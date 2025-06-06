import argparse
import os
import logging
from typing import Dict, Any, Callable, List

import ray
from ray import serve
from starlette.requests import Request

import torch
import torch.nn as nn # Added for NuGraph2_model example structure
import numpy as np
import pandas as pd

# Attempt to import nugraph and torch_geometric components
try:
    import nugraph as ng
    import torch_geometric as pyg
    from torch_geometric.data import Dataset
    from torch_geometric.transforms import Compose
    PYG_NUGRAPH_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import 'nugraph' or 'torch_geometric'. Please ensure they are installed. Error: {e}")
    PYG_NUGRAPH_AVAILABLE = False

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- User-provided HitGraphProducer class ---
class HitGraphProducer():
    def __init__(self,
                 semantic_labeller: Callable = None,
                 event_labeller: Callable = None,
                 label_vertex: bool = False,
                 planes: list[str] = ['u','v','y'],
                 node_pos: list[str] = ['local_wire','local_time'],
                 pos_norm: list[float] = [0.3,0.055],
                 node_feats: list[str] = ['integral','rms'],
                 lower_bound: int = 20,
                 filter_hits: bool = False):
        if not PYG_NUGRAPH_AVAILABLE:
            raise ImportError("torch_geometric is required for HitGraphProducer.")
        self.semantic_labeller = semantic_labeller
        self.event_labeller = event_labeller
        self.label_vertex = label_vertex
        self.planes = planes
        self.node_pos = node_pos
        self.pos_norm = torch.tensor(pos_norm).float()
        self.node_feats = node_feats
        self.lower_bound = lower_bound
        self.filter_hits = filter_hits
        import logging
        logging.basicConfig(level=logging.INFO, force=True)  # force=True for Python 3.8+
        self.logger = logging.getLogger(__name__)
        self.transform = pyg.transforms.Compose((
            pyg.transforms.Delaunay(),
            pyg.transforms.FaceToEdge()))
    
    def create_graph(self, hit_table_hit_id, hit_table_local_plane, hit_table_local_time, \
                     hit_table_local_wire, hit_table_integral, hit_table_rms, \
                     spacepoint_table_spacepoint_id, spacepoint_table_hit_id_u, spacepoint_table_hit_id_v, \
                     spacepoint_table_hit_id_y,
                     # Optional tables, pass None or empty if not available for a request
                     event_table=None, 
                     edep_table=None, 
                     particle_table=None):
        
        # Constructing DataFrames from input lists/arrays
        # Ensure inputs are suitable for DataFrame construction (e.g., lists of scalars or 1D numpy arrays)
        evt = {
            'hit_table': pd.DataFrame({
                'hit_id': hit_table_hit_id, 
                'local_plane': hit_table_local_plane, 
                'local_time': hit_table_local_time,
                'local_wire': hit_table_local_wire, 
                'integral': hit_table_integral, 
                'rms': hit_table_rms
            }),
            'spacepoint_table': pd.DataFrame({
                'spacepoint_id': spacepoint_table_spacepoint_id, 
                'hit_id_u': spacepoint_table_hit_id_u,
                'hit_id_v': spacepoint_table_hit_id_v, 
                'hit_id_y': spacepoint_table_hit_id_y
            })
        }
        if self.event_labeller or self.label_vertex:
            event = evt['event_table'].squeeze()

        hits = evt['hit_table']
        spacepoints = evt['spacepoint_table'].reset_index(drop=True)

        # discard any events with pathologically large hit integrals
        # this is a hotfix that should be removed once the dataset is fixed
        if hits.integral.max() > 1e6:
            print('found event with pathologically large hit integral, skipping')
            return evt.name, None

        # handle energy depositions
        if self.filter_hits or self.semantic_labeller:
            edeps = evt['edep_table']
            energy_col = 'energy' if 'energy' in edeps.columns else 'energy_fraction' # for backwards compatibility
            edeps = edeps.sort_values(by=[energy_col],
                                      ascending=False,
                                      kind='mergesort').drop_duplicates('hit_id')
            hits = edeps.merge(hits, on='hit_id', how='right')

            # if we're filtering out data hits, do that
            if self.filter_hits:
                hitmask = hits[energy_col].isnull()
                filtered_hits = hits[hitmask].hit_id.tolist()
                hits = hits[~hitmask].reset_index(drop=True)
                # filter spacepoints from noise
                cols = [ f'hit_id_{p}' for p in self.planes ]
                spmask = spacepoints[cols].isin(filtered_hits).any(axis='columns')
                spacepoints = spacepoints[~spmask].reset_index(drop=True)

            hits['filter_label'] = ~hits[energy_col].isnull()
            hits = hits.drop(energy_col, axis='columns')

        # reset spacepoint index
        spacepoints = spacepoints.reset_index(names='index_3d')

        # get labels for each particle
        if self.semantic_labeller:
            particles = self.semantic_labeller(evt['particle_table'])
            try:
                hits = hits.merge(particles, on='g4_id', how='left')
            except:
                print('exception occurred when merging hits and particles')
                print('hit table:', hits)
                print('particle table:', particles)
                print('skipping this event')
                return None
            mask = (~hits.g4_id.isnull()) & (hits.semantic_label.isnull())
            if mask.any():
                print(f'found {mask.sum()} orphaned hits.')
                return evt.name, None
            del mask

        data = pyg.data.HeteroData()

        # event metadata
        data['metadata'].run = 6876
        data['metadata'].subrun = 9
        data['metadata'].event = 470

        # spacepoint nodes
        data['sp'].num_nodes = spacepoints.shape[0]

        # draw graph edges
        for i, plane_hits in hits.groupby('local_plane'):

            p = self.planes[i]
            plane_hits = plane_hits.reset_index(drop=True).reset_index(names='index_2d')

            # node position
            pos = torch.tensor(plane_hits[self.node_pos].values).float()
            data[p].pos = pos * self.pos_norm[None,:]

            # node features
            data[p].x = torch.tensor(plane_hits[self.node_feats].values).float()

            # hit indices
            data[p].id = torch.tensor(plane_hits['hit_id'].values).long()

            # 2D edges
            data[p, 'plane', p].edge_index = self.transform(data[p]).edge_index

            # 3D edges
            edge3d = spacepoints.merge(plane_hits[['hit_id','index_2d']].add_suffix(f'_{p}'),
                                       on=f'hit_id_{p}',
                                       how='inner')
            edge3d = edge3d[[f'index_2d_{p}','index_3d']].values.transpose()
            edge3d = torch.tensor(edge3d) if edge3d.size else torch.empty((2,0))
            data[p, 'nexus', 'sp'].edge_index = edge3d.long()

            # truth information
            if self.semantic_labeller:
                data[p].y_semantic = torch.tensor(plane_hits['semantic_label'].fillna(-1).values).long()
                data[p].y_instance = torch.tensor(plane_hits['instance_label'].fillna(-1).values).long()
            if self.label_vertex:
                vtx_2d = torch.tensor([ event[f'nu_vtx_wire_pos_{i}'], event.nu_vtx_wire_time ]).float()
                data[p].y_vtx = vtx_2d * self.pos_norm[None,:]

        for p in self.planes:
            if bool(data[p]): continue
            data[p].pos = torch.empty(0, 2)
            data[p].x = torch.empty(0, 2)
            data[p].id = torch.empty(0)
            data[p, 'plane', p].edge_index = torch.empty((2, 0), dtype=torch.long)
            data[p, 'nexus', 'sp'].edge_index = torch.empty((2, 0), dtype=torch.long)

        # event label
        if self.event_labeller:
            data['evt'].y = torch.tensor(self.event_labeller(event)).long()

        # 3D vertex truth
        if self.label_vertex:
            vtx_3d = [ [ event.nu_vtx_corr_x, event.nu_vtx_corr_y, event.nu_vtx_corr_z ] ]
            data['evt'].y_vtx = torch.tensor(vtx_3d).float()
        
        return data

# --- User-provided HeteroDataset ---
class HeteroDataset(Dataset):
    def __init__(self, hetero_data, transform=None):
        super().__init__(transform=transform)
        self.transform = transform
        self.hetero_data = hetero_data

    def len(self):
        return 1
    def get(self, idx=0):
        return self.transform(self.hetero_data)
    
# --- GNN Model Handler ---
class GNNModelHandler:
    def __init__(self, model_dir_path: str, checkpoint_filename: str = "test-ng2.ckpt", device: str = 'cpu'):
        if not PYG_NUGRAPH_AVAILABLE:
            raise ImportError("nugraph library is required for GNNModelHandler.")
        import logging
        logging.basicConfig(level=logging.INFO, force=True)  # force=True for Python 3.8+
        self.logger = logging.getLogger(__name__)
        
        self.device = torch.device(device)
        self.model_path = os.path.join(model_dir_path, checkpoint_filename)
        
        if not os.path.exists(self.model_path):
            self.logger.error(f"Checkpoint file not found at {self.model_path}")
            raise FileNotFoundError(f"Checkpoint file not found at {self.model_path}")

        try:
            ModelClass = ng.models.nugraph2.NuGraph2 
            self.model = ModelClass.load_from_checkpoint(self.model_path, map_location=self.device)
            self.model.to(self.device) # Ensure model is on the correct device
            self.model.eval()
            self.logger.info(f"NuGraph2 GNN model successfully loaded from checkpoint {self.model_path} to {self.device}")
        except AttributeError as e:
            self.logger.error(f"Error loading NuGraph2 GNN model: 'nugraph.models.nugraph2.NuGraph2' might be an incorrect path or the library structure is different. Original error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading NuGraph2 GNN model from checkpoint {self.model_path}: {e}")
            raise

    def run_inference(self, data_obj: pyg.data.HeteroData) -> Dict[str, np.ndarray]:
        """Runs inference using the model's step method and extracts results."""
        self.model.step(data_obj.to(self.device)) # Call step method
        
        # Extract results from self.model.data as per user's NuGraph2_model.forward
        # This structure ('u', 'x_semantic', etc.) is specific to the model's output
        model_output_data = self.model.data 
        
        results = {}
        for plane in ['u', 'v', 'y']:
            if 'x_semantic' in model_output_data[plane]:
                results[f'x_semantic_{plane}'] = model_output_data[plane]['x_semantic'].cpu().detach().numpy()
            if 'x_filter' in model_output_data[plane]:
                results[f'x_filter_{plane}'] = model_output_data[plane]['x_filter'].cpu().detach().numpy()
        
        if not results:
             self.logger.warning("No data extracted from model output. Model 'data' attribute might be empty or have unexpected structure.")
             self.logger.debug(f"Model output data keys: {model_output_data.keys() if isinstance(model_output_data, dict) else 'Not a dict'}")


        return results

# --- Ray Serve Deployment for the Full Pipeline ---
@serve.deployment
class NuGraphPipelineDeployment:
    def __init__(self, model_dir_path: str, checkpoint_filename: str, device: str, 
                 hit_producer_config: Dict = None, # For HitGraphProducer kwargs
                 planes: List[str] = None, 
                 norm_dict: Dict = None):
        if not PYG_NUGRAPH_AVAILABLE:
            raise ImportError("nugraph or torch_geometric is not available for NuGraphPipelineDeployment.")
        import logging
        logging.basicConfig(level=logging.DEBUG, force=True)  # force=True for Python 3.8+
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Initializing NuGraphPipelineDeployment on device: {device}")
        
        # Initialize HitGraphProducer
        # Allow passing configuration for HitGraphProducer if needed
        hgp_kwargs = hit_producer_config if hit_producer_config else {}
        self.hit_producer = HitGraphProducer(**hgp_kwargs)
        self.logger.info("HitGraphProducer initialized.")

        # Initialize GNNModelHandler
        self.gnn_model_handler = GNNModelHandler(
            model_dir_path=model_dir_path,
            checkpoint_filename=checkpoint_filename,
            device=device
        )
        self.logger.info("GNNModelHandler initialized.")

        # Store configurations for transforms
        self.planes = planes if planes else ['u', 'v', 'y']
        # Default norm dictionary from user's NuGraph2_model example
        self.norm = norm_dict if norm_dict else {
            'u': torch.tensor(np.array([[389.00623, 173.41809, 144.40556, 4.558202 ], [148.03027, 78.83374, 223.77074, 2.2621164]]).astype(np.float32)),
            'v': torch.tensor(np.array([[369.14136, 173.47131, 151.55148, 4.452483 ], [145.24632, 81.39258, 298.7041 , 1.9223225]]).astype(np.float32)),
            'y': torch.tensor(np.array([[547.3887 , 173.13036, 109.57681, 4.1024694], [284.20694, 74.47841, 108.93824, 1.431838 ]]).astype(np.float32))
        }
        self.logger.info(f"Pipeline configured with planes: {self.planes}")

    def _prepare_input_from_json(self, json_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts data for HitGraphProducer.create_graph from JSON.
        Assumes JSON values are lists that can be used to create pandas Series/DataFrames.
        """
        required_keys = [
            'hit_table_hit_id', 'hit_table_local_plane', 'hit_table_local_time',
            'hit_table_local_wire', 'hit_table_integral', 'hit_table_rms',
            'spacepoint_table_spacepoint_id', 'spacepoint_table_hit_id_u',
            'spacepoint_table_hit_id_v', 'spacepoint_table_hit_id_y'
        ]
        prepared_input = {}
        for key in required_keys:
            if key not in json_input:
                raise ValueError(f"Missing required key in JSON input: {key}")
            prepared_input[key] = json_input[key] # Assuming these are already lists/arrays
        
        # Handle optional tables (event_table, edep_table, particle_table)
        # The JSON input should provide these as dictionaries if they are to be used.
        prepared_input['event_table'] = json_input.get('event_table')
        prepared_input['edep_table'] = json_input.get('edep_table')
        prepared_input['particle_table'] = json_input.get('particle_table')
        
        return prepared_input

    async def __call__(self, http_request: Request) -> Dict[str, Any]:
        if http_request.method == "POST":
            try:
                json_input = await http_request.json()
                self.logger.info(f"Received POST request. Keys: {list(json_input.keys())}")

                # 1. Prepare input for HitGraphProducer
                graph_input_args = self._prepare_input_from_json(json_input)

                # 2. Create initial HeteroData object
                hetero_data_obj = self.hit_producer.create_graph(**graph_input_args)
                if hetero_data_obj is None or (isinstance(hetero_data_obj, tuple) and hetero_data_obj[1] is None) : # Adjust based on actual return
                    self.logger.error("HitGraphProducer failed to create a graph.")
                    return {"error": "Graph creation failed in HitGraphProducer.", "status_code": 500}
                # If create_graph returns a tuple (name, data), extract data
                if isinstance(hetero_data_obj, tuple): 
                    hetero_data_obj = hetero_data_obj[0] # Assuming data is the first element if tuple


                # 3. Define and apply transformations
                # Ensure ng.util components are available if PYG_NUGRAPH_AVAILABLE is true
                if not PYG_NUGRAPH_AVAILABLE: # Should have been caught earlier
                     return {"error": "NuGraph/PyG components for transform not available", "status_code": 500}

                transform = Compose([
                    ng.util.PositionFeatures(self.planes),
                    ng.util.FeatureNorm(self.planes, self.norm),
                    ng.util.HierarchicalEdges(self.planes),
                    ng.util.EventLabels() # Assuming default EventLabels config
                ])
                
                temp_dataset = HeteroDataset(hetero_data_obj, transform=transform)
                transformed_data_obj = temp_dataset.get() # Get the single transformed item
                self.logger.info("Data transformed successfully.")

                # 4. Run GNN inference
                raw_results = self.gnn_model_handler.run_inference(transformed_data_obj)
                self.logger.info(f"Inference complete. Result: {raw_results}")

                # 5. Serialize results for JSON response
                serializable_output = {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in raw_results.items()
                }
                return serializable_output

            except ValueError as ve: # Catch data validation or specific processing errors
                self.logger.error(f"ValueError during pipeline processing: {ve}")
                return {"error": f"Data processing error: {str(ve)}", "status_code": 400}
            except FileNotFoundError as fnfe:
                 self.logger.error(f"FileNotFoundError: {fnfe}")
                 return {"error": f"Configuration or model file not found: {str(fnfe)}", "status_code": 500}
            except Exception as e:
                import traceback
                self.logger.error(f"Unhandled error during pipeline processing: {e}\n{traceback.format_exc()}")
                return {"error": f"Internal server error: {str(e)}", "status_code": 500}
        else:
            self.logger.warning(f"Received non-POST request: {http_request.method}")
            return {"error": "Only POST requests are supported.", "status_code": 405}

# --- Main Execution Block ---
if __name__ == "__main__":
    if not PYG_NUGRAPH_AVAILABLE:
        logger.critical("nugraph or torch_geometric library is required but not found. Please install it. Exiting.")
        exit(1)

    parser = argparse.ArgumentParser(description="Ray Serve NuGraph2 Full Pipeline Server")
    parser.add_argument("--model-dir-path", type=str, required=True,
                        help="Local directory path containing the NuGraph2 model checkpoint.")
    parser.add_argument("--checkpoint-filename", type=str, default="test-ng2.ckpt",
                        help="Filename of the model checkpoint within the model directory.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Device to run the model on (cpu or cuda).")
    # Add arguments for HitGraphProducer config, planes, norm_dict if they need to be configurable via CLI
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    if not ray.is_initialized():
        ray.init(num_cpus=4, configure_logging=True, logging_level=logging.INFO) # Adjust as needed
    
    # Here you could load HitGraphProducer config, planes, norm_dict from files or define them
    # For simplicity, using defaults or values embedded in NuGraphPipelineDeployment for now.
    # hit_producer_config_arg = {} 
    # planes_arg = ['u','v','y']
    # norm_dict_arg = None # Will use default in deployment

    nugraph_full_pipeline_app = NuGraphPipelineDeployment.bind(
        model_dir_path=args.model_dir_path,
        checkpoint_filename=args.checkpoint_filename,
        device=args.device,
    )
    serve.start(http_options={"port": 8579})
    serve.run(
        nugraph_full_pipeline_app, 
        name="nugraph_full_pipeline", # Name for the deployment
        route_prefix="/predict_pipeline", # Updated route for the full pipeline
        blocking=True # For serve.run, blocking is default in script context
    )

    logger.info("Press Ctrl+C to stop the server.")

