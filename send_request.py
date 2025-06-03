import tritonclient.grpc as grpcclient
import numpy as np
import argparse
import os # Added for path joining

def load_input_from_npz(npz_path):
    """Loads input data from a .npz file."""
    try:
        data = np.load(npz_path)
        print(f"Loaded input data from {npz_path}")
        return {name: data[name] for name in data.files}
    except Exception as e:
        print(f"Error loading input data from {npz_path}: {e}")
        return None

def load_input_from_h5(h5_path):
    """Loads input data from a .h5 file."""
    try:
        import h5py # Import h5py locally
        input_dict = {}
        with h5py.File(h5_path, "r") as f:
            for name in f.keys():
                input_dict[name] = f[name][:]
        print(f"Loaded input data from {h5_path}")
        return input_dict
    except ImportError:
        print("h5py library is not installed. Please install it to load .h5 files (e.g., pip install h5py).")
        return None
    except Exception as e:
        print(f"Error loading input data from {h5_path}: {e}")
        return None

def main(model_name, input_file_path, file_type):
    # server_url should be accessible from within the container.
    # If using --network=host and server is on host, 'localhost:8001' is correct.
    server_url = 'localhost:8001'
    ssl_enabled = False
    model_version = '' # Empty string for latest version

    try:
        triton_client = grpcclient.InferenceServerClient(url=server_url, ssl=ssl_enabled, verbose=False)
        print(f"Attempting to connect to Triton server at {server_url}")
    except Exception as e:
        print(f"Could not create inference client: {e}")
        return

    try:
        if not triton_client.is_server_live():
            print(f"FAILED: Server at {server_url} is not live.")
            return
        print("Server is live.")
        if not triton_client.is_server_ready():
            print(f"FAILED: Server at {server_url} is not ready.")
            return
        print("Server is ready.")
        if not triton_client.is_model_ready(model_name, model_version):
            print(f"FAILED: Model '{model_name}' is not ready on server {server_url}.")
            # You can get model repository index to see available models
            try:
                repo_index = triton_client.get_model_repository_index()
                print("Available models/versions on the server:")
                for item in repo_index.models:
                    print(f"- {item.name} (versions: {item.versions}, state: {item.state})")
            except Exception as e_repo:
                print(f"Could not retrieve model repository index: {e_repo}")
            return
        print(f"Model '{model_name}' is ready.")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    if file_type == 'npz':
        input_data_map = load_input_from_npz(input_file_path)
    elif file_type == 'h5':
        input_data_map = load_input_from_h5(input_file_path)
    else:
        print(f"Unsupported file type: {file_type}. Choose 'npz' or 'h5'.")
        return

    if not input_data_map:
        print(f"Failed to load input data from {input_file_path}.")
        return

    inputs = []
    print("\nPreparing inference inputs...")
    for input_name, input_array in input_data_map.items():
        print(f"  Input Name: {input_name}, Shape: {input_array.shape}, Dtype: {input_array.dtype}")
        triton_dtype = np_to_triton_dtype(input_array.dtype)
        if triton_dtype is None:
            print(f"Unsupported numpy dtype {input_array.dtype} for input {input_name}")
            return
        infer_input = grpcclient.InferInput(input_name, input_array.shape, triton_dtype)
        infer_input.set_data_from_numpy(input_array)
        inputs.append(infer_input)

    if not inputs:
        print("No inputs were prepared. Check your input file and model configuration.")
        return

    outputs = [] # Let Triton return all outputs, or specify:
    # outputs = [grpcclient.InferRequestedOutput('your_output_name_1'),
    #            grpcclient.InferRequestedOutput('your_output_name_2')]


    print(f"\nSending inference request to model '{model_name}'...")
    try:
        results = triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            model_version=model_version
        )
        print("\nInference successful!")
        print("\nInference Outputs:")
        # Get all output names from the result object
        output_names_from_server = [output.name for output in results.get_response().outputs]

        if not output_names_from_server:
            print("  No outputs returned from the server.")
        else:
            print("  Available output names from server and their results:")
            for name in output_names_from_server:
                try:
                    result_array = results.as_numpy(name)
                    print(f"    - Output '{name}':")
                    print(f"        Shape: {result_array.shape}")
                    print(f"        Dtype: {result_array.dtype}")
                    print(f"        Data (first 5 elements if large): {result_array.flatten()[:5]}")
                except Exception as e_out:
                    print(f"    - Could not get output '{name}' as numpy: {e_out}")

    except grpcclient.InferenceServerException as e:
        print(f"Inference failed: {e}")
        if e.debug_details():
            print(f"Debug details: {e.debug_details()}")
    except Exception as e:
        print(f"An unexpected error occurred during inference: {e}")

def np_to_triton_dtype(np_dtype):
    """Converts NumPy dtype to Triton dtype string."""
    if np_dtype == np.bool_:
        return "BOOL"
    elif np_dtype == np.int8:
        return "INT8"
    elif np_dtype == np.int16:
        return "INT16"
    elif np_dtype == np.int32:
        return "INT32"
    elif np_dtype == np.int64:
        return "INT64"
    elif np_dtype == np.uint8:
        return "UINT8"
    elif np_dtype == np.uint16:
        return "UINT16"
    elif np_dtype == np.uint32:
        return "UINT32"
    elif np_dtype == np.uint64:
        return "UINT64"
    elif np_dtype == np.float16:
        return "FP16"
    elif np_dtype == np.float32:
        return "FP32"
    elif np_dtype == np.float64:
        return "FP64"
    # For object arrays that are actually bytes, Triton expects "BYTES"
    elif np_dtype == np.object_ or np_dtype.type == np.bytes_ or np_dtype.type == np.str_:
        return "BYTES"
    print(f"Warning: np_to_triton_dtype: Unhandled numpy dtype {np_dtype}")
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triton Inference Client Example")
    parser.add_argument('--model_name', type=str, default='nugraph2',
                        help='Name of the model to query')
    # Updated default path to use /data (mounted volume)
    parser.add_argument('--input_file', type=str, default='/data/request_0.h5',
                        help='Path to the input file (.npz or .h5) inside the container (e.g., /data/your_file.npz)')
    parser.add_argument('--file_type', type=str, choices=['npz', 'h5'], default='h5',
                        help='Type of the input file (npz or h5)')

    args = parser.parse_args()

    # Ensure the input file path is absolute or resolve it relative to /data if it's just a filename
    if not os.path.isabs(args.input_file) and not args.input_file.startswith('/data/'):
        # This logic might be too simple if complex paths are given.
        # Assuming if not absolute, it's meant to be directly in /data
        args.input_file = os.path.join('/data', os.path.basename(args.input_file))
        print(f"Relative path detected, assuming input file is at: {args.input_file}")


    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file {args.input_file} does not exist inside the container.")
        print(f"Please ensure the file is present in the mounted /data directory (host: ~/my_triton_data) "
              f"and the path is correct.")
    else:
        main(args.model_name, args.input_file, args.file_type)
