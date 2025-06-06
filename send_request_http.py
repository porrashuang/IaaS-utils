import argparse
import os
import numpy as np
import requests

def load_input_from_npz(npz_path):
    try:
        data = np.load(npz_path)
        print(f"Loaded input data from {npz_path}")
        return {name: data[name] for name in data.files}
    except Exception as e:
        print(f"Error loading input data from {npz_path}: {e}")
        return None

def load_input_from_h5(h5_path):
    try:
        import h5py
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

def numpy_to_jsonable(input_dict):
    """Convert numpy arrays in dict to lists for JSON serialization."""
    jsonable = {}
    for k, v in input_dict.items():
        if isinstance(v, np.ndarray):
            jsonable[k] = v.tolist()
        else:
            jsonable[k] = v
    return jsonable

def print_response_json(response_json):
    if not isinstance(response_json, dict):
        print(response_json)
        return
    print("\nInference Outputs:")
    for key, value in response_json.items():
        if isinstance(value, list):
            arr = np.array(value)
            print(f"  - Output '{key}':")
            print(f"      Shape: {arr.shape}")
            print(f"      Dtype: {arr.dtype}")
            print(f"      Data (first 5 elements if large): {arr.flatten()[:5]}")
        else:
            print(f"  - Output '{key}': {value}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HTTP Inference Client Example")
    parser.add_argument('--input_file', type=str, default='/data/request_0.h5',
                        help='Path to the input file (.npz or .h5)')
    parser.add_argument('--file_type', type=str, choices=['npz', 'h5'], default='h5',
                        help='Type of the input file (npz or h5)')
    parser.add_argument('--url', type=str, default='http://localhost:8579/predict_pipeline',
                        help='HTTP endpoint URL')

    args = parser.parse_args()

    # Ensure the input file path is absolute or resolve it relative to /data if it's just a filename
    if not os.path.isabs(args.input_file) and not args.input_file.startswith('/data/'):
        args.input_file = os.path.join('/data', os.path.basename(args.input_file))
        print(f"Relative path detected, assuming input file is at: {args.input_file}")

    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file {args.input_file} does not exist.")
        exit(1)

    if args.file_type == 'npz':
        input_data_map = load_input_from_npz(args.input_file)
    elif args.file_type == 'h5':
        input_data_map = load_input_from_h5(args.input_file)
    else:
        print(f"Unsupported file type: {args.file_type}. Choose 'npz' or 'h5'.")
        exit(1)

    if not input_data_map:
        print(f"Failed to load input data from {args.input_file}.")
        exit(1)

    json_payload = numpy_to_jsonable(input_data_map)
    print(f"\nSending HTTP POST request to {args.url} ...")
    try:
        resp = requests.post(args.url, json=json_payload)
        print(f"HTTP response status: {resp.status_code}")
        try:
            response_json = resp.json()
            print_response_json(response_json)
        except Exception:
            print("HTTP response text:", resp.text)
    except Exception as e:
        print(f"HTTP request failed: {e}")