import h5py
import numpy as np
import argparse
from pathlib import Path

def filter_demos(input_file, output_file):
    """
    Filter out specified demo indices from an HDF5 file.
    
    Args:
        input_file (str): Path to input HDF5 file
        output_file (str): Path to output HDF5 file
        demos_to_remove (list): List of demo indices to remove (e.g., [0, 5, 10])
    """
    
    f_in = h5py.File(input_file, 'r')
    f_out = h5py.File(output_file, 'w')
    data_grp = f_out.create_group("data")
            
    demos = sorted(list(f_in["data"].keys()))
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    
    # Remove specified demos
    # filtered_indices = [i for i in inds if i not in demos_to_remove]
    # filtered_indices.sort()  # Keep them in order
    
    print(f"Original demos: {len(inds)}")
    # print(f"Demos to remove: {len(demos_to_remove)}")
    # print(f"Remaining demos: {len(filtered_indices)}")
    # print(f"Removed demo indices: {sorted(demos_to_remove)}")

    total_samples = 0
    
    # Copy filtered demos with new sequential naming
    # for new_idx, original_idx in enumerate(filtered_indices):
    new_idx = 0
    for original_demo_key in demos:
        if f_in["data"][original_demo_key].attrs['label'] == 0:

            # original_demo_key = f'demo_{original_idx}'
            new_demo_key = f'demo_{new_idx}'
            
            ep_data_grp = data_grp.create_group(new_demo_key) 
            
            ep_data_grp.create_dataset("actions", data=np.array(f_in["data"][original_demo_key]["actions"][:120]))
            ep_data_grp.create_dataset("states", data=np.array(f_in["data"][original_demo_key]["states"][:120]))
            ep_data_grp.create_dataset("rewards", data=np.array(f_in["data"][original_demo_key]["rewards"][:120]))
            ep_data_grp.create_dataset("dones", data=np.array(f_in["data"][original_demo_key]["dones"][:120]))
            for k in f_in["data"][original_demo_key]["obs"]:
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(f_in["data"][original_demo_key]["obs"][k][:120]))
            for k in f_in["data"][original_demo_key]["next_obs"]:
                ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(f_in["data"][original_demo_key]["next_obs"][k][:120]))

            print(f_in["data"][original_demo_key].keys())

            # episode metadata
            if "model_file" in f_in["data"][original_demo_key].attrs:
                ep_data_grp.attrs["model_file"] = f_in["data"][original_demo_key].attrs["model_file"] # model xml for this episode
            # ep_data_grp.attrs["num_samples"] = f_in["data"][original_demo_key].attrs["num_samples"] # number of transitions in this episode
            ep_data_grp.attrs["num_samples"] = min(f_in["data"][original_demo_key].attrs["num_samples"], 120)
            ep_data_grp.attrs["label"] = 5

            if "camera_info" in f_in["data"][original_demo_key].attrs:
                ep_data_grp.attrs["camera_info"] = f_in["data"][original_demo_key].attrs["camera_info"]

            # total_samples += f_in["data"][original_demo_key].attrs["num_samples"]
            total_samples += min(f_in["data"][original_demo_key].attrs["num_samples"], 120)

            new_idx += 1
    
    # Update any metadata that might reference the number of demos
    if "mask" in f_in:
        f_in.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples  # total number of demos
    data_grp.attrs["env_args"] = f_in["data"].attrs["env_args"]

    f_in.close()
    f_out.close()

def get_demos_to_remove(labels_file, input_file, goal):
    if labels_file is not None:
        with open(labels_file, 'r') as f:
            lines = f.readlines()

        if goal == 'remove_backending':
            lines = [line.strip() for line in lines]
            lines = [line.split(',')[1] for line in lines]
            demos_to_remove = []

            for i, line in enumerate(lines):
                if line == 'back':
                    demos_to_remove.append(i)

        elif goal == 'keep_frontstarting_labels':
            lines = [line.strip() for line in lines]
            demos_to_remove = []

            frontstarting_leftending = 0
            frontstarting_rightending = 0

            for i, line in enumerate(lines):
                start, end = line.split(',')
                if start != 'front':
                    demos_to_remove.append(i)

                else:
                    if end == 'left':
                        frontstarting_leftending += 1
                    elif end == 'right':
                        frontstarting_rightending += 1

            print(f"Frontstarting leftending: {frontstarting_leftending}, rightending: {frontstarting_rightending}")

    else:
        if goal == 'keep_frontstarting_angle':
            f_in = h5py.File(input_file, 'r')
            demo_names = list(f_in["data"].keys())
            demo_names = sorted(demo_names, key=lambda x: int(x.split('_')[1]))  # Sort by demo number
            demos_to_remove = []

            for i, demo_name in enumerate(demo_names):
                nut = f_in["data"][demo_name]["obs"]["object"][0]
                if nut[10] != 0. or nut[11] != 0.:
                    print("somethings fishy.....")

                rot_angle = 2 * np.arccos(nut[13])
                print(f"Demo {i}: {demo_name}, rot_angle: {rot_angle}")
                if rot_angle > 0.524 and rot_angle < 5.758:
                    demos_to_remove.append(i)

            f_in.close()
            

    return demos_to_remove

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Filter demos from HDF5 file')
    parser.add_argument('--input_file', help='Input HDF5 file path')
    parser.add_argument('--output_file', help='Output HDF5 file path')
    # parser.add_argument('--labels_file', default=None, help='Labels file path with demos to remove')
    # parser.add_argument('--goal', choices=['remove_backending_labels', 'keep_frontstarting_angle', 'keep_frontstarting_labels'], required=True, 
                        # help='Goal for filtering demos: remove backending or keep frontstarting')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist.")

    # if not Path(args.labels_file).exists():
    #     print(f"Error: Labels file '{args.labels_file}' does not exist.")
    
    # Validate output directory exists
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Filtering demos from: {args.input_file}")
    print(f"Output file: {args.output_file}")

    # demos_to_remove = get_demos_to_remove(args.labels_file, args.input_file, args.goal)

    # print(f"Demos to remove: {demos_to_remove}")
    
    try:
        filter_demos(args.input_file, args.output_file)
        print("Filtering completed successfully!")
            
    except Exception as e:
        print(f"Error during filtering: {e}")
        # Clean up partial output file if error occurred
        if output_path.exists():
            output_path.unlink()