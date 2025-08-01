import h5py
import cv2
import numpy as np

def update_labels(input_file, output_file, label):
    """
    Update the label of a specific demo in an HDF5 file based on user input and save to a new file.

    Args:
        hdf5_file_path (str): Path to the original HDF5 file.
        demo_index (int): Index of the demo to process (e.g., 'demo_0').
        output_file (h5py.File): HDF5 file object to save the updated data.

    Returns:
        None
    """
    f_in = h5py.File(input_file, 'r')
    f_out = h5py.File(output_file, 'w')
    data_grp = f_out.create_group("data")

    demos = sorted(list(f_in["data"].keys()))
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    total_samples = 0
    new_index = 0

    for demo_key in demos:
    # return

        ep_data_grp = data_grp.create_group(demo_key) 

        ep_data_grp.create_dataset("actions", data=np.array(f_in["data"][demo_key]["actions"]))
        ep_data_grp.create_dataset("states", data=np.array(f_in["data"][demo_key]["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(f_in["data"][demo_key]["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(f_in["data"][demo_key]["dones"]))
        for k in f_in["data"][demo_key]["obs"]:
            ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(f_in["data"][demo_key]["obs"][k]))
        for k in f_in["data"][demo_key]["next_obs"]:
            ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(f_in["data"][demo_key]["next_obs"][k]))
        ep_data_grp.attrs["label"] = int(label)  # Save the label for this demo

        print(f_in["data"][demo_key].keys())

        # episode metadata
        if "model_file" in f_in["data"][demo_key].attrs:
            ep_data_grp.attrs["model_file"] = f_in["data"][demo_key].attrs["model_file"] # model xml for this episode
        ep_data_grp.attrs["num_samples"] = f_in["data"][demo_key].attrs["num_samples"] # number of transitions in this episode

        if "camera_info" in f_in["data"][demo_key].attrs:
            ep_data_grp.attrs["camera_info"] = f_in["data"][demo_key].attrs["camera_info"]

    if "mask" in f_in:
        f_in.copy("mask", f_out)


    # global metadata
    data_grp.attrs["total"] = total_samples  # total number of demos
    data_grp.attrs["env_args"] = f_in["data"].attrs["env_args"]

    f_in.close()
    f_out.close()

# Example usage
# main('path_to_file.hdf5', 'path_to_new_file.hdf5')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the original HDF5 file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output HDF5 file')
    parser.add_argument('--label', type=int, required=True, help='Label to assign to the demos (e.g., 0, 1, 2)')
    args = parser.parse_args()
    update_labels(args.input, args.output, args.label)