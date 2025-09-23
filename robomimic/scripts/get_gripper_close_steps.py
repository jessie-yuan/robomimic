import h5py
import json
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    args = parser.parse_args()

    # extract demonstration list from file
    f = h5py.File(args.dataset, "r")
    demos = sorted(list(f["data"].keys()))
    # put demonstration list in increasing episode order
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    num_successful = 0
    gripper_close_steps = []
    for demo_key in demos:
        if f['data'][demo_key].attrs['label'] in [1,2]:
            num_successful += 1
            last_elements = f['data'][demo_key]['actions'][:, -1]  # Get last element of each sublist
            # print(last_elements)
            switch_indices = np.where((last_elements[:-1] - last_elements[1:] <= -1.25))[0]
            if len(switch_indices) == 0:
                print(last_elements)
            first_switch_index = switch_indices[0] + 1
            gripper_close_steps.append(first_switch_index)

    gripper_close_steps = np.array(gripper_close_steps)

    # report statistics on the data
    print("")
    print("num successful samples:", num_successful)
    print("mean gripper close step:", np.mean(gripper_close_steps))
    print("st dev gripper close step:", np.std(gripper_close_steps))
    print("min gripper close step:", np.min(gripper_close_steps))
    print("max gripper close step:", np.max(gripper_close_steps))

    f.close()

 