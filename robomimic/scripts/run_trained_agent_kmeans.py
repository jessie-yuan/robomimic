"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.env_robosuite import EnvRobosuite
from robomimic.algo import RolloutPolicy
from robomimic.scripts.real_policy import RealPolicy
import robomimic.utils.env_utils as EnvUtils

from tqdm import tqdm
import pandas as pd
import plotly.express as px

import seaborn as sns
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
import cv2

def rollout(action_seq, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, camera_height=96, camera_width=96, env_initial_state=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """

    assert isinstance(env, EnvBase)
    assert not (render and (video_writer is not None))
    assert len(action_seq) == horizon, "action_seq length must match the horizon"

    state_dict = env_initial_state
    obs = env.reset_to(state_dict)

    all_eef_pos = [obs["robot0_eef_pos"]]

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):

            # play action
            next_obs, r, done, _ = env.step(action_seq[step_i])

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=camera_height, width=camera_width, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            if all_eef_pos is not None:
                all_eef_pos.append(next_obs["robot0_eef_pos"])

            # collect transition
            traj["actions"].append(action_seq[step_i])
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                # traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                # traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj, all_eef_pos


def trajectory_clustering(trajs, n_clusters=6, random_state=0, time_series=True  ):
    """
    Parameters:
    trajs (list of np.ndarray): A list where each element is a trajectory
                                represented as a 2D array of shape (n_waypoints, dim_per_waypoint).
    n_clusters (int): The number of clusters or modes to create.
    random_state (int): Seed for reproducibility.

    Returns:
    list of np.ndarray: A list of two trajs, each one is a 120 x 8 array
    modes_prob: A list of probs of the two modes
    """
    # Flatten each trajectory into a single vector (n_waypoints * dim_per_waypoint)

    print("shape", np.shape(trajs))

    # Apply K-means clustering
    ## fork 0:3 4 cluster, chip 0:7, 2 cluster, cup kmeans, 2 cluster
    if time_series:
        traj_vectors = np.array([np.array(traj)[:,0:7] for traj in trajs]) # 3 # 7
        scaler = TimeSeriesScalerMeanVariance()  # Normalize each trajectory
        traj_scaled = scaler.fit_transform(traj_vectors)

        # Apply K-means clustering
        kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=True)
        y_kmeans = kmeans.fit_predict(traj_scaled)
        clustered_trajs = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(y_kmeans):
            clustered_trajs[label].append(trajs[idx])
    else:
        traj_vectors = np.array([np.array(traj)[:,0:3].flatten() for traj in trajs])

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state,
                        init='k-means++', n_init=20, max_iter=400, tol=0.0001,
                            verbose=0, copy_x=True)
        kmeans.fit(traj_vectors)

        # Separate trajectories into clusters based on the labels
        clustered_trajs = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(kmeans.labels_):
            clustered_trajs[label].append(trajs[idx])
    p = 1.0 / len(trajs)
    # Compute the average trajectory for each cluster
    aggregated_trajs = []
    modes_prob = []
    for cluster in clustered_trajs:
        # Calculate the mean trajectory for each cluster
        mean_traj = np.mean(cluster, axis=0)
        aggregated_trajs.append(mean_traj)
        modes_prob.append(len(cluster) * p)
    ## also return the labels for each trajectory
    labels = kmeans.labels_
    return aggregated_trajs, modes_prob, labels

def visualize_plans_w_agg(real_traj,aggregated_traj=None, mode_probs = None, labels = None, current_pose=None):

      
        ## plot them in 3d space with matplotlib
        fig = make_subplots(rows=1, cols=1,subplot_titles=("Origin"),specs = [[{"type": "scatter3d"}]])
        colormap = sns.color_palette("pastel")# Paired
        hex_palette = []
        for k, color in enumerate(colormap):
            if k > 7:
                break
            # print('k/len(colormap)', k/7)
            hex_palette.append(f'rgb({int(color[0]*255)} ,{int(color[1]*255)} ,{int(color[2]*255)})')
        colormap = sns.color_palette("muted")
        for k, color in enumerate(colormap):
            if k > 7:
                break
            hex_palette.append(f'rgb({int(color[0]*255)} ,{int(color[1]*255)} ,{int(color[2]*255)})')
        label_to_color = {label: hex_palette[k] for k, label in enumerate(range(16))}
        if current_pose is not None:
            pos_list = np.array([current_pose])
            fig.add_trace(go.Scatter3d(x=pos_list[:,0], y=pos_list[:,1], z=pos_list[:,2], 
                                             mode='markers',
                                                marker=dict(size=5, color='black'),
                                                name=f'current_pose'), row=1, col=1)
        for j in range(len(real_traj)):
          
            
            # else:
                # colormap = ListedColormap(sns.color_palette('viridis').as_hex())
            traj = real_traj[j]
        
            pos_list = np.array(traj)
            color_label = label_to_color[labels[j]]#j/len(real_traj)
            # print('color', color)
            fig.add_trace(go.Scatter3d(x=pos_list[:-3,0], y=pos_list[:-3,1], z=pos_list[:-3,2],
                                            mode='markers', 
                                           marker=dict(size=5, color=color_label), 
                                           name=f'traj_{j}'), row=1, col=1)
        
        # label_to_color = {label: hex_palette[k] for k, label in enumerate(set(labels))}
        
        for j in range(len(aggregated_traj)):
            pos_list = np.array(aggregated_traj[j])
            color_label  = label_to_color[j+8]#hex_palette[j]
            # if j == 0:
            #     color_name = 'black'
            # elif j == 1:
            #     color_name = 'white'
            # elif j == 2:
            #     color_name = 'grey', current_pose=env.get_observation()['robot0_eef_pos']
            # else: 
            #     color_name = 'lightgoldenrodyellow'
           
            fig.add_trace(go.Scatter3d(x=pos_list[:-3,0], y=pos_list[:-3,1], z=pos_list[:-3,2], 
                                           mode='markers', 
                                           marker=dict(size=5, color=color_label),
                                           name=f'aggregated_traj_{j}_{mode_probs[j]}'), row=1, col=1)
            #
     
        return fig


def visualize_eofpos(real_traj,aggregated_traj=None, mode_probs = None, labels = None, current_pose=None):

      
    ## plot them in 3d space with matplotlib
    fig = make_subplots(rows=1, cols=1,subplot_titles=("Origin"),specs = [[{"type": "scatter3d"}]])
    colormap = sns.color_palette("pastel")# Paired
    hex_palette = []
    for k, color in enumerate(colormap):
        if k > 7:
            break
        # print('k/len(colormap)', k/7)
        hex_palette.append(f'rgb({int(color[0]*255)} ,{int(color[1]*255)} ,{int(color[2]*255)})')
    colormap = sns.color_palette("muted")
    for k, color in enumerate(colormap):
        if k > 7:
            break
        hex_palette.append(f'rgb({int(color[0]*255)} ,{int(color[1]*255)} ,{int(color[2]*255)})')
    label_to_color = {label: hex_palette[k] for k, label in enumerate(range(16))}
    if current_pose is not None:
        pos_list = np.array([current_pose])
        fig.add_trace(go.Scatter3d(x=pos_list[:,0], y=pos_list[:,1], z=pos_list[:,2], 
                                            mode='markers',
                                            marker=dict(size=5, color='black'),
                                            name=f'current_pose'), row=1, col=1)
    for j in range(len(real_traj)):
        
        
        # else:
            # colormap = ListedColormap(sns.color_palette('viridis').as_hex())
        traj = real_traj[j]
    
        pos_list = np.array(traj)
        if j == 0:
            color_label = label_to_color[0]#j/len(real_traj)
        else:
            color_label = label_to_color[3]
        # print('color', color)
        fig.add_trace(go.Scatter3d(x=pos_list[:-3,0], y=pos_list[:-3,1], z=pos_list[:-3,2],
                                        mode='markers', 
                                        marker=dict(size=5, color=color_label), 
                                        name=f'traj_{j}'), row=1, col=1)
    
    # label_to_color = {label: EnvUtilsggregated_traj[j])
    #     color_label  = label_to_color[j+8]#hex_palette[j]
    #     # if j == 0:
    #     #     color_name = 'black'
    #     # elif j == 1:
    #     #     color_name = 'white'
    #     # elif j == 2:
    #     #     color_name = 'grey'
    #     # else: 
    #     #     color_name = 'lightgoldenrodyellow'
        
    #     fig.add_trace(go.Scatter3d(x=pos_list[:-3,0], y=pos_list[:-3,1], z=pos_list[:-3,2], 
    #                                     mode='markers', 
    #                                     marker=dict(size=5, color=color_label),
    #                                     name=f'aggregated_traj_{j}_{mode_probs[j]}'), row=1, col=1)
    #     #
    
    return fig

def run_trained_agent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    # if args.render:
    #     # on-screen rendering can only support one camera
    #     assert len(args.camera_names) == 1

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy = RealPolicy(args.checkpoint_dir, args.checkpoint_num)

    # read rollout settings
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        rollout_horizon = 200

    env_meta = FileUtils.get_env_metadata_from_dataset('/home/jzyuan/uncertainty_aware_steering/robomimic/datasets/square/my_demos/startback_endright_abs.hdf5')

    abs_env_meta = deepcopy(env_meta)
    abs_env_meta['env_kwargs']['controller_configs']['body_parts']['right']['input_type'] = 'absolute'

    env = EnvUtils.create_env_for_data_processing(
        env_meta=abs_env_meta,
        camera_names=args.camera_names, 
        camera_height=96, 
        camera_width=96, 
        reward_shaping=False,
        use_depth_obs=False,
        render=args.render,
        render_offscreen=(args.video_path is not None), 
    )

    # env = EnvRobosuite.create_for_data_processing(
    #     env_name="NutAssemblySquare",
    #     camera_names=args.camera_names,
    #     camera_height=96,
    #     camera_width=96,
    #     reward_shaping=False,
    #     render=args.render,
    #     render_offscreen=(args.video_path is not None), 
    #     robots=["Panda"],)

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []

    env.reset()
    initial_state_dict = env.get_state()

    all_eef_pos_dfs = []

    all_action_seqs = []
    obs = env.get_observation()

    for i in range(args.n_samples):
        all_action_seqs.append(policy.get_action_seq(obs))

    aggregated_trajs, modes_prob, labels = trajectory_clustering(all_action_seqs, n_clusters=args.n_clusters, random_state=0, time_series=True)

    print("modes_prob", modes_prob)
    print("labels", labels)

    # for i in range(rollout_num_episodes):
    for i in tqdm(range(args.n_clusters), desc="Running aggregated rollouts"):

        # # maybe create video writer
        # video_writer = None
        # if write_video:
        #     video_writer = imageio.get_writer(args.video_path, fps=20)

        stats, traj, eef_pos_df = rollout(
            action_seq=aggregated_trajs[i], 
            env=env, 
            horizon=rollout_horizon, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
            env_initial_state=initial_state_dict,
        )
        rollout_stats.append(stats)

        # if write_video:
        #     video_writer.close()

        if eef_pos_df is not None:
            # eef_pos_df['rollout'] = f'Rollout {i}'  # Add group identifier
            all_eef_pos_dfs.append(eef_pos_df)

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # success_modality_label = input("input number: ")
            # labels_dict = {
            #     '0': 'left_success',
            #     '1': 'right_success',
            #     '2': 'left_fail',
            #     '3': 'right_fail',
            #     '4': 'unclear',
            # }

            # ep_data_grp.attrs["success_modality_label"] = labels_dict[success_modality_label]

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    avg_rollout_stats["Median_Horizon"] = np.median(rollout_stats["Horizon"])
    avg_rollout_stats["Mean_Successful_Horizon"] = np.mean(
        [h for h, s in zip(rollout_stats["Horizon"], rollout_stats["Success_Rate"]) if s > 0]
    )
    avg_rollout_stats["Median_Successful_Horizon"] = np.median(
        [h for h, s in zip(rollout_stats["Horizon"], rollout_stats["Success_Rate"]) if s > 0]
    )
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))

    if len(all_eef_pos_dfs) > 0:
        # combined_df = pd.concat(all_eef_pos_dfs, ignore_index=True)

        # fig = px.scatter_3d(combined_df, x='x', y='y', z='z', 
        #                     color='rollout',
        #                     title=f'eef pos over time for {rollout_num_episodes} rollouts',)

        # fig.show()
        fig = visualize_plans_w_agg(all_action_seqs, aggregated_traj=aggregated_trajs, mode_probs=modes_prob, labels=labels, current_pose=None)
        fig.update_layout(title=f'eef pos over time for {args.n_clusters} rollouts')
        fig.write_html("/home/jzyuan/uncertainty_aware_steering/robomimic/robomimic/scripts/eef_pos_over_time.html")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="number of trajectories to sample from the policy",
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        default=6,
        help="number of clusters for k-means",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=208,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview", "robot0_eye_in_hand"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    parser.add_argument(
        "--checkpoint_num",
        type=int,
        default=None,
        help="which checkpoint to load",
    )

    args = parser.parse_args()
    run_trained_agent(args)