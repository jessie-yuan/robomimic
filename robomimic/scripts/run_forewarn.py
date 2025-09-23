
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
# from robomimic.envs.env_robosuite import EnvRobosuite
from robomimic.algo import RolloutPolicy
from robomimic.scripts.real_policy import RealPolicy
import robomimic.utils.env_utils as EnvUtils

import ruamel.yaml as yaml
from tqdm import tqdm

import cv2
from PIL import Image
from gym.spaces import Box, Dict
from dreamer.dreamer import Dreamer
    

def process_rollout(traj, config, wm_model, num_history_images=1, imagined_steps=1):
    with open('/home/jzyuan/uncertainty_aware_steering/robomimic/datasets/square/combined_wm/norm_dict_abs.json', 'r') as file:
        norm_dict = json.load(file)
    for key in norm_dict.keys():
        norm_dict[key] = np.array(norm_dict[key])
    
    length = len(traj['actions'])

    cut_length = length

    ## if you want to downsample the data
    sample_freq = 2

    wrist_images = traj['obs']['robot0_eye_in_hand_image'][:cut_length][::sample_freq]
    front_images = traj['obs']['agentview_image'][:cut_length][::sample_freq]
    # states = f['data'][traj]['obs']['state'][:cut_length][::sample_freq]
    states = []
    for key in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
        states.append(traj['obs'][key])
    states = np.concatenate(states, axis=1)
    states = states[:cut_length][::sample_freq]

    actions = traj['actions'][:cut_length][sample_freq-1::sample_freq]
    
    if length % sample_freq != 0:
        actions = np.concatenate((actions, actions[-1:]), axis=0)
    
    # get the specific snippet for the subtask
    begin_index = 0#start_index#start_index
    # begin_index = subtask_start_index[subtask_id]#start_index#start_index
    
    ## feed in only the first image to get the imagined latent
    original_length = len(wrist_images)
    wrist_images = wrist_images[begin_index:begin_index + num_history_images + imagined_steps]
    front_images = front_images[begin_index:begin_index + num_history_images + imagined_steps]
    # if len(wrist_images) == 0:
    #     print('original length', original_length)
    #     print('traj_id', traj_id)
    #     print('begin_index', begin_index)
    #     print('end index', begin_index + num_history_images + imagined_steps)
    ## pad the images to the length of num_history_images + imagined_steps + start_index - begin_index with the last image
    if len(wrist_images) < num_history_images + imagined_steps :
        padded_length = num_history_images + imagined_steps  - len(wrist_images)
        ## keep the original data type of images
        
        wrist_images = np.concatenate((wrist_images, np.repeat(wrist_images[-1:], padded_length, axis=0)), axis=0)
        front_images = np.concatenate((front_images, np.repeat(front_images[-1:], padded_length, axis=0)), axis=0)
    
    images = np.concatenate((front_images,wrist_images), axis=1)
    
    states = states[begin_index:begin_index + num_history_images+imagined_steps]
    ## normalize states
    ob_min = norm_dict['ob_min']
    ob_max = norm_dict['ob_max']
    states = (states - ob_min)/(ob_max - ob_min)
    states = 2* states - 1
    ## padding the states
    if len(states) < num_history_images + imagined_steps :
        padded_length = num_history_images + imagined_steps- len(states)
        states = np.concatenate((states, np.repeat(states[-1:], padded_length, axis=0)), axis=0)
    
    is_first = np.zeros((num_history_images + imagined_steps + begin_index,1))
    is_first[0,:] = 1
    is_first = is_first
    is_first[0,:] = 1
    is_terminal = np.zeros((num_history_images+imagined_steps + begin_index,1))
    is_terminal = is_terminal

    actions = actions[begin_index:num_history_images+imagined_steps + begin_index]
    ac_min = norm_dict['ac_min']
    ac_max = norm_dict['ac_max']
    actions = (actions - ac_min)/(ac_max - ac_min)
    actions = 2* actions - 1
    is_first = is_first[begin_index:num_history_images+imagined_steps + begin_index]
    
    is_terminal = is_terminal[begin_index:num_history_images+imagined_steps + begin_index]
    
    ## padding the actions and is_first, is_terminal if length < num_history_images + imagined_steps + start_index
    if len(actions) < num_history_images + imagined_steps :
        length = length//sample_freq + (1 if (length) % sample_freq != 0 else 0) - begin_index
        # length = min(length, num_history_images + imagined_steps)
        padded_length = num_history_images + imagined_steps  - len(actions)
        actions = np.concatenate((actions, np.repeat(actions[-1:], padded_length, axis=0)), axis=0)
    else: 
        length = num_history_images + imagined_steps

    images = [Image.fromarray(img.astype('uint8'), 'RGB').resize((96,64)) for img in images]

    dict = {}
    B, T, H, W, C = images.shape
    if T > num_history_images + imagined_steps:
        ## the original shape is B,T,H,W,C
        ## NOW it should be B*6, T//6, H, W, C
        images = images.reshape(-1, T//6, H, W, C)
        states = states.reshape(-1, T//6, states.shape[-1])
        actions = actions.reshape(-1, T//6, actions.shape[-1])
        is_first = is_first.reshape(-1, T//6, 1)
        is_terminal = is_terminal.reshape(-1, T//6, 1)
        actual_lengths = actual_lengths.reshape(B*6)
    # is_first = torch.tensor(is_first)
    # is_terminal = torch.tensor(is_terminal)
    # for i in range(length):
        # for key in images:
    img_keys = ["agentview_image", "robot0_eye_in_hand_image"]
    dict[img_keys[0]] = images[:,:,:H//2,]
    dict[img_keys[1]] = images[:,:,H//2:,]    
    # for key in states:
        # dict[key] = states[key]
    # dict['robot0_eef_pos'] = states[:,:, :3]
    # dict['robot0_eef_quat'] = states[:,:, 3:7]
    # dict['robot0_gripper_qpos'] = states[:,:, 7:]
    dict['state'] = states
    dict['action'] = actions
    dict['is_first'] = is_first[:, :, 0]
    dict['is_terminal'] = is_terminal[:, :, 0]

    batch_embeds = wm_model._wm.get_latent(dict, mode='all', imagined_steps=imagined_steps, actual_lengths= [64], sample_size = 16, total_steps=16)
    return batch_embeds


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

def initialize_vision_model(wm_config):
    action_space = Box(-1, 1, shape = wm_config.action_space)
    wm_config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]

    obs_space = {}
    for key, value in wm_config.observation_space.items():
        if 'robot' in key:
            obs_space[key] = Box(-1, 1, shape = value)
        else: 
            obs_space[key] = Box(0, 1, shape = value)
    obs_space = Dict(obs_space)
    print('loading world model from ckpt path', wm_config.from_ckpt)
    wm_model = Dreamer.from_pretrained(path = wm_config.from_ckpt, obs_space = obs_space,
                                            act_space = action_space,
                                            config = wm_config,
                                            dataset = None,#success_val_dataset,
                                            logger = None,
                                            expert_dataset= None).to(torch.float32)
    wm_model.requires_grad_(requires_grad=False)
    wm_model.eval()

    return wm_model

def run_forewarn(args):
    # some arg checking
    write_video = (args.video_path is not None)
    # if args.render:
    #     # on-screen rendering can only support one camera
    #     assert len(args.camera_names) == 1

    config = yaml.YAML().load('/home/jzyuan/uncertainty_aware_steering/failure_detection/configs/wm_nut_assembly_config.yaml')
    # wm_model = initialize_vision_model(config)
    wm_model = None

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy = RealPolicy(args.checkpoint_dir, args.checkpoint_num)

    # read rollout settings
    n_sampled_trajectories = args.n_samples
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
    obs = env.get_observation()
    
    all_action_seqs = []
    for i in range(n_sampled_trajectories):
        all_action_seqs.append(policy.get_action(obs))

    for i in tqdm(range(n_sampled_trajectories)):

        stats, traj, eef_pos_df = rollout(
            action_seq=all_action_seqs[i], 
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

        batch_embeds = process_rollout(traj, config, wm_model, num_history_images=1, imagined_steps=63)
        # if write_video:
        #     video_writer.close()

        

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
        default=10,
        help="number of trajectories to sample",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=208,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    # parser.add_argument(
    #     "--env",
    #     type=str,
    #     default=None,
    #     help="(optional) override name of env from the one in the checkpoint, and use\
    #         it for rollouts",
    # )

    # Whether to render rollouts to screen
    # parser.add_argument(
    #     "--render",
    #     action='store_true',
    #     help="on-screen rendering",
    # )

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
        default=1,
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

    # parser.add_argument(
    #     "--fixed_initial_state",
    #     action='store_true',
    # )

    parser.add_argument(
        "--checkpoint_num",
        type=int,
        default=None,
        help="which checkpoint to load",
    )

    args = parser.parse_args()
    run_forewarn(args)