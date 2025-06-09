import threading
import time
from collections import namedtuple

import numpy as np

from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import json
from omegaconf import DictConfig, OmegaConf
import hydra
from collections import deque, defaultdict
from scipy.spatial.transform import Rotation
import cv2

PRED_HORIZON = 16
ACT_HORIZON = 8
EXP_WEIGHT = 0
GRIP_THRESH = 0.55

def rmat_to_quat(rot_mat, degrees=False):
    quat = R.from_matrix(torch.Tensor(rot_mat)).as_quat()
    return quat


def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X



class RealPolicy:
    """
    A wrapper for using a robot policy on hardware
    """
    def __init__(self, ckpt_dir, iteration = None, log=False, logger=None):
        torch.multiprocessing.set_start_method('spawn')
        agent_yaml_path = Path(ckpt_dir, "agent_config.yaml")
        exp_config_path = Path(ckpt_dir, "exp_config.yaml")
        obs_config_path = Path(ckpt_dir, "obs_config.yaml")
        ob_dict_path = Path(ckpt_dir, "ob_norm.json")
        ac_dict_path = Path(ckpt_dir, "ac_norm.json")

        f = open(ob_dict_path, 'r')
        data = json.load(f)
        self.ob_max = np.array(data['maximum'])
        self.ob_min = np.array(data['minimum'])

        f = open(ac_dict_path, 'r')
        data = json.load(f)
        self.ac_max = np.array(data['maximum'])
        self.ac_min = np.array(data['minimum'])
 

        hydra_config_path = Path(ckpt_dir, ".hydra", "config.yaml")
        print('agent yaml')
        print(agent_yaml_path)
        agent_config = OmegaConf.load(agent_yaml_path)
        exp_config = OmegaConf.load(exp_config_path)
        obs_config = OmegaConf.load(obs_config_path)
        hydra_config = OmegaConf.load(hydra_config_path)
        
        agent = hydra.utils.instantiate(agent_config)
        if exp_config.params.exp_name is not None:
            model_name = exp_config.params.exp_name
        else:
            model_name = exp_config.exp_name
        load_dict = torch.load(Path(ckpt_dir, f"{model_name}.ckpt"))

        agent.load_state_dict(load_dict["model"])

        # Load transforms
        self.transform = hydra.utils.instantiate(obs_config["transform"])

        self.model_name = model_name
        self.agent = agent.eval().cuda()
        self.cam_indices = hydra_config.task.train_buffer.cam_indexes

        self.act_history = []
        self.prev_img = defaultdict(lambda: None)
        self.prev_obs = None
        self.img_chunk = exp_config.params.img_chunk

    def on_begin_traj(self, traj_idx):
        print(f"beginning new trajectory")

    def _proc_image(self, img):

        # addressing issues w negative strides
        if any(stride < 0 for stride in img.strides):
            img = np.ascontiguousarray(img)

        rgb_img = torch.from_numpy(img).float().permute((2, 0, 1)) / 255
    
        return self.transform(rgb_img)[None].cuda()

    def _get_images_and_states(self, obs):
        """
        Return images and states given observations
        """
        images = {}
        state = np.concatenate((obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]))

        cam_keys = ['agentview_image', 'robot0_eye_in_hand_image']
        for i, key in enumerate(cam_keys):
            if i in self.cam_indices:
                img = obs[key]
                cur_img = self._proc_image(img)
                if self.prev_img[key] is None:
                    self.prev_img[key] = torch.clone(cur_img)
                if self.img_chunk == 2:
                    images['cam'+str(i)] = torch.cat((cur_img, self.prev_img[key]), dim = 0).unsqueeze(0)
                else:
                    images['cam'+str(i)] = cur_img
                self.prev_img[key] = cur_img

        state = (state - self.ob_min)/(self.ob_max-self.ob_min)
        state = state*2 - 1
        if self.prev_obs is None:
            self.prev_obs = state
        
        self.prev_obs = state

        state = torch.from_numpy(state.astype(np.float32))[None].cuda()
        
        return images, state 

    def get_action(self, obs, reward=0, pred_action=None):
        """
        Called at the end of each step.
        """
        images, state = self._get_images_and_states(obs)
        
        if self.act_history:
            ac = self.act_history.pop(0)
        else:
            with torch.no_grad():
                action = self.agent.get_actions(images, state)
            ac_list = action[0].cpu().numpy().astype(np.float32)[:ACT_HORIZON]
            self.act_history = list(ac_list)
            ac = self.act_history.pop(0) 

        if len(ac) == 7:
            ac = (ac+1)/2
            ac = ac*(self.ac_max-self.ac_min)+ self.ac_min  
    
        assert len(ac) == 7, "Assuming 7d action dim!"
        
        return ac, None, None

    def reset_rollout(self):
        self.act_history = []
        self.prev_img = defaultdict(lambda: None)
        self.prev_obs = None
    
    def on_end_traj(self, traj_idx, traj_status):
        pass

    def reset_logger(self):
        pass

    def log_data(self, obs, action, reward=0):
        pass

    def log_obs(self, obs: dict, action: dict):
        pass

    def on_step(self, traj_idx, step_idx):
        return None
    
if __name__ == "__main__":
    policy = RealPolicy(['cam_rs, cam_zed'], "/data/test/diff")


    '''def __init__(self, controller_id='r'):
        self.reader = OculusReader()
        self.reader.run()
        self.controller_id = controller_id
        # LPF filter
        self.vr_pose_filtered = None
        self.global_to_env_mat = vec_to_reorder_mat([-3, -1, 2, 4])
        self.vr_to_global_mat = np.eye(4)
        self.last_pos = np.array([0.,0.,0.])
        self.last_rot = R.identity()

        self.reset_orientation = True

        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = False


        print("Oculus Quest teleop reader instantiated.")
    
    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        pass

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_state(self):
        # Get data from oculus reader
        transforms, buttons = self.reader.get_transformations_and_buttons()

        # Generate output
        button_labels = get_button_labels(self.controller_id)
        if transforms:
            control_en = buttons[button_labels["control_en"]][0] > 0.9
            grasp_en = buttons[button_labels["grasp_en"]]
            if self.reset_orientation and control_en:
                self.vr_to_global_mat = np.linalg.inv(
                    np.asarray(transforms[self.controller_id])
                )
                self.reset_orientation = False

            if not control_en:
                self.reset_orientation = True

            diff_matrix = self.vr_to_global_mat @ np.asarray(
                transforms[self.controller_id]
            )
            
            # print(f"diff_matrix: {diff_matrix[:3, 3]}")
            pose_matrix = self.global_to_env_mat @ diff_matrix
        else:
            control_en = False
            grasp_en = 0
            pose_matrix = np.eye(4)
            self.vr_pose_filtered = None
            self.reset_orientation = True

        rot_mat = np.asarray(pose_matrix)
        vr_pos = rot_mat[:3, 3]
        vr_quat = rmat_to_quat(rot_mat[:3, :3])

        pose = (vr_pos, vr_quat)
        return control_en, grasp_en, pose, buttons
    
    def get_controller_state(self):
        # Get data from oculus reader
        transforms, buttons = self.reader.get_transformations_and_buttons()
        # Generate output
        button_labels = get_button_labels(self.controller_id)
        if transforms:
            control_en = buttons[button_labels["control_en"]][0] > 0.9
            grasp_en = buttons[button_labels["grasp_en"]]
            # if grasp_en:
            #     print(f"grasp enabled")
            # else:
            #     print(f"grasp not enabled")
            if self.reset_orientation and control_en:
                self.vr_to_global_mat = np.linalg.inv(
                    np.asarray(transforms[self.controller_id])
                )
                self.reset_orientation = False
                self.last_pos = np.array([0.,0.,0.])

            if not control_en:
                self.reset_orientation = True

            diff_matrix = self.vr_to_global_mat @ np.asarray(
                transforms[self.controller_id]
            )
            # print(f"diff_matrix: {diff_matrix[:3, 3]}")
            pose_matrix = self.global_to_env_mat @ diff_matrix
        else:
            control_en = False
            grasp_en = 0
            pose_matrix = np.eye(4)
            self.reset_orientation = True

        rot_mat = np.asarray(pose_matrix)
        vr_pos = rot_mat[:3, 3]
        vr_quat = rmat_to_quat(rot_mat[:3, :3])

        pose = (vr_pos, vr_quat)

        

        # Compute delta position
        dpos = vr_pos - self.last_pos
        self.last_pos = vr_pos

        # Compute delta rotation (as Euler angle difference or raw quaternion diff)
        current_rot = R.from_matrix(rot_mat[:3, :3])
        raw_drotation = (current_rot * self.last_rot.inv()).as_rotvec()  # delta rotation in axis-angle
        self.last_rot = current_rot

        if not control_en:
            raw_drotation*=0
            dpos *= 0
            rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

        return dict(
            dpos=dpos,
            rotation=vr_quat,            # absolute orientation
            raw_drotation=raw_drotation, # delta orientation (rotation vector)
            grasp=int(grasp_en),
            reset=0         # reset if not being held
        )'''
