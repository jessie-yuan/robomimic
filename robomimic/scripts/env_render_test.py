import cv2
import numpy as np
from robomimic.envs.env_robosuite import EnvRobosuite

env = EnvRobosuite.create_for_data_processing(
    env_name="NutAssemblySquare",
    camera_names=['agentview', 'robot0_eye_in_hand'],
    camera_height=84,
    camera_width=84,
    reward_shaping=False,
    robots=["Panda"],
    render_collision_mesh=False)

obs = env.reset()

video_img = []
for cam_name in ['agentview', 'robot0_eye_in_hand']:
    video_img.append(env.render(mode="rgb_array", height=84, width=84, camera_name=cam_name))
video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
cv2.imwrite("video_img.png", video_img[:, :, ::-1])