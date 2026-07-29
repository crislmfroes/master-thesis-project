from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, pipeline
from trl import GRPOConfig, GRPOTrainer
import random
from datasets import Dataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.envs.factory import make_env, make_env_pre_post_processors, make_env_config
from lerobot.envs.libero import LiberoEnv as LeRobotLiberoEnv
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, SmolVLAConfig
from lerobot.rewards.topreward.modeling_topreward import TOPRewardConfig, TOPRewardModel
from lerobot.rewards.topreward.processor_topreward import TOPRewardEncoderProcessorStep
from lerobot.rewards.topreward.configuration_topreward import PolicyFeature
from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel
from lerobot.rewards.factory import make_reward_pre_post_processors
from lerobot.rewards.robometer.modeling_robometer import ROBOMETER_FEATURE_PREFIX
from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep

from lerobot.envs.utils import preprocess_observation
from lerobot.types import TransitionKey
from scipy.spatial.transform import Rotation

from copy import deepcopy
from PIL import Image
from typing import Any, Literal, List, Dict
import numpy as np
import cv2
import torch
import gc
import re

import tqdm

random.seed(123)

policy_path = "crislmfroes/smolvla-libero-90"
dataset_path = "crislmfroes/libero_90_lerobot"
#policy_path = "lerobot/smolvla_libero"
#policy = SmolVLAPolicy.from_pretrained(pretrained_name_or_path=policy_path)
#policy = policy.cpu()
#policy.config.device = 'cpu'
policy = None

reward_model = None

trainer: GRPOTrainer = None

# ==========================================
# 1. Configuration & Constants
# ==========================================
MODEL_ID = "Qwen/Qwen3.5-9B"
OUTPUT_DIR = "./vlm-grpo-vla-steering"


# ==========================================
# 2. Environment Factory
# ==========================================

class LiberoEnv:
    def reset(self, **kwargs):
        env_config = make_env_config(env_type="libero", task=kwargs["task_suite"], task_ids=[kwargs["task_id"],], render_mode="human")
        env = make_env(cfg=env_config)
        self.task_suite = kwargs["task_suite"]
        self.task_id = kwargs["task_id"]
        self.env = env
        self.env_config = env_config
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=self.env_config, policy_cfg=None)
        self.env_preprocessor = env_preprocessor
        self.env_postprocessor = env_postprocessor
        obs, info = self.env[self.task_suite][self.task_id].reset(seed=kwargs["seed"])
        print(obs.keys())
        print(obs['robot_state']['eef'].keys())
        self.obs = obs
        self.prev_obs = deepcopy(obs)
        self.info = info
        self.subtask_reward = 0.0
        self.task_reward = 0.0
        self.stage_reward = 0.0
        self.actions = []
        self.initial_obs = deepcopy(self.obs)
        self.last_checkpoint = deepcopy(self.initial_obs)
        self.last_action = np.zeros((1, 7))
        self.frames = []

        #print(self.env[self.task_suite][self.task_id].metadata)
        #print(self.env[self.task_suite][self.task_id].envs[0].unwrapped._env.obj_of_interest)
        #print(self.env[self.task_suite][self.task_id].envs[0].unwrapped._env.sim.model.body_names)
        #print(self.env[self.task_suite][self.task_id].envs[0].unwrapped._env.sim.data.get_body_xpos("object_name"))
        #print(self.env[self.task_suite][self.task_id])
        #exit()
        print('TASK:', self._get_libero_task_description())
        self._load_vla_policy()
        #self._load_reward_model()
        '''for timestep in range(kwargs["start_idx"]):
            print(kwargs["actions"][timestep])
            self.env[self.task_suite][self.task_id].step(actions=np.asarray(kwargs["actions"][timestep]))'''
        return [
            *self._get_current_observation(),
            {
                "type": "text",
                "text": "Environment task: " + self._get_libero_task_description()
            }
        ]
    
    def _load_vla_policy(self):
        global policy
        if policy == None:
            policy = SmolVLAPolicy.from_pretrained(pretrained_name_or_path=policy_path)
            policy.config.n_action_steps = 1
        policy_preprocessor, policy_postprocessor = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=policy_path)
        self.policy = policy
        self.policy_preprocessor = policy_preprocessor
        self.policy_postprocessor = policy_postprocessor
        self.policy.reset()
        #self.policy.to(device=torch.device('cpu'))

    def _unload_vla_policy(self):
        global policy
        policy = None
        self.policy = None
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def _load_reward_model(self):
        global reward_model
        if reward_model == None:
            reward_model_config = RobometerConfig(
                input_features={
                    "observation.images.top": PolicyFeature(type="VISUAL", shape=(256, 256, 3))
                },
                #vlm_name='Qwen/Qwen3-VL-2B-Instruct',
                device='cuda',
                image_key='observation.images.top',
                torch_dtype="float16"
            )
            reward_model = RobometerRewardModel.from_pretrained(pretrained_name_or_path="lerobot/Robometer-4B", config=reward_model_config)
        self.reward_model = reward_model
        self.reward_preprocessor = RobometerEncoderProcessorStep()
        #reward_preprocessor, reward_postprocessor = make_reward_pre_post_processors(reward_cfg=reward_model.config)
        #self.reward_preprocessor = reward_preprocessor
        #self.reward_postprocessor = reward_postprocessor
        #self.reward_model.to(device=torch.device('cpu'))

    def _unload_reward_model(self):
        global reward_model
        reward_model = None
        self.reward_model = None
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def _load_description_generator(self):
        self.description_generator = pipeline(task="image-text-to-text", model="Qwen/Qwen3.5-0.8B")

    def _unload_description_generator(self):
        self.description_generator = None
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
    
    def get_reward(self)->float:
        self._unload_vla_policy()
        self._load_reward_model()
        if len(self.frames) > 0:
            rm_result = self._compute_reward_model(prompt=self._get_libero_task_description(), is_subtask=False, frames=torch.as_tensor(np.concatenate(self.frames, axis=0)).unsqueeze(0))
        else:
            rm_result = (False, 0.0)
        reward = 100.0*self.task_reward + 10.0*rm_result[1]# + self.stage_reward # + 0.5*self._compute_reward_model(self._get_libero_task_description(), is_subtask=False)[1] + 0.5*self.subtask_reward
        self.prev_obs = deepcopy(self.obs)
        self._unload_reward_model()
        return reward
    
    def _compute_reward_model(self, prompt: str, is_subtask=True, frames: torch.Tensor=None):
        #self._load_reward_model()
        '''if is_subtask == True:
            prev_obs = preprocess_observation(observations=self.prev_obs)
        else:
            prev_obs = preprocess_observation(observations=self.initial_obs)
        prev_obs = self.env_preprocessor(prev_obs)
        obs = preprocess_observation(observations=self.obs)
        obs = self.env_preprocessor(obs)
        frames = torch.cat([prev_obs["observation.images.image"], obs["observation.images.image"]]).unsqueeze(0)  # (1, T, C, H, W)'''
        transition = {
            TransitionKey.OBSERVATION: {"observation.images.top": frames},
            TransitionKey.COMPLEMENTARY_DATA: {"task": prompt},
        }
        encoded = self.reward_preprocessor(transition)
        obs_new = encoded[TransitionKey.OBSERVATION]
        batch = {
            key: value.to('cuda') if isinstance(value, torch.Tensor) else value for key, value in obs_new.items()
        }

        with torch.no_grad():
            self.reward_model.config.reward_output = "progress"
            reward = self.reward_model.compute_reward(batch)
            self.reward_model.config.reward_output = "success"
            success = self.reward_model.compute_reward(batch)
        reward = reward.item()
        print(reward)
        #reward = torch.sum(reward).item()
        #success = reward >= 0.5
        success = success.item() > 0.95
        #self._unload_reward_model()
        print(success, reward)
        return success, reward
    
    def _get_libero_task_description(self)->str:
        """
        Get the current task from the Libero environment.

        Returns:
            The current task description.
        """
        return self.env[self.task_suite][self.task_id].call('task_description')[0]
    
    def _get_current_observation(self, generate_text_description=False)->list:
        """
        Get the current visual observation from the Libero environment.

        Returns:
            The image of the current visual observation.
        """
        print('GET OBS')
        img = Image.fromarray(cv2.resize(self.obs['pixels']['image'][0][::-1,::-1,:].copy(), (448, 448)))
        #img.save('debug.jpg')
        return_message = [
            {
                "type": "text",
                "text": "Current observation image:"
            },
            {
                "type": "image",
                "image": img
            }
        ]
        if generate_text_description == True:
            self._load_description_generator()
            caption_message = {"role": "user", "content": [
                {
                    "type": "text",
                    "text": "Generate a descriptive caption of the image, complete with points and bounding box annotations."
                },
                {
                    "type": "image",
                    "image": img
                }
            ]}
            self.description_generator
            description = self.description_generator(images=[img], text=caption_message, return_full_text=False)[0]["generated_text"]
            return_message.append({
                "type": "text",
                "text": f"Image description: {description}"
            })
            self._unload_description_generator()
        return return_message
    
    '''def detect_with_yolo_world(self, classes: list[str])->list:
        """
        Executes ultralytics YOLO World predict method on the current visual observation.

        Args:
            classes: List of class names to detect.
        
        Returns:
            The output from the object detection predict method
        """
        img = Image.fromarray(cv2.resize(self.obs['pixels']['image'][0][::-1,::-1,:].copy(), (448, 448)))
        object_detector.set_classes(classes=classes)
        return [
            {
                "type": "text",
                "text": object_detector.predict(source=img)
            }
        ]'''
    
    def _mat2euler(self, mat: np.ndarray):
        return Rotation.from_matrix(matrix=mat).as_euler(seq="xyz")
    
    def _return_to_home_position(self)->list:
        """
        Return the end-effector of the robotic arm in the Libero environment to its initial position.
        You can use this between two VLA subtasks in order to stitch together different behaviors of the policy.
        """
        print('RETURN TO HOME')
        target_position = self.initial_obs["robot_state"]["eef"]["pos"][0]
        current_position = self.obs["robot_state"]["eef"]["pos"][0]
        target_rotation = self._mat2euler(np.array(self.initial_obs["robot_state"]["eef"]["mat"][0]))
        current_rotation = self._mat2euler(np.array(self.obs["robot_state"]["eef"]["mat"][0]))
        kp = 100.0
        kd = 10.0
        dt = 0.002
        prev_pos_error = 0.0
        prev_rot_error = 0.0
        for timestep in range(20): #while np.linalg.norm(target_position - current_position) > 0.05 or np.linalg.norm(target_rotation - current_rotation) > 0.05:
            pos_error = timestep*(target_position - current_position)/150.0
            rot_error = timestep*(target_rotation - current_rotation)/150.0
            pos_derivative = (pos_error - prev_pos_error) / dt
            rot_derivative = (rot_error - prev_rot_error) / dt
            pos_output = (kp * pos_error) + (kd * pos_derivative)
            rot_output = (kp * rot_error) + (kd * rot_derivative)
            pos_action = pos_output - current_position
            rot_action = rot_output - current_rotation
            if timestep <= 100:
                self.last_action[0, :3] = pos_action
            else:
                self.last_action[0, 0] = 0.0
                self.last_action[0, 1] = 0.0
                self.last_action[0, 2] = 0.0
            if timestep > 100 and False:
                self.last_action[0, 3:6] = rot_action
            else:
                self.last_action[0, 3] = 0.0
                self.last_action[0, 4] = 0.0
                self.last_action[0, 5] = 0.0
            self.last_action[0, 6] = -1.0
            obs, reward, terminated, truncated, info = self.env[self.task_suite][self.task_id].step(self.last_action)
            self._get_current_observation()
            self.obs = obs
            self.info = info
            print(terminated)
            print(truncated)
            print(info)
            if terminated[0] or truncated[0] or self.info['is_success'][0]:
                break
            prev_pos_error = pos_error
            prev_rot_error = rot_error
            current_position = self.obs["robot_state"]["eef"]["pos"][0]
            current_rotation = self._mat2euler(self.obs["robot_state"]["eef"]["mat"][0])
        return self._get_current_observation()
    
    def _return_to_last_checkpoint(self)->list:
        print('RETURN TO CHECKPOINT')
        target_position = self.last_checkpoint["robot_state"]["eef"]["pos"][0]
        current_position = self.obs["robot_state"]["eef"]["pos"][0]
        target_rotation = self._mat2euler(np.array(self.last_checkpoint["robot_state"]["eef"]["mat"][0]))
        current_rotation = self._mat2euler(np.array(self.obs["robot_state"]["eef"]["mat"][0]))
        kp = 100.0
        kd = 10.0
        dt = 0.002
        prev_pos_error = 0.0
        prev_rot_error = 0.0
        for timestep in range(20): #while np.linalg.norm(target_position - current_position) > 0.05 or np.linalg.norm(target_rotation - current_rotation) > 0.05:
            pos_error = timestep*(target_position - current_position)/150.0
            rot_error = timestep*(target_rotation - current_rotation)/150.0
            pos_derivative = (pos_error - prev_pos_error) / dt
            rot_derivative = (rot_error - prev_rot_error) / dt
            pos_output = (kp * pos_error) + (kd * pos_derivative)
            rot_output = (kp * rot_error) + (kd * rot_derivative)
            pos_action = pos_output - current_position
            rot_action = rot_output - current_rotation
            self.last_action[0, :3] = pos_action
            self.last_action[0, 3] = 0.0
            self.last_action[0, 4] = 0.0
            self.last_action[0, 5] = 0.0
            self.last_action[0, 6] = -1.0
            obs, reward, terminated, truncated, info = self.env[self.task_suite][self.task_id].step(self.last_action)
            self._get_current_observation()
            self.obs = obs
            self.info = info
            print(terminated)
            print(truncated)
            print(info)
            if terminated[0] or truncated[0] or self.info['is_success'][0]:
                break
            prev_pos_error = pos_error
            prev_rot_error = rot_error
            current_position = self.obs["robot_state"]["eef"]["pos"][0]
            current_rotation = self._mat2euler(self.obs["robot_state"]["eef"]["mat"][0])
        return self._get_current_observation()
    
    '''def execute_plan(self, steps: list[str])->list:
        """
        Executes a step-by-step plan inside the Libero environment.

        Args:
            steps: List of subtasks to execute in the environment.
            step_durations: List of float values where each float is the amount of seconds to execute the subtask for.
        
        Returns:
            The image of the visual observation after executing the plan.
        """
        steps = json.loads(steps)
        print('steps', steps)
        for step in steps:
            self._run_vla_policy(subtask=step, max_steps=100)
        return self.__get_current_observation()'''
    
    def _close_gripper(self):
        """
        Close the gripper of the robotic arm.
        """
        print('CLOSE GRIPPER')
        for timestep in range(10):
            self.last_action[0, 0] = 0.0
            self.last_action[0, 1] = 0.0
            self.last_action[0, 2] = 0.0
            self.last_action[0, 3] = 0.0
            self.last_action[0, 4] = 0.0
            self.last_action[0, 5] = 0.0
            self.last_action[0, 6] = 1.0
            obs, reward, terminated, truncated, info = self.env[self.task_suite][self.task_id].step(self.last_action)
            self._get_current_observation()
            self.obs = obs
            self.info = info
            if terminated[0] or truncated[0] or self.info['is_success'][0]:
                break
        #success, reward = self._compute_reward_model(prompt=self._get_libero_task_description(), is_subtask=True)
        #self.subtask_reward += reward
        self.prev_obs = deepcopy(self.obs)
        '''if (success == False and not (terminated[0] or truncated[0] or self.info['is_success'][0])) and (max_retries > 0):
            return_val =  self.run_vla_policy(subtask=prompt, max_retries=max_retries-1)
            self.prev_obs = deepcopy(self.obs)
            return return_val'''
        return [
            *self._get_current_observation(),
            #{
            #    "type": "text",
            #    "text": f"Subtask completed: {success}"
            #}
        ]
    
    def _open_gripper(self):
        """
        Open the gripper of the robotic arm.
        """
        print('OPEN GRIPPER')
        for timestep in range(10):
            self.last_action[0, 0] = 0.0
            self.last_action[0, 1] = 0.0
            self.last_action[0, 2] = 0.0
            self.last_action[0, 3] = 0.0
            self.last_action[0, 4] = 0.0
            self.last_action[0, 5] = 0.0
            self.last_action[0, 6] = -1.0
            obs, reward, terminated, truncated, info = self.env[self.task_suite][self.task_id].step(self.last_action)
            self._get_current_observation()
            self.obs = obs
            self.info = info
            if terminated[0] or truncated[0] or self.info['is_success'][0]:
                break
        #success, reward = self._compute_reward_model(prompt=self._get_libero_task_description(), is_subtask=True)
        #self.subtask_reward += reward
        self.prev_obs = deepcopy(self.obs)
        '''if (success == False and not (terminated[0] or truncated[0] or self.info['is_success'][0])) and (max_retries > 0):
            return_val =  self.run_vla_policy(subtask=prompt, max_retries=max_retries-1)
            self.prev_obs = deepcopy(self.obs)
            return return_val'''
        return [
            *self._get_current_observation(),
            #{
            #    "type": "text",
            #    "text": f"Subtask completed: {success}"
            #}
        ]
    
    def _move_arm(self, direction: Literal["left", "right", "forward", "backward", "up", "down"])->list:
        """
        Moves the end-effector of the robotic arm in the given direction.

        Args:
            direction: The direction to move the end-effector.
        """
        print('MOVE ARM', direction)
        direction_to_action = {
            "left": np.array([0.0, 1.0, 0.0]),
            "right": np.array([0.0, -1.0, 0.0]),
            "forward": np.array([1.0, 0.0, 0.0]),
            "backward": np.array([-1.0, 0.0, 0.0]),
            "up": np.array([0.0, 0.0, 1.0]),
            "down": np.array([0.0, 0.0, -1.0]),
        }
        for timestep in range(10):
            self.last_action[0, :3] = direction_to_action[direction]
            self.last_action[0, 3] = 0.0
            self.last_action[0, 4] = 0.0
            self.last_action[0, 5] = 0.0
            obs, reward, terminated, truncated, info = self.env[self.task_suite][self.task_id].step(self.last_action)
            self._get_current_observation()
            self.obs = obs
            self.info = info
            if terminated[0] or truncated[0] or self.info['is_success'][0]:
                break
        success, reward = self._compute_reward_model(prompt=self._get_libero_task_description(), is_subtask=True)
        self.subtask_reward += reward
        self.prev_obs = deepcopy(self.obs)
        '''if (success == False and not (terminated[0] or truncated[0] or self.info['is_success'][0])) and (max_retries > 0):
            return_val =  self.run_vla_policy(subtask=prompt, max_retries=max_retries-1)
            self.prev_obs = deepcopy(self.obs)
            return return_val'''
        return [
            *self._get_current_observation(),
            #{
            #    "type": "text",
            #    "text": f"Subtask completed: {success}"
            #}
        ]
    
    def _pick_and_place(self, object: str, destination: str)->list:
        """
        Pick an object with the robotic arm, and place it on the destination

        Args:
            object: The object to pick.
            destination: The location to place the object on.
        """
        print('PICK AND PLACE')
        return self.run_vla_policy(subtask=f"put the {object} on the {destination}")
    
    def _open_drawer(self, drawer: str)->list:
        """
        Open the specified drawer with the robotic arm.

        Args:
            drawer: The specific drawer to open.
        """
        print('OPEN DRAWER')
        return self.run_vla_policy(subtask=f"open the {drawer} drawer")
    
    def _close_drawer(self, drawer: str)->list:
        """
        Close the specified drawer with the robotic arm.

        Args:
            drawer: The specific drawer to close.
        """
        print('CLOSE DRAWER')
        return self.run_vla_policy(subtask=f"close the {drawer} drawer")
    
    def _open_door(self, door: str)->list:
        """
        Open the specified door with the robotic arm.

        Args:
            door: The specific door to open.
        """
        print('OPEN DOOR')
        return self.run_vla_policy(subtask=f"open the {door} door")
    
    def _close_door(self, door: str)->list:
        """
        Close the specified door with the robotic arm.

        Args:
            door: The specific door to close.
        """
        print('CLOSE DOOR')
        return self.run_vla_policy(subtask=f"close the {door} door")
    
    def _activate_stove(self)->list:
        """
        Activates the stove burner.
        """
        print('ACTIVATE STOVE')
        return self.run_vla_policy(subtask=f"turn on the stove burner")
    
    def _deactivate_stove(self)->list:
        """
        De-activates the stove burner.
        """
        print('DE-ACTIVATE STOVE')
        return self.run_vla_policy(subtask=f"turn off the stove burner")
    
    def run_vla_policy(self, subtask: str, direction_to_move: Literal["left", "right", "forward", "backward", "up", "down"], gripper_action: Literal["open", "close"])->list:
        """
        Move the robotic arm in the libero environment by prompting a pretrained vision-language-action model. You must break the environment task into small, short horizon, atomic subtasks, and feed only the next immediate subtask into the VLA.

        Args:
            subtask: The subtask to feed into the VLA model.
            direction_to_move: The direction to move the end-effector.
            gripper_action: If the VLA should open or close the gripper of the robot.
        """
        prompt = subtask
        max_steps = 50
        max_retries = 0
        #self._open_gripper()
        #self.return_to_home_position()
        print(f'RUN VLA: {prompt}, {max_steps}')
        self._load_vla_policy()
        self.policy.reset()
        self._get_current_observation()
        frames = []
        stage_success = False
        stage_reward = 0.0
        last_reward = 0.0
        initial_noise = torch.zeros((1, 50, 32))
        direction_to_action = {
            "left": torch.tensor([[0.0, 1.0, 0.0],]*50),
            "right": torch.tensor([[0.0, -1.0, 0.0],]*50),
            "forward": torch.tensor([[1.0, 0.0, 0.0],]*50),
            "backward": torch.tensor([[-1.0, 0.0, 0.0],]*50),
            "up": torch.tensor([[0.0, 0.0, 1.0],]*50),
            "down": torch.tensor([[0.0, 0.0, -1.0],]*50),
        }
        gripper_action_to_action = {
            "open": torch.tensor([[-1.0,],]*50),
            "close": torch.tensor([[1.0,],]*50),
        }
        initial_noise[:, :, :3] = direction_to_action[direction_to_move]
        initial_noise[:, :, 6:7] = gripper_action_to_action[gripper_action]
        for i in range(max_steps):
            obs = preprocess_observation(observations=self.obs)
            obs = self.env_preprocessor(obs)
            if i % 50 == 0:
                frames.append(obs["observation.images.image"])
                '''rm_result = self._compute_reward_model(prompt=subtask, is_subtask=True, frames=torch.as_tensor(np.concatenate(frames, axis=0)).unsqueeze(0))
                stage_success = rm_result[0]
                stage_reward += rm_result[1]
                if rm_result[1] > last_reward:
                    last_reward = rm_result[1]
                    self.last_checkpoint = deepcopy(self.obs)
                else:
                    break'''
            obs = self.policy_preprocessor({
                "observation.images.top": obs["observation.images.image"],
                "observation.images.wrist_image": obs["observation.images.image2"],
                "observation.state": obs["observation.state"],
                "task": prompt
            })
            action = self.policy.select_action(batch=obs, noise=initial_noise.cuda())
            action = self.policy_postprocessor(data=action)
            action = {"action": action}
            action = self.env_postprocessor(action)
            action = action["action"].detach().cpu().numpy()
            self.last_action = action
            self.actions.append(self.last_action)
            obs, reward, terminated, truncated, info = self.env[self.task_suite][self.task_id].step(self.last_action)
            self._get_current_observation()
            self.obs = obs
            self.info = info
            if terminated[0] or truncated[0] or self.info['is_success'][0] or stage_success:
                break
        if self.info['is_success'][0] == True:
            self.task_reward += 1.0
        self.stage_reward += stage_reward
        #self.task_reward += stage_reward
        self.frames += frames
        self._get_current_observation()
        #self._unload_vla_policy()
        #rm_result = self._compute_reward_model(prompt=subtask, is_subtask=True, frames=torch.as_tensor(np.concatenate(frames, axis=0)).unsqueeze(0))
        #success = rm_result[0]
        #progress = rm_result[1]*100.0
        #self.subtask_reward += reward
        self.prev_obs = deepcopy(self.obs)
        '''if (success == False and not (terminated[0] or truncated[0] or self.info['is_success'][0])) and (max_retries > 0):
            return_val =  self.run_vla_policy(subtask=prompt, max_retries=max_retries-1)
            self.prev_obs = deepcopy(self.obs)
            return return_val'''
        return [
            *self._get_current_observation(),
            #{
            #    "type": "text",
            #    "text": f"Subtask completed: {success}. Subtask progress: {progress:.2f}%"
            #}
        ]


# ==========================================
# 3. Dataset Preprocessing
# ==========================================
def preprocess_dataset():
    """Loads and shapes dataset to match TRL's conversational template requirements."""
    dataset_metadata = LeRobotDatasetMetadata(repo_id=dataset_path)
    dataset = []
    #env = LiberoEnv()
    for seed in tqdm.trange(100):
        task_id = random.choice(list(range(1)))
        task_suite = "libero_10"
        '''env.reset(task_suite=task_suite, task_id=task_id, seed=seed, start_idx=0)
        for j in range(10):
            env.run_vla_policy(subtask=env._get_libero_task_description())'''
        dataset += [
            {
                "prompt": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": f"You are the high level planner component of a hierarchical vision-language-action model controlling a robotic arm. DO NOT THINK OR TALK, JUST EXECUTE TOOL CALLS!"}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Execute the high level task given by the Libero environment."}
                        ]
                    }
                ],
                "task_suite": "libero_10",
                "task_id": task_id,
                "seed": seed,
                #"actions": env.actions,
                #"start_idx": random.choice(list(range(len(env.actions))))
            }
        ]

    return Dataset.from_list(mapping=dataset)

# ==========================================
# 4. Main Training Loop
# ==========================================

def main():
    global trainer
    '''debug_env = LiberoEnv()
    debug_env.reset(task_suite="libero_10", task_id=0, seed=0)
    task = debug_env._get_libero_task_description()
    debug_env._get_current_observation()
    #debug_env.return_to_home_position()
    debug_env.run_vla_policy(subtask=task)
    debug_env.return_to_home_position()
    exit()'''
    # Load raw conversational dataset
    train_dataset = preprocess_dataset()
    
    # Initialize processor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.min_pixels = 256*256
    processor.max_pixels = 256*256

    # Configure LoRA to save massive amounts of VRAM
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        #bias="none",
        #task_type="CAUSAL_LM"
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4"
    )

    # Configure GRPO Engine
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        per_device_train_batch_size=1, # Adjust based on your VRAM
        gradient_accumulation_steps=8,
        max_steps=100,
        generation_kwargs=dict(
            pad_token_id=processor.tokenizer.pad_token_id
        ),
        chat_template_kwargs=dict(
            enable_thinking=False,
        ),
        max_completion_length=64*(500/50),
        use_liger_kernel=False,
        
        # GRPO Specific configuration settings
        num_generations=8,             # Number of completions to sample per prompt (G parameter)
        #max_prompt_length=512,
        #max_completion_length=16000,    # Space for complex reasoning/thinking block
        #vllm_max_model_length=4096,
        #max_completion_length=4096,
        max_tool_calling_iterations=10,

        # Generation Acceleration with colocated vLLM 
        use_vllm=False,
        #vllm_mode="colocate",          # Shares GPU seamlessly inside the trainer process
        #vllm_gpu_memory_utilization=0.1, 
        
        logging_steps=1,
        log_completions=True,
        #fp16=True,                     # FlashAttention requires bf16/fp16
        bf16=True,
        remove_unused_columns=False,   # Keep visual payload columns intact
        report_to="wandb"               # Set to "wandb" if logging metrics
    )

    # Initialize GRPOTrainer explicitly supporting vision inputs
    trainer = GRPOTrainer(
        model=MODEL_ID,                # Passes string path so vLLM can load base model internally
        environment_factory=LiberoEnv,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        quantization_config=quantization_config,
        processing_class=processor     # Automatically maps image tokens and resizes arrays
    )


    # Launch model optimization
    trainer.train()
    
    # Save optimized LoRA adapter weights
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()
