# 3D-Diffusion-Policy/diffusion_policy_3d/dataset/xarm_dataset.py
import os
from typing import Dict, List
import numpy as np
import torch
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask
# 确保导入 SingleFieldLinearNormalizer
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class XarmDataset(BaseDataset):
    def __init__(self,
                 zarr_path: str,
                 horizon: int,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 obs_keys: List[str] = ['point_cloud', 'agent_pos'],
                 action_keys: List[str] = ['action'],
                 seed: int = 42,
                 val_ratio: float = 0.0,
                 max_train_episodes: int = None,
                 **kwargs):
        try:
            super().__init__()
        except TypeError as e:
            print(f"Warning: super().__init__() in XarmDataset failed. If BaseDataset expects arguments, they were not provided. Error: {e}")
            import torch.utils.data
            if not isinstance(self, torch.utils.data.Dataset):
                 torch.utils.data.Dataset.__init__(self)

        print(f"--- [XarmDataset __init__] ---")
        print(f"Received zarr_path: '{zarr_path}' (type: {type(zarr_path)})")
        print(f"Received horizon: {horizon}")
        print(f"Received pad_before: {pad_before}, pad_after: {pad_after}")
        print(f"Received obs_keys: {obs_keys}")
        print(f"Received action_keys: {action_keys}")
        print(f"Received seed: {seed}, val_ratio: {val_ratio}")
        print(f"Received kwargs: {kwargs}")

        if not zarr_path or not zarr_path.strip():
            raise ValueError(f"XarmDataset received an empty or invalid zarr_path: '{zarr_path}'")

        self.zarr_obs_key_map = {
            'point_cloud': 'point_cloud',
            'agent_pos': 'state'
        }
        self.zarr_action_key_map = {
            'action': 'action'
        }

        zarr_data_keys_to_load = sorted(list(set(list(self.zarr_obs_key_map.values()) + list(self.zarr_action_key_map.values()))))
        print(f"[XarmDataset __init__] Keys to load from Zarr: {zarr_data_keys_to_load}")
        
        expanded_zarr_path = os.path.expanduser(zarr_path)
        if not os.path.exists(expanded_zarr_path):
            raise FileNotFoundError(f"ERROR: [XarmDataset __init__] Expanded zarr_path '{expanded_zarr_path}' does not exist BEFORE calling ReplayBuffer.copy_from_path!")
        
        try:
            self.replay_buffer = ReplayBuffer.copy_from_path(expanded_zarr_path, keys=zarr_data_keys_to_load)
        except Exception as e:
            print(f"ERROR during ReplayBuffer.copy_from_path with zarr_path='{expanded_zarr_path}': {e}")
            raise e

        print(f"[XarmDataset __init__] ReplayBuffer loaded. Backend: {self.replay_buffer.backend}, Num episodes: {self.replay_buffer.n_episodes}, Num steps: {self.replay_buffer.n_steps}")

        if self.replay_buffer.n_episodes == 0:
            print("WARNING: Replay buffer has 0 episodes. Check Zarr data or val_ratio.")
            self.val_mask = np.array([], dtype=bool)
        else:
            self.val_mask = get_val_mask(
                n_episodes=self.replay_buffer.n_episodes,
                val_ratio=val_ratio,
                seed=seed)
        
        train_mask = ~self.val_mask
        
        if np.sum(train_mask) == 0 and self.replay_buffer.n_episodes > 0 :
             print(f"WARNING: No training episodes after val_split (val_ratio={val_ratio}). All {self.replay_buffer.n_episodes} episodes are in validation set.")
        
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask, 
        )
        print(f"[XarmDataset __init__] SequenceSampler created. Length: {len(self.sampler)}")

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.obs_keys = obs_keys # 保存策略期望的 obs_keys
        self.action_keys = action_keys # 保存策略期望的 action_keys
        self.val_ratio = val_ratio
        self.seed = seed
        self.zarr_path = zarr_path 
        print(f"--- [XarmDataset __init__] Finished Initializing Dataset ---")

    def get_normalizer(self) -> LinearNormalizer:
        print(f"--- [XarmDataset get_normalizer] ---")
        normalizer = LinearNormalizer()
        
        data_for_low_dim_normalization = dict()

        # Action
        # self.action_keys 来自 __init__ 的参数, 默认为 ['action']
        action_policy_key = self.action_keys[0] # 通常是 'action'
        action_zarr_key = self.zarr_action_key_map.get(action_policy_key)
        if action_zarr_key and action_zarr_key in self.replay_buffer:
            print(f"Fetching data for action ('{action_policy_key}') normalization from Zarr key: '{action_zarr_key}'")
            data_for_low_dim_normalization[action_policy_key] = self.replay_buffer[action_zarr_key][:]
        else:
            print(f"Warning: Action key '{action_policy_key}' or its Zarr mapping '{action_zarr_key}' not found in replay_buffer for normalization.")

        # Agent_pos (state from Zarr)
        # self.obs_keys 来自 __init__ 的参数, 默认为 ['point_cloud', 'agent_pos']
        agent_pos_policy_key = 'agent_pos' # 这是策略内部期望的键名
        if agent_pos_policy_key in self.obs_keys: # 确保 agent_pos 是我们关心的观测键
            agent_pos_zarr_key = self.zarr_obs_key_map.get(agent_pos_policy_key)
            if agent_pos_zarr_key and agent_pos_zarr_key in self.replay_buffer:
                print(f"Fetching data for '{agent_pos_policy_key}' normalization from Zarr key: '{agent_pos_zarr_key}'")
                data_for_low_dim_normalization[agent_pos_policy_key] = self.replay_buffer[agent_pos_zarr_key][:]
            else:
                print(f"Warning: Obs key '{agent_pos_policy_key}' or its Zarr mapping '{agent_pos_zarr_key}' not found in replay_buffer for normalization.")
        
        if not data_for_low_dim_normalization:
            print("Warning: No data found for low_dim normalization. Will only set identity for point_cloud if applicable.")
        else:
            normalizer.fit(data=data_for_low_dim_normalization, last_n_dims=1, mode='limits')
        
        # 为 point_cloud (以及其他不应被 LinearNormalizer fit 的高维数据) 设置恒等变换
        # 确保这些键在 self.obs_keys 中，并且我们确实有这些数据
        point_cloud_policy_key = 'point_cloud'
        if point_cloud_policy_key in self.obs_keys:
            point_cloud_zarr_key = self.zarr_obs_key_map.get(point_cloud_policy_key)
            if point_cloud_zarr_key and point_cloud_zarr_key in self.replay_buffer:
                print(f"Setting identity normalization for '{point_cloud_policy_key}'")
                # 获取点云数据的实际数据类型以创建匹配的恒等变换器
                # 通常点云数据是 float32
                pc_dtype_numpy = self.replay_buffer[point_cloud_zarr_key].dtype
                pc_dtype_torch = torch.from_numpy(np.array([1], dtype=pc_dtype_numpy)).dtype

                normalizer[point_cloud_policy_key] = SingleFieldLinearNormalizer.create_identity(
                    dtype=pc_dtype_torch
                )
            else:
                 print(f"Warning: Point cloud key '{point_cloud_policy_key}' (Zarr key: '{point_cloud_zarr_key}') not found in replay_buffer. Cannot set identity normalizer for it.")

        print(f"--- [XarmDataset get_normalizer] Finished. Normalizer fitted for keys: {list(data_for_low_dim_normalization.keys()) if data_for_low_dim_normalization else 'None'}. ---")
        return normalizer

    def get_validation_dataset(self):
        print(f"--- [XarmDataset get_validation_dataset] ---")
        if self.replay_buffer.n_episodes == 0:
            print("WARNING: Replay buffer has 0 episodes. Cannot create validation dataset sampler.")
            empty_sampler = SequenceSampler(
                replay_buffer=self.replay_buffer,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=np.array([], dtype=bool)
            )
            return ValidationDatasetWrapper(empty_sampler, self.obs_keys, self.action_keys, self.zarr_obs_key_map, self.zarr_action_key_map, "Validation (empty replay)")

        val_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        print(f"[XarmDataset get_validation_dataset] Validation sampler created. Length: {len(val_sampler)}")
        
        return ValidationDatasetWrapper(val_sampler, self.obs_keys, self.action_keys, self.zarr_obs_key_map, self.zarr_action_key_map, "Validation")

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # print(f"[XarmDataset __getitem__] Sampling index: {idx}")
        data_numpy = self.sampler.sample_sequence(idx)
        
        obs_data_torch = {}
        # 使用 self.obs_keys 来决定哪些观测值被处理并包含在输出中
        for policy_key in self.obs_keys:
            zarr_key_in_data = self.zarr_obs_key_map.get(policy_key)
            if zarr_key_in_data is None:
                 raise KeyError(f"Policy observation key '{policy_key}' not found in zarr_obs_key_map.")
            
            if zarr_key_in_data in data_numpy:
                # 确保数据是 float32，特别是对于点云和 agent_pos
                obs_data_torch[policy_key] = torch.from_numpy(data_numpy[zarr_key_in_data].astype(np.float32))
            else:
                raise KeyError(f"Key '{zarr_key_in_data}' (mapped from policy key '{policy_key}') not found in sampled data from Zarr. Available keys: {list(data_numpy.keys())}")
        
        # 准备策略期望的 action 张量
        # 使用 self.action_keys (通常只有一个元素 'action')
        action_policy_key = self.action_keys[0]
        action_zarr_key = self.zarr_action_key_map.get(action_policy_key)
        if action_zarr_key is None:
            raise KeyError(f"Policy action key '{action_policy_key}' not found in zarr_action_key_map.")

        if action_zarr_key in data_numpy:
            action_tensor = torch.from_numpy(data_numpy[action_zarr_key].astype(np.float32))
        else:
            raise KeyError(f"Action key '{action_zarr_key}' (mapped from policy key '{action_policy_key}') not found in sampled data from Zarr. Available keys: {list(data_numpy.keys())}")
            
        return {
            'obs': obs_data_torch,
            'action': action_tensor
        }

# 辅助类，用于创建验证集实例
class ValidationDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, sampler, obs_keys, action_keys, zarr_obs_key_map, zarr_action_key_map, name="Validation"):
        self.sampler = sampler
        self.obs_keys = obs_keys
        self.action_keys = action_keys
        self.zarr_obs_key_map = zarr_obs_key_map
        self.zarr_action_key_map = zarr_action_key_map
        self.name = name
        print(f"--- [{self.name} ValidationDatasetWrapper __init__] Sampler length: {len(self.sampler)} ---")

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_numpy = self.sampler.sample_sequence(idx)
        
        obs_data_torch = {}
        for policy_key in self.obs_keys:
            zarr_key_in_data = self.zarr_obs_key_map.get(policy_key)
            if zarr_key_in_data is None:
                 raise KeyError(f"[{self.name}] Policy observation key '{policy_key}' not found in zarr_obs_key_map.")

            if zarr_key_in_data in data_numpy:
                obs_data_torch[policy_key] = torch.from_numpy(data_numpy[zarr_key_in_data].astype(np.float32))
            else:
                raise KeyError(f"[{self.name}] Key '{zarr_key_in_data}' (for policy key '{policy_key}') not in Zarr sample. Available keys: {list(data_numpy.keys())}")
        
        action_policy_key = self.action_keys[0]
        action_zarr_key = self.zarr_action_key_map.get(action_policy_key)
        if action_zarr_key is None:
            raise KeyError(f"[{self.name}] Policy action key '{action_policy_key}' not found in zarr_action_key_map.")

        if action_zarr_key in data_numpy:
            action_tensor = torch.from_numpy(data_numpy[action_zarr_key].astype(np.float32))
        else:
            raise KeyError(f"[{self.name}] Action key '{action_zarr_key}' not in Zarr sample. Available keys: {list(data_numpy.keys())}")
            
        return {
            'obs': obs_data_torch,
            'action': action_tensor
        }