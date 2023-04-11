if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import copy
import tempfile
import time
from typing import Dict

import numpy as np
import torch
import zarr

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class R2D2Dataset(BaseImageDataset):
    def __init__(
        self,
        zarr_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        overfit_episodes=None,
    ):

        super().__init__()

        tmp_store_path = tempfile.TemporaryDirectory()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=["images", "joint_states", "actions"],
            store=zarr.LMDBStore(tmp_store_path.name),
        )

        # Convert state and actions from euler angles to rotation 6d.
        self.rotation_transformer = RotationTransformer(
            from_rep="euler_angles",
            to_rep="rotation_6d",
            from_convention="XYZ",
        )
        self.replay_buffer.root["data"][
            "actions"
        ] = self._convert_pos_euler_gripper_to_pos_rot6_gripper(
            self.replay_buffer.root["data"]["actions"],
        )
        self.replay_buffer.root["data"][
            "joint_states"
        ] = self._convert_pos_euler_gripper_to_pos_rot6_gripper(
            self.replay_buffer.root["data"]["joint_states"],
        )

        # Close the store and reopen in read only mode to allow for multiple workers.
        self.replay_buffer.root.store.close()
        self.replay_buffer.root = zarr.group(
            store=zarr.LMDBStore(tmp_store_path.name, readonly=True, lock=False),
        )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )

        if overfit_episodes is not None:
            val_mask = downsample_mask(
                mask=val_mask,
                max_n=1,
                seed=seed,
            )
            train_mask = downsample_mask(
                train_mask,
                max_n=overfit_episodes,
                seed=seed,
            )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            # episode_mask=~self.train_mask,
            episode_mask=self.val_mask,
        )
        # val_set.train_mask = ~self.train_mask
        val_set.train_mask = self.val_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["actions"],
            "joint_states": self.replay_buffer["joint_states"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["images"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _convert_pos_euler_gripper_to_pos_rot6_gripper(self, x):
        T = len(x)
        ret = np.zeros((T, 10), dtype=x.dtype)
        ret[:, :3] = x[:, :3]
        ret[:, 3:9] = self.rotation_transformer.forward(x[:, 3:6])
        ret[:, 9] = x[:, 6]
        return ret

    def _sample_to_data(self, sample):
        joint_states = sample["joint_states"].astype(np.float32)
        actions = sample["actions"].astype(np.float32)
        images = np.moveaxis(sample["images"], -1, 1) / 255

        data = {
            "obs": {
                "images": images,  # T, 3, 256, 256
                "joint_states": joint_states,  # T, 10
            },
            "action": actions,  # T, 10
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    zarr_path = "/home/datasets/r2d2/tasks/pick_up_ball/diffusion/all.zarr"
    dataset = R2D2Dataset(zarr_path, horizon=16)

    from matplotlib import pyplot as plt

    normalizer = dataset.get_normalizer()
    nactions = normalizer["action"].normalize(dataset.replay_buffer["actions"])
    diff = np.diff(nactions, axis=0)
    dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(nactions.shape[-1]):
        row = i // 3
        col = i % 3
        ax = axs[row][col]
        ax.hist(nactions[:, i], bins=50)
        ax.set_title(f"dim {i}")
    plt.savefig("/home/diffusion_policy/media/normalized_actions.png")
    plt.clf()

    normalizer = dataset.get_normalizer()
    nstates = normalizer["joint_states"].normalize(
        dataset.replay_buffer["joint_states"]
    )
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(nactions.shape[-1]):
        row = i // 3
        col = i % 3
        ax = axs[row][col]
        ax.hist(nstates[:, i], bins=50)
        ax.set_title(f"dim {i}")
    plt.savefig("/home/diffusion_policy/media/normalized_states.png")
    plt.clf()

    breakpoint()


if __name__ == "__main__":
    test()
