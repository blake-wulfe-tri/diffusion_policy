if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class R2D2Dataset(BaseImageDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):

        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=["images", "joint_states", "actions"]
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
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
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
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

    def _sample_to_data(self, sample):
        joint_states = sample["joint_states"].astype(np.float32)
        image = np.moveaxis(sample["images"], -1, 1) / 255

        data = {
            "obs": {
                "images": image,  # T, 3, 256, 256
                "joint_states": joint_states,  # T, 7
            },
            "action": sample["actions"].astype(np.float32),  # T, 7
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