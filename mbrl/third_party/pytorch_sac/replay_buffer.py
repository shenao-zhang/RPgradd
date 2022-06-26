import numpy as np
import numpy.random as npr
import torch
from sortedcontainers import SortedSet


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.done_idxs = SortedSet()
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        if done:
            self.done_idxs.add(self.idx)
        elif self.full:
            self.done_idxs.discard(self.idx)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_batch(self, obs, action, reward, next_obs, done, done_no_max):
        def copy_from_to(buffer_start, batch_start, how_many):
            buffer_slice = slice(buffer_start, buffer_start + how_many)
            batch_slice = slice(batch_start, batch_start + how_many)
            np.copyto(self.obses[buffer_slice], obs[batch_slice])
            np.copyto(self.actions[buffer_slice], action[batch_slice])
            np.copyto(self.rewards[buffer_slice], reward[batch_slice])
            np.copyto(self.next_obses[buffer_slice], next_obs[batch_slice])
            np.copyto(self.not_dones[buffer_slice], np.logical_not(done[batch_slice]))
            np.copyto(
                self.not_dones_no_max[buffer_slice],
                np.logical_not(done_no_max[batch_slice]),
            )

        _batch_start = 0
        buffer_end = self.idx + len(obs)
        if buffer_end > self.capacity:
            copy_from_to(self.idx, _batch_start, self.capacity - self.idx)
            _batch_start = self.capacity - self.idx
            self.idx = 0
            self.full = True

        _how_many = len(obs) - _batch_start
        copy_from_to(self.idx, _batch_start, _how_many)
        self.idx = (self.idx + _how_many) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(
            self.not_dones_no_max[idxs], device=self.device
        )

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

    def sample_multistep(self, batch_size, rollout_length):
        last_idx = self.capacity if self.full else self.idx
        last_idx -= rollout_length

        # raw here means the "coalesced" indices that map to valid
        # indicies that are more than T steps away from a done
        done_idxs_sorted = np.array(list(self.done_idxs) + [last_idx])
        n_done = len(done_idxs_sorted)
        done_idxs_raw = done_idxs_sorted - np.arange(1, n_done+1) * rollout_length

        samples_raw = npr.choice(
            last_idx-(rollout_length + 1) * n_done, size=batch_size,
            replace=True # for speed
        )
        samples_raw = sorted(samples_raw)
        js = np.searchsorted(done_idxs_raw, samples_raw)
        offsets = done_idxs_raw[js] - samples_raw + rollout_length
        start_idxs = done_idxs_sorted[js] - offsets

        obses, actions, rewards, next_obses = [], [], [], []
        for h in range(rollout_length):
            obses.append(self.obses[start_idxs + h])
            actions.append(self.actions[start_idxs + h])
            rewards.append(self.rewards[start_idxs + h])
            next_obses.append(self.next_obses[start_idxs + h])
            assert np.all(self.not_dones[start_idxs + h])

        obses = np.stack(obses)
        actions = np.stack(actions)
        #rewards = np.stack(rewards).squeeze(2)
        rewards = np.stack(rewards)
        next_obses = np.stack(next_obses)

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        return obses, actions, rewards, next_obses