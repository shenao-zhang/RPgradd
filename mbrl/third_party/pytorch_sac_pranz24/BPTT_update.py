import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import Optional, Sequence, cast
import mbrl.types

from mbrl.third_party.pytorch_sac_pranz24.model import (
    DeterministicPolicy,
    GaussianPolicy,
    QNetwork,
)
from mbrl.third_party.pytorch_sac_pranz24.utils import hard_update, soft_update


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class BPTT(object):
    def __init__(self, num_inputs, action_space, args):
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = args.device

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(
            device=self.device
        )
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(
            num_inputs, action_space.shape[0], args.hidden_size
        ).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                if args.target_entropy is None:
                    self.target_entropy = -torch.prod(
                        torch.Tensor(action_space.shape).to(self.device)
                    ).item()
                else:
                    self.target_entropy = args.target_entropy
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, batched=False, evaluate=False):
        state = torch.FloatTensor(state)
        if not batched:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        if batched:
            return action.detach().cpu().numpy()
        return action.detach().cpu().numpy()[0]

    def update_parameters(
        self, memory, batch_size, updates, logger=None, reverse_mask=False, mf_update=False,
            rollout_horizon=1, model_env=None,
    ):
        batch = memory.sample(batch_size)
        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            mask_batch,
        ) = batch.astuple()

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        if reverse_mask:
            mask_batch = mask_batch.logical_not()

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        if mf_update:
            pi, log_pi, _ = self.policy.sample(state_batch)
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            policy_loss = (
                (self.alpha * log_pi) - min_qf_pi
            ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        else:
            obs_batch, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
            model_state = model_env.reset(
                initial_obs_batch=cast(np.ndarray, obs_batch),
                return_as_np=True,
            )
            accum_dones = torch.zeros(obs_batch.shape[0], dtype=bool)
            accum_dones = accum_dones.to(self.device)
            rollout_horizon = 3
            for i in range(rollout_horizon):
                if self.device == "cpu":
                    obs_batch = torch.FloatTensor(obs_batch)
                else:
                    obs_batch = torch.cuda.FloatTensor(obs_batch)
                action, log_action, _ = self.policy.sample(obs_batch)
                pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
                    action, model_state, sample=True, no_grad=False
                )
                """
                if i == 0:
                    cum_rew = pred_rewards * (~accum_dones)
                elif i < rollout_horizon - 1:
                    cum_rew += self.gamma ** i * (pred_rewards * (~accum_dones))
                if i == rollout_horizon - 1:
                    pi, log_pi, _ = self.policy.sample(pred_next_obs)
                    #        with eval_mode(self.critic):
                    qf1_pi, qf2_pi = self.critic(pred_next_obs, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    cum_rew += self.gamma ** rollout_horizon * (min_qf_pi * (~accum_dones))
                """
                if i == 0:
                    log_pi = log_action
                    cum_rew = pred_rewards * (~accum_dones) - self.alpha * log_action
                elif i < rollout_horizon - 1:
                    cum_rew = cum_rew + self.gamma ** i * (pred_rewards * (~accum_dones) - self.alpha * log_action)

                if i == rollout_horizon - 1:
                    last_a, log_last_a, _ = self.policy.sample(pred_next_obs)
            #        with eval_mode(self.critic):
                    qf1_pi, qf2_pi = self.critic(pred_next_obs, last_a)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    cum_rew = cum_rew + self.gamma ** rollout_horizon * (min_qf_pi * (~accum_dones) - self.alpha * log_last_a)

                obs_batch = pred_next_obs
                accum_dones |= pred_dones.squeeze()
            policy_loss = -(cum_rew/rollout_horizon).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if logger is not None:
            logger.log("train/batch_reward", reward_batch.mean(), updates)
            logger.log("train_critic/loss", qf_loss, updates)
            logger.log("train_actor/loss", policy_loss, updates)
            if self.automatic_entropy_tuning:
                logger.log("train_actor/target_entropy", self.target_entropy, updates)
            else:
                logger.log("train_actor/target_entropy", 0, updates)
            logger.log("train_actor/entropy", -log_pi.mean(), updates)
            logger.log("train_alpha/loss", alpha_loss, updates)
            logger.log("train_alpha/value", self.alpha, updates)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
        )

    # Save model parameters
    def save_checkpoint(self, env_name=None, suffix="", ckpt_path=None):
        if ckpt_path is None:
            assert env_name is not None
            if not os.path.exists("checkpoints/"):
                os.makedirs("checkpoints/")
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print("Saving models to {}".format(ckpt_path))
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            },
            ckpt_path,
        )

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
