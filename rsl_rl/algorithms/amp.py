import torch
import torch.nn as nn
# from torch.distributions import Normal
import numpy as np
from tensordict import TensorDict  
from torch import autograd

from rsl_rl.networks import EmpiricalNormalization

class AMP(nn.Module):
    """AMP algorithm config."""

    def __init__(
            self,
            input_dim,
            amp_reward_coef,
            task_reward_lerp,
            amp_obs_normalization=False,
            amp_hidden_dims=[1024,512],
            activation="relu",
            device="cpu",
            state_dependent_std=False,
            **kwargs,
        ):
            if kwargs:
              print(
                "AMP.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
              )
            super().__init__()
            self.device = device
            self.input_dim = input_dim

            amp_layers = []
            curr_in_dim = input_dim
            for hidden_dim in amp_hidden_dims:
                amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                amp_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            self.trunk = nn.Sequential(*amp_layers).to(device)
            self.amp_linear = nn.Linear(amp_hidden_dims[-1], 1).to(device)

            self.trunk.train()
            self.amp_linear.train()

            self.amp_reward_coef = amp_reward_coef
            self.task_reward_lerp = task_reward_lerp

            self.obs_noise_std = 0.0

    def update_amp_noise_level(self, current_iter, max_iters):
        """
        [新增] 动态更新噪声水平
        在训练初期给予较强噪声，随后线性衰减至 0。
        建议在 Runner 的 learn 循环中调用此函数。
        """
        # 配置参数：初始噪声强度和衰减轮数
        start_noise = 1.0   # 初始标准差，1.0 是个比较强的值，足以模糊掉初期微小的姿态差异
        decay_iters = 2000  # 在前 2000 轮内线性衰减，之后保持为 0
        
        if current_iter < decay_iters:
            self.obs_noise_std = start_noise * (1.0 - current_iter / decay_iters)
        else:
            self.obs_noise_std = 0.0

    def get_disc_obs(self, obs):
        """
        从 obs 字典中提取判别器 (AMP) 所需的观测数据。
        
        Args:
            obs (dict): 包含所有观测数据的字典 (TensorDict)。
            
        Returns:
            torch.Tensor: 拼接好的判别器输入数据。
        """
        target_key = "discriminator"
        
        # 2. 直接获取
        if target_key in obs:
            return obs[target_key]
        else:
            raise ValueError(f"在 obs 中找不到 '{target_key}'。现有的 keys: {list(obs.keys())}")
        
    def forward(self, x):
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self,
                         expert_state,
                         expert_next_state,
                         lambda_=10):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(
            self, state, next_state, task_reward, normalizer=None, dt=None):
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer(state)
                next_state = normalizer(next_state)

            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            style_reward = dt * self.amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0:
                total_reward = self._lerp_reward(style_reward, task_reward.unsqueeze(-1))
            else:
                total_reward = style_reward + task_reward.unsqueeze(-1)
            # total_reward = style_reward + task_reward.unsqueeze(-1)
            self.train()
        return total_reward.squeeze(), style_reward.squeeze()
    
    def _lerp_reward(self, disc_r, task_r):
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        # r = 4 * disc_r + task_r
        return r
    
    def update(self, policy_generator, expert_generator, optimizer, normalizer=None, num_updates=1):
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0

        for sample_amp_policy, sample_amp_expert in zip(policy_generator, expert_generator):
            # discriminator loss
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            if normalizer is not None:
                # # 先用原始数据更新统计信息
                # normalizer.update(policy_state)
                # normalizer.update(expert_state)

                with torch.no_grad():
                    policy_state = normalizer(policy_state)
                    policy_next_state = normalizer(policy_next_state)
                    # 使用当前的 normalizer 处理 expert 数据
                    expert_state_norm = normalizer(expert_state)
                    expert_next_state_norm = normalizer(expert_next_state)
            # else:
            #     expert_state_norm = expert_state
            #     expert_next_state_norm = expert_next_state

            policy_cat = torch.cat([policy_state, policy_next_state], dim=-1)
            expert_cat = torch.cat([expert_state_norm, expert_next_state_norm], dim=-1)

            # if self.obs_noise_std > 0:
            #     policy_cat += torch.randn_like(policy_cat) * self.obs_noise_std
            #     expert_cat += torch.randn_like(expert_cat) * self.obs_noise_std

            # # --- 开始计算梯度 ---
            # optimizer.zero_grad()

            # Policy Loss
            policy_d = self(policy_cat)
            policy_loss = torch.nn.MSELoss()(
                policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            
            # Expert Loss
            expert_d = self(expert_cat)
            expert_loss = torch.nn.MSELoss()(
                expert_d, torch.ones(expert_d.size(), device=self.device))
            
            # Gradient Penalty
            grad_pen_loss = self.compute_grad_pen(
                expert_state_norm, # split back to state
                expert_next_state_norm, # split back to next_state 
                lambda_=20)
            
            amp_loss = expert_loss + policy_loss + grad_pen_loss

            # gradient step
            optimizer.zero_grad()
            amp_loss.backward()
            optimizer.step()

            if normalizer is not None:
                # 用 expert 数据更新统计信息
                normalizer.update(policy_state)
                normalizer.update(expert_state)

            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates

        return {
            "loss": mean_amp_loss,
            "grad_pen": mean_grad_pen_loss,
            "policy_pred": mean_policy_pred,
            "expert_pred": mean_expert_pred,
            "noise_std": self.obs_noise_std
        }






class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim, buffer_size, device):
        """Initialize a ReplayBuffer object.
        Arguments:
            buffer_size (int): maximum size of buffer
        """
        self.states = torch.zeros(buffer_size, obs_dim).to(device)
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0
    
    def insert(self, states, next_states):
        """Add new states to memory."""
        
        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            self.states[self.step:self.buffer_size] = states[:self.buffer_size - self.step]
            self.next_states[self.step:self.buffer_size] = next_states[:self.buffer_size - self.step]
            self.states[:end_idx - self.buffer_size] = states[self.buffer_size - self.step:]
            self.next_states[:end_idx - self.buffer_size] = next_states[self.buffer_size - self.step:]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield (self.states[sample_idxs].to(self.device),
                   self.next_states[sample_idxs].to(self.device))
         