# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict, Counter

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config): # seems never used?
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

def compute_reinforce_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
):
    scores = token_level_rewards.sum(dim=-1)
    advantages = scores.unsqueeze(-1) * response_mask
    returns = advantages
    return advantages, returns

def _uid_check_and_print(index, expect_n=None, strict=False, prefix="[reinforce_mean]", max_show=10):
    """
    index: list/np.array/torch tensor of uid (len = batch_size)
    expect_n: 期望每个 uid 出现次数（通常 = n_agent）
    strict: True 时如果不等于 expect_n 直接报错
    """
    uid_list = list(index)
    c = Counter(uid_list)

    counts = list(c.values())
    count_min = min(counts) if counts else 0
    count_max = max(counts) if counts else 0

    # 统计分布（前几个）
    dist = Counter(counts)
    top_counts = dist.most_common(5)

    print(f"{prefix}[uid_check] unique_uids={len(c)} batch={len(uid_list)} "
          f"count_min={count_min} count_max={count_max} top_counts={top_counts}")

    if expect_n is not None:
        bad = {k: v for k, v in c.items() if v != expect_n}
        if bad:
            sample_bad = list(bad.items())[:max_show]
            msg = (f"{prefix}[uid_check] group size mismatch: expect {expect_n}, "
                   f"bad_groups(first {max_show})={sample_bad}, unique_uids={len(c)}, batch={len(uid_list)}")
            if strict:
                raise RuntimeError(msg)
            else:
                print(msg)

def compute_reinforce_mean_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    expect_n: int = None,
    strict: bool = False,
    print_check: bool = False,
):
    """
    REINFORCE + 组内 baseline（减去同 uid 的均值）

    token_level_rewards: (bs, response_len)
    response_mask:      (bs, response_len)  0/1 mask（EOS 后为 0）
    index:              uid 列表/数组（len = bs）

    返回：
      advantages: (bs, response_len)  每条样本一个标量 advantage，复制到 token 维度并乘 mask
      returns:    (bs, response_len)  这里不训练 critic，所以 returns = advantages（保持接口一致）
    """
    if print_check or strict:
        _uid_check_and_print(index, expect_n=expect_n, strict=strict, prefix="[reinforce_mean]")

    # 1) 每条样本的总回报 R_i（outcome reward）
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    # 2) 组内 baseline: mu_uid
    #    用 python 分组写法，和你 grpo 的风格一致，最稳妥
    uid_list = list(index)
    id2scores = defaultdict(list)
    for i, uid in enumerate(uid_list):
        id2scores[uid].append(scores[i])

    id2mean = {}
    with torch.no_grad():
        for uid, s_list in id2scores.items():
            if len(s_list) == 0:
                id2mean[uid] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
            else:
                # s_list 是一堆 0-d tensor，stack 更稳
                id2mean[uid] = torch.stack(s_list).mean()

    # 3) A_i = R_i - mu_uid
    advantages_scalar = scores.clone()
    with torch.no_grad():
        for i, uid in enumerate(uid_list):
            advantages_scalar[i] = advantages_scalar[i] - id2mean[uid]

    # 4) 复制到 token 维度，并 mask
    advantages = advantages_scalar.unsqueeze(-1) * response_mask  # (bs, response_len)
    returns = advantages
    return advantages, returns


def _uid_group_mean_and_count(scores: torch.Tensor, uid: torch.Tensor):
    """
    scores: (bs,) float/0-1
    uid: (bs,) int/long
    returns:
      group_mean_per_sample: (bs,)  每条样本所属uid的组均值
      uid2count: dict(uid -> count) 用于debug
    """
    uid2vals = defaultdict(list)
    bsz = scores.shape[0]

    # 注意：uid 可能是 tensor，也可能是 numpy/int，统一转成 python int 做 key 更稳
    for i in range(bsz):
        key = int(uid[i])
        uid2vals[key].append(scores[i])

    uid2mean = {}
    uid2count = {}
    for key, vals in uid2vals.items():
        # vals 是 list[tensor]，stack 后 mean
        v = torch.stack(vals)
        uid2mean[key] = v.mean()
        uid2count[key] = len(vals)

    group_mean = torch.empty_like(scores)
    for i in range(bsz):
        key = int(uid[i])
        group_mean[i] = uid2mean[key]
    return group_mean, uid2count


def compute_reinforce_da_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    uid: torch.Tensor,
    w_min: float = 0.5,
    w_max: float = 1.0,
):
    """
    REINFORCE-DA:
      scores = sum(token_level_rewards)  (outcome reward: 0/1)
      group_mean = mean(scores within same uid)
      w = w_min + (w_max-w_min)*(1-group_mean)
      A_{t} = w * scores * mask_t
    Returns:
      advantages: (bs, response_len)
      returns: same as advantages (actor-only)
      debug: dict with uid2count / stats
    """
    # outcome scalar
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    with torch.no_grad():
        group_mean, uid2count = _uid_group_mean_and_count(scores, uid)
        # w in [w_min, w_max]
        w = w_min + (w_max - w_min) * (1.0 - group_mean)
        # 数值安全：可选 clamp
        w = torch.clamp(w, min=min(w_min, w_max), max=max(w_min, w_max))

    advantages = (w * scores).unsqueeze(-1) * response_mask
    returns = advantages

    debug = {
        "uid2count": uid2count,
        "scores_mean": float(scores.mean().item()),
        "group_mean_mean": float(group_mean.mean().item()),
        "w_min": float(w.min().item()),
        "w_max": float(w.max().item()),
        "w_mean": float(w.mean().item()),
    }
    return advantages, returns, debug


def compute_reinforce_mean_da_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    uid: torch.Tensor,
    w_min: float = 0.5,
    w_max: float = 1.0,
):
    """
    REINFORCE-MEAN-DA:
      scores = outcome
      group_mean = mean(scores within uid)
      w = linear difficulty weight in [w_min,w_max]
      A_t = w * (scores - group_mean) * mask_t
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    with torch.no_grad():
        group_mean, uid2count = _uid_group_mean_and_count(scores, uid)
        w = w_min + (w_max - w_min) * (1.0 - group_mean)
        w = torch.clamp(w, min=min(w_min, w_max), max=max(w_min, w_max))

    advantages = (w * (scores - group_mean)).unsqueeze(-1) * response_mask
    returns = advantages

    debug = {
        "uid2count": uid2count,
        "scores_mean": float(scores.mean().item()),
        "group_mean_mean": float(group_mean.mean().item()),
        "w_min": float(w.min().item()),
        "w_max": float(w.max().item()),
        "w_mean": float(w.mean().item()),
    }
    return advantages, returns, debug

def compute_grpo_da_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    uid,                      # 参数名改为 uid
    w_min: float = 0.5,
    w_max: float = 1.0,
    epsilon: float = 1e-6,
    expect_n: int = None,
    strict: bool = False,
    print_check: bool = False,
):
    """
    GRPO-DA:
      1) outcome score: R_i = sum(token_level_rewards_i)
      2) GRPO组内标准化: z_i = (R_i - mean_uid) / (std_uid + eps)
      3) 样本难度权重（按uid）：w_uid in [w_min, w_max]，用该uid组内平均回报(mean_uid)映射
      4) A_i = w_uid * z_i  （注意：必须在标准化之后乘，否则会被抵消）
      5) 复制到 token 维度，并乘 eos_mask
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)  # (bs,)

    # --- uid check（可选） ---
    if print_check or strict:
        _uid_check_and_print(uid, expect_n=expect_n, strict=strict, prefix="[grpo-da]", max_show=10)

    uid_list = list(uid)

    # --- 分组收集每个uid的 scores ---
    id2scores = defaultdict(list)
    for i, uid_val in enumerate(uid_list):  # 改：uid -> uid_val，避免覆盖参数
        id2scores[uid_val].append(scores[i])

    id2mean = {}
    id2std = {}

    with torch.no_grad():
        for uid_key, s_list in id2scores.items():  # 改：uid -> uid_key
            s = torch.stack(s_list)
            mu = s.mean()
            std = s.std(unbiased=False)
            if torch.isnan(std) or std.item() < epsilon:
                std = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)

            id2mean[uid_key] = mu
            id2std[uid_key] = std

        # 1) 先做 GRPO z-score
        z = scores.clone()
        for i, uid_val in enumerate(uid_list):  # 改：uid -> uid_val
            z[i] = (scores[i] - id2mean[uid_val]) / (id2std[uid_val] + epsilon)

        # 2) 再做 difficulty weight（按 uid）
        # 你目前 reward 是 EM(0/1)，mean_uid 就是这个 prompt 的"成功率"
        # mean 越低 -> 越难 -> 权重越接近 w_max
        w = torch.empty_like(scores)
        for i, uid_val in enumerate(uid_list):  # 改：uid -> uid_val
            m = id2mean[uid_val]
            wi = w_min + (w_max - w_min) * (1.0 - m)
            wi = torch.clamp(wi, min=min(w_min, w_max), max=max(w_min, w_max))
            w[i] = wi

        advantages_scalar = w * z  # 关键：在标准化之后乘

    advantages = advantages_scalar.unsqueeze(-1).expand(-1, response_length) * eos_mask
    returns = advantages

    debug = {
        "scores_mean": float(scores.mean().item()),
        "group_mean_mean": float(torch.stack([id2mean[u] for u in id2mean]).mean().item()) if len(id2mean) > 0 else 0.0,
        "w_min": float(w.min().item()),
        "w_max": float(w.max().item()),
        "w_mean": float(w.mean().item()),
    }
    if print_check:
        print(f"[grpo-da][debug] scores_mean={debug['scores_mean']:.4f} "
              f"group_mean_mean={debug['group_mean_mean']:.4f} "
              f"w_min={debug['w_min']:.4f} w_max={debug['w_max']:.4f} w_mean={debug['w_mean']:.4f}")
    return advantages, returns, debug




def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
