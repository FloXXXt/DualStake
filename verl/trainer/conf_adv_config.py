"""DualStake confidence-calibration reward configuration.

The default values reproduce the balanced configuration in the paper.
"""

import numpy as np

# Evidence confidence is the final <confidence> emitted after retrieval;
# answer confidence is the <final-confidence> emitted after the answer.
LSC_MODE = "stake"
FC_MODE = "stake"

# Balanced DualStake setting: alpha = beta = 0.25.
# Both coefficients are linearly warmed up from step 100 to step 300.
LSC_ALPHA_MAX = 0.25
LSC_START_STEP = 100
LSC_FULL_STEP = 300
FC_ALPHA_MAX = 0.25
FC_START_STEP = 100
FC_FULL_STEP = 300

# The paper uses margin clipping [m-, m+] = [0.1, 0.9].
STAKE_MARGIN_ENABLE = True
STAKE_CORRECT_MARGIN = 0.9
STAKE_WRONG_MARGIN = 0.1


# =============================================================================
# Curriculum 工具函数
# =============================================================================

def get_lsc_alpha(cur_step: int) -> float:
    """返回当前 step 对应的 lsc alpha 值（线性 warmup）"""
    if cur_step < LSC_START_STEP:
        return 0.0
    if cur_step >= LSC_FULL_STEP:
        return LSC_ALPHA_MAX
    ratio = (cur_step - LSC_START_STEP) / (LSC_FULL_STEP - LSC_START_STEP)
    return LSC_ALPHA_MAX * ratio


def get_fc_alpha(cur_step: int) -> float:
    """返回当前 step 对应的 fc alpha 值（线性 warmup）"""
    if cur_step < FC_START_STEP:
        return 0.0
    if cur_step >= FC_FULL_STEP:
        return FC_ALPHA_MAX
    ratio = (cur_step - FC_START_STEP) / (FC_FULL_STEP - FC_START_STEP)
    return FC_ALPHA_MAX * ratio


# =============================================================================
# Calib Reward 计算函数
# =============================================================================

def compute_calib_reward_stake(q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Stake-Calib：R_i^calib = (2*q_i - 1) * c_eff
    lsc 和 fc 的 stake 监督共用本函数。
    c = -1 的格式错误样本 calib 项为 0（中性）。
    q, c: shape (G,)，已归一化到 [0,1]，c=-1 表示无效

    当 STAKE_MARGIN_ENABLE=True 时启用松弛余量：
      答对（q>=0.5）：c_eff = min(c, STAKE_CORRECT_MARGIN)，超过上限后奖励封顶
      答错（q< 0.5）：c_eff = max(c, STAKE_WRONG_MARGIN)，低于下限后惩罚封底
    """
    G = len(q)
    r = np.zeros(G, dtype=np.float32)
    for i in range(G):
        if c[i] < 0:   # 格式错误
            r[i] = 0.0
        else:
            if STAKE_MARGIN_ENABLE:
                if q[i] >= 0.5:   # 答对：置信度奖励封顶
                    c_eff = min(c[i], STAKE_CORRECT_MARGIN)
                else:             # 答错：置信度惩罚封底
                    c_eff = max(c[i], STAKE_WRONG_MARGIN)
            else:
                c_eff = c[i]
            r[i] = (2.0 * q[i] - 1.0) * c_eff
    return r


def compute_fc_consistency_reward(
    c_fc: np.ndarray,
    c_lsc: np.ndarray,
    threshold: float = FC_CONSISTENCY_THRESHOLD,
) -> np.ndarray:
    """
    FC per-sample 方向一致性惩罚：
      s_fc  = sign(c_fc  - threshold)
      s_lsc = sign(c_lsc - threshold)
      R_i = -1  如果 s_fc != s_lsc 且两者方向均非 0（方向明确且相反）
      R_i =  0  其他情况（任意一方无效 / tie / 方向一致）
    范围：{-1, 0}，纯惩罚信号。
    c_fc, c_lsc: shape (G,)，已归一化到 [0,1]，-1 表示无效
    """
    G = len(c_fc)
    r = np.zeros(G, dtype=np.float32)
    for i in range(G):
        if c_fc[i] < 0 or c_lsc[i] < 0:   # 任意一方无效
            r[i] = 0.0
            continue
        s_fc  = 1.0 if c_fc[i]  > threshold else (-1.0 if c_fc[i]  < threshold else 0.0)
        s_lsc = 1.0 if c_lsc[i] > threshold else (-1.0 if c_lsc[i] < threshold else 0.0)
        if s_fc == 0.0 or s_lsc == 0.0:    # 任意一方恰好等于阈值，不惩罚
            r[i] = 0.0
        elif s_fc != s_lsc:                 # 方向相反
            r[i] = -1.0
        else:                               # 方向一致
            r[i] = 0.0
    return r


def compute_fc_rank_consistency_reward(
    c_fc: np.ndarray,
    c_lsc: np.ndarray,
    delta: float = FC_RANK_DELTA,
) -> np.ndarray:
    """
    FC group 内排序一致性（fc 与 lsc 的 Kendall τ，不涉及 q）：
      对每对 (i,j)，定义带 margin 的 soft sign：
        sgn_d(a,b) = +1 if a>b+d, -1 if a<b-d, 0 otherwise
      kappa_ij = sgn_d(c_fc_i, c_fc_j) * sgn_d(c_lsc_i, c_lsc_j)
               = 0 如果任意一方（fc 或 lsc）在该样本上为 -1（无效）
      R_i = (1/(G-1)) * Σ_{j≠i} kappa_ij
    范围：[-1, +1]，group 内零和（对称结构）。
    分母固定为 G-1（无效 pair 贡献 0 但不减分母）。
    c_fc, c_lsc: shape (G,)，已归一化到 [0,1]，-1 表示无效
    """
    G = len(c_fc)
    if G <= 1:
        return np.zeros(G, dtype=np.float32)

    def _sgn(a, b, d):
        if a > b + d:
            return 1.0
        elif a < b - d:
            return -1.0
        else:
            return 0.0

    r = np.zeros(G, dtype=np.float32)
    for i in range(G):
        for j in range(G):
            if i == j:
                continue
            # 任意一方无效，该 pair 贡献 0
            if c_fc[i] < 0 or c_lsc[i] < 0 or c_fc[j] < 0 or c_lsc[j] < 0:
                continue
            s_fc  = _sgn(c_fc[i],  c_fc[j],  delta)
            s_lsc = _sgn(c_lsc[i], c_lsc[j], delta)
            r[i] += s_fc * s_lsc
    r = r / (G - 1)
    return r.astype(np.float32)


# =============================================================================
# 主入口：双路 calib 叠加
# =============================================================================

def apply_dual_calib_to_scores(
    scores: "torch.Tensor",       # (batch_size,) 原始 outcome score（q）
    lsc_confs: "torch.Tensor",    # (batch_size,) last_step_confidences，-1 表示无效
    fc_confs: "torch.Tensor",     # (batch_size,) final_confidences，-1 表示无效
    uid_list: list,
    cur_step: int,
) -> "torch.Tensor":
    """
    双路 calib 叠加：
      R_i = q_i + lsc_alpha(t) * R_i^lsc  +  fc_alpha(t) * R_i^fc

    LSC_MODE / FC_MODE 均为 "none" 时直接返回原始 scores。
    lsc 和 fc 各自按独立 alpha curriculum warmup。
    """
    import torch

    lsc_alpha = get_lsc_alpha(cur_step) if LSC_MODE != "none" else 0.0
    fc_alpha  = get_fc_alpha(cur_step)  if FC_MODE  != "none" else 0.0

    if lsc_alpha == 0.0 and fc_alpha == 0.0:
        return scores

    new_scores = scores.clone()
    unique_uids = list(dict.fromkeys(uid_list))   # 保序去重

    for u in unique_uids:
        idx = [i for i, x in enumerate(uid_list) if x == u]
        G = len(idx)
        if G <= 1:
            continue

        q_arr   = np.array([scores[i].item()    for i in idx], dtype=np.float32)
        lsc_arr = np.array([lsc_confs[i].item() for i in idx], dtype=np.float32)
        fc_arr  = np.array([fc_confs[i].item()  for i in idx], dtype=np.float32)

        # 归一化 confidence 到 [0,1]（原始范围 0-10）；-1 保持 -1 表示无效
        c_lsc = np.where(lsc_arr >= 0, lsc_arr / 10.0, lsc_arr)
        c_fc  = np.where(fc_arr  >= 0, fc_arr  / 10.0, fc_arr)

        delta = np.zeros(G, dtype=np.float32)

        # ------ LSC 分量 ------
        if LSC_MODE == "stake" and lsc_alpha > 0.0:
            r_lsc = compute_calib_reward_stake(q_arr, c_lsc)
            delta += lsc_alpha * r_lsc

        # ------ FC 分量 ------
        if FC_MODE == "stake" and fc_alpha > 0.0:
            r_fc = compute_calib_reward_stake(q_arr, c_fc)
            delta += fc_alpha * r_fc
        elif FC_MODE == "consistency" and fc_alpha > 0.0:
            r_fc = compute_fc_consistency_reward(c_fc, c_lsc)
            delta += fc_alpha * r_fc
        elif FC_MODE == "rank_consistency" and fc_alpha > 0.0:
            r_fc = compute_fc_rank_consistency_reward(c_fc, c_lsc)
            delta += fc_alpha * r_fc

        for local_i, global_i in enumerate(idx):
            new_scores[global_i] = new_scores[global_i] + float(delta[local_i])

    score_delta = float((new_scores - scores).abs().mean())
    margin_str = (f"on(correct<={STAKE_CORRECT_MARGIN}, wrong>={STAKE_WRONG_MARGIN})"
                  if STAKE_MARGIN_ENABLE else "off")
    print(f"[DUAL_CALIB] step={cur_step}, lsc_mode={LSC_MODE}, fc_mode={FC_MODE}, "
          f"lsc_alpha={lsc_alpha:.4f}, fc_alpha={fc_alpha:.4f}, "
          f"margin={margin_str}, score_delta_mean={score_delta:.4f}")
    return new_scores


# =============================================================================
# 校准评估工具（供 ray_trainer 和 main_ppo 调用，不涉及奖励，保持不变）
# =============================================================================

def compute_ece(confidences: list, is_correct: list, n_bins: int = 20):
    """
    计算 Expected Calibration Error。
    confidences: List[Optional[float]]，原始值域 [1, 10]（integer），None 表示无效。
    is_correct:  List[bool]
    返回 float 或 None（有效样本不足时）
    """
    valid_pairs = [
        (c / 10.0, float(acc))
        for c, acc in zip(confidences, is_correct)
        if c is not None
    ]
    if len(valid_pairs) < n_bins:
        return None

    bin_conf_sums = np.zeros(n_bins)
    bin_acc_sums  = np.zeros(n_bins)
    bin_counts    = np.zeros(n_bins)

    for conf, acc in valid_pairs:
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bin_conf_sums[bin_idx] += conf
        bin_acc_sums[bin_idx]  += acc
        bin_counts[bin_idx]    += 1

    ece = 0.0
    total = len(valid_pairs)
    for b in range(n_bins):
        if bin_counts[b] > 0:
            avg_conf = bin_conf_sums[b] / bin_counts[b]
            avg_acc  = bin_acc_sums[b]  / bin_counts[b]
            ece += (bin_counts[b] / total) * abs(avg_conf - avg_acc)
    return float(ece)


def compute_auroc(confidences, is_correct):
    """
    纯 NumPy 计算 AUROC，考虑 None 和 ties。
    """
    valid_pairs = [(c, int(is_correct[i])) for i, c in enumerate(confidences) if c is not None]
    if len(valid_pairs) < 2:
        return None

    confs  = np.array([p[0] for p in valid_pairs], dtype=float)
    labels = np.array([p[1] for p in valid_pairs], dtype=int)

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    pos_confs = confs[labels == 1][:, None]
    neg_confs = confs[labels == 0][None, :]
    diff = pos_confs - neg_confs
    u_stat = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(u_stat / (n_pos * n_neg))
