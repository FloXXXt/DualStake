"""
计算 final confidence 和 last-step confidence 的 ECE / AUROC / Brier Score。

过滤条件：
  - final confidence 有效：confidence >= 1（排除 confidence=0，即模型未输出 <final-confidence> 标签的样本）
  - last-step confidence 有效：per_step_conf_values 中 answer 步骤有非 None 值
  - 只保留两者同时有效的样本（交集）

格式正确（has_valid_format=1）的子集单独统计。
"""
import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_metrics_ext import compute_ece, compute_auroc, compute_brier_score, print_bin_detail


def get_last_step_conf(row):
    """
    从 per_step_conf_values 和 per_step_action_types 中提取 answer 步骤的置信度。
    优先取最后一个 action_type=='answer' 且 conf 非 None 的步骤；
    若无，取最后一个非 None 的置信度。
    """
    try:
        conf_values = ast.literal_eval(str(row["per_step_conf_values"]))
        action_types = ast.literal_eval(str(row["per_step_action_types"]))
    except Exception:
        return None

    if not conf_values or not action_types:
        return None

    for i in range(len(action_types) - 1, -1, -1):
        if i < len(conf_values) and action_types[i] == "answer" and conf_values[i] is not None:
            return conf_values[i]

    for i in range(len(conf_values) - 1, -1, -1):
        if conf_values[i] is not None:
            return conf_values[i]

    return None


def print_metrics(label, confs, is_correct, n_bins, detail=False):
    """打印一组样本的校准指标。"""
    n = len(confs)
    if n == 0:
        print(f"[{label}] 无有效样本，跳过。")
        return

    overall_acc = is_correct.mean()
    avg_conf = confs.mean()
    ece   = compute_ece(confs, is_correct, n_bins=n_bins)
    auroc = compute_auroc(confs, is_correct)
    brier = compute_brier_score(confs, is_correct)

    print(f"\n{'='*55}")
    print(f"  {label}  (n={n})")
    print(f"{'='*55}")
    print(f"  准确率:            {overall_acc:.4f}")
    print(f"  平均置信度(归一化): {avg_conf:.4f}")
    print(f"  置信度范围:         [{confs.min():.2f}, {confs.max():.2f}]")
    print(f"  ECE   (↓ 20桶):   {ece:.4f}")
    print(f"  AUROC (↑):         {auroc:.4f}")
    print(f"  Brier Score (↓):  {brier:.4f}")

    correct_confs   = confs[is_correct == 1]
    incorrect_confs = confs[is_correct == 0]
    if len(correct_confs) > 0:
        print(f"  答对 {len(correct_confs)} 样本，平均置信度: {correct_confs.mean():.4f}")
    if len(incorrect_confs) > 0:
        print(f"  答错 {len(incorrect_confs)} 样本，平均置信度: {incorrect_confs.mean():.4f}")

    if detail:
        print_bin_detail(confs, is_correct, n_bins=n_bins)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    type=str, required=True)
    parser.add_argument("--n_bins", type=int, default=20)
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()

    print(f"\n读取文件：{args.csv}")
    df = pd.read_csv(args.csv)
    n_total = len(df)
    print(f"总样本数: {n_total}")
    print(f"has_valid_format 分布: {dict(df['has_valid_format'].value_counts().sort_index())}")
    print(f"confidence 分布: {dict(df['confidence'].value_counts().sort_index())}")

    # ------------------------------------------------------------------ #
    # 1. 提取 last-step confidence
    # ------------------------------------------------------------------ #
    df["last_step_conf"] = df.apply(get_last_step_conf, axis=1)
    df["last_step_conf"] = pd.to_numeric(df["last_step_conf"], errors="coerce")

    # ------------------------------------------------------------------ #
    # 2. 过滤：同时有效的样本（final conf >= 1 AND last_step_conf 非空）
    # ------------------------------------------------------------------ #
    mask_final = df["confidence"] >= 1           # 排除 confidence=0（未输出标签）
    mask_last  = df["last_step_conf"].notna()    # last-step 有值
    mask_both  = mask_final & mask_last

    df_both = df[mask_both].copy()
    n_both = len(df_both)
    print(f"\n同时有 final conf(>=1) 且有 last-step conf 的样本: {n_both} / {n_total} "
          f"({n_both/n_total*100:.1f}%)")
    print(f"  其中 has_valid_format=1: {df_both['has_valid_format'].sum()} / {n_both}")

    # ------------------------------------------------------------------ #
    # 3. 归一化 /10
    # ------------------------------------------------------------------ #
    df_both["final_conf_norm"]     = df_both["confidence"].astype(float) / 10.0
    df_both["last_conf_norm"]      = df_both["last_step_conf"].astype(float) / 10.0

    # ------------------------------------------------------------------ #
    # 4. 分两组：全部 & 格式正确
    # ------------------------------------------------------------------ #
    groups = {
        "【全部有效样本】": df_both,
        "【格式正确(has_valid_format=1)】": df_both[df_both["has_valid_format"] == 1],
    }

    print("\n\n" + "#"*60)
    print("# Final Confidence (answer步骤的 <final-confidence> 标签)")
    print("#"*60)
    for label, sub in groups.items():
        print_metrics(
            label,
            sub["final_conf_norm"].values,
            sub["is_correct"].values.astype(float),
            args.n_bins,
            args.detail,
        )

    print("\n\n" + "#"*60)
    print("# Last-Step Confidence (per_step_conf_values 中 answer 步骤的 <confidence> 标签)")
    print("#"*60)
    for label, sub in groups.items():
        print_metrics(
            label,
            sub["last_conf_norm"].values,
            sub["is_correct"].values.astype(float),
            args.n_bins,
            args.detail,
        )

    # ------------------------------------------------------------------ #
    # 5. 汇总对比表
    # ------------------------------------------------------------------ #
    print("\n\n" + "="*70)
    print("Summary comparison")
    print("="*70)
    header = f"{'子集':<28} {'类型':<12} {'ECE':>7} {'AUROC':>7} {'Brier':>7} {'n':>6}"
    print(header)
    print("-"*70)

    for label, sub in groups.items():
        n = len(sub)
        if n == 0:
            continue
        for conf_type, col in [("Final Conf", "final_conf_norm"), ("Last-Step", "last_conf_norm")]:
            c = sub[col].values
            y = sub["is_correct"].values.astype(float)
            ece   = compute_ece(c, y, n_bins=args.n_bins)
            auroc = compute_auroc(c, y)
            brier = compute_brier_score(c, y)
            short_label = label.replace("【", "").replace("】", "")
            print(f"{short_label:<28} {conf_type:<12} {ece:>7.4f} {auroc:>7.4f} {brier:>7.4f} {n:>6}")
        print()


if __name__ == "__main__":
    main()
