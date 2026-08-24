"""
compute_metrics_ext.py

从 evaluate_batch_conf_ext.sh 生成的评估产物目录中，计算：
  1. bamboogle：读取 4 次 rep1~rep4 的 CSV，合并后整体计算指标（等效取均值）
  2. simpleqa ：读取单次 CSV，直接计算指标

每个数据集输出与 train.md 表格一致的指标：
  acc | final-conf ECE↓ | final-conf AUROC↑ | final-conf BS↓ |
  last-step-conf ECE↓ | last-step-conf AUROC↑ | last-step-conf BS↓

用法：
  # 计算单个模型产物目录
  python scripts/compute_metrics_ext.py --eval_dir evaluation/<model_name>_ext_<timestamp>

  # 批量计算所有 *_ext_* 目录
  python scripts/compute_metrics_ext.py --eval_root evaluation --all
"""

import argparse
import ast
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../statistics'))
from confidence_metrics import compute_ece, compute_auroc, compute_brier_score


# ============================================================
# last-step confidence 提取（同 calc_last_step_conf.py）
# ============================================================
def get_last_step_conf(row):
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


# ============================================================
# 从 sample_statistics 目录中找 CSV
# ============================================================
def find_csv(stats_dir: str) -> str | None:
    """在 stats_dir/sample_statistics/ 下找到第一个 *_val.csv 文件。"""
    pattern = os.path.join(stats_dir, "sample_statistics", "*_val.csv")
    files = glob.glob(pattern)
    if not files:
        # 有时直接在 stats_dir 下
        pattern2 = os.path.join(stats_dir, "sample_statistics", "*.csv")
        files = glob.glob(pattern2)
    if files:
        return sorted(files)[0]
    return None


# ============================================================
# 计算一个 DataFrame 的全套指标
# ============================================================
def compute_metrics_df(df: pd.DataFrame, n_bins: int = 20) -> dict:
    """
    返回字典：
      acc, n,
      fc_ece, fc_auroc, fc_bs,      # final-conf
      lsc_ece, lsc_auroc, lsc_bs    # last-step-conf
    """
    # last-step conf
    df = df.copy()
    df["last_step_conf"] = df.apply(get_last_step_conf, axis=1)
    df["last_step_conf"] = pd.to_numeric(df["last_step_conf"], errors="coerce")

    # 过滤：final conf valid (confidence_valid==1 且 confidence>=1)
    mask_fc = (df["confidence_valid"] == 1) & (pd.to_numeric(df["confidence"], errors="coerce") >= 1)
    mask_lsc = df["last_step_conf"].notna()
    mask_both = mask_fc & mask_lsc

    df_v = df[mask_both].copy()
    df_v["confidence"] = pd.to_numeric(df_v["confidence"], errors="coerce")
    df_v["fc_norm"]  = df_v["confidence"].astype(float) / 10.0
    df_v["lsc_norm"] = df_v["last_step_conf"].astype(float) / 10.0

    n = len(df_v)
    if n == 0:
        nan = float("nan")
        return dict(acc=nan, n=0,
                    fc_ece=nan, fc_auroc=nan, fc_bs=nan,
                    lsc_ece=nan, lsc_auroc=nan, lsc_bs=nan)

    y  = df_v["is_correct"].values.astype(float)
    fc = df_v["fc_norm"].values
    lc = df_v["lsc_norm"].values

    acc = df["is_correct"].values.astype(float).mean()   # 整体 acc（不过滤）

    return dict(
        acc     = float(acc),
        n       = n,
        fc_ece  = compute_ece(fc, y, n_bins),
        fc_auroc= compute_auroc(fc, y),
        fc_bs   = compute_brier_score(fc, y),
        lsc_ece = compute_ece(lc, y, n_bins),
        lsc_auroc= compute_auroc(lc, y),
        lsc_bs  = compute_brier_score(lc, y),
    )


# ============================================================
# 处理单个 eval_dir（一个模型的扩展评估产物目录）
# ============================================================
def process_eval_dir(eval_dir: str, n_bins: int = 20):
    model_name = os.path.basename(eval_dir)
    print(f"\n{'='*70}")
    print(f"模型目录: {model_name}")
    print(f"{'='*70}")

    results = {}

    # ---- bamboogle: 收集所有 rep 的 CSV ----
    bamboogle_csvs = []
    for rep in range(1, 10):
        rep_dir = os.path.join(eval_dir, "bamboogle", f"rep{rep}")
        if not os.path.isdir(rep_dir):
            break
        csv_path = find_csv(rep_dir)
        if csv_path:
            bamboogle_csvs.append(csv_path)
        else:
            print(f"  ⚠ bamboogle rep{rep}: 未找到 CSV（{rep_dir}）")

    if bamboogle_csvs:
        n_reps = len(bamboogle_csvs)
        print(f"\nbamboogle: 找到 {n_reps} 次重复")
        dfs = [pd.read_csv(f) for f in bamboogle_csvs]
        df_all = pd.concat(dfs, ignore_index=True)
        m = compute_metrics_df(df_all, n_bins)
        results["bamboogle"] = m
        print(f"  合并样本数 (valid conf): {m['n']}  (每次 {len(dfs[0])} 条 × {n_reps} 次)")
        _print_row("bamboogle", m)
    else:
        print("  ⚠ bamboogle: 无 CSV，跳过")

    # ---- simpleqa: 单次 ----
    simpleqa_dir = os.path.join(eval_dir, "simpleqa")
    if os.path.isdir(simpleqa_dir):
        csv_path = find_csv(simpleqa_dir)
        if csv_path:
            print(f"\nsimpleqa: {csv_path}")
            df = pd.read_csv(csv_path)
            m = compute_metrics_df(df, n_bins)
            results["simpleqa"] = m
            print(f"  总样本数: {len(df)}  有效 conf: {m['n']}")
            _print_row("simpleqa", m)
        else:
            print(f"  ⚠ simpleqa: 未找到 CSV（{simpleqa_dir}）")
    else:
        print(f"  ⚠ simpleqa 目录不存在: {simpleqa_dir}")

    # ---- Markdown 格式输出 ----
    print(f"\n--- Markdown 表格行（可直接贴入 train.md）---")
    print(f"| | acc | final-conf ECE↓ | final-conf AUROC↑ | final-conf BS↓ | last-step-conf ECE↓ | last-step-conf AUROC↑ | last-step-conf BS↓ |")
    print(f"|---|---|---|---|---|---|---|---|")
    for ds, m in results.items():
        n_label = f"{m['n']}" if m['n'] else "?"
        label = f"Bamboogle ({len(bamboogle_csvs)}次均)" if ds == "bamboogle" else "SimpleQA"
        print(f"| {label} | {m['acc']:.3f} | {m['fc_ece']:.4f} | {m['fc_auroc']:.4f} | {m['fc_bs']:.4f} | {m['lsc_ece']:.4f} | {m['lsc_auroc']:.4f} | {m['lsc_bs']:.4f} |")

    return results


def _print_row(ds, m):
    print(f"  acc={m['acc']:.3f}  "
          f"fc_ECE={m['fc_ece']:.4f}  fc_AUROC={m['fc_auroc']:.4f}  fc_BS={m['fc_bs']:.4f}  "
          f"lsc_ECE={m['lsc_ece']:.4f}  lsc_AUROC={m['lsc_auroc']:.4f}  lsc_BS={m['lsc_bs']:.4f}")


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="计算 bamboogle/simpleqa 扩展评估指标")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--eval_dir", type=str,
                       help="单个模型的 evaluation 产物目录（如 evaluation/way1-stake_ext_20260515_120000）")
    group.add_argument("--eval_root", type=str,
                       help="evaluation 根目录，配合 --all 批量处理所有 *_ext_* 子目录")
    parser.add_argument("--all", action="store_true",
                        help="批量处理 eval_root 下所有 *_ext_* 目录")
    parser.add_argument("--n_bins", type=int, default=20)
    args = parser.parse_args()

    if args.eval_dir:
        process_eval_dir(args.eval_dir, args.n_bins)
    else:
        if not args.all:
            print("请指定 --all 以批量处理，或直接用 --eval_dir 指定单目录")
            sys.exit(1)
        pattern = os.path.join(args.eval_root, "*_ext_*")
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            print(f"未找到 *_ext_* 目录 in {args.eval_root}")
            sys.exit(1)
        for d in dirs:
            if os.path.isdir(d):
                process_eval_dir(d, args.n_bins)


if __name__ == "__main__":
    main()
