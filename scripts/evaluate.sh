#!/usr/bin/env bash
set -euo pipefail

# Evaluate a DualStake checkpoint on one prepared benchmark split.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a base model or trained actor checkpoint.}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to a directory containing train.parquet and test.parquet.}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/evaluation/$(basename "$DATA_DIR")}" 
NUM_GPUS="${NUM_GPUS:-4}"
NUM_NODES="${NUM_NODES:-1}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_data_num=null \
    data.val_data_num=1024 \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=3072 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.state_masking=true \
    algorithm.no_think_rl=false \
    trainer.logger='[]' \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes="$NUM_NODES" \
    max_turns=4 \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk=3
