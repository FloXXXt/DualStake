#!/usr/bin/env bash
set -euo pipefail

# Start the local E5 retrieval service used by Search-R1 and DualStake.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INDEX_PATH="${INDEX_PATH:?Set INDEX_PATH to the Wikipedia retrieval index.}"
CORPUS_PATH="${CORPUS_PATH:?Set CORPUS_PATH to the matching Wikipedia JSONL corpus.}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
TOPK="${TOPK:-3}"
cd "$PROJECT_ROOT"
python search_r1/search/retrieval_server.py \
    --index_path "$INDEX_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --retriever_model "$RETRIEVER_MODEL" \
    --topk "$TOPK"
