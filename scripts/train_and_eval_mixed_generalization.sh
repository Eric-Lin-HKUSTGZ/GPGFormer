#!/bin/bash

set -euo pipefail

NUM_GPUS=${1:-4}
CONFIG=${2:-configs/ablations_v2/datasets/config_mixed_generalization.yaml}
MASTER_PORT=${3:-29500}
CKPT=${4:-}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

cd "${REPO_DIR}"

echo "Starting mixed-domain training for generalization"
echo "GPUs: ${NUM_GPUS}"
echo "Config: ${CONFIG}"
echo "Master port: ${MASTER_PORT}"

torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" train_mixed.py --config "${CONFIG}"

if [ -z "${CKPT}" ]; then
  CKPT=$(python -c 'import sys, yaml; from pathlib import Path; cfg = yaml.safe_load(Path(sys.argv[1]).read_text()); print(Path(cfg["train"]["out_dir"]) / cfg["dataset"]["name"] / "gpgformer_best.pt")' "${CONFIG}")
fi

echo "Using checkpoint: ${CKPT}"

OUTPUT_DIR=$(python -c 'import sys; from pathlib import Path; ckpt = Path(sys.argv[1]).resolve(); print((ckpt.parent.parent if len(ckpt.parents) >= 2 else ckpt.parent) / "multi_eval")' "${CKPT}")

python eval_multi_dataset.py --config "${CONFIG}" --ckpt "${CKPT}" --output-dir "${OUTPUT_DIR}"
