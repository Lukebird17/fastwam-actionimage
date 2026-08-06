: "${NVCC_PREPEND_FLAGS:=}"
: "${NVCC_APPEND_FLAGS:=}"
export NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS

conda activate curobo
export PYTHONPATH=src
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

export WANDB_API_KEY="${WANDB_API_KEY:-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}"

export TRITON_CACHE_DIR="/tmp/${USER}/triton"
mkdir -p "$TRITON_CACHE_DIR"