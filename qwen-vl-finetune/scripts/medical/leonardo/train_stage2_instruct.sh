#!/bin/bash

# =====================================================================================
# Qwen3-VL Medical Training Script - Stage 2 Alignment (Leonardo)
# =====================================================================================
# This script performs stage 2 instruction tuning training for medical vision-language models
# using Qwen2.5 like model.

# =====================================================================================
# SLURM CONFIGURATION
# =====================================================================================
#SBATCH --job-name=train-lingshu-7b-stage2-mixed
#SBATCH --output=/leonardo_work/AIFAC_L07_002/%u/slurm/%j.out
#SBATCH --error=/leonardo_work/AIFAC_L07_002/%u/slurm/%j.out
#SBATCH --mail-type END,FAIL,TIME_LIMIT
#SBATCH --mail-user $SLURM_MAIL_USER
#SBATCH --account=AIFAC_L07_002
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time 24:00:00
#SBATCH --hint=nomultithread

# =====================================================================================
# ENVIRONMENT SETUP
# =====================================================================================
# Load system modules and activate conda environment
source ~/.bashrc
module purge
module load cuda/12.1
conda activate qwen3vl

# =============================================================================
# PYTHON ENVIRONMENT
# =============================================================================
# Add current directory to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# =============================================================================
# HUGGINGFACE OFFLINE MODE
# =============================================================================
# Disable online features to work in offline environment
export HF_HUB_OFFLINE=1           # Disable downloads of new models
export HF_HUB_DISABLE_TELEMETRY=1 # Disable telemetry
export HF_DATASETS_OFFLINE=1      # Use offline datasets
export HF_UPDATE_DOWNLOAD_COUNTS=0 # Disable download counting
export TRANSFORMERS_OFFLINE=1     # Use offline transformers
export HF_EVALUATE_OFFLINE=1      # Use offline evaluation

# =====================================================================================
# CACHE AND STORAGE PATHS
# =====================================================================================
# Set cache directories for various components
export HF_DATASETS_CACHE=/leonardo_work/AIFAC_L07_002/shared_cache/llavanext/huggingface/datasets
export HF_HOME=/leonardo_work/AIFAC_L07_002/shared_cache/llavanext/huggingface
export DEEPSPEED_CACHE=/leonardo_work/AIFAC_L07_002/shared_cache/llavanext/deepspeed

# =====================================================================================
# PERFORMANCE AND DISTRIBUTED TRAINING
# =====================================================================================
export WANDB_MODE=offline
export OMP_NUM_THREADS=8
export NUM_GPUS=4
export NNODES=1
export RANK=0
export ADDR="localhost"
export PORT="29500"

# =====================================================================================
# MODEL CONFIGURATION
# =====================================================================================
# Language model configuration
LLM_VERSION=lingshu-medical-mllm/Lingshu-7B  # Using HuggingFace model ID
LLM_VERSION_CLEAN="${LLM_VERSION//\//-}"

# =====================================================================================
# TRAINING HYPERPARAMETERS
# =====================================================================================
# Training hyperparameters
LR=1e-5
BATCH_SIZE=1
GRAD_ACCUM_STEPS=4

# =====================================================================================
# DATA CONFIGURATION
# =====================================================================================
# Data paths and processing
DATASETS=datasets=llavamed_instruct,kits_instruct,abdomen_atlas_instruct,radimagenet_instruct,vqa_rad,slake


# =====================================================================================
# EXPERIMENT NAMING AND OUTPUT CONFIGURATION
# =====================================================================================
BASE_RUN_NAME="qwenvl-${LLM_VERSION_CLEAN}-stage2-mixed"
OUTPUT_DIR="/leonardo_work/AIFAC_L07_002/lbutsane/models/qwen3vl/${BASE_RUN_NAME}"


# Display experiment information
echo "============================================================================================"
echo "EXPERIMENT CONFIGURATION"
echo "============================================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Run Name: ${BASE_RUN_NAME}"
echo "LLM Version: ${LLM_VERSION}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "============================================================================================"

# =====================================================================================
# TRAINING EXECUTION
# =====================================================================================
# Stage 2 Alignment Training
# This stage focuses on aligning the vision and language components
# Only tuning vision tower and MLP adapter (not language model)
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03

torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    qwenvl/train/train_qwen.py \
    --deepspeed ${DEEPSPEED} \
    --model_name_or_path "${LLM_VERSION}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"
exit 0;