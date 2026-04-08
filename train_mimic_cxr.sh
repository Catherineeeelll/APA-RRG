#!/bin/bash
# Train APA-RRG on MIMIC-CXR.
#
# Hyperparameters follow Section 4.1.3 of the paper:
#   AdamW, lr 5e-5, weight decay 0.05, batch 16, 15 epochs
#   lambda_cls = 4.0, lambda_str = 0.1
#   PARC temperature tau_p = 0.5, APG threshold tau = 0.5
#   beam size = 3, generation length 80-150
#   18 disease nodes (14 CheXpert + 4 auxiliary)
#
# Single NVIDIA RTX 4090 GPU, ~24 hours.

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python main_train.py \
    --n_gpu 1 \
    --image_dir data/mimic_cxr/images/ \
    --ann_path data/mimic_cxr/mimic_annotation_promptmrg.json \
    --dataset_name mimic_cxr \
    --batch_size 16 \
    --epochs 15 \
    --save_dir results/apa_rrg \
    --seed 9233 \
    --init_lr 5e-5 \
    --min_lr 5e-6 \
    --warmup_lr 5e-7 \
    --weight_decay 0.05 \
    --warmup_steps 5000 \
    --lambda_cls 4.0 \
    --lambda_str 0.1 \
    --clip_k 21 \
    --beam_size 3 \
    --gen_max_len 150 \
    --gen_min_len 80 \
    --apg_threshold 0.5 \
    --proto_temperature 0.5 \
    --cooccurrence_path data/mimic_cxr/disease_cooccurrence.npy \
    --use_dap_graph \
    --use_parc \
    --use_apg \
    --use_structure_loss \
    --load_pretrained results/model_promptmrg/model_promptmrg_20240305.pth \
    2>&1 | tee log.out

echo "Training finished."
