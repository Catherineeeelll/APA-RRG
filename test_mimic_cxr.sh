#!/bin/bash
# Evaluate APA-RRG on the MIMIC-CXR test split.

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python main_test.py \
    --image_dir data/mimic_cxr/images/ \
    --ann_path data/mimic_cxr/mimic_annotation_promptmrg.json \
    --dataset_name mimic_cxr \
    --batch_size 16 \
    --beam_size 3 \
    --gen_max_len 150 \
    --gen_min_len 80 \
    --clip_k 21 \
    --apg_threshold 0.5 \
    --proto_temperature 0.5 \
    --cooccurrence_path data/mimic_cxr/disease_cooccurrence.npy \
    --use_dap_graph \
    --use_parc \
    --use_apg \
    --use_structure_loss \
    --load_pretrained results/apa_rrg/model_best.pth
