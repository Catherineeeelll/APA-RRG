#!/bin/bash
# Zero-shot transfer evaluation on IU X-Ray.
# The model is trained on MIMIC-CXR only and applied directly to the
# entire IU X-Ray collection without fine-tuning, following the protocol
# of PromptMRG.

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python main_test.py \
    --image_dir data/iu_xray/images/ \
    --ann_path data/iu_xray/iu_annotation_promptmrg.json \
    --dataset_name iu_xray \
    --batch_size 16 \
    --beam_size 3 \
    --gen_max_len 150 \
    --gen_min_len 80 \
    --clip_k 21 \
    --apg_threshold 0.5 \
    --proto_temperature 0.5 \
    --cooccurrence_path data/iu_xray/disease_cooccurrence.npy \
    --use_dap_graph \
    --use_parc \
    --use_apg \
    --use_structure_loss \
    --load_pretrained results/apa_rrg/model_best.pth
