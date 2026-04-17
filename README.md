# APA-RRG

Code for the paper *Reliable Radiology Report Generation with
Anatomy-Pathology-Aware Dynamic Reasoning and Representation
Calibration*.

APA-RRG is an anatomy-pathology-aware framework for radiology report
generation that emulates the progressive workflow of a radiologist
through three interdependent modules.

* **DAP-G** (Dynamic Anatomy-Pathology Graph). Infers patient-specific
  disease dependencies by fusing an image-conditioned dynamic adjacency
  with a static co-occurrence prior through a vision-guided gate.
* **PARC** (Pathology-Anchored Representation Calibration). Anchors
  fragile rare-disease representations to a bank of learnable prototypes
  via prediction-driven gated fusion.
* **APG** (Anatomy-Aware Prompt Generation). Aggregates the predicted
  pathology probabilities into six anatomical regions and emits one
  region-aware prompt token per region to guide structured generation.

## Installation

We use Python 3.10 and PyTorch. Set up the environment as follows.

```
conda create -n apa-rrg python=3.10
conda activate apa-rrg
pip install -r requirements.txt
```

After the source code is in place, the first training or evaluation
run will automatically pull `bert-base-uncased` from the HuggingFace
Hub. No manual download is required for the BERT weights.

## Datasets Preparation

APA-RRG is trained on **MIMIC-CXR** and evaluated on both MIMIC-CXR and
**IU X-Ray** (zero-shot transfer). The required files for each dataset
are listed below.

### MIMIC-CXR

Three artifacts are needed under `data/mimic_cxr/`:

1. The chest X-ray images themselves. The official source is
   [PhysioNet](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/),
   which requires credentialed access. A pre-resized copy is also
   distributed by the [R2Gen](https://github.com/zhjohnchan/R2Gen)
   project and is what we use in our experiments.
2. The PromptMRG-style annotation file
   `mimic_annotation_promptmrg.json`, available on
   [Google Drive](https://drive.google.com/file/d/1qR7EJkiBdHPrskfikz2adL-p9BjMRXup/view?usp=sharing).
3. Pre-extracted text features `clip_text_features.json` from the
   training reports, encoded by the MIMIC-CXR pretrained
   [CLIP](https://stanfordmedicine.app.box.com/s/dbebk0jr5651dj8x1cu6b6kqyuuvz3ml)
   model. Download the file directly from
   [Google Drive](https://drive.google.com/file/d/1Zyq-84VOzc-TOZBzlhMyXLwHjDNTaN9A/view?usp=sharing).

### IU X-Ray

Two artifacts are needed under `data/iu_xray/`:

1. The chest X-ray images, available from
   [R2Gen](https://github.com/zhjohnchan/R2Gen).
2. The annotation file `iu_annotation_promptmrg.json` from
   [Google Drive](https://drive.google.com/file/d/1zV5wgi5QsIp6OuC1U95xvOmeAAlBGkRS/view?usp=sharing).

### CheXbert

The Clinical Efficacy metric depends on a trained
[CheXbert](https://github.com/stanfordmlgroup/CheXbert) label
extractor. Download `chexbert.pth` from
[Google Drive](https://drive.google.com/file/d/1Qj5yM62FlASGRnW1hH0DDtCENuqGtt7L/view?usp=sharing)
and save it to `checkpoints/stanford/chexbert/chexbert.pth`.

### Final layout

Once the downloads are in place, the working directory should look
like:

```
apa-rrg
|--data
   |--mimic_cxr
      |--base_probs.json
      |--clip_text_features.json
      |--mimic_annotation_promptmrg.json
      |--images
         |--p10
         |--p11
         ...
   |--iu_xray
      |--iu_annotation_promptmrg.json
      |--images
         |--CXR1000_IM-0003
         |--CXR1001_IM-0004
         ...
|--checkpoints
   |--stanford
      |--chexbert
         |--chexbert.pth
...
```

## Build the Static Co-occurrence Prior

DAP-G fuses a dynamic adjacency with a static co-occurrence prior
estimated from the training-set CheXpert labels. **This step is
mandatory before training or testing**, since the prior is not
committed to the repository.

```
python scripts/build_cooccurrence.py --dataset mimic_cxr
```

The script reads the PromptMRG-style annotation file and writes an
18-by-18 row-normalized symmetric matrix to
`data/mimic_cxr/disease_cooccurrence.npy`. The output is fully
deterministic given the input annotation, so any execution reproduces
the exact same prior.

## Pretrained Weights

We provide two checkpoints for different use cases. Both were trained
with images from [R2Gen](https://github.com/zhjohnchan/R2Gen). If you
use images processed by yourself, you may obtain degraded performance
with these weights. In this case, you need to train a model by
yourself.

**APA-RRG (ours).** The fully trained APA-RRG model, which can be used
for direct evaluation or as a warm-start for further training. All
modules (DAP-G, PARC, APG) are included. Download from
[Google Drive](https://drive.google.com/file/d/1lliZlxwVAlpZk6clUxs-6EE42jdT8o-K/view?usp=drive_link)
and place it under `results/apa_rrg/`.

**PromptMRG backbone.** The base PromptMRG checkpoint released by its
original authors, containing only the visual encoder, memory module,
classification head, and text decoder. DAP-G, PARC, and APG weights
are not included and will be randomly initialized. This checkpoint can
only be used as a warm-start for training. Download
`model_promptmrg_20240305.pth` from
[Google Drive](https://drive.google.com/file/d/1s4AoLnnGOysOQkdILhhFCL59LyQtRHGa/view?usp=drive_link)
and place it under `results/model_promptmrg/`.

## Training

A single command launches end-to-end training on MIMIC-CXR:

```
bash train_mimic_cxr.sh
```

All hyperparameters are documented inline within the shell script and
should not need to be edited for reproduction. Trained checkpoints are
written to `results/apa_rrg/`.

By default the script warm-starts from the PromptMRG backbone. To
warm-start from the fully trained APA-RRG checkpoint instead, change
the `--load_pretrained` path inside the script. To skip warm-starting
entirely, set `--load_pretrained ""`.

## Evaluation

Two helper scripts cover the standard evaluation protocols.

```
bash test_mimic_cxr.sh    # MIMIC-CXR test split
bash test_iu_xray.sh      # IU X-Ray zero-shot transfer
```

To reproduce the per-class rare-disease analysis, run

```
python evaluate_rare_diseases.py \
    --load_pretrained results/apa_rrg/model_best.pth \
    --use_dap_graph --use_parc --use_apg --use_structure_loss
```

The script dumps a JSON file alongside the checkpoint, which is then
consumed by `visualize_rare_disease.py` to render the per-class
comparison figure used in the paper.

## Acknowledgment

We thank the authors of the following projects, on which our code
depends.

* [PromptMRG](https://github.com/jhb86253817/PromptMRG)
* [R2Gen](https://github.com/zhjohnchan/R2Gen)
* [BLIP](https://github.com/salesforce/BLIP)
* [CheXbert](https://github.com/stanfordmlgroup/CheXbert)
