"""Clinical Efficacy metric based on CheXbert label extraction."""

import numpy as np

from .chexbert import CheXbert


# CheXbert outputs labels for 14 CheXpert conditions:
#   0 = blank / not mentioned
#   1 = positive
#   2 = negative
#   3 = uncertain
CONDITIONS = [
    "enlarged cardiomediastinum",
    "cardiomegaly",
    "lung opacity",
    "lung lesion",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural effusion",
    "pleural other",
    "fracture",
    "support devices",
    "no finding",
]


class CheXbertMetrics:
    """Computes example-based Clinical Efficacy metrics."""

    def __init__(self, checkpoint_path, mbatch_size, device):
        self.checkpoint_path = checkpoint_path
        self.mbatch_size = mbatch_size
        self.device = device
        self.chexbert = CheXbert(self.checkpoint_path, self.device).to(self.device)

    def mini_batch(self, gts, res, mbatch_size=16):
        length = len(gts)
        assert length == len(res)
        for i in range(0, length, mbatch_size):
            yield gts[i : min(i + mbatch_size, length)], res[i : min(i + mbatch_size, length)]

    def compute(self, gts, res):
        gts_chexbert, res_chexbert = [], []
        for gt, re in self.mini_batch(gts, res, self.mbatch_size):
            gt_chexbert = self.chexbert(list(gt)).tolist()
            re_chexbert = self.chexbert(list(re)).tolist()
            gts_chexbert += gt_chexbert
            res_chexbert += re_chexbert
        gts_chexbert = np.array(gts_chexbert)
        res_chexbert = np.array(res_chexbert)

        res_chexbert = res_chexbert == 1
        gts_chexbert = gts_chexbert == 1

        tp = (res_chexbert * gts_chexbert).astype(float)
        fp = (res_chexbert * ~gts_chexbert).astype(float)
        fn = (~res_chexbert * gts_chexbert).astype(float)

        tp_eg = tp.sum(1)
        fp_eg = fp.sum(1)
        fn_eg = fn.sum(1)

        scores = {
            "ce_precision": np.nan_to_num(tp_eg / (tp_eg + fp_eg)).mean(),
            "ce_recall": np.nan_to_num(tp_eg / (tp_eg + fn_eg)).mean(),
            "ce_f1": np.nan_to_num(tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg))).mean(),
            "ce_num_examples": float(len(res_chexbert)),
        }
        return scores
