"""CheXbert wrapper used for the Clinical Efficacy metric."""

from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer


class CheXbert(nn.Module):
    """CheXbert label extractor (Smit et al., EMNLP 2020).

    Loads the public ``bert-base-uncased`` weights from the HuggingFace
    Hub and replaces the classification heads with the trained CheXbert
    parameters supplied via ``checkpoint_path``.
    """

    def __init__(self, checkpoint_path, device, p=0.1):
        super().__init__()

        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        config = BertConfig.from_pretrained("bert-base-uncased")

        with torch.no_grad():
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(p)

            hidden_size = self.bert.pooler.dense.in_features

            # 13 conditions with 4 classes (present, absent, uncertain, blank).
            self.linear_heads = nn.ModuleList(
                [nn.Linear(hidden_size, 4, bias=True) for _ in range(13)]
            )
            # The "no finding" head is a binary {yes, no} classifier.
            self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

            state_dict = torch.load(checkpoint_path, map_location=device)["model_state_dict"]

            new_state_dict = OrderedDict()
            new_state_dict["bert.embeddings.position_ids"] = torch.arange(
                config.max_position_embeddings
            ).expand((1, -1))
            for key, value in state_dict.items():
                if "bert" in key:
                    new_key = key.replace("module.bert.", "bert.")
                elif "linear_heads" in key:
                    new_key = key.replace("module.linear_heads.", "linear_heads.")
                new_state_dict[new_key] = value

            self.load_state_dict(new_state_dict)

        self.eval()

    def forward(self, reports):
        for i in range(len(reports)):
            reports[i] = reports[i].strip()
            reports[i] = reports[i].replace("\n", " ")
            reports[i] = reports[i].replace("\\s+", " ")
            reports[i] = reports[i].replace("\\s+(?=[\\.,])", "")
            reports[i] = reports[i].strip()

        with torch.no_grad():
            tokenized = self.tokenizer(
                reports,
                padding="longest",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            last_hidden_state = self.bert(**tokenized)[0]
            cls = last_hidden_state[:, 0, :]
            cls = self.dropout(cls)

            predictions = []
            for i in range(14):
                predictions.append(self.linear_heads[i](cls).argmax(dim=1))

        return torch.stack(predictions, dim=1)
