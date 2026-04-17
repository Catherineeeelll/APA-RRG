"""
APA-RRG main model.

The BLIP_Decoder ties together the visual encoder, the CLIP-based memory
retrieval module, the Dynamic Anatomy-Pathology Graph (DAP-G), the
Pathology-Anchored Representation Calibration (PARC), the Anatomy-Aware
Prompt Generation (APG), and the BERT text decoder.

Module breakdown follows Section 3 of the paper:
    - Section 3.2: visual encoder + CLIP memory retrieval.
    - Section 3.3: DAP-G  (models/dap_graph.py).
    - Section 3.4: PARC   (models/parc.py).
    - Section 3.5: APG    (models/apg.py).
    - Section 3.6: training objective (loss assembled in modules/trainer.py).
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.med import BertConfig, BertLMHeadModel
from models.resnet import blip_resnet
from models.transformer import Transformer

from models.dap_graph import DynamicAnatomyPathologyGraph
from models.parc import PathologyAnchoredCalibration
from models.apg import build_prompt_from_probs, empty_prompt
from models.structure_loss import StructureLoss

warnings.filterwarnings("ignore")


STATE_TOKENS = ["[BLA]", "[POS]", "[NEG]", "[UNC]"]


class BLIP_Decoder(nn.Module):
    """End-to-end APA-RRG model."""

    def __init__(self, args, tokenizer=None, image_size=224, prompt="", bert_path=None):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

        self.use_dap_graph = getattr(args, "use_dap_graph", False)
        self.use_parc = getattr(args, "use_parc", False)
        self.use_apg = getattr(args, "use_apg", False)
        self.use_structure_loss = getattr(args, "use_structure_loss", False)

        # ----- Section 3.2: visual encoder + CLIP memory retrieval -----
        vision_width = 2048
        self.visual_encoder = blip_resnet(args)
        self.vision_proj = nn.Linear(vision_width, 512)

        self.memory = Transformer(
            d_model=512,
            num_encoder_layers=2,
            num_decoder_layers=2,
            num_queries=1,
        )
        self.attn_linear = nn.Linear(1024, 2)

        # Classification head over the 18 disease nodes (14 CheXpert + 4 aux),
        # each with four states {BLA, POS, NEG, UNC}.
        self.cls_head = nn.Linear(2048 + 512, 18 * 4)
        nn.init.normal_(self.cls_head.weight, std=0.001)
        if self.cls_head.bias is not None:
            nn.init.constant_(self.cls_head.bias, 0)

        # ----- BERT text decoder -----
        decoder_config = BertConfig.from_json_file("configs/bert_config.json")
        decoder_config.encoder_width = vision_width
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
        if bert_path is None:
            bert_path = "bert-base-uncased"
        self.text_decoder = BertLMHeadModel.from_pretrained(
            bert_path,
            config=decoder_config,
            ignore_mismatched_sizes=True,
        )
        self._freeze_layers()
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        # ----- Section 3.3: DAP-G -----
        if self.use_dap_graph:
            cooccur_path = getattr(
                args, "cooccurrence_path", "data/mimic_cxr/disease_cooccurrence.npy"
            )
            self.dap_graph = DynamicAnatomyPathologyGraph(
                num_diseases=18,
                hidden_dim=512,
                visual_dim=2048,
                cooccurrence_path=cooccur_path,
            )
            print("[Model] DAP-G enabled")
        else:
            self.dap_graph = None

        # ----- Section 3.4: PARC -----
        if self.use_parc:
            self.parc = PathologyAnchoredCalibration(
                feature_dim=512,
                num_prototypes=18,
                temperature=getattr(args, "proto_temperature", 0.5),
                gate_init_bias=-2.0,
            )
            print("[Model] PARC enabled")
        else:
            self.parc = None

        # ----- Structure loss head (applied on decoder hidden states) -----
        if self.use_structure_loss:
            self.structure_loss = StructureLoss(
                decoder_hidden_dim=decoder_config.hidden_size,
                num_regions=6,
            )
            print("[Model] Structure loss enabled")
        else:
            self.structure_loss = None

        if self.use_apg:
            print("[Model] APG enabled")

        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Parameters: total={n_total:,}, trainable={n_train:,}")

    def _freeze_layers(self) -> None:
        """Freeze the first four ResNet layers and the lower 6 BERT layers."""
        for name, param in self.visual_encoder.named_parameters():
            if any(f"model.{i}" in name for i in range(4)):
                param.requires_grad = False
        for name, param in self.text_decoder.named_parameters():
            if "embeddings" in name:
                param.requires_grad = False
            elif "layer." in name:
                try:
                    layer_num = int(name.split("layer.")[1].split(".")[0])
                    if layer_num < 6:
                        param.requires_grad = False
                except (ValueError, IndexError):
                    pass

    # ------------------------------------------------------------------
    # Shared forward path used by both training and generation.
    # ------------------------------------------------------------------
    def _encode_and_classify(self, image, clip_memory):
        """Run the encoder, memory retrieval, DAP-G, PARC, and the cls head.

        Returns:
            image_embeds: [B, L, 2048] patch features for the cross-attention.
            avg_embeds:   [B, 2048] global visual feature.
            hs_cal:       [B, 512] memory feature after DAP-G + PARC.
            cls_preds:    [B, 4, 18] final classification logits.
        """
        image_embeds, avg_embeds = self.visual_encoder(image)

        clip_memory = torch.permute(clip_memory, (1, 0, 2)).to(image.device)
        query_embed = self.vision_proj(avg_embeds)
        hs = self.memory(clip_memory, None, query_embed.unsqueeze(0), None)
        hs = hs.squeeze(0).squeeze(1)

        # First-pass classification provides the disease prior used by DAP-G.
        cls_features = torch.cat([avg_embeds, hs], dim=1)
        cls_preds = self.cls_head(cls_features).view(-1, 4, 18)

        if self.dap_graph is not None:
            # expects softmax probabilities over the four states.
            disease_probs = F.softmax(cls_preds, dim=1).permute(0, 2, 1)
            graph_feats = self.dap_graph(disease_probs, visual_features=avg_embeds)
            hs = hs + torch.sigmoid(self.dap_graph.scale) * graph_feats

        if self.parc is not None:
            # Use the per-disease positive probability as pi (length 18).
            disease_pos = F.softmax(cls_preds, dim=1)[:, 1, :]
            hs = self.parc(hs, disease_pos)

        # Refresh classification with the calibrated memory feature.
        cls_features = torch.cat([avg_embeds, hs], dim=1)
        cls_preds = self.cls_head(cls_features).view(-1, 4, 18)
        return image_embeds, avg_embeds, hs, cls_preds

    # ------------------------------------------------------------------
    # Training pass.
    # ------------------------------------------------------------------
    def forward(self, image, caption, cls_labels, clip_memory, criterion_cls, base_probs):
        image_embeds, _, _, cls_preds = self._encode_and_classify(image, clip_memory)

        # Logit adjustment over the POS channel using base disease rates
        # base_probs is a length-18 numpy array.
        if base_probs is not None and len(base_probs) > 0:
            base_probs_tensor = torch.from_numpy(base_probs).to(image.device).float()
            base_probs_tensor = torch.clamp(base_probs_tensor, min=1e-6, max=1.0)
            log_base = torch.log(base_probs_tensor).view(1, -1)
            cls_preds[:, 1, :] = cls_preds[:, 1, :] + log_base

        batch_size = cls_preds.size(0)
        cls_labels_reshaped = (
            cls_labels.view(batch_size, -1) if cls_labels.dim() == 1 else cls_labels
        )
        cls_preds_flat = cls_preds.permute(0, 2, 1).contiguous().view(-1, 4)
        cls_labels_flat = cls_labels_reshaped.view(-1).long()
        loss_cls = criterion_cls(cls_preds_flat, cls_labels_flat)

        # Tokenize the caption (which already contains the APG prompt
        # prefix prepared by the dataset).
        text = self.tokenizer(
            caption, padding="longest", truncation=True, return_tensors="pt"
        ).to(image.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, : self.prompt_length] = -100

        decoder_output = self.text_decoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            labels=decoder_targets,
            return_dict=True,
            output_hidden_states=self.use_structure_loss,
        )
        loss_lm = decoder_output.loss

        loss_str = torch.tensor(0.0, device=image.device)
        if self.structure_loss is not None and decoder_output.hidden_states is not None:
            last_hidden = decoder_output.hidden_states[-1]
            loss_str = self.structure_loss(last_hidden, cls_labels_reshaped)

        return loss_lm, loss_cls, loss_str

    # ------------------------------------------------------------------
    # Inference pass.
    # ------------------------------------------------------------------
    def generate(
        self,
        image,
        clip_memory,
        sample=False,
        num_beams=3,
        max_length=150,
        min_length=80,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        image_embeds, _, _, cls_preds = self._encode_and_classify(image, clip_memory)

        cls_preds_softmax = F.softmax(cls_preds, dim=1)
        cls_preds_logits = cls_preds_softmax[:, 1, :14]  # POS prob, 14 CheXpert

        # Build prompts from the predicted probabilities. Each sample
        # receives the six APG region tokens followed by
        # the eighteen per-disease state tokens drawn from the
        # argmax over {BLA, POS, NEG, UNC} of the refreshed classification
        # logits. The two segments are concatenated before being fed to
        # the decoder so that the generated report is conditioned on both
        # anatomical structure and fine-grained disease evidence.
        threshold = getattr(self.args, "apg_threshold", 0.5)
        state_argmax = cls_preds_softmax.argmax(dim=1)  # [B, 18], values in {0,1,2,3}
        prompts = []
        for j in range(image.size(0)):
            region_prompt = build_prompt_from_probs(
                cls_preds_logits[j], threshold=threshold
            )
            state_prompt = (
                " ".join(STATE_TOKENS[int(state_argmax[j, i])] for i in range(18))
                + " "
            )
            prompts.append(region_prompt + state_prompt)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        text = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = text.input_ids.to(image.device)
        attn_masks = text.attention_mask.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]
        attn_masks = attn_masks[:, :-1]

        outputs = self.text_decoder.generate(
            input_ids=input_ids,
            min_length=min_length,
            max_new_tokens=max_length,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            attention_mask=attn_masks,
            **model_kwargs,
        )

        captions = []
        for i, output in enumerate(outputs):
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(prompts[i]):])

        return captions, cls_preds_softmax, cls_preds_logits


def blip_decoder(args, tokenizer, **kwargs):
    return BLIP_Decoder(args, tokenizer, **kwargs)
