"""Architecture de fusion signal–texte–caractéristiques tolérante aux modalités manquantes.

  Signal ECG (12 x T) + masque de dérivations ──► CNN-ResNet1D ──► Transformer ──► jetons temporels
  Compte rendu ──► ClinicalBERT (ou encodeur léger) ──► jetons texte
  Caractéristiques (F) + indicateurs de valeurs manquantes ──► jetons « FT-Transformer »
                                │
        jetons concaténés + embeddings de type de modalité
        (modalité absente -> jeton appris « [ABSENT] », jetons masqués)
                                │
  Requêtes par pathologie (1 requête / label) ──► attention croisée ×N ──► 1 logit / label

Les requêtes par label rendent l'attention interprétable : la masse
d'attention de la requête « AFIB » sur chaque modalité mesure la contribution
de cette modalité à cette pathologie.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = ["signal", "text", "features"]


# --------------------------------------------------------------------------
# Encodeur signal
# --------------------------------------------------------------------------
class ResBlock1d(nn.Module):
    def __init__(self, c_in, c_out, stride, k=7, p_drop=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, k, stride, k // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, 1, k // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(p_drop)
        self.skip = (nn.Identity() if stride == 1 and c_in == c_out else
                     nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm1d(c_out)))

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(self.drop(h)))
        return F.gelu(h + self.skip(x))


class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=4096):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(max_len, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class SignalEncoder(nn.Module):
    """CNN-Transformer. Le masque de dérivations est donné en canaux d'entrée
    supplémentaires : le réseau sait quelles dérivations sont réellement nulles."""

    def __init__(self, d_model=256, n_layers=4, n_heads=8, widths=(64, 128, 192, 256), dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(24, widths[0], 15, 2, 7, bias=False),
                                  nn.BatchNorm1d(widths[0]), nn.GELU())
        blocks, c = [], widths[0]
        for w in widths:
            blocks += [ResBlock1d(c, w, stride=2, p_drop=dropout), ResBlock1d(w, w, 1, p_drop=dropout)]
            c = w
        self.cnn = nn.Sequential(*blocks)
        self.proj = nn.Linear(c, d_model)
        self.pe = SinusoidalPE(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, signal, lead_mask):
        m = lead_mask[:, :, None].expand_as(signal)
        h = self.cnn(self.stem(torch.cat([signal, m], dim=1)))   # (B, C, T')
        h = self.pe(self.proj(h.transpose(1, 2)))                 # (B, T', d)
        return self.transformer(h)


# --------------------------------------------------------------------------
# Encodeur texte
# --------------------------------------------------------------------------
class HFTextEncoder(nn.Module):
    """Encodeur biomédical pré-entraîné (ex. emilyalsentzer/Bio_ClinicalBERT)."""

    def __init__(self, name, d_model=256, n_trainable_layers=2):
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(name)
        for p in self.bert.parameters():
            p.requires_grad = False
        layers = getattr(getattr(self.bert, "encoder", None), "layer", [])
        for layer in list(layers)[len(layers) - n_trainable_layers:] if n_trainable_layers else []:
            for p in layer.parameters():
                p.requires_grad = True
        self.proj = nn.Linear(self.bert.config.hidden_size, d_model)

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.proj(h)


class LightTextEncoder(nn.Module):
    """Encodeur léger entraîné de zéro (vocabulaire haché) : tests, ou ablation."""

    def __init__(self, vocab_size=2 ** 15, d_model=256, n_layers=2, n_heads=8, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = SinusoidalPE(d_model, 1024)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, input_ids, attention_mask):
        h = self.pe(self.emb(input_ids))
        return self.tr(h, src_key_padding_mask=attention_mask == 0)


# --------------------------------------------------------------------------
# Encodeur des caractéristiques (style FT-Transformer)
# --------------------------------------------------------------------------
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_model=256, n_layers=2, n_heads=8, dropout=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.missing = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, x, missing):
        tok = x[..., None] * self.weight + self.bias                 # (B, F, d)
        m = missing[..., None]
        tok = tok * (1 - m) + self.missing * m                       # valeur manquante -> jeton appris
        return self.tr(tok)


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------
class LabelQueryFusion(nn.Module):
    """Requêtes par label -> attention croisée sur tous les jetons de modalités."""

    def __init__(self, n_labels, d_model=256, n_layers=2, n_heads=8, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_labels, d_model) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(d_model), "ln_kv": nn.LayerNorm(d_model),
                "cross": nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True),
                "ln_s": nn.LayerNorm(d_model),
                "self": nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True),
                "ln_f": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(4 * d_model, d_model)),
            }))

    def forward(self, tokens, key_padding_mask):
        q = self.queries.unsqueeze(0).expand(tokens.size(0), -1, -1)
        attn_last = None
        for L in self.layers:
            kv = L["ln_kv"](tokens)
            a, attn_last = L["cross"](L["ln_q"](q), kv, kv, key_padding_mask=key_padding_mask,
                                      need_weights=True, average_attn_weights=True)
            q = q + a
            s = L["ln_s"](q)
            q = q + L["self"](s, s, s, need_weights=False)[0]
            q = q + L["ffn"](L["ln_f"](q))
        return q, attn_last  # (B, C, d), (B, C, N_tokens)


def masked_mean(x, valid):
    w = valid.float().unsqueeze(-1)
    return (x * w).sum(1) / w.sum(1).clamp(min=1.0)


class MultimodalECGNet(nn.Module):
    def __init__(self, n_labels, n_features, d_model=256, text_encoder="light",
                 text_model_name="emilyalsentzer/Bio_ClinicalBERT", text_trainable_layers=2,
                 signal_layers=4, fusion_layers=2, n_heads=8, dropout=0.1):
        super().__init__()
        self.signal_enc = SignalEncoder(d_model, signal_layers, n_heads, dropout=dropout)
        self.text_enc = (HFTextEncoder(text_model_name, d_model, text_trainable_layers)
                         if text_encoder == "hf" else LightTextEncoder(d_model=d_model, n_heads=n_heads))
        self.feat_enc = FeatureTokenizer(n_features, d_model, n_heads=n_heads, dropout=dropout)
        self.type_emb = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.absent_tok = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.fusion = LabelQueryFusion(n_labels, d_model, fusion_layers, n_heads, dropout)
        self.head = nn.Linear(d_model, 1)  # partagé ; spécialisation via les requêtes
        self.label_bias = nn.Parameter(torch.zeros(n_labels))
        # têtes auxiliaires unimodales : chaque encodeur doit rester informatif seul
        self.aux = nn.ModuleList([nn.Linear(d_model, n_labels) for _ in range(3)])

    def forward(self, batch, return_attention=False):
        present = batch["present"]  # (B, 3)
        B = present.size(0)
        tok_s = self.signal_enc(batch["signal"], batch["lead_mask"])
        tok_t = self.text_enc(batch["input_ids"], batch["attention_mask"])
        tok_f = self.feat_enc(batch["features"], batch["feat_missing"])

        valid_s = present[:, :1].bool().expand(-1, tok_s.size(1))
        valid_t = batch["attention_mask"].bool() & present[:, 1:2].bool()
        valid_f = (batch["feat_missing"] < 0.5) & present[:, 2:3].bool()

        parts, valids, spans = [], [], []
        for k, (tok, val) in enumerate([(tok_s, valid_s), (tok_t, valid_t), (tok_f, valid_f)]):
            absent = self.absent_tok[k].expand(B, 1, -1)
            # jeton [ABSENT] visible seulement si la modalité est absente
            parts += [tok + self.type_emb[k], absent + self.type_emb[k]]
            valids += [val, (~val.any(1, keepdim=True))]
            spans.append(tok.size(1) + 1)
        tokens = torch.cat(parts, 1)
        valid = torch.cat(valids, 1)

        q, attn = self.fusion(tokens, key_padding_mask=~valid)
        logits = self.head(q).squeeze(-1) + self.label_bias

        aux = [self.aux[0](masked_mean(tok_s, valid_s)),
               self.aux[1](masked_mean(tok_t, valid_t)),
               self.aux[2](masked_mean(tok_f, valid_f))]
        out = {"logits": logits, "aux_logits": aux}
        if return_attention:
            # masse d'attention par modalité (jeton [ABSENT] inclus dans sa modalité)
            chunks = torch.split(attn, spans, dim=-1)
            out["modality_attention"] = torch.stack([c.sum(-1) for c in chunks], -1)  # (B, C, 3)
        return out
