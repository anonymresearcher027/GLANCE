"""GLANCE training and evaluation.

This module is self-contained and starts from a fresh random brain model and an
identity-initialized residual sentence adapter. Cached text targets remain
outside the optimizer, and all retrieval components are trained jointly from
epoch 1 with constant parameter-group learning rates.

The module is imported by ``run_glance.py``. That wrapper
requires an explicit ``--run`` before it can train; invoking it without
``--run`` only performs the non-training preflight checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_DIM = 768
TOKEN_DIM = 256
MASK_LENGTH = 4
MASK_PROBABILITY = 0.50
RETRIEVAL_TEMPERATURE = 0.07
LOCAL_TEMPERATURE = 0.10
LOCAL_WEIGHT = 0.70
CONSISTENCY_WEIGHT = 0.10
WEIGHT_DECAY = 0.01
BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0
ADAPTER_BOTTLENECK = 64
ADAPTER_ALPHA = 0.05


PHASES: tuple[dict[str, Any], ...] = (
    {
        "name": "JOINT",
        "description": "Joint GLANCE training with constant learning rates",
        "epochs": 200,
        "selection_metric": "combined.recall@10",
        "local_weight": LOCAL_WEIGHT,
        "masked_view": True,
        "consistency_weight": CONSISTENCY_WEIGHT,
        "trainable": [
            "global_base",
            "lead_mask_tokens",
            "predictor_projection",
            "temporal_cls_positions",
            "local_word",
            "sentence_adapter",
        ],
        "learning_rates": {
            "global_base": 3e-4,
            "lead_mask_tokens": 3e-4,
            "predictor_projection": 5e-5,
            "temporal_cls_positions": 2e-5,
            "local_word": 3e-4,
            "sentence_adapter": 1e-5,
        },
    },
)

TOTAL_EPOCHS = sum(int(phase["epochs"]) for phase in PHASES)
PHASE_STARTS = {
    phase["name"]: 1 + sum(int(previous["epochs"]) for previous in PHASES[:index])
    for index, phase in enumerate(PHASES)
}
PHASE_ENDS = {
    phase["name"]: sum(int(previous["epochs"]) for previous in PHASES[:index + 1])
    for index, phase in enumerate(PHASES)
}


@dataclass(frozen=True)
class ExperimentPaths:
    data_root: Path
    sentence_cache: Path
    word_cache: Path
    word_mask: Path
    feature_cache: Path
    window: Path
    channel_order: Path
    split_root: Path
    output_dir: Path


def make_paths(seed: int, output_root: Path, data_root: Path, split_root: Path | None = None, channel_order: Path | None = None) -> ExperimentPaths:
    """Resolve the canonical prepared-data layout without dataset-specific IDs."""
    data = data_root
    if channel_order is None:
        raise ValueError("--channel-order is required for the public release")
    return ExperimentPaths(
        data_root=data,
        sentence_cache=data / "text" / "sentence_embeddings.npy",
        word_cache=data / "text" / "word_embeddings.npy",
        word_mask=data / "text" / "word_mask.npy",
        feature_cache=data / "features" / "normalized_features.npy",
        window=data / "sentence_windows.csv",
        channel_order=channel_order,
        split_root=split_root or (data / "splits"),
        output_dir=output_root,
    )


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state(order_rng: Any) -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "order": order_rng.get_state(),
    }


def restore_rng_state(state: dict[str, Any], order_rng: Any) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    order_rng.set_state(state["order"])


def cpu_state_dict(module: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def atomic_torch_save(payload: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def phase_for_epoch(epoch: int) -> tuple[int, dict[str, Any], int]:
    if not 1 <= epoch <= TOTAL_EPOCHS:
        raise ValueError(f"epoch must be in 1..{TOTAL_EPOCHS}, got {epoch}")
    for index, phase in enumerate(PHASES):
        if epoch <= PHASE_ENDS[phase["name"]]:
            return index, phase, epoch - PHASE_STARTS[phase["name"]] + 1
    raise AssertionError(epoch)


def derive_lead_mapping(common_channels_path: Path) -> tuple[list[str], list[list[str]], list[list[int]]]:
    """Derive lead groups from the channel labels; no fixed lead count is used."""
    labels = pd.read_csv(common_channels_path)["electrode_label"].astype(str).tolist()
    groups: dict[str, list[tuple[int, str]]] = {}
    for label in labels:
        match = re.fullmatch(r"(.+?)(\d+)", label)
        if match is None:
            raise ValueError(f"Cannot derive lead prefix/contact number from channel label {label!r}")
        groups.setdefault(match.group(1), []).append((int(match.group(2)), label))
    lead_names = sorted(groups)
    channels = [[label for _, label in sorted(groups[name])] for name in lead_names]
    index_by_label = {label: index for index, label in enumerate(labels)}
    indices = [[index_by_label[label] for label in lead] for lead in channels]
    flattened = [index for lead in indices for index in lead]
    if not flattened or len(flattened) != len(labels) or len(set(flattened)) != len(flattened):
        raise RuntimeError("Derived lead mapping does not partition common channels exactly")
    return lead_names, channels, indices


def make_ids(windows: pd.DataFrame) -> np.ndarray:
    first: dict[str, int] = {}
    transcript_ids: list[int] = []
    for index, text in enumerate(windows.text.astype(str)):
        key = " ".join(unicodedata.normalize("NFC", str(text)).split())
        first.setdefault(key, index)
        transcript_ids.append(int(windows.sample_id.iloc[first[key]]) if "sample_id" in windows else first[key])
    return np.asarray(transcript_ids, dtype=np.int64)


def load_prepared_data(paths: ExperimentPaths, seed: int) -> dict[str, Any]:
    windows = pd.read_csv(paths.window)
    sentences = np.asarray(np.load(paths.sentence_cache, mmap_mode="r"), dtype=np.float32)
    words = np.asarray(np.load(paths.word_cache, mmap_mode="r"), dtype=np.float32)
    word_mask = np.asarray(np.load(paths.word_mask, mmap_mode="r"), dtype=bool)
    features = np.load(paths.feature_cache, mmap_mode="r")
    valid_path = paths.data_root / "valid_time_mask.npy"
    if valid_path.exists():
        valid = np.asarray(np.load(valid_path, mmap_mode="r"), dtype=bool)
    else:
        valid = np.arange(features.shape[-1])[None, :] < windows.retained_time_points.to_numpy(dtype=int)[:, None]
    splits = {
        name: np.load(paths.split_root / f"seed{seed}" / f"{name}.npy").astype(np.int64)
        for name in ("train", "val", "test")
    }
    if sentences.ndim != 2 or sentences.shape[1] != TARGET_DIM or sentences.shape[0] != len(windows):
        raise ValueError(f"sentence cache shape mismatch: {sentences.shape}, windows={len(windows)}")
    if words.ndim != 3 or words.shape[0] != len(windows) or words.shape[2] != TARGET_DIM:
        raise ValueError(f"word cache shape mismatch: {words.shape}")
    if word_mask.shape != words.shape[:2]:
        raise ValueError(f"word mask shape mismatch: {word_mask.shape} vs {words.shape[:2]}")
    if features.ndim != 4 or features.shape[0] != len(windows):
        raise ValueError(f"feature cache shape mismatch: {features.shape}")
    if features.shape[1] != len(pd.read_csv(paths.channel_order)):
        raise ValueError(f"feature/channel mismatch: features={features.shape[1]}, channel table={len(pd.read_csv(paths.channel_order))}")
    if valid.shape != (len(windows), features.shape[-1]):
        raise ValueError(f"valid-time shape mismatch: {valid.shape}")
    expected_valid = np.arange(features.shape[-1])[None, :] < windows.retained_time_points.to_numpy(dtype=int)[:, None]
    if not np.array_equal(valid, expected_valid):
        raise ValueError("valid-time mask disagrees with sentence-window retained_time_points")
    if not np.isfinite(np.asarray(sentences)).all() or not np.isfinite(np.asarray(words)).all():
        raise ValueError("cached Gemma targets contain NaN or infinite values")
    if np.any(np.asarray(words)[~word_mask] != 0):
        raise ValueError("padding positions in centered word targets are not zero")
    all_indices = np.concatenate(tuple(splits.values()))
    if len(np.unique(all_indices)) != len(windows) or len(all_indices) != len(windows):
        raise ValueError("train/val/test split files are not a complete disjoint partition")
    return {
        "windows": windows,
        "sentences": sentences,
        "words": words,
        "word_mask": word_mask,
        "features": features,
        "valid": valid,
        "splits": splits,
        "tids": make_ids(windows),
    }


def mp_nce(scores: Any, positives: Any) -> Any:
    import torch

    return (torch.logsumexp(scores, 1) - torch.logsumexp(scores.masked_fill(~positives, float("-inf")), 1)).mean()


def sym_nce(scores: Any, positives: Any) -> Any:
    return 0.5 * (mp_nce(scores, positives) + mp_nce(scores.T, positives.T))


def local_scores(brain: Any, valid: Any, words: Any, word_mask: Any) -> Any:
    import torch

    similarity = torch.einsum("qtd,cwd->qctw", brain, words).masked_fill(
        ~valid[:, None, :, None], float("-inf")
    )
    n_time = valid.sum(1).clamp_min(1).float()
    pooled = LOCAL_TEMPERATURE * (torch.logsumexp(similarity / LOCAL_TEMPERATURE, 2) - torch.log(n_time)[:, None, None])
    pooled = pooled.masked_fill(~word_mask[None], 0)
    return pooled.sum(-1) / word_mask.sum(-1).clamp_min(1).float()[None]


def sample_contiguous_temporal_masks(valid_mask: np.ndarray, sample_ids: np.ndarray, seed: int, epoch: int) -> np.ndarray:
    masks = np.zeros_like(np.asarray(valid_mask, dtype=bool), dtype=bool)
    for row, sample_id in enumerate(np.asarray(sample_ids, dtype=np.int64)):
        valid = np.asarray(valid_mask[row], dtype=bool)
        length = int(valid.sum())
        decision_rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(sample_id), 17]))
        if length <= 0 or decision_rng.random() >= MASK_PROBABILITY:
            continue
        mask_length = min(MASK_LENGTH, length)
        start_rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(sample_id)]))
        start = int(start_rng.integers(0, length - mask_length + 1))
        masks[row, np.flatnonzero(valid)[start:start + mask_length]] = True
    return masks


class ResidualTextAdapter(nn.Module):
    """Identity-initialized residual sentence adapter."""

    def __init__(self, dim: int = TARGET_DIM, bottleneck: int = ADAPTER_BOTTLENECK, alpha: float = ADAPTER_ALPHA):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        native = F.normalize(text.float(), dim=-1)
        residual = self.up(self.activation(self.down(native)))
        return F.normalize(native + self.alpha * residual, dim=-1)


class GLANCEModel(nn.Module):
    def __init__(self, lead_indices: list[list[int]], n_channels: int, n_freq: int, n_time: int, device: str):
        super().__init__()
        self.lead_indices = [torch.tensor(indices, dtype=torch.long) for indices in lead_indices]
        self.n_leads = len(lead_indices)
        self.n_channels = n_channels
        self.n_freq = n_freq
        self.n_time = n_time
        self.lead_grus = nn.ModuleList([nn.GRU(len(indices) * n_freq, 128, batch_first=True, bidirectional=True) for indices in lead_indices])
        self.lead_mask_tokens = nn.ParameterList([nn.Parameter(torch.randn(1, 1, len(indices) * n_freq) * 0.02) for indices in lead_indices])
        self.lead_embedding = nn.Parameter(torch.randn(1, 1, len(lead_indices), TOKEN_DIM) * 0.02)
        spatial_layer = nn.TransformerEncoderLayer(TOKEN_DIM, 4, 512, 0.1, "gelu", batch_first=True, norm_first=True)
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, 1, norm=nn.LayerNorm(TOKEN_DIM))
        self.spatial_cls = nn.Parameter(torch.randn(1, 1, TOKEN_DIM) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, TOKEN_DIM) * 0.02)
        self.positional_embedding = nn.Parameter(torch.randn(1, n_time + 1, TOKEN_DIM) * 0.02)
        predictor_layer = nn.TransformerEncoderLayer(TOKEN_DIM, 8, 1024, 0.1, "gelu", batch_first=True, norm_first=True)
        self.predictor = nn.TransformerEncoder(predictor_layer, 2, norm=nn.LayerNorm(TOKEN_DIM))
        self.projection = nn.Linear(TOKEN_DIM, TARGET_DIM)
        self.local_encoder = nn.Sequential(nn.Conv1d(TOKEN_DIM, 128, 5, padding=2), nn.GELU(), nn.Linear(128, 128, bias=False))
        self.word_projector = nn.Linear(TARGET_DIM, 128, bias=False)
        self.to(device)

    def encode_time(self, x: Any, valid: Any, mask: Any | None = None):
        import torch
        import torch.nn as nn

        batch, _, _, n_time = x.shape
        lengths = valid.sum(1).clamp_min(1).to(torch.int64).cpu()
        states = []
        for indices, gru, mask_token in zip(self.lead_indices, self.lead_grus, self.lead_mask_tokens):
            lead_x = x.index_select(1, indices.to(x.device)).permute(0, 3, 1, 2).reshape(batch, n_time, -1)
            if mask is not None:
                lead_x = torch.where(mask.bool().unsqueeze(-1), mask_token.expand(batch, n_time, -1), lead_x)
            packed = nn.utils.rnn.pack_padded_sequence(lead_x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=n_time)
            states.append(out.masked_fill(~valid.unsqueeze(-1), 0))
        lead_states = torch.stack(states, 2) + self.lead_embedding
        lead_tokens = lead_states.reshape(batch * n_time, self.n_leads, TOKEN_DIM)
        spatial_input = torch.cat([self.spatial_cls.expand(batch * n_time, -1, -1), lead_tokens], dim=1)
        spatial_out = self.spatial_encoder(spatial_input)
        time_tokens = spatial_out[:, 0, :].reshape(batch, n_time, TOKEN_DIM).masked_fill(~valid.unsqueeze(-1), 0)
        post_spatial = spatial_out[:, 1:, :].reshape(batch, n_time, self.n_leads, TOKEN_DIM)
        return time_tokens, lead_states, post_spatial

    def forward(self, x: Any, valid_time: Any, artificial_mask: Any | None = None, return_intermediates: bool = False):
        if tuple(x.shape[1:]) != (self.n_channels, self.n_freq, self.n_time):
            raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")
        valid = valid_time.bool()
        time_tokens, lead_states, spatial_tokens = self.encode_time(x, valid, artificial_mask)
        local = self.local_encoder[0](time_tokens.transpose(1, 2))
        local = self.local_encoder[1](local).transpose(1, 2)
        local = F.normalize(self.local_encoder[2](local).float(), dim=-1).masked_fill(~valid.unsqueeze(-1), 0)
        sequence = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), time_tokens], 1)
        sequence = sequence + self.positional_embedding[:, : self.n_time + 1]
        key_padding = torch.cat([torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device), ~valid], 1)
        predicted = self.predictor(sequence, src_key_padding_mask=key_padding)
        predictor_cls = predicted[:, 0].float()
        brain = F.normalize(self.projection(predictor_cls).float(), dim=-1)
        if return_intermediates:
            return brain, local, predictor_cls, lead_states, spatial_tokens, time_tokens, key_padding
        return brain


def build_parameter_groups(model: GLANCEModel, adapter: ResidualTextAdapter, phase: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Any], list[Any]]:
    import torch

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)

    groups_by_name: dict[str, list[Any]] = {
        "global_base": list(model.lead_grus.parameters()) + list(model.spatial_encoder.parameters()) + [model.lead_embedding, model.spatial_cls],
        "lead_mask_tokens": list(model.lead_mask_tokens.parameters()),
        "predictor_projection": list(model.predictor.parameters()) + list(model.projection.parameters()),
        "temporal_cls_positions": [model.cls_token, model.positional_embedding],
        "local_word": list(model.local_encoder.parameters()) + list(model.word_projector.parameters()),
        "sentence_adapter": list(adapter.parameters()),
    }
    for group_name in phase["trainable"]:
        for parameter in groups_by_name[group_name]:
            parameter.requires_grad_(True)

    optimizer_groups: list[dict[str, Any]] = []
    for group_name, learning_rate in phase["learning_rates"].items():
        parameters = [parameter for parameter in groups_by_name[group_name] if parameter.requires_grad]
        optimizer_groups.append({"name": group_name, "params": parameters, "lr": float(learning_rate)})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=WEIGHT_DECAY)
    trainable_model = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_adapter = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    return optimizer, trainable_model, trainable_adapter


def verify_optimizer_and_freezing(model: GLANCEModel, adapter: ResidualTextAdapter, phase: dict[str, Any], optimizer: Any) -> dict[str, Any]:
    import torch

    expected = set(phase["learning_rates"])
    actual = {group["name"] for group in optimizer.param_groups}
    if actual != expected:
        raise AssertionError(f"Optimizer groups mismatch for {phase['name']}: expected={expected}, actual={actual}")
    adapter_trainable = any(parameter.requires_grad for parameter in adapter.parameters())
    if adapter_trainable != ("sentence_adapter" in expected):
        raise AssertionError("sentence-adapter freeze state does not match phase")
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    expected_ids |= {id(parameter) for parameter in adapter.parameters() if parameter.requires_grad}
    if optimizer_ids != expected_ids:
        raise AssertionError("optimizer does not contain exactly the trainable parameters")
    return {
        "optimizer_groups": [{"name": group["name"], "lr": group["lr"], "parameter_count": sum(parameter.numel() for parameter in group["params"])} for group in optimizer.param_groups],
        "trainable_adapter": adapter_trainable,
        "trainable_model_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def configure_phase_modes(model: GLANCEModel, adapter: ResidualTextAdapter, phase: dict[str, Any]) -> None:
    """Set train/evaluation modes for the joint training phase."""
    model.eval()
    adapter.eval()
    model.train()
    model.local_encoder.train()
    if "sentence_adapter" in phase["trainable"]:
        adapter.train()


def make_banks(tids: np.ndarray, indices: np.ndarray, seed: int, directory: Path) -> list[np.ndarray]:
    held_out = np.unique(tids[indices])
    if len(held_out) < 100:
        raise ValueError(f"fewer than 100 held-out transcript identities: {len(held_out)}")
    banks: list[np.ndarray] = []
    directory.mkdir(parents=True, exist_ok=True)
    for bank_index in range(30):
        path = directory / f"bank_{bank_index:02d}.npy"
        if path.exists():
            bank = np.load(path)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([seed, bank_index, 991]))
            bank = np.empty((len(indices), 100), dtype=np.int64)
            for row, transcript_id in enumerate(tids[indices]):
                distractors = rng.choice(held_out[held_out != transcript_id], 99, replace=False)
                candidates = np.concatenate(([transcript_id], distractors))
                rng.shuffle(candidates)
                bank[row] = candidates
            np.save(path, bank)
        if bank.shape != (len(indices), 100) or not np.all(np.sum(bank == tids[indices, None], axis=1) == 1):
            raise ValueError(f"invalid deterministic evaluation bank: {path}")
        banks.append(np.asarray(bank, dtype=np.int64))
    return banks


def metrics_from_ranks(ranks: np.ndarray) -> dict[str, Any]:
    return {
        "recall@1": float(np.mean(ranks <= 1) * 100),
        "recall@5": float(np.mean(ranks <= 5) * 100),
        "recall@10": float(np.mean(ranks <= 10) * 100),
        "recall@50": float(np.mean(ranks <= 50) * 100),
        "mrr": float(np.mean(1.0 / ranks)),
        "n_queries": int(len(ranks)),
    }


def evaluate(model: GLANCEModel, adapter: ResidualTextAdapter, data: dict[str, Any], indices: np.ndarray, banks: list[np.ndarray], tag: str, output_dir: Path, local_status: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    model.eval()
    adapter.eval()
    features = data["features"]
    valid = torch.from_numpy(data["valid"]).bool()
    x = torch.from_numpy(features)
    global_parts: list[Any] = []
    local_parts: list[Any] = []
    valid_parts: list[Any] = []
    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start:start + BATCH_SIZE]
            result = model(x[batch_indices].float().to(device), valid[batch_indices].to(device), return_intermediates=True)
            global_parts.append(result[0].cpu())
            local_parts.append(result[1].cpu())
            valid_parts.append(valid[batch_indices])
    global_brain = torch.cat(global_parts).to(device)
    local_brain = torch.cat(local_parts).to(device)
    valid_eval = torch.cat(valid_parts).to(device)
    sentences = torch.from_numpy(data["sentences"]).float().to(device)
    words = torch.from_numpy(data["words"]).float().to(device)
    word_mask = torch.from_numpy(data["word_mask"]).bool().to(device)
    with torch.no_grad():
        text = adapter(sentences)
        projected_words = F.normalize(model.word_projector(words), dim=-1)
    tids = data["tids"]
    first_row: dict[int, int] = {}
    for row, transcript_id in enumerate(tids):
        first_row.setdefault(int(transcript_id), row)
    rank_values = {key: [] for key in ("global", "local", "combined")}
    with torch.no_grad():
        for bank in banks:
            candidate_indices = np.asarray([[first_row[int(transcript_id)] for transcript_id in row] for row in bank], dtype=np.int64)
            candidates = torch.from_numpy(candidate_indices).to(device)
            global_scores = torch.einsum("qd,qcd->qc", global_brain, text[candidates])
            candidate_words = projected_words[candidates]
            similarity = torch.einsum("qtd,qcwd->qctw", local_brain, candidate_words).masked_fill(~valid_eval[:, None, :, None], float("-inf"))
            n_time = valid_eval.sum(1).clamp_min(1).float()
            local_score = LOCAL_TEMPERATURE * (torch.logsumexp(similarity / LOCAL_TEMPERATURE, 2) - torch.log(n_time)[:, None, None])
            local_score = local_score.masked_fill(~word_mask[candidates], 0).sum(-1) / word_mask[candidates].sum(-1).clamp_min(1).float()
            combined_scores = global_scores + LOCAL_WEIGHT * local_score
            positive = torch.tensor([int(np.flatnonzero(row == tids[indices[row_index]])[0]) for row_index, row in enumerate(bank)], device=device)
            for key, scores in (("global", global_scores), ("local", local_score), ("combined", combined_scores)):
                rank_values[key].extend((torch.argsort(scores, 1, descending=True) == positive[:, None]).nonzero()[:, 1].add(1).cpu().numpy().tolist())
    metrics = {key: {**metrics_from_ranks(np.asarray(values)), "local_weight": LOCAL_WEIGHT if key != "global" else None, "local_branch_status": local_status} for key, values in rank_values.items()}
    metrics["n_banks"] = len(banks)
    metrics["checkpoint_tag"] = tag
    write_json(output_dir / "evaluations" / tag / "metrics.json", metrics)
    return metrics


def scalar_metric(metrics: dict[str, Any], phase: dict[str, Any]) -> float:
    branch, metric_name = phase["selection_metric"].split(".")
    return float(metrics[branch][metric_name])


def source_snapshot(output_dir: Path) -> dict[str, str]:
    snapshot_dir = output_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    files = [
        Path(__file__),
        Path(__file__).with_name("run_glance.py"),
    ]
    hashes = {}
    for source in files:
        destination = snapshot_dir / source.name
        shutil.copy2(source, destination)
        hashes[source.name] = sha256_file(destination)
    return hashes


def checkpoint_payload(model: GLANCEModel, adapter: ResidualTextAdapter, optimizer: Any, order_rng: Any, history: list[dict[str, Any]], phase_summaries: list[dict[str, Any]], phase: dict[str, Any], phase_epoch: int, global_epoch: int, best: dict[str, Any], initial_state_hash: str, current_validation_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "format": "glance_200_v1",
        "brain_model": model.state_dict(),
        "text_adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "phase": phase["name"],
        "phase_epoch_completed": phase_epoch,
        "global_epoch_completed": global_epoch,
        "next_phase_epoch": phase_epoch + 1,
        "history": history,
        "phase_summaries": phase_summaries,
        "best_tracking": best,
        "current_epoch_validation_metrics": current_validation_metrics,
        "current_epoch_validation_score": scalar_metric(current_validation_metrics, phase) if current_validation_metrics is not None else None,
        "initial_state_hash": initial_state_hash,
        "rng_state": capture_rng_state(order_rng),
    }


def save_model_checkpoint(path: Path, model: GLANCEModel, adapter: ResidualTextAdapter, metadata: dict[str, Any]) -> None:
    atomic_torch_save({"brain_model": model.state_dict(), "text_adapter": adapter.state_dict(), **metadata}, path)


def initial_state(model: GLANCEModel, adapter: ResidualTextAdapter, order_rng: Any, seed: int, output_dir: Path) -> tuple[dict[str, Any], str]:
    path = output_dir / "checkpoints" / "initial_state.pt"
    if path.exists():
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["brain_model"], strict=True)
        adapter.load_state_dict(payload["text_adapter"], strict=True)
        restore_rng_state(payload["rng_state"], order_rng)
        return payload, sha256_file(path)
    payload = {
        "format": "glance_200_initial_v1",
        "seed": seed,
        "brain_model": cpu_state_dict(model),
        "text_adapter": cpu_state_dict(adapter),
        "rng_state": capture_rng_state(order_rng),
        "initialization": "fresh random brain model plus identity-initialized ResidualTextAdapter",
        "trained_checkpoint_loading": False,
    }
    atomic_torch_save(payload, path)
    return payload, sha256_file(path)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Run checks without writing experiment state or changing the future run's RNG state."""
    import torch

    if TOTAL_EPOCHS != 200:
        raise AssertionError(f"phase lengths sum to {TOTAL_EPOCHS}, not 200")
    paths = make_paths(args.seed, Path(args.output_root), Path(args.data_root), Path(args.split_root) if args.split_root else None, Path(args.channel_order))
    lead_names, lead_channels, lead_indices = derive_lead_mapping(paths.channel_order)
    data = load_prepared_data(paths, args.seed)
    # Use a forked CPU seed and restore every caller RNG state so preflight cannot
    # change the initialization stream used by the subsequent training launch.
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    with torch.random.fork_rng(devices=[]):
        seed_everything(args.seed)
        feature_shape = data["features"].shape
        model = GLANCEModel(lead_indices, int(feature_shape[1]), int(feature_shape[2]), int(feature_shape[3]), "cpu")
        adapter = ResidualTextAdapter()
        checks = []
        for phase in PHASES:
            optimizer, _, _ = build_parameter_groups(model, adapter, phase)
            checks.append({"phase": phase["name"], **verify_optimizer_and_freezing(model, adapter, phase, optimizer)})
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    report = {
        "phase_total_epochs": TOTAL_EPOCHS,
        "phase_starts": PHASE_STARTS,
        "phase_ends": PHASE_ENDS,
        "seed": args.seed,
        "lead_count_derived": len(lead_names),
        "channels_derived": sum(len(channels) for channels in lead_channels),
        "feature_shape": list(data["features"].shape),
        "n_examples": int(len(data["windows"])),
        "split_rows": {name: int(len(rows)) for name, rows in data["splits"].items()},
        "held_out_identities": {name: int(len(np.unique(data["tids"][rows]))) for name, rows in data["splits"].items()},
        "lead_names": lead_names,
        "checks": checks,
        "paths": {key: str(value) for key, value in vars(paths).items()},
    }
    return report


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    paths = make_paths(args.seed, Path(args.output_root), Path(args.data_root), Path(args.split_root) if args.split_root else None, Path(args.channel_order))
    output_dir = paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "checkpoints" / "latest.pt").exists() and not args.resume:
        raise FileExistsError(f"latest checkpoint exists; pass --resume to continue: {output_dir / 'checkpoints' / 'latest.pt'}")

    seed_everything(args.seed)
    data = load_prepared_data(paths, args.seed)
    lead_names, lead_channels, lead_indices = derive_lead_mapping(paths.channel_order)
    n_channels, n_freq, n_time = map(int, data["features"].shape[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLANCEModel(lead_indices, n_channels, n_freq, n_time, device)
    adapter = ResidualTextAdapter().to(device)
    order_rng = torch.Generator().manual_seed(args.seed + 777)
    initial_payload, initial_hash = initial_state(model, adapter, order_rng, args.seed, output_dir)
    bank_root = output_dir / "evaluation_100way" / "banks" / f"seed{args.seed}"
    val_banks = make_banks(data["tids"], data["splits"]["val"], args.seed, bank_root / "val")
    test_banks = make_banks(data["tids"], data["splits"]["test"], args.seed, bank_root / "test")
    source_hashes = source_snapshot(output_dir)
    config = {
        "model": "GLANCE",
        "training_protocol": "joint_retrieval_200",
        "seed": args.seed,
        "device": device,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRAD_CLIP_NORM,
        "retrieval_temperature": RETRIEVAL_TEMPERATURE,
        "local_scoring_temperature": LOCAL_TEMPERATURE,
        "local_weight": LOCAL_WEIGHT,
        "mask_length": MASK_LENGTH,
        "mask_probability": MASK_PROBABILITY,
        "consistency_weight": CONSISTENCY_WEIGHT,
        "initialization": "fresh random brain model; fresh identity-initialized sentence adapter",
        "trained_initialization_loaded": False,
        "initial_state": {"path": str(output_dir / "checkpoints" / "initial_state.pt"), "sha256": initial_hash},
        "input_shape": ["B", n_channels, n_freq, n_time],
        "expected_input_shape_verified": list(data["features"].shape),
        "lead_count": len(lead_names),
        "lead_names": lead_names,
        "lead_channels": lead_channels,
        "data_paths": {key: str(value) for key, value in vars(paths).items()},
        "split_references": {name: str(paths.split_root / f"seed{args.seed}" / f"{name}.npy") for name in ("train", "val", "test")},
        "candidate_banks": {"validation": str(bank_root / "val"), "test": str(bank_root / "test"), "count": 30, "candidates": 100},
        "dataset_manifest": str(paths.data_root / "manifest.json"),
        "normalization_metadata": str(paths.data_root / "normalization" / "metadata.json"),
        "embedding_metadata": {
            "sentence": str(paths.data_root / "text" / "sentence_metadata.json"),
            "word": str(paths.data_root / "text" / "word_metadata.json"),
        },
        "runtime": {
            "python": sys.version,
            "pytorch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": {
                name: __import__("importlib.metadata", fromlist=["version"]).version(name)
                for name in ("numpy", "pandas", "transformers", "tokenizers")
            },
        },
        "phase_definitions": list(PHASES),
        "source_code_snapshot": source_hashes,
        "selection": "JOINT validation combined R@10 at local weight 0.70; strict > tie rule; epoch 0 diagnostic only",
        "resume": "latest.pt includes current-phase optimizer, counters, history, best tracking, and Python/NumPy/PyTorch/order RNG states",
    }
    write_json(output_dir / "config.json", config)
    (output_dir / "checkpoints" / "initial_state.sha256").write_text(initial_hash + "\n", encoding="utf-8")
    write_json(output_dir / "channel_mapping.json", {"lead_names": lead_names, "lead_channels": lead_channels, "lead_indices": lead_indices})
    pd.DataFrame({"channel_index": range(n_channels), "electrode_label": pd.read_csv(paths.channel_order)["electrode_label"].astype(str)}).to_csv(output_dir / "included_channels.csv", index=False)

    history: list[dict[str, Any]] = []
    phase_summaries: list[dict[str, Any]] = []
    start_phase_index = 0
    start_phase_epoch = 1
    global_epoch = 0
    latest_path = output_dir / "checkpoints" / "latest.pt"
    if args.resume and latest_path.exists():
        latest = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(latest["brain_model"], strict=True)
        adapter.load_state_dict(latest["text_adapter"], strict=True)
        global_epoch = int(latest["global_epoch_completed"])
        phase_name = str(latest["phase"])
        start_phase_index = next(index for index, phase in enumerate(PHASES) if phase["name"] == phase_name)
        start_phase_epoch = int(latest["phase_epoch_completed"]) + 1
        history = latest.get("history", [])
        phase_summaries = latest.get("phase_summaries", [])
        restore_rng_state(latest["rng_state"], order_rng)

    x = torch.from_numpy(data["features"])
    valid = torch.from_numpy(data["valid"]).bool()
    selection_checkpoints: dict[str, str] = {}
    for phase_index in range(start_phase_index, len(PHASES)):
        phase = PHASES[phase_index]
        phase_name = phase["name"]
        phase_start_epoch = PHASE_STARTS[phase_name]
        phase_end_epoch = PHASE_ENDS[phase_name]
        phase_epoch_start = start_phase_epoch if phase_index == start_phase_index else 1
        optimizer, trainable_model, trainable_adapter = build_parameter_groups(model, adapter, phase)
        verify_optimizer_and_freezing(model, adapter, phase, optimizer)
        if args.resume and latest_path.exists() and phase_index == start_phase_index:
            latest = torch.load(latest_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(latest["optimizer"])
            verify_optimizer_and_freezing(model, adapter, phase, optimizer)
            best = latest["best_tracking"]
        else:
            best = {"score": -float("inf"), "epoch": None, "phase_epoch": None, "metrics": None, "checkpoint": None}
        configure_phase_modes(model, adapter, phase)
        for phase_epoch in range(phase_epoch_start, int(phase["epochs"]) + 1):
            cumulative_epoch = phase_start_epoch + phase_epoch - 1
            model.train(model.training)
            configure_phase_modes(model, adapter, phase)
            order = data["splits"]["train"][torch.randperm(len(data["splits"]["train"]), generator=order_rng).numpy()]
            loss_rows: list[dict[str, float]] = []
            for start in range(0, len(order), BATCH_SIZE):
                ids = order[start:start + BATCH_SIZE]
                xb = x[ids].float().to(device)
                vb = valid[ids].to(device)
                text_batch = torch.from_numpy(data["sentences"][ids]).float().to(device)
                positives = torch.from_numpy(data["tids"][ids, None] == data["tids"][ids, None].T).to(device)
                optimizer.zero_grad(set_to_none=True)
                full = model(xb, vb, return_intermediates=True)
                masked = None
                if phase["masked_view"]:
                    artificial_mask = sample_contiguous_temporal_masks(vb.cpu().numpy(), ids, args.seed, cumulative_epoch + 1000)
                    masked = model(xb, vb, torch.from_numpy(artificial_mask).to(device), return_intermediates=True)
                adapted_text = adapter(text_batch)
                global_full_scores = (full[0] @ adapted_text.T) / RETRIEVAL_TEMPERATURE
                combined_full_scores = global_full_scores
                if phase["local_weight"] > 0:
                    projected_words = F.normalize(model.word_projector(torch.from_numpy(data["words"][ids]).float().to(device)), dim=-1)
                    local = local_scores(full[1], vb, projected_words, torch.from_numpy(data["word_mask"][ids]).bool().to(device))
                    combined_full_scores = (full[0] @ adapted_text.T + LOCAL_WEIGHT * local) / RETRIEVAL_TEMPERATURE
                else:
                    local = None
                if phase_name == "JOINT":
                    masked_scores = (masked[0] @ adapted_text.T) / RETRIEVAL_TEMPERATURE
                    retrieval_full = sym_nce(combined_full_scores, positives)
                    masked_loss = sym_nce(masked_scores, positives)
                    retrieval = 0.5 * (retrieval_full + masked_loss)
                else:
                    raise AssertionError(phase_name)
                consistency = torch.zeros((), device=device)
                if phase["consistency_weight"]:
                    consistency = (1 - (masked[0] * full[0].detach()).sum(-1)).mean()
                loss = retrieval + phase["consistency_weight"] * consistency
                if loss.requires_grad is False:
                    raise AssertionError(f"{phase_name} produced a non-differentiable loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_model, GRAD_CLIP_NORM)
                if trainable_adapter:
                    torch.nn.utils.clip_grad_norm_(trainable_adapter, GRAD_CLIP_NORM)
                optimizer.step()
                loss_rows.append({
                    "total": float(loss.detach().cpu()),
                    "retrieval": float(retrieval.detach().cpu()),
                    "masked_retrieval": float(masked_loss.detach().cpu()),
                    "consistency": float(consistency.detach().cpu()),
                })

            should_evaluate = cumulative_epoch % 5 == 0 or cumulative_epoch == phase_end_epoch
            val_metrics = None
            eval_tag = None
            if should_evaluate:
                local_status = "trained"
                eval_tag = f"validation_epoch_{cumulative_epoch:04d}"
                val_metrics = evaluate(model, adapter, data, data["splits"]["val"], val_banks, eval_tag, output_dir, local_status)
                score = scalar_metric(val_metrics, phase)
                if score > float(best["score"]):
                    best = {
                        "score": score,
                        "epoch": cumulative_epoch,
                        "phase_epoch": phase_epoch,
                        "metrics": val_metrics,
                        "checkpoint": str(output_dir / "checkpoints" / f"phase_{phase_name}_best.pt"),
                    }
                    save_model_checkpoint(output_dir / "checkpoints" / f"phase_{phase_name}_best.pt", model, adapter, {
                        "phase": phase_name,
                        "phase_epoch": phase_epoch,
                        "global_epoch": cumulative_epoch,
                        "selection_rule": phase["selection_metric"],
                        "selected_validation_score": score,
                        "validation_metrics": val_metrics,
                        "local_weight_for_combined": LOCAL_WEIGHT,
                    })
            mean_losses = {key: float(np.mean([row[key] for row in loss_rows])) for key in loss_rows[0]}
            row = {
                "phase": phase_name,
                "phase_epoch": phase_epoch,
                "cumulative_epoch": cumulative_epoch,
                "train_loss": mean_losses["total"],
                "retrieval_loss": mean_losses["retrieval"],
                "masked_retrieval_loss": mean_losses["masked_retrieval"],
                "consistency_loss": mean_losses["consistency"],
                "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                "local_weight": phase["local_weight"],
                "validation_evaluated": should_evaluate,
                "validation_tag": eval_tag,
                "validation": val_metrics,
                "best_selection_score_so_far": None if best["epoch"] is None else best["score"],
                "best_selection_epoch_so_far": best["epoch"],
            }
            history.append(row)
            archive_due = cumulative_epoch % 10 == 0 or cumulative_epoch == phase_end_epoch
            if archive_due:
                save_model_checkpoint(output_dir / "checkpoints" / "archive" / f"epoch_{cumulative_epoch:04d}.pt", model, adapter, {
                    "phase": phase_name,
                    "phase_epoch": phase_epoch,
                    "global_epoch": cumulative_epoch,
                    "checkpoint_kind": "periodic_archive" if cumulative_epoch % 10 == 0 and cumulative_epoch != phase_end_epoch else "phase_endpoint_archive",
                    "validation_metrics": val_metrics,
                    "validation_score_for_current_phase": scalar_metric(val_metrics, phase) if val_metrics is not None else None,
                })
            if cumulative_epoch == 100:
                save_model_checkpoint(output_dir / "checkpoints" / "epoch_100.pt", model, adapter, {
                    "phase": phase_name,
                    "phase_epoch": phase_epoch,
                    "global_epoch": cumulative_epoch,
                    "checkpoint_kind": "epoch_100_diagnostic",
                    "validation_metrics": val_metrics,
                    "validation_score_for_current_phase": scalar_metric(val_metrics, phase) if val_metrics is not None else None,
                })
            atomic_torch_save(checkpoint_payload(model, adapter, optimizer, order_rng, history, phase_summaries, phase, phase_epoch, cumulative_epoch, best, initial_hash, val_metrics), latest_path)
            write_json(output_dir / "history.json", history)
            print(json.dumps({"phase": phase_name, "phase_epoch": phase_epoch, "cumulative_epoch": cumulative_epoch, "train_loss": row["train_loss"], "validation": None if val_metrics is None else scalar_metric(val_metrics, phase)}, default=str), flush=True)

        endpoint_path = output_dir / "checkpoints" / "archive" / f"epoch_{phase_end_epoch:04d}.pt"
        phase_summary = {
            "phase": phase_name,
            "description": phase["description"],
            "start_epoch": phase_start_epoch,
            "end_epoch": phase_end_epoch,
            "duration": phase["epochs"],
            "selection_metric": phase["selection_metric"],
            "best_checkpoint": best["checkpoint"],
            "best_epoch": best["epoch"],
            "best_phase_epoch": best["phase_epoch"],
            "best_validation_score": best["score"],
            "best_validation_metrics": best["metrics"],
            "endpoint_checkpoint": str(endpoint_path),
            "endpoint_is_selection_eligible": True,
        }
        phase_summaries = [summary for summary in phase_summaries if summary.get("phase") != phase_name] + [phase_summary]
        selection_checkpoints[phase_name] = str(output_dir / "checkpoints" / f"phase_{phase_name}_best.pt")
        if phase_name == "JOINT":
            save_model_checkpoint(output_dir / "checkpoints" / "final_epoch_200.pt", model, adapter, {
                "phase": phase_name,
                "phase_epoch": phase["epochs"],
                "global_epoch": TOTAL_EPOCHS,
                "checkpoint_kind": "final_epoch_200_diagnostic",
                "validation_metrics": history[-1]["validation"],
                "validation_score_for_current_phase": scalar_metric(history[-1]["validation"], phase) if history[-1]["validation"] is not None else None,
            })
        if phase_index < len(PHASES) - 1:
            import torch

            best_payload = torch.load(output_dir / "checkpoints" / f"phase_{phase_name}_best.pt", map_location=device, weights_only=False)
            model.load_state_dict(best_payload["brain_model"], strict=True)
            adapter.load_state_dict(best_payload["text_adapter"], strict=True)
            next_phase = PHASES[phase_index + 1]
            next_optimizer, _, _ = build_parameter_groups(model, adapter, next_phase)
            verify_optimizer_and_freezing(model, adapter, next_phase, next_optimizer)
            configure_phase_modes(model, adapter, next_phase)
            atomic_torch_save(checkpoint_payload(model, adapter, next_optimizer, order_rng, history, phase_summaries, next_phase, 0, {"score": -float("inf"), "epoch": None, "phase_epoch": None, "metrics": None, "checkpoint": None}, initial_hash, None), latest_path)
            start_phase_epoch = 1
        else:
            break

    # Main final model is the validation-selected JOINT checkpoint; epoch 200 and
    # epoch 100 are retained as diagnostics only.
    selected_path = output_dir / "checkpoints" / "phase_JOINT_best.pt"
    selected_payload = torch.load(selected_path, map_location=device, weights_only=False)
    model.load_state_dict(selected_payload["brain_model"], strict=True)
    adapter.load_state_dict(selected_payload["text_adapter"], strict=True)
    final_selected_test = evaluate(model, adapter, data, data["splits"]["test"], test_banks, "test_JOINT_selected", output_dir, "trained")
    epoch_100_payload = torch.load(output_dir / "checkpoints" / "epoch_100.pt", map_location=device, weights_only=False)
    model.load_state_dict(epoch_100_payload["brain_model"], strict=True)
    adapter.load_state_dict(epoch_100_payload["text_adapter"], strict=True)
    epoch_100_test = evaluate(model, adapter, data, data["splits"]["test"], test_banks, "test_epoch_100", output_dir, "trained")
    final_epoch_payload = torch.load(output_dir / "checkpoints" / "final_epoch_200.pt", map_location=device, weights_only=False)
    model.load_state_dict(final_epoch_payload["brain_model"], strict=True)
    adapter.load_state_dict(final_epoch_payload["text_adapter"], strict=True)
    final_epoch_test = evaluate(model, adapter, data, data["splits"]["test"], test_banks, "test_final_epoch_200", output_dir, "trained")
    phase_tests = {}
    for phase in PHASES:
        payload = torch.load(output_dir / "checkpoints" / f"phase_{phase['name']}_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(payload["brain_model"], strict=True)
        adapter.load_state_dict(payload["text_adapter"], strict=True)
        status = "trained"
        phase_tests[phase["name"]] = evaluate(model, adapter, data, data["splits"]["test"], test_banks, f"test_{phase['name']}_selected", output_dir, status)
    summary = {
        "model": "GLANCE",
        "training_protocol": "joint_retrieval_200",
        "seed": args.seed,
        "output_dir": str(output_dir),
        "initial_state_sha256": initial_hash,
        "phase_summaries": phase_summaries,
        "phase_selected_test_evaluations": phase_tests,
        "main_final_model": {"checkpoint": str(selected_path), "selection_rule": "JOINT validation combined recall@10", "test_evaluation": final_selected_test},
        "epoch_100_diagnostic": {"checkpoint": str(output_dir / "checkpoints" / "epoch_100.pt"), "selection_rule": "not selected; fixed epoch-100 diagnostic", "test_evaluation": epoch_100_test},
        "final_epoch_200_diagnostic": {"checkpoint": str(output_dir / "checkpoints" / "final_epoch_200.pt"), "selection_rule": "not selected; fixed epoch-200 diagnostic", "test_evaluation": final_epoch_test},
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "phase_summaries.json", phase_summaries)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", required=True, help="Canonical prepared-data directory; no raw data are stored in this repository.")
    parser.add_argument("--split-root")
    parser.add_argument("--channel-order", required=True, help="CSV defining the channel order and lead labels.")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train(args)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
