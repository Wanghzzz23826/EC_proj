from __future__ import annotations

import argparse
from collections import deque
import json
import os
import pickle as pkl
import sys
import time
from dataclasses import dataclass

# Must be set before importing jax/brainpy.
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

_DEFAULT_LOCAL_BRAINPY = "/home/wanghz/Study/brainpy/BrainPy-master"
if os.environ.get("EC_V1_USE_LOCAL_BRAINPY", "0") == "1":
    local_brainpy = os.environ.get("EC_V1_BRAINPY_PATH", _DEFAULT_LOCAL_BRAINPY)
    if os.path.isdir(local_brainpy) and local_brainpy not in sys.path:
        sys.path.insert(0, local_brainpy)

import brainpy as bp
import brainpy.math as bm
import brainstate as bst
import jax
import jax.numpy as jnp
import numpy as np

try:
    import wandb
except Exception:
    wandb = None

from common.types import InputParams, NodeParams, SynapseParams, to_brainpy_csr
from brainpy_impl import classification_tools, load_sparse, models2 as bp_models


@dataclass
class VCDNPZSequenceGenerator:
    """
    NPZ-backed VCD sample generator.
    Expects arrays produced by tensorflow_impl/generate_vcd_dataset.py:
      x: [N, time, input], y: [N, chunks], w: [N, chunks]
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        *,
        n_input: int,
        down_sample: int,
        seed: int,
        batch_size: int = 1,
        shuffle: bool = True,
    ):
        if batch_size != 1:
            raise ValueError("Current train_bp.py supports batch_size=1 only.")
        if x.ndim != 3:
            raise ValueError(f"x must be [N, T, C], got {x.shape}")
        if y.ndim != 2 or w.ndim != 2:
            raise ValueError(f"y and w must be [N, chunks], got y={y.shape}, w={w.shape}")
        if x.shape[0] != y.shape[0] or x.shape[0] != w.shape[0]:
            raise ValueError(
                f"Sample size mismatch: x={x.shape[0]}, y={y.shape[0]}, w={w.shape[0]}"
            )
        if y.shape != w.shape:
            raise ValueError(f"y/w shape mismatch: y={y.shape}, w={w.shape}")

        self.x = _project_input_features(np.asarray(x, dtype=np.float32), n_input=n_input)
        self.y = np.asarray(y, dtype=np.int32)
        self.w = np.asarray(w, dtype=np.float32)
        self.n_samples = int(self.x.shape[0])
        self.seq_len = int(self.x.shape[1])
        self.n_chunks = int(self.y.shape[1])
        self.down_sample = int(down_sample)
        self.shuffle = bool(shuffle)
        self.rd = np.random.RandomState(seed)
        self._order = np.arange(self.n_samples, dtype=np.int32)
        self._cursor = 0
        self._reshuffle()

        if self.seq_len % self.down_sample != 0:
            raise ValueError(
                f"seq_len={self.seq_len} must be divisible by down_sample={self.down_sample}"
            )
        inferred_chunks = self.seq_len // self.down_sample
        if inferred_chunks != self.n_chunks:
            raise ValueError(
                f"Chunk mismatch: y has {self.n_chunks}, but seq_len/down_sample={inferred_chunks}"
            )

    def _reshuffle(self):
        if self.shuffle:
            self.rd.shuffle(self._order)
        self._cursor = 0

    def sample(self):
        if self._cursor >= self.n_samples:
            self._reshuffle()
        idx = int(self._order[self._cursor])
        self._cursor += 1
        return self.x[idx : idx + 1], self.y[idx : idx + 1], self.w[idx : idx + 1]


def _project_input_features(x: np.ndarray, *, n_input: int) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected [N, T, C] input array, got {x.shape}")
    if n_input <= 0:
        raise ValueError(f"n_input must be positive, got {n_input}")

    in_dim = int(x.shape[-1])
    if in_dim == n_input:
        return x
    if in_dim > n_input:
        idx = np.linspace(0, in_dim - 1, num=n_input, dtype=np.int32)
        return x[..., idx]

    pad_width = ((0, 0), (0, 0), (0, n_input - in_dim))
    return np.pad(x, pad_width=pad_width, mode="constant")


def _load_npz_field(npz_obj, keys: tuple[str, ...], *, path: str):
    for key in keys:
        if key in npz_obj:
            return npz_obj[key]
    raise KeyError(f"None of keys {keys} found in npz: {path}")


def _load_vcd_npz_arrays(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"VCD npz not found: {path}")

    with np.load(path, allow_pickle=True) as d:
        x = np.asarray(_load_npz_field(d, ("x", "inputs"), path=path), dtype=np.float32)
        y = np.asarray(_load_npz_field(d, ("y", "labels"), path=path), dtype=np.int32)
        w = np.asarray(_load_npz_field(d, ("w", "weights"), path=path), dtype=np.float32)
        if "image_label" in d:
            image_label = np.asarray(d["image_label"], dtype=np.int32)
        elif "image_labels" in d:
            image_label = np.asarray(d["image_labels"], dtype=np.int32)
        else:
            image_label = np.zeros_like(y, dtype=np.int32)

    if x.ndim != 3:
        raise ValueError(f"VCD x must be [N, T, C], got {x.shape} from {path}")
    if y.ndim == 3 and y.shape[-1] == 1:
        y = y[..., 0]
    if w.ndim == 3 and w.shape[-1] == 1:
        w = w[..., 0]
    if image_label.ndim == 3 and image_label.shape[-1] == 1:
        image_label = image_label[..., 0]

    if y.ndim != 2 or w.ndim != 2:
        raise ValueError(f"VCD y/w must be [N, chunks], got y={y.shape}, w={w.shape}")
    if image_label.ndim != 2:
        image_label = np.zeros_like(y, dtype=np.int32)

    n = int(x.shape[0])
    if int(y.shape[0]) != n or int(w.shape[0]) != n or int(image_label.shape[0]) != n:
        raise ValueError(
            f"Sample-count mismatch in {path}: x={x.shape[0]}, y={y.shape[0]}, "
            f"w={w.shape[0]}, image_label={image_label.shape[0]}"
        )
    if y.shape != w.shape:
        raise ValueError(f"Label/weight shape mismatch in {path}: y={y.shape}, w={w.shape}")

    return x, y, w, image_label


def _split_vcd_arrays(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    image_label: np.ndarray,
    *,
    val_ratio: float,
    seed: int,
):
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    n = int(x.shape[0])
    if n < 2:
        raise ValueError(f"Need at least 2 samples to split train/val, got {n}")

    rd = np.random.RandomState(seed)
    perm = rd.permutation(n)
    n_val = int(np.round(n * val_ratio))
    n_val = max(1, min(n - 1, n_val))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train = (x[train_idx], y[train_idx], w[train_idx], image_label[train_idx])
    val = (x[val_idx], y[val_idx], w[val_idx], image_label[val_idx])
    return train, val


def _sample_target_firing_rates(path: str, n_neurons: int, seed: int):
    with open(path, "rb") as f:
        firing_rates = np.asarray(pkl.load(f), dtype=np.float32)
    sorted_rates = np.sort(firing_rates)
    percentiles = (np.arange(sorted_rates.shape[0], dtype=np.float32) + 1.0) / sorted_rates.shape[0]
    rd = np.random.RandomState(seed=seed)
    x = rd.uniform(size=n_neurons)
    return np.sort(np.interp(x, percentiles, sorted_rates)).astype(np.float32)


def _collect_recurrent_weights(model):
    rec_refs = {}
    rec_init = {}
    for i, proj in enumerate(model.rsnn.projs):
        key = f"rec.{i}.w"
        w = proj.proj.comm.weight
        rec_refs[key] = w
        rec_init[key] = jnp.asarray(np.asarray(w.value), dtype=jnp.float32)
    return rec_refs, rec_init


def _collect_trainable_vars(model, train_readout: bool, train_input: bool, train_recurrent: bool):
    grad_vars = {}

    if train_readout:
        grad_vars["readout.W"] = model.output_head.W
        grad_vars["readout.b"] = model.output_head.b

    if train_input:
        grad_vars["input.w"] = model.input_layer.input_proj.weight

    if train_recurrent:
        rec_refs, _ = _collect_recurrent_weights(model)
        grad_vars.update(rec_refs)

    if not grad_vars:
        raise ValueError("No trainable variables selected. Enable at least one of readout/input/recurrent.")

    return grad_vars


def _clip_grads(grads: dict, max_norm: float):
    if max_norm <= 0:
        return grads, jnp.asarray(0.0, dtype=jnp.float32)
    sq = jnp.asarray(0.0, dtype=jnp.float32)
    for g in grads.values():
        sq = sq + jnp.sum(g * g)
    gnorm = jnp.sqrt(sq + 1e-12)
    scale = jnp.minimum(1.0, max_norm / (gnorm + 1e-6))
    return {k: g * scale for k, g in grads.items()}, gnorm


def _configure_runtime(args):
    env_platform = os.environ.get("JAX_PLATFORM_NAME", "")
    if args.jax_platform and env_platform and args.jax_platform != env_platform:
        print(
            f"> Warning: --jax_platform={args.jax_platform} is ignored because "
            f"JAX_PLATFORM_NAME={env_platform} is set before import."
        )
    if getattr(args, "xla_preallocate", None) is not None:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if args.xla_preallocate else "false"
    try:
        devices = jax.devices()
        print(f"> JAX_PLATFORM_NAME={os.environ.get('JAX_PLATFORM_NAME', '')}")
        print("> JAX devices:", ", ".join([str(d) for d in devices]))
    except Exception as e:
        print(f"> Warning: failed to query jax devices: {e}")


def _maybe_init_wandb(args):
    if not args.use_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. Install wandb or disable --use_wandb.")

    run_name = args.wandb_name
    if not run_name:
        run_name = f"{args.task}_n{args.n_neurons}_in{args.n_input}_seed{args.seed}"

    config = dict(vars(args))
    config["device"] = args.jax_platform
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        name=run_name,
        mode=args.wandb_mode,
        config=config,
    )
    return run


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std < 1e-12 or b_std < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _chunk_logits(logits_t: np.ndarray, down_sample: int, n_chunks: int):
    kept = int(n_chunks * down_sample)
    if logits_t.shape[0] < kept:
        raise ValueError(
            f"Not enough timesteps for chunking: got {logits_t.shape[0]}, need {kept}"
        )
    logits = logits_t[:kept]
    logits = logits.reshape(n_chunks, down_sample, logits.shape[-1]).mean(axis=1)
    return logits


def _compute_dynamics_metrics(
    *,
    logits_t: np.ndarray,
    spikes_t: np.ndarray,
    voltage_t: np.ndarray,
    extras: dict,
    y_seq: np.ndarray,
    w_seq: np.ndarray,
    down_sample: int,
    dt: float,
    rec_refs: dict,
    rec_init: dict,
):
    eps = 1e-8
    metrics = {}

    # firing rate distribution
    rate_per_neuron_hz = spikes_t.mean(axis=0) * (1000.0 / max(dt, eps))
    metrics["dyn/firing_rate_mean_hz"] = float(np.mean(rate_per_neuron_hz))
    metrics["dyn/firing_rate_std_hz"] = float(np.std(rate_per_neuron_hz))
    metrics["dyn/firing_rate_p10_hz"] = float(np.percentile(rate_per_neuron_hz, 10))
    metrics["dyn/firing_rate_p50_hz"] = float(np.percentile(rate_per_neuron_hz, 50))
    metrics["dyn/firing_rate_p90_hz"] = float(np.percentile(rate_per_neuron_hz, 90))

    # adaptation dynamics
    q = max(1, spikes_t.shape[0] // 5)
    early_rate = float(np.mean(spikes_t[:q]))
    late_rate = float(np.mean(spikes_t[-q:]))
    metrics["dyn/adaptation_early_rate"] = early_rate
    metrics["dyn/adaptation_late_rate"] = late_rate
    metrics["dyn/adaptation_rate_ratio"] = late_rate / (early_rate + eps)
    if "iasc1_t" in extras:
        iasc1 = np.asarray(extras["iasc1_t"])
        metrics["dyn/adaptation_iasc1_mean_abs"] = float(np.mean(np.abs(iasc1)))
        metrics["dyn/adaptation_iasc1_late_early_ratio"] = float(
            np.mean(np.abs(iasc1[-q:])) / (np.mean(np.abs(iasc1[:q])) + eps)
        )
    if "iasc2_t" in extras:
        iasc2 = np.asarray(extras["iasc2_t"])
        metrics["dyn/adaptation_iasc2_mean_abs"] = float(np.mean(np.abs(iasc2)))
        metrics["dyn/adaptation_iasc2_late_early_ratio"] = float(
            np.mean(np.abs(iasc2[-q:])) / (np.mean(np.abs(iasc2[:q])) + eps)
        )

    # PSC temporal profile
    if "epsc_t" in extras and "ipsc_t" in extras:
        epsc = np.asarray(extras["epsc_t"])
        ipsc = np.asarray(extras["ipsc_t"])
        epsc_profile = epsc.mean(axis=-1)
        ipsc_profile = ipsc.mean(axis=-1)
        t = np.arange(epsc_profile.shape[0], dtype=np.float32)
        if epsc_profile.shape[0] >= 2:
            slope_e = float(np.polyfit(t, epsc_profile, 1)[0])
            slope_i = float(np.polyfit(t, ipsc_profile, 1)[0])
        else:
            slope_e, slope_i = 0.0, 0.0
        metrics["dyn/psc_epsc_mean"] = float(np.mean(epsc_profile))
        metrics["dyn/psc_ipsc_mean"] = float(np.mean(ipsc_profile))
        metrics["dyn/psc_epsc_peak"] = float(np.max(epsc_profile))
        metrics["dyn/psc_ipsc_peak"] = float(np.max(ipsc_profile))
        metrics["dyn/psc_epsc_temporal_slope"] = slope_e
        metrics["dyn/psc_ipsc_temporal_slope"] = slope_i
        metrics["dyn/psc_ei_balance"] = float(
            np.mean(epsc_profile - ipsc_profile) / (np.mean(np.abs(epsc_profile) + np.abs(ipsc_profile)) + eps)
        )

    # spike timing structure
    pop_rate = spikes_t.mean(axis=1)
    metrics["dyn/spike_timing_pop_rate_mean"] = float(np.mean(pop_rate))
    metrics["dyn/spike_timing_pop_rate_cv"] = float(np.std(pop_rate) / (np.mean(pop_rate) + eps))
    if pop_rate.shape[0] > 2:
        metrics["dyn/spike_timing_lag1_autocorr"] = _safe_corr(pop_rate[:-1], pop_rate[1:])
    else:
        metrics["dyn/spike_timing_lag1_autocorr"] = 0.0
    metrics["dyn/spike_timing_burst_index"] = float(np.max(pop_rate) / (np.mean(pop_rate) + eps))

    # recurrent activity stability
    dv = np.diff(voltage_t, axis=0) if voltage_t.shape[0] > 1 else np.zeros_like(voltage_t)
    ds = np.diff(spikes_t, axis=0) if spikes_t.shape[0] > 1 else np.zeros_like(spikes_t)
    metrics["dyn/recurrent_stability_dv_rms"] = float(np.sqrt(np.mean(dv * dv)))
    metrics["dyn/recurrent_stability_ds_rms"] = float(np.sqrt(np.mean(ds * ds)))
    metrics["dyn/recurrent_stability_v_abs_max"] = float(np.max(np.abs(voltage_t)))
    metrics["dyn/recurrent_stability_v_energy"] = float(np.mean(voltage_t * voltage_t))
    rec_weight_norm = 0.0
    rec_weight_drift = 0.0
    for key, w in rec_refs.items():
        w_np = np.asarray(w.value, dtype=np.float32)
        rec_weight_norm += float(np.sum(w_np * w_np))
        diff = w_np - np.asarray(rec_init[key], dtype=np.float32)
        rec_weight_drift += float(np.sum(diff * diff))
    metrics["dyn/recurrent_weight_l2"] = float(np.sqrt(rec_weight_norm))
    metrics["dyn/recurrent_weight_drift_l2"] = float(np.sqrt(rec_weight_drift))

    # evidence accumulation behavior
    if logits_t.shape[-1] >= 2:
        evidence = logits_t[:, 1] - logits_t[:, 0]
    else:
        evidence = np.zeros((logits_t.shape[0],), dtype=np.float32)
    evidence_cum = np.cumsum(evidence)
    metrics["dyn/evidence_final"] = float(evidence_cum[-1])
    metrics["dyn/evidence_auc"] = float(np.mean(evidence_cum))
    metrics["dyn/evidence_slope"] = float((evidence_cum[-1] - evidence_cum[0]) / max(1, evidence_cum.shape[0] - 1))

    chunk_logits = np.asarray(
        _chunk_logits(logits_t, down_sample=down_sample, n_chunks=y_seq.shape[0]),
        dtype=np.float32,
    )
    idx = min(1, chunk_logits.shape[-1] - 1)
    shift = chunk_logits - np.max(chunk_logits, axis=-1, keepdims=True)
    exp = np.exp(shift)
    probs = exp / (np.sum(exp, axis=-1, keepdims=True) + eps)
    prob_1 = probs[:, idx]
    active = w_seq > 0.5
    if np.any(active):
        prob_active = np.asarray(prob_1)[active]
        target_active = y_seq[active]
        metrics["dyn/evidence_active_prob_mean"] = float(np.mean(prob_active))
        metrics["dyn/evidence_active_target_mean"] = float(np.mean(target_active))
        pred_active = (prob_active >= 0.5).astype(np.int32)
        metrics["dyn/evidence_active_acc"] = float(np.mean(pred_active == target_active))
    else:
        metrics["dyn/evidence_active_prob_mean"] = float(np.mean(np.asarray(prob_1)))
        metrics["dyn/evidence_active_target_mean"] = 0.0
        metrics["dyn/evidence_active_acc"] = 0.0

    return metrics, rate_per_neuron_hz


def _extract_trial_summary(logits_t, spikes_t, y_seq, w_seq, *, down_sample: int):
    chunk_logits = np.asarray(
        _chunk_logits(logits_t, down_sample=down_sample, n_chunks=y_seq.shape[0]),
        dtype=np.float32,
    )
    shift = chunk_logits - np.max(chunk_logits, axis=-1, keepdims=True)
    exp = np.exp(shift)
    p = exp / (np.sum(exp, axis=-1, keepdims=True) + 1e-8)
    p1 = p[:, min(1, p.shape[-1] - 1)]
    active = w_seq > 0.5
    prob_mean = float(np.mean(p1[active])) if np.any(active) else float(np.mean(p1))
    rate_mean = float(np.mean(spikes_t))
    return prob_mean, rate_mean


def _compute_trial_to_trial_variability(prob_history, rate_history):
    if len(prob_history) < 2 or len(rate_history) < 2:
        return {
            "dyn/trial_variability_prob_var": 0.0,
            "dyn/trial_variability_rate_var": 0.0,
            "dyn/trial_variability_rate_fano": 0.0,
        }
    probs = np.asarray(prob_history, dtype=np.float32)
    mean_rates = np.asarray(rate_history, dtype=np.float32)
    return {
        "dyn/trial_variability_prob_var": float(np.var(probs)),
        "dyn/trial_variability_rate_var": float(np.var(mean_rates)),
        "dyn/trial_variability_rate_fano": float(np.var(mean_rates) / (np.mean(mean_rates) + 1e-8)),
    }


def _rollout_sequence(model, x_seq, *, dt: float, collect_extra: bool = False):
    time_len = int(x_seq.shape[0])
    t_seq = bm.arange(time_len, dtype=bm.float32) * dt

    def _strip_batch(v):
        if getattr(v, "ndim", 0) > 1 and int(v.shape[0]) == 1:
            return v[0]
        return v

    if not collect_extra:
        def _step(x_t, t_t):
            bp.share.save(t=t_t, dt=dt)
            logits, spikes, voltage = model.update(x_t)
            return logits[0], spikes[0], voltage[0]

        logits_t, spikes_t, voltage_t = bm.for_loop(_step, (x_seq, t_seq))
        return logits_t, spikes_t, voltage_t

    has_psc = hasattr(model.rsnn, "epsc_var") and hasattr(model.rsnn, "ipsc_var")
    if has_psc:
        def _step(x_t, t_t):
            bp.share.save(t=t_t, dt=dt)
            logits, spikes, voltage = model.update(x_t)
            n = model.rsnn.neurons
            iasc1 = _strip_batch(n.Iasc1.value)
            iasc2 = _strip_batch(n.Iasc2.value)
            epsc = _strip_batch(model.rsnn.epsc_var.value)
            ipsc = _strip_batch(model.rsnn.ipsc_var.value)
            return logits[0], spikes[0], voltage[0], iasc1, iasc2, epsc, ipsc

        logits_t, spikes_t, voltage_t, iasc1_t, iasc2_t, epsc_t, ipsc_t = bm.for_loop(_step, (x_seq, t_seq))
        extras = {
            "iasc1_t": iasc1_t,
            "iasc2_t": iasc2_t,
            "epsc_t": epsc_t,
            "ipsc_t": ipsc_t,
        }
    else:
        def _step(x_t, t_t):
            bp.share.save(t=t_t, dt=dt)
            logits, spikes, voltage = model.update(x_t)
            n = model.rsnn.neurons
            iasc1 = _strip_batch(n.Iasc1.value)
            iasc2 = _strip_batch(n.Iasc2.value)
            return logits[0], spikes[0], voltage[0], iasc1, iasc2

        logits_t, spikes_t, voltage_t, iasc1_t, iasc2_t = bm.for_loop(_step, (x_seq, t_seq))
        extras = {
            "iasc1_t": iasc1_t,
            "iasc2_t": iasc2_t,
        }

    return logits_t, spikes_t, voltage_t, extras


def _compute_losses(
    model,
    logits_t,
    spikes_t,
    voltage_t,
    labels,
    weights,
    *,
    down_sample: int,
    target_firing_rates,
    rate_cost: float,
    voltage_cost: float,
    recurrent_weight_regularization: float,
    rec_refs: dict,
    rec_init: dict,
):
    n_chunks = int(labels.shape[0])
    kept = n_chunks * down_sample
    logits_t = logits_t[:kept]
    logits_chunks = bm.reshape(logits_t, (n_chunks, down_sample, logits_t.shape[-1]))
    logits_chunks = bm.mean(logits_chunks, axis=1)

    log_probs = jax.nn.log_softmax(logits_chunks, axis=-1)
    picked = jnp.take_along_axis(log_probs, labels[:, None], axis=-1)[:, 0]
    denom = jnp.maximum(jnp.sum(weights), 1e-6)
    cls_loss = -jnp.sum(picked * weights) / denom

    pred = jnp.argmax(logits_chunks, axis=-1)
    acc = jnp.sum((pred == labels) * weights) / denom

    mean_rate = jnp.mean(spikes_t)

    rate_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if rate_cost > 0:
        rate_vec = bp_models.compute_spike_rate_distribution_loss(
            spikes_t[None, ...], target_firing_rates
        )
        rate_loss = rate_cost * jnp.mean(rate_vec)

    voltage_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if voltage_cost > 0:
        v_offset = model.rsnn.neurons.e_l
        v_scale = jnp.maximum(model.rsnn.neurons.v_th - model.rsnn.neurons.e_l, 1e-3)
        v_loss = bp_models.voltage_loss(voltage_t[None, ...], v_offset, v_scale)
        voltage_loss = voltage_cost * jnp.mean(v_loss)

    rec_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if recurrent_weight_regularization > 0 and len(rec_refs) > 0:
        rec_pen = jnp.asarray(0.0, dtype=jnp.float32)
        for key, w in rec_refs.items():
            diff = w - rec_init[key]
            rec_pen = rec_pen + jnp.sum(diff * diff)
        rec_loss = recurrent_weight_regularization * rec_pen

    l2_loss = model.readout_l2()
    total = cls_loss + rate_loss + voltage_loss + rec_loss + l2_loss

    aux = {
        "cls_loss": cls_loss,
        "rate_loss": rate_loss,
        "voltage_loss": voltage_loss,
        "rec_loss": rec_loss,
        "l2_loss": l2_loss,
        "acc": acc,
        "mean_rate": mean_rate,
    }
    return total, aux


def _save_ckpt(path, grad_vars, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "trainables": {k: np.asarray(v.value, dtype=np.float32) for k, v in grad_vars.items()},
        "meta": meta,
    }
    with open(path, "wb") as f:
        pkl.dump(payload, f)


def _build_model_and_network(args, *, n_output: int, output_mode: str):
    if args.caching:
        load_fn = load_sparse.cached_load_billeh
        input_population, network, bkg_weights = load_fn(
            n_input=args.n_input,
            n_neurons=args.n_neurons,
            core_only=args.core_only,
            data_dir=args.data_dir,
            seed=args.seed,
            connected_selection=args.connected_selection,
            n_output=2,
            neurons_per_output=args.neurons_per_output,
            use_rand_ini_w=args.use_rand_ini_w,
            scale_w_e=args.scale_w_e,
        )
    else:
        load_fn = load_sparse.load_billeh
        input_population, network, bkg_weights = load_fn(
            n_input=args.n_input,
            n_neurons=args.n_neurons,
            core_only=args.core_only,
            data_dir=args.data_dir,
            seed=args.seed,
            connected_selection=args.connected_selection,
            n_output=2,
            neurons_per_output=args.neurons_per_output,
            use_rand_ini_w=args.use_rand_ini_w,
            use_dale_law=args.use_dale_law,
            use_rand_connectivity=args.use_rand_connectivity,
            use_uniform_neuron_type=args.use_uniform_neuron_type,
            scale_w_e=args.scale_w_e,
            localized_readout=args.localized_readout,
        )

    net = classification_tools._normalize_network(network)

    node_params = NodeParams.from_network_node_params(
        net.node_params,
        net.node_type_ids,
        dt=args.dt,
    )
    syn_params = SynapseParams.from_network_synapses(
        net.synapses,
        n_nodes=net.n_nodes,
        n_edges=net.n_edges,
        n_receptors=node_params.n_receptors,
        max_delay=args.max_delay,
        dt=args.dt,
    )

    input_population_norm = dict(input_population)
    if "delays" not in input_population_norm:
        n_input_edges = int(np.asarray(input_population_norm["weights"]).shape[0])
        input_population_norm["delays"] = np.ones(n_input_edges, dtype=np.float32)
    input_params = InputParams.from_input_node_bkg(
        input_population_norm,
        node_params,
        np.asarray(bkg_weights, dtype=np.float32),
    )
    input_csr = to_brainpy_csr(
        input_params, split_receptor=False, split_conn=False
    )
    input_csr.eliminate_zeros()
    in_conn = bp.conn.SparseMatConn(input_csr != 0)
    in_weight = input_csr.data
    bkg = np.asarray(input_params.bkg_weights, dtype=np.float32).reshape(
        node_params.n_nodes, node_params.n_receptors
    )

    with bm.environment(mode=bm.TrainingMode(batch_size=1)):
        rsnn = bp_models.BillehColumnTFAligned(
            node_params=node_params,
            syn_params=syn_params,
            use_dale_law=args.use_dale_law,
            default_input_to_receptor=True,
            spk_reset="hard",
            delay_filter=None,
            keep_all_delays=True,
            laminar_indices=net.laminar_indices if net.laminar_indices else {"L23e": np.arange(net.n_nodes)},
            track_epsc_ipsc=args.track_epsc_ipsc,
            track_psc_trace=args.track_psc_trace,
        )
        input_layer = classification_tools._InputLayerCompat(
            conn=in_conn,
            weight=in_weight,
            tau_syn=node_params.tau_syn,
            use_dale_law=args.use_dale_law,
            bkg_weights=bkg,
            use_decoded_noise=args.use_decoded_noise,
            noise_data=None,
        )
        model = classification_tools.BillehClassificationModel(
            rsnn=rsnn,
            input_layer=input_layer,
            network_meta=net,
            seq_len=args.seq_len,
            down_sample=args.down_sample,
            n_output=n_output,
            output_mode=output_mode,
            neuron_output=False,
            lRout_pop=args.lRout_pop,
            dampening_factor=args.dampening_factor,
            full_output=False,
            L2_factor=args.l2_factor,
            batch_size=1,
        )

    return model, network


def main(args):
    _configure_runtime(args)
    t0 = time.time()
    os.makedirs(args.results_dir, exist_ok=True)
    wandb_run = None

    n_output = 2
    output_mode = "garrett"

    if args.train_vcd_npz_path is None:
        raise ValueError("VCD-only mode requires --train_vcd_npz_path.")

    train_x, train_y, train_w, train_image_label = _load_vcd_npz_arrays(args.train_vcd_npz_path)
    if args.val_vcd_npz_path:
        val_x, val_y, val_w, val_image_label = _load_vcd_npz_arrays(args.val_vcd_npz_path)
    else:
        (train_x, train_y, train_w, train_image_label), (val_x, val_y, val_w, val_image_label) = _split_vcd_arrays(
            train_x,
            train_y,
            train_w,
            train_image_label,
            val_ratio=args.vcd_val_split,
            seed=args.seed,
        )

    inferred_seq_len = int(train_x.shape[1])
    inferred_input_dim = int(train_x.shape[2])
    inferred_chunks = int(train_y.shape[1])
    if inferred_seq_len % inferred_chunks != 0:
        raise ValueError(
            f"VCD npz has incompatible shapes: seq_len={inferred_seq_len}, chunks={inferred_chunks}"
        )
    inferred_down_sample = inferred_seq_len // inferred_chunks

    if args.auto_seq_from_data and args.seq_len != inferred_seq_len:
        print(f"> Override seq_len from data: {args.seq_len} -> {inferred_seq_len}")
        args.seq_len = inferred_seq_len
    if args.seq_len != inferred_seq_len:
        raise ValueError(
            f"Configured seq_len={args.seq_len} but data seq_len={inferred_seq_len}. "
            "Enable --auto_seq_from_data or fix args."
        )

    if args.auto_down_sample_from_data and args.down_sample != inferred_down_sample:
        print(f"> Override down_sample from data: {args.down_sample} -> {inferred_down_sample}")
        args.down_sample = inferred_down_sample
    if args.down_sample != inferred_down_sample:
        raise ValueError(
            f"Configured down_sample={args.down_sample} but data implies {inferred_down_sample}. "
            "Enable --auto_down_sample_from_data or fix args."
        )

    if args.auto_n_input_from_data and args.n_input != inferred_input_dim:
        print(f"> Override n_input from data: {args.n_input} -> {inferred_input_dim}")
        args.n_input = inferred_input_dim

    print(
        f"> Loaded VCD npz train/val: {train_x.shape[0]} / {val_x.shape[0]} "
        f"(seq={train_x.shape[1]}, input={train_x.shape[2]}, chunks={train_y.shape[1]})"
    )

    train_gen = VCDNPZSequenceGenerator(
        train_x,
        train_y,
        train_w,
        n_input=args.n_input,
        down_sample=args.down_sample,
        seed=args.seed,
        batch_size=1,
        shuffle=True,
    )
    val_gen = VCDNPZSequenceGenerator(
        val_x,
        val_y,
        val_w,
        n_input=args.n_input,
        down_sample=args.down_sample,
        seed=args.seed + 1,
        batch_size=1,
        shuffle=False,
    )

    if args.seq_len % args.down_sample != 0:
        raise ValueError("seq_len must be divisible by down_sample.")

    model, network = _build_model_and_network(args, n_output=n_output, output_mode=output_mode)
    print(f"> Network nodes: {int(network['n_nodes'])}")

    if args.rate_cost > 0:
        if args.firing_rates_path is not None and os.path.exists(args.firing_rates_path):
            target_firing_rates = jnp.asarray(
                _sample_target_firing_rates(args.firing_rates_path, int(network["n_nodes"]), args.seed)
            )
            print(f"> Target firing-rate profile loaded: {tuple(target_firing_rates.shape)}")
        else:
            target_firing_rates = jnp.zeros((int(network["n_nodes"]),), dtype=jnp.float32)
            print("> Warning: firing-rates file not found, disabling rate_cost.")
            args.rate_cost = 0.0
    else:
        target_firing_rates = jnp.zeros((int(network["n_nodes"]),), dtype=jnp.float32)

    rec_refs, rec_init = _collect_recurrent_weights(model)
    grad_vars = _collect_trainable_vars(
        model,
        train_readout=args.train_readout,
        train_input=args.train_input,
        train_recurrent=args.train_recurrent,
    )
    print(f"> Number of trainable tensors: {len(grad_vars)}")
    for k, v in grad_vars.items():
        print(f"  - {k}: {tuple(v.shape)}")

    optimizer = bst.optim.Adam(lr=args.learning_rate)
    optimizer.register_trainable_weights(grad_vars)

    def loss_fn_train(x_seq, y_seq, w_seq):
        logits_t, spikes_t, voltage_t, extras_t = _rollout_sequence(
            model, x_seq, dt=args.dt, collect_extra=True
        )

        total, aux = _compute_losses(
            model,
            logits_t,
            spikes_t,
            voltage_t,
            y_seq,
            w_seq,
            down_sample=args.down_sample,
            target_firing_rates=target_firing_rates,
            rate_cost=args.rate_cost,
            voltage_cost=args.voltage_cost,
            recurrent_weight_regularization=args.recurrent_weight_regularization,
            rec_refs=rec_refs,
            rec_init=rec_init,
        )
        return total, (aux, logits_t, spikes_t, voltage_t, extras_t)

    def loss_fn_eval(x_seq, y_seq, w_seq):
        logits_t, spikes_t, voltage_t = _rollout_sequence(model, x_seq, dt=args.dt)
        return _compute_losses(
            model,
            logits_t,
            spikes_t,
            voltage_t,
            y_seq,
            w_seq,
            down_sample=args.down_sample,
            target_firing_rates=target_firing_rates,
            rate_cost=args.rate_cost,
            voltage_cost=args.voltage_cost,
            recurrent_weight_regularization=args.recurrent_weight_regularization,
            rec_refs=rec_refs,
            rec_init=rec_init,
        )

    grad_fn = bm.grad(loss_fn_train, grad_vars=grad_vars, return_value=True, has_aux=True)
    if args.jit_train_step:
        grad_fn = bm.jit(grad_fn)
        loss_fn_eval = bm.jit(loss_fn_eval)

    wandb_run = _maybe_init_wandb(args)
    if wandb_run is not None:
        wandb.config.update(
            {
                "n_nodes": int(network["n_nodes"]),
                "n_trainables": len(grad_vars),
                "n_recurrent_projections": len(model.rsnn.projs),
                "jax_devices": [str(d) for d in jax.devices()],
            },
            allow_val_change=True,
        )
        if args.wandb_watch:
            try:
                wandb.watch(model, log="gradients", log_freq=args.wandb_watch_log_freq)
            except Exception as e:
                print(f"> Warning: wandb.watch(model) failed: {e}")

    @bm.jit
    def reset_model_state():
        model._reset_inplace()
        return jnp.asarray(0, dtype=jnp.int32)

    best_val = float("inf")
    history = []
    global_step = 0
    tv_prob_history = deque(maxlen=max(2, int(args.trial_variability_window)))
    tv_rate_history = deque(maxlen=max(2, int(args.trial_variability_window)))

    for epoch in range(1, args.n_epochs + 1):
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        train_metrics = {
            "loss": zero,
            "cls": zero,
            "rate": zero,
            "volt": zero,
            "rec": zero,
            "l2": zero,
            "acc": zero,
            "mean_rate": zero,
            "grad_norm": zero,
        }

        if args.continuing_state:
            _ = reset_model_state()

        for step in range(1, args.steps_per_epoch + 1):
            if not args.continuing_state:
                _ = reset_model_state()

            x, y, w = train_gen.sample()
            x_seq = bm.asarray(x[0])
            y_seq = jnp.asarray(y[0], dtype=jnp.int32)
            w_seq = jnp.asarray(w[0], dtype=jnp.float32)

            grads, loss, aux_pack = grad_fn(x_seq, y_seq, w_seq)
            aux, logits_t_cache, spikes_t_cache, voltage_t_cache, extras_t_cache = aux_pack
            grads, gnorm = _clip_grads(grads, args.grad_clip_norm)
            optimizer.update(grads)
            global_step += 1

            train_metrics["loss"] = train_metrics["loss"] + loss
            train_metrics["cls"] = train_metrics["cls"] + aux["cls_loss"]
            train_metrics["rate"] = train_metrics["rate"] + aux["rate_loss"]
            train_metrics["volt"] = train_metrics["volt"] + aux["voltage_loss"]
            train_metrics["rec"] = train_metrics["rec"] + aux["rec_loss"]
            train_metrics["l2"] = train_metrics["l2"] + aux["l2_loss"]
            train_metrics["acc"] = train_metrics["acc"] + aux["acc"]
            train_metrics["mean_rate"] = train_metrics["mean_rate"] + aux["mean_rate"]
            train_metrics["grad_norm"] = train_metrics["grad_norm"] + gnorm

            if wandb_run is not None and (global_step % args.wandb_log_interval == 0):
                log_data = jax.device_get(
                    {
                        "train/loss": loss,
                        "train/cls_loss": aux["cls_loss"],
                        "train/rate_loss": aux["rate_loss"],
                        "train/voltage_loss": aux["voltage_loss"],
                        "train/rec_loss": aux["rec_loss"],
                        "train/l2_loss": aux["l2_loss"],
                        "train/acc": aux["acc"],
                        "train/mean_rate": aux["mean_rate"],
                        "train/grad_norm": gnorm,
                    }
                )
                host_log_data = {k: float(v) for k, v in log_data.items()}
                host_log_data["train/epoch"] = int(epoch)
                host_log_data["train/step_in_epoch"] = int(step)

                if args.log_dynamics:
                    cached = {
                        "logits_t": logits_t_cache,
                        "spikes_t": spikes_t_cache,
                        "voltage_t": voltage_t_cache,
                        "y_seq": y_seq,
                        "w_seq": w_seq,
                    }
                    if isinstance(extras_t_cache, dict):
                        for k, v in extras_t_cache.items():
                            cached[k] = v
                    cached = jax.device_get(cached)
                    extras_np = {}
                    for k in ("iasc1_t", "iasc2_t", "epsc_t", "ipsc_t"):
                        if k in cached:
                            extras_np[k] = np.asarray(cached[k], dtype=np.float32)

                    dyn_metrics, rate_hist = _compute_dynamics_metrics(
                        logits_t=np.asarray(cached["logits_t"], dtype=np.float32),
                        spikes_t=np.asarray(cached["spikes_t"], dtype=np.float32),
                        voltage_t=np.asarray(cached["voltage_t"], dtype=np.float32),
                        extras=extras_np,
                        y_seq=np.asarray(cached["y_seq"], dtype=np.int32),
                        w_seq=np.asarray(cached["w_seq"], dtype=np.float32),
                        down_sample=args.down_sample,
                        dt=args.dt,
                        rec_refs=rec_refs,
                        rec_init=rec_init,
                    )
                    host_log_data.update(dyn_metrics)

                    prob_mean, rate_mean = _extract_trial_summary(
                        np.asarray(cached["logits_t"], dtype=np.float32),
                        np.asarray(cached["spikes_t"], dtype=np.float32),
                        np.asarray(cached["y_seq"], dtype=np.int32),
                        np.asarray(cached["w_seq"], dtype=np.float32),
                        down_sample=args.down_sample,
                    )
                    tv_prob_history.append(prob_mean)
                    tv_rate_history.append(rate_mean)
                    host_log_data["dyn/trial_prob_mean"] = prob_mean
                    host_log_data["dyn/trial_rate_mean"] = rate_mean
                    host_log_data.update(
                        _compute_trial_to_trial_variability(tv_prob_history, tv_rate_history)
                    )

                    if args.log_firing_rate_hist and rate_hist.size > 0:
                        host_log_data["dyn/firing_rate_hist"] = wandb.Histogram(rate_hist)

                wandb.log(host_log_data, step=global_step)

            if step % args.log_every == 0 or step == args.steps_per_epoch:
                d = float(step)
                train_avg = jax.device_get(
                    {
                        "loss": train_metrics["loss"] / d,
                        "acc": train_metrics["acc"] / d,
                        "rate": train_metrics["mean_rate"] / d,
                        "g_norm": train_metrics["grad_norm"] / d,
                    }
                )
                print(
                    f"[Epoch {epoch:03d}] step {step:04d}/{args.steps_per_epoch} "
                    f"loss={float(train_avg['loss']):.6f} "
                    f"acc={float(train_avg['acc']):.4f} "
                    f"rate={float(train_avg['rate']):.6f} "
                    f"g_norm={float(train_avg['g_norm']):.6f}"
                )

        for k in train_metrics:
            train_metrics[k] /= args.steps_per_epoch

        val_metrics = {
            "loss": zero,
            "cls": zero,
            "rate": zero,
            "volt": zero,
            "rec": zero,
            "l2": zero,
            "acc": zero,
            "mean_rate": zero,
        }

        if args.continuing_state:
            _ = reset_model_state()

        for _ in range(args.val_steps):
            if not args.continuing_state:
                _ = reset_model_state()

            x, y, w = val_gen.sample()
            x_seq = bm.asarray(x[0])
            y_seq = jnp.asarray(y[0], dtype=jnp.int32)
            w_seq = jnp.asarray(w[0], dtype=jnp.float32)

            loss, aux = loss_fn_eval(x_seq, y_seq, w_seq)
            val_metrics["loss"] = val_metrics["loss"] + loss
            val_metrics["cls"] = val_metrics["cls"] + aux["cls_loss"]
            val_metrics["rate"] = val_metrics["rate"] + aux["rate_loss"]
            val_metrics["volt"] = val_metrics["volt"] + aux["voltage_loss"]
            val_metrics["rec"] = val_metrics["rec"] + aux["rec_loss"]
            val_metrics["l2"] = val_metrics["l2"] + aux["l2_loss"]
            val_metrics["acc"] = val_metrics["acc"] + aux["acc"]
            val_metrics["mean_rate"] = val_metrics["mean_rate"] + aux["mean_rate"]

        for k in val_metrics:
            val_metrics[k] /= args.val_steps

        train_metrics_host = {k: float(v) for k, v in jax.device_get(train_metrics).items()}
        val_metrics_host = {k: float(v) for k, v in jax.device_get(val_metrics).items()}

        rec = {
            "epoch": epoch,
            "train_loss": train_metrics_host["loss"],
            "train_cls_loss": train_metrics_host["cls"],
            "train_rate_loss": train_metrics_host["rate"],
            "train_voltage_loss": train_metrics_host["volt"],
            "train_rec_loss": train_metrics_host["rec"],
            "train_l2_loss": train_metrics_host["l2"],
            "train_acc": train_metrics_host["acc"],
            "train_mean_rate": train_metrics_host["mean_rate"],
            "train_grad_norm": train_metrics_host["grad_norm"],
            "val_loss": val_metrics_host["loss"],
            "val_cls_loss": val_metrics_host["cls"],
            "val_rate_loss": val_metrics_host["rate"],
            "val_voltage_loss": val_metrics_host["volt"],
            "val_rec_loss": val_metrics_host["rec"],
            "val_l2_loss": val_metrics_host["l2"],
            "val_acc": val_metrics_host["acc"],
            "val_mean_rate": val_metrics_host["mean_rate"],
            "elapsed_min": (time.time() - t0) / 60.0,
        }
        history.append(rec)

        print(
            f"[Epoch {epoch:03d}] "
            f"val_loss={val_metrics_host['loss']:.6f} "
            f"val_acc={val_metrics_host['acc']:.4f} "
            f"val_rate={val_metrics_host['mean_rate']:.6f}"
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_metrics_host["loss"],
                    "train/epoch_acc": train_metrics_host["acc"],
                    "train/epoch_mean_rate": train_metrics_host["mean_rate"],
                    "train/epoch_grad_norm": train_metrics_host["grad_norm"],
                    "val/loss": val_metrics_host["loss"],
                    "val/acc": val_metrics_host["acc"],
                    "val/mean_rate": val_metrics_host["mean_rate"],
                    "val/cls_loss": val_metrics_host["cls"],
                    "val/rate_loss": val_metrics_host["rate"],
                    "val/voltage_loss": val_metrics_host["volt"],
                    "val/rec_loss": val_metrics_host["rec"],
                    "val/l2_loss": val_metrics_host["l2"],
                    "time/elapsed_min": (time.time() - t0) / 60.0,
                },
                step=global_step,
            )

        last_ckpt = os.path.join(args.results_dir, "last_trainables.pkl")
        _save_ckpt(last_ckpt, grad_vars, meta={"epoch": epoch, "args": vars(args)})

        if val_metrics_host["loss"] < best_val:
            best_val = val_metrics_host["loss"]
            best_ckpt = os.path.join(args.results_dir, "best_trainables.pkl")
            _save_ckpt(best_ckpt, grad_vars, meta={"epoch": epoch, "args": vars(args)})
            print(f"> Saved best checkpoint to {best_ckpt}")

        with open(os.path.join(args.results_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"> Training finished in {(time.time() - t0) / 60.0:.2f} min")
    print(f"> Best validation loss: {best_val:.6f}")
    if wandb_run is not None:
        wandb.summary["best_val_loss"] = float(best_val)
        wandb.finish()


def build_parser():
    parser = argparse.ArgumentParser(
        description="BrainPy/brainstate VCD-task training (single-machine baseline)"
    )
    parser.add_argument("--data_dir", type=str, default="/home/wanghz/EC_V1/data/GLIF_V1_network")
    parser.add_argument("--results_dir", type=str, default="./results_bp_vcd")
    parser.add_argument("--task", type=str, default="vcd_npz", choices=["vcd_npz"])

    # network loading
    parser.add_argument("--n_input", type=int, default=200)
    parser.add_argument("--n_neurons", type=int, default=1000)
    parser.add_argument("--core_only", action="store_true", default=False)
    parser.add_argument("--connected_selection", action="store_true", default=True)
    parser.add_argument("--neurons_per_output", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--caching", action="store_true", default=False)
    parser.add_argument("--use_rand_ini_w", action="store_true", default=False)
    parser.add_argument("--use_dale_law", action="store_true", default=True)
    parser.add_argument("--use_rand_connectivity", action="store_true", default=False)
    parser.add_argument("--use_uniform_neuron_type", action="store_true", default=False)
    parser.add_argument("--localized_readout", action="store_true", default=True)
    parser.add_argument("--scale_w_e", type=float, default=-1.0)

    # task / data
    parser.add_argument("--train_vcd_npz_path", type=str, default=None)
    parser.add_argument("--val_vcd_npz_path", type=str, default=None)
    parser.add_argument("--vcd_val_split", type=float, default=0.2)
    parser.add_argument("--auto_n_input_from_data", dest="auto_n_input_from_data", action="store_true")
    parser.add_argument("--no_auto_n_input_from_data", dest="auto_n_input_from_data", action="store_false")
    parser.add_argument("--auto_seq_from_data", dest="auto_seq_from_data", action="store_true")
    parser.add_argument("--no_auto_seq_from_data", dest="auto_seq_from_data", action="store_false")
    parser.add_argument("--auto_down_sample_from_data", dest="auto_down_sample_from_data", action="store_true")
    parser.add_argument("--no_auto_down_sample_from_data", dest="auto_down_sample_from_data", action="store_false")
    parser.add_argument("--firing_rates_path", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=600)

    # optimization and losses
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--val_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--jit_train_step", dest="jit_train_step", action="store_true")
    parser.add_argument("--no_jit_train_step", dest="jit_train_step", action="store_false")

    parser.add_argument("--rate_cost", type=float, default=0.1)
    parser.add_argument("--voltage_cost", type=float, default=1e-5)
    parser.add_argument("--recurrent_weight_regularization", type=float, default=0.0)
    parser.add_argument("--l2_factor", type=float, default=0.0)

    # runtime
    parser.add_argument("--down_sample", type=int, default=50)
    parser.add_argument("--max_delay", type=int, default=5)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--dampening_factor", type=float, default=0.5)
    parser.add_argument("--use_decoded_noise", action="store_true", default=False)
    parser.add_argument("--decoded_noise_path", type=str, default=None)
    parser.add_argument("--lRout_pop", type=str, default="all")
    parser.add_argument("--jax_platform", type=str, default="gpu", choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--xla_preallocate", dest="xla_preallocate", action="store_true")
    parser.add_argument("--no_xla_preallocate", dest="xla_preallocate", action="store_false")
    parser.add_argument("--track_epsc_ipsc", dest="track_epsc_ipsc", action="store_true")
    parser.add_argument("--no_track_epsc_ipsc", dest="track_epsc_ipsc", action="store_false")
    parser.add_argument("--track_psc_trace", dest="track_psc_trace", action="store_true")
    parser.add_argument("--no_track_psc_trace", dest="track_psc_trace", action="store_false")

    # wandb + dynamics monitoring
    parser.add_argument("--use_wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb_project", type=str, default="EC_V1_alignment")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_watch", dest="wandb_watch", action="store_true")
    parser.add_argument("--no_wandb_watch", dest="wandb_watch", action="store_false")
    parser.add_argument("--wandb_watch_log_freq", type=int, default=100)
    parser.add_argument("--wandb_log_interval", type=int, default=10)
    parser.add_argument("--log_dynamics", dest="log_dynamics", action="store_true")
    parser.add_argument("--no_log_dynamics", dest="log_dynamics", action="store_false")
    parser.add_argument("--log_firing_rate_hist", dest="log_firing_rate_hist", action="store_true")
    parser.add_argument("--no_log_firing_rate_hist", dest="log_firing_rate_hist", action="store_false")
    parser.add_argument("--trial_variability_window", type=int, default=64)

    # switches that mirror TF training knobs
    parser.add_argument("--train_readout", dest="train_readout", action="store_true")
    parser.add_argument("--no_train_readout", dest="train_readout", action="store_false")
    parser.add_argument("--train_input", dest="train_input", action="store_true")
    parser.add_argument("--no_train_input", dest="train_input", action="store_false")
    parser.add_argument("--train_recurrent", dest="train_recurrent", action="store_true")
    parser.add_argument("--no_train_recurrent", dest="train_recurrent", action="store_false")
    parser.add_argument("--continuing_state", dest="continuing_state", action="store_true")
    parser.add_argument("--no_continuing_state", dest="continuing_state", action="store_false")

    parser.set_defaults(
        train_readout=True,
        train_input=True,
        train_recurrent=True,
        continuing_state=True,
        jit_train_step=True,
        auto_n_input_from_data=True,
        auto_seq_from_data=True,
        auto_down_sample_from_data=True,
        xla_preallocate=False,
        track_epsc_ipsc=True,
        track_psc_trace=False,
        use_wandb=False,
        wandb_watch=False,
        log_dynamics=True,
        log_firing_rate_hist=True,
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.decoded_noise_path is None:
        args.decoded_noise_path = os.path.join(args.data_dir, "additive_noise.mat")

    main(args)
