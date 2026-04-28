from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Literal, Mapping, Optional

import brainstate as bs
import jax
import jax.numpy as jnp
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))



ECWeightMode = Literal["gate", "binary"]


def _as_f32(x: Any) -> jnp.ndarray:
    return jnp.asarray(x, dtype=jnp.float32)


def _as_i32(x: Any) -> jnp.ndarray:
    return jnp.asarray(x, dtype=jnp.int32)


def _repeat_to_length(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 0:
        return np.full((n,), x, dtype=np.float32)
    x = x.reshape(-1)
    if x.shape[0] == n:
        return x.astype(np.float32)
    if x.shape[0] == 1:
        return np.full((n,), x[0], dtype=np.float32)
    raise ValueError(f"Cannot broadcast shape {x.shape} to length {n}.")


def _maybe_get_fixed_sign(obj: Any) -> Optional[np.ndarray]:
    sign_obj = getattr(obj, "sign", None)
    if sign_obj is None:
        return None
    sign_val = getattr(sign_obj, "value", sign_obj)
    return np.asarray(sign_val)


def _extract_sign_amplitude(
    weight: np.ndarray,
    *,
    signed_masks: bool,
    fixed_sign: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weight, dtype=np.float32)
    if fixed_sign is not None:
        fs = np.asarray(fixed_sign)
        sign_fixed = np.where(fs > 0, 1.0, -1.0).astype(np.float32)
        amp = np.where(sign_fixed > 0.0, np.maximum(w, 0.0), np.maximum(-w, 0.0)).astype(
            np.float32
        )
        if signed_masks:
            return sign_fixed, amp

    if signed_masks:
        sign = np.where(w >= 0.0, 1.0, -1.0).astype(np.float32)
        amp = np.abs(w).astype(np.float32)
    else:
        sign = np.ones_like(w, dtype=np.float32)
        amp = np.abs(w).astype(np.float32)
    return sign, amp


def _build_post_ids(indptr: np.ndarray) -> np.ndarray:
    indptr = np.asarray(indptr, dtype=np.int32)
    counts = np.diff(indptr)
    post = np.arange(counts.shape[0], dtype=np.int32)
    return np.repeat(post, counts).astype(np.int32)


def _segment_sum_last_axis_single(
    values_be: jnp.ndarray,
    segment_ids: jnp.ndarray,
    n_segments: int,
) -> jnp.ndarray:
    def _one_row(v):
        return jax.ops.segment_sum(v, segment_ids, n_segments)

    return jax.vmap(_one_row)(values_be)


def _csr_mv_single(
    x_bpre: jnp.ndarray,
    w_e: jnp.ndarray,
    edge_pre_ids: jnp.ndarray,
    edge_post_ids: jnp.ndarray,
    post_num: int,
) -> jnp.ndarray:
    x_edges = jnp.take(x_bpre, edge_pre_ids, axis=-1)
    contrib = x_edges * w_e[None, :]
    return _segment_sum_last_axis_single(contrib, edge_post_ids, post_num)


def _infer_population_size(pop_masks: Mapping[str, jnp.ndarray]) -> int:
    for v in pop_masks.values():
        vv = jnp.asarray(v)
        if vv.ndim >= 1:
            return int(vv.shape[0])
    return 1


def _mask_with_broadcast(
    pop_masks: Mapping[str, jnp.ndarray],
    key: str,
    shape: tuple[int, ...],
    pop_size: int,
) -> jnp.ndarray:
    if key not in pop_masks:
        return jnp.ones((pop_size, *shape), dtype=jnp.float32)
    m = jnp.asarray(pop_masks[key], dtype=jnp.float32)
    if m.ndim == len(shape):
        m = jnp.broadcast_to(m[None, ...], (pop_size, *shape))
    if m.shape[0] != pop_size:
        raise ValueError(
            f"Mask '{key}' population size mismatch: got {m.shape[0]}, expected {pop_size}"
        )
    if m.shape[1:] != shape:
        raise ValueError(f"Mask '{key}' shape mismatch: got {m.shape[1:]}, expected {shape}")
    return m


@dataclass(frozen=True)
class MaskSpec:
    sign: jnp.ndarray
    amplitude: jnp.ndarray
    base_weight: jnp.ndarray


@dataclass(frozen=True)
class RecurrentProjCore:
    key: str
    indices: jnp.ndarray
    indptr: jnp.ndarray
    post_ids: jnp.ndarray
    pre_num: int
    post_num: int
    sign: jnp.ndarray
    amplitude: jnp.ndarray
    base_weight: jnp.ndarray
    delay_steps: int
    tau_decay: jnp.ndarray
    psc_initial: jnp.ndarray


@dataclass(frozen=True)
class FunctionalCore:
    n_nodes: int
    n_receptors: int
    dt: float
    down_sample: int

    v_reset: jnp.ndarray
    e_l: jnp.ndarray
    v_th: jnp.ndarray
    c_m: jnp.ndarray
    tau: jnp.ndarray
    k: jnp.ndarray
    asc_amps: jnp.ndarray
    t_ref: jnp.ndarray

    input_indices: jnp.ndarray
    input_indptr: jnp.ndarray
    input_post_ids: jnp.ndarray
    input_pre_num: int
    input_post_num: int
    input_sign: jnp.ndarray
    input_amplitude: jnp.ndarray
    input_base_weight: jnp.ndarray
    bkg_weights: jnp.ndarray
    bkg_mean_scale: float
    input_x_scale: float

    receptor_tau_decay: jnp.ndarray
    receptor_psc_initial: jnp.ndarray

    recurrent_projs: tuple[RecurrentProjCore, ...]

    readout_sign: jnp.ndarray
    readout_amplitude: jnp.ndarray
    readout_base_weight: jnp.ndarray
    readout_bias: jnp.ndarray
    output_indices: jnp.ndarray
    readout_neuron_indices: Optional[jnp.ndarray]

    exc_indices: jnp.ndarray
    l2_factor: float

    mask_specs: dict[str, MaskSpec]
    rec_keys: tuple[str, ...]

    weight_mode: ECWeightMode = "binary"
    binary_weight_scale: float = 1.0
    binary_input_scale: float = 1.0
    binary_recurrent_scale: float = 1.0
    binary_readout_scale: float = 1.0


@dataclass(frozen=True)
class EffectiveWeights:
    input_w: jnp.ndarray
    rec_w: tuple[jnp.ndarray, ...]
    readout_w: jnp.ndarray


@dataclass(frozen=True)
class RolloutResult:
    logits_t: jnp.ndarray
    spikes_t: jnp.ndarray
    voltage_t: jnp.ndarray
    extras: dict[str, jnp.ndarray]


def build_functional_core(
    model,
    network_meta: dict[str, Any],
    *,
    dt: float,
    down_sample: int,
    signed_masks: bool = True,
    weight_mode: ECWeightMode = "binary",
    binary_weight_scale: float = 1.0,
    binary_input_scale: float | None = None,
    binary_recurrent_scale: float | None = None,
    binary_readout_scale: float | None = None,
    bkg_mean_scale: float = 0.1,
    input_x_scale: float = 1.0,
) -> FunctionalCore:
    if weight_mode not in ("gate", "binary"):
        raise ValueError(
            f"Unsupported weight_mode='{weight_mode}', expected 'gate' or 'binary'."
        )
    if not np.isfinite(binary_weight_scale):
        raise ValueError(f"binary_weight_scale must be finite, got {binary_weight_scale}")
    if binary_weight_scale < 0.0:
        raise ValueError(f"binary_weight_scale must be non-negative, got {binary_weight_scale}")
    if binary_input_scale is None:
        binary_input_scale = float(binary_weight_scale)
    if binary_recurrent_scale is None:
        binary_recurrent_scale = float(binary_weight_scale)
    if binary_readout_scale is None:
        binary_readout_scale = float(binary_weight_scale)
    for name, val in (
        ("binary_input_scale", binary_input_scale),
        ("binary_recurrent_scale", binary_recurrent_scale),
        ("binary_readout_scale", binary_readout_scale),
    ):
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val}")
        if float(val) < 0.0:
            raise ValueError(f"{name} must be non-negative, got {val}")
    if not np.isfinite(bkg_mean_scale):
        raise ValueError(f"bkg_mean_scale must be finite, got {bkg_mean_scale}")
    if not np.isfinite(input_x_scale):
        raise ValueError(f"input_x_scale must be finite, got {input_x_scale}")

    n_nodes = int(model.rsnn.neurons.num)
    n_receptors = int(model.rsnn.neuron_receptors.out.n_receptors)
    n_receptor_nodes = n_nodes * n_receptors

    neurons = model.rsnn.neurons
    v_reset = _repeat_to_length(np.asarray(neurons.v_reset), n_nodes)
    e_l = _repeat_to_length(np.asarray(neurons.e_l), n_nodes)
    v_th = _repeat_to_length(np.asarray(neurons.v_th), n_nodes)
    c_m = _repeat_to_length(np.asarray(neurons.c_m), n_nodes)
    tau = _repeat_to_length(np.asarray(neurons.tau), n_nodes)
    t_ref = _repeat_to_length(np.asarray(neurons.t_ref), n_nodes)
    k = np.asarray(neurons.k, dtype=np.float32).reshape(n_nodes, 2)
    asc_amps = np.asarray(neurons.asc_amps, dtype=np.float32).reshape(n_nodes, 2)

    in_proj = model.input_layer.input_proj
    in_weight = np.asarray(in_proj.weight.value, dtype=np.float32)
    in_fixed_sign = _maybe_get_fixed_sign(in_proj)
    in_sign, in_amp = _extract_sign_amplitude(
        in_weight,
        signed_masks=signed_masks,
        fixed_sign=in_fixed_sign,
    )
    in_base = in_sign * in_amp

    input_indices = np.asarray(in_proj.indices, dtype=np.int32)
    input_indptr = np.asarray(in_proj.indptr, dtype=np.int32)
    input_post_ids = _build_post_ids(input_indptr)
    input_pre_num = int(in_proj.conn.pre_num)
    input_post_num = int(in_proj.conn.post_num)
    if input_post_num != n_receptor_nodes:
        raise ValueError(
            f"Input projection post_num={input_post_num} mismatches n_receptor_nodes={n_receptor_nodes}"
        )

    bkg_weights = np.asarray(model.input_layer.bkg_weights, dtype=np.float32).reshape(-1)
    if bkg_weights.shape[0] != n_receptor_nodes:
        raise ValueError(
            f"bkg_weights length={bkg_weights.shape[0]} mismatches n_receptor_nodes={n_receptor_nodes}"
        )

    receptor_syn = model.rsnn.neuron_receptors.syn
    receptor_tau_decay = np.asarray(receptor_syn.tau_decay, dtype=np.float32).reshape(-1)
    receptor_psc_initial = np.asarray(receptor_syn.psc_initial, dtype=np.float32).reshape(-1)

    recurrent_projs: list[RecurrentProjCore] = []
    rec_keys: list[str] = []
    for i, proj in enumerate(model.rsnn.projs):
        key = f"rec.{i}.w"
        rec_keys.append(key)

        p = proj.proj
        comm = p.comm
        syn = p.syn

        w = np.asarray(comm.weight.value, dtype=np.float32)
        fixed_sign = _maybe_get_fixed_sign(comm)
        sign, amp = _extract_sign_amplitude(
            w,
            signed_masks=signed_masks,
            fixed_sign=fixed_sign,
        )
        base = sign * amp

        indices = np.asarray(comm.indices, dtype=np.int32)
        indptr = np.asarray(comm.indptr, dtype=np.int32)
        post_ids = _build_post_ids(indptr)
        pre_num = int(comm.conn.pre_num)
        post_num = int(comm.conn.post_num)
        if pre_num != n_nodes:
            raise ValueError(f"{key}: pre_num={pre_num} expected {n_nodes}")
        if post_num != n_receptor_nodes:
            raise ValueError(f"{key}: post_num={post_num} expected {n_receptor_nodes}")

        tau_decay_i = np.asarray(syn.tau_decay, dtype=np.float32).reshape(-1)
        psc_initial_i = np.asarray(syn.psc_initial, dtype=np.float32).reshape(-1)
        delay_steps = int(getattr(p, "delay_steps", 0))

        recurrent_projs.append(
            RecurrentProjCore(
                key=key,
                indices=jnp.asarray(indices, dtype=jnp.int32),
                indptr=jnp.asarray(indptr, dtype=jnp.int32),
                post_ids=jnp.asarray(post_ids, dtype=jnp.int32),
                pre_num=pre_num,
                post_num=post_num,
                sign=jnp.asarray(sign, dtype=jnp.float32),
                amplitude=jnp.asarray(amp, dtype=jnp.float32),
                base_weight=jnp.asarray(base, dtype=jnp.float32),
                delay_steps=delay_steps,
                tau_decay=jnp.asarray(tau_decay_i, dtype=jnp.float32),
                psc_initial=jnp.asarray(psc_initial_i, dtype=jnp.float32),
            )
        )

    if not hasattr(model, "output_head"):
        raise ValueError("BrainState path requires readout Dense head (neuron_output=False).")
    readout_w = np.asarray(model.output_head.W.value, dtype=np.float32)
    readout_b = np.asarray(model.output_head.b.value, dtype=np.float32)
    rd_sign, rd_amp = _extract_sign_amplitude(readout_w, signed_masks=signed_masks)
    rd_base = rd_sign * rd_amp
    output_indices = np.asarray(model.output_indices, dtype=np.int32)

    if model.lRout_pop != "all":
        rid = np.asarray(network_meta["laminar_indices"][model.lRout_pop], dtype=np.int32)
    else:
        rid = None

    exc_indices = np.asarray(model.rsnn.laminar_indices.get("L23e", np.arange(n_nodes)), dtype=np.int32)

    mask_specs: dict[str, MaskSpec] = {
        "input.w": MaskSpec(
            sign=jnp.asarray(in_sign, dtype=jnp.float32),
            amplitude=jnp.asarray(in_amp, dtype=jnp.float32),
            base_weight=jnp.asarray(in_base, dtype=jnp.float32),
        ),
        "readout.W": MaskSpec(
            sign=jnp.asarray(rd_sign, dtype=jnp.float32),
            amplitude=jnp.asarray(rd_amp, dtype=jnp.float32),
            base_weight=jnp.asarray(rd_base, dtype=jnp.float32),
        ),
    }
    for rp in recurrent_projs:
        mask_specs[rp.key] = MaskSpec(
            sign=rp.sign,
            amplitude=rp.amplitude,
            base_weight=rp.base_weight,
        )

    return FunctionalCore(
        n_nodes=n_nodes,
        n_receptors=n_receptors,
        dt=float(dt),
        down_sample=int(down_sample),
        v_reset=jnp.asarray(v_reset, dtype=jnp.float32),
        e_l=jnp.asarray(e_l, dtype=jnp.float32),
        v_th=jnp.asarray(v_th, dtype=jnp.float32),
        c_m=jnp.asarray(c_m, dtype=jnp.float32),
        tau=jnp.asarray(tau, dtype=jnp.float32),
        k=jnp.asarray(k, dtype=jnp.float32),
        asc_amps=jnp.asarray(asc_amps, dtype=jnp.float32),
        t_ref=jnp.asarray(t_ref, dtype=jnp.float32),
        input_indices=jnp.asarray(input_indices, dtype=jnp.int32),
        input_indptr=jnp.asarray(input_indptr, dtype=jnp.int32),
        input_post_ids=jnp.asarray(input_post_ids, dtype=jnp.int32),
        input_pre_num=input_pre_num,
        input_post_num=input_post_num,
        input_sign=jnp.asarray(in_sign, dtype=jnp.float32),
        input_amplitude=jnp.asarray(in_amp, dtype=jnp.float32),
        input_base_weight=jnp.asarray(in_base, dtype=jnp.float32),
        bkg_weights=jnp.asarray(bkg_weights, dtype=jnp.float32),
        bkg_mean_scale=float(bkg_mean_scale),
        input_x_scale=float(input_x_scale),
        receptor_tau_decay=jnp.asarray(receptor_tau_decay, dtype=jnp.float32),
        receptor_psc_initial=jnp.asarray(receptor_psc_initial, dtype=jnp.float32),
        recurrent_projs=tuple(recurrent_projs),
        readout_sign=jnp.asarray(rd_sign, dtype=jnp.float32),
        readout_amplitude=jnp.asarray(rd_amp, dtype=jnp.float32),
        readout_base_weight=jnp.asarray(rd_base, dtype=jnp.float32),
        readout_bias=jnp.asarray(readout_b, dtype=jnp.float32),
        output_indices=jnp.asarray(output_indices, dtype=jnp.int32),
        readout_neuron_indices=None if rid is None else jnp.asarray(rid, dtype=jnp.int32),
        exc_indices=jnp.asarray(exc_indices, dtype=jnp.int32),
        l2_factor=float(getattr(model, "L2_factor", 0.0)),
        mask_specs=mask_specs,
        rec_keys=tuple(rec_keys),
        weight_mode=weight_mode,
        binary_weight_scale=float(binary_weight_scale),
        binary_input_scale=float(binary_input_scale),
        binary_recurrent_scale=float(binary_recurrent_scale),
        binary_readout_scale=float(binary_readout_scale),
    )


def get_mask_specs(core: FunctionalCore) -> dict[str, MaskSpec]:
    return core.mask_specs


def _apply_weight_mode(
    sign: jnp.ndarray,
    amplitude: jnp.ndarray,
    gate: jnp.ndarray,
    *,
    mode: ECWeightMode,
    binary_scale: float,
) -> jnp.ndarray:
    if mode == "gate":
        return sign[None, ...] * amplitude[None, ...] * gate
    if mode == "binary":
        return sign[None, ...] * gate * float(binary_scale)
    raise ValueError(f"Unsupported weight mode: {mode}")


def build_effective_weights(core: FunctionalCore, pop_masks: Mapping[str, jnp.ndarray]) -> EffectiveWeights:
    pop_size = _infer_population_size(pop_masks)

    input_gate = _mask_with_broadcast(
        pop_masks, "input.w", tuple(core.input_amplitude.shape), pop_size
    )
    input_w = _apply_weight_mode(
        core.input_sign,
        core.input_amplitude,
        input_gate,
        mode=core.weight_mode,
        binary_scale=core.binary_input_scale,
    )

    rec_w: list[jnp.ndarray] = []
    for rp in core.recurrent_projs:
        gate = _mask_with_broadcast(pop_masks, rp.key, tuple(rp.amplitude.shape), pop_size)
        rec_w.append(
            _apply_weight_mode(
                rp.sign,
                rp.amplitude,
                gate,
                mode=core.weight_mode,
                binary_scale=core.binary_recurrent_scale,
            )
        )

    readout_gate = _mask_with_broadcast(
        pop_masks, "readout.W", tuple(core.readout_amplitude.shape), pop_size
    )
    readout_w = _apply_weight_mode(
        core.readout_sign,
        core.readout_amplitude,
        readout_gate,
        mode=core.weight_mode,
        binary_scale=core.binary_readout_scale,
    )

    return EffectiveWeights(input_w=input_w, rec_w=tuple(rec_w), readout_w=readout_w)


class EventCSRLinearDaleBS(bs.nn.Module):
    def __init__(
        self,
        *,
        edge_pre_ids: jnp.ndarray,
        edge_post_ids: jnp.ndarray,
        pre_num: int,
        post_num: int,
        sign: jnp.ndarray,
        amplitude: jnp.ndarray,
    ):
        super().__init__()
        self.edge_pre_ids = bs.ParamState(_as_i32(edge_pre_ids))
        self.edge_post_ids = bs.ParamState(_as_i32(edge_post_ids))
        self.pre_num = int(pre_num)
        self.post_num = int(post_num)
        self.sign = bs.ParamState(_as_f32(sign))
        self.amplitude = bs.ParamState(_as_f32(amplitude))

    def weight_from_gate(self, gate: jnp.ndarray) -> jnp.ndarray:
        return self.sign.value * self.amplitude.value * gate

    def update(self, x: jnp.ndarray, *, weight: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if weight is None:
            weight = self.sign.value * self.amplitude.value
        return _csr_mv_single(
            x,
            weight,
            self.edge_pre_ids.value,
            self.edge_post_ids.value,
            self.post_num,
        )


class AlphaBS(bs.nn.Module):
    def __init__(self, *, size: int, tau_decay: jnp.ndarray, dt: float):
        super().__init__()
        self.size = int(size)
        self.dt = float(dt)
        self.tau_decay = bs.ParamState(_as_f32(tau_decay).reshape(-1))
        self.psc_initial = bs.ParamState(_as_f32(np.e) / self.tau_decay.value)
        self.reset_state(pop_size=1, batch_size=1)

    def reset_state(self, *, pop_size: int, batch_size: int) -> None:
        shp = (int(pop_size), int(batch_size), self.size)
        self.h = bs.ShortTermState(jnp.zeros(shp, dtype=jnp.float32))
        self.g = bs.ShortTermState(jnp.zeros(shp, dtype=jnp.float32))

    def update(self, x: jnp.ndarray) -> jnp.ndarray:
        decay = jnp.exp(-self.dt / self.tau_decay.value)
        h_old = self.h.value
        g_old = self.g.value

        h_new = decay[None, :] * h_old + x * self.psc_initial.value[None, :]
        g_new = decay[None, :] * g_old + self.dt * decay[None, :] * h_old

        self.h.value = h_new
        self.g.value = g_new
        return g_new


class CUBAMultiReceptorBS(bs.nn.Module):
    def __init__(self, *, n_nodes: int, n_receptors: int):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.n_receptors = int(n_receptors)

    def update(self, conductance: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(conductance.reshape(conductance.shape[0], self.n_nodes, self.n_receptors), axis=-1)


class DelayFIFOAlignPostBS(bs.nn.Module):
    def __init__(
        self,
        *,
        pre_num: int,
        delay_steps: int,
        comm: EventCSRLinearDaleBS,
        syn: AlphaBS,
        out: CUBAMultiReceptorBS,
    ):
        super().__init__()
        self.pre_num = int(pre_num)
        self.delay_steps = max(int(delay_steps), 0)
        self.comm = comm
        self.syn = syn
        self.out = out
        self.reset_state(pop_size=1, batch_size=1)

    def reset_state(self, *, pop_size: int, batch_size: int) -> None:
        self.syn.reset_state(pop_size=pop_size, batch_size=batch_size)
        q_steps = max(self.delay_steps, 1)
        self.delay_queue = bs.ShortTermState(
            jnp.zeros((int(pop_size), int(batch_size), q_steps, self.pre_num), dtype=jnp.float32)
        )
        self.delay_head = bs.ShortTermState(jnp.zeros((int(pop_size),), dtype=jnp.int32))

    def update(self, spike_now: jnp.ndarray, *, weight: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        q = self.delay_queue.value
        h = self.delay_head.value

        if self.delay_steps > 0:
            delayed_spike = q[:, h, :]
            q = q.at[:, h, :].set(spike_now)
            h_new = jnp.asarray((h + 1) % self.delay_steps, dtype=jnp.int32)
            self.delay_queue.value = q
            self.delay_head.value = h_new
        else:
            delayed_spike = spike_now

        comm_out = self.comm.update(delayed_spike, weight=weight)
        g = self.syn.update(comm_out)
        y = self.out.update(g)
        return y, g


class AlphaCUBABS(bs.nn.Module):
    def __init__(
        self,
        *,
        pre_num: int,
        n_nodes: int,
        n_receptors: int,
        delay_steps: int,
        comm: EventCSRLinearDaleBS,
        syn: AlphaBS,
    ):
        super().__init__()
        self.proj = DelayFIFOAlignPostBS(
            pre_num=pre_num,
            delay_steps=delay_steps,
            comm=comm,
            syn=syn,
            out=CUBAMultiReceptorBS(n_nodes=n_nodes, n_receptors=n_receptors),
        )

    def reset_state(self, *, pop_size: int, batch_size: int) -> None:
        self.proj.reset_state(pop_size=pop_size, batch_size=batch_size)

    def update(self, spike_now: jnp.ndarray, *, weight: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.proj.update(spike_now, weight=weight)


class InputLayerBS(bs.nn.Module):
    def __init__(
        self,
        *,
        n_nodes: int,
        n_receptors: int,
        dt: float,
        comm: EventCSRLinearDaleBS,
        tau_decay: jnp.ndarray,
        psc_initial: jnp.ndarray,
        bkg_weights: jnp.ndarray,
        bkg_mean_scale: float,
        input_x_scale: float,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.n_receptors = int(n_receptors)
        self.comm = comm
        self.syn = AlphaBS(size=n_nodes * n_receptors, tau_decay=tau_decay, dt=dt)
        self.syn.psc_initial = bs.ParamState(_as_f32(psc_initial))
        self.out = CUBAMultiReceptorBS(n_nodes=n_nodes, n_receptors=n_receptors)
        self.bkg_weights = bs.ParamState(_as_f32(bkg_weights).reshape(-1))
        self.bkg_mean_scale = float(bkg_mean_scale)
        self.input_x_scale = float(input_x_scale)

    def reset_state(self, *, pop_size: int, batch_size: int) -> None:
        self.syn.reset_state(pop_size=pop_size, batch_size=batch_size)

    def update(self, x: jnp.ndarray, *, weight: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x_eff = x * self.input_x_scale
        input_4n = self.comm.update(x_eff, weight=weight)
        if self.bkg_mean_scale != 0.0:
            input_4n = input_4n + self.bkg_mean_scale * self.bkg_weights.value[None, :]
        g = self.syn.update(input_4n)
        y = self.out.update(g)
        return y, input_4n


class GLIF3TFOrderBS(bs.nn.Module):
    def __init__(
        self,
        *,
        n_nodes: int,
        dt: float,
        v_reset: jnp.ndarray,
        e_l: jnp.ndarray,
        v_th: jnp.ndarray,
        c_m: jnp.ndarray,
        tau: jnp.ndarray,
        k: jnp.ndarray,
        asc_amps: jnp.ndarray,
        t_ref: jnp.ndarray,
        spk_reset: str = "hard",
        spike_gradient: bool = False,
        surrogate_beta: float = 8.0,
    ):
        super().__init__()
        if spk_reset not in ("soft", "hard"):
            raise ValueError(f"spk_reset must be 'soft' or 'hard', got {spk_reset}")

        self.n_nodes = int(n_nodes)
        self.dt = float(dt)
        self.spk_reset = str(spk_reset)
        self.spike_gradient = bool(spike_gradient)
        self.surrogate_beta = float(surrogate_beta)

        self.v_reset = bs.ParamState(_as_f32(v_reset))
        self.e_l = bs.ParamState(_as_f32(e_l))
        self.v_th = bs.ParamState(_as_f32(v_th))
        self.c_m = bs.ParamState(_as_f32(c_m))
        self.tau = bs.ParamState(_as_f32(tau))
        self.k = bs.ParamState(_as_f32(k))
        self.asc_amps = bs.ParamState(_as_f32(asc_amps))
        self.t_ref = bs.ParamState(_as_f32(t_ref))

        self.reset_state(pop_size=1, batch_size=1)

    def reset_state(self, *, pop_size: int, batch_size: int) -> None:
        shp = (int(pop_size), int(batch_size), self.n_nodes)
        v0 = jnp.broadcast_to(self.v_reset.value[None, None, :], shp)
        zeros = jnp.zeros(shp, dtype=jnp.float32)

        self.v = bs.ShortTermState(v0)
        self.iasc1 = bs.ShortTermState(zeros)
        self.iasc2 = bs.ShortTermState(zeros)
        self.spike = bs.ShortTermState(zeros)
        self.refr = bs.ShortTermState(zeros)
        self.input_current = bs.ShortTermState(zeros)

    def update(self, iinp: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        prev_spike = self.spike.value

        spike_pos = prev_spike > 0.0
        refr_add = jnp.where(spike_pos, self.t_ref.value[None, :], 0.0)
        iasc1_add = jnp.where(spike_pos, self.asc_amps.value[None, :, 0], 0.0)
        iasc2_add = jnp.where(spike_pos, self.asc_amps.value[None, :, 1], 0.0)

        refr_new = jax.nn.relu(self.refr.value + refr_add - self.dt)

        exp_k1 = jnp.exp(-self.dt * self.k.value[:, 0])[None, :]
        exp_k2 = jnp.exp(-self.dt * self.k.value[:, 1])[None, :]
        iasc1_new = exp_k1 * self.iasc1.value + iasc1_add
        iasc2_new = exp_k2 * self.iasc2.value + iasc2_add

        exp_tau = jnp.exp(-self.dt / self.tau.value)
        current_factor = (1.0 - exp_tau) * (self.tau.value / self.c_m.value)
        g_e_l = self.c_m.value / self.tau.value * self.e_l.value

        v_pre_reset = exp_tau[None, :] * self.v.value + current_factor[None, :] * (
            iinp + self.iasc1.value + self.iasc2.value + g_e_l[None, :]
        )

        if self.spk_reset == "soft":
            v_new = v_pre_reset + (self.v_reset.value[None, :] - self.v_th.value[None, :]) * prev_spike
        else:
            v_new = v_pre_reset + (self.v_reset.value[None, :] - v_pre_reset) * prev_spike

        v_den = jnp.maximum(self.v_th.value - self.e_l.value, 1e-3)[None, :]
        v_sc = (v_new - self.v_th.value[None, :]) / v_den

        if self.spike_gradient:
            spike_new = jax.nn.sigmoid(self.surrogate_beta * v_sc)
        else:
            spike_new = (v_sc > 0.0).astype(jnp.float32)
        spike_new = spike_new * (refr_new <= 0.0).astype(jnp.float32)

        self.v.value = v_new
        self.iasc1.value = iasc1_new
        self.iasc2.value = iasc2_new
        self.spike.value = spike_new
        self.refr.value = refr_new
        self.input_current.value = iinp

        return spike_new, v_new, iasc1_new, iasc2_new, refr_new


class BillehColumnTFAlignedBS(bs.nn.Module):
    """
    BrainState modular composition aligned with models2 / tensorflow update order.
    """

    def __init__(
        self,
        core: FunctionalCore,
        *,
        pop_size: int = 1,
        batch_size: int = 1,
        spk_reset: str = "hard",
        spike_gradient: bool = False,
        surrogate_beta: float = 8.0,
        track_epsc_ipsc: bool = False,
    ):
        super().__init__()
        self.core = core
        self.pop_size = int(pop_size)
        self.batch_size = int(batch_size)
        self.track_epsc_ipsc = bool(track_epsc_ipsc)

        self.n_nodes = int(core.n_nodes)
        self.n_receptors = int(core.n_receptors)
        self.n_receptor_nodes = self.n_nodes * self.n_receptors

        self.neurons = GLIF3TFOrderBS(
            n_nodes=self.n_nodes,
            dt=float(core.dt),
            v_reset=core.v_reset,
            e_l=core.e_l,
            v_th=core.v_th,
            c_m=core.c_m,
            tau=core.tau,
            k=core.k,
            asc_amps=core.asc_amps,
            t_ref=core.t_ref,
            spk_reset=spk_reset,
            spike_gradient=spike_gradient,
            surrogate_beta=surrogate_beta,
        )

        self.input_layer = InputLayerBS(
            n_nodes=self.n_nodes,
            n_receptors=self.n_receptors,
            dt=float(core.dt),
            comm=EventCSRLinearDaleBS(
                edge_pre_ids=core.input_post_ids,
                edge_post_ids=core.input_indices,
                pre_num=int(core.input_pre_num),
                post_num=int(core.input_post_num),
                sign=core.input_sign,
                amplitude=core.input_amplitude,
            ),
            tau_decay=core.receptor_tau_decay,
            psc_initial=core.receptor_psc_initial,
            bkg_weights=core.bkg_weights,
            bkg_mean_scale=float(core.bkg_mean_scale),
            input_x_scale=float(core.input_x_scale),
        )

        self.recurrent_projs: list[AlphaCUBABS] = []
        for rp in core.recurrent_projs:
            comm = EventCSRLinearDaleBS(
                edge_pre_ids=rp.post_ids,
                edge_post_ids=rp.indices,
                pre_num=int(rp.pre_num),
                post_num=int(rp.post_num),
                sign=rp.sign,
                amplitude=rp.amplitude,
            )
            syn = AlphaBS(size=int(rp.post_num), tau_decay=rp.tau_decay, dt=float(core.dt))
            syn.psc_initial = bs.ParamState(_as_f32(rp.psc_initial))
            self.recurrent_projs.append(
                AlphaCUBABS(
                    pre_num=int(rp.pre_num),
                    n_nodes=self.n_nodes,
                    n_receptors=self.n_receptors,
                    delay_steps=int(rp.delay_steps),
                    comm=comm,
                    syn=syn,
                )
            )

        self.readout_sign = bs.ParamState(_as_f32(core.readout_sign))
        self.readout_amplitude = bs.ParamState(_as_f32(core.readout_amplitude))
        self.readout_bias = bs.ParamState(_as_f32(core.readout_bias))
        self.output_indices = bs.ParamState(_as_i32(core.output_indices))
        self.readout_neuron_indices = (
            None
            if core.readout_neuron_indices is None
            else bs.ParamState(_as_i32(core.readout_neuron_indices))
        )

        self.input_gate = bs.ParamState(
            jnp.ones((self.pop_size, int(core.input_amplitude.shape[0])), dtype=jnp.float32)
        )
        self.rec_gates = [
            bs.ParamState(jnp.ones((self.pop_size, int(rp.amplitude.shape[0])), dtype=jnp.float32))
            for rp in core.recurrent_projs
        ]
        self.readout_gate = bs.ParamState(
            jnp.ones((self.pop_size, *tuple(core.readout_amplitude.shape)), dtype=jnp.float32)
        )

        self.exc_indices = bs.ParamState(_as_i32(core.exc_indices))
        self.n_exc = int(core.exc_indices.shape[0])

        self._vm_step = None
        self._jit_rollout = None

        self.reset_state(pop_size=self.pop_size, batch_size=self.batch_size)

    @classmethod
    def from_brainpy_model(
        cls,
        model,
        network_meta: dict[str, Any],
        *,
        dt: float,
        down_sample: int,
        pop_size: int = 1,
        batch_size: int = 1,
        signed_masks: bool = True,
        bkg_mean_scale: float = 0.1,
        input_x_scale: float = 1.0,
        **kwargs,
    ) -> "BillehColumnTFAlignedBS":
        core = build_functional_core(
            model,
            network_meta,
            dt=dt,
            down_sample=down_sample,
            signed_masks=signed_masks,
            bkg_mean_scale=bkg_mean_scale,
            input_x_scale=input_x_scale,
        )
        return cls(
            core,
            pop_size=pop_size,
            batch_size=batch_size,
            **kwargs,
        )

    def _resize_mask_state(self, st: bs.ParamState, shape_tail: tuple[int, ...]) -> None:
        expected = (self.pop_size, *shape_tail)
        if tuple(st.value.shape) != expected:
            st.value = jnp.ones(expected, dtype=jnp.float32)

    def reset_state(
        self,
        *,
        pop_size: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        if pop_size is not None:
            self.pop_size = int(pop_size)
        if batch_size is not None:
            self.batch_size = int(batch_size)

        self._resize_mask_state(self.input_gate, (int(self.core.input_amplitude.shape[0]),))
        for gate, rp in zip(self.rec_gates, self.core.recurrent_projs):
            self._resize_mask_state(gate, (int(rp.amplitude.shape[0]),))
        self._resize_mask_state(self.readout_gate, tuple(self.core.readout_amplitude.shape))

        self.neurons.reset_state(pop_size=self.pop_size, batch_size=self.batch_size)
        self.input_layer.reset_state(pop_size=self.pop_size, batch_size=self.batch_size)
        for proj in self.recurrent_projs:
            proj.reset_state(pop_size=self.pop_size, batch_size=self.batch_size)

        shp = (self.pop_size, self.batch_size, self.n_exc)
        self.epsc = bs.ShortTermState(jnp.zeros(shp, dtype=jnp.float32))
        self.ipsc = bs.ShortTermState(jnp.zeros(shp, dtype=jnp.float32))

        self._vm_step = None
        self._jit_rollout = None

    def _normalize_mask(self, m: Any, expected_shape: tuple[int, ...], name: str) -> jnp.ndarray:
        arr = jnp.asarray(m, dtype=jnp.float32)
        if tuple(arr.shape) == expected_shape:
            return arr
        if tuple(arr.shape) == expected_shape[1:]:
            return jnp.broadcast_to(arr[None, ...], expected_shape)
        raise ValueError(
            f"Mask '{name}' shape mismatch: got {arr.shape}, expected {expected_shape} or {expected_shape[1:]}"
        )

    def set_population_masks(self, pop_masks: Mapping[str, Any]) -> None:
        expected_input = (self.pop_size, int(self.input_layer.comm.amplitude.value.shape[0]))
        if "input.w" in pop_masks:
            self.input_gate.value = self._normalize_mask(pop_masks["input.w"], expected_input, "input.w")

        expected_readout = (self.pop_size, *tuple(self.readout_amplitude.value.shape))
        if "readout.W" in pop_masks:
            self.readout_gate.value = self._normalize_mask(pop_masks["readout.W"], expected_readout, "readout.W")

        for i, (gate, rp) in enumerate(zip(self.rec_gates, self.core.recurrent_projs)):
            key = rp.key
            if key in pop_masks:
                exp_shape = (self.pop_size, int(self.core.recurrent_projs[i].amplitude.shape[0]))
                gate.value = self._normalize_mask(pop_masks[key], exp_shape, key)

    def effective_weights(self) -> dict[str, Any]:
        input_w = _apply_weight_mode(
            self.input_layer.comm.sign.value,
            self.input_layer.comm.amplitude.value,
            self.input_gate.value,
            mode=self.core.weight_mode,
            binary_scale=self.core.binary_input_scale,
        )

        rec_w = []
        for proj, gate in zip(self.recurrent_projs, self.rec_gates):
            rec_w.append(
                _apply_weight_mode(
                    proj.proj.comm.sign.value,
                    proj.proj.comm.amplitude.value,
                    gate.value,
                    mode=self.core.weight_mode,
                    binary_scale=self.core.binary_recurrent_scale,
                )
            )

        readout_w = _apply_weight_mode(
            self.readout_sign.value,
            self.readout_amplitude.value,
            self.readout_gate.value,
            mode=self.core.weight_mode,
            binary_scale=self.core.binary_readout_scale,
        )
        return {"input_w": input_w, "rec_w": tuple(rec_w), "readout_w": readout_w}

    def _single_step(self, x_t: jnp.ndarray, eff_model: Mapping[str, Any]):
        x_t = _as_f32(x_t)
        bsz = int(x_t.shape[0])

        prev_spike = self.neurons.spike.value

        total_rec_in = jnp.zeros((bsz, self.n_nodes), dtype=jnp.float32)
        epsc_total = jnp.zeros((bsz, self.n_exc), dtype=jnp.float32)
        ipsc_total = jnp.zeros((bsz, self.n_exc), dtype=jnp.float32)

        for i, proj in enumerate(self.recurrent_projs):
            rec_cond, g_full = proj.update(prev_spike, weight=_as_f32(eff_model["rec_w"][i]))
            total_rec_in = total_rec_in + rec_cond

            if self.track_epsc_ipsc and self.n_exc > 0:
                g_reshape = g_full.reshape(bsz, self.n_nodes, self.n_receptors)
                exc_idx = self.exc_indices.value
                epsc_total = epsc_total + g_reshape[:, exc_idx, 0] + g_reshape[:, exc_idx, 1]
                ipsc_total = ipsc_total + g_reshape[:, exc_idx, 2] + g_reshape[:, exc_idx, 3]

        input_current, _ = self.input_layer.update(x_t, weight=_as_f32(eff_model["input_w"]))
        iinp = total_rec_in + input_current

        spike_new, v_new, iasc1_new, iasc2_new, _ = self.neurons.update(iinp)

        if self.readout_neuron_indices is None:
            read_spike = spike_new
        else:
            read_spike = jnp.take(spike_new, self.readout_neuron_indices.value, axis=-1)

        output_all = jnp.einsum("bn,no->bo", read_spike, _as_f32(eff_model["readout_w"]))
        output_all = output_all + self.readout_bias.value[None, :]
        logits = jnp.take(output_all, self.output_indices.value, axis=-1)

        self.epsc.value = epsc_total
        self.ipsc.value = ipsc_total

        return (
            logits,
            spike_new,
            v_new,
            iasc1_new,
            iasc2_new,
            epsc_total,
            ipsc_total,
            total_rec_in,
            input_current,
            iinp,
        )

    def _get_vmapped_step(self):
        if self._vm_step is None:
            short_states = self.states(bs.ShortTermState)
            self._vm_step = bs.transform.vmap(
                self._single_step,
                in_axes=(None, 0),
                out_axes=0,
                in_states=short_states,
                out_states=short_states,
            )
        return self._vm_step

    def population_step(
        self,
        x_t: jnp.ndarray,
        *,
        eff_weights: Mapping[str, Any] | None = None,
    ):
        vm_step = self._get_vmapped_step()
        if eff_weights is None:
            eff_weights = self.effective_weights()
        return vm_step(x_t, eff_weights)

    def rollout(
        self,
        x_seq: jnp.ndarray,
        *,
        pop_masks: Mapping[str, Any] | None = None,
        jit: bool = False,
    ) -> RolloutResult:
        if pop_masks is not None:
            self.set_population_masks(pop_masks)

        x_seq = _as_f32(x_seq)
        if x_seq.ndim != 3:
            raise ValueError(f"x_seq must be [B, T, D], got {x_seq.shape}")
        if int(x_seq.shape[0]) != self.batch_size:
            raise ValueError(
                f"Batch mismatch: model batch_size={self.batch_size}, x_seq batch={x_seq.shape[0]}"
            )

        x_tf = jnp.swapaxes(x_seq, 0, 1)
        eff = self.effective_weights()
        vm_step = self._get_vmapped_step()

        def _run(xs):
            return bs.transform.for_loop(lambda xt: vm_step(xt, eff), xs)

        if jit:
            if self._jit_rollout is None:
                self._jit_rollout = bs.transform.jit(_run)
            outs = self._jit_rollout(x_tf)
        else:
            outs = _run(x_tf)

        (
            logits_t,
            spikes_t,
            voltage_t,
            iasc1_t,
            iasc2_t,
            epsc_t,
            ipsc_t,
            rec_i_t,
            input_i_t,
            iinp_t,
        ) = outs

        logits_t = jnp.transpose(logits_t, (1, 2, 0, 3))
        spikes_t = jnp.transpose(spikes_t, (1, 2, 0, 3))
        voltage_t = jnp.transpose(voltage_t, (1, 2, 0, 3))

        extras = {
            "iasc1_t": jnp.transpose(iasc1_t, (1, 2, 0, 3)),
            "iasc2_t": jnp.transpose(iasc2_t, (1, 2, 0, 3)),
            "epsc_t": jnp.transpose(epsc_t, (1, 2, 0, 3)),
            "ipsc_t": jnp.transpose(ipsc_t, (1, 2, 0, 3)),
            "rec_i_t": jnp.transpose(rec_i_t, (1, 2, 0, 3)),
            "input_i_t": jnp.transpose(input_i_t, (1, 2, 0, 3)),
            "iinp_t": jnp.transpose(iinp_t, (1, 2, 0, 3)),
            "psc_sum_t": jnp.transpose(iinp_t, (1, 2, 0, 3)),
        }
        return RolloutResult(
            logits_t=logits_t,
            spikes_t=spikes_t,
            voltage_t=voltage_t,
            extras=extras,
        )

    def short_state_snapshot(self) -> dict[tuple[Any, ...], jnp.ndarray]:
        return {path: st.value for path, st in self.states(bs.ShortTermState).items()}

    def short_state_restore(
        self,
        snapshot: Mapping[tuple[Any, ...], jnp.ndarray],
        *,
        strict: bool = True,
    ) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
        short_states = self.states(bs.ShortTermState)
        unexpected = [k for k in snapshot.keys() if k not in short_states]
        missing = []
        for k, st in short_states.items():
            if k in snapshot:
                st.value = snapshot[k]
            else:
                missing.append(k)
        if strict and (unexpected or missing):
            raise KeyError(f"State mismatch. unexpected={unexpected[:5]}, missing={missing[:5]}")
        return unexpected, missing


_MODEL_CACHE: dict[tuple[int, int, int, bool, str, bool, float], BillehColumnTFAlignedBS] = {}


def _get_bs_model(
    core: FunctionalCore,
    *,
    pop_size: int,
    batch_size: int,
    track_epsc_ipsc: bool,
    spk_reset: str,
    spike_gradient: bool,
    surrogate_beta: float,
) -> BillehColumnTFAlignedBS:
    key = (
        id(core),
        int(pop_size),
        int(batch_size),
        bool(track_epsc_ipsc),
        str(spk_reset),
        bool(spike_gradient),
        float(surrogate_beta),
    )
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = BillehColumnTFAlignedBS(
            core,
            pop_size=pop_size,
            batch_size=batch_size,
            spk_reset=spk_reset,
            spike_gradient=spike_gradient,
            surrogate_beta=surrogate_beta,
            track_epsc_ipsc=track_epsc_ipsc,
        )
        _MODEL_CACHE[key] = model
    return model


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()


def rollout_population(
    core: FunctionalCore,
    pop_masks: Mapping[str, jnp.ndarray],
    x_seq: jnp.ndarray,
    *,
    collect_extra: bool = False,
    use_jit: bool = True,
    spk_reset: str = "hard",
    spike_gradient: bool = False,
    surrogate_beta: float = 8.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    x_seq = jnp.asarray(x_seq, dtype=jnp.float32)
    if x_seq.ndim != 3:
        raise ValueError(f"x_seq must be [B, T, D], got {x_seq.shape}")

    pop_size = _infer_population_size(pop_masks)
    batch_size = int(x_seq.shape[0])

    model = _get_bs_model(
        core,
        pop_size=pop_size,
        batch_size=batch_size,
        track_epsc_ipsc=bool(collect_extra),
        spk_reset=spk_reset,
        spike_gradient=spike_gradient,
        surrogate_beta=surrogate_beta,
    )
    model.reset_state(pop_size=pop_size, batch_size=batch_size)
    out = model.rollout(x_seq, pop_masks=pop_masks, jit=use_jit)

    extras = out.extras if collect_extra else {}
    return out.logits_t, out.spikes_t, out.voltage_t, extras


def compute_population_losses(
    core: FunctionalCore,
    eff: EffectiveWeights,
    logits_t: jnp.ndarray,
    spikes_t: jnp.ndarray,
    voltage_t: jnp.ndarray,
    labels: jnp.ndarray,
    weights: jnp.ndarray,
    *,
    target_firing_rates: jnp.ndarray,
    rate_cost: float,
    voltage_cost: float,
    recurrent_weight_regularization: float,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    labels = jnp.asarray(labels, dtype=jnp.int32)
    weights = jnp.asarray(weights, dtype=jnp.float32)
    if labels.ndim != 2 or weights.ndim != 2:
        raise ValueError(f"labels/weights must be [B, chunks], got {labels.shape}, {weights.shape}")

    pop_size, batch_size, tlen, _ = logits_t.shape
    n_chunks = int(labels.shape[1])
    kept = n_chunks * int(core.down_sample)
    if kept > tlen:
        raise ValueError(f"Not enough timesteps: have {tlen}, need {kept} for down_sample={core.down_sample}")
    logits_kept = logits_t[:, :, :kept, :]
    logits_chunks = logits_kept.reshape(pop_size, batch_size, n_chunks, core.down_sample, -1).mean(axis=3)

    log_probs = jax.nn.log_softmax(logits_chunks, axis=-1)
    picked = jnp.take_along_axis(log_probs, labels[None, :, :, None], axis=-1)[..., 0]
    denom = jnp.maximum(jnp.sum(weights), 1e-6)
    cls_loss = -jnp.sum(picked * weights[None, :, :], axis=(1, 2)) / denom

    pred = jnp.argmax(logits_chunks, axis=-1)
    acc = jnp.sum((pred == labels[None, :, :]) * weights[None, :, :], axis=(1, 2)) / denom

    mean_rate = jnp.mean(spikes_t, axis=(1, 2, 3))

    rate_loss = jnp.zeros((pop_size,), dtype=jnp.float32)
    if rate_cost > 0:
        rates = jnp.mean(spikes_t, axis=(1, 2))
        target = jnp.asarray(target_firing_rates, dtype=jnp.float32)
        if target.ndim != 1 or target.shape[0] != core.n_nodes:
            raise ValueError(f"target_firing_rates must be [N={core.n_nodes}], got {target.shape}")
        sorted_rates = jnp.sort(rates, axis=-1)
        sorted_target = jnp.sort(target)
        rate_loss = rate_cost * jnp.mean((sorted_rates - sorted_target[None, :]) ** 2, axis=-1)

    voltage_loss = jnp.zeros((pop_size,), dtype=jnp.float32)
    if voltage_cost > 0:
        v32 = (voltage_t - core.e_l[None, None, None, :]) / jnp.maximum(
            core.v_th[None, None, None, :] - core.e_l[None, None, None, :], 1e-3
        )
        v_pos = jax.nn.relu(v32 - 1.0) ** 2
        v_neg = jax.nn.relu(-v32 + 1.0) ** 2
        v_loss = jnp.sum(v_pos + v_neg, axis=-1)
        voltage_loss = voltage_cost * jnp.mean(v_loss, axis=(1, 2))

    rec_loss = jnp.zeros((pop_size,), dtype=jnp.float32)
    if recurrent_weight_regularization > 0 and len(core.recurrent_projs) > 0:
        rec_pen = jnp.zeros((pop_size,), dtype=jnp.float32)
        for i, rp in enumerate(core.recurrent_projs):
            diff = eff.rec_w[i] - rp.base_weight[None, :]
            rec_pen = rec_pen + jnp.sum(diff * diff, axis=-1)
        rec_loss = recurrent_weight_regularization * rec_pen

    l2_loss = jnp.zeros((pop_size,), dtype=jnp.float32)
    if core.l2_factor > 0:
        l2_loss = 0.5 * core.l2_factor * jnp.sum(eff.readout_w * eff.readout_w, axis=(1, 2))

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
    return total, acc, aux


def make_population_eval_fn(
    core: FunctionalCore,
    *,
    target_firing_rates: jnp.ndarray,
    rate_cost: float,
    voltage_cost: float,
    recurrent_weight_regularization: float,
    use_jit_rollout: bool = True,
    spk_reset: str = "hard",
    spike_gradient: bool = False,
    surrogate_beta: float = 8.0,
):
    target_firing_rates = jnp.asarray(target_firing_rates, dtype=jnp.float32)
    rate_cost = float(rate_cost)
    voltage_cost = float(voltage_cost)
    recurrent_weight_regularization = float(recurrent_weight_regularization)

    def _eval(pop_masks, x_seq, labels, weights):
        x_seq = jnp.asarray(x_seq, dtype=jnp.float32)
        labels = jnp.asarray(labels, dtype=jnp.int32)
        weights = jnp.asarray(weights, dtype=jnp.float32)

        eff = build_effective_weights(core, pop_masks)
        logits_t, spikes_t, voltage_t, _ = rollout_population(
            core,
            pop_masks,
            x_seq,
            collect_extra=False,
            use_jit=use_jit_rollout,
            spk_reset=spk_reset,
            spike_gradient=spike_gradient,
            surrogate_beta=surrogate_beta,
        )
        loss, acc, aux = compute_population_losses(
            core,
            eff,
            logits_t,
            spikes_t,
            voltage_t,
            labels,
            weights,
            target_firing_rates=target_firing_rates,
            rate_cost=rate_cost,
            voltage_cost=voltage_cost,
            recurrent_weight_regularization=recurrent_weight_regularization,
        )
        return loss, acc, aux

    return _eval


