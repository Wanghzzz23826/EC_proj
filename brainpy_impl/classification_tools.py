from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import brainpy as bp
import brainpy.math as bm
import numpy as np
from jax.lax import stop_gradient

from common.types import InputParams, NodeParams, SynapseParams, to_brainpy_csr

try:
    from . import models
except ImportError:  # pragma: no cover - fallback for direct script usage
    import models


_OUTPUT_MODE_TO_INDEX = {
    "garrett": np.array([0, 1], dtype=np.int32),
    "vcd_grating": np.array([2, 3], dtype=np.int32),
    "ori_diff": np.array([4, 5], dtype=np.int32),
    "evidence": np.array([6, 7], dtype=np.int32),
    "10class": np.arange(8, 18, dtype=np.int32),
}


def _as_np(x, dtype=None):
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


@dataclass
class _NormalizedNetwork:
    node_params: dict
    node_type_ids: np.ndarray
    n_nodes: int
    n_edges: int
    synapses: dict
    laminar_indices: dict
    readout_ids: dict


def _normalize_network(network: dict) -> _NormalizedNetwork:
    # Support both common/tensorflow-style network dict and brainpy_impl/load_sparse dict.
    if "node_params" in network:
        node_params = network["node_params"]
    elif "neuron_params" in network:
        node_params = network["neuron_params"]
    else:
        raise KeyError("network must contain either 'node_params' or 'neuron_params'")

    if "node_type_ids" in network:
        node_type_ids = _as_np(network["node_type_ids"], np.int32)
    elif "neuron_type_ids" in network:
        node_type_ids = _as_np(network["neuron_type_ids"], np.int32)
    else:
        raise KeyError("network must contain either 'node_type_ids' or 'neuron_type_ids'")

    if "n_nodes" in network:
        n_nodes = int(network["n_nodes"])
    elif "n_neurons" in network:
        n_nodes = int(network["n_neurons"])
    else:
        n_nodes = int(len(node_type_ids))

    if "synapses" in network:
        synapses = network["synapses"]
    elif "syn" in network:
        syn = network["syn"]
        pre = _as_np(syn["pre"], np.int64)
        post = _as_np(syn["post"], np.int64)
        receptor = _as_np(syn["receptor"], np.int64)
        synapses = {
            "indices": np.stack([post * 4 + receptor, pre], axis=-1).astype(np.int64),
            "weights": _as_np(syn["weight"], np.float32),
            "delays": _as_np(syn["delay"], np.float32),
            "dense_shape": (4 * n_nodes, n_nodes),
        }
    else:
        raise KeyError("network must contain either 'synapses' or 'syn'")

    if "n_edges" in network:
        n_edges = int(network["n_edges"])
    else:
        n_edges = int(len(_as_np(synapses["weights"])))

    laminar_indices = {
        k: _as_np(v, np.int32).reshape(-1) for k, v in network.get("laminar_indices", {}).items()
    }
    readout_ids = {
        k: _as_np(v, np.int32).reshape(-1)
        for k, v in network.items()
        if k.startswith("localized_readout_neuron_ids_")
    }

    return _NormalizedNetwork(
        node_params=node_params,
        node_type_ids=node_type_ids,
        n_nodes=n_nodes,
        n_edges=n_edges,
        synapses=synapses,
        laminar_indices=laminar_indices,
        readout_ids=readout_ids,
    )


class BillehClassificationModel(bp.DynSysGroup):
    """
    BrainPy/JAX model counterpart of tensorflow classification model builder.

    The model is stateful: call `forward()` with a full sequence.
    """

    def __init__(
        self,
        rsnn: models.BillehColumn,
        input_layer: models.InputLayer,
        network_meta: _NormalizedNetwork,
        *,
        seq_len: int,
        down_sample: int,
        n_output: int,
        output_mode: str,
        neuron_output: bool,
        lRout_pop: str,
        dampening_factor: float,
        full_output: bool,
        L2_factor: float,
        batch_size: int,
    ):
        super().__init__()
        self.rsnn = rsnn
        self.input_layer = input_layer
        self.network_meta = network_meta
        self.seq_len = int(seq_len)
        self.down_sample = int(down_sample)
        self.n_output = int(n_output)
        self.output_mode = output_mode
        self.neuron_output = neuron_output
        self.lRout_pop = lRout_pop
        self.dampening_factor = float(dampening_factor)
        self.default_full_output = bool(full_output)
        self.L2_factor = float(L2_factor)
        self.batch_size = int(batch_size)

        if self.neuron_output:
            self.output_scale = models.ScaleUp(0.1)
        else:
            if self.lRout_pop != "all":
                if self.lRout_pop not in self.network_meta.laminar_indices:
                    raise KeyError(f"Unknown lRout_pop '{self.lRout_pop}'")
                n_in = int(self.network_meta.laminar_indices[self.lRout_pop].size)
            else:
                n_in = int(self.network_meta.n_nodes)
            self.output_head = bp.dnn.Dense(n_in, 18)
            self.output_indices = _OUTPUT_MODE_TO_INDEX[self.output_mode]

    def _readout_spikes(self, spikes):
        # spikes shape: [batch, n_nodes]
        if self.neuron_output:
            output_spikes = (
                (1.0 / self.dampening_factor) * spikes
                + (1.0 - 1.0 / self.dampening_factor) * stop_gradient(spikes)
            )

            if self.output_mode in ("garrett", "vcd_grating", "ori_diff"):
                if self.output_mode == "garrett":
                    rid = self.network_meta.readout_ids["localized_readout_neuron_ids_0"]
                elif self.output_mode == "vcd_grating":
                    rid = self.network_meta.readout_ids["localized_readout_neuron_ids_1"]
                else:
                    rid = self.network_meta.readout_ids["localized_readout_neuron_ids_2"]
                out = bm.mean(output_spikes[..., rid], axis=-1)
                thresh = bm.zeros_like(out) + 0.01
                logits = bm.stack([thresh, out], axis=-1)
                return self.output_scale(logits)

            if self.output_mode == "evidence":
                rid1 = self.network_meta.readout_ids["localized_readout_neuron_ids_3"]
                rid2 = self.network_meta.readout_ids["localized_readout_neuron_ids_4"]
                out1 = bm.mean(output_spikes[..., rid1], axis=-1)
                out2 = bm.mean(output_spikes[..., rid2], axis=-1)
                logits = bm.stack([out1, out2], axis=-1)
                return self.output_scale(logits)

            if self.output_mode == "10class":
                outs = []
                for i in range(10):
                    rid = self.network_meta.readout_ids[
                        f"localized_readout_neuron_ids_{i + 5}"
                    ]
                    outs.append(bm.mean(output_spikes[..., rid], axis=-1))
                logits = bm.stack(outs, axis=-1)
                return self.output_scale(logits)

            raise ValueError(f"Unrecognized output_mode: {self.output_mode}")

        if self.lRout_pop != "all":
            rid = self.network_meta.laminar_indices[self.lRout_pop]
            out_pop_spikes = spikes[..., rid]
        else:
            out_pop_spikes = spikes

        output_all = self.output_head(out_pop_spikes)
        return output_all[..., self.output_indices]

    def update(self, inp):
        # inp shape: [batch, n_input]
        rnn_input = self.input_layer(inp)
        self.rsnn(rnn_input)
        spikes = self.rsnn.neurons.spike.value
        voltage = self.rsnn.neurons.V.value
        logits = self._readout_spikes(spikes)
        return logits, spikes, voltage

    def _time_reduce(self, output_seq):
        # output_seq shape: [batch, time, n_output]
        seq = int(output_seq.shape[1])
        n_chunk = seq // self.down_sample
        if n_chunk <= 0:
            raise ValueError(
                f"down_sample={self.down_sample} is too large for seq_len={seq}"
            )
        kept = n_chunk * self.down_sample
        output_seq = output_seq[:, :kept, :]
        output_seq = bm.reshape(
            output_seq, (output_seq.shape[0], n_chunk, self.down_sample, output_seq.shape[-1])
        )
        mean_output = bm.mean(output_seq, axis=2)
        return bm.softmax(mean_output, axis=-1)

    def readout_l2(self):
        if self.neuron_output or self.L2_factor <= 0:
            return bm.asarray(0.0)
        return self.L2_factor * 0.5 * bm.sum(self.output_head.W**2)

    def _reset_inplace(self):
        # Reset neuron core states without reallocating Variables.
        neu = self.rsnn.neurons
        neu.Iinp.value = bm.zeros_like(neu.Iinp.value)
        neu.Iasc1.value = bm.zeros_like(neu.Iasc1.value)
        neu.Iasc2.value = bm.zeros_like(neu.Iasc2.value)
        neu.spike.value = bm.zeros_like(neu.spike.value)
        neu.r.value = bm.zeros_like(neu.r.value)
        v_reset = bm.asarray(neu.v_reset)
        if neu.V.value.ndim > v_reset.ndim:
            v_reset = bm.broadcast_to(v_reset, neu.V.value.shape)
        neu.V.value = v_reset
        if hasattr(neu, "input"):
            neu.input.value = bm.zeros_like(neu.input.value)

        # Reset recurrent synapse states.
        self.rsnn.clear_input()
        for proj in self.rsnn.projs:
            proj.proj.syn.h.value = bm.zeros_like(proj.proj.syn.h.value)
            proj.proj.syn.g.value = bm.zeros_like(proj.proj.syn.g.value)

        # Reset receptor aggregation states.
        self.rsnn.neuron_receptors.syn.h.value = bm.zeros_like(
            self.rsnn.neuron_receptors.syn.h.value
        )
        self.rsnn.neuron_receptors.syn.g.value = bm.zeros_like(
            self.rsnn.neuron_receptors.syn.g.value
        )

    def forward(self, inputs, reset_state=True, full_output: Optional[bool] = None):
        """
        Run full sequence forward.

        Args:
            inputs: array of shape [batch, time, n_input].
            reset_state: reset model states before sequence.
            full_output: if True returns (pred, spikes, voltage).
        """
        x = bm.asarray(inputs)
        if x.ndim != 3:
            raise ValueError(f"inputs must have shape [batch, time, n_input], got {x.shape}")

        batch_size = int(x.shape[0])
        if batch_size != self.batch_size:
            raise ValueError(
                f"input batch size {batch_size} mismatches model batch_size {self.batch_size}"
            )
        if reset_state:
            self._reset_inplace()
        runner = bp.DSRunner(
            self,
            dt=1.0,
            progress_bar=False,
        )
        # DSRunner consumes [time, ...]. Current model supports batch_size=1,
        # so drop the singleton batch axis and restore it after run.
        x_time_first = x[0]
        logits_seq, spikes_seq, voltage_seq = runner.run(inputs=x_time_first)
        logits_seq = logits_seq[None, ...]
        spikes_seq = spikes_seq[None, ...]
        voltage_seq = voltage_seq[None, ...]

        pred = self._time_reduce(logits_seq)

        use_full = self.default_full_output if full_output is None else bool(full_output)
        if use_full:
            return pred, spikes_seq, voltage_seq
        return pred


class _InputLayerCompat(bp.DynSysGroup):
    """
    Compatibility input layer for current BrainPy API.

    Mirrors models.InputLayer behavior but avoids bp.init.parameter() signature
    assumptions from older versions.
    """

    def __init__(
        self,
        conn,
        weight,
        tau_syn,
        use_dale_law,
        bkg_weights,
        use_decoded_noise=False,
        noise_data=None,
    ):
        super().__init__()
        self._n_node_receptors = int(tau_syn.shape[0] * tau_syn.shape[-1])
        # Event CSR path can fail under some traced loops in current env.
        # Use value CSR matvec while keeping Dale sign constraints.
        self.input_proj = _CSRLinearValueDale(conn, weight, use_dale_law=use_dale_law)
        self.bkg_weights = bm.asarray(bkg_weights, dtype=bm.float_).reshape(-1)
        self._use_decoded_noise = bool(use_decoded_noise)
        self.noise_data = None if noise_data is None else bm.asarray(noise_data, dtype=bm.float_)

    def gen_noise(self, inp):
        if self._use_decoded_noise:
            if self.noise_data is None:
                raise ValueError("noise_data must be provided when use_decoded_noise=True")
            if inp.ndim == 1:
                noise_idx = bm.random.randint(0, self.noise_data.shape[0], size=(self._n_node_receptors,))
                return self.noise_data[noise_idx]
            noise_idx = bm.random.randint(
                0, self.noise_data.shape[0], size=(inp.shape[0], self._n_node_receptors)
            )
            return self.noise_data[noise_idx]
        if inp.ndim == 1:
            rest_of_brain = bm.random.binomial(10, 0.1) / 10.0
            return self.bkg_weights * rest_of_brain
        rest_of_brain = bm.random.binomial(10, 0.1, size=(inp.shape[0],)) / 10.0
        return self.bkg_weights * rest_of_brain[:, None]

    def update(self, inp):
        squeeze_singleton = inp.ndim > 1 and inp.shape[0] == 1
        inp_use = inp[0] if squeeze_singleton else inp
        input_current = self.input_proj(inp_use)
        input_current += self.gen_noise(inp_use)
        if squeeze_singleton:
            input_current = bm.expand_dims(input_current, axis=0)
        return input_current


class _CSRLinearValueDale(models.CSRLinearValue):
    """CSRLinearValue with optional Dale-law sign lock."""

    def __init__(self, conn, weight, use_dale_law=True, **kwargs):
        super().__init__(conn=conn, weight=weight, transpose=True, **kwargs)
        self.use_dale_law = bool(use_dale_law)
        self.sign = bp.init.parameter(
            bm.asarray(weight) > 0, (self.indices.size,)
        )

    @property
    def weight_dale(self):
        if self.use_dale_law:
            return bm.where(self.sign, bm.relu(self.weight), -bm.relu(-self.weight))
        return self.weight

    def update(self, x):
        if x.ndim == 1:
            return bm.sparse.csrmv(
                self.weight_dale,
                self.indices,
                self.indptr,
                x,
                shape=(self.conn.pre_num, self.conn.post_num),
                transpose=self.transpose,
            )
        if x.ndim > 1:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = jax.vmap(self._batch_csrmv_val)(x)
            return bm.reshape(y, shapes + (y.shape[-1],))
        raise ValueError

    def _batch_csrmv_val(self, x):
        return bm.sparse.csrmv(
            self.weight_dale,
            self.indices,
            self.indptr,
            x,
            shape=(self.conn.pre_num, self.conn.post_num),
            transpose=self.transpose,
        )


class _BillehColumnCompat(models.BillehColumn):
    """
    Compatibility wrapper that skips EPSC/IPSC bookkeeping.

    The bookkeeping path in models.BillehColumn.update currently assumes
    non-batched shapes and can break batched forward. Core neuron/synapse
    dynamics are preserved.
    """

    def update(self, x=None):
        for proj in self.projs:
            proj()
        if x is None:
            x = 0.0
        if self.default_input_to_receptor:
            self.receptor_input += x
            x = self.neuron_receptors(self.receptor_input)
        self.neurons(x)


def create_model(
    network,
    input_population,
    bkg_weights,
    seq_len=100,
    n_input=10,
    n_output=2,
    dtype=None,
    down_sampled_decode_noise_path=None,
    input_weight_scale=1.0,
    gauss_std=0.5,
    dampening_factor=0.2,
    train_recurrent=True,
    train_input=True,
    neuron_output=False,
    lRout_pop="all",
    L2_factor=0.0,
    return_state=False,
    down_sample=50,
    use_decoded_noise=False,
    max_delay=5,
    batch_size=None,
    full_output=False,
    output_mode="garrett",
    neuron_model="GLIF3",
    use_dale_law=True,
    scale=(1, 1),
    _return_interal_variables=False,
):
    """
    Build a BrainPy classification model from Billeh network objects.

    Notes:
    - `return_state`, `train_recurrent`, `train_input`, `dtype`, and `scale` are kept for
      signature compatibility with the tensorflow version.
    - Forward pass is available via returned model's `forward(inputs)` method.
    """
    del (
        n_input,
        dtype,
        input_weight_scale,
        gauss_std,
        train_recurrent,
        train_input,
        return_state,
        scale,
        _return_interal_variables,
    )
    if neuron_model != "GLIF3":
        raise ValueError("Not supported neuron model!")
    if output_mode not in _OUTPUT_MODE_TO_INDEX:
        raise ValueError(f"Unrecognized output_mode: {output_mode}")
    if batch_size is None:
        batch_size = 1
    if int(batch_size) != 1:
        raise NotImplementedError(
            "Current brainpy_impl/models.py is not batch-safe. "
            "Please use batch_size=1 in create_model()."
        )

    expected_out = 10 if output_mode == "10class" else 2
    if int(n_output) != expected_out:
        raise ValueError(
            f"n_output={n_output} mismatches output_mode='{output_mode}' (expected {expected_out})"
        )

    net = _normalize_network(network)
    dt = 1.0
    node_params = NodeParams.from_network_node_params(
        net.node_params,
        net.node_type_ids,
        dt=dt,
    )
    syn_params = SynapseParams.from_network_synapses(
        net.synapses,
        n_nodes=net.n_nodes,
        n_edges=net.n_edges,
        n_receptors=node_params.n_receptors,
        max_delay=max_delay,
        dt=dt,
    )
    input_population_norm = dict(input_population)
    if "delays" not in input_population_norm:
        # Some loaders drop delays after input-reduction. Keep TF/common assumption:
        # all input synapses share one-step delay.
        n_input_edges = int(_as_np(input_population_norm["weights"]).shape[0])
        input_population_norm["delays"] = np.ones(n_input_edges, dtype=np.float32)

    input_params = InputParams.from_input_node_bkg(
        input_population_norm,
        node_params,
        _as_np(bkg_weights),
    )

    laminar_indices = net.laminar_indices if net.laminar_indices else None
    if laminar_indices is None:
        laminar_indices = {"L23e": np.arange(net.n_nodes, dtype=np.int32)}
    rsnn = _BillehColumnCompat(
        node_params=node_params,
        syn_params=syn_params,
        use_dale_law=use_dale_law,
        default_input_to_receptor=True,
        batch_size=None,
        spk_reset="soft",
        laminar_indices=laminar_indices,
    )

    input_csr = to_brainpy_csr(
        input_params, split_receptor=False, split_conn=False
    )
    input_csr.eliminate_zeros()
    in_conn = bp.conn.SparseMatConn(input_csr != 0)
    in_weight = input_csr.data
    bkg = _as_np(input_params.bkg_weights, np.float32).reshape(
        node_params.n_nodes, node_params.n_receptors
    )

    noise_data = None
    if use_decoded_noise:
        if down_sampled_decode_noise_path is None:
            raise ValueError(
                "down_sampled_decode_noise_path must be provided when use_decoded_noise=True"
            )
        from scipy.io import loadmat

        noise_mat = loadmat(down_sampled_decode_noise_path)
        if "additive_noise" not in noise_mat:
            raise KeyError("decoded noise mat must contain key 'additive_noise'")
        noise_data = _as_np(noise_mat["additive_noise"], np.float32).reshape(-1)

    input_layer = _InputLayerCompat(
        conn=in_conn,
        weight=in_weight,
        tau_syn=node_params.tau_syn,
        use_dale_law=use_dale_law,
        bkg_weights=bkg,
        use_decoded_noise=use_decoded_noise,
        noise_data=noise_data,
    )

    return BillehClassificationModel(
        rsnn=rsnn,
        input_layer=input_layer,
        network_meta=net,
        seq_len=seq_len,
        down_sample=down_sample,
        n_output=n_output,
        output_mode=output_mode,
        neuron_output=neuron_output,
        lRout_pop=lRout_pop,
        dampening_factor=dampening_factor,
        full_output=full_output,
        L2_factor=L2_factor,
        batch_size=1,
    )
