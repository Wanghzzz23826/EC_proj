from dataclasses import dataclass
from typing import Union

import jax.numpy as jnp
import numpy as np
from jax import core as jcore
from jaxtyping import Array, Float, Int
from scipy.sparse import coo_matrix, csr_matrix

# an adapter interface that enforces typecheck with jaxtyping
# see jaxtyping.install_import_hook and common/__init__.py


def normalize_key_name(key):
    """Normalize the key name to lower case to suit Python's naming convention."""
    return {key.lower(): val for key, val in key.items()}


@dataclass
class NodeParams:
    dt: float
    n_nodes: int
    n_asc: int
    n_receptors: int
    v_th: Float[Array, "{self.n_nodes}"]  # noqa: F821
    g: Float[Array, "{self.n_nodes}"]  # noqa: F821
    e_l: Float[Array, "{self.n_nodes}"]  # noqa: F821
    k: Float[Array, "{self.n_nodes} {self.n_asc}"]  # noqa: F722
    tau: Float[Array, "{self.n_nodes}"]  # noqa: F821
    c_m: Float[Array, "{self.n_nodes}"]  # noqa: F821
    v_reset: Float[Array, "{self.n_nodes}"]  # noqa: F821
    t_ref: Float[Array, "{self.n_nodes}"]  # noqa: F821
    asc_amps: Float[Array, "{self.n_nodes} {self.n_asc}"]  # noqa: F722
    tau_syn: Float[Array, "{self.n_nodes} {n_receptors}"]  # noqa: F722
    voltage_scale: Float[Array, "{self.n_nodes}"] | None = None  # noqa: F821
    voltage_offset: Float[Array, "{self.n_nodes}"] | None = None  # noqa: F821

    # not needed for brainpy, where we can use ode solver directly
    @property
    def decay(self):
        return np.exp(-self.dt / self.tau)

    @property
    def c_m_discrete(self):
        return 1 / self.c_m * (1 - self.decay) * self.tau

    @property
    def syn_decay(self):
        return np.exp(-self.dt / self.tau_syn)

    @property
    def psc_initial(self):
        return np.e / self.tau_syn

    def __getitem__(self, indices):
        sliced_attrs = {
            key: (value[indices] if isinstance(value, np.ndarray) else value)
            for key, value in self.__dict__.items()
        }
        sliced_attrs["n_nodes"] = len(sliced_attrs["g"])
        return NodeParams(**sliced_attrs)

    @classmethod
    def from_network_node_params(
        cls,
        params,
        node_type_ids: Int[Array, "n_nodes"],  # noqa: F821
        dt: float,
        scaled: bool = False,
    ):
        # normalize the key name to Python's naming convention
        params = normalize_key_name(params)

        # normalize the parameters
        if scaled:
            voltage_scale = params["v_th"] - params["e_l"]
            voltage_offset = params["e_l"]
            v_th = (params["v_th"] - voltage_offset) / voltage_scale
            e_l = (params["e_l"] - voltage_offset) / voltage_scale
            v_reset = (params["v_reset"] - voltage_offset) / voltage_scale
            asc_amps = params["asc_amps"] / voltage_scale[..., None]
        else:
            voltage_scale = None
            voltage_offset = None
            v_th = params["v_th"]
            e_l = params["e_l"]
            v_reset = params["v_reset"]
            asc_amps = params["asc_amps"]
        n_asc = asc_amps.shape[-1]
        assert (
            n_asc == 2
        ), "Due to BrainPy exponential euler odeint limitation, n_asc is hard coded to 2"

        tau = params["c_m"] / params["g"]
        n_receptors = params["tau_syn"].shape[-1]

        # put in the new values
        normalized_params = {
            "v_th": v_th,
            "g": params["g"],
            "e_l": e_l,
            "k": params["k"],
            "tau": tau,
            "v_reset": v_reset,
            "t_ref": params["t_ref"],
            "asc_amps": asc_amps,
            "tau_syn": params["tau_syn"],
            "c_m": params["c_m"],
        }
        if scaled:
            normalized_params += {
                "voltage_scale": voltage_scale,
                "voltage_offset": voltage_offset,
            }

        n_nodes = len(node_type_ids)

        # apply node_type_ids
        normalized_params = {
            key: val[node_type_ids] for key, val in normalized_params.items()
        }

        return cls(
            **normalized_params,
            n_nodes=n_nodes,
            n_asc=n_asc,
            n_receptors=n_receptors,
            dt=dt,
        )


def is_all_same(arr):
    if np.isscalar(arr):
        return True

    return (arr == arr[0]).all()


def _is_tracer(x) -> bool:
    return isinstance(x, jcore.Tracer)


@dataclass
class SynapseParams:
    n_nodes: int
    n_edges: int
    n_receptors: int
    # [post_node*n_receptor pre_node]
    indices: Int[Array, "{self.n_edges} 2"]  # noqa: F722, F821
    weights: Float[Array, "{self.n_edges}"]  # noqa: F821
    delays: Int[Array, "{self.n_edges}"] | int  # noqa: F821
    dense_shape: tuple[int, int]  # (n_receptors*n_nodes , n_nodes)
    max_delay: float
    dt: float

    @classmethod
    def from_network_synapses(
        self,
        synapses,
        n_nodes: int,
        n_edges: int,
        n_receptors: int,
        max_delay: int,
        dt: float,
    ):
        max_delay = min(round(max(synapses["delays"])), max_delay)
        indices = synapses["indices"]
        delays = np.round(np.clip(synapses["delays"], dt, max_delay) / dt).astype(int)
        delays = (
            delays - 1
        )  # delay starts from 1, but at runtime, we need to start from 0

        return SynapseParams(
            n_nodes=n_nodes,
            n_edges=n_edges,
            n_receptors=n_receptors,
            indices=indices,
            delays=delays,
            weights=synapses["weights"],
            dense_shape=synapses["dense_shape"],
            max_delay=max_delay,
            dt=dt,
        )

    def split_by_delays(
        self,
    ) -> list["SynapseParams"]:
        delays, delay_groups = np.unique(self.delays, return_inverse=True)
        ret = []
        for i, d in enumerate(delays):
            group_sel = delay_groups == i
            indices = self.indices[group_sel, :]
            weights = self.weights[group_sel]
            ret.append(
                SynapseParams(
                    n_nodes=self.n_nodes,
                    n_edges=indices.shape[0],
                    n_receptors=self.n_receptors,
                    indices=indices,
                    delays=d,
                    weights=weights,
                    dense_shape=self.dense_shape,
                    max_delay=self.max_delay,
                    dt=self.dt,
                )
            )
        return ret

    def report_post_stats(
        self,
        indices_readout: Int[Array, "n_readout"] = np.arange(32, dtype=int),  # noqa: F821
    ):
        num_readout_synapses = 0
        num_synapses_per_neuron = np.zeros(len(indices_readout), np.int32)
        post_indices, _ = self.indices.T
        post_indices = post_indices // self.n_receptors
        for post_ind in post_indices:
            if post_ind in indices_readout:
                num_readout_synapses += 1
                num_synapses_per_neuron[post_ind] += 1
        print(f"> Readout synapses {num_readout_synapses}")
        print(f"> Synapses per readout neuron {np.mean(num_synapses_per_neuron):.1f}")


@dataclass
class InputParams:
    n_input_nodes: int
    n_nodes: int
    n_receptors: int
    dense_shape: tuple[int, int]  # (n_nodes * n_receptors, n_input_nodes)
    # [post_node*receptor pre_node]
    indices: Int[Array, "n_input_edges 2"]  # noqa: F722, F821
    weights: Float[Array, "n_input_edges"]  # noqa: F821

    bkg_weights: Int[Array, "{self.n_nodes}*{self.n_receptors}"]  # noqa: F821

    @classmethod
    def from_input_node_bkg(
        cls, input_population, node_params: NodeParams, bkg_weights
    ):
        input_weights = input_population["weights"]
        input_indices = input_population["indices"]
        n_nodes = node_params.n_nodes
        n_receptors = node_params.n_receptors
        if node_params.voltage_scale is not None:
            input_weights = (
                input_weights
                / node_params.voltage_scale[input_indices[:, 0] // n_receptors]
            )
        return cls(
            n_nodes=n_nodes,
            n_input_nodes=input_population["n_inputs"],
            n_receptors=n_receptors,
            dense_shape=(n_nodes * n_receptors, input_population["n_inputs"]),
            indices=input_indices,
            weights=input_weights,
            bkg_weights=bkg_weights,
        )


def split_receptor_types(
    indices: Int[Array, "n_edges 2"],  # noqa: F722
    n_receptors: int,  # noqa: F722
) -> (Int[Array, "n_edges"], Int[Array, "n_edges"]):  # noqa: F821
    return np.divmod(indices[:, 0], n_receptors)


def to_csr(
    dense_shape: tuple[int, int],
    indices: Float[Array, "n_edges 2"],  # noqa: F722
    weights: Float[Array, "n_edges"],  # noqa: F821
) -> coo_matrix:
    return coo_matrix((weights, indices.T), shape=dense_shape).tocsr()


def to_brainpy_csr(
    params: Union[SynapseParams, InputParams],
    split_receptor: bool = False,
    split_conn: bool = True,
    zero_tol: float = 0.0,
    drop_zero_in_tracer: bool = False,
) -> tuple[csr_matrix, Float[Array, "n_input_edges"]] | csr_matrix:  # noqa: F821
    # from coo to csr
    # from (post*receptor, pre) to (pre, post)
    indices = params.indices
    if split_receptor:
        indices = indices.copy()
        indices[:, 0], _ = split_receptor_types(params.indices, params.n_receptors)
        dense_shape = (params.dense_shape[1], params.n_nodes)
    else:
        dense_shape = (params.dense_shape[1], params.dense_shape[0])

    ij = np.asarray(indices[:, ::-1], dtype=np.int64)
    weights = params.weights
    # print(f"before: weight length {weights.size} ")

    if _is_tracer(weights) or split_conn:
        if not split_conn:
            raise ValueError("split_conn=False is not supported for traced weights.")
        if drop_zero_in_tracer:
            raise ValueError(
                "drop_zero_in_tracer=True is not supported for traced weights. "
                "Dynamic edge deletion is unsafe under vmap/jit."
            )

        n_cols = int(dense_shape[1])
        key = ij[:, 0] * n_cols + ij[:, 1]
        uniq_key, inv = np.unique(key, return_inverse=True)

        rows = (uniq_key // n_cols).astype(np.int32)
        cols = (uniq_key % n_cols).astype(np.int32)
        conn_csr = coo_matrix(
            (np.ones((rows.shape[0],), dtype=np.float32), (rows, cols)),
            shape=dense_shape,
        ).tocsr()

        inv_j = jnp.asarray(inv, dtype=jnp.int32)
        w_j = jnp.asarray(weights)
        w_agg = jnp.zeros((uniq_key.shape[0],), dtype=w_j.dtype).at[inv_j].add(w_j)
        if zero_tol > 0.0:
            tol = jnp.asarray(zero_tol, dtype=w_agg.dtype)
            w_agg = jnp.where(jnp.abs(w_agg) <= tol, jnp.zeros_like(w_agg), w_agg)

        conn_csr = conn_csr != 0
        assert w_agg.shape[0] == conn_csr.nnz, (
            f"Traced CSR mismatch: weight length {w_agg.shape[0]} "
            f"!= conn nnz {conn_csr.nnz}"
        )
        return conn_csr, w_agg

    csr_mat = to_csr(
        dense_shape,
        ij,
        weights,
    )
    # conn, weight
    if split_conn:
        # Keep conn structure and weight array aligned when explicit zero
        # weights exist in sparse data.
        csr_mat = csr_mat.copy()
        csr_mat.eliminate_zeros()
        if zero_tol > 0.0:
            small = np.abs(csr_mat.data) <= zero_tol
            if np.any(small):
                csr_mat.data[small] = 0.0
                csr_mat.eliminate_zeros()

        conn_csr = csr_mat != 0
        assert csr_mat.data.size == conn_csr.nnz, (
            f"CSR mismatch: weight length {csr_mat.data.size} "
            f"!= conn nnz {conn_csr.nnz}"
        )
        # print(f"after: weight length {csr_mat.data.size} ")
        return csr_mat, csr_mat.data
    else:
        return csr_mat


@dataclass
class OutputParams:
    task_dense_readout_map: dict[str, Int[Array, "_"]]  # noqa: F821
    task_neuron_readout_map: dict[str, Int[Array, "_"] | list[Int[Array, "_"]]]  # noqa: F821

    laminar_indices: Int[Array, "_"] | None  # noqa: F821

    @classmethod
    def from_network(cls, network):
        task_dense_readout_map = {
            "garrett": np.array([0, 1]),
            "vcd_grating": np.array([2, 3]),
            "ori_diff": np.array([4, 5]),
            "evidence": np.array([6, 7]),
            "10class": np.arange(8, 18),
        }
        task_neuron_readout_map = {
            "garrett": network["localized_readout_neuron_ids_0"],
            "vcd_grating": network["localized_readout_neuron_ids_1"],
            "ori_diff": network["localized_readout_neuron_ids_2"],
            "evidence": [
                network["localized_readout_neuron_ids_3"],
                network["localized_readout_neuron_ids_4"],
            ],
            "10class": [
                network[f"localized_readout_neuron_ids_{i + 5}"] for i in range(10)
            ],
        }
        return cls(
            task_dense_readout_map=task_dense_readout_map,
            task_neuron_readout_map=task_neuron_readout_map,
            laminar_indices=network["laminar_indices"],
        )

    def total_readout_size(self, neuron_out: bool):
        def max_indice(readout_map: dict):
            return np.max((v.max() for v in readout_map.values()))

        if neuron_out:
            return max_indice(self.task_neuron_readout_map)
        else:
            return max_indice(self.task_dense_readout_map)