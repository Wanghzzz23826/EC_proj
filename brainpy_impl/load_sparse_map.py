"""LGN 输入通道提取与重映射工具。

这个模块不修改原始 load_sparse.py 的行为，只额外提供：
1. 从 input_population 中提取某一组 V1 神经元真正使用到的 LGN 通道；
2. 构建 old LGN index -> new compact index 的可复现映射；
3. 对 input_population 和 LGN rates 应用同一份映射。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover - 允许在无 JAX 环境下做纯 numpy 映射
    jnp = None


def sort_sparse_indices(indices, *arrays):
    """按 (post, pre) 排序稀疏索引，并同步重排所有附带数组。"""
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(f"indices must have shape (N, 2), got {indices.shape}")

    n_edges = indices.shape[0]
    for a in arrays:
        if len(a) != n_edges:
            raise ValueError(
                f"Length mismatch: indices has {n_edges}, side array has {len(a)}"
            )
    if n_edges == 0:
        return (indices,) + arrays

    max_ind = np.max(indices) + 1
    key = indices[:, 0].astype(np.int64) * np.int64(max_ind) + indices[:, 1].astype(
        np.int64
    )
    order = np.argsort(key)
    return (indices[order],) + tuple(a[order] for a in arrays)


def _is_jax_array(x) -> bool:
    module_name = type(x).__module__
    return module_name.startswith("jax") or module_name.startswith("jaxlib")


def _to_like(reference, array, dtype=None):
    if jnp is not None and reference is not None and _is_jax_array(reference):
        return jnp.asarray(array, dtype=dtype)
    return np.asarray(array, dtype=dtype)


def _as_sorted_unique_int_array(values: Iterable[int], name: str) -> np.ndarray:
    if isinstance(values, set):
        values = list(values)
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if np.any(arr < 0):
        raise ValueError(f"{name} contains negative indices: {arr[arr < 0][:8]}")
    return np.unique(arr)


def _normalize_lgn_mapping(
    external_mapping: Mapping[int, int] | Iterable[int],
    *,
    n_old_input: int | None = None,
) -> tuple[Dict[int, int], np.ndarray]:
    """统一 old LGN -> new compact index 的映射格式。"""
    if external_mapping is None:
        raise ValueError("external_mapping must not be None")

    if isinstance(external_mapping, Mapping):
        if not external_mapping:
            return {}, np.zeros((0,), dtype=np.int64)
        old = np.asarray(list(external_mapping.keys()), dtype=np.int64)
        new = np.asarray(list(external_mapping.values()), dtype=np.int64)
    else:
        old = _as_sorted_unique_int_array(external_mapping, "external_mapping")
        new = np.arange(old.size, dtype=np.int64)

    if old.size != np.unique(old).size:
        raise ValueError("external_mapping contains duplicated old LGN indices")
    if new.size != np.unique(new).size:
        raise ValueError("external_mapping contains duplicated new LGN indices")
    if old.size == 0:
        return {}, old
    if np.any(new < 0):
        raise ValueError("external_mapping contains negative new indices")

    expected_new = np.arange(old.size, dtype=np.int64)
    if not np.array_equal(np.sort(new), expected_new):
        raise ValueError(
            "external_mapping must map to a compact 0..N-1 index space without gaps"
        )
    if n_old_input is not None and np.any(old >= int(n_old_input)):
        raise ValueError(
            f"external_mapping contains LGN index >= n_old_input ({n_old_input})"
        )

    compact_mapping = {
        int(old_idx): int(new_idx)
        for old_idx, new_idx in sorted(
            zip(old.tolist(), new.tolist()), key=lambda x: x[1]
        )
    }
    used_lgn_channels = np.asarray(list(compact_mapping.keys()), dtype=np.int64)
    return compact_mapping, used_lgn_channels


def extract_used_lgn_channels(input_population, selected_neuron_ids):
    """
    提取一组 V1 神经元实际使用到的 LGN 输入通道。

    当前 input_population['indices'] 的真实语义是：
        indices[:, 0] = 4 * post_neuron_id + receptor_id
        indices[:, 1] = pre_lgn_channel_id
    所以这里会先用 indices[:, 0] // 4 还原 post neuron，再取第 2 列的 LGN 通道。
    """
    in_ind = np.asarray(input_population["indices"], dtype=np.int64)
    if in_ind.ndim != 2 or in_ind.shape[1] != 2:
        raise ValueError(f"indices must have shape (N, 2), got {in_ind.shape}")

    selected_neuron_ids = _as_sorted_unique_int_array(
        selected_neuron_ids, "selected_neuron_ids"
    )
    if selected_neuron_ids.size == 0 or in_ind.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    post_neuron_ids = in_ind[:, 0] // 4
    keep = np.isin(post_neuron_ids, selected_neuron_ids)
    if not np.any(keep):
        return np.zeros((0,), dtype=np.int64)
    return np.unique(in_ind[keep, 1].astype(np.int64, copy=False))


def build_lgn_reduction_map(used_lgn_channels):
    """构建 old LGN index -> new compact index 的一一映射。"""
    used_lgn_channels = _as_sorted_unique_int_array(
        used_lgn_channels, "used_lgn_channels"
    )
    return {
        int(old_idx): int(new_idx)
        for new_idx, old_idx in enumerate(used_lgn_channels)
    }


def remap_input_population(input_population, mapping):
    """
    保留映射中出现的 LGN 输入边，并把 LGN 索引压缩到新的紧凑编号。

    返回的字典会尽量保持原数组类型：
    - numpy 输入返回 numpy
    - jax 输入返回 jax array
    """
    compact_mapping, used_lgn_channels = _normalize_lgn_mapping(
        mapping, n_old_input=input_population["n_inputs"]
    )
    new_n_input = len(compact_mapping)

    indices_ref = input_population["indices"]
    weights_ref = input_population["weights"]

    in_ind = np.asarray(indices_ref, dtype=np.int64)
    in_weights = np.asarray(weights_ref, dtype=np.float32)
    if in_ind.ndim != 2 or in_ind.shape[1] != 2:
        raise ValueError(f"indices must have shape (N, 2), got {in_ind.shape}")
    if in_ind.shape[0] != in_weights.shape[0]:
        raise ValueError(
            f"indices/weights size mismatch: {in_ind.shape[0]} vs {in_weights.shape[0]}"
        )

    lookup = np.full(int(input_population["n_inputs"]), -1, dtype=np.int64)
    if new_n_input > 0:
        lookup[used_lgn_channels] = np.arange(new_n_input, dtype=np.int64)

    if in_ind.shape[0] == 0 or new_n_input == 0:
        out = dict(
            n_inputs=new_n_input,
            indices=_to_like(indices_ref, np.zeros((0, 2), dtype=np.int64), dtype=np.int32),
            weights=_to_like(weights_ref, np.zeros((0,), dtype=np.float32), dtype=np.float32),
        )
    else:
        keep = lookup[in_ind[:, 1].astype(np.int64, copy=False)] >= 0
        kept_ind = in_ind[keep].copy()
        kept_w = in_weights[keep].copy()
        kept_ind[:, 1] = lookup[kept_ind[:, 1].astype(np.int64, copy=False)]

        side_arrays = [kept_w]
        has_delays = "delays" in input_population and input_population["delays"] is not None
        if has_delays:
            delays_ref = input_population["delays"]
            delays = np.asarray(delays_ref, dtype=np.float32)
            if delays.shape[0] != in_ind.shape[0]:
                raise ValueError(
                    f"indices/delays size mismatch: {in_ind.shape[0]} vs {delays.shape[0]}"
                )
            side_arrays.append(delays[keep].copy())

        sorted_payload = sort_sparse_indices(kept_ind, *side_arrays)
        out = dict(
            n_inputs=new_n_input,
            indices=_to_like(indices_ref, sorted_payload[0], dtype=np.int32),
            weights=_to_like(weights_ref, sorted_payload[1], dtype=np.float32),
        )
        if has_delays:
            out["delays"] = _to_like(
                input_population["delays"], sorted_payload[2], dtype=np.float32
            )

    if "spikes" in input_population:
        spikes = input_population["spikes"]
        if spikes is None:
            out["spikes"] = None
        else:
            spikes_np = np.asarray(spikes)
            if spikes_np.ndim == 2 and spikes_np.shape[1] == int(input_population["n_inputs"]):
                out["spikes"] = _to_like(spikes, spikes_np[:, used_lgn_channels], dtype=np.float32)
            else:
                out["spikes"] = spikes

    if "spike_times" in input_population and "spike_ids" in input_population:
        spike_ids_ref = input_population["spike_ids"]
        spike_times_ref = input_population["spike_times"]
        spike_ids = np.asarray(spike_ids_ref, dtype=np.int64)
        spike_times = np.asarray(spike_times_ref, dtype=np.int32)
        keep_spikes = lookup[spike_ids] >= 0
        out["spike_ids"] = _to_like(
            spike_ids_ref, lookup[spike_ids[keep_spikes]], dtype=np.int32
        )
        out["spike_times"] = _to_like(
            spike_times_ref, spike_times[keep_spikes], dtype=np.int32
        )
        if "n_steps" in input_population:
            out["n_steps"] = input_population["n_steps"]

    return out


def reduce_input_population_with_mapping(input_population, mapping):
    """
    显式映射版 reduction。

    作用等价于“如果原 reduce_input_population 支持 external_mapping，就按该映射压缩”，
    但这里单独放在独立模块中，不改原始 load_sparse.py。
    """
    return remap_input_population(input_population, mapping)


def reduce_lgn_rates(rates, used_lgn_channels):
    """按选中的 LGN 通道子集裁剪缓存的 rates。"""
    rates = np.asarray(rates)
    used_lgn_channels = _as_sorted_unique_int_array(
        used_lgn_channels, "used_lgn_channels"
    )
    if rates.ndim != 2:
        raise ValueError(f"rates must have shape (T, N), got {rates.shape}")
    return rates[:, used_lgn_channels]


def _load_mapping_payload(lgn_mapping):
    """支持 dict / indices array / .npy path 三种输入。"""
    if lgn_mapping is None:
        return None
    if isinstance(lgn_mapping, (str, Path)):
        path = Path(lgn_mapping).expanduser().resolve()
        if path.suffix != ".npy":
            raise ValueError(
                f"Expected .npy file for lgn_mapping path, got {path}"
            )
        return np.load(path)
    return lgn_mapping


def _annotate_network_with_lgn_mapping(network, selected_indices, remapped_input_population):
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    compact_pre = np.asarray(remapped_input_population["indices"], dtype=np.int64)
    compact_pre = (
        np.unique(compact_pre[:, 1]) if compact_pre.ndim == 2 and compact_pre.size else np.zeros((0,), dtype=np.int64)
    )
    disconnected = np.setdiff1d(
        np.arange(selected_indices.size, dtype=np.int64), compact_pre, assume_unique=True
    )

    network["lgn_selected_indices"] = (
        jnp.asarray(selected_indices, dtype=jnp.int32)
        if jnp is not None
        else selected_indices.astype(np.int32, copy=False)
    )
    network["lgn_connected_compact_indices"] = (
        jnp.asarray(compact_pre, dtype=jnp.int32)
        if jnp is not None
        else compact_pre.astype(np.int32, copy=False)
    )
    network["lgn_disconnected_compact_indices"] = (
        jnp.asarray(disconnected, dtype=jnp.int32)
        if jnp is not None
        else disconnected.astype(np.int32, copy=False)
    )
    network["lgn_mapping_size"] = int(selected_indices.size)
    network["lgn_connected_count"] = int(compact_pre.size)
    network["lgn_disconnected_count"] = int(disconnected.size)
    return network


def reduce_lgn_cache(rates, lgn_mapping):
    """按同一份 lgn_mapping 裁剪 LGN cache，保证 cache 与 connectivity 对齐。"""
    mapping_payload = _load_mapping_payload(lgn_mapping)
    compact_mapping, selected_indices = _normalize_lgn_mapping(mapping_payload)
    _ = compact_mapping
    return reduce_lgn_rates(rates, selected_indices)


def load_sparse_map(
    n_input,
    n_neurons,
    core_only,
    data_dir,
    seed=3000,
    connected_selection=False,
    n_output=2,
    neurons_per_output=16,
    use_rand_ini_w=False,
    use_dale_law=True,
    use_rand_connectivity=False,
    use_uniform_neuron_type=False,
    use_only_one_type=False,
    scale_w_e=-1,
    localized_readout=True,
    TD_input=False,
    n_TD_input=None,
    targets=None,
    lgn_mapping=None,
):
    """
    Reduced LGN pipeline.

    - lgn_mapping is None: 保持和 baseline load_sparse 一致
    - lgn_mapping is provided: 固定使用这份 old LGN -> compact LGN 映射
      绝不调用随机 reduction
    """
    try:
        from . import load_sparse as _baseline_load_sparse
    except ImportError:
        from brainpy_impl import load_sparse as _baseline_load_sparse

    if lgn_mapping is None:
        return _baseline_load_sparse.load_billeh(
            n_input=n_input,
            n_neurons=n_neurons,
            core_only=core_only,
            data_dir=data_dir,
            seed=seed,
            connected_selection=connected_selection,
            n_output=n_output,
            neurons_per_output=neurons_per_output,
            use_rand_ini_w=use_rand_ini_w,
            use_dale_law=use_dale_law,
            use_rand_connectivity=use_rand_connectivity,
            use_uniform_neuron_type=use_uniform_neuron_type,
            use_only_one_type=use_only_one_type,
            scale_w_e=scale_w_e,
            localized_readout=localized_readout,
            TD_input=TD_input,
            n_TD_input=n_TD_input,
            targets=targets,
        )

    mapping_payload = _load_mapping_payload(lgn_mapping)
    compact_mapping, selected_indices = _normalize_lgn_mapping(mapping_payload)
    target_n_input = len(selected_indices)
    if int(n_input) != target_n_input:
        raise ValueError(
            f"n_input={n_input} does not match lgn_mapping size={target_n_input}"
        )

    # 关键点：这里强制先加载 full LGN (17400)，完全绕开 baseline 中的随机 reduction。
    if TD_input:
        td_inputs, input_population_full, network, bkg_weights = _baseline_load_sparse.load_billeh(
            n_input=17400,
            n_neurons=n_neurons,
            core_only=core_only,
            data_dir=data_dir,
            seed=seed,
            connected_selection=connected_selection,
            n_output=n_output,
            neurons_per_output=neurons_per_output,
            use_rand_ini_w=use_rand_ini_w,
            use_dale_law=use_dale_law,
            use_rand_connectivity=use_rand_connectivity,
            use_uniform_neuron_type=use_uniform_neuron_type,
            use_only_one_type=use_only_one_type,
            scale_w_e=scale_w_e,
            localized_readout=localized_readout,
            TD_input=True,
            n_TD_input=n_TD_input,
            targets=targets,
        )
        remapped_input_population = remap_input_population(
            input_population_full, compact_mapping
        )
        network = _annotate_network_with_lgn_mapping(
            network, selected_indices, remapped_input_population
        )
        return td_inputs, remapped_input_population, network, bkg_weights

    input_population_full, network, bkg_weights = _baseline_load_sparse.load_billeh(
        n_input=17400,
        n_neurons=n_neurons,
        core_only=core_only,
        data_dir=data_dir,
        seed=seed,
        connected_selection=connected_selection,
        n_output=n_output,
        neurons_per_output=neurons_per_output,
        use_rand_ini_w=use_rand_ini_w,
        use_dale_law=use_dale_law,
        use_rand_connectivity=use_rand_connectivity,
        use_uniform_neuron_type=use_uniform_neuron_type,
        use_only_one_type=use_only_one_type,
        scale_w_e=scale_w_e,
        localized_readout=localized_readout,
        TD_input=False,
        n_TD_input=n_TD_input,
        targets=targets,
    )

    remapped_input_population = remap_input_population(input_population_full, compact_mapping)
    network = _annotate_network_with_lgn_mapping(
        network, selected_indices, remapped_input_population
    )
    return remapped_input_population, network, bkg_weights


def load_billeh(*args, **kwargs):
    """与 baseline 接口对齐的别名，方便 reduced pipeline 直接替换调用。"""
    return load_sparse_map(*args, **kwargs)
