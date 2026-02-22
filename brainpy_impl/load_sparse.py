import os

import numpy as np
import pickle as pkl
import h5py
import jax
import jax.numpy as jnp
import pandas as pd


def jax_sort_indices(indices: jnp.ndarray, weights: jnp.ndarray):
    # indices shape: (n_synapses, 2)
    max_ind = jnp.max(indices) + 1
    q = indices[:, 0] * max_ind + indices[:, 1]
    sorted_idx = jnp.argsort(q)
    return indices[sorted_idx], weights[sorted_idx]


def sort_sparse_indices(indices, *arrays):
    """Sort sparse (post, pre) indices and apply the same order to side arrays."""
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

def sort_synapses(pre, post, weight, delay, receptor):
    order = jnp.lexsort((pre, post, receptor))
    return (pre[order], post[order],
            weight[order], delay[order], receptor[order])

def load_network(
    path='/home/wanghz/EC_V1/data/GLIF_V1_network/network_dat.pkl',
    h5_path='/home/wanghz/EC_V1/data/GLIF_V1_network/network/v1_nodes.h5',
    data_dir='.',
    core_only=True,
    n_neurons=None,
    seed=3000,
    connected_selection=False,
    use_rand_ini_w=False,
    use_dale_law=True,
    use_rand_connectivity=False,
    use_uniform_neuron_type=False,
    use_only_one_type=False,
    scale_w_e=-1
):
    rd = np.random.RandomState(seed=seed)

    with open(path, "rb") as f:
        d = pkl.load(f)

    n_all_nodes = sum(len(a["ids"]) for a in d["nodes"])

    # Initialize full-id mappings (tensorflow_impl behavior).
    bmtk_id_to_tf_id = np.arange(n_all_nodes)
    tf_id_to_bmtk_id = np.arange(n_all_nodes)

    edges = d["edges"]
    h5_file = h5py.File(h5_path, "r")
    assert np.diff(h5_file["nodes"]["v1"]["node_id"]).var() < 1e-12
    x_all = np.array(h5_file["nodes"]["v1"]["0"]["x"])
    y_all = np.array(h5_file["nodes"]["v1"]["0"]["y"])
    z_all = np.array(h5_file["nodes"]["v1"]["0"]["z"])
    r_all = np.sqrt(x_all ** 2 + z_all ** 2)

    if connected_selection:
        sorted_ind = np.argsort(r_all)
        sel_mask = np.zeros(n_all_nodes, bool)
        sel_mask[sorted_ind[:n_neurons]] = True
        print(f"> Maximum sample radius: {r_all[sorted_ind[n_neurons - 1]]:.2f}")
    elif core_only:
        sel_mask = r_all < 400
        if n_neurons is not None and n_neurons > 0:
            inds = np.where(sel_mask)[0]
            take_inds = rd.choice(inds, size=n_neurons, replace=False)
            sel_mask[:] = False
            sel_mask[take_inds] = True
    elif n_neurons is not None and n_neurons > 0:
        take_inds = rd.choice(np.arange(n_all_nodes), size=n_neurons, replace=False)
        sel_mask = np.zeros(n_all_nodes, bool)
        sel_mask[take_inds] = True
    else:
        sel_mask = np.ones(n_all_nodes, bool)

    n_nodes = int(np.sum(sel_mask))
    tf_id_to_bmtk_id = tf_id_to_bmtk_id[sel_mask]
    bmtk_id_to_tf_id = np.zeros_like(bmtk_id_to_tf_id) - 1
    for tf_id, bmtk_id in enumerate(tf_id_to_bmtk_id):
        bmtk_id_to_tf_id[bmtk_id] = tf_id

    x = x_all[sel_mask]
    y = y_all[sel_mask]
    z = z_all[sel_mask]

    n_edges = 0
    for edge in edges:
        target_tf_ids = bmtk_id_to_tf_id[np.array(edge["target"])]
        source_tf_ids = bmtk_id_to_tf_id[np.array(edge["source"])]
        edge_exists = np.logical_and(target_tf_ids >= 0, source_tf_ids >= 0)
        n_edges += np.sum(edge_exists)

    print(f"> Number of Neurons: {n_nodes}")
    print(f"> Number of Synapses: {n_edges}")

    n_node_types = len(d["nodes"])
    node_params = dict(
        V_th=np.zeros(n_node_types, np.float32),
        g=np.zeros(n_node_types, np.float32),
        E_L=np.zeros(n_node_types, np.float32),
        k=np.zeros((n_node_types, 2), np.float32),
        C_m=np.zeros(n_node_types, np.float32),
        V_reset=np.zeros(n_node_types, np.float32),
        tau_syn=np.zeros((n_node_types, 4), np.float32),
        t_ref=np.zeros(n_node_types, np.float32),
        asc_amps=np.zeros((n_node_types, 2), np.float32),
    )
    node_type_ids = np.zeros(n_nodes, np.int64)
    for i, node_type in enumerate(d["nodes"]):
        tf_ids = bmtk_id_to_tf_id[np.array(node_type["ids"])]
        tf_ids = tf_ids[tf_ids >= 0]
        node_type_ids[tf_ids] = i
        for k, v in node_params.items():
            v[i] = node_type["params"][k]

    if use_uniform_neuron_type:
        node_params = dict(
            V_th=np.zeros(n_node_types, np.float32),
            g=np.zeros(n_node_types, np.float32),
            E_L=np.zeros(n_node_types, np.float32),
            k=np.zeros((n_node_types, 2), np.float32),
            C_m=np.zeros(n_node_types, np.float32),
            V_reset=np.zeros(n_node_types, np.float32),
            tau_syn=np.zeros((n_node_types, 4), np.float32),
            t_ref=np.zeros(n_node_types, np.float32),
            asc_amps=np.zeros((n_node_types, 2), np.float32),
        )
        df = pd.read_csv(os.path.join(data_dir, "network/v1_node_types.csv"), delimiter=" ")
        for i, node_type in enumerate(d["nodes"]):
            tf_ids = bmtk_id_to_tf_id[np.array(node_type["ids"])]
            tf_ids = tf_ids[tf_ids >= 0]
            if use_only_one_type:
                for k, v in node_params.items():
                    v[i] = d["nodes"][19]["params"][k]  # e23Cux2
            else:
                if df.iloc[i]["pop_name"].startswith("e"):
                    for k, v in node_params.items():
                        v[i] = d["nodes"][18]["params"][k]  # e23Cux2
                elif df.iloc[i]["pop_name"].startswith("i"):
                    for k, v in node_params.items():
                        v[i] = d["nodes"][23]["params"][k]  # i23Pvalb
                else:
                    raise ValueError("It is neither excitatory nor inhibitory; check data")

    dense_shape = (4 * n_nodes, n_nodes)
    indices = np.zeros((n_edges, 2), dtype=np.int64)
    weights = np.zeros((n_edges,), dtype=np.float32)
    delays = np.zeros((n_edges,), dtype=np.float32)

    current_edge = 0
    for edge in edges:
        r = edge["params"]["receptor_type"] - 1
        target_tf_ids = bmtk_id_to_tf_id[np.array(edge["target"])]
        source_tf_ids = bmtk_id_to_tf_id[np.array(edge["source"])]
        edge_exists = np.logical_and(target_tf_ids >= 0, source_tf_ids >= 0)
        if not np.any(edge_exists):
            continue
        target_tf_ids = target_tf_ids[edge_exists]
        source_tf_ids = source_tf_ids[edge_exists]

        weights_tf = np.asarray(edge["params"]["weight"], dtype=np.float32)
        if weights_tf.ndim == 0:
            weights_tf = np.full(target_tf_ids.shape, weights_tf, dtype=np.float32)
        else:
            weights_tf = weights_tf[edge_exists]

        delays_tf = edge["params"]["delay"]
        n_new_edge = int(np.sum(edge_exists))
        indices[current_edge: current_edge + n_new_edge] = np.array(
            [target_tf_ids * 4 + r, source_tf_ids]
        ).T
        weights[current_edge: current_edge + n_new_edge] = weights_tf
        delays[current_edge: current_edge + n_new_edge] = delays_tf
        current_edge += n_new_edge

    indices, weights, delays = sort_sparse_indices(indices, weights, delays)

    if use_rand_connectivity:
        with open(os.path.join(data_dir, "../random_connectivity.pkl"), "rb") as f:
            data_tmp = pkl.load(f)
        indices = data_tmp["indices"]
        indices, weights, delays = sort_sparse_indices(indices, weights, delays)

    if use_rand_ini_w:
        if use_dale_law:
            w_ab_value = np.abs(rd.randn(*weights.shape))
            w_ab_value = (w_ab_value - w_ab_value.mean()) / w_ab_value.std()
            w_ab_value = w_ab_value * weights.std() + weights.mean()
            weights = np.sign(weights) * w_ab_value
        else:
            weights = rd.randn(*weights.shape) * weights.std() + weights.mean()
        weights = weights.astype(np.float32)

    if scale_w_e > 0:
        weights[weights > 0] = weights[weights > 0] * scale_w_e

    # Provide both tensorflow-style and brainpy-style keys.
    pos = np.stack([x, y, z], axis=-1)
    receptor = (indices[:, 0] % 4).astype(np.int32, copy=False)
    post = (indices[:, 0] // 4).astype(np.int32, copy=False)
    pre = indices[:, 1].astype(np.int32, copy=False)

    neuron_params = {k: v[node_type_ids] for k, v in node_params.items()}

    network = dict(
        x=x,
        y=y,
        z=z,
        pos=pos,
        n_nodes=n_nodes,
        n_neurons=n_nodes,
        n_edges=n_edges,
        node_params=node_params,
        node_type_ids=node_type_ids,
        neuron_type_ids=node_type_ids,
        neuron_params=neuron_params,
        synapses=dict(
            indices=indices,
            weights=weights,
            delays=delays,
            dense_shape=dense_shape,
        ),
        syn=dict(
            pre=pre,
            post=post,
            receptor=receptor,
            weight=weights,
            delay=delays,
        ),
        tf_id_to_bmtk_id=tf_id_to_bmtk_id,
        bmtk_id_to_tf_id=bmtk_id_to_tf_id,
        bmtk2local=bmtk_id_to_tf_id,
    )

    return network


def load_input(
    path="/home/wanghz/EC_V1/data/GLIF_V1_network/input_dat.pkl",
    start=0,
    duration=3000,
    dt=1,
    bmtk2local=None,
    bmtk_id_to_tf_id=None,
    spike_repr="dense",
):
    with open(path, "rb") as f:
        d = pkl.load(f)

    # Backward-compat alias with tensorflow/common loader signature.
    if bmtk2local is None and bmtk_id_to_tf_id is not None:
        bmtk2local = bmtk_id_to_tf_id
    bmtk2local = None if bmtk2local is None else np.asarray(bmtk2local)
    if spike_repr not in ("dense", "coo", "both"):
        raise ValueError(f"spike_repr must be 'dense', 'coo', or 'both', got {spike_repr}")

    n_steps = int(duration / dt)
    input_populations = []

    for input_population in d:
        post_indices = []
        pre_indices = []
        weights = []
        delays = []

        for edge in input_population[1]:
            r = edge["params"]["receptor_type"] - 1
            target_local = np.asarray(edge["target"], dtype=np.int64)
            source_idx = np.asarray(edge["source"], dtype=np.int64)

            w = np.asarray(edge["params"]["weight"], dtype=np.float32)
            if w.ndim == 0:
                w = np.full(source_idx.shape, w, dtype=np.float32)

            dly = np.asarray(edge["params"]["delay"], dtype=np.float32)
            if dly.ndim == 0:
                dly = np.full(source_idx.shape, dly, dtype=np.float32)

            if bmtk2local is not None:
                target_local = bmtk2local[target_local]
                edge_exists = target_local >= 0
                target_local = target_local[edge_exists]
                source_idx = source_idx[edge_exists]
                w = w[edge_exists]
                dly = dly[edge_exists]

            if target_local.size == 0:
                continue

            post_indices.append(4 * target_local + r)
            pre_indices.append(source_idx)
            weights.append(w)
            delays.append(dly)

        if post_indices:
            post_indices = np.concatenate(post_indices).astype(np.int64, copy=False)
            pre_indices = np.concatenate(pre_indices).astype(np.int64, copy=False)
            indices = np.stack([post_indices, pre_indices], axis=-1)
            weights = np.concatenate(weights).astype(np.float32, copy=False)
            delays = np.concatenate(delays).astype(np.float32, copy=False)

            # Keep (post, pre, weight, delay) aligned under the same permutation.
            indices, weights, delays = sort_sparse_indices(indices, weights, delays)
        else:
            indices = np.zeros((0, 2), dtype=np.int64)
            weights = np.zeros((0,), dtype=np.float32)
            delays = np.zeros((0,), dtype=np.float32)

        n_inputs = len(input_population[0]["ids"])
        rows = []
        cols = []
        for i, sp in zip(input_population[0]["ids"], input_population[0]["spikes"]):
            sp = np.asarray(sp)
            sp = sp[np.logical_and(start < sp, sp < start + duration)] - start
            sp = (sp / dt).astype(np.int32)
            sp = sp[np.logical_and(sp >= 0, sp < n_steps)]
            if sp.size == 0:
                continue
            rows.append(sp)
            cols.append(np.full_like(sp, i, dtype=np.int32))

        if rows:
            spike_times = np.concatenate(rows).astype(np.int32, copy=False)
            spike_ids = np.concatenate(cols).astype(np.int32, copy=False)
        else:
            spike_times = np.zeros((0,), dtype=np.int32)
            spike_ids = np.zeros((0,), dtype=np.int32)

        out = dict(
            n_inputs=n_inputs,
            indices=jnp.asarray(indices, dtype=jnp.int32),
            weights=jnp.asarray(weights, dtype=jnp.float32),
            delays=jnp.asarray(delays, dtype=jnp.float32),
        )
        if spike_repr in ("dense", "both"):
            spikes = np.zeros((n_steps, n_inputs), dtype=np.float32)
            if spike_times.size > 0:
                # np.add.at preserves repeated-event accumulation semantics.
                np.add.at(spikes, (spike_times, spike_ids), 1.0)
            out["spikes"] = jnp.asarray(spikes, dtype=jnp.float32)
        if spike_repr in ("coo", "both"):
            out["spike_times"] = jnp.asarray(spike_times, dtype=jnp.int32)
            out["spike_ids"] = jnp.asarray(spike_ids, dtype=jnp.int32)
            out["n_steps"] = n_steps

        input_populations.append(out)

    return input_populations


def load_TD_input(
    path,
    network,
    n_inputs,
    targets,
    inter_area_min_delay,
    inter_area_max_delay,
    seed,
):
    with open(path, "rb") as f:
        d = pkl.load(f)

    # Estimate connection probability from bottom-up stimulus connections.
    cons = [len(edge["target"]) for edge in d[0][1]]
    num_cons = sum(cons)
    l4e = np.asarray(network["laminar_indices"]["L4e"])
    l4i = np.asarray(network["laminar_indices"]["L4i"])
    con_prob = num_cons / (17400 * (l4e.size + l4i.size))

    # Pool connection weights from stimulus input for top-down resampling.
    weights_pool = np.concatenate([edge["params"]["weight"] for edge in d[0][1]])

    targets_pool = []
    for target_pop in targets.split(","):
        target_pop = target_pop.strip()
        if not target_pop:
            continue
        targets_pool.append(np.asarray(network["laminar_indices"][target_pop]))

    post_indices = []
    pre_indices = []
    weights = []
    rd = np.random.RandomState(seed=seed)

    for target in targets_pool:
        n_pairs = int(target.size * n_inputs)
        n_select = int(0.1 * con_prob * n_pairs)
        if n_pairs == 0 or n_select == 0:
            continue
        n_select = min(n_select, n_pairs)
        tmp = rd.choice(n_pairs, n_select, replace=False)
        post_indices.append(target[np.mod(tmp, target.size)] * 4 + rd.randint(0, 4, tmp.size))
        pre_indices.append(tmp // target.size)
        weights.append(rd.choice(weights_pool, tmp.size, replace=True))

    if post_indices:
        post_indices = np.concatenate(post_indices).astype(np.int64, copy=False)
        pre_indices = np.concatenate(pre_indices).astype(np.int64, copy=False)
        indices = np.stack([post_indices, pre_indices], axis=-1)
        weights = np.concatenate(weights).astype(np.float32, copy=False)
        indices, weights = sort_sparse_indices(indices, weights)
    else:
        indices = np.zeros((0, 2), dtype=np.int64)
        weights = np.zeros((0,), dtype=np.float32)

    delays = rd.randint(
        low=inter_area_min_delay,
        high=inter_area_max_delay,
        size=weights.shape,
    ).astype(np.int32, copy=False)

    return dict(
        n_inputs=n_inputs,
        indices=jnp.asarray(indices, dtype=jnp.int32),
        weights=jnp.asarray(weights, dtype=jnp.float32),
        delays=jnp.asarray(delays, dtype=jnp.int32),
    )


def reduce_input_population(input_population, new_n_input, seed=3000):
    rd = np.random.RandomState(seed=seed)
    if new_n_input <= 0:
        raise ValueError(f"new_n_input must be positive, got {new_n_input}")

    in_ind = np.asarray(input_population["indices"])
    in_weights = np.asarray(input_population["weights"], dtype=np.float32)

    if in_ind.ndim != 2 or in_ind.shape[1] != 2:
        raise ValueError(f"indices must have shape (N, 2), got {in_ind.shape}")
    if in_ind.shape[0] != in_weights.shape[0]:
        raise ValueError(
            f"indices/weights size mismatch: {in_ind.shape[0]} vs {in_weights.shape[0]}"
        )

    n_old_input = int(input_population["n_inputs"])
    assignment = rd.choice(np.arange(new_n_input), size=n_old_input, replace=True)

    if in_ind.shape[0] == 0:
        new_in_ind = np.zeros((0, 2), dtype=np.int64)
        new_in_weights = np.zeros((0,), dtype=np.float32)
    else:
        post = in_ind[:, 0].astype(np.int64, copy=False)
        pre_old = in_ind[:, 1].astype(np.int64, copy=False)
        pre_new = assignment[pre_old].astype(np.int64, copy=False)

        # Merge duplicated (post, new_pre) connections by summing their weights.
        flat_key = post * np.int64(new_n_input) + pre_new
        uniq_key, inv = np.unique(flat_key, return_inverse=True)
        new_in_weights = np.zeros((uniq_key.shape[0],), dtype=np.float32)
        np.add.at(new_in_weights, inv, in_weights)

        new_post = uniq_key // np.int64(new_n_input)
        new_pre = uniq_key % np.int64(new_n_input)
        new_in_ind = np.stack([new_post, new_pre], axis=-1).astype(np.int64, copy=False)
        new_in_ind, new_in_weights = sort_sparse_indices(new_in_ind, new_in_weights)

    return dict(
        n_inputs=new_n_input,
        indices=jnp.asarray(new_in_ind, dtype=jnp.int32),
        weights=jnp.asarray(new_in_weights, dtype=jnp.float32),
        spikes=None,
    )


def set_laminar_indices(df, h5_path, network, L2_neuron_ratio=0.5):
    # locate neuron population
    node_types = df
    node_h5 = h5py.File(h5_path, mode="r")
    node_type_id_to_pop_name = dict()
    for nid in np.unique(node_h5["nodes"]["v1"]["node_type_id"]):
        ind_list = np.where(node_types.node_type_id == nid)[0]
        assert len(ind_list) == 1
        node_type_id_to_pop_name[nid] = node_types.pop_name[ind_list[0]]

    if "tf_id_to_bmtk_id" in network:
        local_to_bmtk_id = np.asarray(network["tf_id_to_bmtk_id"], dtype=np.int64)
    elif "bmtk2local" in network:
        bmtk2local = np.asarray(network["bmtk2local"], dtype=np.int64)
        valid = bmtk2local >= 0
        local_to_bmtk_id = np.zeros(np.sum(valid), dtype=np.int64)
        local_to_bmtk_id[bmtk2local[valid]] = np.where(valid)[0]
    else:
        raise KeyError("network must contain either 'tf_id_to_bmtk_id' or 'bmtk2local'")

    all_node_type_ids = np.asarray(node_h5["nodes"]["v1"]["node_type_id"], dtype=np.int64)
    all_pop_names = np.array(
        [node_type_id_to_pop_name[nid] for nid in all_node_type_ids], dtype=object
    )[local_to_bmtk_id]

    neuron_pop_id_to_name = [
        "i1Htr3a",
        "e23",
        "i23Pvalb",
        "i23Sst",
        "i23Htr3a",
        "e4",
        "i4Pvalb",
        "i4Sst",
        "i4Htr3a",
        "e5",
        "i5Pvalb",
        "i5Sst",
        "i5Htr3a",
        "e6",
        "i6Pvalb",
        "i6Sst",
        "i6Htr3a",
    ]

    rough_neuron_pop_names = np.zeros(all_pop_names.shape, np.int32)
    for i, pop_name in enumerate(all_pop_names):
        for j, pp_name in enumerate(neuron_pop_id_to_name):
            if pop_name.startswith(pp_name):
                rough_neuron_pop_names[i] = j
                break

    laminar_indices = dict()
    # exc neurons
    laminar_indices["L1e"] = np.array([], dtype=np.int32)
    exc_ind = [1, 5, 9, 13]
    for i, layer_number in enumerate([23, 4, 5, 6]):
        laminar_indices[f"L{layer_number}e"] = np.where(rough_neuron_pop_names == exc_ind[i])[0].astype(np.int32)

    # inh neurons
    laminar_indices["L1i"] = np.where(rough_neuron_pop_names == 0)[0].astype(np.int32)
    inh_ind = [2, 6, 10, 14]
    for i, layer_number in enumerate([23, 4, 5, 6]):
        temp = []
        for ii in range(3):
            temp.append(np.where(rough_neuron_pop_names == inh_ind[i] + ii)[0])
        laminar_indices[f"L{layer_number}i"] = np.concatenate(temp).astype(np.int32)

    # split 2/3 layers
    if "y" in network:
        y = np.asarray(network["y"])
    elif "pos" in network:
        y = np.asarray(network["pos"])[:, 1]
    else:
        raise KeyError("network must contain either 'y' or 'pos' for vertical coordinates")

    vertical_coordinates_e = y[laminar_indices["L23e"]]
    vertical_coordinates_i = y[laminar_indices["L23i"]]
    vertical_coordinates = np.hstack((vertical_coordinates_e, vertical_coordinates_i))
    L23_argindices_sorted = np.argsort(vertical_coordinates)
    L23_neuorn_indices = np.hstack((laminar_indices["L23e"], laminar_indices["L23i"]))

    split_at = np.int64(L2_neuron_ratio * vertical_coordinates.size)
    L2_argindices = L23_argindices_sorted[:split_at]
    L2e_argindices = L2_argindices[L2_argindices < vertical_coordinates_e.size]
    laminar_indices["L2e"] = L23_neuorn_indices[L2e_argindices].astype(np.int32)
    # Keep the same boundary convention as tensorflow_impl/load_sparse.py.
    L2i_argindices = L2_argindices[L2_argindices > vertical_coordinates_e.size]
    laminar_indices["L2i"] = L23_neuorn_indices[L2i_argindices].astype(np.int32)

    L3_argindices = L23_argindices_sorted[split_at:]
    L3e_argindices = L3_argindices[L3_argindices < vertical_coordinates_e.size]
    laminar_indices["L3e"] = L23_neuorn_indices[L3e_argindices].astype(np.int32)
    L3i_argindices = L3_argindices[L3_argindices > vertical_coordinates_e.size]
    laminar_indices["L3i"] = L23_neuorn_indices[L3i_argindices].astype(np.int32)

    network["laminar_indices"] = {
        k: jnp.asarray(v, dtype=jnp.int32) for k, v in laminar_indices.items()
    }
    return network


def load_billeh(
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
):
    h5_path = os.path.join(data_dir, "network/v1_nodes.h5")
    network = load_network(
        path=os.path.join(data_dir, "network_dat.pkl"),
        h5_path=h5_path,
        data_dir=data_dir,
        core_only=core_only,
        n_neurons=n_neurons,
        seed=seed,
        connected_selection=connected_selection,
        use_rand_ini_w=use_rand_ini_w,
        use_dale_law=use_dale_law,
        use_rand_connectivity=use_rand_connectivity,
        use_only_one_type=use_only_one_type,
        use_uniform_neuron_type=use_uniform_neuron_type,
        scale_w_e=scale_w_e,
    )

    # Compatibility aliases for tensorflow/common conventions.
    pos = np.asarray(network["pos"])
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    network["x"] = jnp.asarray(x, dtype=jnp.float32)
    network["y"] = jnp.asarray(y, dtype=jnp.float32)
    network["z"] = jnp.asarray(z, dtype=jnp.float32)
    network["n_nodes"] = int(network["n_neurons"])
    network["node_type_ids"] = jnp.asarray(network["neuron_type_ids"], dtype=jnp.int32)
    network["bmtk_id_to_tf_id"] = jnp.asarray(network["bmtk2local"], dtype=jnp.int32)

    bmtk2local = np.asarray(network["bmtk2local"], dtype=np.int64)
    valid = bmtk2local >= 0
    tf_id_to_bmtk_id = np.zeros(np.sum(valid), dtype=np.int64)
    tf_id_to_bmtk_id[bmtk2local[valid]] = np.where(valid)[0]
    network["tf_id_to_bmtk_id"] = jnp.asarray(tf_id_to_bmtk_id, dtype=jnp.int32)

    inputs = load_input(
        start=1000,
        duration=1000,
        dt=1,
        path=os.path.join(data_dir, "input_dat.pkl"),
        bmtk2local=network["bmtk2local"],
    )

    df = pd.read_csv(os.path.join(data_dir, "network/v1_node_types.csv"), delimiter=" ")
    network = set_laminar_indices(df, h5_path, network)

    l5e_types_indices = []
    for a in df.iterrows():
        if a[1]["pop_name"].startswith("e5"):
            l5e_types_indices.append(a[0])
    l5e_types_indices = np.asarray(l5e_types_indices, dtype=np.int32)
    node_type_ids = np.asarray(network["node_type_ids"])
    l5e_neuron_sel = np.zeros(network["n_nodes"], dtype=bool)
    for l5e_type_index in l5e_types_indices:
        l5e_neuron_sel = np.logical_or(l5e_neuron_sel, node_type_ids == l5e_type_index)
    l5e_neuron_indices = np.where(l5e_neuron_sel)[0]
    network["l5e_types"] = jnp.asarray(l5e_types_indices, dtype=jnp.int32)
    network["l5e_neuron_sel"] = jnp.asarray(l5e_neuron_sel)
    print(f"> Number of L5e Neurons: {int(np.sum(l5e_neuron_sel))}")

    # Determine localized readout neurons.
    with h5py.File(h5_path, mode="r") as node_h5:
        node_type_id_to_pop_name = dict()
        node_h5_type_ids = np.asarray(node_h5["nodes"]["v1"]["node_type_id"], dtype=np.int64)
        for nid in np.unique(node_h5_type_ids):
            ind_list = np.where(df.node_type_id == nid)[0]
            assert len(ind_list) == 1
            node_type_id_to_pop_name[nid] = df.pop_name[ind_list[0]]

    local_to_bmtk = np.asarray(network["tf_id_to_bmtk_id"], dtype=np.int64)
    all_pop_names = np.array(
        [node_type_id_to_pop_name[nid] for nid in node_h5_type_ids], dtype=object
    )[local_to_bmtk]

    rough_neuron_pop_names2 = np.zeros(all_pop_names.shape, np.int32)
    for i, pop_name in enumerate(all_pop_names):
        if pop_name[0] == "e":
            rough_neuron_pop_names2[i] = 0
        elif pop_name.count("Htr") > 0:
            rough_neuron_pop_names2[i] = 1
        elif pop_name.count("Sst") > 0:
            rough_neuron_pop_names2[i] = 2
        elif pop_name.count("Pvalb") > 0:
            rough_neuron_pop_names2[i] = 3

    layer_pop_names = np.zeros(all_pop_names.shape, np.int32)
    for i, pop_name in enumerate(all_pop_names):
        if pop_name[1] == "1":
            layer_pop_names[i] = 0
        elif pop_name[1] == "2":
            layer_pop_names[i] = 1
        elif pop_name[1] == "4":
            layer_pop_names[i] = 2
        elif pop_name[1] == "5":
            layer_pop_names[i] = 3
        elif pop_name[1] == "6":
            layer_pop_names[i] = 4

    bounds = []
    for i in range(5):
        sel = layer_pop_names == i
        bounds.append((np.min(y[sel]), np.max(y[sel])))

    pos = np.stack((x, z, y), axis=-1)
    origins = np.tile(
        np.array([[90, -95, np.array(bounds[3]).mean()]])[None], (15, 1, 1)
    )
    origins[:15, 0, :2] = [
        [0, 0],
        [100, -110],
        [-100, -110],
        [-100, 110],
        [100, 110],
        [0, 260],
        [180, 230],
        [-180, 230],
        [270, 95],
        [-270, 95],
        [270, -95],
        [-270, -95],
        [180, -230],
        [-180, -230],
        [0, -260],
    ]

    if localized_readout:
        try:
            for i in range(15):
                origin = origins[i]
                sel = rough_neuron_pop_names2 == 0
                sel = np.logical_and(sel, y < bounds[3][1])
                sel = np.logical_and(sel, y > bounds[3][0])
                sel = np.logical_and(sel, np.sqrt(np.square(pos - origin).sum(-1)) < 55)
                rd = np.random.RandomState(seed=seed)
                sel_ind = np.where(sel)[0]
                sel_ind = rd.choice(sel_ind, replace=False, size=neurons_per_output)
                sel = np.zeros_like(sel)
                sel[sel_ind] = True
                network[f"localized_readout_neuron_ids_{i}"] = jnp.asarray(
                    np.where(sel)[0][None], dtype=jnp.int32
                )
        except ValueError:
            print("Warning: Small neuronal volume, not all readout populations available")
            if "localized_readout_neuron_ids_0" not in network.keys():
                raise ValueError("Neuronal volume too small: No readout population")
    else:
        rd = np.random.RandomState(seed=seed)
        readout_neurons_random = rd.choice(
            l5e_neuron_indices, size=30 * 15, replace=False
        ).reshape((15, 30))
        # Keep localized naming for compatibility with existing code paths.
        for i in range(15):
            network[f"localized_readout_neuron_ids_{i}"] = jnp.asarray(
                readout_neurons_random[i, :][None, :], dtype=jnp.int32
            )

    network["localized_readout_neuron_ids"] = network["localized_readout_neuron_ids_0"]

    input_population = inputs[0]
    bkg = inputs[1]
    bkg_indices = np.asarray(bkg["indices"])
    bkg_w = np.asarray(bkg["weights"], dtype=np.float32)
    bkg_weights = np.zeros((network["n_nodes"] * 4,), np.float32)
    bkg_weights[bkg_indices[:, 0]] = bkg_w
    if n_input != 17400:
        input_population = reduce_input_population(input_population, n_input, seed=seed)

    bkg_weights = jnp.asarray(bkg_weights, dtype=jnp.float32)
    if TD_input:
        TD_inputs = load_TD_input(
            os.path.join(data_dir, "input_dat.pkl"),
            network,
            n_TD_input,
            targets,
            3,
            5,
            seed,
        )
        return TD_inputs, input_population, network, bkg_weights
    else:
        return input_population, network, bkg_weights


def cached_load_billeh(
    n_input,
    n_neurons,
    core_only,
    data_dir,
    seed=3000,
    connected_selection=False,
    n_output=2,
    neurons_per_output=16,
    use_rand_ini_w=False,
    scale_w_e=-1,
):
    store = False
    input_population, network, bkg_weights = None, None, None
    flag_str = (
        f"in{n_input}_rec{n_neurons}_s{seed}_c{core_only}_con{connected_selection}"
    )
    flag_str += f"_out{n_output}_nper{neurons_per_output}"
    cache_path = f".cache/billeh_network_{flag_str}.pkl"
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                input_population, network, bkg_weights = pkl.load(f)
                print(f"> Sucessfully restored Billeh model from {cache_path}")
        except Exception as e:
            print(e)
            store = True
    else:
        store = True
    if input_population is None or network is None or bkg_weights is None:
        input_population, network, bkg_weights = load_billeh(
            n_input,
            n_neurons,
            core_only,
            data_dir,
            seed,
            connected_selection=connected_selection,
            n_output=n_output,
            neurons_per_output=neurons_per_output,
            use_rand_ini_w=use_rand_ini_w,
            scale_w_e=scale_w_e,
        )
    if store:
        os.makedirs(".cache", exist_ok=True)
        with open(cache_path, "wb") as f:
            pkl.dump((input_population, network, bkg_weights), f)
        print(f"> Cached Billeh model in {cache_path}")
    return input_population, network, bkg_weights


def main(base_path):
    import torch

    TD_input_population, input_population, network, bkg_weights = load_billeh(
        n_input=17400,
        n_neurons=5000,
        core_only=False,
        data_dir=base_path,
        seed=3000,
        connected_selection=True,
        n_output=2,
        neurons_per_output=16,
        use_rand_ini_w=False,
        use_rand_connectivity=False,
        use_uniform_neuron_type=False,
        scale_w_e=-1,
        TD_input=True,
        n_TD_input=5000,
        targets="L23e,L5e",
    )

    td_weights = np.asarray(TD_input_population["weights"], dtype=np.float32)
    td_indices = np.asarray(TD_input_population["indices"], dtype=np.int64).T
    td_dense_shape = (4 * int(network["n_nodes"]), int(TD_input_population["n_inputs"]))
    sparse_w_in = torch.sparse_coo_tensor(td_indices, td_weights, td_dense_shape)
    dense_w_in = sparse_w_in.to_dense()
    print("> TD sparse matrix built:", tuple(dense_w_in.shape))
    print("> Input population size:", int(input_population["n_inputs"]))
    print("> Background weight size:", int(np.asarray(bkg_weights).shape[0]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/wanghz/EC_V1/data/GLIF_V1_network",
    )
    args = parser.parse_args()
    main(args.data_dir)
