import brainpy as bp
import brainpy.math as bm
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import pandas as pd
import os
import pickle as pkl
import sys
import h5py
try:
    import tensorflow as tf
except Exception:
    tf = None
from bmtk.simulator.filternet.lgnmodel.fitfuns import makeBasis_StimKernel
from bmtk.simulator.filternet.lgnmodel.spatialfilter import GaussianSpatialFilter
from bmtk.simulator.filternet.lgnmodel.temporalfilter import TemporalFilterCosineBump
from bmtk.simulator.filternet.lgnmodel.util_fns import get_tcross_from_temporal_kernel
try:  # this is old version of bmtk
    from bmtk.simulator.filternet.lgnmodel.util_fns import get_data_metrics_for_each_subclass
except ImportError:  # new bmtk imports
    from bmtk.simulator.filternet.lgnmodel.cellmetrics import get_data_metrics_for_each_subclass


def _load_pickle_compat(path):
    sys.modules.setdefault("numpy._core", np.core)
    if tf is None:
        with open(path, "rb") as f:
            return pkl.load(f)

    # Old LGN cache pickles may deserialize TensorFlow tensors; keep that on CPU
    # so BrainPy/JAX can still use GPU without TensorFlow trying to stage constants there.
    with tf.device("/CPU:0"):
        with open(path, "rb") as f:
            return pkl.load(f)


#create temporal filter 
def create_temporal_filter(inp_dict, nkt=600):

    opt_wts = inp_dict['opt_wts']
    opt_kpeaks = inp_dict['opt_kpeaks']
    opt_delays = inp_dict['opt_delays']

    # Build the temporal basis (BMTK function returns numpy)
    basis = makeBasis_StimKernel(dict(
        neye=0,
        ncos=opt_wts.shape[0],
        kpeaks=opt_kpeaks,
        b=0.3,
        delays=[opt_delays.astype(int)]
    ), nkt)

    # Combine basis with weights → raw kernel
    kernel_np = basis @ opt_wts

    # Convert to jax array for BrainPy
    kernel = jnp.asarray(kernel_np, dtype=jnp.float32)

    return kernel


#用在composition部分 生成temporal kernel  
#将实验数据对齐composition的cell时间特性
def create_one_unit_of_two_subunit_filter(prs, ttp_exp, nkt=600):

    opt_wts = prs["opt_wts"]
    opt_kpeaks = prs["opt_kpeaks"]
    opt_delays = prs["opt_delays"]

    basis = makeBasis_StimKernel(dict(
        neye=0,
        ncos=opt_wts.shape[0],
        kpeaks=opt_kpeaks,
        b=0.3,
        delays=[opt_delays.astype(int)]
    ), nkt)

    kernel_np = basis @ opt_wts  

    tcross_ind = get_tcross_from_temporal_kernel(kernel_np)

    filt_sum = kernel_np[:tcross_ind].sum()

    del_offset = ttp_exp - tcross_ind

    if del_offset >= 0:
     
        delays = prs["opt_delays"]
        delays[0] = delays[0] + del_offset
        delays[1] = delays[1] + del_offset
        prs["opt_delays"] = delays

        basis_new = makeBasis_StimKernel(dict(
            neye=0,
            ncos=opt_wts.shape[0],
            kpeaks=opt_kpeaks,
            b=0.3,
            delays=[prs["opt_delays"].astype(int)]
        ), nkt)

        kernel_new_np = basis_new @ opt_wts
    else:
     
        kernel_new_np = kernel_np

    kernel_new_jax = jnp.asarray(kernel_new_np, dtype=jnp.float32)

    return kernel_new_jax, float(filt_sum)



def create_lgn_units_info(csv_path='/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/GLIF_network/network/lgn_node_types.csv', 
                          h5_path='/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/GLIF_network/network//lgn_nodes.h5',
                          filename='/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/lgn_model/data/lgn_full_col_cells.csv'
                          ):
    # filename = os.path.join('data', filename)
    # Load both the h5 file and the csv file
    csv_file = pd.read_csv(csv_path, sep=' ')
    features = ['id', 'model_id', 'x', 'y', 'ei', 'location', 'spatial_size', 'kpeaks_dom_0', 'kpeaks_dom_1', 'weight_dom_0', 'weight_dom_1', 'delay_dom_0', 'delay_dom_1', 'kpeaks_non_dom_0', 'kpeaks_non_dom_1', 'weight_non_dom_0', 'weight_non_dom_1', 'delay_non_dom_0', 'delay_non_dom_1', 'tuning_angle', 'sf_sep']
    df = pd.DataFrame(columns=features)

    with h5py.File(h5_path, 'r') as h5_file:
        node_id = h5_file['nodes']['lgn']['node_id'][:]
        node_type_id = h5_file['nodes']['lgn']['node_type_id'][:]
        for feature in h5_file['nodes']['lgn']['0'].keys():
            value = h5_file['nodes']['lgn']['0'][feature][:]
            arr = jnp.asarray(value, dtype=jnp.float32)
            df[feature] = np.array(arr)  # back to NumPy for pandas

    node_info = {}
    for index, row in csv_file.iterrows():
        node_info[row['node_type_id']] = {'model_id': row['pop_name'], 'location': row['location'], 'ei': row['ei']}

    df['id'] = node_id
    df['model_id'] = [node_info[int(x)]['model_id'] for x in node_type_id]
    df['location'] = [node_info[node_type_id[i]]['location'] for i in range(len(node_type_id))]
    df['ei'] = [node_info[node_type_id[i]]['ei'] for i in range(len(node_type_id))]

    df.to_csv(filename, index=False, sep=' ', na_rep='NaN')
    return df



def create_lgn_units_info(
    csv_path='/root/autodl-tmp/EC_proj/data/GLIF_V1_network/network/lgn_node_types.csv',
    h5_path='/root/autodl-tmp/EC_proj/data/GLIF_V1_network/network//lgn_nodes.h5',
    file_path='/root/autodl-tmp/EC_proj/data/GLIF_V1_network/lgn_full_col_cells_3.csv'
):
  
    csv_df = pd.read_csv(csv_path, sep=" ")
    node_type_info = {}
    for _, row in csv_df.iterrows():
        node_type_info[row["node_type_id"]] = {
            "model_id": row["pop_name"],
            "location": row["location"],
            "ei": row["ei"],
        }

    with h5py.File(h5_path, "r") as h5_file:
        node_id_np = h5_file["nodes/lgn/node_id"][:]
        node_type_id_np = h5_file["nodes/lgn/node_type_id"][:]

        numeric_data = {}
        for feature in h5_file["nodes/lgn/0"].keys():
            raw = h5_file[f"nodes/lgn/0/{feature}"][:]
            numeric_data[feature] = raw.astype(np.float32)  

    model_ids = []
    locations = []
    eis = []
    for nt in node_type_id_np:
        info = node_type_info[int(nt)]
        model_ids.append(info["model_id"])
        locations.append(info["location"])
        eis.append(info["ei"])

    df = pd.DataFrame({
        "id": node_id_np.astype(np.int32),
        "model_id": model_ids,
        "location": locations,
        "ei": np.asarray(eis, dtype=str),
    })

    for key, arr in numeric_data.items():
        df[key] = arr

    if file_path is not None:
        df.to_csv(file_path, sep=" ", index=False, na_rep="NaN")

    return df

def temporal_filter(X: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
    K_len, N = K.shape
    # Keep this path pure JAX so tf.data generators do not depend on a
    # surrounding brainstate environment (e.g. numpy_func_return).
    X_pad = jnp.pad(X, ((K_len - 1, 0), (0, 0)))

    lhs = X_pad.T[None, :, :, None]  # → [1, N, T_pad, 1]
    rhs = K.T[:, None, :, None]      # → [N, 1, kernel_len, 1]

    Y = jax.lax.conv_general_dilated(
        lhs=lhs,                         # [1, N, T_pad, 1]
        rhs=rhs,                         # [N, 1, K_len, 1]
        window_strides=(1, 1),
        padding='VALID',
        feature_group_count=N,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )

    return Y[0, :, :, 0].T


def transfer_function(x: bm.ndarray) -> bm.ndarray:
    return jnp.maximum(x, 0)



def select_spatial(x: bm.ndarray,
                      y: bm.ndarray,
                      convolved_movie: bm.ndarray) -> bm.ndarray:
    """
    Bilinear interpolation spatial sampling (JAX/BrainPy).

    x              : [N]  float positions on width axis
    y              : [N]  float positions on height axis
    convolved_movie: [T, H, W]  input responses to sample

    returns:
        spatial_responses : [T, N]
    """

    # 1) Compute integer neighbor indices
    x0 = jnp.floor(x).astype(jnp.int32)
    x1 = jnp.ceil(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    y1 = jnp.ceil(y).astype(jnp.int32)

    # 2) Stack index tuples
    # We build (N, 2) arrays of coordinate pairs
    i1 = jnp.stack([y0, x0], axis=-1)  # top-left
    i2 = jnp.stack([y1, x0], axis=-1)  # bottom-left
    i3 = jnp.stack([y0, x1], axis=-1)  # top-right
    i4 = jnp.stack([y1, x1], axis=-1)  # bottom-right

    # 3) Gather values at neighbor pixels
    # Transpose movie to [H, W, T] so index last dims are H,W
    transposed = jnp.transpose(convolved_movie, (1, 2, 0))

    # gather_nd equivalent in JAX:
    sr1 = transposed[i1[:, 0], i1[:, 1]]   # shape [N, T]
    sr2 = transposed[i2[:, 0], i2[:, 1]]
    sr3 = transposed[i3[:, 0], i3[:, 1]]
    sr4 = transposed[i4[:, 0], i4[:, 1]]

    # 4) Compute bilinear interpolation weights
    # fractional parts
    xf = x - jnp.floor(x)
    yf = y - jnp.floor(y)

    w1 = (1 - xf) * (1 - yf)  # top-left weight
    w2 = (1 - xf) * yf        # bottom-left weight
    w3 = xf * (1 - yf)        # top-right weight
    w4 = xf * yf              # bottom-right weight

    # 5) Weighted sum
    # each sr has shape [N, T], weights are [N], so broadcast
    spatial_responses = (sr1 * w1[:, None]
                       + sr2 * w2[:, None]
                       + sr3 * w3[:, None]
                       + sr4 * w4[:, None])

    # 6) Transpose to [T, N]
    spatial_responses = jnp.transpose(spatial_responses, (1, 0))

    return spatial_responses

class LGN(bp.DynamicalSystem):
    def __init__(self,
                 row_size: int,
                 col_size: int,
                 data_dir: str,
                 n_input: int = None,
                 bmtk_compat: bool = True):
        super().__init__()
        self.bmtk_compat = bmtk_compat

        # load the metadata
        filename = f'lgn_full_col_cells_{col_size}x{row_size}.csv'
        root_dir = os.path.abspath(data_dir)
        csv_path = os.path.join(root_dir, 'network', 'lgn_node_types.csv')
        h5_path = os.path.join(root_dir, 'network', 'lgn_nodes.h5')

        lgn_data_path = os.path.join(root_dir, 'tf_data', filename)
        if os.path.exists(lgn_data_path):
            self.info_df = pd.read_csv(lgn_data_path, sep=r'\s+')
        else:
            self.info_df = create_lgn_units_info(csv_path, h5_path, file_path=lgn_data_path)

        # parse basic LGN quantities
        model_ids = self.info_df['model_id'].to_numpy()
        amplitude = np.array([1. if 'ON' in a else -1. for a in model_ids], dtype=np.float32)
        non_dom_amplitude = np.zeros_like(amplitude)
        is_composite = np.array([('ON' in m and 'OFF' in m) for m in model_ids], dtype=np.float32)

        self.x = jnp.asarray(self.info_df['x'].to_numpy(dtype=np.float32))
        self.y = jnp.asarray(self.info_df['y'].to_numpy(dtype=np.float32))
        self.is_composite = jnp.asarray(is_composite)

        # spontaneous firing rates
        spont_path = os.path.join(root_dir, 'tf_data',
                                  f'spontaneous_firing_rates_{col_size}x{row_size}.pkl')
        with open(spont_path, 'rb') as f:
            spontaneous_firing_rates = pkl.load(f)

        # temporal kernels
        tk_path = os.path.join(root_dir, 'tf_data',
                               f'temporal_kernels_{col_size}x{row_size}.pkl')
        with open(tk_path, 'rb') as f:
            loaded = pkl.load(f)

        self.dom_temporal_kernels = jnp.asarray(loaded['dom_temporal_kernels'], dtype=jnp.float32)
        self.non_dom_temporal_kernels = jnp.asarray(loaded['non_dom_temporal_kernels'], dtype=jnp.float32)
        self.non_dominant_x = jnp.asarray(loaded['non_dominant_x'], dtype=jnp.float32)
        self.non_dominant_y = jnp.asarray(loaded['non_dominant_y'], dtype=jnp.float32)
        # Keep amplitude/spontaneous sources aligned with TensorFlow LGN.
        amplitude = loaded.get('amplitude', amplitude)
        non_dom_amplitude = loaded.get('non_dom_amplitude', non_dom_amplitude)
        spontaneous_firing_rates = loaded.get('spontaneous_firing_rates', spontaneous_firing_rates)

        self.amplitude = jnp.asarray(amplitude, dtype=jnp.float32)
        self.non_dom_amplitude = jnp.asarray(non_dom_amplitude, dtype=jnp.float32)
        self.spontaneous_firing_rates = jnp.asarray(spontaneous_firing_rates, dtype=jnp.float32)

        # spatial filters & indices
        spatial_path = os.path.join(root_dir, 'tf_data',
                                    f'spatial_kernels_{col_size}x{row_size}.pkl')
        with open(spatial_path, 'rb') as f:
            sp_loaded = pkl.load(f)

        # Match TensorFlow path exactly: use preprocessed coordinates from spatial cache.
        self.x = jnp.asarray(sp_loaded['x'], dtype=jnp.float32)
        self.y = jnp.asarray(sp_loaded['y'], dtype=jnp.float32)
        self.non_dominant_x = jnp.asarray(sp_loaded['non_dominant_x'], dtype=jnp.float32)
        self.non_dominant_y = jnp.asarray(sp_loaded['non_dominant_y'], dtype=jnp.float32)

        self.gaussian_filters = [jnp.asarray(gf, dtype=jnp.float32) for gf in sp_loaded['gaussian_filters']]
        self.spatial_range_indices = [jnp.asarray(idxs, dtype=jnp.int32)
                                      for idxs in sp_loaded['spatial_range_indices']]
        self.sorted_neuron_ids_indices = jnp.asarray(sp_loaded['sorted_neuron_ids_indices'],
                                                     dtype=jnp.int32)

    @bm.jit(static_argnums=0)
    def spatial_response(self, movie: jnp.ndarray):
        """
        movie shape: [T, H, W]
        returns:
          dom_spatial: [T, N]
          non_dom_spatial: [T, N]
        """
        all_dom = []
        all_non_dom = []

        for idxs, gf in zip(self.spatial_range_indices, self.gaussian_filters):
            # convolve with gaussian filter
            lhs = movie[..., None]   # add channel
            # Cached filters are typically already [H, W, I, O] from TensorFlow.
            rhs = gf if gf.ndim == 4 else gf[..., None, None]

            conv = jax.lax.conv_general_dilated(
                lhs, rhs,
                window_strides=(1, 1),
                padding='SAME',
                dimension_numbers=('NHWC','HWIO','NHWC')
            )
            convolved_movie = conv[..., 0]  # drop channel

            if self.bmtk_compat:
                ones = jnp.ones_like(movie)
                norm = jax.lax.conv_general_dilated(
                    ones[..., None], rhs,
                    window_strides=(1, 1),
                    padding='SAME',
                    dimension_numbers=('NHWC','HWIO','NHWC')
                )
                convolved_movie = convolved_movie / norm[..., 0]

            dom_resp = select_spatial(self.x[idxs], self.y[idxs], convolved_movie)
            non_dom_resp = select_spatial(self.non_dominant_x[idxs],
                                          self.non_dominant_y[idxs],
                                          convolved_movie)

            all_dom.append(dom_resp)
            all_non_dom.append(non_dom_resp)

        all_dom = jnp.concatenate(all_dom, axis=1)
        all_non_dom = jnp.concatenate(all_non_dom, axis=1)

        all_dom = all_dom[:, self.sorted_neuron_ids_indices]
        all_non_dom = all_non_dom[:, self.sorted_neuron_ids_indices]

        return all_dom, all_non_dom

    @bm.jit(static_argnums=0)
    def firing_rates_from_spatial(self,
                                  dom_spatial: jnp.ndarray,
                                  non_dom_spatial: jnp.ndarray):
        dom_filtered = temporal_filter(dom_spatial, self.dom_temporal_kernels)
        non_dom_filtered = temporal_filter(non_dom_spatial, 
                                           self.non_dom_temporal_kernels)

        dom_rates = transfer_function(dom_filtered * self.amplitude
                                      + self.spontaneous_firing_rates)
        non_dom_rates = transfer_function(non_dom_filtered * self.non_dom_amplitude
                                          + self.spontaneous_firing_rates)

        return dom_rates + self.is_composite * non_dom_rates

    def __call__(self, movie: jnp.ndarray):
        dom_spatial, non_dom_spatial = self.spatial_response(movie)
        return self.firing_rates_from_spatial(dom_spatial, non_dom_spatial)
