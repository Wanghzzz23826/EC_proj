"""
Temporal-LGN + 4-receptor-background variant of run_vcd.py.

Key differences from run_vcd_lgn.py:
- input connectivity is loaded through load_sparse_map() with a fixed LGN mapping
- each stimulus is a temporal LGN sequence of shape [im_slice, n_input_neurons]
- background drive is configurable and defaults to receptor-aligned background current
  instead of relying on a uniform scalar default current during blank periods
- a TensorFlow-style stochastic background mode is also available via receptor_noise

This script is intentionally standalone and does not modify any existing module.
"""

import os
import types
import hashlib
from pathlib import Path

os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES",
    os.environ.get(
        "RUN_VCD_4REC_CUDA_VISIBLE_DEVICES",
        os.environ.get("RUN_VCD_LGN_CUDA_VISIBLE_DEVICES", "0"),
    ),
)
os.environ.setdefault(
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    os.environ.get(
        "RUN_VCD_4REC_XLA_PREALLOCATE",
        os.environ.get("RUN_VCD_LGN_XLA_PREALLOCATE", "false"),
    ),
)
os.environ.setdefault(
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    os.environ.get(
        "RUN_VCD_4REC_XLA_ALLOCATOR",
        os.environ.get("RUN_VCD_LGN_XLA_ALLOCATOR", "platform"),
    ),
)
os.environ.setdefault(
    "TF_FORCE_GPU_ALLOW_GROWTH",
    os.environ.get(
        "RUN_VCD_4REC_TF_ALLOW_GROWTH",
        os.environ.get("RUN_VCD_LGN_TF_ALLOW_GROWTH", "true"),
    ),
)

import datetime
import gc

import brainpy as bp
import brainpy.math as bm
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from flax.core import FrozenDict
from omegaconf import OmegaConf
from tqdm import tqdm

from brainpy_impl.load_sparse_map import load_sparse_map, reduce_lgn_cache
from brainpy_impl.models2 import BillehColumn
import brainpy_impl.stim_dataset as stim_dataset_module
from brainpy_impl.stim_dataset import (
    _convert_rates_to_vcd_input,
    _resolve_vcd_lgn_cache_file,
    ensure_vcd_lgn_cache,
)
from classification_tools import (
    InputLayer,
    Model,
    apply_scales,
    build_default_current_add,
    build_readout_indices_xz_centers,
    build_readout_neurons_xz_nearest_max_group,
    delete_small_weights,
    recompute_synapse_weights_from_vtarget,
    reset_synapse_weights_by_mean_w,
    reset_synapse_weights_with_rho,
)
from common.types import InputParams, NodeParams, SynapseParams, to_brainpy_csr
import rsrp


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = os.environ.get(
    "GLIF_V1_DATA_DIR",
    "/root/autodl-tmp/EC_proj/data/GLIF_V1_network",
)
DEFAULT_MAPPING_PATH = PROJECT_DIR / ".cache" / "lgn_reduction" / "used_lgn_channels.npy"
DEFAULT_LGN_CACHE_DIR = PROJECT_DIR / ".cache" / "pair_match_lgn"


def _install_native_lgn_backend():
    """Route stim_dataset LGN requests through the native BrainPy/JAX LGN."""

    class _NativeLGNCompat:
        def __init__(
            self,
            row_size=80,
            col_size=120,
            data_dir="GLIF_network",
            n_input=None,
            dtype=tf.float32,
        ):
            from brainpy_impl.lgn import LGN as NativeLGN

            self.dtype = dtype
            self._impl = NativeLGN(
                row_size=int(row_size),
                col_size=int(col_size),
                data_dir=str(data_dir),
                n_input=n_input,
            )

        def _normalize_movie(self, movie):
            if isinstance(movie, tf.Tensor):
                movie = movie.numpy()
            movie = np.asarray(movie, dtype=np.float32)
            if movie.ndim == 4:
                if movie.shape[-1] != 1:
                    raise ValueError(
                        f"Expected LGN movie with last channel 1, got shape={movie.shape!r}."
                    )
                movie = movie[..., 0]
            elif movie.ndim != 3:
                raise ValueError(f"Expected LGN movie shape [T, H, W] or [T, H, W, 1], got {movie.shape!r}.")
            return movie

        def spatial_response(self, movie, bmtk_compat=True):
            del bmtk_compat
            movie = self._normalize_movie(movie)
            return self._impl.spatial_response(jnp.asarray(movie, dtype=jnp.float32))

        def firing_rates_from_spatial(self, dom_spatial, non_dom_spatial):
            return self._impl.firing_rates_from_spatial(
                jnp.asarray(dom_spatial, dtype=jnp.float32),
                jnp.asarray(non_dom_spatial, dtype=jnp.float32),
            )

        def __call__(self, movie):
            movie = self._normalize_movie(movie)
            return self._impl(jnp.asarray(movie, dtype=jnp.float32))

    stim_dataset_module._LGN_MODEL_MODULE = types.SimpleNamespace(LGN=_NativeLGNCompat)


def _compute_timesteps(schedule_mode, gray_delay, im_slice, readout_period):
    mode = str(schedule_mode).strip().lower()
    if mode == "seperate":
        timesteps = 2 * (gray_delay + im_slice) + gray_delay + readout_period
    elif mode == "within":
        timesteps = 2 * im_slice
    else:
        raise ValueError(
            f"Unsupported schedule_mode={schedule_mode!r}. Expected 'seperate' or 'within'."
        )
    return int(timesteps)


def build_chunk_schedule(schedule_mode, n_batch, timesteps, gray_delay, im_slice, readout_period):
    mode = str(schedule_mode).strip().lower()
    t = jnp.arange(timesteps, dtype=jnp.int32)
    if mode == "seperate":
        edges = jnp.array(
            [gray_delay, gray_delay + im_slice, 2 * gray_delay + im_slice, 2 * gray_delay + 2 * im_slice],
            dtype=jnp.int32,
        )
        stage = jnp.searchsorted(edges, t, side="right")
        input_mode = jnp.array([0, 1, 0, 2, 0], dtype=jnp.int32)[stage]
        img_t = jnp.where(
            stage == 1,
            t - gray_delay,
            jnp.where(stage == 3, t - (2 * gray_delay + im_slice), 0),
        ).astype(jnp.int32)
    elif mode == "within":
        is_img2 = (t >= im_slice).astype(jnp.int32)
        input_mode = 1 + is_img2
        img_t = jnp.where(is_img2 == 0, t, t - im_slice).astype(jnp.int32)
    else:
        raise ValueError(
            f"Unsupported schedule_mode={schedule_mode!r}. Expected 'seperate' or 'within'."
        )

    in_readout = (t >= (timesteps - readout_period)).astype(jnp.int32)
    return jnp.stack(
        [
            jnp.repeat(jnp.arange(n_batch, dtype=jnp.int32), timesteps),
            jnp.tile(input_mode, n_batch),
            jnp.tile(img_t, n_batch),
            jnp.tile(in_readout, n_batch),
        ],
        axis=1,
    )


def set_random_seeds(seed):
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))
    bm.random.seed(int(seed))


def build_orientation_bank(center, delta, step):
    step = float(step)
    delta = float(delta)
    if step <= 0.0:
        raise ValueError(f"orientation_step must be > 0, got {step!r}")
    if delta <= 0.0:
        raise ValueError(f"orientation_delta must be > 0, got {delta!r}")
    n_side = max(1, int(round(delta / step)))
    signed_steps = np.concatenate(
        [np.arange(-n_side, 0, dtype=np.int32), np.arange(1, n_side + 1, dtype=np.int32)]
    )
    return (float(center) + signed_steps.astype(np.float32) * step).astype(np.float32)


def _projected_cache_fingerprint(*arrays, tokens=()):
    hasher = hashlib.sha1()
    for token in tokens:
        hasher.update(str(token).encode("utf-8"))
        hasher.update(b"\0")
    for arr in arrays:
        arr_np = np.asarray(arr)
        hasher.update(np.asarray(arr_np.shape, dtype=np.int64).tobytes())
        hasher.update(str(arr_np.dtype).encode("utf-8"))
        hasher.update(arr_np.tobytes())
    return hasher.hexdigest()[:12]


class PairMatchLGNBank:
    def __init__(self, conf, selected_indices, orientation_values, projected_cache=None):
        self.selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
        self.orientation_values = np.asarray(orientation_values, dtype=np.float32).reshape(-1)
        if self.orientation_values.size == 0:
            raise ValueError("orientation_values must be non-empty.")

        self.current_input = str(conf.input_mode).strip().lower() == "current"
        self.cache_kind = str(conf.lgn_cache_kind).strip().lower()
        if self.cache_kind not in {"rates", "input"}:
            raise ValueError(
                f"Unsupported lgn_cache_kind={conf.lgn_cache_kind!r}. Expected 'rates' or 'input'."
            )
        if self.cache_kind == "input" and not self.current_input:
            raise ValueError(
                "lgn_cache_kind='input' only matches input_mode='current'. "
                "Use lgn_cache_kind='rates' for spike input."
            )

        cache_info = ensure_vcd_lgn_cache(
            data_dir=str(conf.data_dir),
            lgn_cache_dir=str(conf.lgn_cache_dir),
            intensity=float(conf.lgn_intensity),
            im_slice=int(conf.im_slice),
            pre_delay=0,
            post_delay=0,
            orientation_values=self.orientation_values,
            num_phases=int(conf.lgn_cache_num_phases),
            phase_values=None
            if conf.lgn_cache_phase_values is None
            else tuple(float(v) for v in conf.lgn_cache_phase_values),
            cache_kind=self.cache_kind,
            current_input=self.current_input,
            overwrite=False,
            verbose=False,
        )

        self.cache_dir = Path(cache_info["cache_dir"])
        self.phase_values = np.asarray(cache_info["phase_values"], dtype=np.float32).reshape(-1)
        self.cache_in_memory = bool(getattr(conf, "cache_in_memory", False))
        self._base_cache = {}

        self.orientation_keys = np.rint(self.orientation_values * 10.0).astype(np.int32)
        self._orientation_key_to_idx = {
            int(key): idx for idx, key in enumerate(self.orientation_keys.tolist())
        }
        self.projected_cache_enabled = False
        self.projected_cache_dir = None
        self.projected_cache_in_memory = False
        self.projected_cache_dtype = np.float16
        self.projected_output_dim = None
        self._projected_cache = {}
        self._project_sequence = None
        if projected_cache is not None and bool(projected_cache.get("enabled", False)):
            self.projected_cache_enabled = True
            self.projected_cache_dir = Path(str(projected_cache["cache_dir"])).expanduser().resolve()
            self.projected_cache_dir.mkdir(parents=True, exist_ok=True)
            self.projected_cache_in_memory = bool(projected_cache.get("cache_in_memory", False))
            self.projected_cache_dtype = np.dtype(projected_cache.get("disk_dtype", np.float16))
            self.projected_output_dim = int(projected_cache["output_dim"])
            self._project_sequence = projected_cache["project_sequence"]

    def summary(self):
        summary = {
            "cache_dir": str(self.cache_dir),
            "cache_kind": self.cache_kind,
            "num_orientations": int(self.orientation_values.size),
            "num_phases": int(self.phase_values.size),
            "sample_dim": int(
                self.projected_output_dim
                if self.projected_cache_enabled
                else len(self.selected_indices)
            ),
            "projected_cache_enabled": bool(self.projected_cache_enabled),
        }
        if self.projected_cache_enabled:
            summary["projected_cache_dir"] = str(self.projected_cache_dir)
        return summary

    def sample_orientation_key(self, rng):
        idx = int(rng.randint(self.orientation_keys.size))
        return int(self.orientation_keys[idx])

    def sample_different_orientation_key(self, rng, ref_key):
        ref_idx = self._orientation_key_to_idx[int(ref_key)]
        shift = int(rng.randint(1, self.orientation_keys.size))
        return int(self.orientation_keys[(ref_idx + shift) % self.orientation_keys.size])

    def sample_phase_index(self, rng):
        if self.phase_values.size == 1:
            return 0
        return int(rng.randint(self.phase_values.size))

    def _load_base_value(self, orientation_key, phase_idx):
        cache_key = (int(orientation_key), int(phase_idx))
        arr = self._base_cache.get(cache_key)
        if arr is not None:
            return arr

        orientation = float(int(orientation_key)) / 10.0
        full = np.load(
            _resolve_vcd_lgn_cache_file(self.cache_dir, orientation, int(phase_idx)),
            mmap_mode="r",
        )
        arr = reduce_lgn_cache(full, self.selected_indices).astype(np.float32, copy=False)
        if self.cache_in_memory:
            self._base_cache[cache_key] = arr
        return arr

    def _load_current_sequence(self, orientation_key, phase_idx):
        base = self._load_base_value(orientation_key, phase_idx)
        if self.cache_kind == "input":
            return np.asarray(base, dtype=np.float32)
        return _convert_rates_to_vcd_input(base)

    def _projected_cache_file(self, orientation_key, phase_idx):
        return self.projected_cache_dir / f"ori_{int(orientation_key)}_phase_{int(phase_idx):03d}.npy"

    def _load_projected_value(self, orientation_key, phase_idx):
        cache_key = (int(orientation_key), int(phase_idx))
        arr = self._projected_cache.get(cache_key)
        if arr is not None:
            return arr

        cache_file = self._projected_cache_file(orientation_key, phase_idx)
        if cache_file.exists():
            projected = np.load(cache_file, mmap_mode="r")
        else:
            current_seq = self._load_current_sequence(orientation_key, phase_idx)
            projected_f32 = self._project_sequence(current_seq)
            np.save(cache_file, projected_f32.astype(self.projected_cache_dtype, copy=False))
            projected = projected_f32
        arr = np.asarray(projected, dtype=np.float32)
        if self.projected_cache_in_memory:
            self._projected_cache[cache_key] = arr
        return arr

    def sample_sequence(self, rng, orientation_key):
        phase_idx = self.sample_phase_index(rng)
        if self.projected_cache_enabled:
            return self._load_projected_value(orientation_key, phase_idx)
        base = self._load_base_value(orientation_key, phase_idx)

        if self.cache_kind == "input":
            return np.asarray(base, dtype=np.float32)

        if self.current_input:
            return _convert_rates_to_vcd_input(base)

        p = 1.0 - np.exp(-np.asarray(base, dtype=np.float32) / 1000.0)
        return (rng.uniform(size=p.shape) < p).astype(np.float32)


def build_grating_lgn_pair_match_dataset(conf, selected_indices, *, seed, projected_cache=None):
    orientation_values = build_orientation_bank(
        conf.orientation_center,
        conf.orientation_delta,
        conf.orientation_step,
    )
    bank = PairMatchLGNBank(conf, selected_indices, orientation_values, projected_cache=projected_cache)

    invocation_state = {"count": 0}
    sample_dim = (
        int(projected_cache["output_dim"])
        if projected_cache is not None and bool(projected_cache.get("enabled", False))
        else int(len(selected_indices))
    )
    input_shape = (int(conf.im_slice), sample_dim)

    def _generator():
        local_seed = int(seed) + 9973 * invocation_state["count"]
        invocation_state["count"] += 1
        rng = np.random.RandomState(local_seed)

        while True:
            orientation_key1 = bank.sample_orientation_key(rng)
            same_flag = bool(rng.uniform() < 0.5)
            orientation_key2 = (
                orientation_key1
                if same_flag
                else bank.sample_different_orientation_key(rng, orientation_key1)
            )
            x1 = bank.sample_sequence(rng, orientation_key1)
            x2 = bank.sample_sequence(rng, orientation_key2)
            label = np.int32(1 if same_flag else 0)
            yield x1.astype(np.float32, copy=False), x2.astype(np.float32, copy=False), label

    dataset = tf.data.Dataset.from_generator(
        _generator,
        output_signature=(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if bool(getattr(conf, "prefetch_dataset", True)):
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, bank.summary()


def _mean_tree_entry(value):
    if isinstance(value, (list, tuple)):
        flat = [jnp.ravel(jnp.asarray(v)) for v in value]
        if not flat:
            return None
        return float(jnp.mean(jnp.concatenate(flat)))
    arr = jnp.asarray(value)
    return float(jnp.mean(arr))


def _resolve_readout_neurons(network, n_output_max):
    if "readout_neuron_sel" in network:
        return jnp.where(network["readout_neuron_sel"] == 1)[0]

    if "l5e_neuron_sel" not in network:
        raise KeyError(
            "network is missing both 'readout_neuron_sel' and 'l5e_neuron_sel', "
            "so run_vcd_lgn.py cannot build readout neurons."
        )
    if "x" not in network or "z" not in network:
        raise KeyError(
            "network is missing 'x'/'z', so run_vcd_lgn.py cannot build fallback readout neurons."
        )

    l5e_neuron = np.where(np.asarray(network["l5e_neuron_sel"]) == 1)[0]
    if l5e_neuron.size == 0:
        raise ValueError("No L5e neurons found in network['l5e_neuron_sel']; cannot build readout.")

    x = np.asarray(network["x"], dtype=np.float32)
    z = np.asarray(network["z"], dtype=np.float32)
    dist2 = np.square(x[l5e_neuron]) + np.square(z[l5e_neuron])
    k = min(int(n_output_max), int(l5e_neuron.size))
    chosen = l5e_neuron[np.argsort(dist2)[:k]]
    print(
        f"[LGN] network has no readout_neuron_sel; using fallback nearest-center L5e readout set "
        f"with {k} neurons."
    )
    return jnp.asarray(chosen, dtype=jnp.int32)


def main():
    _install_native_lgn_backend()

    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(
         {
            "task": "grating pair match (LGN temporal)",
            "data_dir": str(DEFAULT_DATA_DIR),
            "output_dir": "output",
            "pretrained_model": None,
            "filename": None,
            # network setting
            "DT": 1.0,
            "n_neurons": 2000,
            "n_input_neurons": 861,
            "n_output_max": 2000,
            "mode": "0/1",
            "k_in": 1.3,
            "k_h": 1.5,
            "k_b": 0.15,
            # input receptor-wise scales
            "k_in_ee": 1.0,
            "k_in_ei": 1.0,
            # recurrent/synapse receptor-wise scales
            "k_rec_ee": 0.35,
            "k_rec_ie": 0.7,
            "k_rec_ei": 0.7,
            "k_rec_ii": 0.7,
            # temporal LGN input setting
            "input_mode": "current",
            "schedule_mode": "seperate",
            "im_slice": 80,
            "gray_delay": 50,
            "readout_period": 100,
            "default_current": 0.0,
            "default_current_to_receptor": False,
            "default_input_to_receptor": True,
            "rearrange_input": False,
            "background_mode": "receptor_current",
            "background_during_stimulus": False,
            "preproject_current_batch": True,
            "preproject_current_train_batch": False,
            # LGN reduction/cache setting
            "lgn_mapping_path": str(DEFAULT_MAPPING_PATH),
            "lgn_cache_dir": str(DEFAULT_LGN_CACHE_DIR),
            "lgn_cache_kind": "input",
            "lgn_cache_num_phases": 8,
            "lgn_cache_phase_values": None,
            "lgn_intensity": 1.8,
            "cache_in_memory": False,
            "use_projected_current_cache_train": False,
            "projected_cache_dir": str(PROJECT_DIR / ".cache" / "pair_match_projected_current_4rec"),
            "projected_cache_in_memory": False,
            "prefetch_dataset": True,
            # batch/pop setting
            "batch_size": 100,
            "total_batch_size": 100,
            "pop_size": 1000,
            "total_pop_size": 1000,
            "steps_per_epoch": 1000,
            "eval_batches": 100,
            # training setting
            "seed": 42,
            "lr": 2.0,
            "eps": 1e-3,
            "n_epoch": 2,
            "wandb": True,
            "wandb_entity": None,
            "wandb_project": "rsrp_allen_v1_vcd",
            "eval_interval": 20,
            "reward_type": "accuracy_margin",
            "reward_margin_weight": 0.2,
            "reward_margin_temp": 1.0,
            "wrong_logit_penalty": 0.2,
            "class_weights": None,
            "min_weight": 0.0,
            "mean_synapse_weights": False,
            "mean_synapse_weights_recompute": False,
            "rho_rec_init": 0.25,
            "reset_synapse_weights_with_rho": False,
            # default false to keep temporal LGN throughput sane
            "train_input": True,
            "train_recurrent": True,
            "sigmoid": False,
            "readout": "nearest_max_group",
            "group_size_max": 60,
            # ES
            "es_lr": 0.01,
            "es_sigma_schedule": "constant",
            "es_sigma_init": 0.02,
            "es_sigma_final": 0.001,
            "train_es": True,
            "es_sp_rate_target": 5.0,
            "es_sp_rate_penalty": 0.06,
            # grating task params
            "orientation_center": 135.0,
            "orientation_delta": 15.0,
            "orientation_step": 0.1,
            "orientation_spatial_freq": 0.1,
            "orientation_contrast": 1.0,
        },  
        cli_conf,
    )

    if str(conf.input_mode).strip().lower() not in {"current", "spike"}:
        raise ValueError(
            f"Unsupported input_mode={conf.input_mode!r}. Expected 'current' or 'spike'."
        )
    conf.input_mode = str(conf.input_mode).strip().lower()
    conf.background_mode = str(conf.background_mode).strip().lower()
    if conf.background_mode not in {"constant", "receptor_current", "receptor_noise"}:
        raise ValueError(
            f"Unsupported background_mode={conf.background_mode!r}. "
            "Expected 'constant', 'receptor_current', or 'receptor_noise'."
        )
    if str(conf.input_mode).strip().lower() == "spike" and str(conf.lgn_cache_kind).strip().lower() == "input":
        conf.lgn_cache_kind = "rates"
    if bool(conf.rearrange_input):
        raise ValueError("run_vcd_4rec.py requires rearrange_input=False for fixed reduced LGN mapping.")
    if bool(conf.use_projected_current_cache_train):
        if conf.input_mode != "current":
            raise ValueError("use_projected_current_cache_train only supports input_mode='current'.")
        if bool(conf.train_input):
            raise ValueError("use_projected_current_cache_train requires train_input=False.")
        if conf.background_mode == "receptor_noise":
            raise ValueError(
                "use_projected_current_cache_train does not support background_mode='receptor_noise' "
                "because the stimulus background is stochastic."
            )

    result = {
        "step": 0,
        "acc": 0,
        "train_acc": 0,
        "mean_rho": [],
        "sp_rate": 0,
    }

    if conf.pretrained_model:
        data_load = np.load(conf.pretrained_model, allow_pickle=True).item()
        conf_saved = data_load["conf"]
        conf_current = conf.copy()
        conf = OmegaConf.merge(conf, conf_saved)
        conf.pretrained_model = conf_current.pretrained_model
        if conf.train_es:
            conf.k_in_ee *= float(data_load["es_params"]["k_in_ee"])
            conf.k_in_ei *= float(data_load["es_params"]["k_in_ei"])
            conf.k_rec_ee *= float(data_load["es_params"]["k_rec_ee"])
            conf.k_rec_ie *= float(data_load["es_params"]["k_rec_ie"])
            conf.k_rec_ei *= float(data_load["es_params"]["k_rec_ei"])
            conf.k_rec_ii *= float(data_load["es_params"]["k_rec_ii"])
        result["step"] = data_load["result"]["step"]
        print(conf)

    mapping_path = Path(str(conf.lgn_mapping_path)).expanduser().resolve()
    if not mapping_path.exists():
        raise FileNotFoundError(f"LGN mapping file not found: {mapping_path}")
    selected_indices = np.asarray(np.load(mapping_path), dtype=np.int64).reshape(-1)
    if selected_indices.size == 0:
        raise ValueError(f"Empty LGN mapping: {mapping_path}")
    conf.n_input_neurons = int(selected_indices.size)

    conf.timesteps = _compute_timesteps(
        schedule_mode=conf.schedule_mode,
        gray_delay=conf.gray_delay,
        im_slice=conf.im_slice,
        readout_period=conf.readout_period,
    )
    n_pop = conf.total_pop_size // conf.pop_size
    n_batch = conf.total_batch_size // conf.batch_size
    chunk_indices = build_chunk_schedule(
        schedule_mode=conf.schedule_mode,
        n_batch=n_batch,
        timesteps=conf.timesteps,
        gray_delay=conf.gray_delay,
        im_slice=conf.im_slice,
        readout_period=conf.readout_period,
    )

    set_random_seeds(conf.seed)
    key = jax.random.PRNGKey(conf.seed)
    current_time = datetime.datetime.now()
    conf.filename = os.path.join(conf.output_dir, current_time.strftime("%m%d_%H%M"))
    os.makedirs(conf.output_dir, exist_ok=True)

    if conf.wandb:
        import wandb

        init_kwargs = {
            "project": str(conf.wandb_project),
            "name": current_time.strftime("%m%d_%H%M"),
            "config": OmegaConf.to_container(conf),
        }
        if conf.wandb_entity is not None:
            init_kwargs["entity"] = str(conf.wandb_entity)
        wandb.init(**init_kwargs)

    input_population, network, bkg_weights = load_sparse_map(
        n_input=conf.n_input_neurons,
        n_neurons=conf.n_neurons,
        core_only=True,
        data_dir=str(conf.data_dir),
        seed=3000,
        connected_selection=True,
        n_output=conf.n_output_max,
        lgn_mapping=selected_indices,
    )
    print(
        f"[LGN] connected_compact={network.get('lgn_connected_count', 'NA')} "
        f"disconnected_compact={network.get('lgn_disconnected_count', 'NA')}"
    )
    print(
        f"[4REC] background_mode={conf.background_mode} "
        f"background_during_stimulus={bool(conf.background_during_stimulus)} "
        f"k_b={float(conf.k_b)} default_current={float(conf.default_current)} "
        f"default_current_to_receptor={bool(conf.default_current_to_receptor)}"
    )

    node_params = NodeParams.from_network_node_params(
        network["node_params"],
        network["node_type_ids"],
        dt=conf.DT,
    )
    synapse_params = SynapseParams.from_network_synapses(
        network["synapses"],
        network["n_nodes"],
        network["n_edges"],
        node_params.n_receptors,
        1,
        node_params.dt,
    )
    synapse_params = delete_small_weights(synapse_params, conf.min_weight)

    input_params = InputParams.from_input_node_bkg(
        input_population,
        node_params,
        np.asarray(conf.k_b * bkg_weights, dtype=np.float32),
    )

    scale_params_input = {
        "k_ee": conf.k_in_ee,
        "k_ei": conf.k_in_ei,
        "k": conf.k_in,
    }
    scale_params_synapse = {
        "k_ee": conf.k_rec_ee,
        "k_ie": conf.k_rec_ie,
        "k_ei": conf.k_rec_ei,
        "k_ii": conf.k_rec_ii,
        "k": conf.k_h,
    }
    input_params = apply_scales(input_params, scale_params_input)
    synapse_params = apply_scales(synapse_params, scale_params_synapse)

    if conf.mean_synapse_weights:
        synapse_params = reset_synapse_weights_by_mean_w(synapse_params, network, conf.rho_rec_init)
    if conf.mean_synapse_weights_recompute:
        synapse_params = recompute_synapse_weights_from_vtarget(
            synapse_params,
            node_params,
            network,
            conf.rho_rec_init,
        )

    _input_csr, _ = to_brainpy_csr(input_params, split_receptor=False, split_conn=True)
    input_scale_indices = jnp.asarray(_input_csr.indices % input_params.n_receptors, dtype=jnp.int32)
    _rec_csr, _ = to_brainpy_csr(synapse_params, split_receptor=False, split_conn=True)
    recurrent_scale_indices = [
        jnp.asarray(_rec_csr.indices % synapse_params.n_receptors, dtype=jnp.int32)
    ]

    n_class = 2
    readout_neuron = _resolve_readout_neurons(network, conf.n_output_max)
    group_size = readout_neuron.shape[-1] // n_class
    if conf.readout == "random":
        readout_neuron = readout_neuron[: group_size * n_class]
        readout_indices = np.random.permutation(readout_neuron.shape[-1])
    elif conf.readout == "center":
        readout_neuron = readout_neuron[: group_size * n_class]
        readout_indices = build_readout_indices_xz_centers(
            network=network,
            readout_neuron=readout_neuron,
            n_class=n_class,
            group_size=group_size,
            seed=int(conf.seed),
        )
    elif conf.readout == "nearest_max_group":
        readout_neuron, group_size, readout_indices = build_readout_neurons_xz_nearest_max_group(
            network=network,
            readout_neuron=readout_neuron,
            n_class=n_class,
            seed=int(conf.seed),
            group_size_max_custom=conf.group_size_max,
        )
    else:
        raise ValueError(f"Unknown conf.readout={conf.readout!r}")

    def build_input_layer():
        with bm.environment(mode=bm.BatchingMode(batch_size=conf.batch_size)):
            return InputLayer(
                input_params,
                tau_syn=node_params.tau_syn,
                use_dale_law=True,
                sparse_mode="value" if conf.input_mode == "current" else "event",
                use_decoded_noise=False,
                noise_data=None,
            )

    def build_model():
        with bm.environment(mode=bm.BatchingMode(batch_size=conf.batch_size)):
            recurrent_layer = BillehColumn(
                node_params,
                synapse_params,
                use_dale_law=True,
                default_input_to_receptor=conf.default_input_to_receptor,
                spk_reset="hard",
                delay_off=True,
                batch_size=conf.batch_size,
            )
            input_layer = build_input_layer()
            model = Model(recurrent_layer=recurrent_layer, input_layer=input_layer)
        return model

    projected_cache_train = None
    if bool(conf.use_projected_current_cache_train):
        projection_layer = build_input_layer()
        projected_dim = int(np.asarray(projection_layer.bkg_weights).reshape(-1).shape[0])
        projected_tag = _projected_cache_fingerprint(
            np.asarray(projection_layer.input_proj.weight.value),
            np.asarray(projection_layer.bkg_weights),
            tokens=(
                conf.k_in,
                conf.k_in_ee,
                conf.k_in_ei,
                conf.k_b,
                conf.default_current,
                conf.background_mode,
                bool(conf.background_during_stimulus),
            ),
        )
        projected_bucket = (
            Path(str(conf.projected_cache_dir)).expanduser().resolve()
            / f"map{conf.n_input_neurons}_im{int(conf.im_slice)}_proj{projected_dim}_{projected_tag}"
        )

        def _project_sequence_for_cache(seq):
            seq_bm = bm.asarray(seq, dtype=bm.float_)
            projected = projection_layer.input_proj(seq_bm)
            if bool(conf.background_during_stimulus):
                if conf.background_mode == "constant":
                    projected = projected + build_gray_background_from_input_layer(projection_layer)
                else:
                    projected = projected + build_receptor_background_from_input_layer(projection_layer)
            return np.asarray(jax.device_get(projected), dtype=np.float32)

        projected_cache_train = {
            "enabled": True,
            "cache_dir": str(projected_bucket),
            "cache_in_memory": bool(conf.projected_cache_in_memory),
            "disk_dtype": np.float16,
            "output_dim": projected_dim,
            "project_sequence": _project_sequence_for_cache,
        }

    dataset_train, dataset_meta = build_grating_lgn_pair_match_dataset(
        conf,
        selected_indices,
        seed=int(conf.seed),
        projected_cache=projected_cache_train,
    )
    dataset_test, _ = build_grating_lgn_pair_match_dataset(
        conf,
        selected_indices,
        seed=int(conf.seed) + 100_000,
        projected_cache=None,
    )
    print(
        f"[LGN] mapping={mapping_path} n_input={conf.n_input_neurons} "
        f"cache_dir={dataset_meta['cache_dir']} cache_kind={dataset_meta['cache_kind']} "
        f"orientations={dataset_meta['num_orientations']} phases={dataset_meta['num_phases']} "
        f"sample_dim={dataset_meta['sample_dim']} projected_cache={dataset_meta['projected_cache_enabled']}"
    )
    if dataset_meta.get("projected_cache_dir") is not None:
        print(f"[LGN] projected_current_cache_train={dataset_meta['projected_cache_dir']}")

    dataset_batch = dataset_train.batch(conf.total_batch_size, drop_remainder=True).take(conf.steps_per_epoch)
    test_batch = list(dataset_test.batch(conf.total_batch_size, drop_remainder=True).take(conf.eval_batches))

    def build_gray_background_from_input_layer(input_layer):
        if bool(getattr(conf, "default_current_to_receptor", False)):
            default_add = build_default_current_add(
                default_current=conf.default_current,
                n_neurons=conf.n_neurons,
                n_receptors=synapse_params.n_receptors,
                laminar_indices=network["laminar_indices"],
                dtype=input_layer.bkg_weights.dtype,
            )
            return input_layer.bkg_weights + default_add
        return input_layer.bkg_weights + conf.default_current

    def build_gray_background(model):
        return build_gray_background_from_input_layer(model.input_layer)

    def build_receptor_background_from_input_layer(input_layer):
        background = input_layer.bkg_weights
        if bool(getattr(conf, "default_current_to_receptor", False)) and float(conf.default_current) != 0.0:
            default_add = build_default_current_add(
                default_current=conf.default_current,
                n_neurons=conf.n_neurons,
                n_receptors=synapse_params.n_receptors,
                laminar_indices=network["laminar_indices"],
                dtype=input_layer.bkg_weights.dtype,
            )
            background = background + default_add
        elif float(conf.default_current) != 0.0:
            background = background + conf.default_current
        return background

    def build_receptor_background(model):
        return build_receptor_background_from_input_layer(model.input_layer)

    def build_background_noise(input_layer, prefix_shape):
        noise = bm.random.poisson(1.0, size=prefix_shape)
        return jnp.asarray(noise[..., None] * input_layer.bkg_weights)

    def as_switch_input(x):
        return jnp.asarray(x)

    def build_blank_background(model):
        if conf.background_mode == "constant":
            background = build_gray_background(model)
            return as_switch_input(jnp.broadcast_to(background, (conf.batch_size, background.shape[0])))
        if conf.background_mode == "receptor_current":
            background = build_receptor_background(model)
            return as_switch_input(jnp.broadcast_to(background, (conf.batch_size, background.shape[0])))
        return as_switch_input(build_background_noise(model.input_layer, (conf.batch_size,)))

    def reshape_temporal_input(x):
        x = jnp.asarray(x)
        return x.reshape(conf.batch_size, n_batch, conf.im_slice, x.shape[-1]).transpose(1, 2, 0, 3)

    def temporal_input_frame(x, chunk_idx, img_t):
        x = jnp.asarray(x)
        x = x.reshape(conf.batch_size, n_batch, conf.im_slice, x.shape[-1])
        return x[:, chunk_idx, img_t, :]

    def project_stimulus_sequence(model, inp_seq):
        inp_seq = jnp.asarray(inp_seq)
        projected = as_switch_input(model.input_layer.input_proj(inp_seq))
        if not bool(conf.background_during_stimulus):
            return as_switch_input(projected)
        if conf.background_mode == "constant":
            background = build_gray_background(model)
            return as_switch_input(projected + jnp.broadcast_to(background, projected.shape))
        if conf.background_mode == "receptor_current":
            background = build_receptor_background(model)
            return as_switch_input(projected + jnp.broadcast_to(background, projected.shape))
        return as_switch_input(projected + build_background_noise(model.input_layer, projected.shape[:-1]))

    def project_stimulus_input(model, inp_t):
        return project_stimulus_sequence(model, inp_t)

    def prepare_current_sequence(model, x, *, inputs_are_projected=False, preproject=True):
        if not preproject:
            return as_switch_input(x)
        if inputs_are_projected:
            return as_switch_input(reshape_temporal_input(x))
        x_bt = reshape_temporal_input(x)
        return as_switch_input(project_stimulus_sequence(model, x_bt))

    def current_recurrent_input_at(model, prepared_seq, chunk_idx, img_t, *, inputs_are_projected=False, preproject=True):
        if preproject:
            return as_switch_input(prepared_seq[chunk_idx, img_t])
        frame = temporal_input_frame(prepared_seq, chunk_idx, img_t)
        if inputs_are_projected:
            return as_switch_input(frame)
        return as_switch_input(project_stimulus_input(model, frame))

    def apply_es_scales_to_weights(weights, es_params):
        scaled = {}
        if "input" in weights:
            scales_in = jnp.array(
                [
                    es_params["k_in_ee"],
                    1.0,
                    es_params["k_in_ei"],
                    1.0,
                ],
                dtype=jnp.float32,
            )
            scaled["input"] = weights["input"] * scales_in[input_scale_indices]
        if "recurrent" in weights:
            scales_rec = jnp.array(
                [
                    es_params["k_rec_ee"],
                    es_params["k_rec_ie"],
                    es_params["k_rec_ei"],
                    es_params["k_rec_ii"],
                ],
                dtype=jnp.float32,
            )
            scaled["recurrent"] = [
                weights["recurrent"][j] * scales_rec[recurrent_scale_indices[j]]
                for j in range(len(weights["recurrent"]))
            ]
        return FrozenDict(scaled)

    def run_model_eval(img1, img2, weights, inputs_are_projected=False):
        model = build_model()
        class_acc = bm.Variable(jnp.zeros((conf.batch_size, n_batch, n_class), dtype=jnp.float32))
        spike_sum = bm.Variable(jnp.asarray(0.0, dtype=jnp.float32))

        def _accumulate_step(spk, chunk_idx, in_readout):
            spk_ro = spk[:, readout_neuron]
            spk_cls = spk_ro[:, readout_indices].reshape(conf.batch_size, group_size, n_class).sum(axis=1)
            m = in_readout.astype(jnp.float32)
            class_acc.value = class_acc.value.at[:, chunk_idx, :].add(spk_cls * m)
            spike_sum.value = spike_sum.value + jnp.sum(spk)
            return jnp.asarray(0.0, dtype=jnp.float32)

        @bm.jit
        def _run_model(weights_local):
            if conf.train_input and "input" in weights_local:
                model.input_layer.input_proj.weight.value = weights_local["input"]
            if conf.train_recurrent and "recurrent" in weights_local:
                for j, proj in enumerate(model.recurrent_layer.projs):
                    proj.proj.comm.weight.value = weights_local["recurrent"][j]

            class_acc.value = jnp.zeros((conf.batch_size, n_batch, n_class), dtype=jnp.float32)
            spike_sum.value = jnp.asarray(0.0, dtype=jnp.float32)

            bkg_batch = build_blank_background(model)

            if conf.input_mode == "spike":
                img1_bt = reshape_temporal_input(img1)
                img2_bt = reshape_temporal_input(img2)

                def spike_one_step(chunk_info):
                    chunk_idx = chunk_info[0]
                    mode_id = chunk_info[1]
                    img_t = chunk_info[2]
                    in_ro = chunk_info[3]
                    recurrent_inp = jax.lax.switch(
                        mode_id,
                        [
                            lambda _: bkg_batch,
                            lambda _: project_stimulus_input(model, img1_bt[chunk_idx, img_t]),
                            lambda _: project_stimulus_input(model, img2_bt[chunk_idx, img_t]),
                        ],
                        operand=None,
                    )
                    model.recurrent_layer(recurrent_inp)
                    return _accumulate_step(model.recurrent_layer.neurons.spike.value, chunk_idx, in_ro)

                bm.for_loop(spike_one_step, chunk_indices)
            else:
                preproject_current = bool(conf.preproject_current_batch)
                current_img1 = prepare_current_sequence(
                    model,
                    img1,
                    inputs_are_projected=inputs_are_projected,
                    preproject=preproject_current,
                )
                current_img2 = prepare_current_sequence(
                    model,
                    img2,
                    inputs_are_projected=inputs_are_projected,
                    preproject=preproject_current,
                )

                def current_one_step(chunk_info):
                    chunk_idx = chunk_info[0]
                    mode_id = chunk_info[1]
                    img_t = chunk_info[2]
                    in_ro = chunk_info[3]
                    recurrent_inp = jax.lax.switch(
                        mode_id,
                        [
                            lambda _: bkg_batch,
                            lambda _: current_recurrent_input_at(
                                model,
                                current_img1,
                                chunk_idx,
                                img_t,
                                inputs_are_projected=inputs_are_projected,
                                preproject=preproject_current,
                            ),
                            lambda _: current_recurrent_input_at(
                                model,
                                current_img2,
                                chunk_idx,
                                img_t,
                                inputs_are_projected=inputs_are_projected,
                                preproject=preproject_current,
                            ),
                        ],
                        operand=None,
                    )
                    model.recurrent_layer(recurrent_inp)
                    return _accumulate_step(model.recurrent_layer.neurons.spike.value, chunk_idx, in_ro)

                bm.for_loop(current_one_step, chunk_indices)

            out_logits = class_acc.value.reshape(conf.total_batch_size, n_class)
            sp_rate = spike_sum.value / (conf.total_batch_size * conf.n_neurons * conf.timesteps) * 1000.0
            predict = jnp.argmax(out_logits, axis=-1)
            return predict, sp_rate

        return _run_model(weights)

    def run_model_train(img1, img2, weights, es_params_pop, inputs_are_projected=False):
        model = build_model()
        class_acc = bm.Variable(jnp.zeros((conf.batch_size, n_batch, n_class), dtype=jnp.float32))
        spike_sum = bm.Variable(jnp.asarray(0.0, dtype=jnp.float32))

        def _accumulate_step(spk, chunk_idx, in_readout):
            spk_ro = spk[:, readout_neuron]
            spk_cls = spk_ro[:, readout_indices].reshape(conf.batch_size, group_size, n_class).sum(axis=1)
            m = in_readout.astype(jnp.float32)
            class_acc.value = class_acc.value.at[:, chunk_idx, :].add(spk_cls * m)
            spike_sum.value = spike_sum.value + jnp.sum(spk)
            return jnp.asarray(0.0, dtype=jnp.float32)

        @bm.jit
        def _run_model(weights_local):
            if conf.train_input and "input" in weights_local:
                model.input_layer.input_proj.weight.value = weights_local["input"]
            if conf.train_recurrent and "recurrent" in weights_local:
                for j, proj in enumerate(model.recurrent_layer.projs):
                    proj.proj.comm.weight.value = weights_local["recurrent"][j]

            class_acc.value = jnp.zeros((conf.batch_size, n_batch, n_class), dtype=jnp.float32)
            spike_sum.value = jnp.asarray(0.0, dtype=jnp.float32)

            bkg_batch = build_blank_background(model)

            if conf.input_mode == "spike":
                img1_bt = reshape_temporal_input(img1)
                img2_bt = reshape_temporal_input(img2)

                def spike_one_step(chunk_info):
                    chunk_idx = chunk_info[0]
                    mode_id = chunk_info[1]
                    img_t = chunk_info[2]
                    in_ro = chunk_info[3]
                    recurrent_inp = jax.lax.switch(
                        mode_id,
                        [
                            lambda _: bkg_batch,
                            lambda _: project_stimulus_input(model, img1_bt[chunk_idx, img_t]),
                            lambda _: project_stimulus_input(model, img2_bt[chunk_idx, img_t]),
                        ],
                        operand=None,
                    )
                    model.recurrent_layer(recurrent_inp)
                    return _accumulate_step(model.recurrent_layer.neurons.spike.value, chunk_idx, in_ro)

                bm.for_loop(spike_one_step, chunk_indices)
            else:
                preproject_current = bool(conf.preproject_current_train_batch)
                current_img1 = prepare_current_sequence(
                    model,
                    img1,
                    inputs_are_projected=inputs_are_projected,
                    preproject=preproject_current,
                )
                current_img2 = prepare_current_sequence(
                    model,
                    img2,
                    inputs_are_projected=inputs_are_projected,
                    preproject=preproject_current,
                )

                def current_one_step(chunk_info):
                    chunk_idx = chunk_info[0]
                    mode_id = chunk_info[1]
                    img_t = chunk_info[2]
                    in_ro = chunk_info[3]
                    recurrent_inp = jax.lax.switch(
                        mode_id,
                        [
                            lambda _: bkg_batch,
                            lambda _: current_recurrent_input_at(
                                model,
                                current_img1,
                                chunk_idx,
                                img_t,
                                inputs_are_projected=inputs_are_projected,
                                preproject=preproject_current,
                            ),
                            lambda _: current_recurrent_input_at(
                                model,
                                current_img2,
                                chunk_idx,
                                img_t,
                                inputs_are_projected=inputs_are_projected,
                                preproject=preproject_current,
                            ),
                        ],
                        operand=None,
                    )
                    model.recurrent_layer(recurrent_inp)
                    return _accumulate_step(model.recurrent_layer.neurons.spike.value, chunk_idx, in_ro)

                bm.for_loop(current_one_step, chunk_indices)

            out_spikes = class_acc.value.reshape(conf.total_batch_size, n_class)
            sp_rate = spike_sum.value / (conf.total_batch_size * conf.n_neurons * conf.timesteps) * 1000.0
            return out_spikes, sp_rate

        logits = jnp.zeros((n_pop, conf.total_batch_size, n_class))
        sp_rates = jnp.zeros((n_pop,))
        for i in range(n_pop):
            weights_i = jax.tree_util.tree_map(lambda x: x[i], weights)
            if es_params_pop is not None:
                weights_i_scaled = {}
                if "input" in weights_i:
                    scales_in = jnp.array(
                        [
                            es_params_pop["k_in_ee"][i],
                            1.0,
                            es_params_pop["k_in_ei"][i],
                            1.0,
                        ],
                        dtype=jnp.float32,
                    )
                    weights_i_scaled["input"] = weights_i["input"] * scales_in[input_scale_indices]
                if "recurrent" in weights_i:
                    scales_rec = jnp.array(
                        [
                            es_params_pop["k_rec_ee"][i],
                            es_params_pop["k_rec_ie"][i],
                            es_params_pop["k_rec_ei"][i],
                            es_params_pop["k_rec_ii"][i],
                        ],
                        dtype=jnp.float32,
                    )
                    weights_i_scaled["recurrent"] = [
                        weights_i["recurrent"][j] * scales_rec[recurrent_scale_indices[j]]
                        for j in range(len(weights_i["recurrent"]))
                    ]
                weights_i = FrozenDict(weights_i_scaled)

            logits_i, sp_rate_i = _run_model(weights_i)
            logits = logits.at[i, :, :].set(logits_i)
            sp_rates = sp_rates.at[i].set(sp_rate_i)
            gc.collect()

        return logits, sp_rates

    def evaluate(params, weights, es_params):
        params_eval = (
            jax.tree_util.tree_map(lambda p: jax.nn.sigmoid(p), params)
            if conf.sigmoid
            else params
        )
        mask_weights = rsrp.mask_weights_with_theta(params_eval, weights, conf.mode)
        mask_weights = apply_es_scales_to_weights(mask_weights, es_params)
        correct = 0
        total = 0
        sp_rates = []
        for batch in test_batch:
            img1_test = jnp.asarray(batch[0])
            img2_test = jnp.asarray(batch[1])
            labels_test = jnp.asarray(batch[2])
            predict, sp_rate = run_model_eval(
                img1_test,
                img2_test,
                mask_weights,
                inputs_are_projected=False,
            )
            correct = correct + jnp.sum(predict == labels_test)
            total += int(labels_test.size)
            sp_rates.append(sp_rate)
            del img1_test, img2_test, labels_test, predict, sp_rate
            gc.collect()
        fitness = jnp.asarray(correct, dtype=jnp.float32) / float(total)
        return fitness, jnp.mean(jnp.asarray(sp_rates, dtype=jnp.float32))

    def evaluate_train_batch(params, weights, es_params, img1_batch, img2_batch, labels_batch, *, inputs_are_projected):
        params_eval = (
            jax.tree_util.tree_map(lambda p: jax.nn.sigmoid(p), params)
            if conf.sigmoid
            else params
        )
        mask_weights = rsrp.mask_weights_with_theta(params_eval, weights, conf.mode)
        mask_weights = apply_es_scales_to_weights(mask_weights, es_params)
        predict, _ = run_model_eval(
            img1_batch,
            img2_batch,
            mask_weights,
            inputs_are_projected=inputs_are_projected,
        )
        return jnp.mean(predict == labels_batch)

    def train(dataset_iterable, opt_state, optim_cls, weights, params, es_params, es_opt_state, es_opt_cls, key_local):
        total_steps = int(conf.n_epoch) * int(conf.steps_per_epoch)
        for epoch in range(conf.n_epoch):
            for batch in tqdm(dataset_iterable, desc=f"epoch {epoch + 1}/{conf.n_epoch}"):
                img1, img2, labels = batch[:3]
                img1 = jnp.asarray(img1)
                img2 = jnp.asarray(img2)
                labels = jnp.asarray(labels)

                next_key, key_local = jax.random.split(key_local)

                if conf.train_es:
                    n_pairs = conf.total_pop_size // 2
                    sigma = rsrp.get_es_sigma(
                        result["step"],
                        total_steps=total_steps,
                        sigma_schedule=conf.es_sigma_schedule,
                        sigma_init=conf.es_sigma_init,
                        sigma_final=conf.es_sigma_final,
                    )
                    es_noise = rsrp.sample_gaussian_noise(next_key, es_params, sigma, n_pairs)
                    es_pos = {k: es_params[k] + es_noise[k] for k in es_params}
                    es_neg = {k: es_params[k] - es_noise[k] for k in es_params}
                    es_samples = {
                        k: jnp.concatenate([es_pos[k], es_neg[k]]).reshape(conf.pop_size, n_pop)
                        for k in es_params
                    }
                else:
                    sigma = None
                    es_noise = None
                    es_samples = None

                params_sample = (
                    jax.tree_util.tree_map(lambda p: jax.nn.sigmoid(p), params)
                    if conf.sigmoid
                    else params
                )
                theta = rsrp.sample_bernoulli_parameter(next_key, params_sample, conf.total_pop_size)
                mask_weights = rsrp.mask_weights_with_theta(theta, weights, conf.mode)
                mask_weights = jax.tree_util.tree_map(
                    lambda x: x.reshape((conf.pop_size, n_pop, x.shape[-1])),
                    mask_weights,
                )

                if es_samples is None:
                    logits, sp_rate_pop = jax.vmap(
                        lambda w: run_model_train(
                            img1,
                            img2,
                            w,
                            None,
                            inputs_are_projected=bool(conf.use_projected_current_cache_train),
                        ),
                        in_axes=0,
                    )(mask_weights)
                else:
                    logits, sp_rate_pop = jax.vmap(
                        lambda w, e: run_model_train(
                            img1,
                            img2,
                            w,
                            e,
                            inputs_are_projected=bool(conf.use_projected_current_cache_train),
                        ),
                        in_axes=(0, 0),
                    )(mask_weights, es_samples)
                logits = logits.reshape((conf.total_pop_size, conf.total_batch_size, n_class))
                sp_rate_flat = sp_rate_pop.reshape((conf.total_pop_size,))

                if conf.reward_type == "accuracy":
                    fitness_cls = rsrp.accuracy
                elif conf.reward_type == "accuracy_margin":
                    fitness_cls = rsrp.AccuracyWithMargin(
                        margin_weight=conf.reward_margin_weight,
                        margin_temp=conf.reward_margin_temp,
                    )
                elif conf.reward_type == "softrecall":
                    fitness_cls = rsrp.SoftRecall(
                        class_weights=conf.class_weights,
                        wrong_logit_penalty=conf.wrong_logit_penalty,
                    )
                else:
                    raise ValueError(f"Unknown conf.reward_type={conf.reward_type!r}")

                fitness_task = fitness_cls(logits, labels)
                fitness_ranked = rsrp.centered_rank_transform(fitness_task)

                if conf.sigmoid:
                    params, opt_state = rsrp.rsrp_sigmoid(
                        fitness_ranked,
                        theta,
                        params,
                        opt_state,
                        optim_cls,
                    )
                else:
                    params, opt_state = rsrp.rsrp(
                        fitness_ranked,
                        theta,
                        params,
                        opt_state,
                        optim_cls,
                        conf.eps,
                    )

                if conf.train_es:
                    fitness_es = fitness_task - conf.es_sp_rate_penalty * (
                        sp_rate_flat - conf.es_sp_rate_target
                    ) ** 2
                    reward_for_es = rsrp.centered_rank_transform(fitness_es)
                    es_params, es_opt_state = rsrp.update_es(
                        reward_for_es,
                        es_noise,
                        es_params,
                        es_opt_state,
                        es_opt_cls,
                        sigma,
                    )

                result["step"] += 1
                gc.collect()

                if result["step"] % conf.eval_interval == 0 or result["step"] == 1:
                    result["acc"], result["sp_rate"] = evaluate(params, weights, es_params)
                    result["train_acc"] = float(
                        evaluate_train_batch(
                            params,
                            weights,
                            es_params,
                            img1,
                            img2,
                            labels,
                            inputs_are_projected=bool(conf.use_projected_current_cache_train),
                        )
                    )
                    result["readout_rate"] = float(np.mean(logits) / group_size / conf.readout_period * 1000.0)
                    result["train_sp_rate"] = float(jnp.mean(sp_rate_flat))
                    gc.collect()

                    params_log = (
                        jax.tree_util.tree_map(lambda p: jax.nn.sigmoid(p), params)
                        if conf.sigmoid
                        else params
                    )
                    result["mean_rho"] = dict(rsrp.rho_mean(params_log))
                    es_vals = {k: float(es_params[k]) for k in es_params}

                    if conf.wandb:
                        wandb.log(
                            {
                                "accuracy": result["acc"],
                                "train_accuracy": result["train_acc"],
                                "mean_rho": result["mean_rho"],
                                "sp_rate": result["sp_rate"],
                                "readout_rate": result["readout_rate"],
                                "train_sp_rate": result["train_sp_rate"],
                                **{"es/" + k: v for k, v in es_vals.items()},
                            },
                            step=result["step"],
                        )

                    print(
                        f"{result['step']}: acc={result['acc']} train_acc={result['train_acc']} "
                        f"rho_mean={result['mean_rho']}, "
                        f"sp_rate={result['sp_rate']}, readout_rate={result['readout_rate']}, "
                        f"train_ro_sp_rate={result['train_sp_rate']}"
                    )
                    np.save(
                        conf.filename,
                        {
                            "params": params,
                            "es_params": es_params,
                            "conf": OmegaConf.to_container(conf, resolve=True),
                            "result": result,
                        },
                    )

        return params, opt_state, es_params, es_opt_state, key_local

    global_model = build_model()
    weights_dict = {}
    if conf.train_input:
        weights_dict["input"] = global_model.input_layer.input_proj.weight.value
    if conf.train_recurrent:
        weights_dict["recurrent"] = [proj.proj.comm.weight.value for proj in global_model.recurrent_layer.projs]
    weights = FrozenDict(weights_dict)

    if len(weights) == 0:
        raise ValueError("At least one of train_input/train_recurrent must be True.")

    params = rsrp.init_rho(weights, init_prob=0.0 if conf.sigmoid else 0.5)

    if (conf.mean_synapse_weights or conf.mean_synapse_weights_recompute) and conf.train_recurrent:
        params_dict = dict(params)
        params_dict["recurrent"] = [p * (conf.rho_rec_init / 0.5) for p in params_dict["recurrent"]]
        params = FrozenDict(params_dict)

    if conf.reset_synapse_weights_with_rho and conf.train_recurrent:
        weights, recurrent_rho_init = reset_synapse_weights_with_rho(
            weights=weights,
            pop_name_indices=network["pop_name_indices"],
            n_receptors=synapse_params.n_receptors,
            rec_csr=_rec_csr,
            eps=1e-30 if conf.sigmoid else conf.eps,
            sigmoid_mode=conf.sigmoid,
        )
        params_dict = dict(params)
        params_dict["recurrent"] = recurrent_rho_init
        params = FrozenDict(params_dict)

    if conf.pretrained_model:
        params = FrozenDict(data_load["params"])

    init_rho_msg = {}
    if "input" in params:
        init_rho_msg["input"] = _mean_tree_entry(params["input"])
    if "recurrent" in params:
        init_rho_msg["recurrent"] = _mean_tree_entry(params["recurrent"])
    print(f"initial_rho={init_rho_msg}")

    optim_cls = optax.sgd(learning_rate=conf.lr)
    opt_state = optim_cls.init(params)

    es_params = {
        "k_in_ee": 1.0,
        "k_in_ei": 1.0,
        "k_rec_ee": 1.0,
        "k_rec_ie": 1.0,
        "k_rec_ei": 1.0,
        "k_rec_ii": 1.0,
    }
    es_opt_cls = optax.adam(learning_rate=conf.es_lr)
    es_opt_state = es_opt_cls.init(es_params)

    params, opt_state, es_params, es_opt_state, key = train(
        dataset_batch,
        opt_state,
        optim_cls,
        weights,
        params,
        es_params,
        es_opt_state,
        es_opt_cls,
        key,
    )

    return params, opt_state, es_params, es_opt_state, key


if __name__ == "__main__":
    main()
