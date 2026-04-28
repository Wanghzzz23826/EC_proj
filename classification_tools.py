import glob
import math
import os
import pickle
import tarfile
import numpy as np
import tensorflow as tf
import jax.numpy as jnp
from flax.core import FrozenDict

from brainpy_impl import models2 as models
import brainpy as bp
import brainpy.math as bm
from common.types import to_brainpy_csr


def _expand_cifar10_tar_candidates(path_value):
    if not path_value:
        return []

    path_value = os.path.expanduser(os.path.expandvars(str(path_value)))
    out = []
    try:
        if os.path.isfile(path_value):
            out.append(path_value)
        elif os.path.isdir(path_value):
            patterns = [
                os.path.join(path_value, "cifar-10-python.tar.gz"),
                os.path.join(path_value, "*cifar*10*.tar.gz"),
                os.path.join(path_value, "*.tar.gz"),
            ]
            for pattern in patterns:
                out.extend(sorted(glob.glob(pattern)))
        else:
            out.extend(sorted(glob.glob(path_value)))
    except OSError:
        return []
    return out


def _resolve_cifar10_tar_path(cifar10_path=None):
    raw_candidates = [
        cifar10_path,
        os.environ.get("CIFAR10_TAR_PATH"),
        os.environ.get("CIFAR10_PATH"),
        "/root/autodl-pub/cifar-10",
        "/root/autodl-pub/cifar10",
        "/root/autodl-pub/cifar-10-python.tar.gz",
    ]

    seen = set()
    for raw_value in raw_candidates:
        for candidate in _expand_cifar10_tar_candidates(raw_value):
            if candidate in seen:
                continue
            seen.add(candidate)
            if os.path.isfile(candidate):
                return candidate
    return None


def _load_cifar10_from_local_tar(tar_path):
    def _fetch_entry(batch_dict, byte_key, text_key):
        if byte_key in batch_dict:
            return batch_dict[byte_key]
        if text_key in batch_dict:
            return batch_dict[text_key]
        raise KeyError(f"Missing CIFAR entry {byte_key!r}/{text_key!r} in {tar_path}")

    def _load_batch(archive, member_name):
        extracted = archive.extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f"Missing member {member_name!r} in {tar_path}")
        with extracted:
            batch = pickle.load(extracted, encoding="bytes")
        data = np.asarray(_fetch_entry(batch, b"data", "data"), dtype=np.uint8)
        labels = np.asarray(
            _fetch_entry(batch, b"labels", "labels"),
            dtype=np.uint8,
        ).reshape(-1, 1)
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return data, labels

    with tarfile.open(tar_path, mode="r:gz") as archive:
        train_images = []
        train_labels = []
        for batch_idx in range(1, 6):
            images_i, labels_i = _load_batch(
                archive,
                f"cifar-10-batches-py/data_batch_{batch_idx}",
            )
            train_images.append(images_i)
            train_labels.append(labels_i)
        test_images, test_labels = _load_batch(archive, "cifar-10-batches-py/test_batch")

    x_train = np.concatenate(train_images, axis=0)
    y_train = np.concatenate(train_labels, axis=0)
    return (x_train, y_train), (test_images, test_labels)


def _load_cifar10_data(cifar10_path=None):
    local_tar = _resolve_cifar10_tar_path(cifar10_path)
    if local_tar is not None:
        print(f"[classification_tools] Loading CIFAR10 from local tar cache: {local_tar}")
        return _load_cifar10_from_local_tar(local_tar)
    return tf.keras.datasets.cifar10.load_data()

# dataset
def generate_classification_spike_data(
    data_usage=0,
    n_input_neurons=784,
    im_slice=100,
    pre_delay=50,
    post_delay=150,
    dataset="mnist",
    input_hz=200.0,
    input_mode="spike",
    balanced_batch_size=None,
):
    """
    input_mode:
        "spike"   — rate→Bernoulli spike, output shape per sample: (timesteps, n_input_neurons)
        "current" — deterministic analog current, output shape per sample: (n_input_neurons,)
    """
    # data_usage: 0, train; 1, test

    if dataset.lower() == "cifar100":
        all_ds = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    elif dataset.lower() == "cifar10":
        all_ds = _load_cifar10_data()
    elif dataset.lower() == "mnist":
        all_ds = tf.keras.datasets.mnist.load_data()
    elif dataset.lower() == "fashion_mnist":
        all_ds = tf.keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    images, labels = all_ds[data_usage]
    labels = np.asarray(labels).reshape(-1).astype(np.int32)

    if len(images.shape) > 3:
        images = tf.image.rgb_to_grayscale(images) / 255.0
    else:
        images = images[..., None] / 255.0

    if input_mode == "spike":
        def process_sample(image, label):
            col = int(math.sqrt(n_input_neurons))
            row = n_input_neurons // col
            assert col * row == n_input_neurons
            img = tf.image.resize_with_pad(image, col, row, method="lanczos5")
            tiled_img = tf.tile(img[None, ...], (im_slice, 1, 1, 1))
            z1 = tf.tile(tf.zeros_like(img)[None, ...], (pre_delay, 1, 1, 1))
            z2 = tf.tile(tf.zeros_like(img)[None, ...], (post_delay, 1, 1, 1))
            videos = tf.concat((z1, tiled_img, z2), 0)
            firing_rates = tf.reshape(videos, [-1, n_input_neurons])
            _p = 1 - tf.exp(-input_hz * firing_rates / 1000.0)
            _z = tf.random.uniform(tf.shape(_p)) < _p
            return _z, label

    elif input_mode == "current":
        def process_sample(image, label):
            col = int(math.sqrt(n_input_neurons))
            row = n_input_neurons // col
            assert col * row == n_input_neurons
            img = tf.image.resize_with_pad(image, col, row, method="lanczos5")
            firing_rates = tf.reshape(img, [n_input_neurons])
            input_current = tf.cast(firing_rates * float(input_hz) / 1000.0, tf.float32)
            return input_current, label

    else:
        raise ValueError(f"Unsupported input_mode='{input_mode}', expected 'spike' or 'current'")

    if balanced_batch_size is not None:
        unique_labels = np.unique(labels)
        n_classes = int(unique_labels.size)
        if n_classes <= 0:
            raise ValueError("No labels found when building balanced batches.")
        if balanced_batch_size % n_classes != 0:
            raise ValueError(
                "balanced_batch_size must be divisible by number of classes. "
                f"Got balanced_batch_size={balanced_batch_size}, n_classes={n_classes}."
            )

        per_class_batch = balanced_batch_size // n_classes
        class_batch_datasets = []
        shuffle_buffer = max(256, per_class_batch * 32)
        images_np = np.asarray(images)

        for class_id in unique_labels.tolist():
            class_mask = labels == class_id
            class_images = images_np[class_mask]
            class_labels = labels[class_mask]
            class_ds = tf.data.Dataset.from_tensor_slices((class_images, class_labels))
            class_ds = class_ds.shuffle(
                shuffle_buffer, reshuffle_each_iteration=True
            )
            class_ds = class_ds.batch(per_class_batch, drop_remainder=True)
            class_batch_datasets.append(class_ds)

        def merge_class_batches(*class_batches):
            spikes = tf.concat([b[0] for b in class_batches], axis=0)
            labels_batch = tf.concat([b[1] for b in class_batches], axis=0)
            perm = tf.random.shuffle(tf.range(tf.shape(labels_batch)[0]))
            return tf.gather(spikes, perm), tf.gather(labels_batch, perm)

        data_set = tf.data.Dataset.zip(tuple(class_batch_datasets))
        data_set = data_set.map(
            merge_class_batches, num_parallel_calls=tf.data.AUTOTUNE
        )
        data_set = data_set.unbatch()
        data_set = data_set.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
        data_set = data_set.batch(balanced_batch_size, drop_remainder=True)
        data_set = data_set.prefetch(tf.data.AUTOTUNE)
    else:
        data_set = tf.data.Dataset.from_tensor_slices((images, labels))
        data_set = data_set.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)

    return data_set


def generate_fine_orientation_discrimination_data(
    n_input_neurons=784,
    im_slice=150,
    pre_delay=0,
    post_delay=0,
    input_hz=200.0,
    input_mode="spike",
    orientation_center=45.0,
    orientation_delta=2.0,
    orientation_step=0.1,
    spatial_freq=0.1,
    contrast=1.0,
    return_orientation=False,
):
    """
    持续在线生成“接近 45 度的小角度差异”刺激序列，用于二分类（orientation > center）。

    input_mode:
        "spike"   -> Bernoulli spike sequence, output per sample: (timesteps, n_input_neurons)
        "current" -> deterministic analog current, output per sample: (n_input_neurons,)
    return_orientation:
        False -> return (x, label)
        True  -> return (x, label, orientation)
    """
    col = int(math.sqrt(n_input_neurons))
    row = n_input_neurons // col
    assert col * row == n_input_neurons
    if orientation_step <= 0:
        raise ValueError(f"orientation_step must be > 0, got {orientation_step}")
    if orientation_delta <= 0:
        raise ValueError(f"orientation_delta must be > 0, got {orientation_delta}")
    if contrast <= 0 or contrast > 1:
        raise ValueError(f"contrast must be in (0, 1], got {contrast}")

    timesteps = int(pre_delay + im_slice + post_delay)
    yy, xx = tf.meshgrid(
        tf.cast(tf.range(row), tf.float32),
        tf.cast(tf.range(col), tf.float32),
        indexing="ij",
    )

    center = tf.constant(float(orientation_center), dtype=tf.float32)
    delta = tf.constant(float(orientation_delta), dtype=tf.float32)
    step = tf.constant(float(orientation_step), dtype=tf.float32)
    spatial_freq_tf = tf.constant(float(spatial_freq), dtype=tf.float32)
    contrast_tf = tf.constant(float(contrast), dtype=tf.float32)
    input_hz_tf = tf.constant(float(input_hz), dtype=tf.float32)

    def _sample_orientation():
        # Sample discrete offsets and explicitly exclude 0 offset (center angle),
        # so orientation never equals `center`.
        n_side = tf.cast(tf.round(delta / step), tf.int32)
        idx = tf.random.uniform([], minval=0, maxval=2 * n_side, dtype=tf.int32)
        signed_steps = tf.where(
            idx < n_side,
            -(n_side - idx),       # -n_side ... -1
            idx - n_side + 1,      # 1 ... n_side
        )
        orientation = center + tf.cast(signed_steps, tf.float32) * step
        return orientation

    def _make_grating_frame(orientation):
        theta = np.pi * (180.0 - orientation) / 180.0
        phase = tf.random.uniform([], minval=0.0, maxval=np.pi * 2.0, dtype=tf.float32)
        xy = xx * tf.cos(theta) + yy * tf.sin(theta)
        frame = 0.5 + 0.5 * contrast_tf * tf.sin(2.0 * np.pi * spatial_freq_tf * xy + phase)
        frame = tf.clip_by_value(frame, 0.0, 1.0)
        return frame

    def _make_grating_video(orientation):
        frame = _make_grating_frame(orientation)[..., None]

        stim = tf.tile(frame[None, ...], (im_slice, 1, 1, 1))
        z1 = tf.zeros((pre_delay, row, col, 1), dtype=tf.float32)
        z2 = tf.zeros((post_delay, row, col, 1), dtype=tf.float32)
        videos = tf.concat((z1, stim, z2), axis=0)
        firing_rates = tf.reshape(videos, [timesteps, n_input_neurons])
        return firing_rates

    def process_sample(_):
        orientation = _sample_orientation()
        label = tf.cast(orientation > center, tf.int32)
        if input_mode == "spike":
            firing_rates = _make_grating_video(orientation)
            p = 1.0 - tf.exp(-input_hz_tf * firing_rates / 1000.0)
            x = tf.cast(tf.random.uniform(tf.shape(p)) < p, tf.float32)
        elif input_mode == "current":
            frame = _make_grating_frame(orientation)
            firing_rates = tf.reshape(frame, [n_input_neurons])
            x = tf.cast(firing_rates * input_hz_tf / 1000.0, tf.float32)
        else:
            raise ValueError(f"Unsupported input_mode='{input_mode}', expected 'spike' or 'current'")
        if return_orientation:
            return x, label, tf.cast(orientation, tf.float32)
        return x, label

    # Infinite stream for continuous generation.
    seeds_ds = tf.data.Dataset.from_tensors(tf.constant(0, tf.int32)).repeat()
    data_set = seeds_ds.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    data_set = data_set.prefetch(tf.data.AUTOTUNE)

    return data_set


def generate_fine_orientation_pair_match_data(
    n_input_neurons=784,
    im_slice=100,
    input_hz=200.0,
    input_mode="spike",
    orientation_center=45.0,
    orientation_delta=15.0,
    orientation_step=0.1,
    spatial_freq=0.1,
    contrast=1.0,
    return_orientation=False,
):
    """
    持续在线生成两张取向条纹图，用于二分类（两张图是否相同，P(same)=0.5）。

    input_mode:
        "spike"   -> Bernoulli spike sequence, output per sample:
                     (x1, x2, label), x1/x2 shape = (im_slice, n_input_neurons)
        "current" -> deterministic analog current, output per sample:
                     (x1, x2, label), x1/x2 shape = (n_input_neurons,)
    return_orientation:
        False -> return (x1, x2, label)
        True  -> return (x1, x2, label, orientation1, orientation2)
    """
    col = int(math.sqrt(n_input_neurons))
    row = n_input_neurons // col
    assert col * row == n_input_neurons
    if orientation_step <= 0:
        raise ValueError(f"orientation_step must be > 0, got {orientation_step}")
    if orientation_delta <= 0:
        raise ValueError(f"orientation_delta must be > 0, got {orientation_delta}")
    if contrast <= 0 or contrast > 1:
        raise ValueError(f"contrast must be in (0, 1], got {contrast}")
    if im_slice <= 0:
        raise ValueError(f"im_slice must be > 0, got {im_slice}")

    yy, xx = tf.meshgrid(
        tf.cast(tf.range(row), tf.float32),
        tf.cast(tf.range(col), tf.float32),
        indexing="ij",
    )

    center = tf.constant(float(orientation_center), dtype=tf.float32)
    delta = tf.constant(float(orientation_delta), dtype=tf.float32)
    step = tf.constant(float(orientation_step), dtype=tf.float32)
    spatial_freq_tf = tf.constant(float(spatial_freq), dtype=tf.float32)
    contrast_tf = tf.constant(float(contrast), dtype=tf.float32)
    input_hz_tf = tf.constant(float(input_hz), dtype=tf.float32)

    def _sample_orientation():
        # Sample discrete offsets and explicitly exclude 0 offset (center angle),
        # so orientation never equals `center`.
        n_side = tf.maximum(1, tf.cast(tf.round(delta / step), tf.int32))
        idx = tf.random.uniform([], minval=0, maxval=2 * n_side, dtype=tf.int32)
        signed_steps = tf.where(
            idx < n_side,
            -(n_side - idx),  # -n_side ... -1
            idx - n_side + 1,  # 1 ... n_side
        )
        orientation = center + tf.cast(signed_steps, tf.float32) * step
        return orientation

    def _sample_orientation_not_equal(ref_orientation):
        # Build a different sample by index remapping in the discrete orientation set.
        # This avoids tf.while_loop issues inside tf.data graph tracing.
        n_side = tf.maximum(1, tf.cast(tf.round(delta / step), tf.int32))
        total = 2 * n_side
        ref_signed = tf.cast(tf.round((ref_orientation - center) / step), tf.int32)
        ref_idx = tf.where(
            ref_signed < 0,
            ref_signed + n_side,      # -n_side ... -1 -> 0 ... n_side-1
            ref_signed + n_side - 1,  # 1 ... n_side -> n_side ... 2*n_side-1
        )
        shift = tf.random.uniform([], minval=1, maxval=total, dtype=tf.int32)
        idx = tf.math.mod(ref_idx + shift, total)
        signed_steps = tf.where(
            idx < n_side,
            -(n_side - idx),  # -n_side ... -1
            idx - n_side + 1,  # 1 ... n_side
        )
        orientation = center + tf.cast(signed_steps, tf.float32) * step
        return orientation

    def _make_grating_frame(orientation):
        theta = np.pi * (180.0 - orientation) / 180.0
        phase = tf.random.uniform([], minval=0.0, maxval=np.pi * 2.0, dtype=tf.float32)
        xy = xx * tf.cos(theta) + yy * tf.sin(theta)
        frame = 0.5 + 0.5 * contrast_tf * tf.sin(2.0 * np.pi * spatial_freq_tf * xy + phase)
        frame = tf.clip_by_value(frame, 0.0, 1.0)
        return frame

    def _to_input(orientation):
        frame = _make_grating_frame(orientation)
        if input_mode == "spike":
            stim = tf.tile(frame[None, ..., None], (im_slice, 1, 1, 1))
            firing_rates = tf.reshape(stim, [im_slice, n_input_neurons])
            p = 1.0 - tf.exp(-input_hz_tf * firing_rates / 1000.0)
            x = tf.cast(tf.random.uniform(tf.shape(p)) < p, tf.float32)
            return x
        if input_mode == "current":
            firing_rates = tf.reshape(frame, [n_input_neurons])
            x = tf.cast(firing_rates * input_hz_tf / 1000.0, tf.float32)
            return x
        raise ValueError(
            f"Unsupported input_mode='{input_mode}', expected 'spike' or 'current'"
        )

    def process_sample(_):
        orientation1 = _sample_orientation()
        same_flag = tf.random.uniform([], minval=0.0, maxval=1.0) < 0.5
        orientation2 = tf.cond(
            same_flag,
            lambda: orientation1,
            lambda: _sample_orientation_not_equal(orientation1),
        )
        label = tf.cast(same_flag, tf.int32)
        x1 = _to_input(orientation1)
        x2 = _to_input(orientation2)

        if return_orientation:
            return (
                x1,
                x2,
                label,
                tf.cast(orientation1, tf.float32),
                tf.cast(orientation2, tf.float32),
            )
        return x1, x2, label

    # Infinite stream for continuous generation.
    seeds_ds = tf.data.Dataset.from_tensors(tf.constant(0, tf.int32)).repeat()
    data_set = seeds_ds.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    data_set = data_set.prefetch(tf.data.AUTOTUNE)

    return data_set


def generate_cifar10_pair_match_data(
    data_usage=0,
    n_input_neurons=784,
    im_slice=100,
    input_hz=200.0,
    input_mode="spike",
    n_reference_images=40,
    seed=0,
    return_image_ids=False,
    align_image_mean=False,
    cifar10_path=None,
):
    """
    持续在线生成两张 CIFAR10 灰度图，用于二分类（两张图是否相同，P(same)=0.5）。

    data_usage:
        0 -> 从 CIFAR10 train split 采样参考图
        1 -> 从 CIFAR10 test split 采样参考图
    n_reference_images:
        参考图池大小，默认 40；train/test 各自独立采样。
    input_mode:
        "spike"   -> Bernoulli spike sequence, output per sample:
                     (x1, x2, label), x1/x2 shape = (im_slice, n_input_neurons)
        "current" -> deterministic analog current, output per sample:
                     (x1, x2, label), x1/x2 shape = (n_input_neurons,)
    return_image_ids:
        False -> return (x1, x2, label)
        True  -> return (x1, x2, label, image_id1, image_id2)
    align_image_mean:
        True  -> 对 resize 后每张图做严格均值对齐（不 clip）
        False -> 不做均值对齐，保持原始像素统计
    """
    if data_usage not in (0, 1):
        raise ValueError(f"data_usage must be 0 (train) or 1 (test), got {data_usage}")
    if n_reference_images < 2:
        raise ValueError(
            f"n_reference_images must be >= 2 for same/different matching, got {n_reference_images}"
        )
    if im_slice <= 0:
        raise ValueError(f"im_slice must be > 0, got {im_slice}")

    col = int(math.sqrt(n_input_neurons))
    row = n_input_neurons // col
    assert col * row == n_input_neurons

    (x_train, _), (x_test, _) = _load_cifar10_data(cifar10_path=cifar10_path)
    source_images = x_train if data_usage == 0 else x_test
    total_images = int(source_images.shape[0])
    if n_reference_images > total_images:
        raise ValueError(
            f"n_reference_images={n_reference_images} exceeds split size {total_images}"
        )

    rng = np.random.RandomState(seed=seed + int(data_usage))
    selected_idx = rng.choice(total_images, size=n_reference_images, replace=False)

    # Use NumPy grayscale conversion on CPU to avoid GPU BLAS initialization issues.
    selected_rgb = source_images[selected_idx].astype(np.float32) / 255.0  # [K, H, W, 3]
    gray_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    selected_gray = np.tensordot(selected_rgb, gray_weights, axes=([-1], [0]))[..., None]

    with tf.device("/CPU:0"):
        selected_images = tf.convert_to_tensor(selected_gray, dtype=tf.float32)
        # Resize once to reduce per-sample overhead in tf.data map.
        resized_images = tf.map_fn(
            lambda _img: tf.image.resize_with_pad(_img, row, col, method="lanczos5"),
            selected_images,
            fn_output_signature=tf.float32,
        )
        if align_image_mean:
            # Strict per-image mean alignment (no clipping):
            # all images are shifted to share the same scalar mean after resize.
            target_mean = tf.reduce_mean(resized_images)
            per_image_mean = tf.reduce_mean(
                resized_images, axis=(1, 2, 3), keepdims=True
            )
            resized_images = resized_images + (target_mean - per_image_mean)
    n_pool = int(n_reference_images)
    input_hz_tf = tf.constant(float(input_hz), dtype=tf.float32)

    def _to_input(frame):
        if input_mode == "spike":
            stim = tf.tile(frame[None, ...], (im_slice, 1, 1, 1))
            firing_rates = tf.reshape(stim, [im_slice, n_input_neurons])
            p = 1.0 - tf.exp(-input_hz_tf * firing_rates / 1000.0)
            return tf.cast(tf.random.uniform(tf.shape(p)) < p, tf.float32)
        if input_mode == "current":
            firing_rates = tf.reshape(frame, [n_input_neurons])
            return tf.cast(firing_rates * input_hz_tf / 1000.0, tf.float32)
        raise ValueError(
            f"Unsupported input_mode='{input_mode}', expected 'spike' or 'current'"
        )

    def process_sample(_):
        image_id1 = tf.random.uniform([], minval=0, maxval=n_pool, dtype=tf.int32)
        same_flag = tf.random.uniform([], minval=0.0, maxval=1.0) < 0.5

        # Draw a different index by cyclic shift to avoid while_loop in graph.
        shift = tf.random.uniform([], minval=1, maxval=n_pool, dtype=tf.int32)
        diff_id = tf.math.mod(image_id1 + shift, n_pool)
        image_id2 = tf.where(same_flag, image_id1, diff_id)

        frame1 = tf.gather(resized_images, image_id1)
        frame2 = tf.gather(resized_images, image_id2)
        x1 = _to_input(frame1)
        x2 = _to_input(frame2)
        label = tf.cast(same_flag, tf.int32)

        if return_image_ids:
            return x1, x2, label, image_id1, image_id2
        return x1, x2, label

    seeds_ds = tf.data.Dataset.from_tensors(tf.constant(0, tf.int32)).repeat()
    data_set = seeds_ds.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    data_set = data_set.prefetch(tf.data.AUTOTUNE)
    return data_set


from dataclasses import replace

def rearrange_input(
    input_params,
    n_input_neurons,
    seed,
    patch_size=7,
    mode="grid",
    stride=None,
):
    """
    按空间局部感受野重排输入连接，使得同一神经元主要接收来自输入图像同一块区域的连接。

    - 仅修改 indices[:, 1]（pre 节点，即输入像素索引），不改变边数、post 节点或权重。
    - 每个 post 神经元会被分配一个 patch，其所有输入边的 pre 都从该 patch 覆盖的像素中采样。

    Parameters
    ----------
    mode : str
        - "grid": 将图像均匀划分为不重叠（或按 stride 重叠）的 patch 网格，神经元按顺序轮询
          分配到各 patch，使空间覆盖和神经元分配都更均匀。
        - "central_random": 每个神经元随机一个 patch 中心（可能重叠、分布不均）。
        - "global_random": 生成均匀随机的 pre_node（0 ~ conf.n_input_neurons-1）
    stride : int or None
        仅当 mode=="grid" 时有效。patch 的步长；若为 None 则取 patch_size（不重叠）。
        若 stride < patch_size 则 patch 之间有重叠。
    """
    indices_np = np.asarray(input_params.indices, dtype=np.int64)
    rng = np.random.RandomState(seed=seed)
    mode_norm = str(mode).strip().lower()

    if mode_norm not in ("global_random", "grid", "central_random"):
        raise ValueError(
            f"Unsupported rearrange_input mode='{mode}'. "
            "Expected one of: 'global_random', 'grid', 'central_random'."
        )

    if mode_norm == "global_random":
        # 生成均匀随机的 pre_node（0 ~ conf.n_input_neurons-1）
        n_edges = indices_np.shape[0]
        new_pre = rng.randint(0, n_input_neurons, size=n_edges, dtype=np.int64)
        indices_np[:, 1] = new_pre
        input_params_new = replace(
            input_params,
            indices=indices_np,
            n_input_nodes=n_input_neurons,
            dense_shape=(input_params.dense_shape[0], n_input_neurons),
        )
        print("input_params.indices[:, 1]已随机对应输入图像。")
        return input_params_new

    rf_cols = max(1, int(np.sqrt(n_input_neurons)))
    rf_rows = (n_input_neurons + rf_cols - 1) // rf_cols

    n_receptors = int(input_params.n_receptors)
    post_flat = indices_np[:, 0].astype(np.int64)
    post_neuron = post_flat // n_receptors
    
    unique_neurons = np.unique(post_neuron)

    if mode_norm == "grid":
        stride = stride if stride is not None else patch_size
        stride = max(1, min(stride, patch_size))
        # 网格起点：使 patch 尽量铺满 [0, rf_rows) x [0, rf_cols)
        r_starts = np.arange(0, max(1, rf_rows - patch_size + 1), stride)
        c_starts = np.arange(0, max(1, rf_cols - patch_size + 1), stride)
        if r_starts.size == 0:
            r_starts = np.array([max(0, (rf_rows - patch_size) // 2)])
        if c_starts.size == 0:
            c_starts = np.array([max(0, (rf_cols - patch_size) // 2)])
        patches_list = []
        for r0 in r_starts:
            for c0 in c_starts:
                r_end = min(r0 + patch_size, rf_rows)
                c_end = min(c0 + patch_size, rf_cols)
                rr = np.arange(r0, r_end)
                cc = np.arange(c0, c_end)
                grid_r, grid_c = np.meshgrid(rr, cc, indexing="ij")
                idx = (grid_r * rf_cols + grid_c).ravel()
                idx = idx[idx < n_input_neurons]
                if idx.size > 0:
                    patches_list.append(idx.astype(np.int64))
        if not patches_list:
            patches_list = [np.arange(n_input_neurons, dtype=np.int64)]
        n_patches = len(patches_list)

    half = patch_size // 2
    for rank, nid in enumerate(unique_neurons):
        mask = post_neuron == nid
        if not np.any(mask):
            continue

        if mode_norm == "grid" and patches_list:
            patch_id = rank % n_patches
            patch_indices = patches_list[patch_id]
        elif mode_norm == "central_random":
            center_r = rng.randint(0+half, rf_rows-half)
            center_c = rng.randint(0+half, rf_cols-half)
            r_start = max(0, center_r - half)
            r_end = min(rf_rows, r_start + patch_size)
            c_start = max(0, center_c - half)
            c_end = min(rf_cols, c_start + patch_size)
            rr = np.arange(r_start, r_end)
            cc = np.arange(c_start, c_end)
            grid_r, grid_c = np.meshgrid(rr, cc, indexing="ij")
            patch_indices = (grid_r * rf_cols + grid_c).ravel()
            patch_indices = np.asarray(
                patch_indices[patch_indices < n_input_neurons], dtype=np.int64
            )
            if patch_indices.size == 0:
                patch_indices = np.arange(n_input_neurons, dtype=np.int64)
        else:
            # Defensive fallback; normal flow is guarded by the upfront mode check.
            raise RuntimeError(f"Unexpected rearrange_input mode branch: {mode_norm}")

        k = int(mask.sum())
        n_patch = len(patch_indices)
        if k <= n_patch:
            # 无放回采样，patch 内每个像素至多被选一次，分布均匀
            new_pre = rng.choice(patch_indices, size=k, replace=False)
        else:
            # 边数多于 patch 像素数：平铺 patch 像素再取前 k 个并打乱，使每个像素被选次数尽量接近
            repeats = (k + n_patch - 1) // n_patch
            tiled = np.tile(patch_indices, repeats)[:k]
            rng.shuffle(tiled)
            new_pre = tiled
        indices_np[mask, 1] = new_pre.astype(np.int64)

    input_params_new = replace(
        input_params,
        indices=indices_np,
        n_input_nodes=n_input_neurons,
        dense_shape=(input_params.dense_shape[0], n_input_neurons),
    )
    print(
        f"input_params.indices[:, 1] 已按空间 patch 重排（mode={mode}, patch_size={patch_size}），"
    )
    return input_params_new

def reset_synapse_weights_by_mean_w(synapse_params, network, rho_init=0.5, pop_names=None):
    """
    按每个 (pre_pop, post_pop) 的 mean w 重置 synapse_params.weights。
    network 需包含 'pop_name_indices'。若 pop_names 为 None，则从 network 推断。
    返回新的 SynapseParams（原对象不变）。
    """
    pop_name_indices = network["pop_name_indices"]
    if pop_names is None:
        pop_names = [k for k in pop_name_indices.keys() if len(pop_name_indices[k]) > 0]

    idx_syn = np.asarray(synapse_params.indices)
    n_rec = synapse_params.n_receptors
    post_neuron_id = np.asarray(idx_syn[:, 0] // n_rec, dtype=np.int64)
    pre_neuron_id = np.asarray(idx_syn[:, 1], dtype=np.int64)
    w_rec = np.asarray(synapse_params.weights).flatten()

    # 1) 神经元 id -> 群体名
    neuron_to_pop = {}
    for pop in pop_names:
        for nid in np.asarray(pop_name_indices[pop]).ravel():
            neuron_to_pop[int(nid)] = pop

    # 2) 每个 (pre_pop, post_pop) 的 mean w
    mean_w_map = {}
    for pre_pop in pop_names:
        pre_ids = np.asarray(pop_name_indices[pre_pop]).ravel()
        if pre_ids.size == 0:
            continue
        for post_pop in pop_names:
            post_ids = np.asarray(pop_name_indices[post_pop]).ravel()
            if post_ids.size == 0:
                continue
            mask = np.isin(pre_neuron_id, pre_ids) & np.isin(post_neuron_id, post_ids)
            if mask.sum() == 0:
                continue
            mean_w_map[(pre_pop, post_pop)] = float(np.mean(w_rec[mask]))

    # 3) 为每条突触赋值为其 (pre_pop, post_pop) 的 mean w
    new_weights = np.empty_like(w_rec, dtype=np.float32)
    for i in range(len(pre_neuron_id)):
        pre_pop = neuron_to_pop.get(int(pre_neuron_id[i]))
        post_pop = neuron_to_pop.get(int(post_neuron_id[i]))
        if pre_pop is None or post_pop is None:
            new_weights[i] = w_rec[i]
        else:
            new_weights[i] = mean_w_map[(pre_pop, post_pop)]

    w_shape = np.asarray(synapse_params.weights).shape
    print(f"synapse_params.weights 已按各 (pre_pop, post_pop) 的 mean w 重置。weights乘以{0.5/rho_init}")
    return replace(synapse_params, weights=np.asarray(new_weights.reshape(w_shape))*(0.5/rho_init))


DEFAULT_POP_ORDER = [
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

DEFAULT_VTARGET_SPARSE = {
    "i1Htr3a": {"i1Htr3a": 1.73, "e23": 0.53, "i23Pvalb": 0.48, "i23Sst": 0.57, "i23Htr3a": 0.78, "e4": 0.42, "e5": 0.42, "i5Pvalb": 0.0, "i5Sst": 0.0, "e6": 0.42},
    "e23": {"i1Htr3a": 0.0, "e23": 0.36, "i23Pvalb": 1.49, "i23Sst": 0.86, "i23Htr3a": 1.31, "e4": 0.34, "i4Pvalb": 1.39, "i4Sst": 0.69, "i4Htr3a": 0.91, "e5": 0.74, "i5Pvalb": 1.32, "i5Sst": 0.53, "e6": 0.0},
    "i23Pvalb": {"i1Htr3a": 0.37, "e23": 0.48, "i23Pvalb": 0.68, "i23Sst": 0.42, "i23Htr3a": 0.41, "e4": 0.56, "i4Pvalb": 0.68, "i4Sst": 0.42, "i4Htr3a": 0.41, "e5": 0.20, "i5Pvalb": 0.79},
    "i23Sst": {"i1Htr3a": 0.47, "e23": 0.31, "i23Pvalb": 0.50, "i23Sst": 0.15, "i23Htr3a": 0.52, "e4": 0.30, "i4Pvalb": 0.50, "i4Sst": 0.15, "i4Htr3a": 0.52, "e5": 0.22, "i5Sst": 0.0},
    "i23Htr3a": {"i1Htr3a": 0.0, "e23": 0.28, "i23Pvalb": 0.18, "i23Sst": 0.32, "i23Htr3a": 0.37, "e4": 0.29, "i4Pvalb": 0.18, "i4Sst": 0.32, "i4Htr3a": 0.37, "e5": 0.0},
    "e4": {"e23": 0.78, "i23Pvalb": 1.39, "i23Sst": 0.69, "i23Htr3a": 0.91, "e4": 0.83, "i4Pvalb": 1.29, "i4Sst": 0.51, "i4Htr3a": 0.51, "e5": 0.63, "i5Pvalb": 1.25, "i5Sst": 0.52, "i5Htr3a": 0.91, "e6": 0.96},
    "i4Pvalb": {"e23": 0.56, "i23Pvalb": 0.68, "i23Sst": 0.42, "i23Htr3a": 0.41, "e4": 0.64, "i4Pvalb": 0.68, "i4Sst": 0.42, "i4Htr3a": 0.41, "e5": 0.73, "i5Pvalb": 0.94, "i5Sst": 0.42, "i5Htr3a": 0.41},
    "i4Sst": {"i1Htr3a": 0.39, "e23": 0.30, "i23Pvalb": 0.50, "i23Sst": 0.15, "i23Htr3a": 0.52, "e4": 0.29, "i4Pvalb": 0.50, "i4Sst": 0.15, "i4Htr3a": 0.52, "e5": 0.28, "i5Pvalb": 0.45, "i5Sst": 0.28, "i5Htr3a": 0.52},
    "i4Htr3a": {"e23": 0.29, "i23Pvalb": 0.18, "i23Sst": 0.32, "i23Htr3a": 0.37, "e4": 0.29, "i4Pvalb": 0.18, "i4Sst": 0.32, "i4Htr3a": 0.37, "e5": 0.0, "i5Pvalb": 0.18, "i5Sst": 0.33, "i5Htr3a": 0.37},
    "e5": {"i1Htr3a": 0.76, "e23": 0.47, "i23Pvalb": 1.25, "i23Sst": 0.52, "i23Htr3a": 0.91, "e4": 0.38, "i4Pvalb": 1.25, "i4Sst": 0.52, "i4Htr3a": 0.91, "e5": 0.75, "i5Pvalb": 1.20, "i5Sst": 0.52, "i5Htr3a": 1.31, "e6": 0.40, "i6Pvalb": 2.50, "i6Sst": 0.52, "i6Htr3a": 1.31},
    "i5Pvalb": {"i1Htr3a": 0.0, "e23": 0.0, "i23Pvalb": 0.51, "e4": 0.0, "i4Pvalb": 0.94, "i4Sst": 0.42, "i4Htr3a": 0.41, "e5": 0.81, "i5Pvalb": 1.19, "i5Sst": 0.41, "i5Htr3a": 0.41, "e6": 0.81, "i6Pvalb": 1.19, "i6Sst": 0.41, "i6Htr3a": 0.41},
    "i5Sst": {"i1Htr3a": 0.31, "e23": 0.25, "i23Sst": 0.39, "e4": 0.28, "i4Pvalb": 0.45, "i4Sst": 0.28, "i4Htr3a": 0.52, "e5": 0.27, "i5Pvalb": 0.40, "i5Sst": 0.40, "i5Htr3a": 0.52, "e6": 0.27, "i6Pvalb": 0.40, "i6Sst": 0.40, "i6Htr3a": 0.52},
    "i5Htr3a": {"e4": 0.29, "i4Pvalb": 0.18, "i4Sst": 0.33, "i4Htr3a": 0.37, "e5": 0.28, "i5Pvalb": 0.18, "i5Sst": 0.33, "i5Htr3a": 0.37, "e6": 0.28, "i6Pvalb": 0.18, "i6Sst": 0.33, "i6Htr3a": 0.37},
    "e6": {"e23": 0.0, "e4": 0.0, "e5": 0.23, "i5Pvalb": 2.50, "i5Sst": 0.52, "i5Htr3a": 1.31, "e6": 0.94, "i6Pvalb": 3.80, "i6Sst": 0.52, "i6Htr3a": 1.31},
    "i6Pvalb": {"e23": 0.81, "e4": 0.81, "e5": 0.81, "i5Pvalb": 1.19, "i5Sst": 0.41, "i5Htr3a": 0.41, "e6": 0.81, "i6Pvalb": 1.19, "i6Sst": 0.41, "i6Htr3a": 0.41},
    "i6Sst": {"e5": 0.27, "i5Pvalb": 0.40, "i5Sst": 0.40, "i5Htr3a": 0.52, "e6": 0.27, "i6Pvalb": 0.40, "i6Sst": 0.40, "i6Htr3a": 0.52},
    "i6Htr3a": {"e5": 0.28, "i5Pvalb": 0.18, "i5Sst": 0.33, "i5Htr3a": 0.37, "e6": 0.28, "i6Pvalb": 0.18, "i6Sst": 0.33, "i6Htr3a": 0.37},
}


def recompute_synapse_weights_from_vtarget(
    synapse_params,
    node_params,
    network,
    rho_init=0.5,
    tau_syn_by_receptor=(5.5, 8.5, 2.8, 5.8),
    vtarget_sparse=None,
    pop_order=None,
    eps=1e-8,
):
    """
    按 Vtarget(PSP mV) 与 GLIF 公式重算 recurrent synapse 权重。

    公式:
      tau_m != tau_syn:
        w = Vtarget * C * (1/tau_m - 1/tau_syn) * exp((1/tau_m - 1/tau_syn) * tau_syn)
      tau_m == tau_syn:
        w = Vtarget * C * e / (2 * tau_m)

    参数说明:
      - pre/post 神经元种类由 network["pop_name_indices"] 提供
      - tau_m, C 使用 post 神经元的 node_params.tau / node_params.c_m
      - tau_syn 由 receptor 索引映射（默认 [5.5, 8.5, 2.8, 5.8]）

    说明:
      - 若某个 pre/post 组合在 vtarget_sparse 中缺失，则该突触保持原始权重。
      - 默认保持原始符号（避免改变 E/I 方向），仅更新绝对值幅度。
    """
    if vtarget_sparse is None:
        vtarget_sparse = DEFAULT_VTARGET_SPARSE
    if pop_order is None:
        pop_order = DEFAULT_POP_ORDER

    idx = np.asarray(synapse_params.indices, dtype=np.int64)
    n_rec = int(synapse_params.n_receptors)
    post_receptor_flat = idx[:, 0]
    pre_neuron_id = np.asarray(idx[:, 1], dtype=np.int64)
    receptor_id = np.asarray(post_receptor_flat % n_rec, dtype=np.int64)
    post_neuron_id = np.asarray(post_receptor_flat // n_rec, dtype=np.int64)

    tau_m_all = np.asarray(node_params.tau, dtype=np.float32)
    c_m_all = np.asarray(node_params.c_m, dtype=np.float32)
    tau_m = tau_m_all[post_neuron_id]
    c_m = c_m_all[post_neuron_id]
    tau_syn = np.asarray(tau_syn_by_receptor, dtype=np.float32)[receptor_id]

    n_nodes = int(network["n_nodes"])
    neuron_to_popid = np.full(n_nodes, -1, dtype=np.int32)
    pop_to_id = {p: i for i, p in enumerate(pop_order)}
    for pop_name, neuron_ids in network["pop_name_indices"].items():
        if pop_name not in pop_to_id:
            continue
        ids = np.asarray(neuron_ids, dtype=np.int64).ravel()
        neuron_to_popid[ids] = pop_to_id[pop_name]

    pre_popid = neuron_to_popid[pre_neuron_id]
    post_popid = neuron_to_popid[post_neuron_id]

    n_pop = len(pop_order)
    vtarget_mat = np.full((n_pop, n_pop), np.nan, dtype=np.float32)
    for pre_name, row in vtarget_sparse.items():
        if pre_name not in pop_to_id:
            continue
        i = pop_to_id[pre_name]
        for post_name, v in row.items():
            if post_name in pop_to_id:
                j = pop_to_id[post_name]
                vtarget_mat[i, j] = np.float32(v)

    valid_pop = (pre_popid >= 0) & (post_popid >= 0)
    vtarget = np.full(idx.shape[0], np.nan, dtype=np.float32)
    valid_idx = np.where(valid_pop)[0]
    vtarget[valid_idx] = vtarget_mat[pre_popid[valid_idx], post_popid[valid_idx]]

    delta = (1.0 / tau_m) - (1.0 / tau_syn)
    w_new_abs = np.empty_like(delta, dtype=np.float32)
    neq = np.abs(delta) > eps
    w_new_abs[neq] = (
        vtarget[neq]
        * c_m[neq]
        * delta[neq]
        * np.exp(delta[neq] * tau_syn[neq])
    )
    w_new_abs[~neq] = vtarget[~neq] * c_m[~neq] * (np.e / (2.0 * tau_m[~neq]))

    w_old = np.asarray(synapse_params.weights, dtype=np.float32).reshape(-1)
    w_new = np.sign(w_old) * np.abs(w_new_abs)

    valid_vtarget = np.isfinite(vtarget)
    w_new[~valid_vtarget] = w_old[~valid_vtarget]

    w_shape = np.asarray(synapse_params.weights).shape
    print(f"synapse_params.weights 已按各 (pre_pop, post_pop) 的生物psp数据重置。共{valid_vtarget.sum()}条突触权重被重置。weights乘以{0.5/rho_init}")
    return replace(synapse_params, weights=w_new.reshape(w_shape)*(0.5/rho_init))


def delete_small_weights(
    synapse_params,
    min_weight=0.1,
):
    """
    删除 `synapse_params.weights < min_weight` 的突触，并同步过滤相关字段。

    返回:
      - synapse_params_new: 过滤后的 synapse_params
      - network: 若传入 network，则同步更新其 n_edges/synapses 后返回
    """
    w = np.abs(np.asarray(synapse_params.weights).reshape(-1))
    keep_mask = w >= float(min_weight)

    if np.all(keep_mask):
        print(f"Pruned recurrent synapses (<{min_weight}): removed 0, kept {synapse_params.n_edges}")
        return synapse_params

    idx = np.asarray(synapse_params.indices)
    delays = np.asarray(synapse_params.delays)
    w_full = np.asarray(synapse_params.weights)
    n_edges_new = int(keep_mask.sum())

    # delays 在常规构图时是长度为 n_edges 的数组；若是标量(按 delay 切分后)，则保持原值。
    if delays.ndim == 0:
        delays_new = delays
    else:
        delays_new = delays[keep_mask]

    synapse_params_new = replace(
        synapse_params,
        n_edges=n_edges_new,
        indices=idx[keep_mask],
        weights=w_full[keep_mask],
        delays=delays_new,
    )

    print(f"Pruned recurrent synapses (<{min_weight}): removed {(~keep_mask).sum()}, kept {n_edges_new}")
    return synapse_params_new

def reset_synapse_weights_with_rho(
    weights,
    pop_name_indices,
    n_receptors,
    rec_csr,
    eps,
    sigmoid_mode=False,
    pop_names=None,
    weight_scale=0.5,
    robust_q_low = 0.0,
    robust_q_high = 1.0,
):
    """
    对 recurrent 连接按 (pre_pop, post_pop) 分组：
    1) 将原始权重组内线性重标定到 [0,1]，并做 [eps, 1-eps] clip，作为 rho 初始化；
    2) 将组内权重统一设为 w_mean，满足 sum(rho * w_mean) == sum(w_orig)。

    注意：recurrent 的可训练权重顺序跟 BrainPy CSR(nnz) 对齐，而非 synapse_params.indices 原始顺序。
    """

    recurrent_list = [np.asarray(w, dtype=np.float32).reshape(-1) for w in weights["recurrent"]]
    recurrent_shapes = [np.asarray(w).shape for w in weights["recurrent"]]
    recurrent_sizes = [arr.size for arr in recurrent_list]
    w_flat_orig = np.concatenate(recurrent_list, axis=0)

    if int(rec_csr.nnz) != w_flat_orig.shape[0]:
        raise ValueError(
            "recurrent weights size and recurrent CSR nnz mismatch: "
            f"{w_flat_orig.shape[0]} vs {int(rec_csr.nnz)}"
        )
    # CSR: indices 是列索引(这里对应 post*receptor)；行索引(pre)需由 indptr 还原。
    post_receptor_flat = np.asarray(rec_csr.indices, dtype=np.int64)
    row_counts = np.diff(np.asarray(rec_csr.indptr, dtype=np.int64))
    pre_neuron_id = np.repeat(
        np.arange(row_counts.shape[0], dtype=np.int64),
        row_counts,
    )
    if pre_neuron_id.shape[0] != post_receptor_flat.shape[0]:
        raise ValueError(
            "CSR indptr/indices inconsistent: "
            f"pre size {pre_neuron_id.shape[0]} vs post size {post_receptor_flat.shape[0]}"
        )
    post_neuron_id = post_receptor_flat // n_receptors

    if pop_names is None:
        pop_names = [k for k in pop_name_indices.keys() if len(pop_name_indices[k]) > 0]

    neuron_to_pop = {}
    for pop in pop_names:
        for nid in np.asarray(pop_name_indices[pop]).ravel():
            neuron_to_pop[int(nid)] = pop

    group_to_indices = {}
    for i in range(w_flat_orig.shape[0]):
        pre_pop = neuron_to_pop.get(int(pre_neuron_id[i]))
        post_pop = neuron_to_pop.get(int(post_neuron_id[i]))
        if pre_pop is None or post_pop is None:
            continue
        key = (pre_pop, post_pop)
        if key not in group_to_indices:
            group_to_indices[key] = []
        group_to_indices[key].append(i)

    rho_flat = np.full_like(w_flat_orig, 0.5, dtype=np.float32)
    w_flat_new = w_flat_orig.copy()
    rho_sum_guard = float(eps)

    # 使用稳健分位数缩放，降低极端离群点对 rho 的挤压效应。
    # 例如 q=0.02/0.98 时，仅对两端各 2% 的值做截断后再线性映射。

    robust_min_group_size = 16

    for idx_list in group_to_indices.values():
        idx = np.asarray(idx_list, dtype=np.int64)
        if idx.size == 0:
            continue
        w_group = w_flat_orig[idx]
        use_robust = idx.size >= robust_min_group_size
        if use_robust and robust_q_low != 0.0 and robust_q_high != 1.0:
            w_lo = float(np.quantile(w_group, robust_q_low))
            w_hi = float(np.quantile(w_group, robust_q_high))
        else:
            w_lo = float(np.min(w_group))
            w_hi = float(np.max(w_group))

        if w_hi > w_lo:
            w_scaled = np.clip(w_group, w_lo, w_hi)
            rho_group = (w_scaled - w_lo) / (w_hi - w_lo)
        else:
            rho_group = np.full_like(w_group, 0.5, dtype=np.float32)
        rho_group = np.clip(rho_group, eps, 1.0 - eps).astype(np.float32)
        rho_flat[idx] = rho_group

        sum_w = float(np.sum(w_group))
        sum_rho = float(np.sum(rho_group))
        w_mean = sum_w / max(sum_rho, rho_sum_guard)
        w_flat_new[idx] = np.float32(w_mean)

    recurrent_new = []
    recurrent_rho = []
    start = 0
    for shape, size in zip(recurrent_shapes, recurrent_sizes):
        end = start + size
        recurrent_new.append(jnp.asarray(w_flat_new[start:end].reshape(shape)*weight_scale, dtype=jnp.float32))
        recurrent_rho.append(jnp.asarray(rho_flat[start:end].reshape(shape), dtype=jnp.float32))
        start = end

    if sigmoid_mode:
        sigmoid_eps = 1e-6
        recurrent_rho = [jnp.log(r) - jnp.log(1.0 - r)for r in recurrent_rho]

    weights_new = dict(weights)
    weights_new["recurrent"] = recurrent_new
    print(f"根据计算同时重置了recurrent weights 和初始化 rho。")
    return FrozenDict(weights_new), recurrent_rho

def apply_scales(params, scale_params: dict):
    """
    对 InputParams 或 SynapseParams 做 receptor-wise 缩放（ee, ie, ei, ii）并乘全局 k。
    scale_params 需包含: k_ee, k_ie, k_ei, k_ii, k。
    返回新的 params（类型与传入相同），不修改原对象。
    """
    ridx = np.asarray(
        params.indices[:, 0] % int(params.n_receptors), dtype=np.int32
    )
    scales = np.asarray(
        [
            scale_params["k_ee"] if "k_ee" in scale_params else 1.0,
            scale_params["k_ie"] if "k_ie" in scale_params else 1.0,
            scale_params["k_ei"] if "k_ei" in scale_params else 1.0,
            scale_params["k_ii"] if "k_ii" in scale_params else 1.0,
        ],
        dtype=np.float32,
    )
    k = float(scale_params["k"]) if "k" in scale_params else 1.0
    w = np.asarray(params.weights, dtype=np.float32)
    w = w * scales[ridx] * k
    print(f"weights 已按 receptor-wise 和 global scale 缩放。k_ee={scales[0]}, k_ie={scales[1]}, k_ei={scales[2]}, k_ii={scales[3]}, k={k}")
    return replace(params, weights=w)


def build_default_current_add(
    default_current,
    n_neurons: int,
    n_receptors: int,
    laminar_indices: dict,
    dtype=jnp.float32,
):
    """
    构建与 input_layer.bkg_weights 对齐的 default current 增量（flatten 后长度 n_neurons*n_receptors）。
    注入规则：兴奋性神经元加到 receptor 0，抑制性神经元加到 receptor 2。
    """
    n_neurons = int(n_neurons)
    n_receptors = int(n_receptors)
    if n_receptors <= 2:
        raise ValueError(
            f"n_receptors={n_receptors} 不支持 e->0 / i->2 的注入规则"
        )

    is_exc_np = np.zeros((n_neurons,), dtype=np.float32)
    for pop_key, neuron_idx in laminar_indices.items():
        if str(pop_key).endswith("e"):
            idx = np.asarray(neuron_idx, dtype=np.int32).ravel()
            if idx.size > 0:
                valid = (idx >= 0) & (idx < n_neurons)
                is_exc_np[idx[valid]] = 1.0

    is_exc = jnp.asarray(is_exc_np, dtype=dtype)
    dc = jnp.asarray(default_current, dtype=dtype)
    default_add = jnp.zeros((n_neurons, n_receptors), dtype=dtype)
    default_add = default_add.at[:, 0].set(dc * is_exc)
    default_add = default_add.at[:, 2].set(dc * (1.0 - is_exc))
    return default_add.reshape(-1)


class InputLayer(bp.DynSysGroup):
    def __init__(
        self,
        input_params,
        tau_syn,
        use_dale_law,
        sparse_mode="event",
        use_decoded_noise=False,
        noise_data=None,
    ):
        super().__init__()
        self._n_node_receptors = int(tau_syn.shape[0] * tau_syn.shape[-1])
        
        self._use_decoded_noise = bool(use_decoded_noise)
        self.noise_data = None if noise_data is None else bm.asarray(noise_data, dtype=bm.float_)
        input_csr, weight = to_brainpy_csr(
            input_params, split_receptor=False, split_conn=True,
        )
        conn = bp.conn.SparseMatConn(input_csr)
        if sparse_mode == "event":
            self.input_proj = models.EventCSRLinearDale(
                conn, weight, use_dale_law=use_dale_law
            )
        elif sparse_mode == "value":
            self.input_proj = models.CSRLinearValue(
                conn, weight, use_dale_law=use_dale_law
            )
        else:
            raise ValueError(
                f"Unsupported sparse_mode='{sparse_mode}', expected 'event' or 'value'"
            )
        self.bkg_weights = bp.init.parameter(
            bm.asarray(input_params.bkg_weights, dtype=bm.float_).reshape(-1),sizes=input_params.bkg_weights.shape
        )

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
            # Single Poisson source at constant 1 kHz (assuming 1 ms simulation step).
            rest_of_brain = bm.random.poisson(1.0)
            return self.bkg_weights * rest_of_brain
        rest_of_brain = bm.random.poisson(1.0, size=(inp.shape[0],))
        return self.bkg_weights * rest_of_brain[:, None]

    def update(self, inp):
        input_current = self.input_proj(inp)
        input_current += self.gen_noise(inp)
        return input_current
    
class Model(bp.DynSysGroup):
    def __init__(
            self,
            recurrent_layer: models.BillehColumn,
            input_layer: InputLayer,
        ):
        super().__init__()
        self.input_layer = input_layer
        self.recurrent_layer = recurrent_layer
    
    def update(self, input=None):
        # inp shape: [batch, n_input]
        rnn_input = self.input_layer(input)
        self.recurrent_layer(rnn_input)


def build_readout_indices_xz_centers(
    network: dict,
    readout_neuron,
    n_class: int = 10,
    group_size: int | None = None,
    seed: int = 42,
):
    """
    根据 readout neurons 在 x-z 平面上的坐标，选取 n_class 个较均匀的中心（farthest point sampling），
    再将 readout neurons 按 x-z 距离做“容量受限”的分配，最后生成一个 interleave 排序后的
    readout_indices，使得 reshape(..., group_size, n_class) 后得到 10 组。

    Parameters
    ----------
    network : dict
        需要包含 network["x"], network["z"]，并且 readout_neuron 的 id 对应到它们的索引。
    readout_neuron : array-like
        readout neurons 的 id 列表（长度应为 group_size * n_class）。
    n_class : int
        分类组数，默认 10。
    group_size : int | None
        每组大小。如果为 None，则自动计算 group_size = len(readout_neuron) // n_class。
    seed : int
        随机种子，仅用于初始化 farthest point sampling 的起点（其余为确定性分配）。
    """
    readout_neuron_np = np.asarray(readout_neuron, dtype=np.int64)
    n_total = int(readout_neuron_np.shape[0])

    if group_size is None:
        if n_total % n_class != 0:
            raise ValueError(
                f"len(readout_neuron)={n_total} 不能整除 n_class={n_class}，请先截断到 group_size*n_class。"
            )
        group_size = n_total // n_class
    else:
        group_size = int(group_size)
        if group_size * n_class != n_total:
            raise ValueError(
                f"group_size*n_class={group_size*n_class} 但 len(readout_neuron)={n_total} 不一致。"
            )

    if "x" not in network or "z" not in network:
        raise KeyError("network 里需要包含 keys: 'x' 和 'z' 用于 x-z 距离聚类。")

    x = np.asarray(network["x"])
    z = np.asarray(network["z"])

    coords_xz = np.stack([x[readout_neuron_np], z[readout_neuron_np]], axis=1)  # (n_total, 2)

    rng = np.random.default_rng(int(seed))
    k = int(n_class)

    # 最远点采样（farthest point sampling）选中心：起点随机，其余确定性
    idx0 = int(rng.integers(0, n_total))
    chosen_mask = np.zeros(n_total, dtype=bool)
    chosen_mask[idx0] = True
    chosen_idx = [idx0]

    min_dist2 = np.sum((coords_xz - coords_xz[idx0]) ** 2, axis=1)
    for _ in range(1, k):
        min_dist2_masked = min_dist2.copy()
        min_dist2_masked[chosen_mask] = -1.0  # 让已选点不会被再次选中
        next_idx = int(np.argmax(min_dist2_masked))
        chosen_idx.append(next_idx)
        chosen_mask[next_idx] = True

        dist2_new = np.sum((coords_xz - coords_xz[next_idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2_new)

    center_coords_xz = coords_xz[np.array(chosen_idx, dtype=int)]  # (k,2)

    # 每个点到每个中心的距离平方
    dist2 = np.sum(
        (coords_xz[:, None, :] - center_coords_xz[None, :, :]) ** 2,
        axis=-1,
    )  # (n_total,k)

    # 容量受限分配：每组恰好 group_size 个点
    center_order = np.argsort(dist2, axis=1)  # (n_total,k)
    caps = np.full(k, group_size, dtype=int)
    cap_remaining = caps.copy()

    min_dist2_to_nearest = dist2[np.arange(n_total), center_order[:, 0]]
    point_order = np.argsort(min_dist2_to_nearest)

    labels = -np.ones(n_total, dtype=int)
    for pi in point_order:
        for c in center_order[pi]:
            if cap_remaining[c] > 0:
                labels[pi] = c
                cap_remaining[c] -= 1
                break

    # 容错：理论上不会发生，但避免实现差异导致的 -1
    if np.any(labels < 0):
        unassigned = np.where(labels < 0)[0]
        for pi in unassigned:
            labels[pi] = int(np.argmin(dist2[pi]))

    # 构造 reshape(group_size, n_class) 后类维度对应 labels 的 interleave 顺序
    cluster_positions = []
    for c in range(k):
        pos = np.where(labels == c)[0]
        # 类内按靠近中心的顺序排（可让结果更稳定/更“中心化”）
        pos = pos[np.argsort(dist2[pos, c])]
        if pos.shape[0] != group_size:
            raise RuntimeError(
                f"cluster {c} size mismatch: got {pos.shape[0]}, expected {group_size}"
            )
        cluster_positions.append(pos)

    readout_indices = np.empty(n_total, dtype=int)
    for j in range(group_size):
        for c in range(k):
            # interleave: (j*k + c) -> 第 j 行第 c 列
            readout_indices[j * k + c] = int(cluster_positions[c][j])
    
    print(f"根据中心生成了{n_class}组读出神经元，每组{group_size}个点")
    return readout_indices


def build_readout_neurons_xz_nearest_max_group(
    network: dict,
    readout_neuron,
    n_class: int = 10,
    seed: int = 42,
    group_size_max_custom: int | None = None,
):
    """
    B: 在满足“最邻近中心归属”的条件下，group_size 取（理论最大 或 自定义上限）中的较小值。

    做法：
    1) 用 x-z farthest point sampling 选出 n_class 个中心（中心仍来自 readout 点）
    2) 对每个 readout 点，计算它到各中心的 x-z 距离，归属到最近中心
    3) 令 count[c] = 归属到中心 c 的点数，则能满足“每类都满足最邻近归属”的最大 group_size 为 min(count)
    4) 每类只取 group_size_max 个最近的点；把所有类按 interleave 顺序拼成 readout_neuron
       使得 reshape(..., group_size, n_class) 后自然得到 10 组
    5) 返回 readout_indices 为恒等置换（因为 readout_neuron 已按 interleave 排好）
    """
    readout_neuron_np = np.asarray(readout_neuron, dtype=np.int64)
    n_total = int(readout_neuron_np.shape[0])
    k = int(n_class)

    if n_total < k:
        raise ValueError(f"readout_neuron too small: got {n_total}, need >= n_class={k}")

    if "x" not in network or "z" not in network:
        raise KeyError("network 里需要包含 keys: 'x' 和 'z' 用于 x-z 距离聚类。")

    x = np.asarray(network["x"])
    z = np.asarray(network["z"])
    coords_xz = np.stack([x[readout_neuron_np], z[readout_neuron_np]], axis=1)  # (n_total,2)

    rng = np.random.default_rng(int(seed))

    # farthest point sampling (x-z) 选中心
    idx0 = int(rng.integers(0, n_total))
    chosen_mask = np.zeros(n_total, dtype=bool)
    chosen_mask[idx0] = True
    chosen_idx = [idx0]

    min_dist2 = np.sum((coords_xz - coords_xz[idx0]) ** 2, axis=1)
    for _ in range(1, k):
        min_dist2_masked = min_dist2.copy()
        min_dist2_masked[chosen_mask] = -1.0
        next_idx = int(np.argmax(min_dist2_masked))
        chosen_idx.append(next_idx)
        chosen_mask[next_idx] = True

        dist2_new = np.sum((coords_xz - coords_xz[next_idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2_new)

    centers_xz = coords_xz[np.array(chosen_idx, dtype=int)]  # (k,2)

    # 最近中心归属
    dist2_all = np.sum(
        (coords_xz[:, None, :] - centers_xz[None, :, :]) ** 2,
        axis=-1,
    )  # (n_total,k)
    nearest_center = np.argmin(dist2_all, axis=1)  # (n_total,)

    counts = np.bincount(nearest_center, minlength=k)
    group_size_max_theoretical = int(np.min(counts))
    if group_size_max_theoretical <= 0:
        raise RuntimeError(
            f"Nearest-center counts too imbalanced, min(counts)={group_size_max_theoretical}. "
            "Try different seed or check readout_neuron geometry."
        )

    if group_size_max_custom is not None:
        group_size_max_custom = int(group_size_max_custom)
        if group_size_max_custom <= 0:
            raise ValueError(f"group_size_max_custom must be positive, got {group_size_max_custom}")
        group_size_max = min(group_size_max_theoretical, group_size_max_custom)
    else:
        group_size_max = group_size_max_theoretical

    if group_size_max <= 0:
        raise RuntimeError(
            f"group_size_max after applying custom limit is invalid: group_size_max={group_size_max}"
        )

    # 每个类取 group_size_max 个最近点（保证这些点的最近中心仍是该类）
    class_positions = []
    for c in range(k):
        pos = np.where(nearest_center == c)[0]
        # 按点到该中心距离升序，取前 group_size_max
        pos = pos[np.argsort(dist2_all[pos, c])]
        pos = pos[:group_size_max]
        if pos.shape[0] != group_size_max:
            raise RuntimeError(
                f"class {c} selection size mismatch: got {pos.shape[0]}, expected {group_size_max}"
            )
        class_positions.append(pos)

    n_selected = group_size_max * k

    # 交错拼接：索引 p=j*k+c 对应第 c 类的第 j 个点
    order_positions = np.empty(n_selected, dtype=int)
    for j in range(group_size_max):
        for c in range(k):
            order_positions[j * k + c] = int(class_positions[c][j])

    readout_neuron_selected = readout_neuron_np[order_positions]

    # 因为 readout_neuron_selected 已经 interleave 排好，readout_indices 恒等即可
    readout_indices = np.arange(n_selected, dtype=int)

    print(f"根据最近中心生成了{n_class}组读出神经元，每组{group_size_max}个点")

    return readout_neuron_selected, group_size_max, readout_indices

def validate_recurrent_weights_mapping(weights_recurrent, synapse_params, rec_csr, atol=1e-6):
    """
    校验 `weights['recurrent']` 与 `synapse_params.weights` 在 (pre, post*receptor) 上是否一致。

    约定：
    - `rec_csr` 为 `to_brainpy_csr(..., split_conn=True)` 生成的 CSR，因此：
      - `rec_csr.indices` 是列索引，对应 `post*receptor`
      - `rec_csr.indptr` 用于从行索引（对应 `pre`）还原一一对应的 nnz 序列
    """
    w_rec_flat = np.concatenate(
        [np.asarray(w, dtype=np.float32).reshape(-1) for w in weights_recurrent], axis=0
    )
    if w_rec_flat.shape[0] != int(rec_csr.nnz):
        raise ValueError(
            "validate failed: recurrent weight size and rec_csr.nnz mismatch: "
            f"{w_rec_flat.shape[0]} vs {int(rec_csr.nnz)}"
        )

    # 从 synapse_params 构建 (pre, post*receptor) -> sum(weight) 的映射
    idx_syn = np.asarray(synapse_params.indices, dtype=np.int64)
    w_syn = np.asarray(synapse_params.weights, dtype=np.float32).reshape(-1)
    syn_map = {}
    for i in range(w_syn.shape[0]):
        key = (int(idx_syn[i, 1]), int(idx_syn[i, 0]))  # (pre, post*receptor)
        syn_map[key] = syn_map.get(key, 0.0) + float(w_syn[i])

    # 按 rec_csr nnz 顺序还原 (pre, post*receptor)
    post_rec_csr = np.asarray(rec_csr.indices, dtype=np.int64)
    row_counts = np.diff(np.asarray(rec_csr.indptr, dtype=np.int64))
    pre_csr = np.repeat(np.arange(row_counts.shape[0], dtype=np.int64), row_counts)
    if pre_csr.shape[0] != post_rec_csr.shape[0]:
        raise ValueError(
            "validate failed: CSR indptr/indices inconsistent: "
            f"{pre_csr.shape[0]} vs {post_rec_csr.shape[0]}"
        )

    missing = 0
    mismatch = 0
    max_abs_err = 0.0
    worst = None

    for i in range(w_rec_flat.shape[0]):
        key = (int(pre_csr[i]), int(post_rec_csr[i]))
        if key not in syn_map:
            missing += 1
            continue
        expected = syn_map[key]
        actual = float(w_rec_flat[i])
        err = abs(actual - expected)
        if err > atol:
            mismatch += 1
            if err > max_abs_err:
                max_abs_err = err
                worst = (key, actual, expected)

    print(
        "[validate_recurrent_mapping] "
        f"total={w_rec_flat.shape[0]}, missing={missing}, mismatch(>{atol})={mismatch}, max_abs_err={max_abs_err:.6e}"
    )
    if worst is not None:
        key, actual, expected = worst
        print(
            "[validate_recurrent_mapping] worst key="
            f"{key}, recurrent={actual:.6e}, synapse_agg={expected:.6e}"
        )


def check_reset_with_rho(
    weights_rec_before,
    weights_rec_after,
    recurrent_rho_init,
    pop_name_indices,
    n_receptors,
    rec_csr,
    eps,
    sigmoid_mode=False,
    weight_scale=0.5,
    atol=1e-5,
):
    """
    验证 reset_synapse_weights_with_rho 是否满足以下约束（逐 (pre_pop, post_pop) 分组）：
    1) rho 来自原始权重的 min-max + clip(eps, 1-eps)
    2) 组内重置后权重为常数 w_mean
    3) 若函数内部对重置后的权重做了整体缩放（例如 *0.5），则应满足
       sum(rho * (weight_scale * w_mean)) == weight_scale * sum(w_orig)（在 atol 容差内）
    """
    w0 = np.concatenate([np.asarray(w).reshape(-1) for w in weights_rec_before], axis=0)
    w1 = np.concatenate([np.asarray(w).reshape(-1) for w in weights_rec_after], axis=0)
    rho = np.concatenate([np.asarray(r).reshape(-1) for r in recurrent_rho_init], axis=0)

    if sigmoid_mode:
        rho = np.clip(1.0 / (1.0 + np.exp(-rho)), eps, 1.0 - eps)

    if w0.shape[0] != int(rec_csr.nnz) or w1.shape[0] != int(rec_csr.nnz):
        raise ValueError(
            "check_reset_with_rho failed: recurrent size and rec_csr.nnz mismatch: "
            f"before={w0.shape[0]}, after={w1.shape[0]}, nnz={int(rec_csr.nnz)}"
        )

    post_rec = np.asarray(rec_csr.indices, dtype=np.int64)
    row_counts = np.diff(np.asarray(rec_csr.indptr, dtype=np.int64))
    pre = np.repeat(np.arange(row_counts.shape[0], dtype=np.int64), row_counts)
    post = post_rec // int(n_receptors)

    neuron_to_pop = {}
    for pop, ids in pop_name_indices.items():
        for nid in np.asarray(ids).ravel():
            neuron_to_pop[int(nid)] = pop

    groups = {}
    for i in range(w0.shape[0]):
        pre_pop = neuron_to_pop.get(int(pre[i]))
        post_pop = neuron_to_pop.get(int(post[i]))
        if pre_pop is None or post_pop is None:
            continue
        groups.setdefault((pre_pop, post_pop), []).append(i)

    bad = []
    for key, idx_list in groups.items():
        idx = np.asarray(idx_list, dtype=np.int64)
        g0 = w0[idx]
        g1 = w1[idx]
        gr = rho[idx]

        if np.min(gr) < eps - 1e-8 or np.max(gr) > 1.0 - eps + 1e-8:
            bad.append((key, "rho_out_of_range"))

        if not np.allclose(g1, g1[0], atol=atol, rtol=0.0):
            bad.append((key, "w_not_constant_in_group"))

        lhs = float(np.sum(gr * g1))
        rhs = float(np.sum(g0)) * float(weight_scale)
        if not np.isclose(lhs, rhs, atol=atol, rtol=1e-6):
            bad.append((key, f"sum_constraint_fail lhs={lhs:.6e} rhs_scaled={rhs:.6e}"))

        mn = float(np.min(g0))
        mx = float(np.max(g0))
        if mx > mn:
            gt = (g0 - mn) / (mx - mn)
        else:
            gt = np.full_like(g0, 0.5, dtype=np.float32)
        gt = np.clip(gt, eps, 1.0 - eps)
        if not np.allclose(gr, gt, atol=atol, rtol=1e-6):
            bad.append((key, "rho_not_from_minmax_clip"))

    print(f"[check_reset_with_rho] groups={len(groups)}, bad={len(bad)}")
    for item in bad[:20]:
        print("[check_reset_with_rho] BAD:", item)
    return bad
