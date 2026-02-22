import math

import tensorflow as tf


# could be simplified with bp.encoding.Encoder
def generate_pure_classification_data_set_from_generator(
    data_usage=0,
    n_input_neurons=256,
    intensity=1,
    im_slice=100,
    pre_delay=50,
    post_delay=150,
    pre_chunks=3,
    resp_chunks=1,
    post_chunks=2,
    current_input=True,
    dataset="mnist",
    path=None,
    rot90=False,
):
    from_lgn = False
    # hard code lgn scale for the case from_lgn=False
    mimc_lgn_std, mimc_lgn_mean = 0.02082, 0.02
    # data_usage: 0, train; 1, test

    if dataset.lower() == "cifar100":
        all_ds = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    elif dataset.lower() == "cifar10":
        all_ds = tf.keras.datasets.cifar10.load_data()
    elif dataset.lower() == "mnist":
        all_ds = tf.keras.datasets.mnist.load_data()
    elif dataset.lower() == "fashion_mnist":
        all_ds = tf.keras.datasets.fashion_mnist.load_data()

    if data_usage == 0:
        images, labels = all_ds[data_usage]
    else:
        images, labels = all_ds[data_usage]
        # choose fixed validation set to minimize the variance
        # images = images[0:1280] # normally, the batch size is 64
        # labels = labels[0:1280]

    # LGN module only can receive gray-scale images with the value in [-intensity,intensity] from black to white
    if len(images.shape) > 3:
        images = tf.image.rgb_to_grayscale(images) / 255
    else:
        images = images[..., None] / 255

    if rot90:
        images = tf.image.rot90(images)

    if from_lgn:
        assert n_input_neurons == 17400
        lgn = lgn_model.LGN()  # noqa: F821
    seq_len = pre_delay + im_slice + post_delay
    chunk_size = 50  # ms
    n_chunks = int(seq_len / chunk_size)

    assert n_chunks == resp_chunks + pre_chunks + post_chunks

    def _g():
        for ind in range(images.shape[0]):
            if from_lgn:
                # LGN model only receives 120 x 240, the core part only receives an eclipse TODO
                img = tf.image.resize_with_pad(images[ind], 120, 240, method="lanczos5")
                # maintain the images for a while
                tiled_img = tf.tile(img[None, ...], (im_slice, 1, 1, 1))
                # make it in [-intensity, intensity]
                tiled_img = (tiled_img - 0.5) * intensity / 0.5
            else:
                col = int(math.sqrt(n_input_neurons))
                row = n_input_neurons // col
                assert col * row == n_input_neurons
                # to mimic the 17400 dim of LGN output
                img = tf.image.resize_with_pad(images[ind], col, row, method="lanczos5")
                # maintain the images for a while
                tiled_img = tf.tile(img[None, ...], (im_slice, 1, 1, 1))

            # add an empty period before a period of real image for continuing classification
            z1 = tf.tile(tf.zeros_like(img)[None, ...], (pre_delay, 1, 1, 1))
            z2 = tf.tile(tf.zeros_like(img)[None, ...], (post_delay, 1, 1, 1))
            videos = tf.concat((z1, tiled_img, z2), 0)
            if from_lgn:
                spatial = lgn.spatial_response(videos)
                firing_rates = lgn.firing_rates_from_spatial(*spatial)
            else:
                firing_rates = tf.reshape(videos, [-1, n_input_neurons])
            # sample rate
            # assuming dt = 1 ms
            _p = 1 - tf.exp(-firing_rates / 1000.0)
            # _z = tf.cast(fixed_noise < _p, dtype)
            if current_input:
                _z = _p * 1.3
                if not from_lgn:
                    _z = _z * mimc_lgn_std
                    _z = (_z - tf.reduce_mean(_z)) / tf.math.reduce_std(
                        _z
                    ) * mimc_lgn_std + mimc_lgn_mean
            else:
                _z = tf.cast(tf.random.uniform(tf.shape(_p)) < _p, tf.float32)
            label = tf.concat(
                [tf.zeros(pre_chunks)]
                + [labels[ind] * tf.ones(resp_chunks)]
                + [tf.zeros(post_chunks)],
                axis=0,
            )
            weight = tf.concat(
                [0 * tf.ones(pre_chunks)]
                + [tf.ones(resp_chunks)]
                + [0 * tf.ones(post_chunks)],
                axis=0,
            )
            # for plotting, label the image when it holds on
            image_labels = tf.concat(
                [tf.zeros(int(pre_delay / chunk_size))]
                + [labels[ind] * tf.ones(int(im_slice / chunk_size))]
                + [tf.zeros(int(post_delay / chunk_size))],
                axis=0,
            )
            yield _z, label, image_labels, weight

    output_dtypes = (tf.float32, tf.int32, tf.int32, tf.float32)
    # when using generator for dataset, it should not contain the batch dim
    output_shapes = (
        tf.TensorShape((seq_len, n_input_neurons)),
        tf.TensorShape((n_chunks)),
        tf.TensorShape((n_chunks)),
        tf.TensorShape((n_chunks)),
    )
    data_set = tf.data.Dataset.from_generator(
        _g, output_dtypes, output_shapes=output_shapes
    ).map(
        lambda _a, _b, _c, _d: (
            tf.cast(_a, tf.float32),
            tf.cast(_b, tf.int32),
            tf.cast(_c, tf.int32),
            tf.cast(_d, tf.float32),
        )
    )
    return data_set
