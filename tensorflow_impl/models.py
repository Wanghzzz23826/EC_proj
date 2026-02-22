import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers, layers


def gauss_pseudo(v_scaled, sigma, amplitude):
    return tf.math.exp(-tf.square(v_scaled) / tf.square(sigma)) * amplitude


def pseudo_derivative(v_scaled, dampening_factor):
    return dampening_factor * tf.maximum(1 - tf.abs(v_scaled), 0)


@tf.custom_gradient
def spike_gauss(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)
        # dz_dv_scaled = pseudo_derivative(v_scaled, .3)
        # dz_dv_scaled = 1.

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name="spike_gauss"), grad


def exp_convolve(tensor, decay=0.8, reverse=False, initializer=None, axis=0):
    rank = len(tensor.get_shape())
    perm = np.arange(rank)
    perm[0], perm[axis] = perm[axis], perm[0]
    tensor = tf.transpose(tensor, perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor[0])

    def scan_fun(_acc, _t):
        return _acc * decay + _t

    filtered = tf.scan(scan_fun, tensor, reverse=reverse, initializer=initializer)

    filtered = tf.transpose(filtered, perm)
    return filtered


class SparseLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        indices,
        weights,
        dense_shape,
        bkg_weights,
        down_sampled_decode_noise_path=None,
        use_decoded_noise=False,
        dtype=tf.float32,
        scale=[1, 1],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale = scale
        self._indices = indices
        self._weights = weights
        self._dense_shape = dense_shape
        self._max_batch = int(2**31 / weights.shape[0])
        self._dtype = dtype
        self._bkg_weights = bkg_weights
        self._use_decoded_noise = use_decoded_noise
        if use_decoded_noise:
            from scipy.io import loadmat

            tmp = loadmat(down_sampled_decode_noise_path)
            self.noise_data = tf.convert_to_tensor(
                tmp["additive_noise"].reshape(-1), dtype=self.compute_dtype
            )

    def call(self, inp):
        tf_shp = tf.unstack(tf.shape(inp))
        shp = inp.shape.as_list()
        for i, a in enumerate(shp):
            if a is None:
                shp[i] = tf_shp[i]

        sparse_w_in = tf.sparse.SparseTensor(
            self._indices, self._weights, self._dense_shape
        )
        inp = tf.reshape(inp, (shp[0] * shp[1], shp[2]))

        input_current = tf.sparse.sparse_dense_matmul(
            sparse_w_in, tf.cast(inp, tf.float32), adjoint_b=True
        )
        input_current = tf.transpose(input_current)
        input_current = tf.cast(input_current, self._dtype)
        if self._use_decoded_noise:
            # quick noise: sample every step
            gen_ind_quick = tf.random.uniform(
                shape=(shp[0], shp[1], self._dense_shape[0]),
                maxval=28406000,
                dtype=tf.int64,
            )  # batch, seq_len, neurons*4
            # slow noise: sample every trial
            gen_ind_slow = tf.random.uniform(
                shape=(shp[0], 1, self._dense_shape[0]), maxval=28406000, dtype=tf.int64
            )  # batch, 1, neurons*4
            gen_ind_slow = tf.tile(
                gen_ind_slow, [1, shp[1], 1]
            )  # batch, seq_len, neurons*4
            quick_noise = tf.gather(self.noise_data, gen_ind_quick)
            slow_noise = tf.gather(self.noise_data, gen_ind_slow)

            noise_input = (
                tf.cast(
                    tf.ones_like(self._bkg_weights[None, None]) * self.scale[0],
                    self.compute_dtype,
                )
                * quick_noise
                + tf.cast(
                    tf.ones_like(self._bkg_weights[None, None]) * self.scale[1],
                    self.compute_dtype,
                )
                * slow_noise
            )
        else:
            rest_of_brain = tf.reduce_sum(
                tf.cast(
                    tf.random.uniform((shp[0], shp[1], 10)) < 0.1, self.compute_dtype
                ),
                -1,
            )
            noise_input = (
                tf.cast(self._bkg_weights[None, None], self.compute_dtype)
                * rest_of_brain[..., None]
                / 10.0
            )

        input_current = tf.reshape(input_current, (shp[0], shp[1], -1)) + noise_input
        return input_current


class SignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, positive):
        self._positive = positive

    def __call__(self, w):
        sign_corrected_w = tf.where(self._positive, tf.nn.relu(w), -tf.nn.relu(-w))
        return sign_corrected_w


class SparseSignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, mask, positive):
        self._mask = mask
        self._positive = positive

    def __call__(self, w):
        sign_corrected_w = tf.where(self._positive, tf.nn.relu(w), -tf.nn.relu(-w))
        return tf.where(self._mask, sign_corrected_w, tf.zeros_like(sign_corrected_w))


class StiffRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength, initial_value):
        super().__init__()
        self._strength = strength
        self._initial_value = tf.Variable(initial_value, trainable=False)

    def __call__(self, x):
        return self._strength * tf.reduce_sum(tf.square(x - self._initial_value))


class L2Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength):
        super().__init__()
        self._strength = strength

    def __call__(self, x):
        return self._strength * tf.nn.l2_loss(x)


class GLIF3(tf.keras.layers.Layer):
    def __init__(
        self,
        node_params,
        node_type_ids,
        dt=1.0,
        gauss_std=0.5,
        dampening_factor=0.3,
        spike_gradient=False,
        _return_interal_variables=False,
        scale_voltage: bool = False,
        spk_reset: str = "soft",
    ):
        super().__init__()
        self._params = node_params

        if scale_voltage:
            voltage_scale = self._params["V_th"] - self._params["E_L"]
            voltage_offset = self._params["E_L"]
            self._params["V_th"] = (
                self._params["V_th"] - voltage_offset
            ) / voltage_scale
            self._params["E_L"] = (self._params["E_L"] - voltage_offset) / voltage_scale
            self._params["V_reset"] = (
                self._params["V_reset"] - voltage_offset
            ) / voltage_scale
            self._params["asc_amps"] = (
                self._params["asc_amps"] / voltage_scale[..., None]
            )

        self._node_type_ids = node_type_ids
        self._dt = dt

        self._return_interal_variables = _return_interal_variables

        # for random spike, the instantaneous firing rate when v = v_th
        self._spike_gradient = spike_gradient

        n_receptors = node_params["tau_syn"].shape[1]
        self._n_receptors = n_receptors
        self._n_neurons = len(node_type_ids)
        self._dampening_factor = tf.cast(dampening_factor, self.compute_dtype)
        self._gauss_std = tf.cast(gauss_std, self.compute_dtype)

        tau = self._params["C_m"] / self._params["g"]
        self._decay = np.exp(-dt / tau)
        self._current_factor = 1 / self._params["C_m"] * (1 - self._decay) * tau
        self._syn_decay = np.exp(-dt / np.array(self._params["tau_syn"]))
        self._psc_initial = np.e / np.array(self._params["tau_syn"])

        self.state_size = (
            self._n_neurons,  # z
            self._n_neurons,  # v
            self._n_neurons,  # r
            self._n_neurons,  # asc 1
            self._n_neurons,  # asc 2
        )

        # useless now; it was for training the neuron parameters
        def _f(_v, trainable=False):
            return tf.Variable(
                tf.cast(self._gather(_v), self.compute_dtype), trainable=trainable
            )

        def inv_sigmoid(_x):
            return tf.math.log(_x / (1 - _x))

        # useless
        def custom_val(_v, trainable=False):
            _v = tf.Variable(
                tf.cast(inv_sigmoid(self._gather(_v)), self.compute_dtype),
                trainable=trainable,
            )

            def _g():
                return tf.nn.sigmoid(_v.read_value())

            return _v, _g

        self.v_reset = _f(self._params["V_reset"])
        self.syn_decay = _f(self._syn_decay)
        self.psc_initial = _f(self._psc_initial)
        self.t_ref = _f(self._params["t_ref"])
        self.asc_amps = _f(self._params["asc_amps"], trainable=False)
        # self.param_k = _f(self._params['k'], trainable=True)
        _k = self._params["k"]
        # _k[_k < .0031] = .0007
        # self.param_k, self.param_k_read = custom_val(_k, trainable=False)
        self.k = _f(_k)
        self.v_th = _f(self._params["V_th"])
        self.e_l = _f(self._params["E_L"])
        self.param_g = _f(self._params["g"])
        self.decay = _f(self._decay)
        self.current_factor = _f(self._current_factor)
        if scale_voltage:
            self.voltage_scale = _f(voltage_scale)
            self.voltage_offset = _f(voltage_offset)
        else:
            self.voltage_scale = 1.0
            self.voltage_offset = 0.0
        assert spk_reset in ("soft", "hard")
        self.spk_reset = spk_reset

    def zero_state(self, batch_size, dtype=tf.float32):
        z0_buf = tf.zeros((batch_size, self._n_neurons), dtype)
        v0 = tf.ones((batch_size, self._n_neurons), dtype) * tf.cast(
            self.v_th * 0.0 + 1.0 * self.v_reset, dtype
        )
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_10 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_20 = tf.zeros((batch_size, self._n_neurons), dtype)
        return z0_buf, v0, r0, asc_10, asc_20

    def random_state(self, batch_size, dtype=tf.float32):
        z0_buf = tf.cast(
            tf.random.uniform((batch_size, self._n_neurons), 0, 2, tf.int32),
            dtype,
        )
        v0 = tf.random.uniform(
            (batch_size, self._n_neurons),
            tf.cast(self.v_reset, dtype),
            tf.cast(self.v_th, dtype),
            dtype,
        )
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_10 = tf.random.normal(
            (batch_size, self._n_neurons), mean=-0.28, stddev=1.75, dtype=dtype
        )  # min -87 max 59
        asc_20 = tf.random.normal(
            (batch_size, self._n_neurons), mean=-0.28, stddev=1.75, dtype=dtype
        )
        return z0_buf, v0, r0, asc_10, asc_20

    def _gather(self, prop):
        return tf.gather(prop, self._node_type_ids)

    def call(self, inputs, state):
        batch_size = inputs.shape[0]
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]

        prev_z, v, r, asc_1, asc_2 = state
        new_r = tf.nn.relu(r + prev_z * self.t_ref - self._dt)

        k = self.k
        asc_amps = self.asc_amps
        new_asc_1 = tf.exp(-self._dt * k[:, 0]) * asc_1 + prev_z * asc_amps[:, 0]
        new_asc_2 = tf.exp(-self._dt * k[:, 1]) * asc_2 + prev_z * asc_amps[:, 1]

        reset_current = prev_z * (self.v_reset - self.v_th)
        input_current = inputs
        decayed_v = self.decay * v

        gathered_g = self.param_g * self.e_l
        c1 = input_current + asc_1 + asc_2 + gathered_g
        new_v = decayed_v + self.current_factor * c1
        if self.spk_reset == "soft":
            new_v += reset_current
        else:
            new_v += (self.v_reset - new_v) * prev_z

        normalizer = self.v_th - self.e_l
        v_sc = (new_v - self.v_th) / normalizer

        new_z = spike_gauss(v_sc, self._gauss_std, self._dampening_factor)

        new_z = tf.where(new_r > 0.0, tf.zeros_like(new_z), new_z)

        if self._return_interal_variables:
            outputs = (
                new_z,
                new_v * self.voltage_scale + self.voltage_offset,
                new_asc_1,
                new_asc_2,
            )
        else:
            outputs = (new_z, new_v * self.voltage_scale + self.voltage_offset)
        new_state = (
            new_z,
            new_v,
            new_r,
            new_asc_1,
            new_asc_2,
        )

        return outputs, new_state


class Alpha(tf.keras.layers.Layer):
    def __init__(self, num_units, tau_syn, dt=1.0, **kwargs):
        super(Alpha, self).__init__(**kwargs)
        self.tau_syn = tau_syn
        self.dt = dt
        self.syn_decay = np.exp(-dt / tau_syn)
        self.psc_initial = np.e / tau_syn
        self.num_units = num_units
        self.state_size = (
            num_units,  # psc
            num_units,  # psc rise
        )

    def zero_state(self, batch_size, dtype=tf.float32):
        psc_rise0 = tf.zeros((batch_size, self.num_units), dtype)
        psc0 = tf.zeros((batch_size, self.num_units), dtype)
        return psc0, psc_rise0

    def call(self, inputs, state):
        psc, psc_rise = state
        new_psc_rise = self.syn_decay * psc_rise + inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self.dt * self.syn_decay * psc_rise

        return (new_psc, new_psc_rise), (new_psc, new_psc_rise)


class SparseLinearDale(tf.keras.layers.Layer):
    def __init__(
        self,
        indices,
        weights,
        dense_shape,
        trainable=True,
        use_dale_law=True,
    ):
        super().__init__()

        self.weights_positive = tf.Variable(
            weights >= 0.0, name="weights_sign", trainable=False
        )
        if use_dale_law:
            self.weight_values = tf.Variable(
                weights,
                name="sparse_weights",
                constraint=SignedConstraint(self.weights_positive),
                trainable=trainable,
            )
        else:
            self.weight_values = tf.Variable(
                weights,
                name="sparse_weights",
                constraint=None,
                trainable=trainable,
            )
        self.indices = tf.Variable(indices, trainable=False)
        self.dense_shape = dense_shape

    def call(self, inputs):
        sparse_w = tf.sparse.SparseTensor(
            self.indices,
            self.weight_values,
            self.dense_shape,
        )

        currents = tf.sparse.sparse_dense_matmul(
            sparse_w, tf.cast(inputs, tf.float32), adjoint_b=True
        )
        # reorder batch to axis 0
        currents = tf.transpose(currents)
        return currents


class Projection(tf.keras.layers.Layer):
    def __init__(
        self,
        network,
        dt=1.0,
        max_delay=5,
        trainable=True,
        use_dale_law=True,
        linear_layer=SparseLinearDale,
    ):
        super().__init__()
        self._dt = dt

        n_receptors = network["node_params"]["tau_syn"].shape[1]
        self._params = network["node_params"]
        self._n_receptors = n_receptors
        self._n_neurons = int(network["n_nodes"])

        tau = self._params["C_m"] / self._params["g"]
        self._decay = np.exp(-dt / tau)

        # synapses: target_ids, source_ids, weights, delays

        self.max_delay = int(
            np.round(np.min([np.max(network["synapses"]["delays"]), max_delay]))
        )

        self.state_size = self._n_neurons * self.max_delay

        indices, weights, dense_shape = (
            network["synapses"]["indices"],
            network["synapses"]["weights"],
            network["synapses"]["dense_shape"],
        )
        delays = np.round(
            np.clip(network["synapses"]["delays"], dt, self.max_delay) / dt
        ).astype(np.int32)
        dense_shape = dense_shape[0], self.max_delay * dense_shape[1]
        indices = indices.copy()
        indices[:, 1] = indices[:, 1] + self._n_neurons * (delays - 1)
        weights = weights.astype(np.float32)

        self.linear = linear_layer(
            indices, weights, dense_shape, trainable, use_dale_law
        )

    def zero_state(self, batch_size, dtype=tf.float32):
        z0_buf = tf.zeros((batch_size, self._n_neurons * self.max_delay), dtype)
        return z0_buf

    def call(self, inputs, state):
        batch_size = inputs.shape[0]
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]

        (z_buf,) = state
        new_z = inputs
        shaped_z_buf = tf.reshape(z_buf, (-1, self.max_delay, self._n_neurons))

        i = self.linear(z_buf)

        new_shaped_z_buf = tf.concat((new_z[:, None], shaped_z_buf[:, :-1]), 1)
        new_z_buf = tf.reshape(new_shaped_z_buf, (-1, self._n_neurons * self.max_delay))

        new_state = (new_z_buf,)

        return i, new_state


class ConvTranspose1DComm(layers.Layer):
    """
    A wrapper for 1D transposed convolution (similar to ConvTranspose1dComm in JAX).
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int = 1,
        kernel_initializer="ones",
        padding: str = "same",
        kernel_dilation: int = 1,
        name: str = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.kernel_dilation = kernel_dilation

        self.conv_transpose = layers.Conv1DTranspose(
            filters=1,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            dilation_rate=kernel_dilation,
            kernel_initializer=initializers.get(kernel_initializer),
            data_format="channels_last",
            use_bias=False,
        )

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(f"Expected input to be 2D, but got shape {input_shape}")
        super().build(input_shape)

    def call(self, inputs):
        outputs = self.conv_transpose(inputs[..., None])
        return outputs[..., 0]


class SimpleNet(tf.keras.layers.Layer):
    def __init__(
        self,
        network,
        comm,
        dt=1.0,
        gauss_std=0.5,
        dampening_factor=0.3,
        spike_gradient=False,
        max_delay=5,
        _return_interal_variables=True,
    ):
        super().__init__()
        self.neurons_in = GLIF3(
            network["node_params"],
            network["node_type_ids"],
            dt,
            gauss_std,
            dampening_factor,
            spike_gradient,
            _return_interal_variables,
            scale_voltage=False,
        )
        self.neurons_out = GLIF3(
            network["node_params"],
            network["node_type_ids"][::-1],
            dt,
            gauss_std,
            dampening_factor,
            spike_gradient,
            _return_interal_variables,
            scale_voltage=False,
        )
        self._dt = dt
        n_receptors = network["node_params"]["tau_syn"].shape[1]
        assert n_receptors == 4
        self._n_receptors = n_receptors
        self._n_neurons = len(network["node_type_ids"])
        self.comm = comm
        self.syn = Alpha(
            self._n_neurons * self._n_receptors,
            network["node_params"]["tau_syn"][network["node_type_ids"], :].flatten(),
            dt,
        )

        self._return_interal_variables = _return_interal_variables

        self.state_size = (
            self.neurons_in.state_size,
            self.neurons_out.state_size,
            self.syn.state_size,
        )

    def zero_state(self, batch_size, dtype=tf.float32):
        return (
            *self.neurons_in.zero_state(batch_size, dtype),
            *self.neurons_out.zero_state(batch_size, dtype),
            *self.syn.zero_state(batch_size, dtype),
        )

    def call(self, inputs, state):
        batch_size = inputs.shape[0]
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]

        state_neurons_in, state_neurons_out, state_syn = (
            state[: len(self.neurons_in.state_size)],
            state[
                len(self.neurons_in.state_size) : 2 * len(self.neurons_in.state_size)
            ],
            state[2 * len(self.neurons_in.state_size) :],
        )

        output_neurons_in, new_state_neurons_in = self.neurons_in(
            inputs, state_neurons_in
        )
        spike_neurons_in = output_neurons_in[0]
        weighted_spikes = self.comm(spike_neurons_in)
        output_syn, new_state_syn = self.syn(weighted_spikes, state_syn)
        # output_syn[0]: psc
        psc = tf.reshape(output_syn[0], (-1, self._n_neurons, self._n_receptors))
        psc = tf.reduce_sum(psc, axis=-1)
        output_neurons_out, new_state_neurons_out = self.neurons_out(
            psc, state_neurons_out
        )

        new_state = (new_state_neurons_in, new_state_neurons_out, new_state_syn)

        outputs = (
            tf.concat(output_neurons_in, axis=-1),
            tf.concat(output_neurons_out, axis=-1),
            *output_syn,
        )

        return outputs, new_state


class BillehColumn(tf.keras.layers.Layer):
    def __init__(
        self,
        network,
        input_population,
        bkg_weights,
        dt=1.0,
        gauss_std=0.5,
        dampening_factor=0.3,
        input_weight_scale=1.0,
        recurrent_weight_scale=1.0,
        spike_gradient=False,
        max_delay=5,
        train_recurrent=True,
        train_input=True,
        train_bkg=False,
        use_dale_law=True,
        _return_interal_variables=False,
        scale_voltage: bool = False,
        default_input_to_receptor=True,
        spk_reset: str = "soft",
        delay_filter: list[int] | None = None,
    ):
        super().__init__()
        self._params = network["node_params"]

        if scale_voltage:
            voltage_scale = self._params["V_th"] - self._params["E_L"]
            voltage_offset = self._params["E_L"]
            self._params["V_th"] = (
                self._params["V_th"] - voltage_offset
            ) / voltage_scale
            self._params["E_L"] = (self._params["E_L"] - voltage_offset) / voltage_scale
            self._params["V_reset"] = (
                self._params["V_reset"] - voltage_offset
            ) / voltage_scale
            self._params["asc_amps"] = (
                self._params["asc_amps"] / voltage_scale[..., None]
            )
        else:
            voltage_scale = 1.0
            voltage_offset = 0.0

        self._node_type_ids = network["node_type_ids"]
        self._dt = dt

        self._return_interal_variables = _return_interal_variables

        # for random spike, the instantaneous firing rate when v = v_th
        self._spike_gradient = spike_gradient

        n_receptors = network["node_params"]["tau_syn"].shape[1]
        self._n_receptors = n_receptors
        self._n_neurons = int(network["n_nodes"])
        self._dampening_factor = tf.cast(dampening_factor, self.compute_dtype)
        self._gauss_std = tf.cast(gauss_std, self.compute_dtype)

        tau = self._params["C_m"] / self._params["g"]
        self._decay = np.exp(-dt / tau)
        self._current_factor = 1 / self._params["C_m"] * (1 - self._decay) * tau
        self._syn_decay = np.exp(-dt / np.array(self._params["tau_syn"]))
        self._psc_initial = np.e / np.array(self._params["tau_syn"])

        # synapses: target_ids, source_ids, weights, delays

        self.max_delay = int(
            np.round(np.min([np.max(network["synapses"]["delays"]), max_delay]))
        )

        self.state_size = (
            self._n_neurons * self.max_delay,  # z buffer
            self._n_neurons,  # v
            self._n_neurons,  # r
            self._n_neurons,  # asc 1
            self._n_neurons,  # asc 2
            n_receptors * self._n_neurons,  # psc rise
            n_receptors * self._n_neurons,  # psc
        )

        # useless now; it was for training the neuron parameters
        def _f(_v, trainable=False):
            return tf.Variable(
                tf.cast(self._gather(_v), self.compute_dtype), trainable=trainable
            )

        def inv_sigmoid(_x):
            return tf.math.log(_x / (1 - _x))

        # useless
        def custom_val(_v, trainable=False):
            _v = tf.Variable(
                tf.cast(inv_sigmoid(self._gather(_v)), self.compute_dtype),
                trainable=trainable,
            )

            def _g():
                return tf.nn.sigmoid(_v.read_value())

            return _v, _g

        self.v_reset = _f(self._params["V_reset"])
        self.syn_decay = _f(self._syn_decay)
        self.psc_initial = _f(self._psc_initial)
        self.t_ref = _f(self._params["t_ref"])
        self.asc_amps = _f(self._params["asc_amps"], trainable=False)
        self.param_k = _f(self._params["k"], trainable=False)
        # _k = self._params["k"]
        # _k[_k < .0031] = .0007
        # self.param_k, self.param_k_read = custom_val(_k, trainable=False)
        self.v_th = _f(self._params["V_th"])
        self.e_l = _f(self._params["E_L"])
        self.param_g = _f(self._params["g"])
        self.decay = _f(self._decay)
        self.current_factor = _f(self._current_factor)
        if scale_voltage:
            self.voltage_scale = _f(voltage_scale)
            self.voltage_offset = _f(voltage_offset)
        else:
            self.voltage_scale = voltage_scale
            self.voltage_offset = voltage_offset

        self.recurrent_weights = None
        self.disconnect_mask = None

        indices, weights, dense_shape = (
            network["synapses"]["indices"],
            network["synapses"]["weights"],
            network["synapses"]["dense_shape"],
        )
        if scale_voltage:
            weights = (
                weights
                / voltage_scale[self._node_type_ids[indices[:, 0] // self._n_receptors]]
            )
        delays = np.round(
            np.clip(network["synapses"]["delays"], dt, self.max_delay) / dt
        ).astype(np.int32)
        dense_shape = dense_shape[0], self.max_delay * dense_shape[1]
        indices = indices.copy()
        indices[:, 1] = indices[:, 1] + self._n_neurons * (delays - 1)
        weights = weights.astype(np.float32)
        if delay_filter is None:
            delay_filter = np.unique(delays)[0]
        else:
            delay_filter = np.array(delay_filter)
        indices = indices[np.isin(delays - 1, delay_filter), :]
        weights = weights[np.isin(delays - 1, delay_filter)]
        print(f"> Recurrent synapses {len(indices)}")
        input_weights = input_population["weights"].astype(np.float32)
        input_indices = input_population["indices"]
        if scale_voltage:
            input_weights = (
                input_weights
                / voltage_scale[
                    self._node_type_ids[input_indices[:, 0] // self._n_receptors]
                ]
            )
        print(f"> Input synapses {len(input_indices)}")
        input_dense_shape = (
            self._n_receptors * self._n_neurons,
            input_population["n_inputs"],
        )

        self.recurrent_weight_positive = tf.Variable(
            weights >= 0.0, name="recurrent_weights_sign", trainable=False
        )
        self.input_weight_positive = tf.Variable(
            input_weights >= 0.0, name="input_weights_sign", trainable=False
        )
        if use_dale_law:
            self.recurrent_weight_values = tf.Variable(
                weights * recurrent_weight_scale,
                name="sparse_recurrent_weights",
                constraint=SignedConstraint(self.recurrent_weight_positive),
                trainable=train_recurrent,
            )
        else:
            self.recurrent_weight_values = tf.Variable(
                weights * recurrent_weight_scale,
                name="sparse_recurrent_weights",
                constraint=None,
                trainable=train_recurrent,
            )
        self.recurrent_indices = tf.Variable(indices, trainable=False)
        self.recurrent_dense_shape = dense_shape

        if use_dale_law:
            self.input_weight_values = tf.Variable(
                input_weights * input_weight_scale,
                name="sparse_input_weights",
                constraint=SignedConstraint(self.input_weight_positive),
                trainable=train_input,
            )
        else:
            self.input_weight_values = tf.Variable(
                input_weights * input_weight_scale,
                name="sparse_input_weights",
                constraint=None,
                trainable=train_input,
            )

        self.input_indices = tf.Variable(input_indices, trainable=False)
        self.input_dense_shape = input_dense_shape
        if scale_voltage:
            bkg_weights = bkg_weights / np.repeat(
                voltage_scale[self._node_type_ids], self._n_receptors
            )
        # this actutually is not used; we used the decoded noise
        self.bkg_weights = tf.Variable(
            bkg_weights * 10.0, name="rest_of_brain_weights", trainable=train_bkg
        )

        self.default_input_to_receptor = default_input_to_receptor
        assert spk_reset in ("soft", "hard")
        self.spk_reset = spk_reset

    def zero_state(self, batch_size, dtype=tf.float32):
        z0_buf = tf.zeros((batch_size, self._n_neurons * self.max_delay), dtype)
        v0 = tf.ones((batch_size, self._n_neurons), dtype) * tf.cast(
            self.v_th * 0.0 + 1.0 * self.v_reset, dtype
        )
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_10 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_20 = tf.zeros((batch_size, self._n_neurons), dtype)
        psc_rise0 = tf.zeros((batch_size, self._n_neurons * self._n_receptors), dtype)
        psc0 = tf.zeros((batch_size, self._n_neurons * self._n_receptors), dtype)
        return z0_buf, v0, r0, asc_10, asc_20, psc_rise0, psc0

    def random_state(self, batch_size, dtype=tf.float32):
        z0_buf = tf.cast(
            tf.random.uniform(
                (batch_size, self._n_neurons * self.max_delay), 0, 2, tf.int32
            ),
            dtype,
        )
        v0 = tf.random.uniform(
            (batch_size, self._n_neurons),
            tf.cast(self.v_reset, dtype),
            tf.cast(self.v_th, dtype),
            dtype,
        )
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_10 = tf.random.normal(
            (batch_size, self._n_neurons), mean=-0.28, stddev=1.75, dtype=dtype
        )  # min -87 max 59
        asc_20 = tf.random.normal(
            (batch_size, self._n_neurons), mean=-0.28, stddev=1.75, dtype=dtype
        )
        psc_rise0 = tf.random.normal(
            (batch_size, self._n_neurons * self._n_receptors),
            mean=0.29,
            stddev=0.77,
            dtype=dtype,
        )  # -3.8~33.6
        psc0 = tf.random.normal(
            (batch_size, self._n_neurons * self._n_receptors),
            mean=1.17,
            stddev=3.19,
            dtype=dtype,
        )  # -21~147
        return z0_buf, v0, r0, asc_10, asc_20, psc_rise0, psc0

    def _gather(self, prop):
        return tf.gather(prop, self._node_type_ids)

    def call(self, inputs, state):
        batch_size = inputs.shape[0]
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]

        z_buf, v, r, asc_1, asc_2, psc_rise, psc = state

        shaped_z_buf = tf.reshape(z_buf, (-1, self.max_delay, self._n_neurons))
        prev_z = shaped_z_buf[:, 0]

        psc_rise = tf.reshape(
            psc_rise, (batch_size, self._n_neurons, self._n_receptors)
        )
        psc = tf.reshape(psc, (batch_size, self._n_neurons, self._n_receptors))

        sparse_w_rec = tf.sparse.SparseTensor(
            self.recurrent_indices,
            self.recurrent_weight_values,
            self.recurrent_dense_shape,
        )

        i_rec = tf.sparse.sparse_dense_matmul(
            sparse_w_rec, tf.cast(z_buf, tf.float32), adjoint_b=True
        )
        i_rec = tf.transpose(i_rec)

        rec_inputs = tf.cast(i_rec, self.compute_dtype)
        if self.default_input_to_receptor:
            rec_inputs += inputs

        rec_inputs = tf.reshape(
            rec_inputs, (batch_size, self._n_neurons, self._n_receptors)
        )

        new_psc_rise = self.syn_decay * psc_rise + rec_inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise

        new_r = tf.nn.relu(r + prev_z * self.t_ref - self._dt)

        k = self.param_k
        asc_amps = self.asc_amps
        new_asc_1 = tf.exp(-self._dt * k[:, 0]) * asc_1 + prev_z * asc_amps[:, 0]
        new_asc_2 = tf.exp(-self._dt * k[:, 1]) * asc_2 + prev_z * asc_amps[:, 1]

        reset_current = prev_z * (self.v_reset - self.v_th)
        input_current = tf.reduce_sum(new_psc, -1)
        if not self.default_input_to_receptor:
            input_current += inputs

        decayed_v = self.decay * v

        gathered_g = self.param_g * self.e_l
        c1 = input_current + asc_1 + asc_2 + gathered_g
        new_v = decayed_v + self.current_factor * c1
        if self.spk_reset == "soft":
            new_v += reset_current
        else:
            new_v += (self.v_reset - new_v) * prev_z

        normalizer = self.v_th - self.e_l
        v_sc = (new_v - self.v_th) / normalizer

        new_z = spike_gauss(v_sc, self._gauss_std, self._dampening_factor)

        new_z = tf.where(new_r > 0.0, tf.zeros_like(new_z), new_z)

        new_psc = tf.reshape(new_psc, (batch_size, self._n_neurons * self._n_receptors))
        new_psc_rise = tf.reshape(
            new_psc_rise, (batch_size, self._n_neurons * self._n_receptors)
        )

        new_shaped_z_buf = tf.concat((new_z[:, None], shaped_z_buf[:, :-1]), 1)
        new_z_buf = tf.reshape(new_shaped_z_buf, (-1, self._n_neurons * self.max_delay))

        if self._return_interal_variables:
            outputs = (
                tf.concat(
                    (
                        new_z,
                        new_v * self.voltage_scale + self.voltage_offset,
                        new_asc_1,
                        new_asc_2,
                    ),
                    axis=-1,
                ),
                new_psc_rise,
                new_psc,
                tf.repeat(input_current, 4, axis=-1),
            )
        else:
            outputs = (new_z, new_v * self.voltage_scale + self.voltage_offset)
        new_state = (
            new_z_buf,
            new_v,
            new_r,
            new_asc_1,
            new_asc_2,
            new_psc_rise,
            new_psc,
        )

        return outputs, new_state


def huber_quantile_loss(u, tau, kappa):
    branch_1 = tf.abs(tau - tf.cast(u <= 0, tf.float32)) / (2 * kappa) * tf.square(u)
    branch_2 = tf.abs(tau - tf.cast(u <= 0, tf.float32)) * (tf.abs(u) - 0.5 * kappa)
    return tf.where(tf.abs(u) <= kappa, branch_1, branch_2)


def compute_spike_rate_distribution_loss(_spikes, target_rate):
    _rate = tf.reduce_mean(_spikes, (0, 1))
    ind = tf.range(target_rate.shape[0])
    rand_ind = tf.random.shuffle(ind)
    _rate = tf.gather(_rate, rand_ind)
    sorted_rate = tf.sort(_rate)

    u = sorted_rate - target_rate
    tau = (tf.cast(tf.range(target_rate.shape[0]), tf.float32) + 1) / target_rate.shape[
        0
    ]
    loss = huber_quantile_loss(u, tau, 0.002)

    return loss


class SpikeRateDistributionRegularization:
    def __init__(self, target_rates, rate_cost=0.5):
        self._rate_cost = rate_cost
        self._target_rates = target_rates

    def __call__(self, spikes):
        reg_loss = (
            compute_spike_rate_distribution_loss(spikes, self._target_rates)
            * self._rate_cost
        )
        reg_loss = tf.reduce_sum(reg_loss)

        return reg_loss


class VoltageRegularization:
    def __init__(self, cell, voltage_cost=1e-5):
        self._voltage_cost = voltage_cost
        self._cell = cell

    def __call__(self, voltages):
        voltage_32 = (
            tf.cast(voltages, tf.float32) - self._cell.voltage_offset
        ) / self._cell.voltage_scale
        v_pos = tf.square(tf.nn.relu(voltage_32 - 1.0))
        v_neg = tf.square(tf.nn.relu(-voltage_32 + 1.0))
        voltage_loss = (
            tf.reduce_mean(tf.reduce_sum(v_pos + v_neg, -1)) * self._voltage_cost
        )
        return voltage_loss


class SpikeVoltageRegularization(tf.keras.layers.Layer):
    def __init__(self, cell, rate_cost=0.1, voltage_cost=0.01, target_rate=0.02):
        self._rate_cost = rate_cost
        self._voltage_cost = voltage_cost
        self._target_rate = target_rate
        self._cell = cell
        super().__init__()

    def call(self, inputs, **kwargs):
        spike = inputs[0]
        voltage = inputs[1]
        # upper_threshold = self._cell.threshold
        # if 'a_buf' in inputs[2].keys():
        #     upper_threshold += self._cell.beta[:, None, None, :] * inputs[2]['a_buf']

        rate = tf.reduce_mean(tf.cast(spike, tf.float32), axis=(0, 1))
        global_rate = tf.reduce_mean(rate)
        self.add_metric(global_rate, name="rate", aggregation="mean")

        reg_loss = tf.reduce_sum(tf.square(rate - self._target_rate)) * self._rate_cost
        self.add_loss(reg_loss)
        self.add_metric(reg_loss, name="rate_loss", aggregation="mean")

        voltage_32 = tf.cast(voltage, tf.float32)
        v_th_32 = tf.cast(self._cell.v_th, tf.float32)
        v_reset_32 = tf.cast(self._cell.v_reset, tf.float32)
        diff = v_th_32 - v_reset_32
        v_pos = tf.square(tf.clip_by_value(tf.nn.relu(voltage_32 - v_th_32), 0.0, 1.0))
        v_neg = tf.square(
            tf.clip_by_value(tf.nn.relu(-voltage_32 + v_reset_32 - diff), 0.0, 1.0)
        )
        voltage_loss = (
            tf.reduce_mean(tf.reduce_sum(v_pos + v_neg, -1)) * self._voltage_cost
        )
        self.add_loss(voltage_loss)
        self.add_metric(voltage_loss, name="voltage_loss", aggregation="mean")
        return inputs