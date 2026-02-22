from functools import partial
from typing import Any, Callable, Literal, Optional, Sequence, Union

import brainpy as bp
import brainpy.math as bm
import jax
import numpy as np
from brainpy.check import is_initializer
from brainpy.initialize import OneInit, ZeroInit, variable_
from brainpy.types import ArrayType, Shape, Sharding
from jax.lax import stop_gradient
from jaxtyping import Array, Float, Int, Shaped
#今天早上的版本的暂存


from common.types import (
    InputParams,
    NodeParams,
    OutputParams,
    SynapseParams,
    is_all_same,
    to_brainpy_csr,
)


# modified from bp.dyn.Gif
class Glif3(bp.dyn.GradNeuDyn):
    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sequence[str]] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: Literal["soft", "hard"] = "soft",
        detach_spk: bool = False,
        method: str = "exp_auto",
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,
        # neuron parameters
        v_reset: Union[float, ArrayType, Callable] = -70.0,
        e_l: Union[float, ArrayType, Callable] = -70.0,
        v_th: Union[float, ArrayType, Callable] = -50.0,
        c_m: Union[float, ArrayType, Callable] = 1 / 20.0,
        tau: Union[float, ArrayType, Callable] = 20.0,
        k: Union[float, ArrayType, Callable] = 0.2,
        asc_amps: Union[float, ArrayType, Callable] = 0.0,
        t_ref: Union[float, ArrayType, Callable] = 0.0,
        v_initializer: Union[Callable, ArrayType] = OneInit(-70.0),
        Iasc_initializer: [Union[Callable, ArrayType]] = ZeroInit(),
    ):
        # initialization
        super().__init__(
            size=size,
            name=name,
            keep_size=keep_size,
            mode=mode,
            sharding=sharding,
            spk_fun=spk_fun,
            detach_spk=detach_spk,
            method=method,
            spk_dtype=spk_dtype,
            spk_reset=spk_reset,
            scaling=scaling,
        )
        # parameters
        self.v_reset = self.offset_scaling(self.init_param(v_reset))
        self.e_l = self.offset_scaling(self.init_param(e_l))
        self.v_th = self.offset_scaling(self.init_param(v_th))
        self.c_m = self.init_param(c_m)
        self.tau = self.init_param(tau)
        assert k.shape[-1] == 2
        self.k = self.init_param(k, shape=k.shape)
        self.asc_amps = self.init_param(asc_amps, shape=asc_amps.shape)

        # initializers
        self._v_initializer = is_initializer(v_initializer)
        self._Iasc_initializer = is_initializer(Iasc_initializer)

        # integral
        self.integral = bp.odeint(method=method, f=self.derivative)

        self.t_ref = self.init_param(t_ref)

        # variables
        if init_var:
            self.reset_state(self.mode)

    @classmethod
    def from_node_params(cls, node_params: NodeParams, **xargs):
        return cls(
            **xargs,
            size=node_params.n_nodes,
            v_reset=node_params.v_reset,
            e_l=node_params.e_l,
            v_th=node_params.v_th,
            c_m=node_params.c_m,
            tau=node_params.tau,
            k=node_params.k,
            asc_amps=node_params.asc_amps,
            t_ref=node_params.t_ref,
            v_initializer=bm.asarray(node_params.v_reset),
        )

    # TODO add exponent euler impl for system var in brainpy
    def dIasc1(self, Iasc1, t):
        return -self.k[:, 0] * Iasc1

    def dIasc2(self, Iasc2, t):
        return -self.k[:, 1] * Iasc2

    def dv(self, V, t, Iasc1, Iasc2):
        Isum = self.Iinp.value
        return -(V - self.e_l) / self.tau + (Isum + Iasc1 + Iasc2) / self.c_m

    # unused, may be useful in case want to enforce specific update order of v and Iasc
    def dvdIasc(self, V, Iasc1, Iasc2, t):
        Isum = self.Iinp.value
        dIasc1 = -self.k[:, 0] * Iasc1
        dIasc2 = -self.k[:, 1] * Iasc2
        dv = -(V - self.e_l) / self.tau + (Isum + Iasc1 + Iasc2) / self.c_m
        return dv, dIasc1, dIasc2

    @property
    def derivative(self):
        return bp.JointEq(self.dIasc1, self.dIasc2, self.dv)

    def reset_state(self, batch_size=None, **kwargs):
        self.Iinp = self.init_variable(bm.zeros, batch_size)
        self.V = self.offset_scaling(
            self.init_variable(self._v_initializer, batch_size)
        )
        self.Iasc1 = self.std_scaling(
            self.init_variable(self._Iasc_initializer, batch_size)
        )
        self.Iasc2 = self.std_scaling(
            self.init_variable(self._Iasc_initializer, batch_size)
        )
        self.spike = self.init_variable(
            partial(bm.zeros, dtype=self.spk_dtype), batch_size
        )
        self.r = self.init_variable(bm.zeros, batch_size)

    def update(self, x=None):
        t = bp.share.load("t")
        dt = bp.share.load("dt")
        x = 0.0 if x is None else x

        # integrate membrane potential
        self.Iinp.value = self.sum_current_inputs(self.V.value, init=x)
        Iasc1, Iasc2, V = self.integral(self.Iasc1, self.Iasc2, self.V.value, t, dt)
        V += self.sum_delta_inputs()

        # refractory
        r = bm.relu(self.r + self.spike * self.t_ref - dt)
        if isinstance(self.mode, bm.TrainingMode):
            r = stop_gradient(r)

        # spike, refractory, spiking time, and membrane potential reset
        if isinstance(self.mode, bm.TrainingMode):
            spike_no_grad = (
                stop_gradient(self.spike.value) if self.detach_spk else self.spike.value
            )
            if self.spk_reset == "soft":
                V -= (self.v_th - self.v_reset) * spike_no_grad
            else:
                V += (self.v_reset - V) * spike_no_grad
            Iasc1 += spike_no_grad * self.asc_amps[:, 0]
            Iasc2 += spike_no_grad * self.asc_amps[:, 1]
            spike = self.spk_fun(V - self.v_th)

        else:
            if self.spk_reset == "soft":
                V -= (self.v_th - self.v_reset) * self.spike
            else:
                V += (self.v_reset - V) * self.spike
            Iasc1 = bm.where(self.spike, Iasc1 + self.asc_amps[:, 0], Iasc1)
            Iasc2 = bm.where(self.spike, Iasc2 + self.asc_amps[:, 1], Iasc2)
            spike = V >= self.v_th

        spike = spike * bm.logical_not(r > 0.0)

        self.V.value = V
        self.Iasc1.value = Iasc1
        self.Iasc2.value = Iasc2
        self.spike.value = spike
        self.r.value = r
        return spike

    def return_info(self):
        return self.spike


class GLIF3_ODE(Glif3):
    """
    ODE-based GLIF3 model (continuous-time with event detection).
    Kept for reference/debug. For TF-aligned discrete update, use GLIF3_TFOrder.
    """

    def __init__(
        self,
        *args,
        input_var: bool = True,
        spike_fun: Callable = None,
        **kwargs,
    ):
        self.input_var = input_var
        if spike_fun is not None:
            kwargs["spk_fun"] = spike_fun
        super().__init__(*args, **kwargs, init_var=False)
        self.reset_state(self.mode)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        if self.input_var:
            self.input = variable_(bm.zeros, self.varshape, batch_size)

    def update(self, x=None):
        if self.input_var:
            if x is not None:
                self.input += x
            x = self.input.value
        else:
            x = 0.0 if x is None else x
        return super().update(x)

    def clear_input(self):
        if self.input_var:
            self.input.value = bm.zeros_like(self.input)


# does BrainPy have an easier way to define constraint on variable?

class GLIF3_TFOrder(Glif3):
    """
    GLIF3 neuron with TensorFlow update ordering.
    This bypasses ODE integrator and uses the discrete update steps
    consistent with tensorflow_impl.models.BillehColumn.
    """

    def __init__(
        self,
        *args,
        input_var: bool = True,
        spike_fun: Callable = None,
        **kwargs,
    ):
        self.input_var = input_var
        if spike_fun is not None:
            kwargs["spk_fun"] = spike_fun
        # use init_var from parent to create variables, but we'll override update
        super().__init__(*args, **kwargs, init_var=False)
        self.reset_state(self.mode)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        if self.input_var:
            self.input = variable_(bm.zeros, self.varshape, batch_size)

    def update(self, x=None):
        # TF order: decay -> add current -> reset -> spike
        t = bp.share.load("t")
        dt = bp.share.load("dt")

        if self.input_var:
            if x is not None:
                self.input += x
            x = self.input.value
        else:
            x = 0.0 if x is None else x

        prev_z = self.spike.value

        # refractory
        r = bm.relu(self.r + prev_z * self.t_ref - dt)
        if isinstance(self.mode, bm.TrainingMode):
            r = stop_gradient(r)

        # after-spike currents (discrete)
        # TF uses old asc in membrane update, then advances asc
        k = self.k
        asc_amps = self.asc_amps
        Iasc1_old = self.Iasc1.value
        Iasc2_old = self.Iasc2.value
        Iasc1_new = bm.exp(-dt * k[:, 0]) * Iasc1_old + prev_z * asc_amps[:, 0]
        Iasc2_new = bm.exp(-dt * k[:, 1]) * Iasc2_old + prev_z * asc_amps[:, 1]

        # input current and membrane update
        self.Iinp.value = self.sum_current_inputs(self.V.value, init=x)
        input_current = self.Iinp.value
        decayed_v = bm.exp(-dt / self.tau) * self.V.value
        current_factor = (1.0 - bm.exp(-dt / self.tau)) * (self.tau / self.c_m)
        # g * E_L where g = C_m / tau
        g_e_l = (self.c_m / self.tau) * self.e_l
        new_v = decayed_v + current_factor * (input_current + Iasc1_old + Iasc2_old + g_e_l)

        # reset
        if self.spk_reset == "soft":
            new_v = new_v + (self.v_reset - self.v_th) * prev_z
        else:
            new_v = new_v + (self.v_reset - new_v) * prev_z

        # spike
        v_sc = (new_v - self.v_th) / (self.v_th - self.e_l)
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(v_sc)
            if self.detach_spk:
                spike = stop_gradient(spike)
        else:
            spike = v_sc > 0.0

        spike = spike * bm.logical_not(r > 0.0)

        # commit state
        self.V.value = new_v
        self.Iasc1.value = Iasc1_new
        self.Iasc2.value = Iasc2_new
        self.spike.value = spike
        self.r.value = r

        return spike

    def clear_input(self):
        if self.input_var:
            self.input.value = bm.zeros_like(self.input)


# Default GLIF3 for this project: TF-aligned discrete update.
GLIF3 = GLIF3_TFOrder


class EventCSRLinearDale(bp.dnn.CSRLinear):
    """
    EventCSRLinear that enforces Dale's law on weights.
    No weight can change sign during training
    """

    def __init__(
        self,
        conn: bp.conn.TwoEndConnector,
        weight: Union[float, ArrayType, Callable],
        use_dale_law: bool = True,
        sharding: Optional[Sharding] = None,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        transpose: bool = True,
    ):
        super().__init__(
            name=name,
            mode=mode,
            conn=conn,
            weight=weight,
            sharding=sharding,
            transpose=transpose,
        )
        self.use_dale_law = use_dale_law
        self.sign = bp.init.parameter(
            weight > 0, (self.indices.size,), sharding=sharding
        )

    @property
    def weight_dale(self):
        if self.use_dale_law:
            return bm.where(self.sign, bm.relu(self.weight), -bm.relu(-self.weight))
        else:
            return self.weight

    def stdp_update(
        self,
        on_pre: dict | None = None,
        on_post: dict | None = None,
        w_min: float | None = None,
        w_max: float | None = None,
    ):
        super().stdp_update(on_pre, on_post, w_min, w_max)
        self.weight.value = self.weight_dale

    def update(self, x):
        if x.ndim == 1:
            return bm.event.csrmv(
                self.weight_dale,
                self.indices,
                self.indptr,
                x,
                shape=(self.conn.pre_num, self.conn.post_num),
                transpose=self.transpose,
            )
        elif x.ndim > 1:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = jax.vmap(self._batch_csrmv)(x)
            return bm.reshape(y, shapes + (y.shape[-1],))
        else:
            raise ValueError

    def _batch_csrmv(self, x):
        return bm.event.csrmv(
            self.weight_dale,
            self.indices,
            self.indptr,
            x,
            shape=(self.conn.pre_num, self.conn.post_num),
            transpose=self.transpose,
        )


class CUBAMultiReceptor(bp.dyn.CUBA):
    """
    CUBA that sums up a fixed number of receptor currents for each neuron
    assume inputs are flattened array of neuron and receptor, like (n_neurons, n_receptors).flatten()
    """

    def __init__(
        self,
        n_receptors: int = 1,
        name: Optional[str] = None,
        scaling: Optional[bm.Scaling] = None,
    ):
        super().__init__(name=name, scaling=scaling)
        self.n_receptors = n_receptors


    def update(self, conductance, potential=None):
        conductance = super().update(conductance, potential)

        return bm.ein_reduce(
            conductance,
            "... (neuron receptor) -> ... neuron",
            receptor=self.n_receptors,
            reduction="sum",
        )


class ConvTranspose1dComm(bp.dnn.Layer):
    """
    A wrapper over jax.lax.conv_transpose.
    Similar to bp.dnn.ConvTranpose1D(in_channels=1, out_channels=1, *args), but only applied to neuron dimension
    TODO: support circular convolution
    """

    num_spatial_dims = 1

    def __init__(
        self,
        kernel_size: int,
        stride: int = 1,
        kernel_initializer: Union[Callable, ArrayType, bp.init.Initializer] = OneInit(),
        padding: Union[str, int, tuple[int, int]] = "SAME",
        kernel_dilation: int = 1,
        sharding: Optional[Sharding] = None,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
    ):
        super().__init__(mode=mode, name=name)

        self.stride = stride
        if isinstance(padding, str):
            assert padding in ["SAME", "VALID"]
        elif isinstance(padding, int):
            padding = tuple((padding, padding) for _ in range(self.num_spatial_dims))
        elif isinstance(padding, tuple[int, int]):
            padding = (padding,)
        else:
            raise ValueError
        self.padding = padding
        self.kernel_size = kernel_size
        self.kernel_dilation = kernel_dilation
        self.sharding = sharding

        self.kernel_initializer = kernel_initializer
        kernel = bp.init.parameter(
            self.kernel_initializer, (self.kernel_size,), sharding=sharding
        )
        if isinstance(self.mode, bm.TrainingMode):
            kernel = bm.TrainVar(kernel)
        self.kernel = kernel

    def _check_input_dim(self, x, batching: bool):
        if x.ndim != self.num_spatial_dims + int(batching):
            raise ValueError(
                f"expected {self.num_spatial_dims + int(batching)}D input (got {x.ndim}D input)"
            )

    def update(self, pre_val):
        batching = isinstance(self.mode, bm.BatchingMode)
        self._check_input_dim(pre_val, batching)
        if not batching and pre_val.ndim == self.num_spatial_dims:
            pre_val = bm.unsqueeze(pre_val, 0)

        # a dummy channel dimension
        pre_val = bm.unsqueeze(pre_val, -1)
        y = jax.lax.conv_transpose(
            lhs=bm.as_jax(pre_val.astype(self.kernel.dtype)),
            rhs=bm.as_jax(self.kernel[:, None, None]),
            strides=(self.stride,),
            padding=self.padding,
            rhs_dilation=(self.kernel_dilation,),
            # N: batch, H: num_units, C: 1
            dimension_numbers=("NHC", "HIO", "NHC"),
        )[..., 0]
        return y if batching else y[0]


class Alpha(bp.dyn.SynDyn, bp.mixin.AlignPost):
    """
    Difference from bp.dyn.Alpha:
        impl AlignPost, so can be placed after comm, see FullProjAlignPost
        correct discrete ODE that follows Alpha synapse model in Billeh's paper
    """

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        keep_size: bool = False,
        sharding: Optional[Sequence[str]] = None,
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
        # synapse parameters
        tau_decay: Union[float, ArrayType, Callable] = 10.0,
    ):
        super().__init__(
            name=name, mode=mode, size=size, keep_size=keep_size, sharding=sharding
        )

        # parameters
        self.tau_decay = self.init_param(tau_decay)
        self.psc_initial = self.init_param(np.e / tau_decay)

        self.reset_state(self.mode)

    def reset_state(self, batch_or_mode=None, **kwargs):
        self.h = self.init_variable(bm.zeros, batch_or_mode)
        self.g = self.init_variable(bm.zeros, batch_or_mode)

    def update(self, x):
        dt = bp.share.load("dt")
        # update synaptic variables
        syn_decay = bm.exp(-dt / self.tau_decay)
        h = syn_decay * self.h.value
        g = syn_decay * self.g.value + dt * syn_decay * self.h.value
        self.h.value = h
        self.g.value = g

        if x is not None:
            self.add_current(x)
        return self.g.value

    def add_current(self, inp):
        self.h += inp * self.psc_initial


class DummySyn(bp.dyn.SynDyn, bp.mixin.AlignPost):
    """
    For debug purposes. Should just pass values
    """

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        **xargs,
    ):
        super().__init__(
            size=size,
            **xargs,
        )
        self.reset_state(self.mode)

    def reset_state(self, batch_or_mode=None, **kwargs):
        # for add_current
        self.g = self.init_variable(bm.zeros, batch_or_mode)

    def update(self, x):
        if x is not None:
            self.add_current(x)
        g = self.g.value
        self.g.value = bm.zeros(self.g.shape)
        return g

    def add_current(self, inp):
        self.g += inp


class AlphaCUBA(bp.Projection):
    def __init__(
        self,
        pre,
        post,
        conn,
        weight,
        use_dale_law: bool,
        delay: float,
        tau,
        n_receptors,
        proj=bp.dyn.FullProjAlignPost,
    ):
        super().__init__()
        self.proj = proj(
            pre=pre,
            delay=delay,
            comm=EventCSRLinearDale(conn, weight, use_dale_law=use_dale_law),
            syn=Alpha(post.num * n_receptors, tau_decay=tau),
            out=CUBAMultiReceptor(n_receptors=n_receptors),
            post=post,
        )

    # TODO: is flattened synapse receptors compatible with STDP_Song2000?
    def from_synapse_params(
        syn_params: SynapseParams | list[SynapseParams],
        tau_syn: Float[Array, "n_neurons n_receptors"],  # noqa: F722
        use_dale_law: bool,
        **xargs,
    ) -> list["AlphaCUBA"]:
        """
        split by delays to create AlphaCUBA from SynapseParams
        """
        return classes_from_synapse_params(
            AlphaCUBA,
            syn_params=syn_params,
            # synapse conn is 2d, so flatten neuron and receptor params
            tau=tau_syn.flatten(),
            n_receptors=tau_syn.shape[-1],
            use_dale_law=use_dale_law,
            **xargs,
        )


def classes_from_synapse_params(
    cls,
    syn_params: SynapseParams | list[SynapseParams],
    delay_filter: Optional[list[int]] = None,
    **xargs,
) -> list:
    """
    split by delays to create cls from SynapseParams

    ugly, I know
    """
    delays_is_scalar = False
    if isinstance(syn_params, list):
        syn_params_list = syn_params
    else:
        if not is_all_same(syn_params.delays):
            syn_params_list = syn_params.split_by_delays()
            delays_is_scalar = True
        else:
            syn_params_list = [syn_params]

    ret = []
    for p in syn_params_list:
        conn_csr, weight = to_brainpy_csr(p, split_receptor=False)
        conn = bp.conn.SparseMatConn(conn_csr)
        delay_steps = p.delays if delays_is_scalar else p.delays[0]
        if delay_filter is None or delay_steps in delay_filter:
            # NOTE: BrainPy's internal delay handling is effectively 1 step earlier.
            # Shift by +1 step here to align with TensorFlow delay indexing.
            delay_steps = int(delay_steps) + 1
            ret.append(
                cls(
                    conn=conn,
                    weight=weight,
                    # bp.Projection uses continuous delay instead of discrete step
                    delay=delay_steps * p.dt,
                    **xargs,
                )
            )

    return ret


class SynOutSequence(bp.DynSysGroup):
    """
    SynDyn and SynOut(BindCondData) cannot be chained simply in a Sequence().
    This is just a wrapper to call them eagerly, instead of on-demand via NeuDyn.add_input().
    Mainly for debug convenience. More common usage of SynDyn and SynOut is in a bp.dyn.Projection.

    Tips: to recreate same receptor model, use ParamDescriptors, see BillehColumn.neuron_receptors_desc
    """

    def __init__(
        self,
        syn: bp.dyn.SynDyn,
        out: bp.dyn.SynOut,
    ):
        super(SynOutSequence, self).__init__()
        self.syn = syn
        self.out = out

    def update(self, x):
        g = self.syn(x)
        self.out.bind_cond(g)
        return self.out()


class BillehColumn(bp.DynSysGroup):
    def __init__(
        self,
        node_params: NodeParams,
        syn_params: SynapseParams,
        use_dale_law=True,
        default_input_to_receptor=True,
        batch_size=None,
        spk_reset="soft",
        delay_filter: Optional[list[int]] = None,
        laminar_indices=None,
        track_epsc_ipsc: bool = False,
    ):
        # super(BillehColumn, self).__init__()
        super().__init__()
        self._track_epsc_ipsc = track_epsc_ipsc
        self.laminar_indices = laminar_indices or {}
        if self._track_epsc_ipsc:
            if "L23e" not in self.laminar_indices:
                raise ValueError("laminar_indices must contain 'L23e' when track_epsc_ipsc=True")
            n_exc = len(self.laminar_indices["L23e"])
        else:
            n_exc = 0
        # self.epsc_var =  self.init_variable(bm.zeros,(0, n_exc), batch_size)
        # self.ipsc_var =  self.init_variable(bm.zeros,(0, n_exc), batch_size)

        self.epsc_var = variable_(bm.zeros, (n_exc,), batch_size)
        self.ipsc_var = variable_(bm.zeros, (n_exc,), batch_size)


        # Use the default GLIF3 alias (TF-order discrete update).
        self.neurons = GLIF3.from_node_params(
            node_params, Iasc_initializer=ZeroInit(), spk_reset=spk_reset
        )
        if delay_filter is None:
            # align with tensorflow_impl: keep only the minimum-delay group
            if hasattr(syn_params, "delays"):
                if np.isscalar(syn_params.delays):
                    delay_filter = [int(syn_params.delays)]
                else:
                    delay_filter = [int(np.min(syn_params.delays))]
        self.projs = AlphaCUBA.from_synapse_params(
            syn_params,
            tau_syn=node_params.tau_syn,
            pre=self.neurons,
            post=self.neurons,
            use_dale_law=use_dale_law,
            delay_filter=delay_filter,
        )
        self.default_input_to_receptor = default_input_to_receptor
        self._define_neuron_receptors(node_params)

        self.reset_state(batch_size)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        # maybe use InputVar
        self.receptor_input = variable_(bm.zeros, self._n_receptor_nodes, batch_size)
        
        # keep only the current-step value; DSRunner will stack over time
        self.epsc_var = variable_(bm.zeros, (self.epsc_var.value.shape[-1],), batch_size)
        self.ipsc_var = variable_(bm.zeros, (self.ipsc_var.value.shape[-1],), batch_size)

    def _define_neuron_receptors(self, node_params: NodeParams):
        # bit duplicated code with AlphaCUBA
        n_receptors = node_params.n_receptors
        self._n_receptor_nodes = node_params.n_nodes * n_receptors
        syn_desc = Alpha.desc(
            self._n_receptor_nodes, tau_decay=node_params.tau_syn.flatten()
        )
        out_desc = CUBAMultiReceptor.desc(n_receptors=n_receptors)

        # in case InputLayer or other Projection wants to use syn and out, store them as desc
        self.neuron_receptors_desc = {
            "syn": syn_desc,
            "out": out_desc,
        }

        # for receptor_input
        self.neuron_receptors = SynOutSequence(syn_desc(), out_desc())

    def update(self, x=None):
        for proj in self.projs:
            proj()
            # postsynaptic excitatory neurons

        if self._track_epsc_ipsc:
            epsc_total = 0.0
            ipsc_total = 0.0
            exc_idx = self.laminar_indices["L23e"]
            # Alpha.g exact
            for proj in self.projs:
                g = proj.proj.syn.g.value          # (n_neurons * 4,)
                g = g.reshape(self.neurons.num, 4) # (neuron, receptor)

                epsc = g[exc_idx, 0] + g[exc_idx, 1]  # AMPA + NMDA
                ipsc = g[exc_idx, 2] + g[exc_idx, 3]  # GABA_A + GABA_B

                epsc_total += epsc
                ipsc_total += ipsc

            self.epsc_var.value = epsc_total
            self.ipsc_var.value = ipsc_total
 
        if x is None:
            x = 0.0
        if self.default_input_to_receptor:
            self.receptor_input += x
            # perhaps self.projs[i].syn.add_current(self.receptor_input) also works
            x = self.neuron_receptors(self.receptor_input)
        self.neurons(x)

    def clear_input(self):
        self.receptor_input.value = bm.zeros_like(self.receptor_input)


class InputLayer(bp.DynSysGroup):
    """
    SparseLayer originally
    intended to map input from lgn and add simulated noise from rest of brain

    TODO: rewrite with projection and verify
    """

    def __init__(
        self,
        conn: bp.conn.CSRConn,
        weight,
        tau_syn: Float[Array, "n_nodes n_receptors"],  # noqa: F722
        use_dale_law: bool,
        bkg_weights: Float[Array, "n_nodes n_receptors"],  # noqa: F722, F821
        use_decoded_noise=False,
        noise_data=None,
        neuron_receptors_desc=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        n_nodes = tau_syn.shape[0]
        n_receptors = tau_syn.shape[-1]
        self._n_node_receptors = n_nodes * n_receptors
        self.input_proj = EventCSRLinearDale(conn, weight, use_dale_law)
        self.bkg_weights = bp.init.parameter(bkg_weights.flatten())
        self._use_decoded_noise = use_decoded_noise
        if use_decoded_noise:
            self.noise_data = bp.init.parameter(noise_data)

        if neuron_receptors_desc is None:
            # synapse conn is 2d, so flatten neuron and receptor params
            self.syn = Alpha(n_nodes * n_receptors, tau_decay=tau_syn.flatten())
            self.cuba = CUBAMultiReceptor(n_receptors)
        else:
            self.syn = neuron_receptors_desc["syn"]()
            self.cuba = neuron_receptors_desc["cuba"]()

    @classmethod
    def from_input_params(
        cls, input_params: InputParams, tau_syn, post, neuron_receptors_desc=None
    ):
        conn_csr, weight = to_brainpy_csr(input_params, split_receptor=False)
        conn = bp.conn.SparseMatConn(conn_csr)
        return cls(
            conn,
            weight,
            tau_syn,
            post,
            use_dale_law=True,
            neuron_receptors_desc=neuron_receptors_desc,
        )

    def gen_noise(self, inp):
        if self._use_decoded_noise:
            # TODO:
            raise ValueError
        else:
            # 10 neurons firing with p = 0.1
            rest_of_brain = bm.random.binomial(10, 0.1, size=(inp.shape[:-1])) / 10.0
            noise_input = self.bkg_weights * rest_of_brain
        return noise_input

    def update(self, inp):
        input_current = self.input_proj(inp)
        noise_input = self.gen_noise(inp)
        input_current += noise_input
        input_current = self.cuba(self.syn(input_current))

        return input_current


def gather_neuron(
    v: Shaped[Array, "*batch time n_neurons"],  # noqa: F722, F821
    indices: Int[Array, "n_indices"],  # noqa: F821
) -> Shaped[Array, "*batch time n_indices"]:  # noqa: F722, F821
    return bm.take_along_axis(v, indices, axis=-1)


class ScaleUp(bp.layers.Layer):
    def __init__(self, scale: float, mode: Optional[bm.Mode] = None):
        super(ScaleUp, self).__init__(mode=mode)
        scale = bp.init.parameter(scale, (1,))
        if isinstance(self.mode, bm.TrainingMode):
            scale = bm.TrainVar(scale)
        self.scale = scale

    def update(self, x):
        return x * (1 + bm.softplus(self.scale))


class OutputLayer(bp.DynamicalSystem):
    """
    output processing common for all tasks
    readout_func: task-specific processing

    TODO: incomplete
    """

    def __init__(
        self,
        n_nodes: int,
        n_total_readout: int,
        neuron_output: bool,
        readout_func: Callable,
        lRout_pop: str,
        down_sample: int,
        seq_len: int,
        n_output: int,
        laminar_indices: Int[Array, "_"],  # noqa: F821
    ):
        super().__init__()
        self.neuron_output = neuron_output
        self.readout_func = readout_func
        self.lRout_pop = lRout_pop
        self.down_sample = down_sample
        self.seq_len = seq_len
        self.n_output = n_output
        if self.neuron_output:
            self.scaleup = ScaleUp(0.1)
        else:
            self.output_head = bp.dnn.Dense(n_nodes, n_total_readout)

    def update(self, spikes):
        if self.neuron_output:
            output = self.scaleup(self.readout_func(spikes))
        else:
            if self.lRout_pop != "all":
                out_pop_spikes = gather_neuron(
                    spikes, self.network["laminar_indices"][self.lRout_pop]
                )
            else:
                out_pop_spikes = spikes

            output_all = self.output_head(out_pop_spikes)
            output = self.readout_func(output_all)

        mean_output = bm.ein_reduce(
            output,
            "... (chunk down_sample) output -> ... chunk output",
            down_sample=self.down_sample,
            reduction="mean",
        )
        # bp.losses.cross_entropy_sparse uses logits
        # mean_output = bm.nn.softmax(mean_output, axis=-1)

        outputs = mean_output

        return outputs


def huber_quantile_loss(u, tau, kappa):
    branch_1 = bm.abs(tau - u <= 0) / (2 * kappa) * u**2
    branch_2 = bm.abs(tau - u <= 0) * (bm.abs(u) - 0.5 * kappa)
    return bm.where(bm.abs(u) <= kappa, branch_1, branch_2)


def compute_spike_rate_distribution_loss(
    spikes,
    target_rate: Float[Array, "time rate"],  # noqa: F722
):
    rate = bm.mean(spikes, axis=(0, 1))
    ind = bm.arange(target_rate.shape[0])
    rand_ind = bm.random.permutation(ind)
    rate = rate[rand_ind]
    sorted_rate = bm.sort(rate)

    u = sorted_rate - target_rate
    tau = (bm.arange(target_rate.shape[0]) + 1.0) / target_rate.shape[0]
    loss = huber_quantile_loss(u, tau, 0.002)

    return loss


def voltage_loss(v, voltage_offset, voltage_scale):
    voltage_32 = (v - voltage_offset) / voltage_scale
    v_pos = bm.relu(voltage_32 - 1.0) ** 2
    v_neg = bm.relu(-voltage_32 + 1.0) ** 2
    voltage_loss = bm.sum(v_pos + v_neg, axis=-1)
    return voltage_loss


class StiffRegularizer(bp.layers.Layer):
    def __init__(self, initial_value):
        super().__init__()
        self.initial_value = initial_value

    def update(self, x):
        return bm.reduce_sum(bm.square(x - self.initial_value))


def cross_entropy_sparse_with_mask_weight(y, target, weight):
    if y.ndim == target.ndim + 1:
        target = target.unsqueeze(-1)
    # center of mass by weight
    # if weight is 0,1 matrix, this reduces to masked average
    return bm.sum(bm.losses.cross_entropy_sparse(y, target) * weight, axis=-1) / bm.sum(
        weight, axis=-1
    )


class CSRLinearValue(bp.dnn.CSRLinear):
    def update(self, x):
        if x.ndim == 1:
            return bm.sparse.csrmv(
                self.weight, self.indices, self.indptr, x,
                shape=(self.conn.pre_num, self.conn.post_num),
                transpose=self.transpose,
            )
        elif x.ndim > 1:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = jax.vmap(self._batch_csrmv_val)(x)
            return bm.reshape(y, shapes + (y.shape[-1],))
        else:
            raise ValueError

    def _batch_csrmv_val(self, x):
        return bm.sparse.csrmv(
            self.weight, self.indices, self.indptr, x,
            shape=(self.conn.pre_num, self.conn.post_num),
            transpose=self.transpose,
        )
