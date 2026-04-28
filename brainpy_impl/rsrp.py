import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from functools import partial
import math
import optax
from typing import Any, Dict, Tuple

# rsrp tool
def conn(w, p, mode):
    if mode == '-1/1':
        return w * p - w * jnp.logical_not(p)
    elif mode == '0/1':
        return w * p

def accuracy(predict, label_batch, mean_axis=-1) -> jnp.ndarray:
    hit = (predict == label_batch)
    return jnp.mean(hit, axis=mean_axis)


def binary_score_margin(
    response_scores: jax.Array,
    label_batch: jax.Array,
    threshold: float = 0.01,
    mean_axis=(-1, -2),
) -> jnp.ndarray:
    labels = label_batch.astype(response_scores.dtype)
    signed_margin = (2.0 * labels - 1.0) * (response_scores - threshold)
    return jnp.mean(signed_margin, axis=mean_axis)

def centered_rank_transform(x: jnp.ndarray) -> jnp.ndarray:
    scale = math.sqrt(12.0)
    shape = x.shape
    x     = x.ravel()

    x = jnp.argsort(jnp.argsort(x))
    x = x / (len(x) - 1) - 0.5
    x = x * scale
    return x.reshape(shape)

def init_rho(params: FrozenDict, init_prob: float = 0.5) -> FrozenDict:
    return jax.tree_util.tree_map(lambda p:jnp.full_like(p, init_prob), params)

def sample_bernoulli_parameter(key: jax.Array, params: FrozenDict, batch_size: Tuple = ()) -> FrozenDict:
    num_vars = len(jax.tree_util.tree_leaves(params))
    treedef = jax.tree_util.tree_structure(params)
    all_keys = jax.random.split(key, num=num_vars)
    theta = jax.tree_util.tree_map(
        lambda p, k: jax.random.uniform(k, (batch_size,*p.shape)) < p,
        params, jax.tree_util.tree_unflatten(treedef, all_keys))
    return theta

def mask_weights_with_theta(theta: FrozenDict, weights: FrozenDict, mode)-> FrozenDict:
    return jax.tree_util.tree_map(lambda p, w: conn(w, p, mode), theta, weights)


def nes_grad(fitness: jax.Array, theta: FrozenDict, rho: FrozenDict) -> jax.Array:
    """
    Compute EC grad estimation, grad = -R * (theta - rho)
    """
    def _nes_grad(_rho,_theta):
        R = fitness.reshape((-1,) + (1,) * (_theta.ndim - 1)).astype(_rho.dtype)
        # lr = (_rho*(1-_rho))**0.5
        return -jnp.mean((_theta - _rho) * R, axis=0)
    
    return jax.tree_util.tree_map(lambda _rho, _theta:_nes_grad(_rho,_theta), rho, theta)

def determinstic_param(params: FrozenDict, weights: FrozenDict, mode)-> FrozenDict:
    return jax.tree_util.tree_map(lambda p, w: conn(w, (p>0.5), mode), params, weights)

def rho_mean(params: FrozenDict):
    return jax.tree_util.tree_map(lambda p: jnp.mean(jnp.abs(p-0.5)), params)

@partial(jax.jit, static_argnums=(5, 7))
def rsrp(predict, labels, theta, params, opt_state, opt_cls, eps, fitness_cls=accuracy):
    fitness = fitness_cls(predict, labels)
    fitness = centered_rank_transform(fitness)
    grads = nes_grad(fitness, theta, params)
    updates, new_opt_state = opt_cls.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_params = jax.tree_util.tree_map(lambda p: jnp.clip(p, eps, 1 - eps), new_params)
    return new_params, new_opt_state
