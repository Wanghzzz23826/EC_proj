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

from jax.nn import one_hot


class SoftRecall:
    """
    Soft recall 的可配置封装，支持按类别加权。
    可作为 fitness_cls 传入 rsrp/rsrp_sigmoid（实例可调用）。
    """

    def __init__(
        self,
        mean_axis=(-1,),
        class_weights=None,
        smoothing_a=5,
        wrong_logit_penalty=0.0,
    ):
        """
        Args:
            mean_axis: 对样本求 mean 的轴（在 one-hot 之前的轴）
            class_weights: None 或 shape (num_classes,) 的数组；给出则按类别加权平均 recall
            smoothing_a: 平滑系数，越大 score 衰减越慢
            wrong_logit_penalty: 惩罚错误类 logit 之和的系数，>0 时使非正确类 logit 尽量低
        """
        self.mean_axis = mean_axis
        self.class_weights = class_weights
        self.smoothing_a = smoothing_a
        self.wrong_logit_penalty = wrong_logit_penalty

    def __call__(self, logits, label_batch):
        """
        Score: 若 logit 与 label 足够接近（logit[label] 最大）则 score=1，否则平滑衰减。
        """
        num_classes = logits.shape[-1]
        label_onehot = one_hot(label_batch, num_classes=num_classes)
        true_logits = jnp.sum(logits * label_onehot, axis=-1)
        greater_logits = logits >= jnp.expand_dims(true_logits, -1)
        count_greater_logits = jnp.sum(greater_logits, axis=-1)

        a = self.smoothing_a
        score = a / (a + count_greater_logits)

        wrong_mask = 1.0 - label_onehot
        mean_wrong_logits = jnp.sum(logits * wrong_mask, axis=-1)/(num_classes-1)
        mean_all_logits = jnp.mean(logits, axis=-1)
        if self.wrong_logit_penalty > 0:
            score = score + self.wrong_logit_penalty * (mean_all_logits - mean_wrong_logits)/mean_all_logits

        sum_axis = tuple(i - 1 for i in self.mean_axis)
        true_positives = jnp.sum(jnp.expand_dims(score, -1) * label_onehot, axis=sum_axis)
        actual_positives = jnp.sum(label_onehot, axis=sum_axis)
        recall = true_positives / actual_positives

        if self.class_weights is None:
            avg_recall = jnp.nanmean(recall, axis=-1)
        else:
            w = jnp.broadcast_to(self.class_weights, recall.shape)
            valid = actual_positives > 0
            numer = jnp.sum(jnp.where(valid, recall * w, 0.0), axis=-1)
            denom = jnp.sum(jnp.where(valid, w, 0.0), axis=-1)
            denom = jnp.maximum(denom, 1e-8)
            avg_recall = numer / denom

        return avg_recall


def softrecall(
    logits,
    label_batch,
    mean_axis=(-1,),
    class_weights=None,
    wrong_logit_penalty=0.0,
):
    """
    函数形式入口，与 SoftRecall 行为一致，兼容旧代码。
    """
    return SoftRecall(
        mean_axis=mean_axis,
        class_weights=class_weights,
        wrong_logit_penalty=wrong_logit_penalty,
    )(logits, label_batch)

def accuracy(logits, label_batch, mean_axis=-1) -> jnp.ndarray:
    predict = jnp.argmax(logits, axis=-1) 
    hit = (predict == label_batch)
    return jnp.mean(hit, axis=mean_axis)


class AccuracyWithMargin:
    """
    Accuracy + margin reward.

    - Keeps 0/1 correctness as the main signal.
    - Adds a continuous bonus/penalty from margin = true_logit - max_wrong_logit,
      so the optimizer also cares about "how much larger".
    """

    def __init__(self, margin_weight=0.2, margin_temp=1.0):
        self.margin_weight = float(margin_weight)
        self.margin_temp = float(margin_temp)

    def __call__(self, logits, label_batch):
        num_classes = logits.shape[-1]
        label_onehot = one_hot(label_batch, num_classes=num_classes)

        true_logits = jnp.sum(logits * label_onehot, axis=-1)
        wrong_logits = jnp.where(label_onehot > 0.5, -jnp.inf, logits)
        max_wrong_logits = jnp.max(wrong_logits, axis=-1)
        margin = true_logits - max_wrong_logits

        predict = jnp.argmax(logits, axis=-1)
        hit = (predict == label_batch).astype(jnp.float32)

        # Bounded margin term in [-1, 1], robust to outliers.
        margin_term = jnp.tanh(margin / self.margin_temp)
        sample_score = hit + self.margin_weight * margin_term
        return jnp.mean(sample_score, axis=-1)

def centered_rank_transform(x: jnp.ndarray) -> jnp.ndarray:
    scale = math.sqrt(12.0)
    shape = x.shape
    x     = x.ravel()

    x = jnp.argsort(jnp.argsort(x))
    x = x / (len(x) - 1) - 0.5
    x = x * scale
    return x.reshape(shape)

def get_es_sigma(
    current_step: int,
    total_steps: int,
    sigma_schedule: str = "exponential",
    sigma_init: float = 0.02,
    sigma_final: float = 0.001,
) -> float:
    """ES sigma schedule (used by antithetic ES for receptor-wise scales)."""
    if sigma_schedule == "constant":
        return sigma_init
    if total_steps <= 1:
        return sigma_init
    decay_rate = (sigma_final / sigma_init) ** (1.0 / total_steps)
    return max(sigma_init * (decay_rate ** current_step), sigma_final)


def sample_gaussian_noise(key: jax.Array, params_dict, sigma: float, batch_size: int):
    """Sample Gaussian noise for ES params with antithetic sampling."""
    keys = jax.random.split(key, len(params_dict))
    return {
        k: jax.random.normal(keys[i], (batch_size,)) * sigma
        for i, k in enumerate(params_dict)
    }


def es_grad(reward: jnp.ndarray, noise: dict, sigma: float) -> dict:
    """ES gradient with antithetic sampling.

    reward is expected to be ordered as [pos, neg] where each part has length n_pairs.
    """
    n_pairs = len(noise[list(noise.keys())[0]])
    f_pos, f_neg = reward[:n_pairs], reward[n_pairs : 2 * n_pairs]
    diff = f_pos - f_neg
    return {k: -jnp.mean(diff * noise[k]) / sigma for k in noise}


def update_es(reward, noise, es_params, es_opt_state, es_opt_cls, sigma):
    """Update ES parameters (generic ES update for real-valued receptor scales)."""
    grads = es_grad(reward, noise, sigma)
    updates, new_opt_state = es_opt_cls.update(grads, es_opt_state, es_params)
    return optax.apply_updates(es_params, updates), new_opt_state

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
        # lr = 2*(_rho*(1-_rho))**0.5
        return -jnp.mean((_theta - _rho) * R, axis=0)
    
    return jax.tree_util.tree_map(lambda _rho, _theta:_nes_grad(_rho,_theta), rho, theta)

def determinstic_param(params: FrozenDict, weights: FrozenDict, mode)-> FrozenDict:
    return jax.tree_util.tree_map(lambda p, w: conn(w, (p>0.5), mode), params, weights)

def rho_mean(params: FrozenDict, init_value: float = 0.5):
    return jax.tree_util.tree_map(lambda p: jnp.mean(jnp.abs(p-init_value)), params)

@partial(jax.jit, static_argnums=(4,))
def rsrp(fitness, theta, params, opt_state, opt_cls, eps):
    grads = nes_grad(fitness, theta, params)
    updates, new_opt_state = opt_cls.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_params = jax.tree_util.tree_map(lambda p: jnp.clip(p, eps, 1 - eps), new_params)
    return new_params, new_opt_state

def nes_grad_sigmoid(fitness: jax.Array, theta: FrozenDict, rho: FrozenDict) -> jax.Array:
    """
    Compute EC grad estimation, grad = -R * (theta - rho)
    """
    def _nes_grad(_rho,_theta):
        R = fitness.reshape((-1,) + (1,) * (_theta.ndim - 1)).astype(_rho.dtype)
        # lr = 0.5/jnp.sqrt(jnp.clip(_rho, 1e-8, 1-1e-8)*(1-jnp.clip(_rho, 1e-8, 1-1e-8)))
        return -jnp.mean((_theta - _rho) * R, axis=0)
    
    return jax.tree_util.tree_map(lambda _rho, _theta:_nes_grad(_rho,_theta), rho, theta)
@partial(jax.jit, static_argnums=(4,))
def rsrp_sigmoid(fitness, theta, params, opt_state, opt_cls):
    params_sigmoid = jax.tree_util.tree_map(lambda p: jax.nn.sigmoid(p), params)
    grads = nes_grad_sigmoid(fitness, theta, params_sigmoid)
    updates, new_opt_state = opt_cls.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state