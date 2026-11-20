import os
import sys
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from brax import envs

sys.path.insert(0, "/home/phani.gande.ug24/scaling-crl")
from envs.ant_maze import AntMaze
from train_dual_node_breakthrough import ContinuousDepthActor
from evaluator import CrlEvaluator

print("=== STARTING STOCHASTIC POLICY RE-EVALUATION ===", flush=True)

raw_eval_env = AntMaze(
    backend="spring",
    exclude_current_positions_from_observation=False,
    terminate_when_unhealthy=False,
    maze_layout_name="u4_maze"
)

num_eval_envs = 128
episode_length = 1000

actor = ContinuousDepthActor(
    action_size=raw_eval_env.action_size,
    norm_type="layer_norm",
    network_width=512,
    network_depth=4,
    skip_connections=4,
    use_relu=0,
    arch_type="node",
    ode_solver="rk4",
    ode_steps=16,
    ode_num_layers=3,
    ode_t0=0.0,
    ode_t1=1.0,
    ode_rtol=0.001,
    ode_atol=0.0001,
)

def actor_step(actor_params, env, env_state, key, extra_fields=()):
    means, log_stds = actor.apply(actor_params, env_state.obs)
    stds = jnp.exp(log_stds)
    actions = jax.random.normal(key, shape=means.shape, dtype=means.dtype) * stds + means
    actions = nn.tanh(actions)
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, {
        "observation": env_state.obs,
        "action": actions,
        "reward": nstate.reward,
        "discount": 1 - nstate.done,
        "extras": {"state_extras": state_extras},
    }

eval_env = envs.training.wrap(raw_eval_env, episode_length=episode_length)
eval_env_wrapped = envs.training.EvalWrapper(eval_env)

@jax.jit
def run_eval_batch(actor_params, key):
    reset_keys = jax.random.split(key, num_eval_envs)
    first_state = eval_env_wrapped.reset(reset_keys)
    
    def f(carry, unused):
        state, k = carry
        k, step_k = jax.random.split(k)
        nstate, _ = actor_step(actor_params, eval_env_wrapped, state, step_k)
        return (nstate, k), ()

    (final_state, _), _ = jax.lax.scan(f, (first_state, key), (), length=episode_length)
    metrics = final_state.info["eval_metrics"]
    return {
        "success": jnp.mean(metrics.episode_metrics["success"]),
        "success_any": jnp.mean(metrics.episode_metrics["success"] > 0.0),
        "dist": jnp.mean(metrics.episode_metrics["dist"]),
    }

run_dir = "/home/phani.gande.ug24/scaling-crl/runs/ant_u4_maze_breakthrough_node_w512_s16_20261117-023418"
ckpts = [
    ("30M", f"{run_dir}/step_30061568.pkl"),
    ("50M", f"{run_dir}/step_49742848.pkl"),
    ("70M", f"{run_dir}/step_69424128.pkl"),
    ("90M", f"{run_dir}/step_89105408.pkl"),
    ("100M", f"{run_dir}/step_98946048.pkl"),
    ("120M", f"{run_dir}/step_118627328.pkl"),
    ("Latest (125M)", f"{run_dir}/latest_ckpt.pkl"),
]

print(f"{'Checkpoint':<15} | {'TimeAtGoal':<12} | {'SuccessAny':<12} | {'AvgDist':<10}", flush=True)
print("-" * 55, flush=True)

rng = jax.random.PRNGKey(12345)
for label, path in ckpts:
    if not os.path.exists(path):
        continue
    with open(path, "rb") as f:
        data = pickle.load(f)
    params = data["actor_params"]
    rng, eval_k = jax.random.split(rng)
    res = run_eval_batch(params, eval_k)
    tag = float(res["success"])
    s_any = float(res["success_any"])
    dist = float(res["dist"])
    print(f"{label:<15} | {tag:<12.2f} | {s_any:<12.3f} | {dist:<10.1f}", flush=True)

print("=== RE-EVALUATION COMPLETE ===", flush=True)
