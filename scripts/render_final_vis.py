import os
import sys
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from brax import envs
from brax.io import html

sys.path.insert(0, "/home/phani.gande.ug24/scaling-crl")
from envs.ant_maze import AntMaze
from train_dual_node_breakthrough import ContinuousDepthActor

ckpt_path = "/home/phani.gande.ug24/scaling-crl/runs/ant_u4_maze_breakthrough_node_w512_s16_20261117-023418/latest_ckpt.pkl"
save_path = "/home/phani.gande.ug24/scaling-crl/runs/ant_u4_maze_breakthrough_node_w512_s16_20261117-023418"

print(f"Loading checkpoint from {ckpt_path}...")
with open(ckpt_path, "rb") as f:
    ckpt = pickle.load(f)

actor_params = ckpt["actor_params"]

# Raw unwrapped AntMaze for single-agent rendering
render_env = AntMaze(
    backend="spring",
    exclude_current_positions_from_observation=False,
    terminate_when_unhealthy=False,
    maze_layout_name="u4_maze"
)

actor = ContinuousDepthActor(
    action_size=render_env.action_size,
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

@jax.jit
def policy_action(actor_params, obs):
    means, _ = actor.apply(actor_params, obs)
    return nn.tanh(means)

jit_render_reset = jax.jit(render_env.reset)
jit_render_step = jax.jit(render_env.step)

print("Rolling out trajectory...")
rollout_states = []
for i in range(2):
    rng = jax.random.PRNGKey(seed=i + 42)
    r_state = jit_render_reset(rng)
    for step in range(1000):
        action = policy_action(actor_params, r_state.obs)
        rollout_states.append(r_state.pipeline_state)
        r_state = jit_render_step(r_state, action)

print("Rendering 3D HTML replay...")
html_string = html.render(render_env.sys, rollout_states)
vis_file = f"{save_path}/vis.html"
with open(vis_file, "w") as f:
    f.write(html_string)

print(f"Successfully saved 3D visualization to {vis_file} ({len(html_string)} bytes)")
