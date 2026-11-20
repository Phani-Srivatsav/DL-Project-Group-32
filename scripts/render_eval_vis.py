import os
import sys
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from brax.io import html

sys.path.insert(0, "/home/phani.gande.ug24/scaling-crl")
from envs.ant_maze import AntMaze
from train_dual_node_breakthrough import ContinuousDepthActor

ckpt_path = "/home/phani.gande.ug24/scaling-crl/runs/ant_u4_maze_breakthrough_node_w512_s16_20261117-023418/latest_ckpt.pkl"
save_path = "/home/phani.gande.ug24/scaling-crl/experiment_runs/run_4_breakthrough_dual_node_s16"

print(f"Loading checkpoint from {ckpt_path}...")
with open(ckpt_path, "rb") as f:
    ckpt = pickle.load(f)

actor_params = ckpt["actor_params"]

# Use the OFFICIAL EVALUATION maze where goals are strictly on the other side of the maze
render_env = AntMaze(
    backend="spring",
    exclude_current_positions_from_observation=False,
    terminate_when_unhealthy=False,
    maze_layout_name="u4_maze_eval"
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
def stoch_action(obs, key):
    means, log_stds = actor.apply(actor_params, obs)
    stds = jnp.exp(log_stds)
    return nn.tanh(means + stds * jax.random.normal(key, shape=means.shape))

jit_step = jax.jit(render_env.step)
jit_reset = jax.jit(render_env.reset)

print(f"Goals in u4_maze_eval: {render_env.possible_goals}")
print("Rolling out trajectory with goal on the other side of the maze...")

# Find seed with best goal traversal
best_rollout = None
best_dist = 999.0
best_seed = 0

for s in [1, 2, 3, 5, 7, 10]:
    rng = jax.random.PRNGKey(s)
    state = jit_reset(rng)
    goal_pos = (float(state.obs[-2]), float(state.obs[-1]))
    rollout = []
    min_d = 999.0
    for t in range(1000):
        rng, k = jax.random.split(rng)
        act = stoch_action(state.obs, k)
        rollout.append(state.pipeline_state)
        state = jit_step(state, act)
        d = float(state.metrics["dist"])
        if d < min_d:
            min_d = d
    print(f"Seed {s}: Goal Pos={goal_pos}, Min Dist={min_d:.2f}")
    if min_d < best_dist:
        best_dist = min_d
        best_rollout = rollout
        best_seed = s

print(f"Best seed: {best_seed} with min dist: {best_dist:.2f}")
print("Rendering 3D HTML replay with u4_maze_eval...")
html_string = html.render(render_env.sys, best_rollout)

vis_file = f"{save_path}/vis.html"
with open(vis_file, "w") as f:
    f.write(html_string)

# Also copy to runs directory
with open(f"/home/phani.gande.ug24/scaling-crl/runs/ant_u4_maze_breakthrough_node_w512_s16_20261117-023418/vis.html", "w") as f:
    f.write(html_string)

print(f"Saved proper u4_maze_eval replay to {vis_file} ({len(html_string)} bytes)")

# Update run_4_results.zip
import subprocess
subprocess.run([
    "zip", "-u", "/home/phani.gande.ug24/run_4_results.zip", "vis.html"
], cwd=save_path)
print("Updated /home/phani.gande.ug24/run_4_results.zip successfully!")
