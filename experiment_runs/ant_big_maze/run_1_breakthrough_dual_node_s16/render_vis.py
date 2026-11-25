import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
from brax.io import html

from envs.ant_maze import AntMaze
from train_dual_node_breakthrough import (
    ContinuousDepthActor,
    SA_encoder,
    G_encoder,
    load_params,
)


def main():
    parser = argparse.ArgumentParser(description="Render vis.html from a saved Dual Neural ODE checkpoint.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="runs/ant_big_maze_breakthrough_node_w512_s16_20261119-201920/latest_ckpt.pkl",
        help="Path to checkpoint .pkl file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output vis.html path (defaults to <ckpt_dir>/vis.html)",
    )
    parser.add_argument("--num-render", type=int, default=10)
    parser.add_argument("--vis-length", type=int, default=1000)
    parser.add_argument("--eval-actor", type=int, default=10)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.output) if args.output else ckpt_path.parent / "vis.html"

    t0 = time.time()
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    ckpt = load_params(str(ckpt_path))
    actor_params = ckpt["actor_params"]
    critic_params = ckpt["critic_params"]
    print(f"Checkpoint epoch={ckpt.get('epoch')}, env_steps={ckpt.get('env_steps')}", flush=True)

    render_env = AntMaze(
        backend="spring",
        exclude_current_positions_from_observation=False,
        terminate_when_unhealthy=False,
        maze_layout_name="big_maze_eval",
    )
    obs_dim = 29
    action_size = render_env.action_size

    actor = ContinuousDepthActor(
        action_size=action_size,
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
        ode_rtol=1e-3,
        ode_atol=1e-4,
    )
    sa_encoder = SA_encoder(
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
        ode_rtol=1e-3,
        ode_atol=1e-4,
    )
    g_encoder = G_encoder(
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
        ode_rtol=1e-3,
        ode_atol=1e-4,
    )

    def policy_action(obs, key):
        obs_2d = obs[None, :]
        if args.eval_actor > 1:
            keys = jax.random.split(key, args.eval_actor)
            means, log_stds = actor.apply(actor_params, obs_2d)
            stds = jnp.exp(log_stds)
            actions = jnp.stack(
                [nn.tanh(means + stds * jax.random.normal(k, shape=means.shape, dtype=means.dtype)) for k in keys]
            )
            state = obs_2d[:, :obs_dim]
            goal = obs_2d[:, obs_dim:]
            sa_reprs = jax.vmap(lambda a: sa_encoder.apply(critic_params["sa_encoder"], state, a))(actions)
            g_repr = g_encoder.apply(critic_params["g_encoder"], goal)
            q_values = -jnp.sqrt(jnp.sum((sa_reprs - g_repr) ** 2, axis=-1))
            best_action_idx = jnp.argmax(q_values, axis=0)
            best_action = jnp.take_along_axis(actions, best_action_idx[None, :, None], axis=0)[0]
            return best_action[0]
        else:
            means, _ = actor.apply(actor_params, obs_2d)
            return nn.tanh(means)[0]

    @jax.jit
    def rollout_episode(rng):
        rng, reset_key = jax.random.split(rng)
        init_state = render_env.reset(reset_key)

        def step_fn(carry, _):
            st, k = carry
            k, act_key = jax.random.split(k)
            action = policy_action(st.obs, act_key)
            next_st = render_env.step(st, action)
            return (next_st, k), (st.pipeline_state, next_st.metrics["success"])

        _, (pipe_states, successes) = jax.lax.scan(step_fn, (init_state, rng), None, length=args.vis_length)
        return pipe_states, successes

    print(f"Running {args.num_render} episodes x {args.vis_length} steps on {jax.devices()}...", flush=True)
    rollout_states = []
    total_success_eps = 0
    for i in range(args.num_render):
        ep_t0 = time.time()
        pipe_states, successes = rollout_episode(jax.random.PRNGKey(seed=i + 1))
        pipe_states = jax.device_get(pipe_states)
        successes = jax.device_get(successes)
        ep_time_at_goal = float(successes.sum())
        ep_any = ep_time_at_goal > 0
        total_success_eps += int(ep_any)
        print(
            f"  Episode {i+1}/{args.num_render}: TimeAtGoal={ep_time_at_goal:.0f}, Success={ep_any} ({time.time()-ep_t0:.1f}s)",
            flush=True,
        )
        for t in range(args.vis_length):
            rollout_states.append(jax.tree_util.tree_map(lambda x: x[t], pipe_states))

    print(
        f"Rollouts done ({total_success_eps}/{args.num_render} succeeded). Rendering HTML ({len(rollout_states)} frames)...",
        flush=True,
    )
    html_string = html.render(render_env.sys, rollout_states)
    out_path.write_text(html_string)
    print(f"Saved 3D policy replay to {out_path} ({out_path.stat().st_size / (1024*1024):.1f} MB) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
