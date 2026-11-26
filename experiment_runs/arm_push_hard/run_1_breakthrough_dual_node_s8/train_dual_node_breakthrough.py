import os
import csv
import jax
import flax
import tyro
import time
import optax
import wandb
import pickle
import random
import wandb_osh
import numpy as np
import flax.linen as nn
import jax.numpy as jnp

from brax import envs
from etils import epath
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import NamedTuple, Any
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling
from brax.io import html
from jax.experimental.ode import odeint

from evaluator import CrlEvaluator
from buffer import TrajectoryUniformSamplingQueue


@dataclass
class Args:
    exp_name: str = "train_dual_node_breakthrough"
    seed: int = 1000
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "clean_JaxGCRL_node"
    wandb_entity: str = ""
    wandb_mode: str = "offline"
    wandb_dir: str = "."
    wandb_group: str = "."
    capture_vis: bool = True
    vis_length: int = 1000
    checkpoint: bool = True
    checkpoint_every: int = 5
    resume_path: str = ""

    # Environment specific arguments
    env_id: str = "ant_u4_maze"
    episode_length: int = 1000
    terminate_when_unhealthy: int = 0  # 0: Do not early-terminate on stumble
    # Filled in at runtime
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0

    # Algorithm specific arguments (Table 7 defaults)
    total_env_steps: int = 125000000
    num_epochs: int = 125
    num_envs: int = 512
    eval_env_id: str = "ant_u4_maze_eval"
    num_eval_envs: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 512
    gamma: float = 0.995  # Longer planning horizon for 4-turn maze
    logsumexp_penalty_coeff: float = 0.1

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62

    # Architecture arguments
    actor_type: str = "node"  # "resnet" or "node"
    critic_type: str = "node"   # "node" or "resnet"
    critic_network_width: int = 512
    actor_network_width: int = 512
    actor_depth: int = 4
    critic_depth: int = 4       # used when critic_type == "resnet"
    actor_skip_connections: int = 4
    critic_skip_connections: int = 4

    # Neural ODE arguments
    ode_solver: str = "rk4"     # "euler", "rk4", or "dopri5"
    ode_steps: int = 16  # 16 steps RK4 = 64 NFE (matches ResNet-64 depth)
    ode_num_layers: int = 3     # MLP layers inside vector field f_theta(h, t)
    ode_t0: float = 0.0
    ode_t1: float = 1.0
    ode_rtol: float = 1e-3      # used for dopri5
    ode_atol: float = 1e-4      # used for dopri5
    grad_clip_norm: float = 0.0 # >0 enables global norm gradient clipping for critic

    num_episodes_per_env: int = 1
    training_steps_multiplier: int = 1
    use_all_batches: int = 0
    num_sgd_batches_per_training_step: int = 800

    eval_actor: int = 10  # 10: Test-time contrastive search guided by Neural ODE critic
    expl_actor: int = 1
    entropy_param: float = 0.5
    disable_entropy: int = 0
    use_relu: int = 0
    num_render: int = 10
    save_buffer: int = 1

    # Computed at runtime
    env_steps_per_actor_step: int = 0
    num_prefill_env_steps: int = 0
    num_prefill_actor_steps: int = 0
    num_training_steps_per_epoch: int = 0


lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
small_init = variance_scaling(1e-2, "fan_in", "uniform")
bias_init = nn.initializers.zeros


def residual_block(x, width, normalize, activation):
    identity = x
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = x + identity
    return x


class ODEFunc(nn.Module):
    """Time-conditioned derivative network dh/dt = f_theta(h, t)."""
    width: int = 256
    num_layers: int = 2
    norm_type: str = "layer_norm"
    use_relu: int = 0

    @nn.compact
    def __call__(self, h: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        normalize = (lambda x: nn.LayerNorm()(x)) if self.norm_type == "layer_norm" else (lambda x: x)
        activation = nn.relu if self.use_relu else nn.swish

        t_feat = jnp.broadcast_to(jnp.asarray(t, dtype=h.dtype), h.shape[:-1] + (1,))
        x = jnp.concatenate([h, t_feat], axis=-1)

        for _ in range(max(1, self.num_layers - 1)):
            x = nn.Dense(self.width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)

        # Small output init keeps initial flow smooth and close to identity
        dh_dt = nn.Dense(self.width, kernel_init=small_init, bias_init=bias_init)(x)
        return dh_dt


def integrate_fixed_step_scan(ode_func, ode_params, h0, t0, t1, steps, solver):
    """Memory-efficient fixed-step Euler or RK4 integrator using jax.lax.scan."""
    dt = (t1 - t0) / steps

    def dyn(h, t):
        return ode_func.apply({"params": ode_params}, h, t)

    def step_fn(carry, _):
        h, t = carry
        if solver == "euler":
            h_next = h + dt * dyn(h, t)
        elif solver == "rk4":
            k1 = dyn(h, t)
            k2 = dyn(h + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = dyn(h + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = dyn(h + dt * k3, t + dt)
            h_next = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError(f"Unknown fixed-step ODE solver: {solver}")
        return (h_next, t + dt), None

    (h_final, _), _ = jax.lax.scan(
        step_fn, (h0, jnp.asarray(t0, dtype=h0.dtype)), None, length=steps
    )
    return h_final


def get_nfe_per_pass(ode_solver: str, ode_steps: int) -> int:
    if ode_solver == "euler":
        return ode_steps
    elif ode_solver == "rk4":
        return 4 * ode_steps
    elif ode_solver == "dopri5":
        return -1  # Adaptive (variable NFE)
    return 0


class ContinuousDepthEncoder(nn.Module):
    """Shared encoder backbone supporting ResNet or Neural ODE continuous depth."""
    norm_type: str = "layer_norm"
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = 4
    use_relu: int = 0
    arch_type: str = "node"
    ode_solver: str = "rk4"
    ode_steps: int = 16  # 16 steps RK4 = 64 NFE (matches ResNet-64 depth)
    ode_num_layers: int = 3
    ode_t0: float = 0.0
    ode_t1: float = 1.0
    ode_rtol: float = 1e-3
    ode_atol: float = 1e-4

    def setup(self):
        self.in_dense = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)
        self.in_norm = nn.LayerNorm() if self.norm_type == "layer_norm" else (lambda x: x)
        if self.arch_type == "resnet":
            self.res_layers = [
                (
                    [nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init) for _ in range(4)],
                    [nn.LayerNorm() if self.norm_type == "layer_norm" else (lambda x: x) for _ in range(4)],
                )
                for _ in range(self.network_depth // 4)
            ]
        elif self.arch_type == "node":
            self.ode_func = ODEFunc(
                width=self.network_width,
                num_layers=self.ode_num_layers,
                norm_type=self.norm_type,
                use_relu=self.use_relu,
            )
        self.out_dense = nn.Dense(64, kernel_init=lecun_uniform, bias_init=bias_init)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        activation = nn.relu if self.use_relu else nn.swish
        x = activation(self.in_norm(self.in_dense(x)))

        if self.arch_type == "resnet":
            for denses, norms in self.res_layers:
                identity = x
                for dense, norm in zip(denses, norms):
                    x = activation(norm(dense(x)))
                x = x + identity
        elif self.arch_type == "node":
            t0 = jnp.asarray(self.ode_t0, dtype=x.dtype)
            if self.is_initializing():
                _ = self.ode_func(x, t0)
            else:
                ode_params = self.variables["params"]["ode_func"]
                if self.ode_solver in ("euler", "rk4"):
                    x = integrate_fixed_step_scan(
                        self.ode_func, ode_params, x, self.ode_t0, self.ode_t1, self.ode_steps, self.ode_solver
                    )
                elif self.ode_solver == "dopri5":
                    ts = jnp.array([self.ode_t0, self.ode_t1], dtype=x.dtype)
                    def dyn(y, t, params):
                        return self.ode_func.apply({"params": params}, y, t)
                    sol = odeint(dyn, x, ts, ode_params, rtol=self.ode_rtol, atol=self.ode_atol)
                    x = sol[-1]
                else:
                    raise ValueError(f"Unsupported ode_solver: {self.ode_solver}")
        else:
            raise ValueError(f"Unsupported arch_type: {self.arch_type}")

        return self.out_dense(x)


class SA_encoder(nn.Module):
    norm_type: str = "layer_norm"
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = 4
    use_relu: int = 0
    arch_type: str = "node"
    ode_solver: str = "rk4"
    ode_steps: int = 16  # 16 steps RK4 = 64 NFE (matches ResNet-64 depth)
    ode_num_layers: int = 3
    ode_t0: float = 0.0
    ode_t1: float = 1.0
    ode_rtol: float = 1e-3
    ode_atol: float = 1e-4

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([s, a], axis=-1)
        return ContinuousDepthEncoder(
            norm_type=self.norm_type,
            network_width=self.network_width,
            network_depth=self.network_depth,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            arch_type=self.arch_type,
            ode_solver=self.ode_solver,
            ode_steps=self.ode_steps,
            ode_num_layers=self.ode_num_layers,
            ode_t0=self.ode_t0,
            ode_t1=self.ode_t1,
            ode_rtol=self.ode_rtol,
            ode_atol=self.ode_atol,
        )(x)


class G_encoder(nn.Module):
    norm_type: str = "layer_norm"
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = 4
    use_relu: int = 0
    arch_type: str = "node"
    ode_solver: str = "rk4"
    ode_steps: int = 16  # 16 steps RK4 = 64 NFE (matches ResNet-64 depth)
    ode_num_layers: int = 3
    ode_t0: float = 0.0
    ode_t1: float = 1.0
    ode_rtol: float = 1e-3
    ode_atol: float = 1e-4

    @nn.compact
    def __call__(self, g: jnp.ndarray):
        return ContinuousDepthEncoder(
            norm_type=self.norm_type,
            network_width=self.network_width,
            network_depth=self.network_depth,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            arch_type=self.arch_type,
            ode_solver=self.ode_solver,
            ode_steps=self.ode_steps,
            ode_num_layers=self.ode_num_layers,
            ode_t0=self.ode_t0,
            ode_t1=self.ode_t1,
            ode_rtol=self.ode_rtol,
            ode_atol=self.ode_atol,
        )(g)


class ContinuousDepthActor(nn.Module):
    """Actor policy parameterized as a Continuous-Depth Neural ODE."""
    action_size: int
    norm_type: str = "layer_norm"
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = 4
    use_relu: int = 0
    arch_type: str = "node"   # "node" or "resnet"
    ode_solver: str = "rk4"
    ode_steps: int = 16  # 16 steps RK4 = 64 NFE (matches ResNet-64 depth)
    ode_num_layers: int = 3
    ode_t0: float = 0.0
    ode_t1: float = 1.0
    ode_rtol: float = 1e-3
    ode_atol: float = 1e-4
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -5.0

    def setup(self):
        self.in_dense = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)
        self.in_norm = nn.LayerNorm() if self.norm_type == "layer_norm" else (lambda x: x)
        if self.arch_type == "resnet":
            self.res_layers = [
                (
                    [nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init) for _ in range(4)],
                    [nn.LayerNorm() if self.norm_type == "layer_norm" else (lambda x: x) for _ in range(4)],
                )
                for _ in range(self.network_depth // 4)
            ]
        elif self.arch_type == "node":
            self.ode_func = ODEFunc(
                width=self.network_width,
                num_layers=self.ode_num_layers,
                norm_type=self.norm_type,
                use_relu=self.use_relu,
            )
        self.mean_dense = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)
        self.log_std_dense = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)

    def __call__(self, x: jnp.ndarray):
        activation = nn.relu if self.use_relu else nn.swish
        x = activation(self.in_norm(self.in_dense(x)))

        if self.arch_type == "resnet":
            for denses, norms in self.res_layers:
                identity = x
                for dense, norm in zip(denses, norms):
                    x = activation(norm(dense(x)))
                x = x + identity
        elif self.arch_type == "node":
            t0 = jnp.asarray(self.ode_t0, dtype=x.dtype)
            if self.is_initializing():
                _ = self.ode_func(x, t0)
            else:
                ode_params = self.variables["params"]["ode_func"]
                if self.ode_solver in ("euler", "rk4"):
                    x = integrate_fixed_step_scan(
                        self.ode_func, ode_params, x, self.ode_t0, self.ode_t1, self.ode_steps, self.ode_solver
                    )
                elif self.ode_solver == "dopri5":
                    ts = jnp.array([self.ode_t0, self.ode_t1], dtype=x.dtype)
                    def dyn(y, t, params):
                        return self.ode_func.apply({"params": params}, y, t)
                    sol = odeint(dyn, x, ts, ode_params, rtol=self.ode_rtol, atol=self.ode_atol)
                    x = sol[-1]
                else:
                    raise ValueError(f"Unsupported ode_solver: {self.ode_solver}")
        else:
            raise ValueError(f"Unsupported arch_type: {self.arch_type}")

        mean = self.mean_dense(x)
        log_std = self.log_std_dense(x)
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std



@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


class Transition(NamedTuple):
    """Container for a transition."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


if __name__ == "__main__":
    args = tyro.cli(Args)

    print("Arguments:", flush=True)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}", flush=True)
    print("\n", flush=True)

    args.env_steps_per_actor_step = args.num_envs * args.unroll_length
    args.num_prefill_env_steps = args.min_replay_size * args.num_envs
    args.num_prefill_actor_steps = int(np.ceil(args.min_replay_size / args.unroll_length))
    args.num_training_steps_per_epoch = max(
        1,
        (args.total_env_steps - args.num_prefill_env_steps) // (args.num_epochs * args.env_steps_per_actor_step),
    )

    print(f"env_steps_per_actor_step: {args.env_steps_per_actor_step}", flush=True)
    print(f"num_prefill_env_steps: {args.num_prefill_env_steps}", flush=True)
    print(f"num_prefill_actor_steps: {args.num_prefill_actor_steps}", flush=True)
    print(f"num_training_steps_per_epoch: {args.num_training_steps_per_epoch}", flush=True)

    run_name = (
        f"{args.env_id}_critic:{args.critic_type}({args.ode_solver}-s{args.ode_steps})_"
        f"actor:{args.actor_type}(d{args.actor_depth})_bs{args.batch_size}_seed{args.seed}"
    )
    print(f"run_name: {run_name}", flush=True)

    if args.track:
        if args.wandb_group == ".":
            args.wandb_group = None
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity or None,
            mode=args.wandb_mode,
            group=args.wandb_group,
            dir=args.wandb_dir,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        if args.wandb_mode == "offline":
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()

    save_path = None
    if args.checkpoint:
        if args.resume_path and os.path.exists(args.resume_path):
            save_path = Path(args.resume_path).parent
        else:
            short_run_name = f"runs/{args.env_id}_breakthrough_node_w{args.actor_network_width}_s{args.ode_steps}_ep{args.episode_length}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            save_path = Path(args.wandb_dir) / Path(short_run_name)
            os.makedirs(save_path, exist_ok=True)
        print(f"Saving artifacts & checkpoints to: {save_path}", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

    def make_env(env_id=args.env_id):
        print(f"making env with env_id: {env_id}", flush=True)
        if env_id == "ant":
            from envs.ant import Ant
            env = Ant(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
            args.obs_dim = 29
            args.goal_start_idx = 0
            args.goal_end_idx = 2
        elif "ant" in env_id and "maze" in env_id:
            if "gen" not in env_id:
                from envs.ant_maze import AntMaze
                env = AntMaze(
                    backend="spring",
                    exclude_current_positions_from_observation=False,
                    terminate_when_unhealthy=bool(args.terminate_when_unhealthy),
                    maze_layout_name=env_id[4:],
                )
                args.obs_dim = 29
                args.goal_start_idx = 0
                args.goal_end_idx = 2
            else:
                from envs.ant_maze_generalization import AntMazeGeneralization
                gen_idx = env_id.find("gen")
                maze_layout_name = env_id[4 : gen_idx - 1]
                generalization_config = env_id[gen_idx + 4 :]
                env = AntMazeGeneralization(
                    backend="spring",
                    exclude_current_positions_from_observation=False,
                    terminate_when_unhealthy=True,
                    maze_layout_name=maze_layout_name,
                    generalization_config=generalization_config,
                )
                args.obs_dim = 29
                args.goal_start_idx = 0
                args.goal_end_idx = 2
        elif env_id == "humanoid":
            from envs.humanoid import Humanoid
            env = Humanoid(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
            args.obs_dim = 268
            args.goal_start_idx = 0
            args.goal_end_idx = 3
        elif "humanoid" in env_id and "maze" in env_id:
            from envs.humanoid_maze import HumanoidMaze
            env = HumanoidMaze(backend="spring", maze_layout_name=env_id[9:])
            args.obs_dim = 268
            args.goal_start_idx = 0
            args.goal_end_idx = 3
        elif env_id == "reacher":
            from envs.reacher import Reacher
            env = Reacher(backend="spring")
            args.obs_dim = 10
            args.goal_start_idx = 4
            args.goal_end_idx = 7
        elif env_id == "pusher":
            from envs.pusher import Pusher
            env = Pusher(backend="spring")
            args.obs_dim = 20
            args.goal_start_idx = 10
            args.goal_end_idx = 13
        elif env_id == "ant_ball":
            from envs.ant_ball import AntBall
            env = AntBall(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
            args.obs_dim = 31
            args.goal_start_idx = 28
            args.goal_end_idx = 30
        elif env_id == "ant_push":
            from envs.ant_push import AntPush
            env = AntPush(backend="mjx")
            args.obs_dim = 31
            args.goal_start_idx = 0
            args.goal_end_idx = 2
        elif env_id == "arm_reach":
            from envs.manipulation.arm_reach import ArmReach
            env = ArmReach(backend="mjx")
            args.obs_dim = 13
            args.goal_start_idx = 7
            args.goal_end_idx = 10
        elif env_id in ["arm_binpick_easy", "arm_binpick_hard"]:
            if env_id == "arm_binpick_easy":
                from envs.manipulation.arm_binpick_easy import ArmBinpickEasy
                env = ArmBinpickEasy(backend="mjx")
            else:
                from envs.manipulation.arm_binpick_hard import ArmBinpickHard
                env = ArmBinpickHard(backend="mjx")
            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3
        elif env_id in ["arm_push_easy", "arm_push_hard"]:
            if env_id == "arm_push_easy":
                from envs.manipulation.arm_push_easy import ArmPushEasy
                env = ArmPushEasy(backend="mjx")
            else:
                from envs.manipulation.arm_push_hard import ArmPushHard
                env = ArmPushHard(backend="mjx")
            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3
        else:
            raise NotImplementedError(f"Unsupported env_id: {env_id}")
        return env

    env = make_env(args.env_id)
    env = envs.training.wrap(env, episode_length=args.episode_length)

    obs_size = env.observation_size
    action_size = env.action_size
    env_keys = jax.random.split(env_key, args.num_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step = jax.jit(env.step)
    print(f"obs_size: {obs_size}, action_size: {action_size}", flush=True)

    if not args.eval_env_id:
        args.eval_env_id = args.env_id
    eval_env = make_env(args.eval_env_id)
    eval_env = envs.training.wrap(eval_env, episode_length=args.episode_length)

    # Actor (Neural ODE or ResNet)
    actor = ContinuousDepthActor(
        action_size=action_size,
        norm_type="layer_norm",
        network_width=args.actor_network_width,
        network_depth=args.actor_depth,
        skip_connections=args.actor_skip_connections,
        use_relu=args.use_relu,
        arch_type=args.actor_type,
        ode_solver=args.ode_solver,
        ode_steps=args.ode_steps,
        ode_num_layers=args.ode_num_layers,
        ode_t0=args.ode_t0,
        ode_t1=args.ode_t1,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, obs_size])),
        tx=optax.adam(learning_rate=args.actor_lr),
    )

    # Critic (Neural ODE or ResNet)
    sa_encoder = SA_encoder(
        network_width=args.critic_network_width,
        network_depth=args.critic_depth,
        skip_connections=args.critic_skip_connections,
        use_relu=args.use_relu,
        arch_type=args.critic_type,
        ode_solver=args.ode_solver,
        ode_steps=args.ode_steps,
        ode_num_layers=args.ode_num_layers,
        ode_t0=args.ode_t0,
        ode_t1=args.ode_t1,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
    )
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, args.obs_dim]), np.ones([1, action_size]))

    g_encoder = G_encoder(
        network_width=args.critic_network_width,
        network_depth=args.critic_depth,
        skip_connections=args.critic_skip_connections,
        use_relu=args.use_relu,
        arch_type=args.critic_type,
        ode_solver=args.ode_solver,
        ode_steps=args.ode_steps,
        ode_num_layers=args.ode_num_layers,
        ode_t0=args.ode_t0,
        ode_t1=args.ode_t1,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
    )
    g_encoder_params = g_encoder.init(g_key, np.ones([1, args.goal_end_idx - args.goal_start_idx]))

    critic_tx = (
        optax.chain(optax.clip_by_global_norm(args.grad_clip_norm), optax.adam(learning_rate=args.critic_lr))
        if args.grad_clip_norm > 0.0
        else optax.adam(learning_rate=args.critic_lr)
    )
    critic_state = TrainState.create(
        apply_fn=None,
        params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
        tx=critic_tx,
    )

    sa_param_count = sum(x.size for x in jax.tree_util.tree_leaves(sa_encoder_params))
    g_param_count = sum(x.size for x in jax.tree_util.tree_leaves(g_encoder_params))
    actor_param_count = sum(x.size for x in jax.tree_util.tree_leaves(actor_state.params))
    print(f"Parameter counts -> Actor: {actor_param_count:,} | Critic SA: {sa_param_count:,} | Critic G: {g_param_count:,}", flush=True)

    target_entropy = -args.entropy_param * action_size
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": log_alpha},
        tx=optax.adam(learning_rate=args.alpha_lr),
    )

    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
    )

    start_epoch = 0
    if args.resume_path and os.path.exists(args.resume_path):
        print(f"Resuming checkpoint from {args.resume_path} ...", flush=True)
        ckpt = load_params(args.resume_path)
        training_state = training_state.replace(
            env_steps=jnp.asarray(ckpt["env_steps"]),
            gradient_steps=jnp.asarray(ckpt["gradient_steps"]),
            alpha_state=alpha_state.replace(params=ckpt["alpha_params"], opt_state=ckpt["alpha_opt"]),
            actor_state=actor_state.replace(params=ckpt["actor_params"], opt_state=ckpt["actor_opt"]),
            critic_state=critic_state.replace(params=ckpt["critic_params"], opt_state=ckpt["critic_opt"]),
        )
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resumed at epoch {start_epoch}, env_steps={int(ckpt['env_steps'])}", flush=True)

    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        extras={"state_extras": {"truncation": 0.0, "seed": 0.0}},
    )

    def jit_wrap(buffer):
        buffer.insert_internal = jax.jit(buffer.insert_internal)
        buffer.sample_internal = jax.jit(buffer.sample_internal)
        return buffer

    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=args.max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=args.batch_size,
            num_envs=args.num_envs,
            episode_length=args.episode_length,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)
    resumed_buffer = False
    if args.resume_path and os.path.exists(args.resume_path):
        buffer_ckpt_path = Path(args.resume_path).parent / "latest_buffer.pkl"
        if buffer_ckpt_path.exists():
            print(f"Restoring replay buffer from {buffer_ckpt_path} ...", flush=True)
            try:
                buf_ckpt = load_params(str(buffer_ckpt_path))
                from buffer import ReplayBufferState
                buffer_state = ReplayBufferState(
                    data=jnp.asarray(buf_ckpt["data"]),
                    insert_position=jnp.asarray(buf_ckpt["insert_position"], dtype=jnp.int32),
                    sample_position=jnp.asarray(buf_ckpt["sample_position"], dtype=jnp.int32),
                    key=jnp.asarray(buf_ckpt["key"]),
                )
                resumed_buffer = True
                print("Replay buffer restored successfully from checkpoint!", flush=True)
            except Exception as e:
                print(f"Warning: Failed to restore buffer: {e}. Will prefill from scratch.", flush=True)

    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
        actions = nn.tanh(means)
        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    def actor_step(training_state, env, env_state, key, extra_fields):
        means, log_stds = actor.apply(training_state.actor_state.params, env_state.obs)
        stds = jnp.exp(log_stds)
        actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))
        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    def multi_sample_actor_step(training_state, env, env_state, key, K, extra_fields):
        keys = jax.random.split(key, K)
        means, log_stds = actor.apply(training_state.actor_state.params, env_state.obs)
        stds = jnp.exp(log_stds)
        actions = jnp.stack([nn.tanh(means + stds * jax.random.normal(k, shape=means.shape, dtype=means.dtype)) for k in keys])
        state = env_state.obs[:, :args.obs_dim]
        goal = env_state.obs[:, args.obs_dim:]
        sa_reprs = jax.vmap(lambda a: sa_encoder.apply(training_state.critic_state.params["sa_encoder"], state, a))(actions)
        g_repr = g_encoder.apply(training_state.critic_state.params["g_encoder"], goal)
        q_values = -jnp.sqrt(jnp.sum((sa_reprs - g_repr) ** 2, axis=-1))
        best_action_idx = jnp.argmax(q_values, axis=0)
        best_actions = jnp.take_along_axis(actions, best_action_idx[None, :, None], axis=0)[0]
        nstate = env.step(env_state, best_actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs,
            action=best_actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    @jax.jit
    def get_experience(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            if args.expl_actor == 1:
                env_state, transition = actor_step(training_state, env, env_state, current_key, extra_fields=("truncation", "seed"))
            elif args.expl_actor == 0:
                env_state, transition = deterministic_actor_step(training_state, env, env_state, extra_fields=("truncation", "seed"))
            else:
                env_state, transition = multi_sample_actor_step(training_state, env, env_state, current_key, args.expl_actor, extra_fields=("truncation", "seed"))
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=args.unroll_length)
        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(training_state, env_state, buffer_state, key)
            training_state = training_state.replace(env_steps=training_state.env_steps + args.env_steps_per_actor_step)
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=args.num_prefill_actor_steps)[0]

    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        actor_batch_size = args.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:actor_batch_size], transitions)

        def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
            obs = transitions.observation
            state = obs[:, :args.obs_dim]
            future_state = transitions.extras["future_state"]
            goal = future_state[:, args.goal_start_idx : args.goal_end_idx]
            observation = jnp.concatenate([state, goal], axis=1)

            means, log_stds = actor.apply(actor_params, observation)
            stds = jnp.exp(log_stds)
            x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            action = nn.tanh(x_ts)
            log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)

            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
            sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
            g_repr = g_encoder.apply(g_encoder_params, goal)

            qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

            if args.disable_entropy:
                loss = -jnp.mean(qf_pi)
            else:
                loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)
            return loss, log_prob

        def alpha_loss(alpha_params, log_prob):
            alpha = jnp.exp(alpha_params["log_alpha"])
            loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - target_entropy))
            return jnp.mean(loss)

        (actorloss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
            training_state.actor_state.params,
            training_state.critic_state.params,
            training_state.alpha_state.params["log_alpha"],
            transitions,
            key,
        )
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        alphaloss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
        new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

        training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)
        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actorloss,
            "alph_aloss": alphaloss,
            "log_alpha": training_state.alpha_state.params["log_alpha"],
        }
        return training_state, metrics

    @jax.jit
    def update_critic(transitions, training_state, key):
        critic_batch_size = args.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:critic_batch_size], transitions)

        def critic_loss(critic_params, transitions, key):
            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
            obs = transitions.observation[:, :args.obs_dim]
            action = transitions.action

            sa_repr = sa_encoder.apply(sa_encoder_params, obs, action)
            g_repr = g_encoder.apply(g_encoder_params, transitions.observation[:, args.obs_dim:])

            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
            loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp ** 2)
            return loss, logsumexp

        (loss, logsumexp), grad = jax.value_and_grad(critic_loss, has_aux=True)(
            training_state.critic_state.params, transitions, key
        )
        critic_grad_norm = optax.global_norm(grad)
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state=new_critic_state)

        metrics = {
            "logsumexp": logsumexp.mean(),
            "critic_loss": loss,
            "critic_grad_norm": critic_grad_norm,
        }
        return training_state, metrics

    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key = jax.random.split(key, 3)

        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, actor_key)
        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)
        training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        return (training_state, key), metrics

    @jax.jit
    def training_step(training_state, env_state, buffer_state, key, t):
        experience_key1, experience_key2, sampling_key, training_key, sgd_batches_key = jax.random.split(key, 5)

        env_state, buffer_state = get_experience(training_state, env_state, buffer_state, experience_key1)
        training_state = training_state.replace(env_steps=training_state.env_steps + args.env_steps_per_actor_step)

        transitions_list = []
        for _ in range(args.num_episodes_per_env):
            buffer_state, new_transitions = replay_buffer.sample(buffer_state)
            transitions_list.append(new_transitions)

        transitions = jax.tree_util.tree_map(lambda *arrays: jnp.concatenate(arrays, axis=0), *transitions_list)

        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx), transitions, batch_keys
        )
        transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions)

        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)

        num_full_batches = len(transitions.observation) // args.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[: num_full_batches * args.batch_size], transitions)
        transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]), transitions)

        if args.use_all_batches == 0:
            num_total_batches = transitions.observation.shape[0]
            selected_indices = jax.random.permutation(sgd_batches_key, num_total_batches)[: args.num_sgd_batches_per_training_step]
            transitions = jax.tree_util.tree_map(lambda x: x[selected_indices], transitions)

        (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)
        return (training_state, env_state, buffer_state), metrics

    @jax.jit
    def training_epoch(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es, bs), metrics = training_step(ts, es, bs, train_key, t)
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            jnp.arange(args.num_training_steps_per_epoch * args.training_steps_multiplier),
        )
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    if not resumed_buffer:
        key, prefill_key = jax.random.split(key, 2)
        print("Prefilling replay buffer...", flush=True)
        training_state, env_state, buffer_state, _ = prefill_replay_buffer(training_state, env_state, buffer_state, prefill_key)
    else:
        print("Skipping buffer prefill: Replay buffer was restored from checkpoint.", flush=True)

    if args.eval_actor == 0:
        evaluator = CrlEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
    elif args.eval_actor == 1:
        key, eval_actor_key = jax.random.split(key)
        evaluator = CrlEvaluator(
            lambda training_state, env, env_state, extra_fields: actor_step(
                training_state, env, env_state, eval_actor_key, extra_fields
            ),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
    else:
        key, eval_actor_key = jax.random.split(key)
        evaluator = CrlEvaluator(
            lambda training_state, env, env_state, extra_fields: multi_sample_actor_step(
                training_state, env, env_state, eval_actor_key, args.eval_actor, extra_fields
            ),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )

    critic_nfe = get_nfe_per_pass(args.ode_solver, args.ode_steps) if args.critic_type == "node" else args.critic_depth

    csv_file = None
    if save_path is not None:
        csv_file = save_path / "metrics.csv"
        if not csv_file.exists():
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "env_steps", "time_at_goal", "success_any", "episode_dist",
                    "critic_loss", "actor_loss", "critic_grad_norm", "critic_nfe",
                    "training_sps", "walltime_hours"
                ])

    training_walltime = 0.0
    print("Starting training....", flush=True)
    start_time = time.time()
    recent_time_at_goal = []

    for ne in range(start_epoch, args.num_epochs):
        t = time.time()
        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(training_state, env_state, buffer_state, epoch_key)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time

        sps = (args.env_steps_per_actor_step * args.num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/envsteps": training_state.env_steps.item(),
            "training/critic_nfe": critic_nfe,
            **{f"training/{name}": value for name, value in metrics.items()},
        }

        metrics = evaluator.run_evaluation(training_state, metrics)

        time_at_goal = float(metrics.get("eval/episode_success", 0.0))
        success_any = float(metrics.get("eval/episode_success_any", 0.0))
        ep_dist = float(metrics.get("eval/episode_dist", 0.0))
        c_loss = float(metrics.get("training/critic_loss", 0.0))
        a_loss = float(metrics.get("training/actor_loss", 0.0))
        c_gnorm = float(metrics.get("training/critic_grad_norm", 0.0))
        hours_passed = (time.time() - start_time) / 3600.0

        recent_time_at_goal.append(time_at_goal)
        if len(recent_time_at_goal) > 5:
            recent_time_at_goal.pop(0)
        avg5_tag = float(np.mean(recent_time_at_goal))

        if np.isnan(c_loss) or np.isnan(a_loss):
            print(f"[WARNING] NaN detected at epoch {ne}! c_loss={c_loss}, a_loss={a_loss}", flush=True)

        print(
            f"[Epoch {ne+1}/{args.num_epochs}] Steps: {int(metrics['training/envsteps']):,} | "
            f"TimeAtGoal: {time_at_goal:.2f} (Last5Avg: {avg5_tag:.2f}) | "
            f"SuccessRate: {success_any:.3f} | Dist: {ep_dist:.1f} | "
            f"CriticLoss: {c_loss:.3f} | GradNorm: {c_gnorm:.3f} | "
            f"NFE: {critic_nfe} | SPS: {sps:.0f} | Elapsed: {hours_passed:.2f}h",
            flush=True,
        )

        if csv_file is not None:
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ne, int(metrics["training/envsteps"]), time_at_goal, success_any, ep_dist,
                    c_loss, a_loss, c_gnorm, critic_nfe, sps, hours_passed
                ])

        if args.checkpoint and save_path is not None:
            if ne < 3 or ne >= args.num_epochs - 5 or (ne + 1) % args.checkpoint_every == 0:
                ckpt_payload = {
                    "epoch": ne,
                    "env_steps": int(training_state.env_steps),
                    "gradient_steps": int(training_state.gradient_steps),
                    "alpha_params": training_state.alpha_state.params,
                    "alpha_opt": training_state.alpha_state.opt_state,
                    "actor_params": training_state.actor_state.params,
                    "actor_opt": training_state.actor_state.opt_state,
                    "critic_params": training_state.critic_state.params,
                    "critic_opt": training_state.critic_state.opt_state,
                }
                save_params(str(save_path / "latest_ckpt.pkl"), ckpt_payload)
                if args.save_buffer:
                    buffer_payload = {
                        "data": np.asarray(buffer_state.data),
                        "insert_position": int(buffer_state.insert_position),
                        "sample_position": int(buffer_state.sample_position),
                        "key": np.asarray(buffer_state.key),
                    }
                    save_params(str(save_path / "latest_buffer.pkl"), buffer_payload)
                if (ne + 1) % 10 == 0 or ne == args.num_epochs - 1:
                    save_params(str(save_path / f"step_{int(training_state.env_steps)}.pkl"), ckpt_payload)

        if args.track:
            wandb.log(metrics, step=ne)
            if args.wandb_mode == "offline":
                trigger_sync()

    if args.checkpoint and save_path is not None:
        with open(save_path / "args.pkl", "wb") as f:
            pickle.dump(args, f)
        if args.save_buffer:
            buffer_payload = {
                "data": np.asarray(buffer_state.data),
                "insert_position": int(buffer_state.insert_position),
                "sample_position": int(buffer_state.sample_position),
                "key": np.asarray(buffer_state.key),
            }
            save_params(str(save_path / "latest_buffer.pkl"), buffer_payload)
            print(f"Final replay buffer saved to {save_path / 'latest_buffer.pkl'}", flush=True)

    if args.capture_vis and save_path is not None:
        print("Rendering final policy with Neural ODE Critic Guidance...", flush=True)
        try:
            render_env = make_env(args.eval_env_id)
            jit_render_reset = jax.jit(render_env.reset)
            jit_render_step = jax.jit(render_env.step)

            if args.eval_actor > 1:
                @jax.jit
                def policy_action(actor_params, critic_params, obs, key):
                    obs_b = obs[None, :] if obs.ndim == 1 else obs
                    keys = jax.random.split(key, args.eval_actor)
                    means, log_stds = actor.apply(actor_params, obs_b)
                    stds = jnp.exp(log_stds)
                    actions = jnp.stack([nn.tanh(means + stds * jax.random.normal(k, shape=means.shape, dtype=means.dtype)) for k in keys])
                    state = obs_b[:, :args.obs_dim]
                    goal = obs_b[:, args.obs_dim:]
                    sa_reprs = jax.vmap(lambda a: sa_encoder.apply(critic_params["sa_encoder"], state, a))(actions)
                    g_repr = g_encoder.apply(critic_params["g_encoder"], goal)
                    q_values = -jnp.sqrt(jnp.sum((sa_reprs - g_repr) ** 2, axis=-1))
                    best_action_idx = jnp.argmax(q_values, axis=0)
                    best_act = jnp.take_along_axis(actions, best_action_idx[None, :, None], axis=0)[0]
                    return best_act[0] if obs.ndim == 1 else best_act
            else:
                @jax.jit
                def policy_action(actor_params, critic_params, obs, key):
                    obs_b = obs[None, :] if obs.ndim == 1 else obs
                    means, _ = actor.apply(actor_params, obs_b)
                    act = nn.tanh(means)
                    return act[0] if obs.ndim == 1 else act

            rollout_states = []
            for i in range(args.num_render):
                rng = jax.random.PRNGKey(seed=i + 1)
                r_state = jit_render_reset(rng)
                for _ in range(args.vis_length):
                    rng, act_key = jax.random.split(rng)
                    action = policy_action(training_state.actor_state.params, training_state.critic_state.params, r_state.obs, act_key)
                    rollout_states.append(r_state.pipeline_state)
                    r_state = jit_render_step(r_state, action)

            html_string = html.render(render_env.sys, rollout_states)
            render_path = f"{save_path}/vis.html"
            with open(render_path, "w") as f:
                f.write(html_string)
            print(f"Saved 3D policy replay to {render_path}", flush=True)
            if args.track:
                wandb.log({"vis": wandb.Html(html_string)})
        except Exception as e:
            print(f"Error rendering final policy: {e}", flush=True)

    print("Training complete!", flush=True)