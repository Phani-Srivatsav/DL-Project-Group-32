# Multi-Environment Experimental Directory

This directory organizes all experimental runs, model checkpoints, training logs, and evaluations by environment. Each environment folder contains self-contained runs with standalone execution code, metrics, and visualization artifacts.

---

## Benchmark Environments

### 1. [`ant_u4_maze`](ant_u4_maze/)
**Goal-Conditioned Locomotion in 4-Corridor Maze (Brax / Spring Physics)**
- **Baseline Runs:** Single-Node Baseline (`run_1`), Dual-Node Unoptimized (`run_2`), Dual-Node RK4-S8 (`run_3`).
- **Breakthrough Model:** Dual Continuous-Depth Neural ODE with RK4 solver, $s=16$, Width 512, 125M environment steps (`run_4`).
- **Key Findings:**
  - `u4_maze` (10 corridor goals): Peak score **296.69** (Time-at-Goal) with **52.34%** success, matching ResNet-64 (296.32) with 42% fewer parameters.
  - `u4_maze_eval` (held-out far corridor): Run 3 reached **20.59** (23.4% success).
- Full details in [`ant_u4_maze/README.md`](ant_u4_maze/README.md).

---

### 2. [`ant_big_maze`](ant_big_maze/)
**High-Complexity Spatial Navigation (Brax / Spring Physics)**
- Scaled maze topology testing long-horizon temporal feature learning.
- Runner script: [`scripts/train_ant_big_maze.sh`](../scripts/train_ant_big_maze.sh).
- Maintained by: Dhruv.

---

### 3. [`arm_push_hard`](arm_push_hard/)
**Robotic Manipulation & Physics Interaction (Brax / MJX Physics)**
- 7-DOF robotic arm manipulation testing contact physics and multi-body dynamics.
- Runner script: [`scripts/train_arm_push.sh`](../scripts/train_arm_push.sh).
- Maintained by: Aarav.

---

## Directory Structure
```
experiment_runs/
├── ant_u4_maze/
│   ├── run_1_single_node_baseline/
│   ├── run_2_dual_node_unoptimized/
│   ├── run_3_dual_node_optimized_s8/
│   ├── run_4_breakthrough_dual_node_s16/
│   └── README.md
├── ant_big_maze/
│   └── README.md
├── arm_push_hard/
│   └── README.md
└── README.md
```
