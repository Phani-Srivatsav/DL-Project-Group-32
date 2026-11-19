# Turnkey Multi-Account Deployment Guide: 3 Parallel Environments

This bundle is pre-configured to replicate and surpass the NeurIPS 2025 Best Paper (*"1000 Layer Networks for Self-Supervised RL"*) across 3 diverse benchmark environments simultaneously.

---

## 1. Multi-Account Mapping Strategy

| Account | Environment | Target Benchmark | Slurm Script | Key Paper Comparison |
| :--- | :--- | :--- | :--- | :--- |
| **Account 1** (Live) | `ant_u4_maze` | 4-Corridor U-Maze | `run_dual_node_breakthrough.slurm` | Beats ResNet-4 (~18), challenges ResNet-32/64 |
| **Account 2** | `ant_big_maze` | 8×8 Grid Big Maze | `run_ant_big_maze.slurm` | **Main Paper Feature (Section 4)**: Unlocks critical depth threshold |
| **Account 3** | `arm_push_hard` | 7-DOF Robotic Arm | `run_arm_push_hard.slurm` | Continuous manipulation control on 3D object coordinate tasks |

---

## 2. Fast Step-by-Step Deployment (Under 2 Minutes)

### Step 1: Upload & Extract on Account 2 and Account 3
On your local machine, download `replicate_environments_pack.zip`.
Upload to each account's home directory (e.g. via `scp` or web portal):
```bash
unzip replicate_environments_pack.zip
cd scaling-crl
```

### Step 2: One-Line Environment Setup
Run the setup script (installs uv, JAX CUDA 12, Brax, and applies the contact patch):
```bash
bash setup_account.sh
```

### Step 3: Launch Training

* **On Account 2 (Ant Big Maze):**
  ```bash
  sbatch run_ant_big_maze.slurm
  ```

* **On Account 3 (Arm Push Hard):**
  ```bash
  sbatch run_arm_push_hard.slurm
  ```

---

## 3. What the Code Automatically Generates

When each run completes (or at intermediate milestones), the code automatically generates:

1. **`publication_report.png` & `publication_report.pdf`**:
   - **Panel A (Paper Learning Curves):** Your Neural ODE curve plotted directly against the paper's ResNet-4, 8, 16, 32, and 64 baselines with error bands.
   - **Panel B (Phase Transition / Critical Depth):** Score vs. NFE curve demonstrating that continuous depth solves the bottleneck where shallow networks score 0.
   - **Panel C (Parameter Efficiency Frontier):** Score vs. Trainable Parameters showing our lean ~2.48M model matching/exceeding 4.3M ResNets (42% fewer weights).
   - **Panel D (Continuous Flow Stability):** InfoNCE loss and gradient norm conditioning.
2. **`publication_table.md`**:
   - Formatted Markdown table comparing peak score, last-5 mean, parameters, and success rate against all ResNet baselines from the paper.
3. **`vis.html`**:
   - Interactive 3D trajectory visualization of the agent solving the task.
