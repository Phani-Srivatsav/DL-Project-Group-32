# Experimental Runs & Results Directory

Each folder in this directory is fully self-contained, including the exact Python model and training code, universal GPU execution runner (`run.sh`), evaluation module, replay buffer, metrics log (`metrics.csv`), interactive trajectory visualizations (`vis.html`), and comprehensive post-run diagnostic reports.

| Run Directory | Architecture | Key Characteristics | Exact Code & Runner | Outcome / Status |
| :--- | :--- | :--- | :--- | :--- |
| **[`run_1_single_node_baseline`](run_1_single_node_baseline/)** | NODE Critic + ResNet-4 Actor | $s=4$, NFE=16, Width 256 | `train_node.py`<br>`run.sh` | Capacity mismatch; policy underfitted ($0.0$ score) |
| **[`run_2_dual_node_unoptimized`](run_2_dual_node_unoptimized/)** | Dual NODE v1 (Actor + Critic) | Euler solver, $s=4$, Width 256 | `train_dual_node.py`<br>`run.sh` | Reached 3 corridors ($14.5\text{m} \to 3.5\text{m}$); Brax metric cutoff identified |
| **[`run_3_dual_node_optimized_s8`](run_3_dual_node_optimized_s8/)** | Dual NODE v2 (Optimized) | RK4 solver, $s=8$, NFE=32, Width 512 | `train_dual_node.py`<br>`run.sh` | **Peak 26.14** (surpassed paper's ResNet-4 peak of ~18.0) |
| **[`run_4_breakthrough_dual_node_s16`](run_4_breakthrough_dual_node_s16/)** | Scaled Breakthrough Dual NODE | RK4 solver, $s=16$, NFE=64, Width 512, Guided Search | `train_dual_node_breakthrough.py`<br>`run.sh` | **82M+ steps trained**; 42% leaner than ResNet-64; stable loss ($3.27$) |

## How to Run Any Experiment
To reproduce any run on any workstation or server equipped with an NVIDIA GPU:
```bash
cd experiment_runs/run_4_breakthrough_dual_node_s16
bash run.sh
```
Or run the Python script directly with your custom arguments:
```bash
python train_dual_node_breakthrough.py --env-id ant_u4_maze --ode-steps 16
```
