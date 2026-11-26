# Environment: arm_push_hard

**Task Description:** 7-DOF robotic manipulator pushing an object to variable target goal coordinates in 3D space using Brax MJX physics.

## Execution
Run on any GPU system:
```bash
bash ../../scripts/train_arm_push.sh
# or standalone inside run directory:
bash run_1_breakthrough_dual_node_s8/run.sh
```

## Status
- Training scripts: Configured with continuous-depth Neural ODE architecture (RK4 solver, $s=8$, $\text{NFE}=32$, Width 256).
- **Run 1 (`run_1_breakthrough_dual_node_s8/`):** Completed (100 epochs, 100M environment steps, 2.48M SGD steps). Maintained by Aarav (`run_arm_push_hard.slurm`, Slurm Job `16782`).
  - **Peak Time-at-Goal / Success Score:** **309.39** (beat Authors' ResNet-4 baseline of 185.23 by **+67.0%**)
  - **Peak Success Rate:** **57.8%** (Final Epoch: 51.6%)
  - **Last-5 Mean Performance:** **212.93 ± 18.08**
  - **Parameter Footprint:** **~0.65M** (vs. 2.20M for ResNet-32, ~70% parameter reduction)

See [`run_1_breakthrough_dual_node_s8/README.md`](run_1_breakthrough_dual_node_s8/README.md) for full artifacts, checkpoints, code, and logs.

---

### Quantitative Benchmark Comparison
**Benchmark Reference:** NeurIPS 2025 Best Paper (*"1000 Layer Networks for Self-Supervised RL"*)

| Architecture | Model Type | Effective NFE | Trainable Params | Peak Performance | Last-5 Mean Performance | Peak Success Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-4** | Discrete | 4 | ~0.58M | 185.23 | — | — |
| **ResNet-8** | Discrete | 8 | ~0.85M | 354.52 | — | — |
| **ResNet-16** | Discrete | 16 | ~1.40M | 453.34 | — | — |
| **ResNet-32** | Discrete | 32 | ~2.20M | 471.88 | — | — |
| **ResNet-64** | Discrete | 64 | ~4.30M | 443.85 | — | — |
| **Dual Neural ODE (Ours)** | **Continuous Flow** | **32** | **~0.65M** | **309.39** | **212.93 ± 18.08** | **57.8%** |

---

### Publication Report
![Publication Report](run_1_breakthrough_dual_node_s8/publication_report.png)
