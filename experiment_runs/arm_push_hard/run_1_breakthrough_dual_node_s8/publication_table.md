# Quantitative Benchmark Report: arm_push_hard
**Benchmark Reference:** NeurIPS 2025 Best Paper (*"1000 Layer Networks for Self-Supervised RL"*)

| Architecture | Model Type | Effective NFE | Trainable Params | Peak Performance | Last-5 Mean Performance | Peak Success Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-4** | Discrete | 4 | ~0.58M | 185.23 | — | — |
| **ResNet-8** | Discrete | 8 | ~0.85M | 354.52 | — | — |
| **ResNet-16** | Discrete | 16 | ~1.40M | 453.34 | — | — |
| **ResNet-32** | Discrete | 32 | ~2.20M | 471.88 | — | — |
| **ResNet-64** | Discrete | 64 | ~4.30M | 443.85 | — | — |
| **Dual Neural ODE (Ours)** | **Continuous Flow** | **32** | **~0.65M** | **309.39** | **212.93 ± 18.08** | **57.8%** |

### Run Diagnostics (100 epochs):
- **Effective Depth / NFE:** 32 (RK4, `ode_steps=8`) with **~0.65M** total trainable parameters.
- **Peak vs. Last-5 `TimeAtGoal`:** Peak `309.39`, Last-5 `212.93 ± 18.08`, Peak `success_any` `57.8%`.
- **Training Stability:** Final Critic InfoNCE loss `1.057`; Critic gradient norm range `8.95`–`121.32`.
