# Quantitative Benchmark Report: ant_u4_maze
**Benchmark Reference:** NeurIPS 2025 Best Paper (*"1000 Layer Networks for Self-Supervised RL"*)

| Architecture | Model Type | Effective NFE | Trainable Params | Peak Performance | Last-5 Mean Performance | Peak Success Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-4** | Discrete | 4 | ~0.58M | 13.66 | — | — |
| **ResNet-8** | Discrete | 8 | ~0.85M | 114.88 | — | — |
| **ResNet-16** | Discrete | 16 | ~1.40M | 145.01 | — | — |
| **ResNet-32** | Discrete | 32 | ~2.20M | 257.71 | — | — |
| **ResNet-64** | Discrete | 64 | ~4.30M | 296.32 | — | — |
| **Dual Neural ODE (Ours)** | **Continuous Flow** | **64** | **~2.48M** | **296.69** | **290.71 ± 4.03** | **52.3%** |

### Key Scientific Takeaways for Project Presentation:
1. **Critical Depth Phenomenon:** Just as demonstrated in the NeurIPS paper, shallow models fail to navigate maze bottlenecks or complex object coordinates. Continuous flows provide the necessary temporal-feature depth.
2. **Superior Parameter Efficiency:** The Dual Neural ODE matches/surpasses ResNet-64's depth (NFE=64) while using only **~2.48M parameters** (~42% leaner than ResNet-64's 4.3M parameters).
3. **Smooth Flow Conditioning:** Gradient norms stay stable (~3.5–4.5) across all 64 integration passes without gradient explosion.
