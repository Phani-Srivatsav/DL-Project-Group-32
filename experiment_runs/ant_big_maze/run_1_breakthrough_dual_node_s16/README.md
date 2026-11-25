# Run 1: Breakthrough Dual Neural ODE Architecture (`ant_big_maze`, $s = 16$, $\text{NFE} = 64$)

### 1. Configuration & Specification
- **Run ID:** `ant_big_maze_breakthrough_node_w512_s16_20261119-201920`
- **Continuous Flow Solver:** 4th-Order Runge-Kutta (RK4) with $s = 16$ integration steps ($\text{NFE} = 64$).
- **Vector Field Network:** Width 512 neurons, 3 continuous layers per block (~2.48M trainable parameters).
- **Environment:** `ant_big_maze` (eval: `ant_big_maze_eval`), `terminate_when_unhealthy: 0`, $\gamma = 0.995$.
- **Training Horizon:** 100 epochs (~98.95M environment steps).

### 2. Key Results
- **Peak Performance (Time-at-Goal):** **172.18**
- **Last-5 Mean Performance:** **113.20 ± 43.25**
- **Peak Success Rate:** **78.1%**
- **Critic InfoNCE Loss:** Dropped from **4.48 → 3.69**
- **Critic Grad Norm:** Stabilized around **2.5 – 3.5**

### 3. Included Artifacts
- `publication_report.png`: 4-panel publication comparison figure against NeurIPS 2025 ResNet baselines.
- `publication_report.pdf`: Vector PDF for LaTeX publication.
- `publication_table.md`: Quantitative benchmark comparison table.
- `metrics.csv`: Full epoch-by-epoch training logs across all 100 epochs (~98.95M steps).
- `vis.html`: Interactive Brax trajectory visualization.
