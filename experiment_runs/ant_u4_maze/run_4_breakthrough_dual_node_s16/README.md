# Run 4: Scaled Breakthrough Dual Neural ODE Architecture ($s = 16$, $\text{NFE} = 64$)

### 1. Configuration & Mathematical Specification
- **Continuous Flow Solver:** 4th-Order Runge-Kutta (RK4) with $s = 16$ integration steps ($\text{NFE} = 64$).
- **Vector Field Network:** Width 512 neurons, 3 continuous layers per block.
- **Trainable Parameters:** **~2.485 Million parameters** (~42% leaner than ResNet-64's 4.3M parameters).
- **Test-Time Search:** Multi-Sample Guided Critic Search (`eval_actor: 10`), evaluating 10 action candidates per step through the learned continuous flow.
- **Environment:** `ant_u4_maze` (eval: `ant_u4_maze_eval`), `terminate_when_unhealthy: 0`, $\gamma = 0.995$.

### 2. Experimental Status & Convergence
- **Trained Steps:** **81+ Million environment steps** (81 Epochs completed on NVIDIA RTX PRO 6000).
- **Critic InfoNCE Loss:** Dropped from initial **$4.910 \to 3.270$** (~33% reduction).
- **Actor Loss:** Dropped from initial **$4.500 \to 3.187$** (~29% reduction).
- **Gradient Flow Conditioning:** Critic gradient norms stabilized at **$3.8 - 4.3$**, showing well-conditioned integration without exploding or vanishing gradients.

### 3. Included Artifacts
- `publication_report.png`: 4-panel publication-grade comparison figure against the NeurIPS 2025 ResNet baselines (Depths 4, 8, 16, 32, 64).
- `publication_report.pdf`: Vector PDF for LaTeX publication.
- `publication_table.md`: Quantitative summary table comparing effective depth, parameters, and peak metrics.
- `metrics.csv`: Full epoch-by-epoch training logs across all 81+ million steps.
