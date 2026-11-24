# Run 5: Tuned Breakthrough Dual Neural ODE ($s=24$, $\text{NFE}=96$)

## Overview & Hypothesis
Run 5 builds directly upon the success of **Run 4 (Score 269.34)** by systematically tuning four key mathematical hyperparameters while strictly maintaining 100% fidelity to the author's benchmark protocol (NeurIPS 2025 "1000 Layer Networks for Self-Supervised RL"):
1. **Extended ODE Integration Depth ($s=24$, $\text{NFE}=96$):**
   - In Run 4, $s=16$ RK4 steps ($\text{NFE}=64$) enabled the policy to break into the 3rd and 4th corridors.
   - Increasing to $s=24$ RK4 steps ($\text{NFE}=96$) increases continuous representational depth by 50% without adding a single trainable parameter, allowing sharper vector-field velocity modeling around 90-degree maze turns.
2. **Longer Horizon Geometric Goal Relabeling ($\gamma = 0.998$):**
   - With the baseline $\gamma = 0.995$, the effective planning horizon is $H_{\text{eff}} \approx \frac{1}{1 - \gamma} = 200$ steps.
   - Because the 4-turn maze requires 700–850 locomotion steps to navigate all corridors, $\gamma=0.998$ extends $H_{\text{eff}} \approx 500$ steps, substantially increasing the sampling probability of distant goals in the final corridor.
3. **Exploration Entropy Regularization ($\text{entropy\_param} = 0.7$):**
   - Increases target entropy $\mathcal{H}_{\text{target}} = -0.7 \times 8 = -5.6$ (vs $-4.0$), preventing the policy from collapsing into deterministic wall-friction traps.
4. **Gradient Clipping ($\text{grad\_clip\_norm} = 1.0$):**
   - Clips the global norm of backpropagated gradients through the 24 RK4 integration steps to eliminate gradient spikes during high-friction wall contact.

---

## Configuration Comparison

| Hyperparameter | Run 4 (Baseline Breakthrough) | Run 5 (Tuned Setup) | Rationale |
| :--- | :---: | :---: | :--- |
| **ODE Solver** | Runge-Kutta 4 (RK4) | Runge-Kutta 4 (RK4) | High 4th-order trajectory precision |
| **Integration Steps ($s$)** | $16$ | **$24$** | 50% deeper vector field integration |
| **Model NFE** | $64$ | **$96$** | Deeper representation without parameter bloat |
| **Goal Discount ($\gamma$)** | $0.995$ ($H \approx 200$) | **$0.998$ ($H \approx 500$)** | Samples distant corridor goals 2.5x more often |
| **Entropy Parameter** | $0.5$ ($\mathcal{H} = -4.0$) | **$0.7$ ($\mathcal{H} = -5.6$)** | Preserves exploratory pivoting at corners |
| **Gradient Clipping** | $0.0$ (Off) | **$1.0$** | Stabilizes deep ODE backpropagation |
| **Network Width** | $512$ | $512$ | Same high-capacity MLP vector field |
| **Total Env Steps** | $125,000,000$ | $125,000,000$ | Identical benchmark budget |
| **Eval Actor Mode** | `10` (Guided Search) | `10` (Guided Search) | Test-time contrastive energy minimization |

---

## Execution Instructions

### Via SLURM (HPC Cluster):
```bash
sbatch run_dual_node_s24.slurm
```

### Locally / Interactively:
```bash
bash run.sh
```

### Monitoring:
```bash
tail -f slurm_logs/slurm-<JOBID>.out
```
