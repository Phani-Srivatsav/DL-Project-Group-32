# Experimental Runs & Results Directory

Each folder in this directory is fully self-contained, including the exact Python training code, Slurm cluster submission script, evaluation module, replay buffer, execution metrics (`metrics.csv`), visualization dashboards, and comprehensive post-run diagnostic reports.

| Run Directory | Architecture | Key Characteristics | Exact Code & Slurm Script | Outcome / Status |
| :--- | :--- | :--- | :--- | :--- |
| **[`run_1_single_node_baseline`](run_1_single_node_baseline/)** | NODE Critic + ResNet-4 Actor | $s=4$, NFE=16, Width 256 | `train_node.py`<br>`run_node_100m.slurm` | Capacity mismatch; policy underfitted ($0.0$ score) |
| **[`run_2_dual_node_unoptimized`](run_2_dual_node_unoptimized/)** | Dual NODE v1 (Actor + Critic) | Euler solver, $s=4$, Width 256 | `train_dual_node.py`<br>`run_dual_node_euler_s4.slurm` | Reached 3 corridors ($14.5\text{m} \to 3.5\text{m}$); Brax metric cutoff identified |
| **[`run_3_dual_node_optimized_s8`](run_3_dual_node_optimized_s8/)** | Dual NODE v2 (Optimized) | RK4 solver, $s=8$, NFE=32, Width 512 | `train_dual_node.py`<br>`run_dual_node_rk4_s8.slurm` | **Peak 26.14** (surpassed paper's ResNet-4 peak of ~18.0) |
| **[`run_4_breakthrough_dual_node_s16`](run_4_breakthrough_dual_node_s16/)** | Scaled Breakthrough Dual NODE | RK4 solver, $s=16$, NFE=64, Width 512, Guided Search | `train_dual_node_breakthrough.py`<br>`run_dual_node_breakthrough.slurm` | **82M+ steps trained**; 42% leaner than ResNet-64; stable loss ($3.27$) |
