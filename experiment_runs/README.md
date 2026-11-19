# Experimental Runs & Results Directory

This directory contains the organized results, metrics, visualizations, and diagnostic reports for each experimental iteration:

| Run Directory | Architecture | Key Characteristics | Outcome / Status |
| :--- | :--- | :--- | :--- |
| **[`run_1_single_node_baseline`](run_1_single_node_baseline/)** | NODE Critic + ResNet-4 Actor | $s=8$, NFE=32, Width 256 | Capacity mismatch; policy underfitted ($0.0$ score) |
| **[`run_2_dual_node_unoptimized`](run_2_dual_node_unoptimized/)** | Dual NODE v1 | Symmetrical NODE flow, $s=8$ | Reached 3 corridors ($14.5\text{m} \to 3.5\text{m}$); Brax metric cutoff identified |
| **[`run_3_dual_node_optimized_s8`](run_3_dual_node_optimized_s8/)** | Dual NODE v2 | $s=8$, NFE=32, Unhealthy abort disabled | **Peak 26.14** (surpassed paper's ResNet-4 peak of ~18.0) |
| **[`run_4_breakthrough_dual_node_s16`](run_4_breakthrough_dual_node_s16/)** | Scaled Dual NODE | $s=16$, NFE=64, Width 512, Guided Search | **81M+ steps trained**; 42% leaner than ResNet-64; stable loss ($3.27$) |
