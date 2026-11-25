# Environment: ant_big_maze

**Task Description:** Continuous-control goal-conditioned locomotion across an extended maze layout with multiple bottleneck corridors.

## Execution
Run on any GPU system:
```bash
bash ../../scripts/train_ant_big_maze.sh
```

## Status
- Training scripts: Configured with continuous-depth Neural ODE architecture.
- **Run 1 (`run_1_breakthrough_dual_node_s16/`):** Completed (~98.95M env steps, 100 epochs). Peak Time-at-Goal **172.18**, Peak Success Rate **78.1%**. See [`run_1_breakthrough_dual_node_s16/README.md`](run_1_breakthrough_dual_node_s16/README.md) for artifacts and metrics.
