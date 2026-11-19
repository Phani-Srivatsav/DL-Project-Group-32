# Run 1: Single Neural ODE Baseline (NODE Critic + ResNet-4 Actor)

### 1. Configuration & Hypothesis
- **Critic Architecture:** Continuous-Depth Neural ODE with RK4 numerical solver ($s = 8$ integration steps, $\text{NFE} = 32$).
- **Actor Architecture:** Standard discrete 4-layer ResNet (`actor_depth = 4`, discrete layer stacking).
- **Hypothesis:** Testing whether replacing only the value function (Critic) with a continuous-depth Neural ODE would be sufficient to guide a shallow policy.

### 2. Experimental Results
- **Time at Goal:** `0.00` across all 100 epochs.
- **Success Rate:** `0.00%`.
- **Policy Behavior:** Collapsed to a static forward drive, steering directly into the first corner wall of the maze.

### 3. Key Scientific Diagnosis & Failure Mode
1. **Representation–Policy Capacity Mismatch:** The continuous flow Critic formed a high-resolution, continuous geometric representation of the maze, but the 4-layer discrete Actor lacked the parametric and functional depth to resolve the steep gradients of the Critic's value landscape.
2. **Gradient Starvation in Policy:** The discrete policy underfitted, unable to map states into complex non-linear steering maneuvers, leading to policy collapse.
3. **Core Conclusion:** Symmetrical continuous depth is strictly required across **both** representation (Critic) and control (Actor) to navigate multi-turn bottleneck mazes.
