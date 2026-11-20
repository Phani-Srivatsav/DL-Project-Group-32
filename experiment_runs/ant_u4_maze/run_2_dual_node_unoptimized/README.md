# Run 2: Dual Neural ODE Architecture v1 (Unoptimized)

### 1. Configuration & Hypothesis
- **Critic Architecture:** Continuous-Depth Neural ODE (RK4, $s = 8$, $\text{NFE} = 32$).
- **Actor Architecture:** Continuous-Depth Neural ODE Actor (symmetrical continuous flow).
- **Hypothesis:** Symmetrical continuous depth across both Actor and Critic will resolve the representation–policy mismatch identified in Run 1.

### 2. Experimental Results
- **Quantitative Metric (`time_at_goal`):** Logged `0.00` in `metrics.csv`.
- **Qualitative 3D Trajectory (`vis.html`):** The ant successfully navigated through **3 entire corridors of the U-maze**, closing distance from an initial $14.54\text{m}$ down to $3.53\text{m}$ from the goal!

### 3. Key Scientific Diagnosis & Evaluation Artifacts
1. **Distance Metric Thresholding:** The goal completion radius is strictly defined as $\text{dist} < 0.5\text{m}$. Reaching $3.53\text{m}$ demonstrated that navigation was actively learned, but the quadruped got trapped at the final turn before the threshold.
2. **Brax `EvalWrapper` Permanent Abort Bug:** `terminate_when_unhealthy: 1` was active. When the ant experienced a slight body tilt or roll while turning, Brax permanently ceased accumulating `time_at_goal` for the remainder of the 1000-step episode.
3. **Deterministic Friction Trapping:** Using purely deterministic evaluation (`eval_actor: 0`) caused the quadruped to get caught in local kinematic friction minima against maze corners.
