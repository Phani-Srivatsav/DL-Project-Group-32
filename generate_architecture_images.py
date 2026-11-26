#!/usr/bin/env python3
"""
Generate publication-quality slide/academic architecture diagrams matching the
visual style of the user's reference slide (NeurIPS presentation style):
- Left side: Clean neural network / dynamical flow diagrams with circular nodes & layers.
- Right side: Mathematical formulation in LaTeX typography.
- Bottom: Figure caption and presentation footer.
"""

import os
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow, FancyArrowPatch

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs('/tmp/matplotlib_cache', exist_ok=True)

OUT_DIR = "/home/phani.gande.ug24/scaling-crl/architecture_diagrams"
os.makedirs(OUT_DIR, exist_ok=True)

# Color Palette matching modern academic slides
BEAMER_RED = "#8b1538"
SLATE_DARK = "#1a1a2e"
NODE_BLUE = "#1f77b4"
NODE_ORANGE = "#ff7f0e"
NODE_PURPLE = "#9467bd"
NODE_GREEN = "#2ca02c"
EDGE_COLOR = "#7f7f7f"
BG_BOX = "#f8f9fa"
BORDER_BOX = "#ced4da"

def draw_slide_template(fig, ax, slide_title, slide_num, total_slides=5):
    """Draws top banner, bottom footer, and slide framing."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Top Beamer-style banner
    top_bar = patches.Rectangle((0, 96), 100, 4, facecolor=BEAMER_RED, edgecolor='none')
    ax.add_patch(top_bar)
    ax.text(50, 98, "Continuous-Depth Neural ODEs for Self-Supervised RL", 
            color='white', fontsize=12, fontweight='bold', ha='center', va='center')
    
    # Bottom footer
    bot_bar = patches.Rectangle((0, 0), 100, 4, facecolor=BEAMER_RED, edgecolor='none')
    ax.add_patch(bot_bar)
    ax.text(10, 2, "Group 32 | Deep Learning Project", color='white', fontsize=10, ha='left', va='center')
    ax.text(50, 2, slide_title, color='white', fontsize=10, ha='center', va='center')
    ax.text(95, 2, f"{slide_num} / {total_slides}", color='white', fontsize=10, ha='right', va='center')

def draw_network_layer(ax, x, y_center, num_nodes, node_color=NODE_BLUE, radius=1.4, spacing=4.0):
    """Draws a vertical column of neural network nodes and returns their center coordinates."""
    y_starts = [y_center + (i - (num_nodes - 1) / 2.0) * spacing for i in range(num_nodes)]
    coords = []
    for y in y_starts:
        circle = Circle((x, y), radius, facecolor='white', edgecolor=node_color, linewidth=2.0, zorder=5)
        ax.add_patch(circle)
        coords.append((x, y))
    return coords

def connect_layers(ax, layer1_coords, layer2_coords, edge_color=EDGE_COLOR, alpha=0.45, lw=1.0):
    """Draws weighted edges connecting two neural network layers."""
    for (x1, y1) in layer1_coords:
        for (x2, y2) in layer2_coords:
            ax.plot([x1, x2], [y1, y2], color=edge_color, alpha=alpha, linewidth=lw, zorder=3)


# ==============================================================================
# SLIDE 1: Discrete ResNet vs Continuous Neural ODE
# ==============================================================================
def make_slide_1():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    draw_slide_template(fig, ax, "Discrete ResNet vs. Continuous Neural ODE Flow", 1)
    
    # Left Side: Diagrams
    # Sub-box 1: Discrete ResNet (Stacked Layers)
    rect1 = FancyBboxPatch((4, 52), 44, 40, boxstyle="round,pad=0.8", facecolor="#fcfcfc", edgecolor=BORDER_BOX, lw=1.5)
    ax.add_patch(rect1)
    ax.text(6, 88, "(A) Discrete ResNet-L Architecture (Wang et al., NeurIPS 2025)", fontsize=11, fontweight='bold', color=SLATE_DARK)
    
    l1 = draw_network_layer(ax, 10, 68, 4, NODE_BLUE)
    l2 = draw_network_layer(ax, 18, 68, 4, NODE_BLUE)
    l3 = draw_network_layer(ax, 26, 68, 4, NODE_BLUE)
    l4 = draw_network_layer(ax, 34, 68, 4, NODE_BLUE)
    l5 = draw_network_layer(ax, 42, 68, 1, NODE_BLUE)
    connect_layers(ax, l1, l2)
    connect_layers(ax, l2, l3)
    connect_layers(ax, l3, l4)
    connect_layers(ax, l4, l5)
    
    # Residual skip arrows
    arrow1 = FancyArrowPatch((10, 78), (26, 78), connectionstyle="arc3,rad=-0.4", color=NODE_BLUE, lw=1.8, arrowstyle="->")
    arrow2 = FancyArrowPatch((18, 78), (34, 78), connectionstyle="arc3,rad=-0.4", color=NODE_BLUE, lw=1.8, arrowstyle="->")
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    ax.text(22, 55, r"Fixed Discrete Layers: $L \in \{4, 8, 16, 32, 64\}$", fontsize=9.5, ha='center', color="#444")
    
    # Sub-box 2: Continuous Neural ODE
    rect2 = FancyBboxPatch((4, 8), 44, 40, boxstyle="round,pad=0.8", facecolor="#fcfcfc", edgecolor=BORDER_BOX, lw=1.5)
    ax.add_patch(rect2)
    ax.text(6, 44, "(B) Continuous-Depth Neural ODE Flow (Ours, RK4)", fontsize=11, fontweight='bold', color=SLATE_DARK)
    
    l_in = draw_network_layer(ax, 10, 24, 4, NODE_BLUE)
    # Continuous Vector Field box in middle
    ode_box = FancyBboxPatch((18, 15), 18, 18, boxstyle="round,pad=0.5", facecolor="#eef5fb", edgecolor=NODE_BLUE, lw=2.0)
    ax.add_patch(ode_box)
    ax.text(27, 26, r"Continuous Flow", fontsize=10, fontweight='bold', ha='center', color=NODE_BLUE)
    ax.text(27, 22, r"$\frac{dh(t)}{dt} = f_\theta(h(t), t)$", fontsize=11, ha='center', color=SLATE_DARK)
    ax.text(27, 18, r"$t \in [0, 1]$", fontsize=9.5, ha='center', color="#555")
    
    l_out = draw_network_layer(ax, 42, 24, 1, NODE_ORANGE)
    connect_layers(ax, l_in, [(18, 24)], edge_color=NODE_BLUE)
    arrow_flow = FancyArrowPatch((13, 24), (18, 24), color=NODE_BLUE, lw=2, arrowstyle="->")
    arrow_out = FancyArrowPatch((36, 24), (41, 24), color=NODE_ORANGE, lw=2, arrowstyle="->")
    ax.add_patch(arrow_flow)
    ax.add_patch(arrow_out)
    
    ax.text(27, 11, r"Depth is Integration Steps: NFE = $4 \times s$", fontsize=9.5, ha='center', color="#444")
    
    # Right Side: Mathematical Formulations
    ax.text(54, 87, r"Discrete ResNet Layer Formulation:", fontsize=13, fontweight='bold', color=SLATE_DARK)
    ax.text(54, 78, r"$h_{k+1} = h_k + f(h_k; W_k), \quad k \in \{0, 1, \dots, L-1\}$", fontsize=12, color=SLATE_DARK)
    ax.text(54, 71, r"$\bullet$ Weight Matrices: Dedicated $W_k$ for every layer $k$", fontsize=11, color="#333")
    ax.text(54, 65, r"$\bullet$ Total Parameters: $\mathcal{O}(L \cdot d^2)$ (Linear growth with depth)", fontsize=11, color="#333")
    ax.text(54, 59, r"$\bullet$ Discrete Step: Fixed discrete leap, Euler-approximation error", fontsize=11, color="#333")
    
    ax.plot([52, 96], [51, 51], color="#e0e0e0", lw=1.5)
    
    ax.text(54, 43, r"Continuous Neural ODE Formulation:", fontsize=13, fontweight='bold', color=SLATE_DARK)
    ax.text(54, 34, r"$\frac{dh(t)}{dt} = f_\theta(h(t), t), \quad h(1) = h(0) + \int_0^1 f_\theta(h(t), t) \, dt$", fontsize=12, color=SLATE_DARK)
    ax.text(54, 27, r"$\bullet$ Vector Field: Single shared parameter set $\theta$ across all depth", fontsize=11, color="#333")
    ax.text(54, 21, r"$\bullet$ Total Parameters: $\mathcal{O}(d^2)$ (Constant regardless of depth!)", fontsize=11, color=NODE_GREEN, fontweight='bold')
    ax.text(54, 15, r"$\bullet$ Continuous Flow: Decoupled depth via RK4 numerical solver steps", fontsize=11, color="#333")
    
    # Caption
    ax.text(50, 5, r"Figure 1: Mathematical and architectural comparison between Discrete ResNet and Continuous-Depth Neural ODE.", 
            fontsize=10.5, ha='center', color=BEAMER_RED, style='italic', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_discrete_resnet_vs_continuous_neural_ode.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/01_discrete_resnet_vs_continuous_neural_ode.pdf")
    plt.close()
    print("Generated Slide 1")


# ==============================================================================
# SLIDE 2: Dual Neural ODE Architecture (Actor + Critic)
# ==============================================================================
def make_slide_2():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    draw_slide_template(fig, ax, "Dual Neural ODE Architecture: Actor-Critic Co-Design", 2)
    
    # Left Box: Actor & Critic Flows
    rect_arch = FancyBboxPatch((4, 9), 45, 83, boxstyle="round,pad=0.8", facecolor="#fcfcfc", edgecolor=BORDER_BOX, lw=1.5)
    ax.add_patch(rect_arch)
    ax.text(6, 88, "Dual Continuous-Depth System", fontsize=12, fontweight='bold', color=SLATE_DARK)
    
    # Critic State-Action Flow
    ax.text(6, 82, r"1. Critic Flow: $\phi_\theta(s, a)$", fontsize=10.5, fontweight='bold', color=NODE_BLUE)
    c1 = draw_network_layer(ax, 8, 73, 3, NODE_BLUE, radius=1.1, spacing=3.0)
    box_c1 = FancyBboxPatch((15, 68), 16, 10, boxstyle="round,pad=0.3", facecolor="#eef5fb", edgecolor=NODE_BLUE, lw=1.5)
    ax.add_patch(box_c1)
    ax.text(23, 73, r"ODE $f_{\theta_{sa}}$ (RK4)", fontsize=9.5, ha='center', va='center', fontweight='bold', color=NODE_BLUE)
    c2 = draw_network_layer(ax, 38, 73, 2, NODE_BLUE, radius=1.1, spacing=3.0)
    ax.add_patch(FancyArrowPatch((11, 73), (15, 73), color=NODE_BLUE, lw=1.5, arrowstyle="->"))
    ax.add_patch(FancyArrowPatch((31, 73), (35, 73), color=NODE_BLUE, lw=1.5, arrowstyle="->"))
    ax.text(43, 73, r"$\phi(s, a)$", fontsize=11, color=SLATE_DARK, va='center')
    
    # Critic Goal Flow
    ax.text(6, 61, r"2. Critic Goal Flow: $\psi_\theta(g)$", fontsize=10.5, fontweight='bold', color=NODE_PURPLE)
    g1 = draw_network_layer(ax, 8, 52, 2, NODE_PURPLE, radius=1.1, spacing=3.0)
    box_g1 = FancyBboxPatch((15, 47), 16, 10, boxstyle="round,pad=0.3", facecolor="#f6effa", edgecolor=NODE_PURPLE, lw=1.5)
    ax.add_patch(box_g1)
    ax.text(23, 52, r"ODE $f_{\theta_g}$ (RK4)", fontsize=9.5, ha='center', va='center', fontweight='bold', color=NODE_PURPLE)
    g2 = draw_network_layer(ax, 38, 52, 2, NODE_PURPLE, radius=1.1, spacing=3.0)
    ax.add_patch(FancyArrowPatch((11, 52), (15, 52), color=NODE_PURPLE, lw=1.5, arrowstyle="->"))
    ax.add_patch(FancyArrowPatch((31, 52), (35, 52), color=NODE_PURPLE, lw=1.5, arrowstyle="->"))
    ax.text(43, 52, r"$\psi(g)$", fontsize=11, color=SLATE_DARK, va='center')
    
    # Dot Product Interaction
    ax.text(23, 38, r"Contrastive Energy: $\mathcal{E}(s, a, g) = \langle \phi(s, a), \psi(g) \rangle$", 
            fontsize=10.5, ha='center', fontweight='bold', color=BEAMER_RED)
    ax.plot([6, 44], [34, 34], color="#e0e0e0", lw=1.0)
    
    # Actor Flow
    ax.text(6, 29, r"3. Continuous Actor: $\pi_\theta(a|s, g)$", fontsize=10.5, fontweight='bold', color=NODE_ORANGE)
    a1 = draw_network_layer(ax, 8, 19, 4, NODE_ORANGE, radius=1.1, spacing=3.0)
    box_act = FancyBboxPatch((15, 14), 16, 10, boxstyle="round,pad=0.3", facecolor="#fff6ed", edgecolor=NODE_ORANGE, lw=1.5)
    ax.add_patch(box_act)
    ax.text(23, 19, r"ODE $f_{\theta_\pi}$ (RK4)", fontsize=9.5, ha='center', va='center', fontweight='bold', color=NODE_ORANGE)
    a2 = draw_network_layer(ax, 38, 19, 2, NODE_ORANGE, radius=1.1, spacing=3.0)
    ax.add_patch(FancyArrowPatch((11, 19), (15, 19), color=NODE_ORANGE, lw=1.5, arrowstyle="->"))
    ax.add_patch(FancyArrowPatch((31, 19), (35, 19), color=NODE_ORANGE, lw=1.5, arrowstyle="->"))
    ax.text(43, 19, r"$\mu, \sigma^2$", fontsize=11, color=SLATE_DARK, va='center')
    
    # Right Side: Objectives & Formulas
    ax.text(53, 87, "Symmetric InfoNCE Objective (Contrastive Critic):", fontsize=12, fontweight='bold', color=SLATE_DARK)
    ax.text(53, 76, r"$\mathcal{L}_{\text{NCE}} = -\frac{1}{B}\sum_{i=1}^B \log \frac{\exp(\phi(s_i, a_i)^\top \psi(g_i))}{\sum_{j=1}^B \exp(\phi(s_i, a_i)^\top \psi(g_j))}$", fontsize=11.5, color=SLATE_DARK)
    ax.text(53, 67, r"$+ \lambda_{\text{reg}} \cdot \left(\log \sum_{j=1}^B \exp(\phi(s_i, a_i)^\top \psi(g_j))\right)^2$", fontsize=11.5, color=BEAMER_RED)
    ax.text(53, 60, r"$\bullet$ $\lambda_{\text{reg}} = 0.1$: LogSumExp regularizer bounds partition function", fontsize=10.5, color="#444")
    
    ax.plot([51, 96], [54, 54], color="#e0e0e0", lw=1.5)
    
    ax.text(53, 47, "Actor Policy Loss (Energy Minimization):", fontsize=12, fontweight='bold', color=SLATE_DARK)
    ax.text(53, 38, r"$\mathcal{L}_{\text{actor}} = \mathbb{E}_{a \sim \pi} \left[ \alpha \log \pi(a|s, g) - \phi(s, a)^\top \psi(g) \right]$", fontsize=11.5, color=SLATE_DARK)
    ax.text(53, 29, "Adaptive Temperature Optimization (SAC):", fontsize=12, fontweight='bold', color=SLATE_DARK)
    ax.text(53, 20, r"$\mathcal{L}(\alpha) = \mathbb{E} \left[ -\alpha \cdot (\log \pi(a|s, g) + \mathcal{H}_{\text{target}}) \right]$", fontsize=11.5, color=SLATE_DARK)
    ax.text(53, 12, r"$\bullet$ $\mathcal{H}_{\text{target}} = -0.7 \times \text{action\_dim} = -5.6$ (Preserves turning exploration)", fontsize=10.5, color="#444")
    
    ax.text(50, 5, r"Figure 2: Architecture of the Dual Neural ODE showing symmetric continuous critic encoders and continuous actor.", 
            fontsize=10.5, ha='center', color=BEAMER_RED, style='italic', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_dual_neural_ode_architecture.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/02_dual_neural_ode_architecture.pdf")
    plt.close()
    print("Generated Slide 2")


# ==============================================================================
# SLIDE 3: Numerical ODE Solver Dynamics: RK4 vs Forward Euler
# ==============================================================================
def make_slide_3():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    draw_slide_template(fig, ax, "Numerical Solver Dynamics: RK4 vs. Forward Euler", 3)
    
    # Left Box: Vector field and trajectory comparison
    rect_box = FancyBboxPatch((4, 9), 44, 83, boxstyle="round,pad=0.8", facecolor="#fcfcfc", edgecolor=BORDER_BOX, lw=1.5)
    ax.add_patch(rect_box)
    ax.text(6, 88, "Solver Trajectory Integration in Maze", fontsize=12, fontweight='bold', color=SLATE_DARK)
    
    # Draw a maze corner obstacle
    obstacle = patches.Rectangle((22, 40), 6, 35, facecolor="#bbb", edgecolor="#666", lw=1.5)
    ax.add_patch(obstacle)
    ax.text(25, 57, "Maze Wall", color='white', fontweight='bold', ha='center', va='center', rotation=90)
    
    # Draw vector field arrows
    for xi in [10, 16, 32, 38]:
        for yi in [25, 45, 65]:
            ax.add_patch(FancyArrowPatch((xi, yi), (xi+2, yi+3), color="#cfd8dc", lw=1.0, arrowstyle="->"))
            
    # Trajectory 1: Euler (Drifting into wall) -> Red dashed
    t_euler = np.linspace(0, 1, 20)
    x_euler = 12 + 10 * t_euler
    y_euler = 30 + 15 * t_euler + 5 * t_euler**2
    ax.plot([12, 17, 22.5], [30, 42, 57], 'o--', color="#d32f2f", lw=2.2, label="Euler (Drifts into Wall, Run 2)")
    ax.scatter([22.5], [57], color="#d32f2f", s=90, marker='x', zorder=10)
    ax.text(22.5, 60, "Collision!", color="#d32f2f", fontweight='bold', ha='center')
    
    # Trajectory 2: RK4 (Smooth curve around wall) -> Green solid
    ax.plot([12, 16, 21, 28, 36], [30, 36, 34, 38, 55], 's-', color="#2e7d32", lw=2.5, label="RK4 (Exact Obstacle Avoidance, Run 4)")
    ax.text(36, 58, "Goal Reached", color="#2e7d32", fontweight='bold', ha='center')
    
    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.85), fontsize=9.2, frameon=True, framealpha=0.9)
    ax.text(26, 13, "Run 2 (Euler): Score ~1.5 | Run 4 (RK4): Score 269.34", fontsize=9.5, ha='center', fontweight='bold', color=SLATE_DARK)
    
    # Right Side: Mathematical Equations for RK4
    ax.text(52, 87, "4th-Order Runge-Kutta (RK4) Integration Scheme:", fontsize=12.5, fontweight='bold', color=SLATE_DARK)
    
    ax.text(54, 78, r"Given step size $\Delta t = \frac{1}{s}$, the state advances via 4 slope stages:", fontsize=11, color="#333")
    
    # 4 stages
    ax.text(56, 70, r"$k_1 = f_\theta(h_n, t_n)$", fontsize=11.5, color=NODE_BLUE)
    ax.text(56, 63, r"$k_2 = f_\theta\left(h_n + \frac{\Delta t}{2} k_1, \, t_n + \frac{\Delta t}{2}\right)$", fontsize=11.5, color=NODE_BLUE)
    ax.text(56, 56, r"$k_3 = f_\theta\left(h_n + \frac{\Delta t}{2} k_2, \, t_n + \frac{\Delta t}{2}\right)$", fontsize=11.5, color=NODE_BLUE)
    ax.text(56, 49, r"$k_4 = f_\theta\left(h_n + \Delta t \, k_3, \, t_n + \Delta t\right)$", fontsize=11.5, color=NODE_BLUE)
    
    ax.text(54, 39, r"Weighted 4th-Order Integration Update:", fontsize=12, fontweight='bold', color=SLATE_DARK)
    ax.text(56, 31, r"$h_{n+1} = h_n + \frac{\Delta t}{6} \left( k_1 + 2k_2 + 2k_3 + k_4 \right)$", fontsize=12, color=BEAMER_RED)
    
    ax.plot([52, 96], [25, 25], color="#e0e0e0", lw=1.5)
    
    ax.text(54, 19, r"Theoretical Error Analysis:", fontsize=11.5, fontweight='bold', color=SLATE_DARK)
    ax.text(54, 13, r"$\bullet$ Forward Euler (Run 2): Local truncation error $\mathcal{O}(\Delta t^2)$", fontsize=10.5, color="#555")
    ax.text(54, 8, r"$\bullet$ Runge-Kutta 4 (Run 4): Local truncation error $\mathcal{O}(\Delta t^5)$ (High precision)", fontsize=10.5, color=NODE_GREEN, fontweight='bold')
    
    ax.text(50, 4.5, r"Figure 3: Runge-Kutta 4 (RK4) integration mechanism vs. 1st-order Euler drift in constrained maze navigation.", 
            fontsize=10.5, ha='center', color=BEAMER_RED, style='italic', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_ode_solver_mechanics_rk4_vs_euler.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/03_ode_solver_mechanics_rk4_vs_euler.pdf")
    plt.close()
    print("Generated Slide 3")


# ==============================================================================
# SLIDE 4: Self-Supervised Geometric Goal Relabeling
# ==============================================================================
def make_slide_4():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    draw_slide_template(fig, ax, "Self-Supervised Geometric Goal Relabeling in Replay Buffer", 4)
    
    # Left Box: Trajectory Timeline
    rect_box = FancyBboxPatch((4, 9), 45, 83, boxstyle="round,pad=0.8", facecolor="#fcfcfc", edgecolor=BORDER_BOX, lw=1.5)
    ax.add_patch(rect_box)
    ax.text(6, 88, "Trajectory Uniform Replay Queue", fontsize=12, fontweight='bold', color=SLATE_DARK)
    
    # Draw trajectory states along a line
    y_line = 65
    ax.plot([8, 44], [y_line, y_line], color="#bbb", lw=3.0, zorder=1)
    
    states = [(10, "$s_t$"), (18, "$s_{t+1}$"), (26, "$s_{t+\Delta t_1}$"), (34, "$s_{t+\Delta t_2}$"), (42, "$s_{t+\Delta t_3}$")]
    for x, label in states:
        c = Circle((x, y_line), 2.2, facecolor='white', edgecolor=NODE_BLUE, lw=2.0, zorder=5)
        ax.add_patch(c)
        ax.text(x, y_line, label, fontsize=9.5, ha='center', va='center')
        
    # Sampling arcs
    arc1 = FancyArrowPatch((10, y_line+3), (26, y_line+3), connectionstyle="arc3,rad=-0.5", color=NODE_ORANGE, lw=2.0, arrowstyle="->")
    arc2 = FancyArrowPatch((10, y_line+3), (42, y_line+3), connectionstyle="arc3,rad=-0.7", color=NODE_GREEN, lw=2.0, arrowstyle="->")
    ax.add_patch(arc1)
    ax.add_patch(arc2)
    ax.text(18, 77, r"Short Horizon ($\gamma=0.995$)", fontsize=9.5, color=NODE_ORANGE, ha='center')
    ax.text(26, 84, r"Full Maze Horizon ($\gamma=0.998$)", fontsize=9.5, color=NODE_GREEN, ha='center')
    
    # Maze corridors schematic
    ax.text(6, 48, "Maze Corridor Traversal Coverage:", fontsize=11, fontweight='bold', color=SLATE_DARK)
    
    # Corridor blocks
    c_colors = ["#c8e6c9", "#a5d6a7", "#81c784", "#66bb6a"]
    for i, c_name in enumerate(["Corridor 1", "Corridor 2", "Corridor 3", "Corridor 4"]):
        c_box = FancyBboxPatch((7 + i*9.5, 34), 8.5, 9, boxstyle="round,pad=0.2", facecolor=c_colors[i], edgecolor="#388e3c", lw=1.2)
        ax.add_patch(c_box)
        ax.text(11.25 + i*9.5, 38.5, c_name, fontsize=8.5, ha='center', va='center', fontweight='bold', color="#1b5e20")
        
    ax.plot([7, 25.5], [30, 30], color=NODE_ORANGE, lw=3.0)
    ax.text(16, 26, r"$\gamma=0.995$ covers Corridors 1-2 ($H \approx 200$)", fontsize=9.5, color=NODE_ORANGE, ha='center', fontweight='bold')
    
    ax.plot([7, 44.5], [20, 20], color=NODE_GREEN, lw=3.0)
    ax.text(26, 16, r"$\gamma=0.998$ covers Corridors 1-4 ($H \approx 500$)", fontsize=9.5, color=NODE_GREEN, ha='center', fontweight='bold')
    
    # Right Side: Mathematical Relabeling Formulation
    ax.text(53, 87, "Geometric Future State Goal Relabeling:", fontsize=12.5, fontweight='bold', color=SLATE_DARK)
    ax.text(53, 78, r"In buffer.py, future states are sampled with geometric distribution:", fontsize=11, color="#333")
    
    ax.text(55, 68, r"$\mathcal{P}(\text{Goal} = s_{t+\Delta t} \mid s_t) = (1 - \gamma) \cdot \gamma^{\Delta t}, \quad \Delta t \geq 1$", fontsize=12, color=BEAMER_RED)
    
    ax.text(53, 58, "Effective Planning Horizon:", fontsize=12, fontweight='bold', color=SLATE_DARK)
    ax.text(55, 49, r"$H_{\text{eff}} = \mathbb{E}[\Delta t] = \frac{1}{1 - \gamma}$", fontsize=13, color=SLATE_DARK)
    
    ax.plot([51, 96], [42, 42], color="#e0e0e0", lw=1.5)
    
    ax.text(53, 34, "Quantitative Horizon Comparison:", fontsize=12, fontweight='bold', color=SLATE_DARK)
    ax.text(55, 26, r"$\bullet$ Baseline $\gamma = 0.995 \to H_{\text{eff}} \approx \frac{1}{0.005} = 200 \text{ steps}$", fontsize=11, color="#333")
    ax.text(58, 20, r"$\to$ Stalls at 3rd corner; insufficient distant goal signal", fontsize=10, color="#666")
    ax.text(55, 13, r"$\bullet$ Tuned $\gamma = 0.998 \to H_{\text{eff}} \approx \frac{1}{0.002} = 500 \text{ steps}$", fontsize=11, color=NODE_GREEN, fontweight='bold')
    ax.text(58, 7, r"$\to$ Samples 4th corridor goals 2.5$\times$ more frequently across full maze", fontsize=10, color=NODE_GREEN)
    
    ax.text(50, 4.5, r"Figure 4: Geometric goal relabeling mechanism in replay buffer and effective planning horizon scaling.", 
            fontsize=10.5, ha='center', color=BEAMER_RED, style='italic', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_contrastive_goal_relabeling_and_planning.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/04_contrastive_goal_relabeling_and_planning.pdf")
    plt.close()
    print("Generated Slide 4")


# ==============================================================================
# SLIDE 5: Code Pipeline & Test-Time Guided Search
# ==============================================================================
def make_slide_5():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    draw_slide_template(fig, ax, "Code Pipeline & Test-Time Contrastive Energy Guidance", 5)
    
    # Left Box: Flowchart
    rect_box = FancyBboxPatch((4, 9), 45, 83, boxstyle="round,pad=0.8", facecolor="#fcfcfc", edgecolor=BORDER_BOX, lw=1.5)
    ax.add_patch(rect_box)
    ax.text(6, 88, "Execution Pipeline in train_dual_node.py", fontsize=12, fontweight='bold', color=SLATE_DARK)
    
    # 4 Flowchart stages
    stages = [
        (75, "1. Environment Rollout", "Brax JAX Engine (512 Parallel Envs)", "#e3f2fd", NODE_BLUE),
        (56, "2. Replay Buffer Queue", "Geometric Relabeling (10,000 Cap)", "#fff3e0", NODE_ORANGE),
        (37, "3. Continuous Optimization", "Dual RK4 ODE SGD Step (800 Batches)", "#f3e5f5", NODE_PURPLE),
        (18, "4. Test-Time Contrastive Search", "eval_actor=10 (Critic Gradient Steering)", "#e8f5e9", NODE_GREEN)
    ]
    
    for y, title, subtitle, bg, border in stages:
        s_box = FancyBboxPatch((8, y-5), 37, 10, boxstyle="round,pad=0.4", facecolor=bg, edgecolor=border, lw=1.5)
        ax.add_patch(s_box)
        ax.text(10, y+2, title, fontsize=10.5, fontweight='bold', color=border)
        ax.text(10, y-2, subtitle, fontsize=9.5, color="#444")
        if y > 20:
            arr = FancyArrowPatch((26, y-5), (26, y-11), color="#999", lw=1.5, arrowstyle="->")
            ax.add_patch(arr)
            
    # Right Side: Test-Time Search Mathematical Formulation
    ax.text(53, 87, "Test-Time Contrastive Guided Search (eval_actor = 10):", fontsize=12, fontweight='bold', color=SLATE_DARK)
    ax.text(53, 78, r"At test time, the policy samples $K=10$ candidate actions from continuous actor:", fontsize=11, color="#333")
    
    ax.text(55, 69, r"$\{a^{(1)}, a^{(2)}, \dots, a^{(K)}\} \sim \pi_\theta(\cdot \mid s_t, g)$", fontsize=12, color=SLATE_DARK)
    
    ax.text(53, 59, "Critic Contrastive Energy Ranking:", fontsize=12, fontweight='bold', color=SLATE_DARK)
    ax.text(55, 49, r"$a^* = \arg\max_{a^{(k)}} \, \phi_\theta(s_t, a^{(k)})^\top \psi_\theta(g)$", fontsize=13, color=BEAMER_RED)
    
    ax.plot([51, 96], [42, 42], color="#e0e0e0", lw=1.5)
    
    ax.text(53, 34, "Why Guided Search Solves Maze Bottlenecks:", fontsize=12, fontweight='bold', color=SLATE_DARK)
    ax.text(53, 26, r"$\bullet$ Direct policy rollouts occasionally pick actions driving toward walls", fontsize=10.5, color="#444")
    ax.text(53, 20, r"$\bullet$ The continuous critic acts as an energy function over state-action space", fontsize=10.5, color="#444")
    ax.text(53, 14, r"$\bullet$ Scoring candidates through the continuous critic steers the quadruped", fontsize=10.5, color=NODE_GREEN, fontweight='bold')
    ax.text(55, 8, "around sharp 90-degree corners, boosting Time-at-Goal from ~20 to 269.34!", fontsize=10.5, color=NODE_GREEN, fontweight='bold')
    
    ax.text(50, 4.5, r"Figure 5: Training pipeline execution flow and test-time contrastive energy minimization guidance.", 
            fontsize=10.5, ha='center', color=BEAMER_RED, style='italic', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_test_time_guided_search_pipeline.png", dpi=300)
    plt.savefig(f"{OUT_DIR}/05_test_time_guided_search_pipeline.pdf")
    plt.close()
    print("Generated Slide 5")

def zip_artifacts():
    zip_path = "/home/phani.gande.ug24/scaling-crl/architecture_diagrams.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUT_DIR):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, os.path.dirname(OUT_DIR))
                zf.write(abs_path, rel_path)
    print(f"Packed all diagrams into: {zip_path}")

if __name__ == "__main__":
    make_slide_1()
    make_slide_2()
    make_slide_3()
    make_slide_4()
    make_slide_5()
    zip_artifacts()
