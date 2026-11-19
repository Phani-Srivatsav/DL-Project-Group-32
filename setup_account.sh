#!/bin/bash
# ==============================================================================
# Turnkey Setup Script for New HPC Slurm Accounts
# Sets up uv, Python 3.10 virtual environment, JAX with CUDA 12, and dependencies
# ==============================================================================

set -e

echo "=== [1/4] Checking Python & UV Package Manager ==="
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "=== [2/4] Setting Up Virtual Environment (.venv) ==="
uv venv --python 3.10
source .venv/bin/activate

echo "=== [3/4] Installing JAX with CUDA 12 & Dependencies ==="
uv pip install -U "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv pip install \
    flax==0.8.4 \
    brax==0.10.1 \
    mujoco==3.1.5 \
    optax==0.2.2 \
    tyro==0.8.4 \
    wandb==0.16.6 \
    wandb_osh==1.2.0 \
    matplotlib==3.10.0 \
    pandas \
    scipy

echo "=== [4/4] Patching Brax 0.10.1 Contact Bug ==="
BRAX_CONTACT=$(find .venv -name "contact.py" | grep "brax/base/contact.py" || true)
if [ -n "$BRAX_CONTACT" ]; then
    echo "Applying Brax contact fix to $BRAX_CONTACT"
    cat << 'EOF' > "$BRAX_CONTACT"
from typing import Optional
from brax import math
from brax.base import Contact
from brax.base import System
from brax.base import Transform
import jax
from jax import numpy as jp
from mujoco import mjx

def get(sys: System, x: Transform) -> Optional[Contact]:
    data = mjx.make_data(sys)
    if data.ncon == 0:
        return None
    @jax.vmap
    def local_to_global(pos1, quat1, pos2, quat2):
        pos = pos1 + math.rotate(pos2, quat1)
        mat = math.quat_to_3x3(math.quat_mul(quat1, quat2))
        return pos, mat
    x = x.concatenate(Transform.zero((1,)))
    xpos = x.pos[sys.geom_bodyid - 1]
    xquat = x.rot[sys.geom_bodyid - 1]
    geom_xpos, geom_xmat = local_to_global(
        xpos, xquat, sys.geom_pos, sys.geom_quat
    )
    d = data.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)
    d = mjx.collision(sys, d)
    c = d.contact
    elasticity = (sys.elasticity[c.geom1] + sys.elasticity[c.geom2]) * 0.5
    body1 = jp.array(sys.geom_bodyid)[c.geom1] - 1
    body2 = jp.array(sys.geom_bodyid)[c.geom2] - 1
    link_idx = (body1, body2)
    return Contact(elasticity=elasticity, link_idx=link_idx, **c.__dict__)
EOF
fi

echo "=== ENVIRONMENT SETUP COMPLETE ==="
echo "You can now launch training with:"
echo "  sbatch run_ant_big_maze.slurm"
echo "  # or"
echo "  sbatch run_arm_push_hard.slurm"
