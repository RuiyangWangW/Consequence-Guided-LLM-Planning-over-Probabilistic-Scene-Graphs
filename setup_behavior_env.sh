#!/usr/bin/env bash

# Source this file from an SSH terminal:
#   source ~/safety_filter/setup_behavior_env.sh

# ---------------------------------------------------------------------------
# Installation paths
# ---------------------------------------------------------------------------

export BEHAVIOR_ROOT="/mnt/check/ruiyangw/omnigibson/BEHAVIOR-1K"
export SAFETY_FILTER_ROOT="$HOME/safety_filter"

export CONDA_ENVS_PATH="/mnt/check/ruiyangw/conda/envs"
export CONDA_PKGS_DIRS="/mnt/check/ruiyangw/conda/pkgs"

CONDA_SETUP="$HOME/miniconda3/etc/profile.d/conda.sh"
SIMULATOR_FILE="$BEHAVIOR_ROOT/OmniGibson/omnigibson/simulator.py"

# ---------------------------------------------------------------------------
# Validate required paths
# ---------------------------------------------------------------------------

_setup_error() {
    echo "ERROR: $1" >&2
    return 1 2>/dev/null || exit 1
}

[[ -f "$CONDA_SETUP" ]] \
    || _setup_error "Conda setup file not found: $CONDA_SETUP"

[[ -d "$BEHAVIOR_ROOT" ]] \
    || _setup_error "BEHAVIOR-1K directory not found: $BEHAVIOR_ROOT"

[[ -d "$SAFETY_FILTER_ROOT" ]] \
    || _setup_error "Project directory not found: $SAFETY_FILTER_ROOT"

[[ -f "$SIMULATOR_FILE" ]] \
    || _setup_error "OmniGibson simulator file not found: $SIMULATOR_FILE"

# ---------------------------------------------------------------------------
# Activate the behavior Conda environment
# ---------------------------------------------------------------------------

source "$CONDA_SETUP"

conda activate behavior \
    || _setup_error "Could not activate the behavior Conda environment"

# ---------------------------------------------------------------------------
# CUDA 12.8 / CuRobo
# ---------------------------------------------------------------------------

export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"

export PATH="$CONDA_PREFIX/bin:$PATH"

CUDA_TARGET="$CONDA_PREFIX/targets/x86_64-linux"

export CPATH="$CUDA_TARGET/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_TARGET/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_TARGET/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# RTX A5000 compute capability.
export TORCH_CUDA_ARCH_LIST="8.6"

[[ -x "$CUDACXX" ]] \
    || _setup_error "CUDA compiler not found: $CUDACXX"

if ! "$CUDACXX" --version 2>/dev/null | grep -q "release 12.8"; then
    _setup_error "CuRobo requires CUDA 12.8, but nvcc does not report CUDA 12.8"
fi

# ---------------------------------------------------------------------------
# OmniGibson runtime configuration
# ---------------------------------------------------------------------------

# Use the second RTX A5000.
export OMNIGIBSON_GPU_ID=1

# Run without a local X/GLFW window.
export OMNIGIBSON_HEADLESS=1

# Enable the standalone Isaac Sim WebRTC Streaming Client.
export OMNIGIBSON_REMOTE_STREAMING="webrtc"

# Informational values used by our WebRTC configuration.
export OMNIGIBSON_SERVER_IP="10.237.196.234"
export OMNIGIBSON_SIGNALING_PORT=49100
export OMNIGIBSON_STREAM_PORT=47998

# ---------------------------------------------------------------------------
# Verify the required Isaac Sim 5.1 WebRTC compatibility patch
# ---------------------------------------------------------------------------

grep -q 'enable_extension("omni.kit.livestream.webrtc")' "$SIMULATOR_FILE" \
    || _setup_error "The omni.kit.livestream.webrtc patch is missing"

grep -q 'publicEndpointAddress' "$SIMULATOR_FILE" \
    || _setup_error "The WebRTC publicEndpointAddress patch is missing"

grep -q 'fixedHostPort", 47998' "$SIMULATOR_FILE" \
    || _setup_error "The WebRTC fixedHostPort patch is missing"

grep -q 'publicEndpointPort", 47998' "$SIMULATOR_FILE" \
    || _setup_error "The WebRTC publicEndpointPort patch is missing"

grep -q 'allowDynamicResize", True' "$SIMULATOR_FILE" \
    || _setup_error "The WebRTC allowDynamicResize patch is missing"

# ---------------------------------------------------------------------------
# Enter the project directory
# ---------------------------------------------------------------------------

cd "$SAFETY_FILTER_ROOT" \
    || _setup_error "Could not enter project directory: $SAFETY_FILTER_ROOT"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "BEHAVIOR / OmniGibson environment ready"
echo "  Conda environment: $CONDA_DEFAULT_ENV"
echo "  Python:            $(command -v python)"
echo "  CUDA compiler:     $(command -v nvcc)"
echo "  CUDA version:      12.8"
echo "  BEHAVIOR root:     $BEHAVIOR_ROOT"
echo "  Project directory: $SAFETY_FILTER_ROOT"
echo "  OmniGibson GPU:    $OMNIGIBSON_GPU_ID"
echo "  Headless:          $OMNIGIBSON_HEADLESS"
echo "  Streaming:         $OMNIGIBSON_REMOTE_STREAMING"
echo "  WebRTC server:     $OMNIGIBSON_SERVER_IP"
echo "  Signaling port:    $OMNIGIBSON_SIGNALING_PORT"
echo "  Stream port:       $OMNIGIBSON_STREAM_PORT"
