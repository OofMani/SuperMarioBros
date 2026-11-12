#!/usr/bin/env bash
# Rebuild nes-py 8.2.1 for this machine with:
#   - CPU-specific optimisation (about 12% faster emulation on Apple Silicon)
#   - a small patch adding multiple save states (Snapshot/LoadSnapshot/FreeSnapshot),
#     used by wrappers.py to start some training episodes mid-level
#
# Usage (inside the conda env):  bash tools/build_nes_py.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

python -m pip download -q --no-deps --no-binary :all: nes-py==8.2.1 -d "$WORK"
tar xzf "$WORK"/nes_py-8.2.1.tar.gz -C "$WORK"
patch -d "$WORK/nes_py-8.2.1" -p1 < "$HERE/nes_py_snapshots.patch"

if [[ "$(uname -m)" == "arm64" && "$(uname -s)" == "Darwin" ]]; then
    ARCH_FLAGS="-mcpu=apple-m1"
else
    ARCH_FLAGS="-march=native"
fi
export CFLAGS="-O3 $ARCH_FLAGS -flto -DNDEBUG"
export CXXFLAGS="$CFLAGS"
export LDFLAGS="-flto"

python -m pip install -q --no-deps --no-build-isolation --force-reinstall "$WORK/nes_py-8.2.1"
python -c "from nes_py.nes_env import _LIB; assert hasattr(_LIB, 'Snapshot'); print('nes-py rebuilt with snapshots')"
