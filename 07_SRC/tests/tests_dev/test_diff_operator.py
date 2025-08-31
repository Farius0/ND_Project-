# ==================================================
# =============== TEST: DiffOperator ===============
# ==================================================
import numpy as np
import torch
import pytest
from pathlib import Path

from core.config import LayoutConfig, GlobalConfig
from operators.image_io import ImageIO
from operators.diff_operator import diffop

# ---------- helpers ----------
def _maybe_image_paths():
    roots = [
        Path.cwd() / "03_EXAMPLES_DATA" / "Images",
        Path.cwd().parent / "03_EXAMPLES_DATA" / "Images",
        Path.cwd().parent.parent / "03_EXAMPLES_DATA" / "Images",
        Path.cwd().parent.parent.parent / "03_EXAMPLES_DATA" / "Images",
    ]
    for r in roots:
        if r.exists():
            imgs = sorted([str(p) for p in r.rglob("*.png")])
            if len(imgs) >= 1:
                return imgs
    return []

def _synthetic_hwc(h=64, w=96, c=3, seed=0, dtype="float32"):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((h, w, c)).astype(dtype)
    # light smoothing to avoid extreme derivatives on random noise
    x = (x
         + np.roll(x, 1, 0) + np.roll(x, -1, 0)
         + np.roll(x, 1, 1) + np.roll(x, -1, 1)) / 5.0
    return x

# ---------- fixtures ----------
@pytest.fixture(scope="session")
def io():
    layout_cfg = LayoutConfig(layout_name="HWC", layout_framework="numpy", layout_ensured_name="HWC")
    global_cfg = GlobalConfig(framework="numpy", output_format="numpy")
    return ImageIO(layout_cfg=layout_cfg, global_cfg=global_cfg)
@pytest.fixture(scope="module")
def io_numpy():
    layout_cfg = LayoutConfig(layout_name="HWC", layout_framework="numpy", layout_ensured_name="HWC")
    global_cfg = GlobalConfig(framework="numpy", output_format="numpy")
    return ImageIO(layout_cfg=layout_cfg, global_cfg=global_cfg)

@pytest.fixture(scope="module")
def io_torch():
    layout_cfg = LayoutConfig(layout_name="HWC", layout_framework="numpy", layout_ensured_name="CHW")
    global_cfg = GlobalConfig(framework="torch", output_format="numpy")
    return ImageIO(layout_cfg=layout_cfg, global_cfg=global_cfg)

# ---------- tests (minimal, robust) ----------
@pytest.mark.parametrize("func", ["gradient", "divergence", "laplacian", "hessian"])
@pytest.mark.parametrize("diff_strategy", ["vectorized", "parallel"])
def test_diff_numpy(io_numpy, func, diff_strategy):
    # read from disk if available, otherwise synthetic fallback
    imgs = _maybe_image_paths()
    if imgs:
        img = io_numpy.read_image(imgs[0], framework="numpy", enable_uid=True)
    else:
        img = _synthetic_hwc(seed=1)

    out = diffop(
        img,
        func=func,
        diff_strategy=diff_strategy,
        framework="numpy",
        output_format="numpy",
        backend="sequential",
    )

    # minimal yet meaningful checks
    assert out is not None, f"{func}: output is None"
    assert isinstance(out, np.ndarray), f"{func}: output must be a numpy ndarray"
    assert out.size > 0, f"{func}: output is empty"

    # Hessian/gradient may add front axes; we only require spatial coherence
    assert out.shape[-2] > 2 and out.shape[-1] > 2, f"{func}: unexpected spatial shape {out.shape}"

@pytest.mark.parametrize("func", ["sobel", "sobel_gradient", "sobel_hessian"])  # one torch-route smoke test
def test_diff_torch(io_torch, func):
    imgs = _maybe_image_paths()
    if imgs:
        img = io_torch.read_image(imgs[0], framework="torch", enable_uid=True)
    else:
        np_img = _synthetic_hwc(seed=2)
        img = io_torch.read_image(np_img, framework="torch", enable_uid=True)

    out = diffop(
        img,
        func=func,
        diff_strategy="torch",
        framework="torch",
        output_format="numpy",
        backend="sequential",
    )

    assert isinstance(out, np.ndarray)
    assert out.ndim >= 2 and out.shape[-2] > 2 and out.shape[-1] > 2

@pytest.mark.parametrize("framework", ["numpy", "torch"]) 
@pytest.mark.parametrize("output_format", ["numpy", "torch"])       
def test_diff_mixed(io, framework, output_format):
    imgs = _maybe_image_paths()
    if imgs:
        img = io.read_image(imgs[0], framework="numpy", enable_uid=True)
    else:
        np_img = _synthetic_hwc(seed=3)
        img = io.read_image(np_img, framework="numpy", enable_uid=True)

    out = diffop(
        img,
        func="sobel",
        diff_strategy="torch" if framework == "torch" else "vectorized",
        framework=framework,
        output_format=output_format,
        backend="sequential",
    )

    if output_format == "torch":
        assert isinstance(out, torch.Tensor)
    else:
        assert isinstance(out, np.ndarray)
        
    assert out.ndim >= 2 and out.shape[-2] > 2 and out.shape[-1] > 2