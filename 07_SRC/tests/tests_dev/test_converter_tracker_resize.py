# ==================================================
# ========= TESTS: Converter_Tracker_Resize ========
# ==================================================
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Optional torch import (tests must still run without torch)
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

from core.config import LayoutConfig, GlobalConfig
from operators.image_io import ImageIO
from operators.axis_tracker import AxisTracker

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

def _synthetic_hwc(h=64, w=96, c=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((h, w, c)).astype("float32")
    x = (x
         + np.roll(x, 1, 0) + np.roll(x, -1, 0)
         + np.roll(x, 1, 1) + np.roll(x, -1, 1)) / 5.0
    return x

def device_matches(dev) -> bool:
    """
    Normalize device comparison.
    Accepts 'cpu', 'cuda', 'cuda:0', 'cuda:1', ... as valid device strings.
    """
    s = str(dev)
    if s.startswith("cuda"):
        return True
    return s == "cpu"

# ---------- fixtures ----------
@pytest.fixture(scope="module")
def io_torch():
    layout_cfg = LayoutConfig(layout_name="HWC", layout_framework="numpy", layout_ensured_name="NCHW")
    global_cfg = GlobalConfig(framework="torch", output_format="numpy", add_batch_dim=True)
    return ImageIO(layout_cfg=layout_cfg, global_cfg=global_cfg)

# ---------- tests ----------
@pytest.mark.parametrize("require_data", [True, False])
def test_read_two_images_and_tagging(io_torch: ImageIO, require_data: bool):    
    imgs = _maybe_image_paths()
    if require_data and len(imgs) < 2:
        pytest.skip("No image data on disk; skipping data-dependent test.")

    if require_data:
        img1 = io_torch.read_image(imgs[0], framework="torch", enable_uid=True)
        img2 = io_torch.read_image(imgs[1], framework="torch", enable_uid=True)
    else:
        img1_np = _synthetic_hwc(seed=1)
        img2_np = _synthetic_hwc(seed=2)
        img1 = io_torch.read_image(img1_np, framework="torch", enable_uid=True)
        img2 = io_torch.read_image(img2_np, framework="torch", enable_uid=True)

    assert io_torch.has_tag(img1, "torch")
    assert io_torch.has_tag(img2, "torch")

    tag1 = io_torch.track(img1).get_tag()
    tag2 = io_torch.track(img2).get_tag()
    assert tag1 and tag2

    uid1 = tag1.get("uid")
    uid2 = tag2.get("uid")
    assert uid1 and uid2 and uid1 != uid2

    out1 = io_torch.to_output(img1, framework="numpy", tag_as="output")
    out2 = io_torch.to_output(img2, framework="numpy", tag_as="output")
    assert io_torch.has_tag(out1, "numpy")
    assert io_torch.has_tag(out2, "numpy")
    assert isinstance(out1, np.ndarray) and isinstance(out2, np.ndarray)
    assert out1.ndim in (2, 3) and out2.ndim in (2, 3)


@pytest.mark.parametrize("require_data", [True, False])
def test_axis_tracker_moveaxis_contract_respected(io_torch: ImageIO, require_data: bool):
    imgs = _maybe_image_paths()
    if require_data and imgs is None:
        pytest.skip("No image data available.")

    if require_data:
        img = io_torch.read_image(imgs[0], framework="torch", enable_uid=True)
    else:
        img_np = _synthetic_hwc(seed=1)  # NumPy input
        img = io_torch.read_image(img_np, framework="torch", enable_uid=True)

    tr = AxisTracker(img, operator=io_torch, framework="torch")
    tag_before = tr.get_tag().copy()
    orig_shape = tuple(tr.image.shape)

    tr2 = tr.moveaxis(src=0, dst=-1)
    tag_after = tr2.get_tag()

    assert tag_after is not None
    assert tag_before.get("uid") == tag_after.get("uid") # Identity must remain the same
    assert tr2.image.shape != orig_shape
    assert sorted(tr2.image.shape) == sorted(orig_shape)


@pytest.mark.parametrize("stack", [True, False])
@pytest.mark.parametrize("require_data", [True, False])
def test_load_batch_match_to_first(io_torch: ImageIO,stack: bool, require_data: bool):
    imgs = _maybe_image_paths()
    if require_data and len(imgs) < 2:
        pytest.skip("No image data available for batch tests.")

    if require_data:
        paths = [str(p) for p in imgs[:2]]
        batch = io_torch.load_batch(paths, to="torch", match_to="first", stack=stack)
    else:
        first_np  = _synthetic_hwc(seed=4)  # reference
        second_np = _synthetic_hwc(seed=5)
        batch = io_torch.load_batch([first_np, second_np], to="torch", match_to="first", stack=stack)

    if stack:
        # Expect a batched tensor/array (e.g., (N,C,H,W)); we do not enforce exact layout here.
        assert hasattr(batch, "shape") and len(batch.shape) >= 4
        spatial = batch.shape[-2:]
        assert spatial[0] > 0 and spatial[1] > 0 # Basic sanity: positive spatial dims
    else:
        assert isinstance(batch, list) and len(batch) == 2
        s0, s1 = batch[0].shape, batch[1].shape
        assert s0 == s1
        

@pytest.mark.parametrize("use_cuda", [False, True])
def test_device_preservation(io_torch: ImageIO,use_cuda: bool):
    if use_cuda and not (TORCH_AVAILABLE and CUDA_AVAILABLE):
        pytest.skip("CUDA not available.")

    x_np = _synthetic_hwc(seed=7)  # NumPy input
    timg = io_torch.read_image(x_np, framework="torch", enable_uid=True)

    if TORCH_AVAILABLE:
        if use_cuda:
            timg = timg.to("cuda")
        tout = io_torch.to_output(timg, framework="torch", tag_as="output")
        assert device_matches(tout.device)
        assert tout.dtype == timg.dtype
        assert tout.shape == timg.shape


def test_summary_and_tag_summary_do_not_raise(io_torch: ImageIO):
    imgs = _maybe_image_paths()
    if imgs:
        timg = io_torch.read_image(imgs[0], framework="torch", enable_uid=True)
    else:
        timg = io_torch.read_image(_synthetic_hwc(seed=2), framework="torch", enable_uid=True)

    tr = AxisTracker(timg, operator=io_torch, framework="torch")
    tr.tag_summary()  # must not raise

    out = io_torch.to_output(timg, framework="numpy", tag_as="output")
    io_torch.summary(out, framework="numpy")  # must not raise
