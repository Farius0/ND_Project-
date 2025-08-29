# ==================================================
# ========= TESTS: Converter_Tracker_Resize ========
# ==================================================
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

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

# Project imports (architecture remains unchanged)
from core.config import LayoutConfig, GlobalConfig
from operators.image_io import ImageIO
from utils.axis_tracker import AxisTracker


# ===================
# Helpers
# ===================

def _make_np(shape: Tuple[int, ...], seed: int = 123) -> np.ndarray:
    """
    Create a deterministic NumPy array with random values.
    A fixed seed ensures test reproducibility across runs.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=shape).astype("float32")


def device_matches(dev) -> bool:
    """
    Normalize device comparison.
    Accepts 'cpu', 'cuda', 'cuda:0', 'cuda:1', ... as valid device strings.
    """
    s = str(dev)
    if s.startswith("cuda"):
        return True
    return s == "cpu"


# ===================
# Fixtures
# ===================

@pytest.fixture(scope="session")
def images_root() -> Path:
    """
    Locate an images directory containing PNG files.
    Tests that require data will skip if none is found.
    """
    candidates = [
        Path.cwd() / "03_EXAMPLES_DATA" / "Images",
        Path.cwd().parent / "03_EXAMPLES_DATA" / "Images",
        Path.cwd().parent.parent / "03_EXAMPLES_DATA" / "Images",
        Path.cwd().parent.parent.parent / "03_EXAMPLES_DATA" / "Images",
    ]
    for p in candidates:
        if p.exists() and any(p.rglob("*.png")):
            return p
    return candidates[0]


@pytest.fixture(scope="session")
def image_paths(images_root: Path) -> List[Path]:
    """
    Return all PNG image paths under images_root, sorted.
    """
    return sorted(images_root.rglob("*.png")) if images_root.exists() else []


@pytest.fixture(scope="session")
def layout_cfg() -> LayoutConfig:
    """
    Input contract: source images are NumPy in HWC/NHWC.
    We do not enforce any layout in tests; we only read/observe tags.
    """
    return LayoutConfig(
        layout_name="HWC",
        layout_framework="numpy",
        layout_ensured_name="NCHW",  # not enforced here; only referenced by the project
    )


@pytest.fixture(scope="session")
def global_cfg() -> GlobalConfig:
    """
    Internal work in torch, output in numpy, batch dim added by default.
    """
    return GlobalConfig(
        framework="torch",
        output_format="numpy",
        add_batch_dim=True,
    )


@pytest.fixture(scope="session")
def io(layout_cfg: LayoutConfig, global_cfg: GlobalConfig) -> ImageIO:
    """
    Instantiate ImageIO with provided configs.
    """
    return ImageIO(layout_cfg=layout_cfg, global_cfg=global_cfg)


# ===================
# Tests (NumPy input only — first tag defines the contract)
# ===================

@pytest.mark.parametrize("require_data", [True, False])
def test_read_two_images_and_tagging(io: ImageIO, image_paths: List[Path], require_data: bool):
    """
    Input is always NumPy (input contract). ImageIO performs the first tag (contract) and internal conversion.
    Verifies:
      - tags exist and contain distinct UIDs,
      - conversion to numpy output is stable and shape is sensible.
    """
    if require_data and len(image_paths) < 2:
        pytest.skip("No image data on disk; skipping data-dependent test.")

    if require_data:
        # Paths accepted; ImageIO still performs the first tagging/contract.
        img1 = io.read_image(str(image_paths[0]), framework="torch", enable_uid=True)
        img2 = io.read_image(str(image_paths[1]), framework="torch", enable_uid=True)
    else:
        # Pure NumPy inputs (HWC), deterministic via seeds.
        img1_np = _make_np((32, 32, 3), seed=1)
        img2_np = _make_np((48, 24, 3), seed=2)
        img1 = io.read_image(img1_np, framework="torch", enable_uid=True)
        img2 = io.read_image(img2_np, framework="torch", enable_uid=True)

    assert io.has_tag(img1, "torch")
    assert io.has_tag(img2, "torch")

    tag1 = io.track(img1).get_tag()
    tag2 = io.track(img2).get_tag()
    assert tag1 and tag2

    uid1 = tag1.get("uid")
    uid2 = tag2.get("uid")
    assert uid1 and uid2 and uid1 != uid2

    out1 = io.to_output(img1, framework="numpy", tag_as="output")
    out2 = io.to_output(img2, framework="numpy", tag_as="output")
    assert isinstance(out1, np.ndarray) and isinstance(out2, np.ndarray)
    assert out1.ndim in (2, 3) and out2.ndim in (2, 3)


@pytest.mark.parametrize("require_data", [True, False])
def test_axis_tracker_moveaxis_contract_respected(io: ImageIO, image_paths: List[Path], require_data: bool):
    """
    Validate that AxisTracker.moveaxis preserves tag readability and identity (UID).
    No layout/axes policy is enforced by this test; it only observes behavior.
    """
    if require_data and not image_paths:
        pytest.skip("No image data available.")

    if require_data:
        img = io.read_image(str(image_paths[0]), framework="torch", enable_uid=True)
    else:
        img_np = _make_np((32, 32, 3), seed=3)  # NumPy input
        img = io.read_image(img_np, framework="torch", enable_uid=True)

    tr = AxisTracker(img, operator=io, framework="torch")
    tag_before = tr.get_tag().copy()
    orig_shape = tuple(tr.image.shape)

    # Example move: (C,H,W) -> (H,W,C) or any valid permutation depending on current shape.
    tr2 = tr.moveaxis(src=0, dst=-1)
    tag_after = tr2.get_tag()

    assert tag_after is not None
    # Identity must remain the same
    assert tag_before.get("uid") == tag_after.get("uid")
    # Shape changed by permutation but retains the same factors
    assert tr2.image.shape != orig_shape
    assert sorted(tr2.image.shape) == sorted(orig_shape)


@pytest.mark.parametrize("stack", [True, False])
@pytest.mark.parametrize("require_data", [True, False])
def test_load_batch_match_to_first(io: ImageIO, image_paths: List[Path], stack: bool, require_data: bool):
    """
    The first element defines the reference when match_to='first'.
    Inputs are NumPy arrays or paths; the operator handles conversion/resize internally.
    """
    if require_data and len(image_paths) < 2:
        pytest.skip("No image data available for batch tests.")

    if require_data:
        paths = [str(p) for p in image_paths[:2]]
        batch = io.load_batch(paths, to="torch", match_to="first", stack=stack)
    else:
        first_np  = _make_np((64, 48, 3), seed=4)  # reference
        second_np = _make_np((32, 32, 3), seed=5)
        batch = io.load_batch([first_np, second_np], to="torch", match_to="first", stack=stack)

    if stack:
        # Expect a batched tensor/array (e.g., (N,C,H,W)); we do not enforce exact layout here.
        assert hasattr(batch, "shape") and len(batch.shape) >= 4
        spatial = batch.shape[-2:]
        # Basic sanity: positive spatial dims
        assert spatial[0] > 0 and spatial[1] > 0
    else:
        assert isinstance(batch, list) and len(batch) == 2
        s0, s1 = batch[0].shape, batch[1].shape
        assert s0 == s1


@pytest.mark.parametrize("use_cuda", [False, True])
def test_device_preservation(io: ImageIO, use_cuda: bool):
    """
    Validate torch->torch path preserves dtype/shape and uses a valid device.
    Device comparison is tolerant: 'cuda' and 'cuda:0' are both accepted as CUDA.
    """
    if use_cuda and not (TORCH_AVAILABLE and CUDA_AVAILABLE):
        pytest.skip("CUDA not available.")

    x_np = _make_np((3, 32, 32), seed=6)  # NumPy input
    timg = io.read_image(x_np, framework="torch", enable_uid=True)

    if TORCH_AVAILABLE:
        if use_cuda:
            timg = timg.to("cuda")
        tout = io.to_output(timg, framework="torch", tag_as="output")
        assert device_matches(tout.device)
        assert tout.dtype == timg.dtype
        assert tout.shape == timg.shape


def test_summary_and_tag_summary_do_not_raise(io: ImageIO):
    """
    Sanity check: summary() and AxisTracker.tag_summary() must not raise.
    No windows should be opened (tests are non-interactive).
    """
    x_np = _make_np((32, 32, 3), seed=7)
    timg = io.read_image(x_np, framework="torch", enable_uid=True)

    tr = AxisTracker(timg, operator=io, framework="torch")
    tr.tag_summary()  # must not raise

    out = io.to_output(timg, framework="numpy", tag_as="output")
    io.summary(out, framework="numpy")  # must not raise
