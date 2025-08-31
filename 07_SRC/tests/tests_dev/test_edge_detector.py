# ==================================================
# =============== TEST: EdgeDetector ===============
# ==================================================
import numpy as np, torch, pytest
from pathlib import Path

from core.config import LayoutConfig, GlobalConfig
from operators.image_io import ImageIO
from operators.edge_detector import edge_detect

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

# ---------- tests ----------
@pytest.mark.parametrize("edge_strategy", ["gradient", "sign_change", "laplacian", "combined"])
@pytest.mark.parametrize("diff_strategy", ["vectorized", "classic"])
def test_numpy_edges(io_numpy, edge_strategy, diff_strategy):
    imgs = _maybe_image_paths()
    if imgs:
        img = io_numpy.read_image(imgs[0], framework="numpy", enable_uid=True)
    else:
        img = _synthetic_hwc(seed=1)

    out = edge_detect(
        img,
        edge_strategy=edge_strategy,
        diff_strategy=diff_strategy,
        processor_strategy="vectorized",
        conv_strategy="gaussian",
        framework="numpy",
        output_format="numpy"
    )
    assert out is not None and isinstance(out, np.ndarray)
    assert out.size > 0
    assert out.shape[-2] > 2 and out.shape[-1] > 2

@pytest.mark.parametrize("edge_strategy", ["gradient", "sobel_gradient", "marr_hildreth", "canny"])
def test_torch_edges(io_torch, edge_strategy):
    imgs = _maybe_image_paths()
    if imgs:
        img = io_torch.read_image(imgs[0], framework="torch", enable_uid=True)
    else:
        img = io_torch.read_image(_synthetic_hwc(seed=2), framework="torch", enable_uid=True)

    out = edge_detect(
        img,
        edge_strategy=edge_strategy,
        diff_strategy="torch",
        processor_strategy="torch",
        conv_strategy="torch",
        framework="torch",
        output_format="numpy"
    )
    assert isinstance(out, np.ndarray)
    assert out.ndim >= 2 and out.shape[-2] > 2 and out.shape[-1] > 2
    
@pytest.mark.parametrize("framework", ["numpy", "torch"]) 
@pytest.mark.parametrize("output_format", ["numpy", "torch"])    
def test_mixed_edges(io, framework, output_format):
    imgs = _maybe_image_paths()
    if imgs:
        img = io.read_image(imgs[0], framework="numpy", enable_uid=True)
    else:
        img = io.read_image(_synthetic_hwc(seed=3), framework="numpy", enable_uid=True)

    out = edge_detect(
        img,
        edge_strategy="gradient",
        diff_strategy="torch" if framework == "torch" else "vectorized",
        processor_strategy="torch" if framework == "torch" else "vectorized",
        conv_strategy="torch" if framework == "torch" else "gaussian",
        framework=framework,
        output_format=output_format
    )
    
    if output_format == "torch":
        assert isinstance(out, torch.Tensor)
    else:
        assert isinstance(out, np.ndarray)
        
    assert out.ndim >= 2 and out.shape[-2] > 2 and out.shape[-1] > 2