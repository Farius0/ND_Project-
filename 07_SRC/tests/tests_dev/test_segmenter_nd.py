# ==================================================
# ================= TEST: SegmenterND ==============
# ==================================================
import numpy as np, pytest
from pathlib import Path

from core.config import LayoutConfig, GlobalConfig
from operators.image_io import ImageIO
from operators.segmenter_nd import segmenter_nd

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

def _synthetic_hwc(h=96, w=128, c=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((h, w, c), dtype=np.float32)
    x[h//4: h//2, w//6: w//3] += 0.5 # light rectangle
    x[2*h//3: -1, 2*w//3: -1] *= 0.3 # dark rectangle
    return np.clip(x, 0.0, 1.0)

# ---------- fixtures ----------
@pytest.fixture(scope="session")
def io():
    layout_cfg = LayoutConfig(layout_name="HWC", layout_framework="numpy", layout_ensured_name="HWC")
    global_cfg = GlobalConfig(framework="numpy", output_format="numpy")
    return ImageIO(layout_cfg=layout_cfg, global_cfg=global_cfg)

@pytest.mark.parametrize("framework", ["numpy", "torch"])
@pytest.mark.parametrize(
    "mode, kwargs",
    [
        ("kmeans",          dict(num_classes=4, use_channels=True)),
        ("otsu",            dict()),
        ("entropy",         dict()),
        ("iterative",       dict()),
        ("multi",           dict(multi_thresholds=[0.33, 0.66])),

    ],
)
def test_segmenter_modes(io, mode, framework, kwargs):
    imgs = _maybe_image_paths()
    if imgs:
        img = io.read_image(imgs[0], framework="numpy", enable_uid=True)
    else:
        img = io.read_image(_synthetic_hwc(seed=1), framework="numpy", enable_uid=True)

    out = segmenter_nd(
        img,
        segmenter_mode=mode,
        framework=framework,
        output_format="numpy",
        layout_name="HWC",
        layout_framework="numpy",
        layout_ensured_name="HWC" if framework == "numpy" else "CHW",
        processor_strategy="vectorized" if framework == "numpy" else "torch",
        **kwargs,
    )

    assert isinstance(out, np.ndarray)
    assert out.shape[:2] == img.shape[:2]
    assert np.isfinite(out).all()
    assert np.unique(out).size > 1, f"{mode} produced a constant output"