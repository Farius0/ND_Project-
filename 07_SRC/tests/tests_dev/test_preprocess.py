# ==================================================
# ================= TEST: Preprocessor =============
# ==================================================
import numpy as np, pytest
from pathlib import Path

from operators.preprocessor import preprocess

# -------- helpers --------
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
    x = rng.standard_normal((h, w, c)).astype("float32")
    x = (x
         + np.roll(x, 1, 0) + np.roll(x, -1, 0)
         + np.roll(x, 1, 1) + np.roll(x, -1, 1)) / 5.0
    return x

# ---------- tests ----------
@pytest.mark.parametrize("flags", [
    dict(normalize=True),
    dict(stretch=True),
    dict(equalize=True),
    dict(denoise=True),
    dict(gamma_correct=True),
    dict(local_contrast=True),
])
def test_numpy_preprocess_smoke(flags):
    imgs = _maybe_image_paths()
    path_or_img = imgs[0] if imgs else _synthetic_hwc(seed=1)
    
    img = preprocess(
        path_or_img,
        processor_strategy="parallel",
        framework="numpy",
        output_format="numpy",
        layout_name="HWC",
        layout_framework="numpy",
        layout_ensured_name="HWC",
    )
    
    out = preprocess(
        path_or_img,
        processor_strategy="parallel",
        framework="numpy",
        output_format="numpy",
        layout_name="HWC",
        layout_framework="numpy",
        layout_ensured_name="HWC",
        **flags,
    )

    assert isinstance(out, np.ndarray)
    assert out.shape[-2:] == img.shape[-2:], "Spatial shape must be preserved"
    assert np.isfinite(out).all(), "Output must be finite"

@pytest.mark.parametrize("case", [
    dict(name="torch_stretch_norm_denoise", stretch=True, normalize=True, denoise=False, equalize=False, local_contrast=False, clip=True),
    dict(name="torch_norm_only", normalize=True, clip=True),
    dict(name="torch_raw", normalize=False),
])
def test_torch_preprocess_minimal(case):
    imgs = _maybe_image_paths()
    path_or_img = imgs[0] if imgs else _synthetic_hwc(seed=1)
    out = preprocess(
        path_or_img,
        processor_strategy="torch",
        framework="torch",
        output_format="numpy",
        layout_name="HWC",
        layout_framework="numpy",
        layout_ensured_name="CHW",
        **{k:v for k,v in case.items() if k != "name"},
    )

    assert isinstance(out, np.ndarray)
    assert out.ndim >= 2 and out.shape[-2] > 2 and out.shape[-1] > 2
    assert np.isfinite(out).all()

    if case.get("normalize") or case.get("stretch"):
        mn, mx = float(out.min()), float(out.max())
        assert mn >= -1e-3 and mx <= 1.0 + 1e-3, f"Expected roughly [0,1] range, got [{mn:.4g}, {mx:.4g}]"