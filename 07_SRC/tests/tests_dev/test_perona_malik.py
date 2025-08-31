# ==================================================
# ============== TEST: Perona–Malik ================
# ==================================================
import numpy as np, torch, pytest
from pathlib import Path

from core.config import LayoutConfig, GlobalConfig, ImageProcessorConfig
from operators.image_io import ImageIO
from operators.image_operator import Operator
from operators.image_processor import ImageProcessor
from operators.metrics import PSNR
from algorithms.perona_malik import pm

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
    x = rng.standard_normal((h, w, c)).astype("float32")
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x
# ---------- fixtures ----------
@pytest.fixture(scope="session")
def io():
    layout_cfg = LayoutConfig(layout_name="HWC", layout_framework="numpy", layout_ensured_name="HWC")
    global_cfg = GlobalConfig(framework="numpy", output_format="numpy")
    return ImageIO(layout_cfg=layout_cfg, global_cfg=global_cfg)

# ---------- noise via ImageProcessor ----------
def _noising(img, sigma=0.1, framework="numpy", output_format="numpy", processor_strategy="vectorized"):

    func = lambda x: Operator(
        x,
        clip=True,
        layout_cfg=LayoutConfig(layout_name="HWC", layout_framework="numpy"),
        global_cfg=GlobalConfig(framework=framework, output_format=output_format),
    ).noise(sigma=sigma)

    proc = ImageProcessor(
        img_process_cfg=ImageProcessorConfig(function=func, processor_strategy=processor_strategy),
        layout_cfg=LayoutConfig(layout_name="HWC", layout_framework="numpy"),
        global_cfg=GlobalConfig(framework=framework, output_format=output_format),
    )
    return proc(img)

# -------------------- tests --------------------
@pytest.mark.parametrize(
    "variant, framework, diff_strategy, conv_strategy, processor_strategy",
    [
        ("pm", "numpy", "vectorized", "gaussian", "vectorized"),
        ("enhanced", "numpy", "vectorized", "gaussian", "vectorized"),
        ("pm", "numpy", "classic", "fft", "classic"),
        ("enhanced", "torch", "torch", "torch", "torch"),
        ("pm", "torch", "torch", "torch", "torch"),
    ],
)
def test_pm_effect(io, variant, framework, diff_strategy, conv_strategy, processor_strategy):
    imgs = _maybe_image_paths()
    if imgs:
        img = io.read_image(imgs[0], framework="numpy", enable_uid=True)
    else:
        img = io.read_image(_synthetic_hwc(seed=3), framework="numpy", enable_uid=True)

    noisy = _noising(img, sigma=0.1, framework=framework, output_format="numpy", processor_strategy=processor_strategy)

    denoised = pm(
        noisy,
        algorithm=variant,
        framework=framework,
        output_format="numpy",
        layout_name="HWC",
        layout_framework="numpy",
        diff_strategy=diff_strategy,
        conv_strategy=conv_strategy,
        processor_strategy=processor_strategy,
        sigma=1.0,
        steps=10,
        dt=0.04,
        alpha=0.1,
        disable_tqdm=True,
    )

    assert isinstance(denoised, np.ndarray) and denoised.shape[-2:] == img.shape[-2:]
    assert np.isfinite(denoised).all()

    psnr_noisy = PSNR(img, noisy)
    psnr_deno  = PSNR(img, denoised)
    assert psnr_deno >= psnr_noisy - 1e-4, f"Expected denoised PSNR >= noisy PSNR, got {psnr_deno:.3f} vs {psnr_noisy:.3f}"