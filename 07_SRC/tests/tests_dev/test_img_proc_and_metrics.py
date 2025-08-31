# ==================================================
# ===== TEST: ImageProcessor & Metrics (Dev) =======
# ==================================================
import numpy as np, pytest, warnings, random
from pathlib import Path
warnings.filterwarnings('ignore')

from core.config import LayoutConfig, GlobalConfig, ImageProcessorConfig
from operators.image_processor import ImageProcessor
from operators.image_io import ImageIO
from operators.metrics import MetricEvaluator, SSIM
from operators.image_operator import Operator

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
    x[h//4: h//2, w//6: w//3] += 0.5  # brighter patch
    x[2*h//3: -1, 2*w//3: -1] *= 0.3  # darker patch
    return np.clip(x, 0.0, 1.0)

# ---------- fixtures ----------
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
@pytest.mark.parametrize("fw, strategy", [
    ("numpy", "vectorized"),
    ("torch", "torch"),
])
def test_noising_and_metrics(io_numpy, io_torch, fw, strategy):
    
    if fw == "torch":
        io = io_torch
    else:
        io = io_numpy

    imgs = _maybe_image_paths()
    if imgs:
        img = io.read_image(imgs[0], framework=fw, enable_uid=True)
    else:
        img = io.read_image(_synthetic_hwc(seed=1), framework=fw, enable_uid=True)

    noised = _noising(
        img, sigma=0.1, framework=fw, output_format="numpy", processor_strategy=strategy,
    )

    if not isinstance(img, np.ndarray):
        img = io.to_output(img, framework="numpy", enable_uid=True)
       
    assert isinstance(noised, np.ndarray)    
    assert noised.shape[-2:] == (img.shape[-2], img.shape[-1])
    assert np.isfinite(noised).all()

    evaluator = MetricEvaluator(metrics=["mse", "psnr", "ssim",], return_dict=True)
    evaluator.available_metrics["ssim"] = (SSIM, {"return_map": False})
    
    res = evaluator(img, noised)

    assert isinstance(res, dict) and len(res) >= 3
    assert res.get("psnr", 0.0) < float("inf")