# ==================================================
# ============ TEST: Feature Extractor =============
# ==================================================
import numpy as np, pytest, warnings, random
from pathlib import Path
warnings.filterwarnings('ignore')

from core.config import LayoutConfig, GlobalConfig
from operators.image_io import ImageIO
from operators.feature_extractor import feature_extractor

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
    x[h//4: h//2, w//6: w//3] += 0.5
    x[2*h//3: -1, 2*w//3: -1] *= 0.3
    return np.clip(x, 0.0, 1.0)

# ---------- fixtures ----------
@pytest.fixture(scope="module")
def io():
    layout_cfg = LayoutConfig(layout_name="HWC", layout_framework="numpy", layout_ensured_name="HWC")
    global_cfg = GlobalConfig(framework="numpy", output_format="numpy")
    return ImageIO(layout_cfg=layout_cfg, global_cfg=global_cfg)

# ---------- tests ----------
features_1 = ["gaussian", "curvature"] # [features_sequence]
features_2 = [["intensity", "mean", "std", "median", "kurtosis", "skewness"]] # [[features_combo]]
features_3 = [["intensity", "mean", "std", "median", "kurtosis", "skewness"], "gaussian", "curvature"] # [[features_combo], features_sequence]

@pytest.mark.parametrize(
    "features, framework, diff_strategy, processor_strategy, conv_strategy",
    [
        (features_1, "torch", "torch", "torch", "torch"),
        (features_2, "numpy", "vectorized", "vectorized", "gaussian"),
        (features_3, "torch", "torch", "torch", "torch"),
        (["gaussian_eigen"], "numpy", "vectorized", "vectorized", "gaussian"),
        (["morpho_hat"], "numpy", "vectorized", "vectorized", "gaussian"),
        (["fft"], "torch", "torch", "torch", "torch"),
        (["gabor"], "numpy", "parallel", "parallel", "fft"),
        (["lbp"], "torch", "torch", "torch", "torch"),
        # (["glcm"], "numpy", "vectorized", "vectorized", "gaussian"), # Ok but take too long
        ("spectral_entropy", "torch", "torch", "torch", "torch"),
        ("median", "numpy", "classic", "classic", "fft"),
        ("bandpass", "torch", "torch", "torch", "torch"),
    ],
)
def test_feature_extractor(io, features, framework, diff_strategy, processor_strategy, conv_strategy):
    imgs = _maybe_image_paths()
    if imgs:
        img = io.read_image(imgs[0], framework="numpy", enable_uid=True)
    else:
        img = io.read_image(_synthetic_hwc(seed=1), framework="numpy", enable_uid=True)

    out = feature_extractor(
        img,
        features=features,
        diff_strategy=diff_strategy,
        processor_strategy=processor_strategy,
        conv_strategy=conv_strategy,
        framework=framework,
        output_format="numpy",
        layout_name="HWC",
        layout_framework="numpy",
        stack=False,
        combine_features=True,
    )

    if isinstance(out, list):
        assert len(out) >= 1
        for v in out:
            assert isinstance(v, np.ndarray) and v.shape[-2:] == img.shape[-2:]
            assert np.isfinite(v).all()
    else:
        assert isinstance(out, np.ndarray) and out.shape[-2:] == img.shape[-2:]
        assert np.isfinite(out).all()
        
# Combo multiscale (1 block)       
features_4 = {
            "block_1":{
                "comb":["gaussian", "curvature"],
                "param": {"sigma":[5.0, 7.0,]},
                },        
            }

# Sequence multiscale (1 block)
features_5 = {
            "block_1":{
                "seq":["mean", "median", "laplacian"],
                "param": {"sigma":[1.0, 3.0,], "window_size":[3, 5]},
                },        
            }
# Combo and sequence multiscale (1 block)
features_6 = {
            "block_1":{
                "comb":["gaussian", "curvature"],                
                "seq":["mean", "median", "laplacian"],
                "param": {"sigma":[1.0, 3.0,], "window_size":[5,]},
                },        
            } 

# Combo and sequence multiscale (2 blocks)
features_7 = {
            "block_1":{
                "comb":["gaussian", "sobel_edge"],
                "param": {"sigma":[5.0, 7.0,]},
                }, 
            "block_2":{
                "seq":["kurtosis", "skewness"],
                "param": {"sigma":[1.0, 3.0,], "window_size":[3,]},
                },        
            }

# Combo and sequence multiscale (3 blocks)
features_8 = {
            "block_1":{
                "comb":["gradient", "sobel_gradient"],
                "param": {"sigma":[1.0, 3.0,]},
                }, 
            "block_2":{
                "seq":["morpho_hat", "grad_morph"],
                "param": {"sigma":[1.0, 3.0,], "window_size":[3,]},
                },   
            "block_3":{
                "comb":["gaussian", "curvature"],                
                "seq":["mean", "median",],
                "param": {"sigma":[1.0,], "window_size":[3]},
                },        
            }      
        
@pytest.mark.parametrize(
    "features, framework, diff_strategy, processor_strategy, conv_strategy",
    [
        (features_4, "numpy", "vectorized", "vectorized", "gaussian"),
        (features_5, "torch", "torch", "torch", "torch"),
        (features_6, "numpy", "vectorized", "vectorized", "gaussian"),
        (features_7, "torch", "torch", "torch", "torch"),
        (features_8, "numpy", "parallel", "parallel", "fft"),
    ],
)
def test_feature_extractor_block_mode(io, features, framework, diff_strategy, processor_strategy, conv_strategy):
    imgs = _maybe_image_paths()
    if imgs:
        img = io.read_image(imgs[0], framework="numpy", enable_uid=True)
    else:
        img = io.read_image(_synthetic_hwc(seed=1), framework="numpy", enable_uid=True)

    out = feature_extractor(
        img,
        features=features,
        diff_strategy=diff_strategy,
        processor_strategy=processor_strategy,
        conv_strategy=conv_strategy,
        framework=framework,
        output_format="numpy",
        layout_name="HWC",
        layout_framework="numpy",
        stack=False,
        block_mode=True,
        combine_features=True
    )

    if isinstance(out, list):
        assert len(out) >= 1
        for v in out:
            assert isinstance(v, np.ndarray) and v.shape[-2:] == img.shape[-2:]
            assert np.isfinite(v).all()
    else:
        assert isinstance(out, np.ndarray) and out.shape[-2:] == img.shape[-2:]
        assert np.isfinite(out).all()