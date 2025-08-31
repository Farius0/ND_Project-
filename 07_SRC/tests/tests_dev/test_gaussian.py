# ==================================================
# ============== TEST: Gaussian Convolution ========
# ==================================================
import numpy as np, torch, pytest
from pathlib import Path

from core.config import LayoutConfig, GlobalConfig
from operators.image_io import ImageIO
from operators.gaussian import conv

# ---------- helpers ----------
def _maybe_image_paths():
    roots = [
        Path.cwd() / '03_EXAMPLES_DATA' / 'Images',
        Path.cwd().parent / '03_EXAMPLES_DATA' / 'Images',
        Path.cwd().parent.parent / '03_EXAMPLES_DATA' / 'Images',
        Path.cwd().parent.parent.parent / "03_EXAMPLES_DATA" / "Images",        
    ]
    for r in roots:
        if r.exists():
            imgs = sorted([str(p) for p in r.rglob('*.png')])
            if len(imgs) >= 1:
                return imgs
    return []

def _synthetic_hwc(h=96, w=128, c=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((h, w, c)).astype('float32')
    x = (x + np.roll(x, 1, 0) + np.roll(x, -1, 0) + np.roll(x, 1, 1) + np.roll(x, -1, 1)) / 5.0
    return x

# ---------- fixtures ----------
@pytest.fixture(scope="session")
def io():
    layout_cfg = LayoutConfig(layout_name="HWC", layout_framework="numpy", layout_ensured_name="HWC")
    global_cfg = GlobalConfig(framework="numpy", output_format="numpy")
    return ImageIO(layout_cfg=layout_cfg, global_cfg=global_cfg)

@pytest.fixture(scope='module')
def io_numpy():
    layout_cfg = LayoutConfig(layout_name='HWC', layout_framework='numpy', layout_ensured_name='HWC')
    global_cfg = GlobalConfig(framework='numpy', output_format='numpy')
    return ImageIO(layout_cfg=layout_cfg, global_cfg=global_cfg)

@pytest.fixture(scope='module')
def io_torch():
    layout_cfg = LayoutConfig(layout_name='HWC', layout_framework='numpy', layout_ensured_name='CHW')
    global_cfg = GlobalConfig(framework='torch', output_format='numpy')
    return ImageIO(layout_cfg=layout_cfg, global_cfg=global_cfg)

# -------------------- tests --------------------
@pytest.mark.parametrize('conv_strategy', ['fft', 'gaussian'])
@pytest.mark.parametrize('processor_strategy', ['classic', 'parallel'])
def test_numpy_conv_outputs_and_effect(io_numpy, conv_strategy, processor_strategy):
    imgs = _maybe_image_paths()
    if imgs:
        img = io_numpy.read_image(imgs[0], framework='numpy', enable_uid=True)
    else:
        img = _synthetic_hwc(seed=1)

    result, kernel = conv(
        img=img, dim=2, size=None, sigma=3.0, angle=0.0,
        framework='numpy', output_format='numpy', backend='sequential',
        conv_strategy=conv_strategy, processor_strategy=processor_strategy
    )

    assert isinstance(result, np.ndarray) and isinstance(kernel, np.ndarray)
    assert result.shape[-2:] == img.shape[-2:] and result.ndim == img.ndim
    assert kernel.ndim == 2

    var_orig = float(np.var(img))
    var_blur = float(np.var(result))
    hf_orig  = float(np.abs(np.fft.fftshift(np.fft.fftn(img))).mean())
    hf_blur  = float(np.abs(np.fft.fftshift(np.fft.fftn(result))).mean())

    assert var_blur < var_orig, 'Blur should reduce variance'
    assert hf_blur < hf_orig, 'Blur should attenuate high frequencies'

def test_torch_conv_route(io_torch):
    imgs = _maybe_image_paths()
    if imgs:
        img = io_torch.read_image(imgs[0], framework='torch', enable_uid=True)
    else:
        img = io_torch.read_image(_synthetic_hwc(seed=2), framework='torch', enable_uid=True)

    result, kernel = conv(
        img=img, dim=2, size=None, sigma=2.0, angle=0.0,
        framework='torch', output_format='numpy', backend='sequential', conv_strategy='torch', 
        processor_strategy='torch', layout_framework='torch', layout_name='CHW',
    )

    assert isinstance(result, np.ndarray) and result.ndim >= 3
    assert isinstance(kernel, np.ndarray) and kernel.ndim == 2
    assert result.shape[-2] > 2 and result.shape[-1] > 2

@pytest.mark.parametrize("framework", ["numpy", "torch"]) 
@pytest.mark.parametrize("output_format", ["numpy", "torch"])      
def test_mixed_conv_route(io, framework, output_format):
    imgs = _maybe_image_paths()
    if imgs:
        img = io.read_image(imgs[0], framework='numpy', enable_uid=True)
    else:
        img = io.read_image(_synthetic_hwc(seed=3), framework='numpy', enable_uid=True)

    result, kernel = conv(
        img=img, 
        dim=2, 
        size=None, 
        sigma=2.0, 
        angle=0.0,
        framework=framework, 
        output_format=output_format, 
        backend='sequential',
        conv_strategy='torch' if framework == "torch" else "gaussian", 
        processor_strategy='torch' if framework == "torch" else "vectorized"
    )

    if output_format == "torch":
        assert isinstance(result, torch.Tensor) and result.ndim >= 3
        assert isinstance(kernel, torch.Tensor) and kernel.ndim == 2
    else:
        assert isinstance(result, np.ndarray) and result.ndim >= 3
        assert isinstance(kernel, np.ndarray) and kernel.ndim == 2
            
    assert result.shape[-2] > 2 and result.shape[-1] > 2