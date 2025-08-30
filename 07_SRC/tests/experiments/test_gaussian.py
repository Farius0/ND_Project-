
import numpy as np, matplotlib.pyplot as plt, random
from pathlib import Path

from core.config import (LayoutConfig, GlobalConfig,)
from operators.gaussian import conv
from operators.image_io import ImageIO
from utils.nd_tools import show_plane
from utils.decorators import timer

# === Make runner ===
def make_runner(strategy, 
                processor_strategy, 
                input_format="numpy",
                output_format="numpy", 
                dim=2, 
                backend='sequential'):
    @timer(return_result=True, 
           name = "Convolve" + "_" + strategy + "_" + processor_strategy)
    def _runner(img=image, 
                dim=dim, 
                size=None,
                sigma=21.0, 
                angle=0):
        return conv(img, 
                    dim, 
                    size, 
                    sigma, 
                    angle, 
                    framework=input_format, 
                    output_format=output_format, 
                    conv_strategy=strategy, 
                    processor_strategy=processor_strategy,
                    backend=backend,)
    return _runner

if __name__ == "__main__":

    root = Path.cwd().parent.parent.parent / "03_EXAMPLES_DATA" / "Images"
    images_path = sorted([str(p) for p in root.rglob("*.png")])
    rand = random.randint(0, len(images_path) - 2)
    
    layout_cfg = LayoutConfig(
        layout_name="HWC",
        layout_framework="numpy",
        layout_ensured_name="HWC",
    )
    global_cfg = GlobalConfig( 
        framework="numpy",
        output_format="numpy",   
    )

    # ====[ Init ImageIO ]====
    io = ImageIO(
        layout_cfg=layout_cfg,
        global_cfg=global_cfg,
    )

    image = io.read_image(images_path[rand], framework="numpy")  
    
    vectorized_ = make_runner("gaussian", "vectorized")
    classic_ = make_runner("fft", "classic")
    torch_ = make_runner("torch", "torch", input_format="torch", dim=2)
    parallel_ = make_runner("gaussian", "parallel", backend="threading") # sequential, loky

    vectorized_res=vectorized_(image)
    classic_res=classic_(image)
    torch_res=torch_(image)
    parallel_res=parallel_(image)

    # Step 4: Analysis
    diff = np.abs(classic_res[0] - image)
    var_orig = np.var(image)
    var_blur = np.var(classic_res[0])
    hf_orig = np.abs(np.fft.fftshift(np.fft.fftn(image))).mean()
    hf_blur = np.abs(np.fft.fftshift(np.fft.fftn(classic_res[0]))).mean()

    print("\n🧪 Test ND Results:")
    print(f"Mean diff         : {diff.mean():.6f}")
    print(f"Var (orig, blur)  : {var_orig:.6f} → {var_blur:.6f}")
    print(f"HF power (orig → blur): {hf_orig:.3f} → {hf_blur:.3f}")

    assert diff.mean() > 1e-4, "❌ Convolution has no visible effect!"
    assert var_blur < var_orig, "❌ Blur did not reduce variance!"
    assert hf_blur < hf_orig, "❌ High frequencies not attenuated!"

    _, ((a,b,c), (d,e,f), (g,h,i), (j,k,l)) = plt.subplots(4,3, figsize=(12,10))

    show_plane(a, image, title="Ground Truth")
    show_plane(b, classic_res[1], title="Kernel")
    show_plane(c, classic_res[0], title="Blurry image")

    show_plane(d, image, title="Ground Truth")
    show_plane(e, vectorized_res[1], title="Kernel")
    show_plane(f, vectorized_res[0], title="Blurry image")

    show_plane(g, image, title="Ground Truth")
    show_plane(h, torch_res[1], title="Kernel")
    show_plane(i, torch_res[0], title="Blurry image")

    show_plane(j, image, title="Ground Truth")
    show_plane(k, parallel_res[1], title="Kernel")
    show_plane(l, parallel_res[0], title="Blurry image")
