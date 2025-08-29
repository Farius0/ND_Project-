# ==================================================
# ======== TEST: ImageProcessor & Metrics ==========
# ==================================================

import warnings; warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt, random
from pathlib import Path

from core.config import LayoutConfig, GlobalConfig, ImageProcessorConfig
from operators.image_operator import Operator as operator
from operators.image_processor import ImageProcessor
from operators.image_io import ImageIO 
from operators.metrics import MetricEvaluator, SSIM
from utils.decorators import timer

# === Fonction à chronométrer ===
def noising(img, 
            sigma, 
            framework="numpy", 
            output_format="numpy", 
            layout_framework="numpy",
            layout_name="HWC",
            processor_strategy='vectorized', 
            backend='sequential', 
            ):
    
    # ====[ Fallback ]====
    processor_strategy=processor_strategy or "vectorized" if framework == "numpy" else "torch"   
        
    # ====[ Configuration ]====
    proc_params = {"processor_strategy": processor_strategy,}
    layout_params = {"layout_name": layout_name, "layout_framework": layout_framework}
    global_params = {"framework": framework, "output_format": output_format, "backend": backend}    
    
    func = lambda x: operator(x, 
                            clip=True, 
                            layout_cfg=LayoutConfig(**layout_params),
                            global_cfg=GlobalConfig(**global_params),
                            ).noise(sigma=sigma)
    
    processor = ImageProcessor(
                            img_process_cfg = ImageProcessorConfig(function=func, **proc_params,),
                            layout_cfg=LayoutConfig(**layout_params),
                            global_cfg=GlobalConfig(**global_params),
                            )
    
    return processor(img)

def make_runner(processor_strategy, 
                name, 
                input_format="numpy", 
                output_format="numpy", 
                backend='sequential', 
                ):
    @timer(return_result=True, name=name)
    def _runner(img=image, sigma=0.1):
        return noising(img, sigma, 
                       processor_strategy=processor_strategy, 
                       framework=input_format, 
                       output_format=output_format,
                       backend=backend, 
                       )
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

    vectorized_ = make_runner("vectorized", "noising_vectorized")
    classic_ = make_runner("classic", "noising_classic")
    torch_ = make_runner("torch", "noising_torch", input_format="torch")
    parallel_ = make_runner("parallel", "noising_parallel", backend="threading") # sequential, loky

    # vectorized_res=vectorized_(image)
    # classic_res=classic_(image)
    torch_res=torch_(image)
    # parallel_res=parallel_(image)

    # === Metric Evaluation ===

    evaluator = MetricEvaluator(metrics=["mse", "mae", "rmse", "nrmse", "psnr", "ssim", "ms-ssim", "lpips"],
            return_dict=True
        )

    evaluator.available_metrics["ssim"] = (SSIM, {"return_map": False})

    results = evaluator(image, torch_res)

    print("\n=== Metric Evaluation Results ===")
    for key, val in results.items():
        print(f"{key.upper():8s} : {val:.3f}")


    #=== Image Wrapping and Metric Addition ===
    image_wr_metric = io.treat_image_and_add_metric(torch_res,
                                            image, 
                                            metric="psnr",
                                            subfolder= "fonts",
                                            fname = "Poppins-BoldItalic.ttf",
                                            )

    plt.imshow(image_wr_metric)
    plt.axis("off")

    # io.save_comparison(folder="results",
    #                 file_name="noising",
    #                 names_list= [["noised"],],
    #                 images_list= [[torch_res],],
    #                 reference_image_list=[image,],
    #                 metric="psnr",
    #                 trajectories_list=None,
    #                 save=True,
    #                 scale_plot=(2, 2))