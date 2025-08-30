# ==================================================
# ============== TESTS: edge_detector ==============
# ==================================================

import warnings; warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt, random
from pathlib import Path

from utils.decorators import timer
from operators.image_io import ImageIO
from utils.nd_tools import show_plane
from operators.edge_detector import edge_detect
from core.config import (LayoutConfig, GlobalConfig,)

# === Make runner ===
def make_runner(edge_strategy, 
                diff_strategy, 
                processor_strategy, 
                conv_strategy, 
                input_format="numpy", 
                output_format="numpy"):
    @timer(return_result=True, return_elapsed=True, name = edge_strategy + "_" + diff_strategy)
    def _runner(img):
        return edge_detect(img, 
                           edge_strategy=edge_strategy, 
                           diff_strategy=diff_strategy, 
                           conv_strategy=conv_strategy,
                           processor_strategy=processor_strategy, 
                           framework=input_format, 
                           output_format=output_format)
    return _runner

if __name__ == "__main__":
    
    root = Path.cwd().parent.parent.parent / "03_EXAMPLES_DATA" / "Images"
    images_path = [str(p) for p in root.rglob("*.png")]
    rand = random.randint(0, len(images_path) - 1)

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
    
    edge_strategy, cmap = "gradient", "inferno"

    vectorized_ = make_runner(edge_strategy,"vectorized", "vectorized", "gaussian")
    classic_ = make_runner(edge_strategy, "classic", "classic", "gaussian")
    torch_ = make_runner(edge_strategy, "torch", "torch", "torch", input_format="torch")
    parallel_ = make_runner(edge_strategy, "parallel", "parallel", "gaussian")

    vectorized_res, vectorized_time=vectorized_(image)
    classic_res, classic_time=classic_(image)
    torch_res, torch_time=torch_(image)
    parallel_res, parallel_time=parallel_(image)

    _, ((a,b), (c,d)) = plt.subplots(2, 2, figsize=(12, 8))

    show_plane(a, 
               vectorized_res[slice(None), slice(None), 0], 
               title=f"{edge_strategy}_vectorized_{vectorized_time:.2f}s", 
               cmap=cmap)
    show_plane(b, 
               classic_res[slice(None), slice(None),  0], 
               title=f"{edge_strategy}_classic_{classic_time:.2f}s", 
               cmap=cmap)
    show_plane(c, 
               torch_res[slice(None), slice(None), 0], 
               title=f"{edge_strategy}_torch_{torch_time:.2f}s", 
               cmap=cmap)
    show_plane(d, 
               parallel_res[slice(None), slice(None), 0], 
               title=f"{edge_strategy}_parallel_{parallel_time:.2f}s", 
               cmap=cmap)