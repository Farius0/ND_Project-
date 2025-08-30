# ==================================================
# =============== TEST: SegmenterND ================
# ==================================================

import numpy as np, matplotlib.pyplot as plt, random
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path

from utils.decorators import timer
from utils.nd_tools import show_plane
from operators.image_io import ImageIO
from operators.segmenter_nd import segmenter_nd
from core.config import (LayoutConfig, GlobalConfig)

# === Make runner ===
def make_runner(segmenter_mode, processor_strategy, input_format="numpy", output_format="numpy", layout_name="HWC", 
                layout_framework="numpy", layout_ensured_name="CHW", threshold=0.5, multi_thresholds=(0.33, 0.66), num_classes=4, use_channels=True):
    @timer(return_result=True, name = segment_mode + "_" + input_format + "_" + output_format)
    def _runner(img):
        return segmenter_nd(img, segmenter_mode, processor_strategy=processor_strategy, framework=input_format, output_format=output_format,
                          layout_name=layout_name, layout_framework=layout_framework, layout_ensured_name=layout_ensured_name,threshold=threshold, 
                          multi_thresholds=multi_thresholds, num_classes=num_classes, use_channels=use_channels, seeds=None, n_seeds=3,)
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
    
    segment_mode, i, cmap, colorbar = "kmeans", 0, "tab10", False # "viridis" or "tab10" or "gray" "binary"

    vectorized_ = make_runner(segment_mode, "vectorized")
    torch_ = make_runner(segment_mode, "torch", input_format="torch")
#     classic_ = make_runner("otsu", "classic")
    parallel_ = make_runner(segment_mode, "parallel")

    vectorized_segment = vectorized_(image)
    torch_segment = torch_(image)
    # classic_segment = classic_(image)
    parallel_segment = parallel_(image)
    

    fig, ((a, c), (b, d)) = plt.subplots(2, 2, figsize=(15, 10), dpi=100)
    
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # show_plane(a, image, 
    #            title="Original Image", 
    #            cmap=cmap, 
    #            colorbar=colorbar)
    # show_plane(b, 
    #            vectorized_segment[slice(None), slice(None), i], 
    #            title=f"{segment_mode}_vectorized_segment", 
    #            cmap=cmap,
    #            colorbar=colorbar)
    # show_plane(c,
    #            torch_segment[slice(None), slice(None), i], 
    #            title=f"{segment_mode}_torch_segment", 
    #            cmap=cmap,
    #            colorbar=colorbar)
    # show_plane(d,
    #            parallel_segment[slice(None), slice(None), i], 
    #            title=f"{segment_mode}_parallel_segment", 
    #            cmap=cmap,
    #            colorbar=colorbar)
    
    # plt.show()
    
    # show_plane(a, image, 
    #            title="Original Image", 
    #            cmap=cmap, 
    #            colorbar=colorbar)
    # show_plane(b, 
    #            vectorized_segment.astype(np.float32), 
    #            title="torch_norm", 
    #            cmap=cmap,
    #            colorbar=colorbar)
    # show_plane(c,
    #            torch_segment.astype(np.float32), 
    #            title="torch", 
    #            cmap=cmap,
    #            colorbar=colorbar)
    # show_plane(d,
    #            parallel_segment.astype(np.float32), 
    #            title="parallel", 
    #            cmap=cmap,
    #            colorbar=colorbar)
    # plt.show()
    
    show_plane(a, image, 
               title="Original Image", 
               cmap=cmap, 
               colorbar=colorbar)
    show_plane(b, 
               vectorized_segment[slice(None), slice(None)], 
               title=f"{segment_mode}_vectorized_segment", 
               cmap=cmap,
               colorbar=colorbar)
    show_plane(c,
               torch_segment[slice(None), slice(None)], 
               title=f"{segment_mode}_torch_segment", 
               cmap=cmap,
               colorbar=colorbar)
    show_plane(d,
               parallel_segment[slice(None), slice(None)], 
               title=f"{segment_mode}_parallel_segment", 
               cmap=cmap,
               colorbar=colorbar)    