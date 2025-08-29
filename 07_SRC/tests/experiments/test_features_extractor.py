# ==================================================
# ============ TEST: Feature Extractor =============
# ==================================================

import matplotlib.pyplot as plt, random, torch, numpy as np
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path

from utils.decorators import timer
from operators.image_io import ImageIO
from utils.nd_tools import show_plane, plot_histogram_bins
from operators.feature_extractor import feature_extractor
from core.config import (LayoutConfig, GlobalConfig)

# === Make runner ===
def make_runner(features, 
                diff_strategy, 
                processor_strategy, 
                conv_strategy, 
                edge_strategy="gradient",
                input_format="numpy", 
                output_format="numpy",):
    @timer(return_result=True, return_elapsed=True, name = diff_strategy + "_" + processor_strategy + "_" + conv_strategy)
    def _runner(img):
        return feature_extractor(img, 
                                features=features, 
                                edge_strategy=edge_strategy,
                                diff_strategy=diff_strategy, 
                                processor_strategy=processor_strategy, 
                                conv_strategy=conv_strategy,
                                framework=input_format, 
                                output_format=output_format,
                                layout_name='HWC',
                                layout_framework='numpy', 
                                )
    return _runner

if __name__ == "__main__":

    root = Path.cwd().parent.parent / "03_EXAMPLES_DATA" / "Images"
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

    features  = ["gaussian_eigen",]

    vectorized_ = make_runner(features,"vectorized", "vectorized", "gaussian")
    classic_ = make_runner(features, "classic", "classic", "gaussian")
    torch_ = make_runner(features, "torch", "torch", "torch", input_format="torch")
    parallel_ = make_runner(features, "parallel", "parallel", "gaussian")

    vectorized_res, vectorized_time=vectorized_(image)
    classic_res, classic_time=classic_(image)
    torch_res, torch_time=torch_(image)
    parallel_res, parallel_time=parallel_(image)

    _, ((a,b), (c,d)) = plt.subplots(2, 2, figsize=(12, 8))

    # Visualize features
    c_idx=0
    f_idx=0
    cmap="auto" # "seismic", "jet", "Blues", "Reds", "inferno", "gray", "flag", "coolwarm", "plasma", "Greys_r"
    colorbar = False

    show_plane(a, 
            vectorized_res, 
            title=f"{features}_vectorized_{vectorized_time:.2f}s", 
            cmap=cmap,
            channel_index=c_idx,
            feature_index=f_idx,
            norm = False,
            colorbar=colorbar,
            feature_type=features[0])
    show_plane(b, 
            classic_res, 
            title=f"{features}_classic_{classic_time:.2f}s", 
            cmap=cmap,
            channel_index=c_idx,
            feature_index=f_idx,
            norm = False,
            colorbar=colorbar,
            feature_type=features[0])
    show_plane(c, 
            torch_res, 
            title=f"{features}_torch_{torch_time:.2f}s", 
            cmap=cmap,
            channel_index=c_idx,
            feature_index=f_idx,
            norm = False,
            colorbar=colorbar,
            feature_type=features[0])
    show_plane(d, 
            parallel_res, 
            title=f"{features}_parallel_{parallel_time:.2f}s", 
            cmap=cmap,
            channel_index=c_idx,
            feature_index=f_idx,
            norm = False,
            colorbar=colorbar,
            feature_type=features[0])

    # Visualize histogram
    # i = 2
    # bins_to_show = (0 * i, 3 * i, 5 * i, 7 * i)

    # plot_histogram_bins(hist_bins=vectorized_res[slice(None), slice(None), :, 0],
    #                     bins_to_show=(0, 3, 5, 7),
    #                     norm=True,
    #                     colorbar=False,
    #                     figsize=(15, 15),
    #                     color="red", 
    #                     density=True, 
    #                     range=None, 
    #                     cumulative=False, 
    #                     hist_ratio=0.3,
    #                     ncols=1,
    #                     cmap="plasma",
    #                     title_prefix="Histogram Bin")

    # feature_maps = {
    #     "contrast": glcm_contrast_map,
    #     "dissimilarity": glcm_dissimilarity_map}

    # print(io.track(vectorized_res).tag_summary())