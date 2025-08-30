# ==================================================
# =============== TEST: TestPeronaMalik ===============
# ==================================================

import matplotlib.pyplot as plt, random, torch, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path

from utils.decorators import timer
from utils.nd_tools import show_plane
from operators.image_io import ImageIO
from operators.image_operator import Operator
from algorithms.perona_malik import pm
from core.config import (LayoutConfig, GlobalConfig)

# === Make runner ===
def make_runner(algorithm, processor_strategy, diff_strategy, conv_strategy, 
                input_format="numpy", output_format="numpy"):
    @timer(return_result=True, name = algorithm + "_" + processor_strategy + "_" + diff_strategy + "_" + conv_strategy)
    def _runner(img):
        return pm(img, algorithm=algorithm, processor_strategy=processor_strategy, diff_strategy=diff_strategy, 
                  conv_strategy=conv_strategy, framework=input_format, output_format=output_format)
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

    image_noised = Operator(image, 
                            clip=True,
                            layout_cfg=layout_cfg,
                            global_cfg=global_cfg,
                            ).noise(sigma=0.1)

    algorithm, cmap = "pm", "inferno"

    vectorized_ = make_runner(algorithm, "vectorized", "vectorized", "gaussian")
    classic_ = make_runner(algorithm, "classic", "classic", "fft")
    torch_ = make_runner(algorithm, "torch", "torch", "torch", input_format="torch")
    parallel_ = make_runner(algorithm, "parallel", "parallel", "fft")

    vectorized_res=vectorized_(image_noised)
    classic_res=classic_(image_noised)
    torch_res=torch_(image_noised)
    parallel_res=parallel_(image_noised)

    noised_wr_metric = io.treat_image_and_add_metric(image_noised,
                                            image, 
                                            metric="psnr",
                                            subfolder= "fonts",
                                            fname = "Poppins-BoldItalic.ttf",
                                            )
    vectorized_wr_metric = io.treat_image_and_add_metric(vectorized_res,
                                            image, 
                                            metric="psnr",
                                            subfolder= "fonts",
                                            fname = "Poppins-BoldItalic.ttf",
                                            )

    torch_wr_metric = io.treat_image_and_add_metric(torch_res,
                                            image, 
                                            metric="psnr",
                                            subfolder= "fonts",
                                            fname = "Poppins-BoldItalic.ttf",
                                            )
    
    classic_wr_metric = io.treat_image_and_add_metric(classic_res,
                                            image, 
                                            metric="psnr",
                                            subfolder= "fonts",
                                            fname = "Poppins-BoldItalic.ttf",
                                            )    
    
    parallel_wr_metric = io.treat_image_and_add_metric(parallel_res,
                                            image, 
                                            metric="psnr",
                                            subfolder= "fonts",
                                            fname = "Poppins-BoldItalic.ttf",
                                            )

    _, ((a,b), (c,d), (e,f)) = plt.subplots(3, 2, figsize=(15, 15))

    show_plane(a, image[slice(None), slice(None), slice(None)] , title="Original image", cmap=cmap)
    show_plane(b, noised_wr_metric[slice(None), slice(None), slice(None)] , title="Noised image", cmap=cmap)
    show_plane(c, vectorized_wr_metric[slice(None), slice(None), slice(None)] , title=f"{algorithm}_vectorized", cmap=cmap)
    show_plane(d, torch_wr_metric[slice(None), slice(None), slice(None)] , title=f"{algorithm}_torch", cmap=cmap)
    show_plane(e, classic_wr_metric[slice(None), slice(None),  slice(None)] , title=f"{algorithm}_classic", cmap=cmap)
    show_plane(f, parallel_wr_metric[slice(None), slice(None), slice(None)] , title=f"{algorithm}_parallel", cmap=cmap)
    plt.show()

    # def srch_opt(image, alpha=None, dt=None, sigma=None, steps=20, algorithm="enhanced"):
    #     return pm(image, alpha=alpha, dt=dt, sigma=sigma, steps=steps, framework="torch", algorithm=algorithm,
    #               processor_strategy="torch", diff_strategy="torch", conv_strategy="torch", disable_tqdm=True)
    
    # # Recherche des meilleurs paramètres pour la fonction Perona_Malik enhanced
    # param_grid = {"alpha": [5e-2, 8e-2, 1.e-1], "dt": [3e-2, 4e-2, 5e-2], "steps": [10, 20, 30], 
    #               "algorithm": ["enhanced", "pm"]} # Main parameters
    # sub_grid = {} # No sub-parameters
    # func_args = {"image": image_noised, "sigma": 1e-5} # Fixed arguments for the function

    # best_params, best_score, table, PM_image = search_opt_general(func=srch_opt, u_truth=image, param_grid=param_grid, func_args=func_args, 
    #     sub_param_grid=sub_grid, return_results=True, verbose=True)

    # print(best_params)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # print(table)
    # pd.reset_option('display.max_rows')
    # Best parameter that can be used (Perona-Malik Enhanced):
    # 'alpha': 0.10, 'dt': 0.04, 'steps': 20 
    # 'alpha': 0.10, 'dt': 0.05, 'steps': 25
    # 'alpha': 0.05, 'dt': 0.05, 'steps': 50
