# ==================================================
# ============== TESTS: Diff_Operator ==============
# ==================================================
import numpy as np, matplotlib.pyplot as plt, random
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path

from utils.decorators import timer
from utils.nd_tools import show_plane
from operators.image_io import ImageIO
from operators.diff_operator import diffop
from core.config import LayoutConfig, GlobalConfig

# === Make runner ===
def make_runner(diff_strategy, 
                func="sobel_hessian", 
                input_format="numpy", 
                output_format="numpy", 
                backend='sequential'):
    @timer(return_result=True, name = func + "_" + diff_strategy)
    def _runner(img=image):
        return diffop(img,
                      func=func, 
                      diff_strategy=diff_strategy, 
                      framework=input_format,
                      output_format=output_format,                    
                      backend=backend, 
                      )
    return _runner

if __name__ == "__main__":
    
    cwd = Path.cwd().parent.parent.parent / "03_EXAMPLES_DATA" / "Images"
    images_path = [str(p) for p in cwd.rglob("*.png")]
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

    vectorized_ = make_runner("vectorized")
    classic_ = make_runner("classic")
    torch_ = make_runner("torch", input_format="torch")
    parallel_ = make_runner("parallel", backend="sequential") # threading, loky

    vectorized_res=vectorized_(image)
    classic_res=classic_(image)
    torch_res=torch_(image)
    parallel_res=parallel_(image)

    _, ((a,b), (c,d)) = plt.subplots(2, 2, figsize=(12, 8))

    # show_plane(a, vectorized_res[slice(None), slice(None), 0] , title="Laplacian vectorized", cmap="seismic")
    # show_plane(b, classic_res[slice(None), slice(None), 0] , title="Laplacian classic", cmap="seismic")
    # show_plane(c, torch_res[slice(None), slice(None), 0] , title="Laplacian torch", cmap="seismic")
    # show_plane(d, parallel_res[slice(None), slice(None), 0] , title="Laplacian parallel", cmap="seismic")

    # print("Mean_torch:", np.mean(torch_res), "std_torch:", np.std(torch_res), "min_torch:", np.min(torch_res), "max_torch:", np.max(torch_res))
    # print("Mean_vectorized:", np.mean(vectorized_res), "std_vectorized:", np.std(vectorized_res), "min_vectorized:", np.min(vectorized_res), "max_vectorized:", np.max(vectorized_res))
    # print("Mean_classic:", np.mean(classic_res), "std_classic:", np.std(classic_res), "min_classic:", np.min(classic_res), "max_classic:", np.max(classic_res))
    # print("Mean_parallel:", np.mean(parallel_res), "std_parallel:", np.std(parallel_res), "min_parallel:", np.min(parallel_res), "max_parallel:", np.max(parallel_res))

    i, j = 0, 0

    # # gradient
    # show_plane(a, vectorized_res[i][slice(None), slice(None), 0] , title="Laplacian vectorized", cmap="seismic")
    # show_plane(b, classic_res[i][slice(None), slice(None), 0] , title="Laplacian classic", cmap="seismic")
    # show_plane(c, torch_res[i][slice(None), slice(None), 0] , title="Laplacian torch", cmap="seismic")
    # show_plane(d, parallel_res[i][slice(None), slice(None), 0] , title="Laplacian parallel", cmap="seismic")

    # print("Mean_torch:", np.mean(torch_res[i]), "std_torch:", np.std(torch_res[i]), "min_torch:", np.min(torch_res[i]), "max_torch:", np.max(torch_res[i]))
    # print("Mean_vectorized:", np.mean(vectorized_res[i]), "std_vectorized:", np.std(vectorized_res[i]), "min_vectorized:", np.min(vectorized_res[i]), "max_vectorized:", np.max(vectorized_res[i]))
    # print("Mean_classic:", np.mean(classic_res[i]), "std_classic:", np.std(classic_res[i]), "min_classic:", np.min(classic_res[i]), "max_classic:", np.max(classic_res[i]))
    # print("Mean_parallel:", np.mean(parallel_res[i]), "std_parallel:", np.std(parallel_res[i]), "min_parallel:", np.min(parallel_res[i]), "max_parallel:", np.max(parallel_res[i]))

    # # hessian
    show_plane(a, vectorized_res[i,j][slice(None), slice(None), 0] , title="Laplacian vectorized", cmap="seismic")
    show_plane(b, classic_res[i,j][slice(None), slice(None), 0] , title="Laplacian classic", cmap="seismic")
    show_plane(c, torch_res[i,j][slice(None), slice(None), 0] , title="Laplacian torch", cmap="seismic")
    show_plane(d, parallel_res[i,j][slice(None), slice(None), 0] , title="Laplacian parallel", cmap="seismic")

    print("Mean_torch:", np.mean(torch_res[i,j]), "std_torch:", np.std(torch_res[i,j]), "min_torch:", np.min(torch_res[i,j]), "max_torch:", np.max(torch_res[i,j]))
    print("Mean_vectorized:", np.mean(vectorized_res[i,j]), "std_vectorized:", np.std(vectorized_res[i,j]), "min_vectorized:", np.min(vectorized_res[i,j]), "max_vectorized:", np.max(vectorized_res[i,j]))
    print("Mean_classic:", np.mean(classic_res[i,j]), "std_classic:", np.std(classic_res[i,j]), "min_classic:", np.min(classic_res[i,j]), "max_classic:", np.max(classic_res[i,j]))
    print("Mean_parallel:", np.mean(parallel_res[i,j]), "std_parallel:", np.std(parallel_res[i,j]), "min_parallel:", np.min(parallel_res[i,j]), "max_parallel:", np.max(parallel_res[i,j]))