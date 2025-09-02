# ==================================================
# =============== TESTS: Preprocessor ==============
# ==================================================

import numpy as np, itertools, pandas as pd, torch, matplotlib.pyplot as plt, random, time
import warnings; warnings.filterwarnings('ignore')
from functools import partial
from pathlib import Path

from utils.decorators import timer
from utils.nd_tools import show_plane
from operators.image_io import ImageIO
from operators.preprocessor import preprocess
from core.config import (LayoutConfig, GlobalConfig, ImageProcessorConfig, PreprocessorConfig)

# === Make runner ===
def make_runner(processor_strategy, input_format="numpy", output_format="numpy", layout_name="HWC", 
                layout_framework="numpy", layout_ensured_name="HWC", normalize=False, stretch=False, equalize=False,
                denoise=False, aggregate=False, remove_artifacts=False, local_contrast=False, gamma_correct=False):
    @timer(return_result=True, name = input_format + "_" + output_format)
    def _runner(img):
        return preprocess(img, processor_strategy=processor_strategy, framework=input_format, output_format=output_format,
                          layout_name=layout_name, layout_framework=layout_framework, layout_ensured_name=layout_ensured_name, 
                          normalize=normalize, stretch=stretch, denoise=denoise, aggregate=aggregate, remove_artifacts=remove_artifacts, 
                          local_contrast=local_contrast, gamma_correct=gamma_correct, equalize=equalize)
    return _runner

if __name__ == "__main__":

      torch_stretch = make_runner("torch", input_format="torch", layout_ensured_name="CHW", stretch=True, normalize=True,
                                    local_contrast=False, denoise=True, equalize=False)
      torch_norm = make_runner("torch", input_format="torch", normalize=True, layout_ensured_name="CHW")
      torch_ = make_runner("torch", input_format="torch", normalize=False, layout_ensured_name="CHW")

      # numpy_norm = make_runner("classic", input_format="numpy", normalize=True)
      # numpy_stretch = make_runner("classic", input_format="numpy", layout_ensured_name="HWC", stretch=True)
      # numpy_ = make_runner("classic", input_format="numpy", normalize=False)

      root = Path.cwd().parent.parent.parent / "03_EXAMPLES_DATA" / "Images"
      images_path = sorted([str(p) for p in root.rglob("*.png")])
      rand = random.randint(0, len(images_path) - 2)

      torch_res_stretch=torch_stretch(images_path[rand])
      torch_res_norm=torch_norm(images_path[rand])
      torch_res_=torch_(images_path[rand])
      # numpy_res_stretch=numpy_stretch(images_path[rand])
      # numpy_res_norm=numpy_norm(images_path[rand])
      # numpy_res=numpy_(images_path[rand])

      cmap, colorbar = "gray", True

      fig, ((a), (b), (c)) = plt.subplots(3, 1, figsize=(15, 10), dpi=100)

      # fig.subplots_adjust(wspace=0.1, hspace=0.1)

      show_plane(a, 
                  torch_res_stretch[slice(None), slice(None), 0], 
                  title="torch_stretch", 
                  cmap=cmap, 
                  colorbar=colorbar,
                  norm=True)
      show_plane(b, 
                  torch_res_norm[slice(None), slice(None), 0], 
                  title="torch_norm", 
                  cmap=cmap,
                  colorbar=colorbar)
      show_plane(c, 
                  torch_res_[slice(None), slice(None), 0], 
                  title="torch", 
                  cmap=cmap,
                  colorbar=colorbar)
      plt.show()

      print("=== [ Preprocess ] ===")
      print("Mean_torch_stretch:", np.mean(torch_res_stretch), "std_torch_stretch:", np.std(torch_res_stretch), 
            "min_torch_stretch:", np.min(torch_res_stretch), "max_torch_stretch:", np.max(torch_res_stretch))
      print("Mean_torch_norm:", np.mean(torch_res_norm), "std_torch_norm:", np.std(torch_res_norm), 
            "min_torch_norm:", np.min(torch_res_norm), "max_torch_norm:", np.max(torch_res_norm))
      print("Mean_torch_:", np.mean(torch_res_), "std_torch_:", np.std(torch_res_), 
            "min_torch_:", np.min(torch_res_), "max_torch_:", np.max(torch_res_), "\n")

      # io = ImageIO(
      #         layout_cfg=LayoutConfig(layout_name="HWC", layout_framework="numpy", layout_ensured_name="NCHW",),
      #         global_cfg=GlobalConfig(framework="torch", output_format="torch", normalize=False, add_batch_dim=True,),
      #             )

      # # ====[ Test batch resized with stack ]====
      # batch_paths = images_path[:4]
      # batch = io.load_batch(batch_paths, to="torch", stack=True, match_to="first")
      # # io.track(batch).tag_summary()
      # print("Mean batch:", torch.mean(batch), "std batch:", torch.std(batch), "min batch:", torch.min(batch), "max batch:", torch.max(batch))
      # batch_norm = make_runner("torch", input_format="torch", output_format="torch", normalize=True, layout_ensured_name="NCHW")
      # batch_res_norm=batch_norm(batch)
      # print("Mean batch norm:", torch.mean(batch_res_norm), "std batch norm:", torch.std(batch_res_norm),\
      #     "min batch norm:", torch.min(batch_res_norm), "max batch norm:", torch.max(batch_res_norm))
      # io.track(batch_res_norm).tag_summary()