# ==================================================
# =========== MODULE: artifact_cleaner =============
# ==================================================
from __future__ import annotations

from typing import Optional, Literal, Any, Dict, Union, Tuple


import pydicom, numpy as np, os, torch
from skimage.morphology import closing
import matplotlib.pyplot as plt
from pathlib import Path

# from utils.decorators import timer
from core.operator_core import OperatorCore
from core.layout_axes import get_layout_axes
from operators.image_processor import ImageProcessor
from operators.feature_extractor import feature_extractor
from core.config import LayoutConfig, GlobalConfig, ImageProcessorConfig

__all__ = ["ArtifactCleanerND",]

ArrayLike = np.ndarray | torch.Tensor
Framework = Literal["numpy", "torch"]
PathLike = Union[str, Path]

class ArtifactCleanerND(OperatorCore):
    """
    ND-compatible cleaner for structured image artifacts (e.g., horizontal stripes).
    Dual backend (NumPy / Torch), applied slice-by-slice via ImageProcessor.
    """
    def __init__(
        self,

        framework: Framework = "numpy",
        layout_name: str = "DHW",
        layout_framework: Framework = "numpy",
        threshold: float = 0.5,
    ) -> None:
        
        # ====[ Configuration ]====
        self._proc_params: Dict[str, Any] = {"processor_strategy": "parallel"}
        self._layout_params: Dict[str, Any] = {"layout_name": layout_name, "layout_framework": layout_framework}
        self._global_params: Dict[str, Any] = {"framework": framework, "output_format": framework}

        # ====[ Mirror inherited params locally for easy access ]====
        self.layout_cfg: LayoutConfig = LayoutConfig(**self._layout_params)
        self.global_cfg: GlobalConfig = GlobalConfig(**self._global_params)
        self.img_process_cfg = ImageProcessorConfig(**self._proc_params)
        self.framework: Framework = self.global_cfg.framework.lower()
        self.output_format: Framework = self.global_cfg.output_format.lower()
        self.threshold: float = threshold
        self.device: str = "cuda" if (torch.cuda.is_available() and self.framework == "torch") else self.global_cfg.device
        
        # === Layout: resolved axes, layout name, layout dict ===
        self.axes: Dict[str, Any] = self.layout_cfg.resolve(include_meta=True)
        self.layout_name: str = self.axes.get("layout_name")
        self.layout: Dict[str, Any] = self.axes.get("layout")
        self.layout_framework: Framework = self.axes.get("layout_framework")

        self.channel_axis: Optional[int] = self.axes.get("channel_axis")
        self.batch_axis: Optional[int] = self.axes.get("batch_axis")
        self.direction_axis: Optional[int] = self.axes.get("direction_axis")
        self.height_axis: Optional[int] = self.axes.get("height_axis")
        self.width_axis: Optional[int] = self.axes.get("width_axis")
        self.depth_axis: Optional[int] = self.axes.get("depth_axis")

        # ====[ Initialize OperatorCore with all axes ]====
        super().__init__(
            layout_cfg = self.layout_cfg,
            global_cfg = self.global_cfg, 
        )
        
        # ====[ Create ImageProcessor ]====
        self.processor: ImageProcessor = ImageProcessor(
            img_process_cfg = self.img_process_cfg,
            layout_cfg = self.layout_cfg,
            global_cfg = self.global_cfg,
        )        
    
    # ---------------- Feature extraction pipeline ----------------

    def extractor(
        self,
        img: ArrayLike,
        features: Optional[list[str]] = None,
        window_size: Optional[int] = None,
    ) -> ArrayLike:
        """
        Thin wrapper to call the project-wide `feature_extractor` with consistent layout.
        """
        return feature_extractor(
            img,
            features=features,
            framework=self.framework,
            output_format=self.output_format,
            layout_name="CHW" if self.layout_name == "DHW" else self.layout_name,
            layout_framework="numpy",
            conv_strategy=None,
            processor_strategy=None,
            window_size=window_size,
            stack=False,
        )
        
    def _extract_features(self, img2d: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        Build smoothed Sobel-based features gx, gy, then edge maps and median filter.
        """
        tagger = self.track(img2d)

        sobel = self.extractor(img2d, features=["sobel_gradient"])
        gx, gy = tagger.copy_to(sobel[0]).get(), tagger.copy_to(sobel[1]).get()

        gx = self.extractor(gx, features=["mean"], window_size=3)
        gy = self.extractor(gy, features=["mean"], window_size=3)

        gx = self.extractor(gx, features=["sobel_edge"])
        gy = self.extractor(gy, features=["sobel_edge"])

        gx = self.extractor(gx, features=["median"], window_size=3)
        gy = self.extractor(gy, features=["median"], window_size=3)

        return gx, gy

    def _generate_mask(
        self, gx: ArrayLike, gy: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike, int, bool, bool]:
        """
        Build a binary mask from gx-gy contrast and derive band statistics.
        """
        diff = gx - gy
        mask = torch.where(diff != 1., 0., 1.) if isinstance(diff, torch.Tensor) else np.where(diff != 1., 0., 1.)
        mean_ax = mask.float().mean(dim=1) if isinstance(mask, torch.Tensor) else mask.mean(axis=1)
        cond_1, cond_2 = mean_ax.max() > self.threshold, (mean_ax > self.threshold).sum() > 2
        line_idx = torch.argmax(mean_ax).item() if isinstance(mean_ax, torch.Tensor) else np.argmax(mean_ax)
        return mask, mean_ax, line_idx, cond_1, cond_2

    def _interpolate_band(
        self,
        img: ArrayLike,
        mask: ArrayLike,
        gy: ArrayLike,
        mean_ax: ArrayLike,
        line_idx: int,
    ) -> ArrayLike:
        """
        Zero-out the detected band and inpaint rows by simple vertical interpolation.
        """
        H = int(img.shape[0])
        count = int((mean_ax > self.threshold).sum().item() if isinstance(mean_ax, torch.Tensor) else np.sum(mean_ax > self.threshold))
        thickness = count + 1

        x_start, x_end = max(0, line_idx - thickness), min(H, line_idx + thickness)
        rnge = list(range(0, x_start)) + list(range(x_end, H))

        if isinstance(mask, torch.Tensor):
            # Clean mask outside band
            mask[rnge, :] = torch.where(mask[rnge, :] == 1, 0, mask[rnge, :])
            
            # Morphological closing on gy (work in numpy/bool, then back to torch)
            closed_np = closing(gy.detach().cpu().numpy(), np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool))
            closed = torch.tensor(closed_np, dtype=torch.bool, device=img.device)

            img_copy = img.clone()
            img_copy[x_start:x_end, :] = torch.where(mask[x_start:x_end, :] == 1., 0., img_copy[x_start:x_end, :])
            img_copy[x_start:x_end, :] = torch.where(closed[x_start:x_end, :] == 1., img[x_start:x_end, :], img_copy[x_start:x_end, :])
        else:
            mask[rnge, :] = np.where(mask[rnge, :] == 1., 0., mask[rnge, :])
           
            closed = closing(gy, np.array([[0,1,0],[0,1,0],[0,1,0]], dtype=bool))
            
            img_copy = img.copy()
            img_copy[x_start:x_end, :] = np.where(mask[x_start:x_end,:]==1., 0., img_copy[x_start:x_end,:])
            img_copy[x_start:x_end, :] = np.where(closed[x_start:x_end,:]==1., img[x_start:x_end,:], img_copy[x_start:x_end,:])

        # Interpolation of rows within the band (simple average of neighbors)
        for i, x in enumerate(reversed(range(x_start - thickness//2, x_end - thickness//2))):
            top = img_copy[x_start + i - 1, :] if x_start > 0 else img_copy[x_end + i + 1, :]
            bot = img_copy[x_end - i + 1, :] if x_end + 1 < H else img_copy[x_start - i - 1, :]
            interp = ((top.float() + bot.float()) // 2).to(img.dtype) if isinstance(img, torch.Tensor) \
                     else ((top.astype(np.float32, copy=False) + bot.astype(np.float32, copy=False)) // 2).astype(img.dtype, copy=False)
            img_copy[x, :] = interp

        return img_copy

    def _clean_slice(self, slice2d: ArrayLike) -> ArrayLike:
        """
        Detect a stripe band on a 2D slice and inpaint it if confident.
        """
        gx, gy = self._extract_features(slice2d)
        mask, mean_ax, line_idx, cond_1, cond_2 = self._generate_mask(gx, gy)

        # Optional pre-clear above the main line
        slice2d[: max(line_idx - 3, 0), :] = 0.0

        if cond_1 and cond_2:
            return self._interpolate_band(slice2d, mask, gy, mean_ax, line_idx)
        return slice2d

    # @log_exceptions(logger_name="error_logger")
    # @timer(return_result=True, return_elapsed=False, name ="artifact_cleaning")
    def __call__(self, image: ArrayLike, axis: int = 1) -> ArrayLike:
        """
        Clean a 2D image or a 3D volume along a given axis (0: axial, 1: coronal, 2: sagittal).
        """
        img = self.convert_once(image=image, tag_as="input", framework=self.framework)
        
        if img.ndim == 3 and self.layout_name == "DHW":

            if axis < 0 or axis > 2:
                raise ValueError(f"Invalid axis {axis}. Expected 0, 1, or 2 for a 3D volume.")            

            # Move and track (layout set to CHW during slice processing)
            tracker = self.track(img).moveaxis(axis, 0) # 0: axial 1: coronal 2: sagittal
            layout = get_layout_axes(self.framework, "CHW")
            layout.pop("name", None)
            layout.pop("description", None)
            layout.update({"layout_name": "CHW", 
                           "shape_after": tracker.image.shape})
            tracker.update_tags(layout)
            img = tracker.get()  
            
            # Clean slice-by-slice through ImageProcessor
            self.processor.function = self._clean_slice
            result = self.processor(img) 
            
            # Move back to original axis and restore tags
            tracker2 = self.track(result).moveaxis(0, axis)
            tracker2.update_tags(self.layout, add_new_axis=True)
            tracker2.update_tags({"status": "cleaned", 
                                  "layout_name": self.layout_name,
                                  "shape_after": tracker2.image.shape})
            return tracker2.get()
        
        elif img.ndim == 2:
            self.processor.function = self._clean_slice
            return self.processor(image)
        
        else:
            raise ValueError("Image must be 2D or a 3D volume with layout 'DHW'.")
      
if __name__ == "__main__":
    
    # # # Load Data
     
    # dir = r'C:\LC-OCT_Images_Group' # B
    dir = r"C:\Users\ainau\Downloads\Université de Bordeaux\Stage Orléans\03_EXAMPLES_IMAGES" # H

    os.chdir(dir)

    files = [str(file) for file in Path(dir).rglob('*.visualisable.dcm') if (os.path.getsize(file)//1024) > 2000]


    # # # Select volume or slice

    # da_1 = pydicom.dcmread(files[9])
    # da_2 = pydicom.dcmread(files[10])
    # da_3 = pydicom.dcmread(files[11]) 
    # data_1 = da_1.pixel_array
    # data_2 = da_2.pixel_array
    # data_3 = da_3.pixel_array
    # image = data[:, 80, :]
    
    
    # # # Volume Configuration with torch backend
    
    # artifact_cleaner = ArtifactCleanerND(framework="torch", layout_name="DHW", layout_framework="torch") 

    # data_cleaned_1 = artifact_cleaner(data_1)
    # data_cleaned_2 = artifact_cleaner(data_2)
    # data_cleaned_3 = artifact_cleaner(data_3)
    
    # print(data_cleaned_1.shape, data_cleaned_2.shape, data_cleaned_3.shape)


    # # # Slice Configuration with torch backend
    
    # artifact_cleaner = ArtifactCleanerND(framework="torch", layout_name="HW", layout_framework="torch")  
    # image_cleaned = artifact_cleaner(image)
    

    # # # Visualization    

    # plt.figure(figsize=(15, 10), dpi=150)

    # plt.subplot(211)
    # plt.imshow(data_1[:,2,:], cmap="gray")
    # plt.imshow(image, cmap="gray")

    # plt.subplot(212)
    # plt.imshow(data_cleaned_1[:,2,:], cmap="gray")
    # plt.imshow(image_cleaned.detach().cpu().numpy(), cmap="gray")

    # plt.show()
