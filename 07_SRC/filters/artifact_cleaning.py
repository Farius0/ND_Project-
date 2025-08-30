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
        """
        Initialize an ND artifact cleaner with layout and backend configuration.

        Parameters
        ----------
        framework : {"numpy", "torch"}, default "numpy"
            Backend framework used for processing (NumPy or PyTorch).
        layout_name : str, default "DHW"
            Input layout name describing axis order (e.g., "DHW", "HWC").
        layout_framework : {"numpy", "torch"}, default "numpy"
            Layout resolution method based on backend convention.
        threshold : float, default 0.5
            Threshold used to identify and remove structured artifacts.
        """
        
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
        Extract features from an image using the project-wide `feature_extractor`.

        This method ensures consistent layout handling and backend compatibility.

        Parameters
        ----------
        img : ArrayLike
            Input image (NumPy array or PyTorch tensor).
        features : list of str, optional
            Specific features to extract (e.g., "entropy", "gradient", "glcm").
            If None, all default features will be computed.
        window_size : int, optional
            Size of the local window for window-based features (e.g., GLCM, entropy).

        Returns
        -------
        ArrayLike
            Output feature map(s), matching the backend and layout conventions.
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
        Extract Sobel-based edge features from a 2D image slice.

        Applies gradient filters (gx, gy), computes edge magnitude,
        and applies a median filter to reduce noise and artifacts.

        Parameters
        ----------
        img2d : ArrayLike
            2D input image (single slice) as a NumPy array or PyTorch tensor.

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            - Edge magnitude map (float).
            - Smoothed version (e.g., median-filtered) of the edge map.
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
        Generate a binary mask from the difference between gx and gy gradients,
        and compute band-level summary statistics for detection.

        The mask highlights regions where gx - gy == 1, indicating potential artifact patterns.
        Summary statistics are computed along axis 1 (typically height or depth).

        Parameters
        ----------
        gx : ArrayLike
            Horizontal gradient feature map (e.g., from Sobel filter).
        gy : ArrayLike
            Vertical gradient feature map.

        Returns
        -------
        Tuple
            - mask : ArrayLike
                Binary mask indicating artifact regions (1 = candidate, 0 = background).
            - mean_ax : ArrayLike
                Mean mask value along axis 1 (per-band activation).
            - line_idx : int
                Index of the band with the maximum mask activation.
            - cond_1 : bool
                True if max(mean_ax) > threshold.
            - cond_2 : bool
                True if more than 2 bands exceed the threshold.
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
        Clean and inpaint a horizontal artifact band using vertical interpolation.

        The method identifies a corrupted band around `line_idx` (based on activation in `mean_ax`),
        zeros out detected artifact regions in the band using the `mask`,
        refines edges using morphological closing on `gy`,
        and finally fills the band by interpolating values from adjacent rows.

        Parameters
        ----------
        img : ArrayLike
            2D input image slice to clean (NumPy or Torch).
        mask : ArrayLike
            Binary mask indicating corrupted regions (1 = artifact).
        gy : ArrayLike
            Vertical edge map used to detect structural continuity.
        mean_ax : ArrayLike
            Mean mask activation across rows (used to determine thickness).
        line_idx : int
            Central row index of the corrupted band.

        Returns
        -------
        ArrayLike
            Cleaned image with the artifact band inpainted.
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
        Detect and optionally remove a horizontal stripe artifact from a 2D image slice.

        Applies feature extraction and binary mask generation to locate stripe regions.
        If artifact presence is confidently detected, the band is inpainted by vertical interpolation.
        Otherwise, the slice is returned unmodified (except optional zeroing above the line).

        Parameters
        ----------
        slice2d : ArrayLike
            2D input image slice (NumPy or Torch).

        Returns
        -------
        ArrayLike
            Cleaned 2D image slice, with artifact region inpainted if detected.
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
        Clean a 2D image or a 3D volume from structured artifacts along a given axis.

        For 3D inputs, the method processes the volume slice-by-slice along the specified axis
        (0: axial, 1: coronal, 2: sagittal). For 2D inputs, it applies the cleaning directly.

        Parameters
        ----------
        image : ArrayLike
            Input 2D image or 3D volume (NumPy array or PyTorch tensor).
        axis : int, default 1
            Axis along which to process the volume if the input is 3D:
            0 = axial slices, 1 = coronal slices, 2 = sagittal slices.

        Returns
        -------
        ArrayLike
            Cleaned image or volume, with updated layout and metadata tags.
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
