# ==================================================
# =============  MODULE: preprocessor  =============
# ==================================================
from __future__ import annotations

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Optional, Union, Tuple, List, Dict, Any, Literal
import matplotlib.pyplot as plt, numpy as np, pydicom, torch, json
from skimage.exposure import equalize_hist
from pathlib import Path

from core.operator_core import OperatorCore
from core.config import (LayoutConfig, GlobalConfig, ImageProcessorConfig, PreprocessorConfig)
from operators.feature_extractor import feature_extractor
from operators.image_processor import ImageProcessor
from operators.image_io import ImageIO
from filters.artifact_cleaning import ArtifactCleanerND
from filters.perona_enhancing import PeronaEnhancer
# from utils.decorators import log_exceptions

# Public API
__all__ = ["PreprocessorND", "preprocess"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Framework = Literal["numpy", "torch"]

class PreprocessorND(OperatorCore):
    """
    Modular ND image preprocessing with dual-backend (NumPy/Torch) and layout-aware control.

    Supports a chain of configurable steps such as normalization, clipping, stretching,
    equalization, gamma correction, denoising, and artifact removal.

    Notes
    -----
    - All steps are enabled/disabled via the `PreprocessorConfig` flags.
    - Execution order is controlled by `self.pipeline`, which can be updated or reordered.
    - Built on OperatorCore: preserves UID tagging, axis layout, and traceability.
    - Compatible with both 2D and 3D images (and higher-D when meaningful).
    """
    def __init__(
        self,
        preprocess_cfg: PreprocessorConfig = PreprocessorConfig(),
        img_process_cfg: ImageProcessorConfig = ImageProcessorConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize the ND-aware preprocessing pipeline.

        Parameters
        ----------
        preprocess_cfg : PreprocessorConfig
            Flags and parameters defining which preprocessing steps to apply (e.g., clip, normalize, stretch).
        img_process_cfg : ImageProcessorConfig
            Strategy for backend execution (vectorized, parallel, fallback, etc.).
        layout_cfg : LayoutConfig
            Axis configuration and layout tracking (e.g., channel_axis, depth_axis).
        global_cfg : GlobalConfig
            General framework behavior, output format, device, and backend strategy.
        """
        # ====[ Configuration ]====
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg
        self.preprocess_cfg: PreprocessorConfig = preprocess_cfg
        self.img_process_cfg: ImageProcessorConfig = img_process_cfg
        
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
        
        # ====[ Store Preprocessor-specific parameters ]====
        self.preprocess_fallback: Optional[Union[str, bool]] = self.preprocess_cfg.fallback
        self.clip: bool = bool(self.preprocess_cfg.clip)
        self.clip_range: Tuple[float, float] = self.preprocess_cfg.clip_range
        self.normalize_mode: str = self.preprocess_cfg.normalize_mode
        self.stretch: bool = bool(self.preprocess_cfg.stretch)
        self.p_low: float = float(self.preprocess_cfg.p_low)
        self.p_high: float = float(self.preprocess_cfg.p_high)
        self.out_range: Tuple[float, float] = self.preprocess_cfg.out_range
        self.aggregate: bool = bool(self.preprocess_cfg.aggregate)
        self.agg_mode: str = self.preprocess_cfg.agg_mode
        self.block_size: int = self.preprocess_cfg.block_size
        self.keys: Optional[str] = self.preprocess_cfg.keys
        self.gamma_correct: bool = bool(self.preprocess_cfg.gamma_correct)
        self.gamma: float = float(self.preprocess_cfg.gamma)
        self.remove_artifacts: bool = bool(self.preprocess_cfg.remove_artifacts)
        self.denois: bool = bool(self.preprocess_cfg.denoise)
        self.local_contrast: bool = bool(self.preprocess_cfg.local_contrast)
        self.equalize: bool = bool(self.preprocess_cfg.equalize)
        
        # === Pipeline definition (name, function, priority) ===
        self.pipeline: List[Tuple[str, callable, int]] = [
            ("normalize", self.normalize_nd, 0),
            ("stretch", self.stretch_contrast_nd, 1),
            ("equalize", self.equalize_hist_nd, 2),
            ("gamma_correct", self.gamma_correction_nd, 3),
            ("local_contrast", self.local_contrast_nd, 4),
            ("aggregate", self.aggregate_nd, 5),
            ("remove_artifacts", self.remove_artifacts_2d, 6),
            ("denoise", self.perona_enhancing, 7),
        ]

        # ====[ Mirror inherited params locally for easy access ]====
        self.framework: Framework = self.global_cfg.framework.lower()
        self.output_format: Framework = self.global_cfg.output_format.lower()
        self.layout_ensured_name: str = self.layout_cfg.layout_ensured_name
        self.add_batch_dim: Optional[bool] = self.global_cfg.add_batch_dim
        self.add_channel_dim: Optional[bool] = self.global_cfg.add_channel_dim
        self.normalize: bool = bool(self.global_cfg.normalize)
        self.verbose: bool = bool(self.global_cfg.verbose)        
        self.device: str = (
            "cuda"
            if (torch.cuda.is_available() and self.framework == "torch")
            else self.global_cfg.device
        )        

        # Inherit tagging, layouts, etc.
        super().__init__(
            layout_cfg=layout_cfg,
            global_cfg=global_cfg,
        )
        
        # Inherit image I/O
        self.image_io: ImageIO = ImageIO(
            layout_cfg=self.layout_cfg,
            global_cfg=self.global_cfg.update_config(normalize=False),
        )
        
        # ====[ Create ImageProcessor ]====
        self.processor: ImageProcessor = ImageProcessor(
            img_process_cfg = self.img_process_cfg,
            layout_cfg = self.layout_cfg,
            global_cfg = self.global_cfg.update_config(output_format=self.framework),
        )

        # if self.resize_shape is not None:
        #     self.resizer: ResizeOperator = ResizeOperator(
        #         layout_cfg = self.layout_cfg,
        #         global_cfg = self.global_cfg
        #     )
        
    # ---------------------- Orchestration ----------------------        
        
    def compose(self, image: ArrayLike) -> ArrayLike:
        """
        Apply the preprocessing pipeline sequentially to the input image.

        Only the enabled steps (as defined by `PreprocessorConfig`) are applied,
        and the order of execution is defined by `self.pipeline`, sorted by priority.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image to preprocess.

        Returns
        -------
        ArrayLike
            Preprocessed image after all enabled steps are applied.

        Notes
        -----
        - Each step in the pipeline is a tuple: (step_name, function, priority).
        - Steps are only executed if the corresponding flag (e.g., `self.clip`) is True.
        - Verbose mode logs each step name and output stats (shape, dtype, min/max).
        - Compatible with both NumPy and Torch tensors.
        """
        img = image
        
        self.pipeline.sort(key=lambda step: step[2]) 
        
        for step_name, function, _ in self.pipeline:
            if getattr(self, step_name, False):
                if self.verbose:
                    print(f"[compose] Step: {step_name}")
                img = function(img)
            if self.verbose:
                print(f"[compose] Step: {step_name}")
                print(f" → shape: {img.shape}, dtype: {img.dtype}, min: {img.min():.3f}, max: {img.max():.3f}")

        return img
    
    def set_priority(self, step_name: str, new_priority: int) -> None:
        """
        Update the execution priority of a preprocessing step in the pipeline.

        Parameters
        ----------
        step_name : str
            Name of the step to update (as registered in `self.pipeline`).
        new_priority : int
            Execution priority. Lower values are executed earlier.

        Raises
        ------
        ValueError
            If the step name is not found in the pipeline.

        Notes
        -----
        - After updating the priority, the pipeline is automatically re-sorted.
        - Useful for customizing the order of preprocessing (e.g., normalize before clip).
        - If `self.verbose` is enabled, logs the change to stdout.
        """
        for i, (name, func, _) in enumerate(self.pipeline):
            if name == step_name:
                self.pipeline[i] = (name, func, new_priority)
                break
        else:
            raise ValueError(f"Step '{step_name}' not found in pipeline.")

        # Re-sort pipeline after priority update
        self.pipeline.sort(key=lambda step: step[2])

        if self.verbose:
            print(f"[set_priority] Step '{step_name}' now has priority {new_priority}")
    
    def save_pipeline(self, filepath: Union[str, Path]) -> None:
        """
        Save the current preprocessing pipeline configuration to a JSON file.

        Stores the list of enabled steps and their execution priorities,
        allowing for later inspection or reuse.

        Parameters
        ----------
        filepath : str or Path
            Destination path for the JSON file.

        Notes
        -----
        - Only step names and priorities are saved (functions are not serialized).
        - The output JSON file contains a list of {"step": ..., "priority": ...} entries.
        - Enables reproducibility and transparency of preprocessing pipelines.
        - Verbose mode prints the save confirmation with full path.
        """
        data = [
            {"step": name, "priority": priority}
            for name, _, priority in self.pipeline
        ]
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        if self.verbose:
            print(f"[save_pipeline] Pipeline saved to {filepath}")

    def load_pipeline(self, filepath: Union[str, Path]) -> None:
        """
        Load and apply step priorities for the preprocessing pipeline from a JSON file.

        Reads the file saved by `save_pipeline()` and updates the current pipeline order
        accordingly. Missing steps fall back to low priority (default: 99).

        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file containing a list of {"step": ..., "priority": ...} entries.

        Notes
        -----
        - Steps present in the file will override the current priorities.
        - Steps missing from the file will be assigned fallback priority 99.
        - The pipeline is automatically re-sorted after loading.
        - Verbose mode prints confirmation and the loaded file path.
        """
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)

        # Create a mapping for faster access
        priority_map = {entry["step"]: entry["priority"] for entry in data}

        new_pipeline = []
        for name, func, _ in self.pipeline:
            prio = priority_map.get(name, 99)  # fallback if missing
            new_pipeline.append((name, func, prio))
        self.pipeline = sorted(new_pipeline, key=lambda x: x[2])

        if self.verbose:
            print(f"[load_pipeline] Pipeline priorities loaded from {filepath}")


    # @log_exceptions(logger_name="error_logger")
    def __call__(self, image: Union[ArrayLike, str, Path]) -> ArrayLike:
        """
        Run the configured preprocessing pipeline on an ND image or image path.

        Automatically detects the input type, applies layout-aware reading if needed,
        and executes all enabled preprocessing steps in priority order.

        Parameters
        ----------
        image : ndarray | Tensor | str | Path
            Input image, either already loaded or specified by file path.

        Returns
        -------
        ndarray | Tensor
            Preprocessed image, tagged and formatted according to `GlobalConfig`.

        Raises
        ------
        TypeError
            If the input is not a supported type.

        Notes
        -----
        - Paths are loaded using `ImageIO.read_image()` with configured layout/backend.
        - All preprocessing steps are applied via `compose()`.
        - Final output is passed through `to_output()` to apply tagging, UID, and output format.
        - Compatible with both NumPy and Torch backends.
        """
        # === Detect image type ===
        if isinstance(image, (str, Path)):
            # Image path
            img = self.image_io.read_image(image, framework=self.framework)
        elif isinstance(image, (np.ndarray, torch.Tensor)):
            # Loaded image
            img = image
        else:
            raise TypeError(f"[PreprocessorND] Unsupported input type: {type(image)}")

        img = self.compose(img)
        
        return self.to_output(img)
    
    def normalize_nd(self, image: ArrayLike) -> ArrayLike:
        """
        Normalize an ND image using the configured mode and optional clipping.

        Supports multiple normalization strategies, applied element-wise across the image.

        Modes
        -----
        - 'minmax' : (x - min) / (max - min)
        - 'zscore' : (x - mean) / std
        - 'robust' : (x - Q1) / (Q3 - Q1)

        Parameters
        ----------
        image : ndarray or torch.Tensor
            Input image to normalize.

        Returns
        -------
        ArrayLike
            Normalized image in the same framework as input (NumPy or Torch).

        Notes
        -----
        - Normalization mode is taken from `self.normalize_mode` (defined in `PreprocessorConfig`).
        - Clipping to `self.clip_range` is applied if `self.clip` is True.
        - Uses backend-specific implementation (`normalize_np` or `normalize_torch`).
        - A small epsilon is added to denominators to prevent division by zero.
        """
        eps = 1e-8
        mode, clip_range = self.normalize_mode, self.clip_range

        def normalize_np(x: np.ndarray) -> np.ndarray:
            if mode == "minmax":
                vmin, vmax = np.min(x), np.max(x)
                x = (x - vmin) / (vmax - vmin + eps)
            elif mode == "zscore":
                mu, sigma = np.mean(x), np.std(x) + eps
                x = (x - mu) / sigma
            elif mode == "robust":
                q1, q3 = np.percentile(x, [25, 75])
                x = (x - q1) / (q3 - q1 + eps)
            return np.clip(x, *clip_range) if self.clip  else x

        def normalize_torch(x: torch.Tensor) -> torch.Tensor:
            x = x.float()
            if mode == "minmax":
                vmin, vmax = x.min(), x.max()
                x = (x - vmin) / (vmax - vmin + eps)
            elif mode == "zscore":
                mu, sigma = x.mean(), x.std() + eps
                x = (x - mu) / sigma
            elif mode == "robust":
                q1 = torch.quantile(x.flatten(), 0.25)
                q3 = torch.quantile(x.flatten(), 0.75)
                x = (x - q1) / (q3 - q1 + eps)
            return torch.clamp(x, *clip_range) if self.clip  else x

        self.processor.function = normalize_torch if self.framework == "torch" else normalize_np

        return self.processor(image)    
    
    def stretch_contrast_nd(self, image: ArrayLike) -> ArrayLike:
        """
        Apply dynamic contrast stretching using percentile-based rescaling.

        The input image is rescaled based on low and high percentiles,
        then optionally clipped to a target output range.

        Parameters
        ----------
        image : ndarray or torch.Tensor
            Input image to stretch.

        Returns
        -------
        ArrayLike
            Contrast-stretched image in the same backend (NumPy or Torch).

        Notes
        -----
        - The percentiles `p_low` and `p_high` (e.g., 1% and 99%) are used to define the input range.
        - The output range is defined by `out_range` (default: [0, 1]).
        - Clipping is applied to the output if `self.clip` is True.
        - Uses NumPy backend even for Torch tensors to estimate percentiles safely.
        - Internally sets the processor function via `self.processor`.
        """
        eps = 1e-8
        p_low = self.p_low
        p_high = self.p_high
        out_range = self.out_range

        def stretch_np(x: np.ndarray) -> np.ndarray:
            vmin, vmax = np.percentile(x, [p_low, p_high])
            x_stretch = (x - vmin) / (vmax - vmin + eps)
            x_stretch = x_stretch * (out_range[1] - out_range[0]) + out_range[0]
            return np.clip(x_stretch, *out_range) if self.clip else x_stretch

        def stretch_torch(x: torch.Tensor) -> torch.Tensor:           
            vmin, vmax = map(float, np.percentile(x.float().cpu().numpy(), [p_low, p_high]))
            x_stretch = (x - vmin) / (vmax - vmin + eps)
            x_stretch = x_stretch * (out_range[1] - out_range[0]) + out_range[0]
            return torch.clamp(x_stretch, *out_range) if self.clip else x_stretch

        self.processor.function = stretch_torch if self.framework == "torch" else stretch_np

        return self.processor(image)
    
    def aggregate_nd(self, image: ArrayLike) -> ArrayLike:
        """
        Apply block-wise spatial aggregation (mean or median) along a layout-aware axis.

        Splits the image along a selected axis into non-overlapping blocks of fixed size,
        applies a reduction (mean or median), and appends a final block if the division is not exact.

        Behavior
        --------
        - Axis selection is guided by `self.keys`:
            * 'axial', 'coronal', or 'sagittal' → mapped to known axes.
            * 'auto' or None → uses the first available axis in (depth, height, width).
        - Aggregation mode is defined in `self.agg_mode`: "mean" or "median".
        - Block size is taken from `self.block_size` (default: 5).

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            ND image or batch to aggregate.

        Returns
        -------
        ArrayLike
            Aggregated image, with the same type (NumPy or Torch) as input.

        Raises
        ------
        ValueError
            If the target axis cannot be inferred or if the mode is unsupported.

        Notes
        -----
        - Handles both NumPy and Torch backends.
        - The residual block (if any) is averaged and appended at the end.
        - Maintains layout tracking if used inside `OperatorCore`-derived classes.
        """
        m = int(self.block_size)
        keys = self.keys
        mode = self.agg_mode
        
        # === Axis selection ===
        if keys == "auto" or keys is None:
            target = self.depth_axis or self.height_axis or self.width_axis
            if target is None:
                raise ValueError("No spatial axis could be inferred for averaging.")
        else:
            axis_map = {
                "axial": self.depth_axis,
                "coronal": self.height_axis,
                "sagittal": self.width_axis
            }
            target = axis_map.get(keys)
            if target is None:
                raise ValueError(f"Unknown axis key '{keys}'")

        # === Aggregator selector ===
        agg_func = torch.mean if self.framework == "torch" else np.mean
        if mode == "median":
            agg_func = torch.median if self.framework == "torch" else np.median

        def aggregate_numpy(x: np.ndarray) -> np.ndarray:
            
            x = np.moveaxis(x, target, 1)
            B, L = x.shape[:2]
            rest = x.shape[2:]
            stop = (L // m) * m

            sliced = x[:, :stop]
            blocks = sliced.reshape(B, stop // m, m, *rest)
            reduced = agg_func(blocks, axis=2)

            if stop < L:
                residual = x[:, stop:]
                residual_mean = agg_func(residual, axis=1, keepdims=True)
                reduced = np.concatenate([reduced, residual_mean], axis=1)

            return np.moveaxis(reduced, 1, target)

        def aggregate_torch(x: torch.Tensor) -> torch.Tensor:
            x = x.float().transpose(target, 1)
            B, L = x.shape[:2]
            rest = x.shape[2:]
            stop = (L // m) * m

            sliced = x[:, :stop]
            blocks = sliced.reshape(B, stop // m, m, *rest)
            if mode == "mean":
                reduced = blocks.mean(dim=2)
            elif mode == "median":
                reduced = blocks.median(dim=2).values
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            if stop < L:
                residual = x[:, stop:]
                residual_mean = residual.mean(dim=1, keepdim=True) if mode == "mean" \
                                else residual.median(dim=1).values.unsqueeze(1)
                reduced = torch.cat([reduced, residual_mean], dim=1)

            return reduced.transpose(1, target)

        self.processor.function = aggregate_torch if self.framework == "torch" else aggregate_numpy

        return self.processor(image)
    
    def gamma_correction_nd(self, image: ArrayLike) -> ArrayLike:
        """
        Apply gamma correction to an ND image using the configured gamma value.

        Adjusts image brightness by applying a power-law transformation to values in [0, 1].

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image, assumed to be normalized in the range [0, 1].
        gamma : float (from self.gamma)
            Gamma correction factor:
            - gamma > 1 → darkens the image.
            - gamma < 1 → brightens the image.

        Returns
        -------
        ArrayLike
            Gamma-corrected image, with the same type as input (NumPy or Torch).

        Notes
        -----
        - Input values are clipped to [0, 1] before applying the power transform.
        - The `gamma` value is taken from `self.gamma` (set via `PreprocessorConfig`).
        - Backend-specific function is assigned via `self.processor.function`.
        """
        def gamma_np(x: np.ndarray) -> np.ndarray:
            x = np.clip(x, 0.0, 1.0)
            return np.power(x, 1.0 / self.gamma)

        def gamma_torch(x: torch.Tensor) -> torch.Tensor:
            x = torch.clamp(x, 0.0, 1.0)
            return x.pow(1.0 / self.gamma)

        self.processor.function = gamma_torch if self.framework == "torch" else gamma_np

        return self.processor(image)

    def equalize_hist_nd(self, image: ArrayLike, nbins: int = 256) -> ArrayLike:
        """
        Apply slice-wise histogram equalization to an ND image.

        Uses `skimage.exposure.equalize_hist` on each slice independently
        to enhance local contrast by redistributing pixel intensities.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image to equalize. Should be normalized to [0, 1] beforehand.
        nbins : int, default 256
            Number of bins used for histogram computation.

        Returns
        -------
        ArrayLike
            Equalized image with the same type as input (NumPy or Torch).

        Notes
        -----
        - The operation is performed slice-wise for ND images (e.g., over depth or batch).
        - When using the Torch backend with `self.preprocess_fallback=True`, the image is
        temporarily converted to NumPy for equalization, and the processor is restored after.
        - The equalization is performed using `skimage.exposure.equalize_hist`.
        - The processor function and framework are dynamically updated during execution.
        """
        def hist_eq_np(x: np.ndarray) -> np.ndarray:
            return equalize_hist(np.clip(x, 0, 1), nbins=nbins)
        
        if self.preprocess_fallback and self.framework == "torch": 
            # Convert to numpy for histogram equalization
            # This is a workaround for the lack of histogram equalization in Torch
            image = self.convert_once(image,tag_as="input", framework="numpy",)
            self.processor.framework = "numpy"
            self.processor.strategy = "parallel"

        self.processor.function = hist_eq_np
        result = self.processor(image)

        if self.preprocess_fallback and self.framework == "torch":
            # Restore original framework
            self.processor.framework = self.framework
            self.processor.strategy = "auto"

        return result

    def remove_artifacts_2d(self, image: ArrayLike, framework: Optional[Framework] = None) -> ArrayLike:
        """
        Remove structural artifacts from a 2D or ND image using a layout-aware cleaner.

        Delegates the operation to `ArtifactCleanerND`, which handles common artifacts
        (e.g., stripes, edge bleed, acquisition noise) in 2D slices.

        Parameters
        ----------
        image : ndarray or torch.Tensor
            Input image (2D, 3D, or batch of 2D slices).
        framework : str or None, optional
            Override for backend to use ('numpy' or 'torch'). If None, uses current framework.

        Returns
        -------
        ArrayLike
            Cleaned image with the same type as input (NumPy or Torch).

        Notes
        -----
        - Uses `ArtifactCleanerND`, initialized with current layout and backend.
        - Automatically layout-aware (e.g., works for (C, H, W) or (B, C, H, W) inputs).
        - For ND images, cleaning is applied slice-wise (along appropriate axis).
        """
        fw = framework or self.framework        
        artifact_cleaner = ArtifactCleanerND(framework=fw,
                                             layout_name=self.layout_name,
                                             layout_framework=self.layout_framework,)        
        return artifact_cleaner(image)
    
    def perona_enhancing(self, image: ArrayLike, framework: Optional[Framework] = None) -> ArrayLike:
        """
        Apply Perona–Malik edge-preserving enhancement to an image.

        Enhances structures while preserving edges by anisotropic diffusion,
        using a custom Perona-Malik variant adapted to the current layout.

        Parameters
        ----------
        image : ndarray or torch.Tensor
            Input image to enhance.
        framework : str or None, optional
            Backend to use ('numpy' or 'torch'). Defaults to current framework.

        Returns
        -------
        ArrayLike
            Enhanced image with preserved edges and reduced noise/artifacts.

        Notes
        -----
        - Uses `PeronaEnhancer`, configured with current layout and backend.
        - Supports 2D, 3D, and batched inputs depending on `PeronaEnhancer` capabilities.
        - Layout handling ensures correct spatial axis selection and tagging.
        """
        fw = framework or self.framework
        perona = PeronaEnhancer(framework=fw,
                                layout_name=self.layout_name,
                                layout_framework=self.layout_framework,
                                layout_ensured_name=self.layout_ensured_name,)
        return perona(image)

    def local_contrast_nd(self, image: ArrayLike, kernel_size: int = 3, eps: float = 1e-8) -> ArrayLike:
        """
        Enhance local contrast by normalizing each pixel with its local mean and std.

        For each pixel, computes a local window (mean and std), then rescales:
            output = (x - mean_local) / (std_local + eps)

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image to enhance.
        kernel_size : int, default 3
            Size of the local window used to compute mean and standard deviation.
        eps : float, default 1e-8
            Small stabilizer to prevent division by zero.

        Returns
        -------
        ArrayLike
            Locally contrast-enhanced image, in the same framework as input.

        Notes
        -----
        - Uses `feature_extractor()` to compute local mean and std in a layout-aware way.
        - Compatible with 2D, 3D, and ND images using correct axis mapping.
        - Backend (`numpy` or `torch`) and layout handling are automatically respected.
        """
        def contrast_np(x: np.ndarray) -> np.ndarray:
            mean_local = feature_extractor(x, features =["mean"], framework=self.framework, output_format="numpy", 
                                           layout_name=self.layout_name, layout_framework=self.layout_framework, window_size=kernel_size, stack=False)   
            std_local = feature_extractor(x, features=["std"], framework=self.framework, output_format="numpy",
                                          layout_name=self.layout_name, layout_framework=self.layout_framework, window_size=kernel_size, stack=False,)         
            return (x - mean_local) / (std_local + eps)

        def contrast_torch(x: torch.Tensor) -> torch.Tensor:
            mean_local = feature_extractor(x, features =["mean"], framework=self.framework, output_format="torch", 
                                           layout_name=self.layout_name, layout_framework=self.layout_framework, window_size=kernel_size, stack=False)
            std_local = feature_extractor(x, features=["std"], framework=self.framework, output_format="torch",
                                          layout_name=self.layout_name, layout_framework=self.layout_framework, window_size=kernel_size, stack=False,)
            return (x - mean_local) / (std_local + eps)
        self.processor.function = contrast_torch if self.framework == "torch" else contrast_np
        return self.processor(image)
   

# ======================================================================
#                      Convenience wrapper
# ======================================================================

def preprocess(
    img: ArrayLike,
    processor_strategy: Optional[str] = None,
    framework: Framework = "numpy",
    output_format: Framework = "numpy",
    layout_name: str = "HWC",
    layout_framework: Framework = "numpy",
    layout_ensured_name: str = "HWC",
    clip: bool = False,
    normalize: bool = False,
    stretch: bool = False,
    denoise: bool = False,
    aggregate: bool = False,
    remove_artifacts: bool = False,
    local_contrast: bool = False,
    gamma_correct: bool = False,
    equalize: bool = False,
) -> ArrayLike:
    """
    Run the full ND-aware preprocessing pipeline with selected options.

    This is a convenience wrapper for instantiating and applying `PreprocessorND`
    with layout-aware and dual-backend configurations.

    Parameters
    ----------
    img : ArrayLike
        Input image (NumPy array or Torch tensor).
    processor_strategy : str, optional
        Processing strategy. Defaults to "parallel" (NumPy) or "torch" (Torch).
    framework : {'numpy', 'torch'}, default 'numpy'
        Backend used internally for preprocessing.
    output_format : {'numpy', 'torch'}, default 'numpy'
        Desired format of the output.
    layout_name : str, default 'HWC'
        Layout description of the input (e.g., 'HWC', 'CHW', etc.).
    layout_framework : {'numpy', 'torch'}, default 'numpy'
        Layout backend used for resolving axis names.
    layout_ensured_name : str, default 'HWC'
        Target layout name to enforce on the processed output.

    Flags (bool)
    ------------
    clip : bool
        Clip values to [0, 1].
    normalize : bool
        Apply global normalization (min-max, z-score, or robust).
    stretch : bool
        Perform percentile-based contrast stretching.
    denoise : bool
        Apply basic denoising (if defined in pipeline).
    aggregate : bool
        Perform spatial aggregation (e.g., block-based averaging).
    remove_artifacts : bool
        Apply 2D artifact removal per slice.
    local_contrast : bool
        Enhance local contrast using sliding window normalization.
    gamma_correct : bool
        Apply gamma correction.
    equalize : bool
        Perform histogram equalization (slice-wise).

    Returns
    -------
    ArrayLike
        Preprocessed image, with layout tracking and proper formatting.

    Notes
    -----
    - The order of execution is determined by internal pipeline priorities.
    - Uses `PreprocessorND`, configured dynamically from the provided flags.
    - All layout and framework behavior is handled via `LayoutConfig` and `GlobalConfig`.
    """
    
    # ====[ Fallback ]====
    processor_strategy=processor_strategy or "parallel" if framework == "numpy" else "torch"   
        
    # ====[ Configuration ]====
    proc_params: Dict[str, Any] = {"processor_strategy": processor_strategy,}
    layout_params: Dict[str, Any] = {"layout_name": layout_name, "layout_framework": layout_framework, "layout_ensured_name": layout_ensured_name, }
    global_params: Dict[str, Any] = {"framework": framework, "output_format": output_format, "normalize": normalize}    
    preprocess_params: Dict[str, Any] = {"clip": clip, "stretch": stretch, "denoise": denoise, "aggregate": aggregate, "remove_artifacts": remove_artifacts, 
                         "local_contrast": local_contrast, "gamma_correct": gamma_correct, "equalize": equalize}

    process = PreprocessorND(
                            preprocess_cfg=PreprocessorConfig(**preprocess_params),
                            img_process_cfg=ImageProcessorConfig(**proc_params),
                            layout_cfg=LayoutConfig(**layout_params),
                            global_cfg=GlobalConfig(**global_params),
                            ) 
    return process(img)