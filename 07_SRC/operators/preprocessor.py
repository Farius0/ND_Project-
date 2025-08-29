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
    Modular ND image preprocessing with dual-backend (NumPy/Torch) and layout-aware steps.

    Notes
    -----
    - Steps are toggled via PreprocessorConfig flags (e.g., clip, stretch, equalize, ...).
    - Execution order is defined by `self.pipeline` (name, function, priority) and can be updated.
    """

    def __init__(
        self,
        preprocess_cfg: PreprocessorConfig = PreprocessorConfig(),
        img_process_cfg: ImageProcessorConfig = ImageProcessorConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize the preprocessing pipeline.

        Parameters
        ----------
        preprocess_cfg : PreprocessorConfig
            Preprocessing flags and parameters.
        img_process_cfg : ImageProcessorConfig
            Processor strategy and options.
        layout_cfg : LayoutConfig
            Axis/layout configuration.
        global_cfg : GlobalConfig
            Global behavior (framework, output_format, device, etc.).
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
        Apply preprocessing pipeline sequentially using registered steps.

        Parameters
        ----------
        image : np.ndarray | torch.Tensor
            Input image.

        Returns
        -------
        Processed image.
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
        Update the execution priority of a preprocessing step.

        Parameters
        ----------
        step_name : str
            Name of the step as defined in `self.pipeline`.
        new_priority : int
            Lower values run earlier.
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
        Save the pipeline steps and their priorities to a JSON file.

        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file.
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
        Load the pipeline step priorities from a JSON file.

        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file containing steps and priorities.
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
        Run the preprocessing pipeline on an ND image or path.

        Parameters
        ----------
        image : ndarray | Tensor | str | Path
            Either an already-loaded image or a path to an image.

        Returns
        -------
        ndarray | Tensor
            Processed image with proper tagging/output formatting.
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
        Normalize values according to `normalize_mode` with optional clipping.

        Modes
        -----
        - 'minmax' : (x - min) / (max - min)
        - 'zscore' : (x - mean) / std
        - 'robust' : (x - Q1) / (Q3 - Q1)
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
        Stretch contrast using dynamic percentile rescaling.
        ND-compatible via ImageProcessor.
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
        Apply spatial averaging (mean or median) in ND images along a layout-aware axis.
        Supports slicing in blocks of m along the specified axis.
        
        Behavior
        --------
        - Selects the target axis via `keys` ('axial','coronal','sagittal') or auto.
        - Aggregates non-overlapping blocks of length `block_size` along that axis.
        - Appends a residual aggregate if the length is not divisible by `block_size`.        

        Parameters
        ----------
        image : np.ndarray | torch.Tensor
            ND image or batch.

        Returns
        -------
        Averaged image (same type as input).
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
        Apply gamma correction to an ND image.
        ND-compatible, layout-aware, via ImageProcessor.
        
        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image assumed normalized in [0, 1].
        gamma : float
            Gamma correction factor. >1 darkens, <1 brightens.

        Returns
        -------
        Image after gamma correction (same type).
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
        Histogram equalization (slice-wise) using skimage's equalize_hist.

        Notes
        -----
        - When running with Torch backend and `preprocess_fallback=True`, the
          image is temporarily converted to NumPy for equalization, then we
          restore the processor settings.
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
        Artifact removal (2D-first) via ArtifactCleanerND with current layout.

        Parameters
        ----------
        image : ndarray | Tensor
            Input image.
        framework : str | None
            Override backend ('numpy' or 'torch'). Defaults to current framework.
        """
        fw = framework or self.framework        
        artifact_cleaner = ArtifactCleanerND(framework=fw,
                                             layout_name=self.layout_name,
                                             layout_framework=self.layout_framework,)        
        return artifact_cleaner(image)
    
    def perona_enhancing(self, image: ArrayLike, framework: Optional[Framework] = None) -> ArrayLike:
        """
        Edge-preserving enhancement using Perona-Malik variant.
        """
        fw = framework or self.framework
        perona = PeronaEnhancer(framework=fw,
                                layout_name=self.layout_name,
                                layout_framework=self.layout_framework,
                                layout_ensured_name=self.layout_ensured_name,)
        return perona(image)

    def local_contrast_nd(self, image: ArrayLike, kernel_size: int = 3, eps: float = 1e-8) -> ArrayLike:
        """
        Local contrast enhancement using mean and standard deviation.

        Parameters
        ----------
        kernel_size : int, default 3
            Window size for local mean.
        eps : float, default 1e-8
            Stabilizer for division.
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
    Convenience entrypoint to run the ND preprocessor with flags.

    Notes
    -----
    - `processor_strategy` default: "parallel" for NumPy, "torch" for Torch,
      unless explicitly provided.
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