# ==================================================
# ===========  MODULE: feature_extractor  ==========
# ==================================================
from __future__ import annotations

from itertools import product
from typing import (
    Any,
    Callable,
    Dict,
    # Iterable,
    List,
    Literal,
    # Mapping,
    # MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from itertools import product
from skimage.feature import local_binary_pattern
from numpy.lib.stride_tricks import sliding_window_view
import torch.nn.functional as F,torch, numpy as np, pywt
from scipy.ndimage import grey_opening, grey_closing, generate_binary_structure

from core.operator_core import OperatorCore
from core.layout_axes import get_layout_axes
from core.config import (LayoutConfig, GlobalConfig, ImageProcessorConfig, 
    DiffOperatorConfig, EdgeDetectorConfig, NDConvolverConfig, FeatureConfig)
# from operators.image_io import ImageIO
from operators.image_processor import ImageProcessor
from operators.edge_detector import EdgeDetector
from operators.diff_operator import DiffOperator
from operators.gaussian import (GaussianKernelGenerator as kernel_generator, 
                                NDConvolver as convolver)

# Public API
__all__ = ["FeatureExtractorND", "feature_extractor"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Framework = Literal["auto", "numpy", "torch"]
Number = Union[int, float]

# ==================================================
# =============== FeatureExtractor =================
# ==================================================

class FeatureExtractorND(OperatorCore):
    """
    N-D feature extractor with dual-backend (NumPy/Torch) support and layout-aware tagging.

    Notes
    -----
    - Follows the project's OperatorCore conventions (convert_once, to_output, track, tagging).
    - Preserves dtype/device when feasible; avoids forcing float casts on stacking.
    - Feature specification supports three shapes:
      * string: single feature
      * list[str] or list[mixed]: independent features (seq)
      * dict: {"seq": [...], "comb": [... or list[list[str]]], "param": {grid}}
    """
    def __init__(
        self,
        feature_cfg: FeatureConfig = FeatureConfig(),
        edge_detector_cfg: EdgeDetectorConfig = EdgeDetectorConfig(),
        diff_operator_cfg: DiffOperatorConfig = DiffOperatorConfig(),
        ndconvolver_cfg: NDConvolverConfig = NDConvolverConfig(),
        img_process_cfg: ImageProcessorConfig = ImageProcessorConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize an N-Dimensional feature extractor with full operator configuration.

        Parameters
        ----------
        feature_cfg : FeatureConfig
            Feature extraction settings (GLCM, Gabor, wavelet, ridge, etc.).
        edge_detector_cfg : EdgeDetectorConfig
            Configuration for edge detectors used in some feature modes.
        diff_operator_cfg : DiffOperatorConfig
            Configuration for gradient-based operations (e.g., for ridge or flow).
        ndconvolver_cfg : NDConvolverConfig
            Convolution parameters for Gaussian or oriented filters.
        img_process_cfg : ImageProcessorConfig
            General image processing options and backend control.
        layout_cfg : LayoutConfig
            Layout specification (axis ordering, overrides).
        global_cfg : GlobalConfig
            Global framework and processing policy configuration.

        Returns
        -------
        None
        """

        # --- Config mirrors ---
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg
        self.conv_cfg: NDConvolverConfig = ndconvolver_cfg
        self.diff_cfg: DiffOperatorConfig = diff_operator_cfg
        self.img_process_cfg: ImageProcessorConfig = img_process_cfg
        self.edge_detect_cfg: EdgeDetectorConfig = edge_detector_cfg
        self.feature_cfg: FeatureConfig = feature_cfg

        # --- Resolved axes & layout meta ---
        self.axes: Dict[str, Any] = self.layout_cfg.resolve(include_meta=True)
        self.layout_name: str = self.axes.get("layout_name")
        self.layout: Dict[str, Any] = self.axes.get("layout")
        self.layout_framework: Literal["numpy", "torch"] = self.axes.get("layout_framework")

        self.channel_axis: Optional[int] = self.axes.get("channel_axis")
        self.batch_axis: Optional[int] = self.axes.get("batch_axis")
        self.direction_axis: Optional[int] = self.axes.get("direction_axis")
        self.height_axis: Optional[int] = self.axes.get("height_axis")
        self.width_axis: Optional[int] = self.axes.get("width_axis")
        self.depth_axis: Optional[int] = self.axes.get("depth_axis")
        
        # --- Store features-specific parameters ---
        self.features: Any = self.feature_cfg.features
        self.window_size: Optional[int] = self.feature_cfg.window_size
        self.n_bins: int = int(self.feature_cfg.n_bins)
        self.glcm_levels: int = int(self.feature_cfg.glcm_levels)
        self.glcm_mode: str = self.feature_cfg.glcm_mode
        self.gabor_freqs: Sequence[float] = self.feature_cfg.gabor_freqs
        self.low: float = float(self.feature_cfg.low)
        self.high: float = float(self.feature_cfg.high)
        self.sharpness: float = float(self.feature_cfg.sharpness)
        self.soft: bool = bool(self.feature_cfg.soft)
        self.wavelet: str = self.feature_cfg.wavelet
        self.level: int = int(self.feature_cfg.level)
        self.aggregate: Optional[str] = self.feature_cfg.aggregate
        self.beta: float = float(self.feature_cfg.beta)
        self.c: float = float(self.feature_cfg.c)
        self.ridge_mode: str = self.feature_cfg.ridge_mode
        self.n_components: int = int(self.feature_cfg.n_components)
        self.stack: bool = bool(self.feature_cfg.stack)
        self.combined: bool = bool(self.feature_cfg.combined)
        self.return_feat_names: bool = bool(self.feature_cfg.return_feat_names)
        self.block_mode: bool = bool(self.feature_cfg.block_mode)
        self.morpho_operation: str = self.feature_cfg.operation
        self.footprint: Optional[ArrayLike] = self.feature_cfg.footprint
        
        # --- Global mirrors ---
        self.framework: Literal["numpy", "torch"] = self.global_cfg.framework.lower()
        self.output_format: Literal["numpy", "torch"] = self.global_cfg.output_format.lower()
        self.add_batch_dim: Optional[bool] = self.global_cfg.add_batch_dim
        self.add_channel_dim: Optional[bool] = self.global_cfg.add_channel_dim
        self.normalize: bool = bool(self.global_cfg.normalize)
        self.verbose: bool = bool(self.global_cfg.verbose)        
        self.device: str = (
            "cuda"
            if (torch.cuda.is_available() and self.framework == "torch")
            else self.global_cfg.device
        )
        self.dim: int = int(self.conv_cfg.dim)
        self.size: Union[int, Sequence[int]] = self.conv_cfg.size
        self.sigma: Union[float, Sequence[float]] = self.conv_cfg.sigma
        self.angle: Union[float, Sequence[float]] = self.conv_cfg.angle
        self.complete_nms: bool = bool(self.edge_detect_cfg.complete_nms)
        self.as_float: bool = bool(self.edge_detect_cfg.as_float)
        self.mode: str = self.edge_detect_cfg.mode
        self.alpha: float = float(self.edge_detect_cfg.alpha)

        # ====[ Initialize OperatorCore with all axes ]====
        super().__init__(
            layout_cfg = self.layout_cfg,
            global_cfg = self.global_cfg,
        )  
        
        # ====[ Create KernelGenerator ]====
        self.kernel_gen: kernel_generator = kernel_generator(
            layout_cfg=LayoutConfig(
                layout_name="HW" if self.dim == 2 else "DHW",
                layout_framework=self.framework,
            ),
            global_cfg=self.global_cfg.update_config(output_format=self.framework),
        )
        self.kernel: ArrayLike = self.kernel_gen.generate(
            dim=self.dim,
            size=self.size,
            sigma=self.sigma,
            angle=self.angle,
            symmetry=False,
        )   
        
        # ====[ Create ImageProcessor ]====
        self.processor: ImageProcessor = ImageProcessor(
            img_process_cfg=self.img_process_cfg,
            layout_cfg=self.layout_cfg,
            global_cfg=self.global_cfg,
        )
        
        # ====[ Create Convolver ]====
        self.convolve: convolver = convolver(
            ndconvolver_cfg=self.conv_cfg,
            img_process_cfg=self.img_process_cfg,
            layout_cfg=self.layout_cfg,
            global_cfg=self.global_cfg,
        )
           
        # ====[ Create DiffOperator ]====
        self.diff: DiffOperator = DiffOperator(
            diff_operator_cfg=self.diff_cfg,
            ndconvolver_cfg=self.conv_cfg,
            layout_cfg=self.layout_cfg,
            global_cfg=self.global_cfg,
        )
        
        # ====[ Create EdgeDetector ]====
        self.edge: EdgeDetector = EdgeDetector(
            edge_detector_cfg=self.edge_detect_cfg,
            diff_operator_cfg=self.diff_cfg,
            ndconvolver_cfg=self.conv_cfg,
            img_process_cfg=self.img_process_cfg,
            layout_cfg=self.layout_cfg,
            global_cfg=self.global_cfg,
        )
        
    def __call__(self, image: ArrayLike) -> Union[ArrayLike, List[ArrayLike], Tuple[Any, Any]]:
        """
        Apply the configured feature extraction pipeline to an input image.

        Supports sequential and combined features, multiple parameter blocks, and flexible
        output modes (stacked, separate, or with feature name return).

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor) to extract features from.

        Returns
        -------
        ArrayLike | List[ArrayLike] | Tuple
            - If stack=True:
                → single array with features stacked along a new axis ("F").
            - If stack=False and return_feat_names=False:
                → list of feature arrays.
            - If return_feat_names=True:
                → tuple (features, names) where features can be stacked or separate.
        """
        image = self.convert_once(image, tag_as="input")
        features_blocks: Dict[str, Any] = {}
        fmap: List[ArrayLike] = []
        fname: List[str] = []
        
        features_raw: Any = (
            self.features if self.features != "all" else self._default_features()
        )        
        if not self.block_mode:
            features_blocks["block_1"] = features_raw
        else:
            features_blocks = features_raw
            
        for block_name, features_block in features_blocks.items():

            features = self._normalize_feature_spec(features_block)  
            
            param_grid: Dict[str, Sequence[Any]] = features.get("param", {})
            param_keys: List[str] = list(param_grid.keys())
            param_values: List[Sequence[Any]] = list(param_grid.values())
            param_combinations: List[Tuple[Any, ...]] = (list(product(*param_values)) if param_grid else [()])
        
            # === COMBINED MODE (chained features) ===
            if getattr(self, "combined", False) and "comb" in features and features["comb"]:  # Use self.combined
                out = self.output_format

                for param_combo in param_combinations:
                    name = ""
                    feat_name = ""
                    for k, v in zip(param_keys, param_combo):
                        setattr(self, k, v)  
                        name += f"_{k}_{v}"
                    
                    # ===[ Reset output format ]====   
                    output: Union[ArrayLike, Dict[str, ArrayLike]] = image
                    self.output_format = self.framework
                    
                    if "gaussian" in features["comb"] or "gaussian_eigen" in features["comb"]:
                        self.kernel = self.kernel_gen.generate(
                            dim=self.dim,
                            size=self.size, 
                            sigma=self.sigma,  
                            angle=self.angle, 
                            symmetry=False,
                        )   
                    
                    for i, feat in enumerate(features["comb"]): 
                        if i == len(features["comb"]) - 1:
                            self.output_format = out
                        if isinstance(output, dict):
                            output = output[list(output.keys())[0]]
                        output = self._extract(output, feat)
                        feat_name += f"_{feat}"
                    
                    if isinstance(output, dict):
                        fmap.extend(self._normalize_feature_axis(list(output.values())))
                        fname.extend([f"{f}_{block_name}_{feat_name}_{name}" for f in list(output.keys())])
                    else:  
                        fmap.append(self._normalize_feature_axis(output)) 
                        fname.append(f"{block_name}_{feat_name}_{name}")   
                
            # ========================================
            if "seq" in features and features["seq"]:
                # === STANDARD MODE (independent features) ===
                for feat in features["seq"]:
                    extracted = self._extract(image, feat)
                    if isinstance(extracted, dict):
                        fmap.extend(self._normalize_feature_axis(list(extracted.values())))
                        fname.extend([f"{f}_{block_name}_{feat}" for f in list(extracted.keys())])
                    else:          
                        fmap.append(self._normalize_feature_axis(extracted))
                        fname.append(f"{block_name}_{feat}")
        
         # === OUTPUT ===
        if len(fmap) == 0:
            raise ValueError(
                "[FeatureExtractorND] No feature produced. Check 'features' specification."
            )         
        if len(fmap) == 1 and not self.stack:
            return fmap[0] if not self.return_feat_names else (fmap[0], fname[0])
        elif len(fmap) > 1 and not self.stack:
            return fmap if not self.return_feat_names else (fmap, fname)
        else:
            stacked = torch.stack([f.float() for f in fmap], dim=-1) if all(isinstance(f, torch.Tensor) for f in fmap) else np.stack(fmap, axis=-1)
            # layout = ["layout_name", "ndim", "direction_axis", "batch_axis", "channel_axis", "depth_axis", "height_axis", "width_axis",]
            updated_layout: Dict[str, Any] = {}
            tracker = self.track(self.to_output(image)) if image.ndim == fmap[0].ndim else self.track(fmap[0])
            tag = tracker.get_tag().copy()
            # for keys in layout:                
            #     if keys in tag and tag[keys] is not None:
            #         updated_layout[keys] = tag[keys] if keys not in ["layout_name", "ndim"] \
            #             else tag[keys] + "F" if keys == "layout_name" else tag[keys] + 1
            updated_layout.update({"shape_after": stacked.shape, "status": "feature", 
                                   "layout_name": tag["layout_name"] + "F", "ndim": tag["ndim"] + 1,})    
            result = tracker.clone_to(stacked, updates=updated_layout)
            result.update_tag("feature_axis", len(stacked.shape) - 1, add_new_axis=True)
            return result.get() if not self.return_feat_names else (result.get(), fname)

    # ---------------------------------------------------------------------
    # Feature registry / normalization
    # ---------------------------------------------------------------------
    def _default_features(self) -> List[str]:
        return [
            "histogram", "kurtosis", "skewness", "lbp", "glcm", "wavelet",
            "gabor", "fft", "spectral_entropy", "structure_features",
            "gradient_edge", "sobel_edge", "laplacian_edge", "marr_hildreth_edge",
            "sign_change_edge", "combined_edge", "canny", "intensity",
            "mean", "std", "median", "gaussian", "gradient", "divergence",
            "laplacian", "sobel", "entropy", "sobel_gradient", "scharr", "bandpass",
            "grad_morph", "morpho_hat", "ridge", "local_similarity", "local_pca",
            "curvature", "hessian", "gaussian_eigen", "sobel_hessian",
        ]
    
    @staticmethod
    def _normalize_feature_spec(features: Any) -> Dict[str, Any]:
        """
        Normalize user-defined feature specifications to a unified internal structure.

        Parameters
        ----------
        features : str | list | dict
            Feature specification provided by the user. Supported formats:
            - str : single feature name
            - list : list of feature names, or nested lists for combined features
            - dict : with optional keys "seq", "comb", and "param"

        Returns
        -------
        dict
            Standardized dictionary with the following keys:
            - "seq" : list of independent features
            - "comb" : list of chained features (combined mode)
            - "param" : parameter grid for feature variations
        """

        out: Dict[str, Any] = {"seq": [], "comb": [], "param": {}}

        if isinstance(features, dict):
            for k in out:
                if k in features:
                    out[k] = features[k]

        elif isinstance(features, list):
            for f in features:
                if isinstance(f, list):
                    out["comb"].extend(f)
                elif isinstance(f, dict):
                    out["param"].update(f)
                else:
                    out["seq"].append(f)

        elif isinstance(features, str):
            out["seq"].append(features)

        else:
            raise ValueError(f"Unsupported feature specification: {type(features)}")

        return out


    def _normalize_feature_axis(self, data: Any) -> Any:
        """
        Recursively normalize feature axis containers.

        Parameters
        ----------
        data : Any
            Feature data, possibly a nested list of arrays.

        Returns
        -------
        Any
            Flattened or normalized version of the data, with recursive application
            if input is a list. Otherwise, returns data unchanged.
        """
        if isinstance(data, list):
            return [self._normalize_feature_axis(d) for d in data]
        return data

    # ------------------ Feature dispatch ------------------ #
    def _extract(self, image: ArrayLike, feat: str) -> Any:
        """
        Extract a specific feature from the input image.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).
        feat : str
            Feature name to extract. Must match one of the supported keys.

        Returns
        -------
        Any
            Feature output, format depends on the method used.

        Raises
        ------
        ValueError
            If the requested feature name is not supported.
        """
        base_kw = dict(
            window_size=self.window_size,
            framework=self.framework,
        )

        mapping: Dict[str, Callable[[ArrayLike], Any]] = {
            "std": lambda img: self.local_std(img),           
            "lbp": lambda img: self.lbp_simple(img),
            "mean": lambda img: self.local_mean(img), 
            "gaussian": lambda img: self.gaussian(img),
            "intensity": lambda img: self.intensity(img),
            "median": lambda img: self.local_median(img),            
            "entropy": lambda img: self.entropy_local(img),
            "kurtosis": lambda img: self.kurtosis_local(img),
            "skewness": lambda img: self.skewness_local(img),  
            "gaussian_eigen": lambda img: self.gaussian_eigen(img),       
            "canny": lambda img: self.edge_detect(img, key="canny"),            
            "sobel": lambda img: self.diff_operator(img, key="sobel"),                     
            "fft": lambda img: self.fft_magnitude_local(img, **base_kw),      
            "scharr": lambda img: self.diff_operator(img, key="scharr"),                     
            "hessian": lambda img: self.diff_operator(img, key="hessian"),
            "structure_features": lambda img: self.structure_features(img),
            "grad_morph": lambda img: self.gradient_morpho(img, **base_kw),
            "gradient": lambda img: self.diff_operator(img, key="gradient"),
            "spectral_entropy": lambda img: self.entropy_spectral_local(img),
            "laplacian": lambda img: self.diff_operator(img, key="laplacian"),
            "combined_edge": lambda img: self.edge_detect(img, key="combined"),           
            "gradient_edge": lambda img: self.edge_detect(img, key="gradient"),            
            "divergence": lambda img: self.diff_operator(img, key="divergence"),
            "laplacian_edge": lambda img: self.edge_detect(img, key="laplacian"),
            "sobel_edge": lambda img: self.edge_detect(img, key="sobel_gradient"),
            "sign_change_edge": lambda img: self.edge_detect(img, key="sign_change"), 
            "sobel_hessian": lambda img: self.diff_operator(img, key="sobel_hessian"),       
            "sobel_gradient": lambda img: self.diff_operator(img, key="sobel_gradient"),
            "marr_hildreth_edge": lambda img: self.edge_detect(img, key="marr_hildreth"),
            "curvature": lambda img: self.curvatures_nd(img, mode="eigen", eigen_select=2),
            "morpho_hat": lambda img: self.morpho_hat_nd(img, operation=self.morpho_operation, size=self.window_size, footprint=self.footprint,),
            "histogram": lambda img: self._handle_histogram(self.local_histogram_bins(img, n_bins=self.n_bins, **base_kw)),
            "ridge": lambda img: self.ridge_filter_nd(img, sigma=self.sigma, beta=self.beta, c=self.c, mode=self.ridge_mode),
            "glcm": lambda img: self.glcm_simple(img, window_size=self.window_size, levels=self.glcm_levels, mode=self.glcm_mode),
            "gabor": lambda img: self.gabor_bank_nd(img, frequencies=self.gabor_freqs, window_size=self.window_size, sigma=self.sigma,),                 
            "bandpass": lambda img: self.bandpass_filter_nd(img, f_low=self.low, f_high=self.high, soft=self.soft, sharpness=self.sharpness),                        
            "local_similarity" : lambda img: self.local_self_similarity_nd(img, window_size=self.window_size, method="ssd", reduction="mean"),  
            "local_pca" : lambda img: self.local_pca_nd(img, window_size=self.window_size, n_components=self.n_components, reduction="first_pc"),                 
            "wavelet": lambda img: self.wavelet_decomposition_local(img, wavelet=self.wavelet, level=self.level, mode='reflect', aggregate=self.aggregate),
        }

        if feat not in mapping:
            raise ValueError(f"[FeatureExtractorND] Unknown feature: '{feat}'")

        return mapping[feat](image)

    def _handle_histogram(self, h: ArrayLike) -> ArrayLike:
        """
        Normalize histogram layout for consistency across backends.

        Parameters
        ----------
        h : ArrayLike
            Histogram array, possibly with feature bins along the last axis.

        Returns
        -------
        ArrayLike
            Histogram with bins moved to the first axis (if NumPy); unchanged if Torch.
        """

        if isinstance(h, torch.Tensor):
            return h
        return np.moveaxis(h, -1, 0)
    
    # def _handle_gabor(self, g: ArrayLike) -> ArrayLike:
    #     """
    #     Normalize Gabor feature layout across backends.

    #     Parameters
    #     ----------
    #     g : ArrayLike
    #         Gabor response array, typically with frequency bands on the last axis.

    #     Returns
    #     -------
    #     ArrayLike
    #         Gabor responses with bands moved to the first axis (if NumPy); unchanged if Torch.
    #     """
    #     if isinstance(g, torch.Tensor):
    #         return g
    #     return np.moveaxis(g, -1, 0)

    #     if isinstance(g, torch.Tensor):
    #         return g
    #     return np.moveaxis(g, -1, 0)
    
    def edge_detect(self, image: ArrayLike, key: str) -> Any:
        """
        Apply edge detection to the input image using a specified method.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).
        key : str
            Name of the edge detection method to apply. Must be one of:
            {'gradient', 'sobel_gradient', 'laplacian', 'marr_hildreth',
            'sign_change', 'combined', 'canny'}.

        Returns
        -------
        Any
            Output of the edge detector, format depends on the configuration.

        Raises
        ------
        ValueError
            If an unknown method key is provided.
        """
        valid_methods = {"gradient", "sobel_gradient", "laplacian", "marr_hildreth", "sign_change", "combined", "canny"}

        if key not in valid_methods:
            raise ValueError(f"Unknown edge detection method '{key}'. Available methods: {sorted(valid_methods)}")

        self.edge.output_format = self.output_format
        self.edge.method = key
        return self.edge(image)
    
    def diff_operator(self, image: ArrayLike, key: str, output_format: str | None = None) -> Any:
        """
        Apply a differential operator (gradient, divergence, laplacian, etc.) to an image.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).
        key : str
            Name of the operator to apply. Must be one of:
            {'gradient', 'divergence', 'laplacian', 'sobel', 'sobel_gradient',
            'sobel_hessian', 'scharr', 'hessian'}.
        output_format : str or None, optional
            Override the output format (e.g., "torch", "numpy"). If None, use default.

        Returns
        -------
        Any
            Output of the selected operator. Type depends on configuration.

        Raises
        ------
        ValueError
            If an unknown operator name is given.
        """
        self.diff.output_format = self.output_format if output_format is None else output_format
        
        mapping: Dict[str, Any] = {
            "gradient": self.diff.gradient,
            "divergence": self.diff.divergence,
            "laplacian": self.diff.laplacian,
            "hessian": self.diff.hessian,
            "sobel": self.diff.sobel,
            "sobel_gradient": self.diff.sobel_gradient,
            "sobel_hessian": self.diff.sobel_hessian,
            "scharr": self.diff.scharr,
        }

        if key not in mapping:
            raise ValueError(
                "Unknown diff operator '{k}'. Available: "
                "gradient, divergence, laplacian, sobel, sobel_gradient, sobel_hessian, scharr, hessian".format(
                    k=key
                )
            )
        return mapping[key](image)
    
   # ------------------ Core helpers ------------------ #        
    def unfold_nd(
        self, image: ArrayLike, window_size: int, framework: Framework = "auto"
    ) -> ArrayLike:
        """
        Extract ND local patches using either torch or numpy backend.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image, with or without channel axis.
        window_size : int
            Size of the local sliding window along each spatial dimension.
        framework : str, optional
            'torch', 'numpy', or 'auto' (auto-detect based on image type).

        Returns
        -------
        patches : np.ndarray or torch.Tensor
            - If channel detected → (C, N, *window_shape)
            - If no channel → (N, *window_shape)
        """
        # === Detect framework ===
        if framework == "auto":
            framework = "torch" if torch.is_tensor(image) else "numpy"

        batch_axis = self.get_axis(image, "batch_axis")
        channel_axis = self.get_axis(image, "channel_axis")            
            
        # Padding spatial dimensions only
        pad: List[int] = []

        if framework == "torch":
            if not isinstance(image, torch.Tensor):
                raise TypeError("Expected torch.Tensor for torch mode.")
            
            for i in reversed(range(image.ndim)):
                if i == channel_axis:
                    pad.extend([0, 0])
                else:
                    pad.extend([window_size // 2, window_size // 2])
                    
            image = F.pad(image.unsqueeze(0).unsqueeze(0), pad=pad, mode="reflect").squeeze(0).squeeze(0)

            if batch_axis is not None:
                image = image.squeeze(batch_axis)

            has_channel = channel_axis is not None

            if has_channel and channel_axis != 0:
                image = image.movedim(channel_axis, 0)

            ndim = image.ndim - (1 if has_channel else 0)
            spatial_axes = list(range(1, image.ndim)) if has_channel else list(range(image.ndim))
 
            # Unfold spatial dimensions
            patches = image
            for axis in spatial_axes:
                patches = patches.unfold(axis, window_size, 1)

            patch_shape = [window_size] * ndim
            if has_channel:
                n_channels = image.shape[0]
                n_patches = patches.shape[1]
                patches = patches.contiguous().view(n_channels, -1, *patch_shape)
            else:
                n_patches = np.prod(patches.shape[:-ndim])
                patches = patches.contiguous().view(n_patches, *patch_shape)

            return patches

        elif framework == "numpy":
            if not isinstance(image, np.ndarray):
                raise TypeError("Expected np.ndarray for numpy mode.")
            
            for i in range(image.ndim):
                if i == channel_axis:
                    pad.append((0, 0))
                else:
                    pad.append((window_size // 2, window_size // 2))
                    
            image = np.pad(image, pad_width=pad, mode='reflect')    

            if batch_axis is not None:
                image = np.squeeze(image, axis=batch_axis)

            has_channel = channel_axis is not None
            ndim = image.ndim - (1 if has_channel else 0)
            window_shape = (window_size,) * ndim

            if has_channel and channel_axis != 0:
                image = np.moveaxis(image, channel_axis, 0)

            if has_channel:
                n_channels = image.shape[0]
                patches = sliding_window_view(image, window_shape=window_shape, axis=tuple(range(1, ndim + 1)))
                n_patches = np.prod(patches.shape[1:1 + ndim])
                patches = patches.reshape(n_channels, n_patches, *window_shape)
            else:
                patches = sliding_window_view(image, window_shape=window_shape, axis=tuple(range(image.ndim)))
                n_patches = np.prod(patches.shape[:-ndim])
                patches = patches.reshape(n_patches, *window_shape)

            return patches

        else:
            raise ValueError(f"Unsupported framework '{framework}'. Use 'torch', 'numpy', or 'auto'.")

    def _get_axes(self, arr: ArrayLike) -> List[int]:
        """
        Determine the spatial axes to apply differential operators on.

        Parameters
        ----------
        arr : np.ndarray | torch.Tensor
            Input image or tensor.

        Returns
        -------
        axes : list[int]
            List of axes eligible for gradient/divergence computation.
        """
        ndim = arr.ndim
        axes = list(range(ndim))

        if ndim == 2:
            return axes  # Default fallback for simple 2D

        tag = self.get_tag(arr, self.framework) if self.has_tag(arr, self.framework) else {}

        def to_positive(axis: Optional[int]) -> Optional[int]:
            return axis if axis is None or axis >= 0 else axis + ndim

        channel_ax = to_positive(tag.get("channel_axis", self.axes.get("channel_axis")))
        batch_ax = to_positive(tag.get("batch_axis", self.axes.get("batch_axis")))
        direction_ax = to_positive(tag.get("direction_axis", self.axes.get("direction_axis")))

        # Remove non-spatial axes
        for ax in (channel_ax, batch_ax, direction_ax):
            if ax is not None and ax in axes:
                axes.remove(ax)
        return axes

    # ------------------ Simple features ------------------ #

    def intensity(self, image: ArrayLike) -> ArrayLike:
        """
        Return the raw intensity of the input image.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).

        Returns
        -------
        ArrayLike
            Same image, possibly converted to the configured output format.
        """
        return self.to_output(image)

    def local_mean(self, image: ArrayLike, output_format: Literal["numpy", "torch"] | None = None) -> ArrayLike:
        """
        Compute the local mean of an image over spatial dimensions only.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).
        output_format : {'numpy', 'torch'}, optional
            Desired output format. If None, uses the default configuration.

        Returns
        -------
        ArrayLike
            Smoothed image with local mean filtering applied, in the requested format.
        """
        out = output_format or self.output_format
        
        self.convolve.strategy = "torch" if self.framework == "torch" else "uniform"
        
        spatial_ndim = len(self._get_axes(image)) # Exclude batch/channel from the kernel
        kernel_shape = [self.window_size] * spatial_ndim
        kernel_size = np.prod(kernel_shape)
        kernel = np.ones(kernel_shape) / kernel_size
        
        result = self.convolve(image, 
                             kernel,
                             sigma=self.sigma,
                             size=self.window_size,)

        return self.to_output(result, framework=out)

    def local_std(
        self,
        image: ArrayLike,
        framework: Literal["numpy", "torch"] | None = None,
        output_format: Literal["numpy", "torch"] | None = None
    ) -> ArrayLike:
        """
        Compute the local standard deviation over spatial dimensions only.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).
        framework : {'numpy', 'torch'}, optional
            Backend to use for internal computations.
        output_format : {'numpy', 'torch'}, optional
            Desired output format for the result.

        Returns
        -------
        ArrayLike
            Smoothed image of local standard deviations in the selected format.
        """
        fw = framework or self.framework
        out = output_format or self.output_format
        tracker = self.track(image)
        image_squared = tracker.copy_to(image**2).get()
        mean = self.local_mean(image, output_format=fw)
        mean_sq = self.local_mean(image_squared, output_format=fw)
        result = tracker.copy_to(((mean_sq - mean**2 + 1e-6) ** 0.5)).get()
        return self.to_output(result, out)

    def local_median(self, image: ArrayLike, output_format: Literal["numpy", "torch"] | None = None) -> ArrayLike:
        """
        Compute the local median over a sliding window, across spatial dimensions only.

        Applies a robust median filter using a NumPy fallback when necessary,
        and handles batch/channel dimensions appropriately.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).
        output_format : {'numpy', 'torch'}, optional
            Desired output format for the result. If None, uses the configured default.

        Returns
        -------
        ArrayLike
            Filtered image with local medians in the selected format.
        """
        is_torch = isinstance(image, torch.Tensor)
        
        out = output_format or self.output_format
        self.convolve.strategy = "median"
        self.convolve.processor.strategy = "auto"
        self.convolve.framework, self.convolve.output_format = "numpy", "numpy"
        self.convolve.processor.framework, self.convolve.processor.output_format = "numpy", "numpy"
        
        if is_torch:
            # Median filter fallback
            image = self.to_output(image, framework="numpy")
            
        # NumPy fallback
        result = self.convolve(image, 
                             self.kernel,
                             sigma=self.sigma,
                             size=self.window_size,)
        
        # Restore parameters
        self.convolve.strategy = "torch"
        self.convolve.framework, self.convolve.output_format = "torch", "torch"
        self.convolve.processor.framework, self.convolve.processor.output_format = "torch", "torch"
        
        return self.to_output(result, framework=out)
    
    def gaussian(self, image: ArrayLike, output_format: Literal["numpy", "torch"] | None = None) -> ArrayLike:
        """
        Apply a Gaussian filter over spatial dimensions using a predefined kernel.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).
        output_format : {'numpy', 'torch'}, optional
            Desired output format. If None, uses the configured default.

        Returns
        -------
        ArrayLike
            Smoothed image after Gaussian filtering, in the selected format.
        """
        out = output_format or self.output_format
        
        result = self.convolve(image, 
                             self.kernel,
                             sigma=self.sigma,)
        
        return self.to_output(result, framework=out)
    
    def gaussian_eigen(self, image: ArrayLike, framework: Literal["numpy", "torch"] | None = None) -> ArrayLike:
        """
        Compute curvature-based eigenvalues after Gaussian smoothing.

        Applies a Gaussian filter over the image, then extracts curvature
        eigenvalues using the selected backend.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).
        framework : {'numpy', 'torch'}, optional
            Backend to use for computation. If None, uses the default.

        Returns
        -------
        ArrayLike
            Array of curvature eigenvalues derived from the smoothed image.
        """
        fw = framework or self.framework
        gaussian = self.gaussian(image, output_format=fw)
        return self.curvatures_nd(gaussian, mode="eigen")

    def kurtosis_local(
        self,
        image: ArrayLike,
        framework: Literal["numpy", "torch"] | None = None,
        output_format: Literal["numpy", "torch"] | None = None
    ) -> ArrayLike:
        """
        Compute the local kurtosis over a sliding window (N-D compatible).

        Uses the definition of kurtosis as the normalized fourth central moment.
        Applies local mean and standard deviation, and tracks metadata throughout.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).
        framework : {'numpy', 'torch'}, optional
            Backend used for computation. If None, uses the configured default.
        output_format : {'numpy', 'torch'}, optional
            Desired format for the result. If None, uses the default.

        Returns
        -------
        ArrayLike
            Image of local kurtosis values, in the selected output format.
        """
        eps = 1e-6
        fw = framework or self.framework
        out = output_format or self.output_format
        tracker = self.track(image)
        mean = self.local_mean(image, output_format=fw)
        std = self.local_std(image, framework=fw, output_format=fw)
        diff = tracker.copy_to((image - mean) ** 4).get()
        fourth_moment = self.local_mean(diff, output_format=fw)
        kurtosis = tracker.copy_to((fourth_moment / (std ** 4 + eps))).get()
        return self.to_output(kurtosis, framework=out)

    def skewness_local(
        self,
        image: ArrayLike,
        framework: Literal["numpy", "torch"] | None = None,
        output_format: Literal["numpy", "torch"] | None = None
    ) -> ArrayLike:
        """
        Compute the local skewness over a sliding window (N-D compatible).

        Uses the third central moment normalized by the cube of the local standard deviation.
        The computation is backend-agnostic and preserves metadata via tracking.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).
        framework : {'numpy', 'torch'}, optional
            Backend to use for internal computation. If None, uses default.
        output_format : {'numpy', 'torch'}, optional
            Desired output format. If None, uses configured default.

        Returns
        -------
        ArrayLike
            Image of local skewness values in the selected output format.
        """
        eps = 1e-6
        fw = framework or self.framework
        out = output_format or self.output_format
        tracker = self.track(image)
        mean = self.local_mean(image, output_format=fw)
        std = self.local_std(image, framework=fw, output_format=fw)
        diff = tracker.copy_to((image - mean) ** 3).get()
        third_moment = self.local_mean(diff, output_format=fw)
        skew = tracker.copy_to((third_moment/ (std ** 3 + eps))).get()
        return self.to_output(skew, framework=out)
    
    # ------------------ Entropy (local / spectral) ------------------ #

    def entropy_local(self, image: ArrayLike, bins: int = 16) -> ArrayLike:
        """
        Compute local entropy over a sliding window using histogram-based estimation.

        The computation is N-D compatible and backend-aware (NumPy or Torch).
        Histogram bins are computed per local patch and entropy is estimated accordingly.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).
        bins : int, default=16
            Number of bins used to compute the local histogram.

        Returns
        -------
        ArrayLike
            Image of local entropy values.
        """
        eps = 1e-8

        def entropy_np(x: np.ndarray) -> float:
            hist, _ = np.histogram(x, bins=bins, range=(0, 1), density=False)
            hist = hist.astype(np.float32, copy=False)
            hist /= (hist.sum() + eps)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist + eps))

        def entropy_torch(x: torch.Tensor) -> torch.Tensor:
            hist = torch.histc(x.flatten(), bins=bins, min=0.0, max=1.0)
            hist = hist / (hist.sum() + eps)
            hist = hist[hist > 0]
            return -torch.sum(hist * torch.log2(hist + eps))

        self.processor.function = entropy_torch if self.framework == "torch" else entropy_np

        return self.processor(image)
  
    def entropy_spectral_local(self, image: ArrayLike) -> ArrayLike:
        """
        Compute local spectral entropy from FFT magnitude (N-D compatible).

        Applies a Fourier Transform over local patches and estimates entropy
        from the power spectrum, using the appropriate backend (NumPy or Torch).

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).

        Returns
        -------
        ArrayLike
            Image of local spectral entropy values.

        Raises
        ------
        ValueError
            If the input type or configured framework is unsupported.
        """
        if self.framework == "torch" or isinstance(image, torch.Tensor):
            return self._entropy_spectral_local_torch(image)
        elif self.framework == "numpy" or isinstance(image, np.ndarray):
            return self._entropy_spectral_local_numpy(image)
        else:
            raise ValueError("Unsupported input type or framework.")

    def _entropy_spectral_local_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute local spectral entropy using sliding windows in N-D Torch tensors.

        Applies FFT on local patches to estimate spectral entropy,
        accounting for the presence of a channel axis if defined.

        Parameters
        ----------
        image : torch.Tensor
            Input N-D image (with or without channel axis).

        Returns
        -------
        torch.Tensor
            Spectral entropy map with same spatial shape as input (channels preserved if present).
        """
        eps = 1e-8
        dims = image.dim()
        p = self.window_size // 2
        
        channel_axis = self.get_axis(image, "channel_axis")
        
        # Padding spatial dimensions only
        pad = []
        for i in reversed(range(dims)):
            if i == channel_axis:
                pad.extend([0, 0])
            else:
                pad.extend([p, p])

        padded = F.pad(image.unsqueeze(0).unsqueeze(0), pad=pad, mode="reflect").squeeze(0).squeeze(0)

        # Handle window shape depending on channel
        if channel_axis is not None:
            window_size = [1 if i == channel_axis else self.window_size for i in range(dims)]
        else:
            window_size = [self.window_size] * dims

        # Sliding window extraction
        unfold = padded
        for axis, size in enumerate(window_size):
            if size > 1:
                unfold = unfold.unfold(dimension=axis, size=size, step=1)

        if channel_axis is not None:
            patches = unfold.reshape(unfold.shape[0], -1, *([self.window_size] * (dims - 1)))
        else:
            patches = unfold.reshape(-1, *([self.window_size] * dims))

        # FFT on patches
        fft_axes = tuple(range(1, patches.dim())) if channel_axis is None else tuple(range(2, patches.dim()))
        fft_vals = torch.fft.fftn(patches, dim=fft_axes)
        mag = torch.abs(fft_vals)
        mag /= (mag.sum(dim=fft_axes, keepdim=True) + eps)

        # Entropy calculation
        entropy = -torch.sum(mag * torch.log2(mag + eps), dim=fft_axes)

        # Reshape back
        if channel_axis is not None:
            output_shape = [image.shape[channel_axis]] + [s for i, s in enumerate(image.shape) if i != channel_axis]
        else:
            output_shape = [s for s in image.shape]

        tracker = self.track(image)
        result = tracker.copy_to(entropy.reshape(output_shape)).get()
        return self.to_output(result)
        
    def _entropy_spectral_local_numpy(self, image: np.ndarray) -> np.ndarray:
        """
        Compute local spectral entropy using sliding windows in N-D NumPy arrays.

        Applies FFT on local patches to estimate spectral entropy,
        accounting for the presence of a channel axis if defined.

        Parameters
        ----------
        image : np.ndarray
            Input N-D image (with or without channel axis).

        Returns
        -------
        np.ndarray
            Spectral entropy map with same spatial shape as input (channels preserved if present).
        """
        eps = 1e-8
        ndim = image.ndim
        p = self.window_size // 2
        
        channel_axis = self.get_axis(image, "channel_axis")
        
        # Padding
        if channel_axis is not None:
            pad_width = [(0,0) if i == channel_axis else (1, 1) for i in range(ndim)]
        else:
            pad_width = [(p, p)] * ndim

        padded = np.pad(image, pad_width=pad_width, mode='reflect')

        # Handle window shape depending on channel
        if channel_axis is not None:
            window_shape = tuple(
                1 if i == channel_axis else self.window_size
                for i in range(image.ndim)
            )
        else:
            window_shape = (self.window_size,) * image.ndim

        # Apply sliding window
        patches = sliding_window_view(padded, window_shape=window_shape)

        # Collapse spatial dimensions to patches
        if channel_axis is not None:
            # Move channel back if needed
            patches = np.moveaxis(patches, channel_axis, 0)
            patches = patches.reshape(patches.shape[0], -1, *([self.window_size] * (image.ndim - 1)))
        else:
            patches = patches.reshape(-1, *([self.window_size] * image.ndim))

        # FFT over patch dimensions (skip first dim if multi-channel)
        fft_axes = tuple(range(1, patches.ndim)) if channel_axis is None else tuple(range(2, patches.ndim))
        fft_vals = np.fft.fftn(patches, axes=fft_axes)
        mag = np.abs(fft_vals)
        mag /= (np.sum(mag, axis=fft_axes, keepdims=True) + eps)

        # Entropy computation
        entropy = -np.sum(mag * np.log2(mag + eps), axis=fft_axes)

        # Reshape back
        if channel_axis is not None:
            output_shape = [image.shape[channel_axis]] + [s for i, s in enumerate(image.shape) if i != channel_axis] 
        else:
            output_shape = [s for s in image.shape]

        tracker = self.track(image)
        result = tracker.copy_to(np.moveaxis(entropy.reshape(output_shape), 0, channel_axis)).get() if channel_axis is not None \
            else tracker.copy_to(entropy.reshape(output_shape)).get()
        return self.to_output(result)

     # ------------------ LBP ------------------ #

    def lbp_simple(self, image: ArrayLike, radius: int = 1, mode: str = "default") -> ArrayLike:
        """
        Compute Local Binary Pattern (LBP) codes for 2D or 3D images.

        Parameters
        ----------
        image : torch.Tensor or np.ndarray
            Input 2D or 3D image.
        radius : int
            Radius for neighborhood (default=1 → 3x3 patch).
        mode : str
            'default', 'uniform', or 'rotation_invariant'.

        Returns
        -------
        torch.Tensor or np.ndarray
            LBP-coded feature map.
        """
        if self.framework == "torch" and isinstance(image, torch.Tensor):
            return self._lbp_torch_nd(image, radius=radius, mode=mode)
        else:
            return self._lbp_numpy_nd(image, radius=radius, mode=mode)
        
    def _lbp_torch_nd(self, image: torch.Tensor, radius: int = 1, mode: str = "default") -> torch.Tensor:
        """
        Compute Local Binary Pattern (LBP) codes for 2D or 3D images (Torch).
        
        Parameters
        ----------
        image : torch.Tensor
            ND image (2D or 3D).
        radius : int
            Neighborhood radius (default=1).
        mode : str
            'default' or 'uniform'.
        
        Returns
        -------
        torch.Tensor
            LBP-coded image.
        """
        ndim = image.ndim
        device = image.device

        channel_axis = self.get_axis(image, "channel_axis")  # Detect if channel exists

        # Padding spatial dimensions only
        pad = []
        for i in reversed(range(ndim)):  # F.pad expects reversed order
            if i == channel_axis:
                pad.extend([0, 0])
            else:
                pad.extend([radius, radius])

        padded = F.pad(image.unsqueeze(0).unsqueeze(0), pad=pad, mode="reflect").squeeze(0).squeeze(0)

        # Sliding window extraction
        window_sizes = [1 if (channel_axis is not None and i == channel_axis) else (2 * radius + 1) for i in range(ndim)]
        
        unfold = padded
        
        for axis, size in enumerate(window_sizes):
            if size > 1:
                unfold = unfold.unfold(dimension=axis, size=size, step=1)

        # Reshape patches
        if channel_axis is not None:
            # With channel: (C, ...) -> keep C
            patches = unfold.reshape(unfold.shape[channel_axis], -1, (2 * radius + 1) ** (ndim - 1))
            center_idx = ((2 * radius + 1) ** (ndim - 1)) // 2
            center = patches[:, :, center_idx]
            neighbors = torch.cat((patches[:, :, :center_idx], patches[:, :, center_idx + 1:]), dim=2)
            comparisons = (neighbors >= center.unsqueeze(2)).int()
            compare_axis = 2
        else:
            # Without channel
            patches = unfold.reshape(-1, (2 * radius + 1) ** ndim)
            center_idx = ((2 * radius + 1) ** ndim) // 2
            center = patches[:, center_idx]
            neighbors = torch.cat((patches[:, :center_idx], patches[:, center_idx + 1:]), dim=1)
            comparisons = (neighbors >= center.unsqueeze(1)).int()
            compare_axis = 1

        # LBP code computation
        if mode == 'default':
            weights = torch.tensor([1 << i for i in range(comparisons.shape[compare_axis])], device=device)
            lbp = (comparisons * weights).sum(dim=compare_axis)

        elif mode == 'uniform':
            diffs = comparisons.diff(dim=compare_axis)
            transitions = diffs.abs().sum(dim=compare_axis) + (comparisons[..., 0] != comparisons[..., -1]).int()
            lbp = (transitions <= 2).int()

        else:
            raise ValueError(f"Unsupported mode '{mode}' for torch LBP.")

        # Reshape output correctly
        if channel_axis is not None:
            output_shape = [image.shape[channel_axis]] + [s for i, s in enumerate(image.shape) if i != channel_axis]
        else:
            output_shape = [s for s in image.shape]
            
        tracker = self.track(image)
        result = tracker.copy_to(lbp.view(*output_shape)).get()
        return self.to_output(result)

    def _lbp_numpy_nd(self, image: np.ndarray, radius: int = 1, mode: str = "default") -> np.ndarray:
        """
        Compute Local Binary Pattern (LBP) codes for 2D or 3D images (NumPy).

        Parameters
        ----------
        image : np.ndarray
            ND image (2D or 3D).
        radius : int
            Neighborhood radius (default=1).
        mode : str
            'default' or 'uniform'.

        Returns
        -------
        np.ndarray
            LBP-coded image.
            
        """     
        ndim = image.ndim
        
        if ndim == 2:
            method = 'default' if mode == 'default' else 'uniform'
            tracker = self.track(image)
            result = tracker.copy_to(local_binary_pattern(image, P=8 * radius, R=radius, method=method)).get()
            return self.to_output(result)
        
        else:
            channel_axis = self.get_axis(image, "channel_axis")  # Detect channel if needed

            window_size = 2 * radius + 1

            # Padding
            if channel_axis is not None:
                pad_width = [(0,0) if i == channel_axis else (radius, radius) for i in range(ndim)]
            else:
                pad_width = [(radius, radius)] * ndim

            padded = np.pad(image, pad_width=pad_width, mode='reflect')

            # Extract patches
            if channel_axis is not None:
                view_shape = [1 if i == channel_axis else window_size for i in range(ndim)]
                patches = sliding_window_view(padded, window_shape=tuple(view_shape))
                patches = np.moveaxis(patches, channel_axis, 0)  # Channels first
                patches = patches.reshape(patches.shape[0], -1, window_size**(ndim-1))
                center_idx = (window_size**(ndim-1)) // 2
                center = patches[:, :, center_idx]
                neighbors = np.concatenate((patches[:, :, :center_idx], patches[:, :, center_idx+1:]), axis=-1)
            else:
                patches = sliding_window_view(padded, window_shape=(window_size,)*ndim)
                patches = patches.reshape(-1, window_size**ndim)
                center_idx = (window_size**ndim) // 2
                center = patches[:, center_idx]
                neighbors = np.concatenate((patches[:, :center_idx], patches[:, center_idx+1:]), axis=-1)

            # Comparisons
            comparisons = (neighbors >= center[..., np.newaxis]).astype(np.uint8, copy=False)

            # LBP coding
            if mode == 'default':
                weights = np.array([1 << i for i in range(comparisons.shape[-1])], dtype=np.uint32)
                lbp = np.sum(comparisons * weights, axis=-1)

            elif mode == 'uniform':
                diffs = np.diff(comparisons, axis=-1)
                transitions = np.sum(diffs != 0, axis=-1) + (comparisons[..., 0] != comparisons[..., -1])
                lbp = (transitions <= 2).astype(np.uint8)

            else:
                raise ValueError(f"Unsupported mode '{mode}' for numpy LBP.")

            # Reshape back
            if channel_axis is not None:
                output_shape = [image.shape[channel_axis]] + [s for i, s in enumerate(image.shape) if i != channel_axis]
            else:
                output_shape = [s for s in image.shape]
        
            tracker = self.track(image)
            result = tracker.copy_to(np.moveaxis(lbp.reshape(output_shape), 0, channel_axis)).get() if channel_axis is not None \
                else tracker.copy_to(lbp.reshape(output_shape)).get()
            return self.to_output(result)

    # ------------------ GLCM ------------------ #
    def glcm_simple(
        self,
        image: ArrayLike,
        window_size: int = 5,
        levels: int = 8,
        offsets: Optional[List[Tuple[int, ...]]] = None,
        mode: Literal["single", "mean"] = "single",
        framework: Framework = "auto",
    ) -> Dict[str, ArrayLike]:
        """
        Compute Haralick-like texture descriptors from local GLCMs (Gray-Level Co-occurrence Matrices).

        Supports N-dimensional images and automatically selects the backend (NumPy or Torch).
        Outputs selected texture descriptors (e.g., contrast, homogeneity, entropy) per window.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).
        window_size : int, default=5
            Size of the sliding window used to compute GLCMs.
        levels : int, default=8
            Number of gray levels used in quantization.
        offsets : list of tuple[int], optional
            List of relative offsets (directions) used to compute GLCMs.
            If None, a default set is inferred for the image dimensionality.
        mode : {'single', 'mean'}, default='single'
            - 'single': returns descriptors per offset.
            - 'mean'  : returns average over all offsets.
        framework : {'numpy', 'torch', 'auto'}, default='auto'
            Backend used for computation. 'auto' infers from the input type.

        Returns
        -------
        dict
            Dictionary of texture descriptor maps, keyed by descriptor name.
            For example: {'contrast': array, 'homogeneity': array, ...}
        """

        if framework == "torch" or (framework == "auto" and isinstance(image, torch.Tensor)):
            return self._glcm_torch_nd(image, window_size, levels, offsets, mode)
        else:
            return self._glcm_numpy_nd(image, window_size, levels, offsets, mode)

    # ==========================================================
    # Torch backend
    # ==========================================================

    def _glcm_torch_nd(
        self,
        image: torch.Tensor,
        window_size: int = 3,
        levels: int = 16,
        offsets: Optional[List[Tuple[int, ...]]] = None,
        mode: Literal["single", "mean"] = "mean",
        idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute local GLCM features (contrast, dissimilarity, ASM, homogeneity).

        Parameters
        ----------
        image : torch.Tensor
            Input tensor (2D or 3D), with or without channels.
        window_size : int
            Size of local patch window (must be odd).
        levels : int
            Number of quantization levels.
        offsets : list of tuples or None
            List of offsets to use. If None, uses standard 8-connectivity (2D) or 26-connectivity (3D).
        mode : str
            'single' (use first offset) or 'mean' (average over all offsets).
        idx : int
            Offset index.

        Returns
        -------
        dict
            Dictionary with feature maps: 'contrast', 'dissimilarity', 'asm', 'homogeneity'.
        """
        eps = 1e-8
        ndim = image.ndim
        device = image.device
        
        tracker = self.track(image)

        # === Detect channel axis if any ===
        channel_axis = self.get_axis(image, "channel_axis")

        # === Quantization ===
        image = torch.clamp((image * (levels - 1)).long(), 0, levels - 1)

        # === Padding only spatial dimensions ===
        pad: List[int] = []
        for i in reversed(range(ndim)):
            if i == channel_axis:
                pad.extend([0, 0])
            else:
                pad.extend([window_size // 2, window_size // 2])
        padded = F.pad(image.unsqueeze(0).unsqueeze(0), pad=pad, mode="reflect").squeeze(0).squeeze(0)

        # === Build sliding windows ===
        window_sizes = [1 if (channel_axis is not None and i == channel_axis) else window_size for i in range(ndim)]
        patches = padded
        for axis, size in enumerate(window_sizes):
            if size > 1:
                patches = patches.unfold(axis, size, step=1)

        if channel_axis is not None:
            # (C, n_patches, window_size^(ndim-1))
            patches = patches.reshape(image.shape[channel_axis], -1, (window_size ** (ndim - 1)))
            n_channels = patches.shape[0]
            n_patches = patches.shape[1]
            center_idx = (window_size ** (ndim - 1) - 1) // 2
        else:
            # (n_patches, window_size^ndim)
            patches = patches.reshape(-1, (window_size ** ndim))
            n_channels = None
            n_patches = patches.shape[0]
            center_idx = (window_size ** ndim - 1) // 2

        # === Define offsets ===
        if offsets is None:
            if ndim == 2 or (ndim == 3 and channel_axis is not None):
                offsets = [(0, 1), (1, 0), (1, 1), (-1, 1),
                        (0, -1), (-1, 0), (-1, -1), (1, -1)]
            else:
                offsets = [(dz, dy, dx) for dz, dy, dx in product([-1, 0, 1], repeat=3) if (dz, dy, dx) != (0, 0, 0)]

        if mode == 'single':
            offsets = [offsets[idx]]

        features_accumulator = {
            'contrast': 0.0,
            'dissimilarity': 0.0,
            'asm': 0.0,
            'homogeneity': 0.0
        }

        # === Precompute meshgrid for Haralick features ===
        i_idx, j_idx = torch.meshgrid(
            torch.arange(levels, device=device),
            torch.arange(levels, device=device),
            indexing='ij'
        )
        diff = (i_idx - j_idx).abs()

        # === Compute GLCM features per offset ===
        window_dims = torch.tensor([window_size] * (ndim-1 if channel_axis is not None else ndim), device=image.device)
        multipliers = torch.cumprod(torch.cat([torch.ones(1, device=image.device), window_dims.flip(0)]), dim=0)[:-1].flip(0)

        for offset in offsets:
            offset_tensor = torch.tensor(offset, device=device)
            shift = int((offset_tensor * multipliers).sum().item())
            
            if channel_axis is not None:
                src = patches[:, :, center_idx]
                dst_idx = center_idx + shift
                dst = patches[:, :, dst_idx]
                src = src.flatten()
                dst = dst.flatten()
            else:
                src = patches[:, center_idx]
                dst_idx = center_idx + shift
                dst = patches[:, dst_idx]

            # Encode (i,j) pair as single integer
            joint = levels * src + dst

            # Build GLCM
            glcm_flat = torch.zeros((n_patches if n_channels is None else n_patches * n_channels, levels * levels), device=device)
            glcm_flat.scatter_add_(1, joint.unsqueeze(-1), torch.ones_like(joint, dtype=torch.float32).unsqueeze(-1))
            glcm = glcm_flat.view(-1, levels, levels)
            glcm = glcm / (glcm.sum(dim=(1, 2), keepdim=True) + eps)

            # Haralick features
            contrast = (glcm * diff ** 2).sum(dim=(1, 2))
            dissimilarity = (glcm * diff).sum(dim=(1, 2))
            asm = (glcm ** 2).sum(dim=(1, 2))
            homogeneity = (glcm / (1.0 + diff)).sum(dim=(1, 2))

            features_accumulator['contrast'] += contrast
            features_accumulator['dissimilarity'] += dissimilarity
            features_accumulator['asm'] += asm
            features_accumulator['homogeneity'] += homogeneity

        # === Average over offsets ===
        for key in features_accumulator:
            features_accumulator[key] /= len(offsets)

        # === Reshape output correctly ===
        spatial_shape = [s for i, s in enumerate(image.shape) if i != channel_axis]
        if channel_axis is not None:
            output_shape = [image.shape[channel_axis]] + spatial_shape
        else:
            output_shape = spatial_shape

        for k in features_accumulator:
            result=tracker.copy_to(features_accumulator[k].view(*output_shape)).get()
            features_accumulator[k] = self.to_output(result)

        return features_accumulator

    # ==========================================================
    # NumPy backend
    # ==========================================================

    def _glcm_numpy_nd(
        self,
        image: np.ndarray,
        window_size: int = 3,
        levels: int = 16,
        offsets: Optional[List[Tuple[int, ...]]] = None,
        mode: Literal["single", "mean"] = "mean",
        idx: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Compute local GLCM features (contrast, dissimilarity, ASM, homogeneity) for NumPy arrays.

        Parameters
        ----------
        image : np.ndarray
            Input image (2D, 3D) with or without channels.
        window_size : int
            Size of the local patch window.
        levels : int
            Number of quantization levels.
        offsets : list of tuples or None
            Offsets to use. If None, uses standard 8 or 26 directions.
        mode : str
            'single' (first offset) or 'mean' (average over offsets).
        idx : int
            offset index.

        Returns
        -------
        dict
            Feature maps: 'contrast', 'dissimilarity', 'asm', 'homogeneity'.
        """
        eps = 1e-8
        ndim = image.ndim
        
        tracker = self.track(image)

        # === Detect channel axis ===
        channel_axis = self.get_axis(image, "channel_axis")

        # === Quantization ===
        image = np.clip((image * (levels - 1)).astype(np.int32, copy=False), 0, levels - 1)

        # === Padding spatial dimensions only ===
        pad_width: List[Tuple[int, int]] = []
        for i in range(ndim):
            if i == channel_axis:
                pad_width.append((0, 0))
            else:
                pad_width.append((window_size // 2, window_size // 2))
        padded = np.pad(image, pad_width=pad_width, mode='reflect')

        # === Extract patches ===
        if channel_axis is not None:
            view_shape = [1 if i == channel_axis else window_size for i in range(ndim)]
            patches = sliding_window_view(padded, window_shape=tuple(view_shape))
            patches = np.moveaxis(patches, channel_axis, 0)  # Channels first
            patches = patches.reshape(patches.shape[0], -1, (window_size ** (ndim - 1)))
            n_channels = patches.shape[0]
            n_patches = patches.shape[1]
            center_idx = (window_size ** (ndim - 1) - 1) // 2
        else:
            patches = sliding_window_view(padded, window_shape=(window_size,) * ndim)
            patches = patches.reshape(-1, (window_size ** ndim))
            n_channels = None
            n_patches = patches.shape[0]
            center_idx = (window_size ** ndim - 1) // 2

        # === Define offsets ===
        if offsets is None:
            if ndim == 2 or (ndim == 3 and channel_axis is not None):
                offsets = [(0, 1), (1, 0), (1, 1), (-1, 1),
                        (0, -1), (-1, 0), (-1, -1), (1, -1)]
            else:
                offsets = [(dz, dy, dx) for dz, dy, dx in product([-1, 0, 1], repeat=3) if (dz, dy, dx) != (0, 0, 0)]

        if mode == 'single':
            offsets = [offsets[idx]]

        features_accumulator = {
            'contrast': 0.0,
            'dissimilarity': 0.0,
            'asm': 0.0,
            'homogeneity': 0.0
        }

        # === Precompute meshgrid for Haralick features ===
        i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')
        diff = np.abs(i_idx - j_idx)

        # === Compute GLCM features ===
        window_dims = [window_size] * (ndim-1 if channel_axis is not None else ndim)
        multipliers = np.cumprod([1] + window_dims[::-1])[:-1][::-1]

        for offset in offsets:
            shift = sum(d * m for d, m in zip(offset, multipliers))

            if channel_axis is not None:
                src = patches[:, :, center_idx].reshape(-1)
                dst_idx = center_idx + shift
                dst = patches[:, :, dst_idx].reshape(-1)
            else:
                src = patches[:, center_idx]
                dst_idx = center_idx + shift
                dst = patches[:, dst_idx]

            # Encode (i,j) pair
            joint = levels * src + dst

            glcm_flat = np.zeros((n_patches if n_channels is None else n_patches * n_channels, levels * levels), dtype=np.float32)
            np.add.at(glcm_flat, (np.arange(glcm_flat.shape[0]), joint), 1)
            glcm = glcm_flat.reshape(-1, levels, levels)
            glcm /= (glcm.sum(axis=(1,2), keepdims=True) + eps)

            contrast = np.sum(glcm * diff**2, axis=(1,2))
            dissimilarity = np.sum(glcm * diff, axis=(1,2))
            asm = np.sum(glcm**2, axis=(1,2))
            homogeneity = np.sum(glcm / (1.0 + diff), axis=(1,2))

            features_accumulator['contrast'] += contrast
            features_accumulator['dissimilarity'] += dissimilarity
            features_accumulator['asm'] += asm
            features_accumulator['homogeneity'] += homogeneity

        # === Average over offsets ===
        for key in features_accumulator:
            features_accumulator[key] /= len(offsets)

        # === Reshape outputs ===
        spatial_shape = [s for i, s in enumerate(image.shape) if i != channel_axis]
        if channel_axis is not None:
            output_shape = [image.shape[channel_axis]] + spatial_shape
        else:
            output_shape = spatial_shape            

        for k in features_accumulator:
            result=tracker.copy_to(np.moveaxis(features_accumulator[k].reshape(output_shape), 0, channel_axis)).get() if channel_axis is not None \
                else tracker.copy_to(features_accumulator[k].reshape(output_shape)).get()
            features_accumulator[k] = self.to_output(result)

        return features_accumulator
    
    # ------------------ Local Histogram Bins ------------------ #    
    def local_histogram_bins(
        self,
        image: ArrayLike,
        window_size: int = 5,
        n_bins: int = 8,
        framework: Framework = "auto"
    ) -> ArrayLike:
        """
        Compute local histograms over a sliding window using a fixed number of bins.

        The function supports both NumPy and Torch backends, automatically selecting
        the appropriate implementation based on input type or explicit choice.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).
        window_size : int, default=5
            Size of the local sliding window.
        n_bins : int, default=8
            Number of histogram bins.
        framework : {'numpy', 'torch', 'auto'}, default='auto'
            Backend used for computation. If 'auto', inferred from the input type.

        Returns
        -------
        ArrayLike
            Local histograms with one histogram per window position.
            Output shape may include a new bin axis depending on backend.
        """
        if framework == "auto":
            framework = "torch" if torch.is_tensor(image) else "numpy"

        if framework == "torch":
            return self.local_histogram_bins_torch(image, window_size, n_bins)
        elif framework == "numpy":
            return self.local_histogram_bins_numpy(image, window_size, n_bins)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    def local_histogram_bins_torch(
        self,
        image: torch.Tensor,
        window_size: int = 5,
        n_bins: int = 8
    ) -> torch.Tensor:
        """
        Compute local histograms over sliding windows using PyTorch (N-D compatible, multi-channel aware).

        Each local patch is quantized into `n_bins`, and a normalized histogram is computed
        per spatial location, preserving the channel axis if present.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor (any number of dimensions).
        window_size : int, default=5
            Size of the local window applied over spatial axes.
        n_bins : int, default=8
            Number of bins used for quantization.

        Returns
        -------
        torch.Tensor
            Local histogram tensor with shape:
            - (C, ..., n_bins) if channel axis is present
            - (..., n_bins) otherwise
            All spatial dimensions are preserved, and an extra histogram axis is appended.
        """
        eps = 1e-8
        device = image.device
        ndim = image.ndim

        # Detect channel axis
        channel_axis = self.get_axis(image, "channel_axis")

        # Quantization
        img_q = torch.clamp((image * n_bins).long(), 0, n_bins - 1)

        # Padding spatial dimensions only
        pad = []
        for i in reversed(range(ndim)):
            if i == channel_axis:
                pad.extend([0, 0])
            else:
                pad.extend([window_size // 2, window_size // 2])
        padded = F.pad(img_q.unsqueeze(0).unsqueeze(0), pad=pad, mode="reflect").squeeze(0).squeeze(0)

        # Window size per dimension
        window_sizes = [1 if (channel_axis is not None and i == channel_axis) else window_size for i in range(ndim)]
        
        patches = padded
        for axis, size in enumerate(window_sizes):
            if size > 1:
                patches = patches.unfold(axis, size, step=1)

        if channel_axis is not None:
            patches = patches.reshape(image.shape[channel_axis], -1, window_size ** (ndim-1))
        else:
            patches = patches.reshape(-1, window_size ** ndim)

        # One-hot and histogram
        one_hot = F.one_hot(patches, num_classes=n_bins).float()
        hist = one_hot.sum(dim=-2) / (patches.shape[-1] + eps)  # (N, n_bins)

        # Reshape output
        spatial_shape = [s for i, s in enumerate(image.shape) if i != channel_axis]
        if channel_axis is not None:
            output_shape = [image.shape[channel_axis]] + spatial_shape
            hist = hist.view(*output_shape, n_bins)
        else:
            output_shape = spatial_shape
            hist = hist.view(*output_shape, n_bins)

        tracker = self.track(image)
        return self.to_output(tracker.copy_to(hist).get())

        
    def local_histogram_bins_numpy(
        self,
        image: np.ndarray,
        window_size: int = 5,
        n_bins: int = 8
    ) -> np.ndarray:
        """
        Compute local histograms over sliding windows using NumPy (N-D compatible, multi-channel aware).

        Each local patch is quantized into `n_bins`, and a normalized histogram is computed
        per spatial location. Handles presence of a channel axis and preserves original layout.

        Parameters
        ----------
        image : np.ndarray
            Input image (N-D NumPy array).
        window_size : int, default=5
            Size of the local window applied over spatial axes. Must be odd.
        n_bins : int, default=8
            Number of histogram bins used for quantization.

        Returns
        -------
        np.ndarray
            Local histogram array with shape:
            - (..., n_bins) if no channel axis
            - shape with n_bins appended to each channel slice otherwise
        """
        eps = 1e-8
        ndim = image.ndim

        channel_axis = self.get_axis(image, "channel_axis")

        if window_size % 2 == 0:
            raise ValueError("Window size must be odd.")

        img_q = np.clip((image * n_bins).astype(np.int32, copy=False), 0, n_bins - 1)

        # Padding
        pad_width = []
        for i in range(ndim):
            if i == channel_axis:
                pad_width.append((0, 0))
            else:
                pad_width.append((window_size // 2, window_size // 2))
        padded = np.pad(img_q, pad_width=pad_width, mode='reflect')

        # Sliding windows
        if channel_axis is not None:
            window_shape = tuple(1 if i == channel_axis else window_size for i in range(ndim))
            patches = sliding_window_view(padded, window_shape)
            patches = np.moveaxis(patches, channel_axis, 0)
            patches = patches.reshape(patches.shape[0], -1, window_size ** (ndim - 1))
        else:
            patches = sliding_window_view(padded, (window_size,) * ndim)
            patches = patches.reshape(-1, window_size ** ndim)

        # Histogram
        hist = np.stack([(patches == b).sum(axis=-1) for b in range(n_bins)], axis=-1).astype(np.float32, copy=False)
        hist /= (patches.shape[-1] + eps)

        spatial_shape = [s for i, s in enumerate(image.shape) if i != channel_axis]
        if channel_axis is not None:
            output_shape = [image.shape[channel_axis]] + spatial_shape
            hist = np.moveaxis(hist.reshape(*output_shape, n_bins), 0, channel_axis)
        else:
            output_shape = spatial_shape
            hist = hist.reshape(*output_shape, n_bins)

        tracker = self.track(image)
        return self.to_output(tracker.copy_to(hist).get())
    
    # ------------------ FFT ------------------ #    
    
    def fft_magnitude_local(
        self, image: ArrayLike, window_size: int = 5, framework: Framework = "auto"
    ) -> ArrayLike:
        """
        Compute local spectral energy (sum of FFT magnitudes) over ND patches.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            ND image with or without channels.
        window_size : int
            Size of local FFT window.
        framework : str
            'torch', 'numpy', or 'auto'.

        Returns
        -------
        array-like
            ND feature map of shape reduced by (window_size - 1) per spatial dimension.
        """
        
        if framework == "auto":
            framework = "torch" if torch.is_tensor(image) else "numpy"

        # === Get number of spatial dimensions (excluding channel if present)
        channel_axis = self.get_axis(image, "channel_axis")
        n_spatial_dims = image.ndim - (1 if channel_axis is not None else 0)
        fft_axes = tuple(range(-n_spatial_dims, 0))

        # === Extract ND patches
        patches = self.unfold_nd(image, window_size, framework=framework)
        is_multichannel = patches.ndim >= 3 and channel_axis is not None

        # === FFT + Magnitude + Sum
        if framework == "torch":
            fft_vals = torch.fft.fftn(patches, dim=fft_axes)
            magnitude = torch.abs(fft_vals)
            energy = magnitude.sum(dim=fft_axes)
        elif framework == "numpy":
            fft_vals = np.fft.fftn(patches, axes=fft_axes)
            magnitude = np.abs(fft_vals)
            energy = magnitude.sum(axis=fft_axes)
        else:
            raise ValueError(f"Unsupported framework '{framework}'.")

        # === Output shape
        spatial_shape = [s for i, s in enumerate(image.shape)
                        if i != channel_axis]

        if is_multichannel:
            output_shape = [image.shape[channel_axis]] + spatial_shape
            energy = energy.view(*output_shape) if framework == "torch" else np.moveaxis(energy.reshape(*output_shape), 0, channel_axis)
        else:
            output_shape = spatial_shape
            energy = energy.view(*output_shape) if framework == "torch" else energy.reshape(*output_shape)

        # === Track layout and return
        tracker = self.track(image)
        return self.to_output(tracker.copy_to(energy).get())

    # ------------------ Gabor ------------------ #

    def make_gabor_filter_nd(
        self,
        u: ArrayLike,
        size: int,
        frequency: float,
        theta: Optional[Union[float, Tuple[float, float]]] = None,
        sigma: Union[float, Sequence[float]] = 1.0,
        framework: Literal["numpy", "torch"] = "torch",
    ) -> ArrayLike:  
        """
        Create an ND Gabor filter (2D or 3D), torch or numpy.

        Parameters
        ----------
        size : int
            Size of the filter (must be odd).
        frequency : float
            Spatial frequency of the sinusoidal carrier.
        theta : float or tuple of floats, optional
            Orientation. 2D: float (angle in radians), 3D: tuple (theta, phi).
        sigma : float or list/tuple
            Standard deviation of Gaussian envelope. Single value or per-axis.
        framework : str
            'torch' or 'numpy'.

        Returns
        -------
        kernel : np.ndarray or torch.Tensor
            Gabor filter in (H, W) or (D, H, W) format.
        """
        if size % 2 == 0:
            raise ValueError("Size must be odd.")
        
        device = self.device
        
        spatial_axes = self._get_axes(u)
        dim = len(spatial_axes)

        if framework == "torch":
            xp = torch
            linspace = lambda: torch.linspace(-1, 1, steps=size, device=device)
            cos, sin, pi = torch.cos, torch.sin, torch.pi
            exp = torch.exp
            theta = torch.tensor(theta, device=device)
        elif framework == "numpy":
            xp = np
            linspace = lambda: np.linspace(-1, 1, num=size)
            cos, sin, pi = np.cos, np.sin, np.pi
            exp = np.exp
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        # Default theta
        if theta is None:
            if framework == "numpy":
                theta = 0.0 if dim == 2 else (0.0, 0.0)
            else:
                theta = torch.tensor(0.0) if dim == 2 else (torch.tensor(0.0), torch.tensor(0.0))

        # Build ND grid
        grid_1d = [linspace() for _ in range(dim)]
        mesh = xp.meshgrid(*grid_1d, indexing='ij')
        coords = xp.stack(mesh, axis=0)  # (dim, ...)

        # Carrier modulation
        if dim == 2:
            x, y = coords
            x_theta = x * cos(theta) + y * sin(theta)
        else:
            if not (isinstance(theta, (tuple, list)) and len(theta) == 2):
                raise ValueError("Theta must be a tuple (theta, phi) for 3D.")
            
            if framework == "torch" and not isinstance(theta, torch.Tensor):
                theta_, phi_ = theta
                theta_, phi_ = torch.tensor(theta_).to(device), torch.tensor(phi_).to(device)

            x, y, z = coords
            x_theta = (x * sin(theta_) * cos(phi_) +
                    y * sin(theta_) * sin(phi_) +
                    z * cos(theta_))

        carrier = xp.cos(2 * pi * frequency * x_theta)

        # Gaussian envelope
        if isinstance(sigma, (float, int)):
            sigma = [sigma] * dim
        elif isinstance(sigma, (list, tuple)):
            if len(sigma) != dim:
                raise ValueError(f"Sigma must have {dim} elements.")
        else:
            raise ValueError("Sigma must be float, int, list, or tuple.")

        envelope = exp(-sum((coords[i] ** 2) / (2 * sigma[i] ** 2) for i in range(dim)))

        gabor = carrier * envelope

        # Final shape: (H, W) or (D, H, W)
        
        layout_name = "HW" if dim == 2 else "DHW"
        axes_tags = get_layout_axes(self.framework, layout_name.upper())
        axes_tags.pop("name", None)
        axes_tags.pop("description", None)
        
        tagger = self.track(u)
        
        tracker = tagger.copy_to(gabor)
        tracker.update_tags({
            "status": "Kernel",
            "layout_name": layout_name,
            "shape_after": gabor.shape,
            **axes_tags
        })

        return tracker.get().to(device) if framework == "torch" else tracker.get()

    def gabor_bank_nd(
        self,
        image: ArrayLike,
        frequencies: Sequence[float],
        orientations: Optional[Sequence[Union[float, Tuple[float, float]]]] = None,
        window_size: int = 31,
        sigma: Union[float, Sequence[float]] = 1.0,
    ) -> Dict[str, ArrayLike]:
        """
        Apply a Gabor filter bank to a 2D or 3D image and return a dict of filtered responses.

        Parameters
        ----------
        image : torch.Tensor or np.ndarray
            Input image, 2D or 3D, with or without channels.
        frequencies : list of float
            List of spatial frequencies.
        orientations : list of float or tuple, optional
            List of orientations. If None, auto-generated (8 angles in 2D, 4x4 angles in 3D).
        window_size : int
            Size of the Gabor kernel (must be odd).
        sigma : float or list of float
            Standard deviation(s) of the Gaussian envelope.

        Returns
        -------
        dict[str, Tensor or ndarray]
            Dictionary mapping feature names to Gabor response maps.
        """
        framework = "torch" if isinstance(image, torch.Tensor) else "numpy"
        device = image.device if framework == "torch" else None

        spatial_axes = self._get_axes(image)
        spatial_ndim = len(spatial_axes)
        if spatial_ndim not in (2, 3):
            raise NotImplementedError("Only 2D or 3D Gabor filters are supported.")

        # === Auto-orientations ===
        if orientations is None:
            if spatial_ndim == 2:
                orientations = [i * np.pi / 8 for i in range(8)]
            else:
                theta_vals = np.linspace(0, np.pi, num=4)
                phi_vals = np.linspace(0, 2 * np.pi, num=4)
                orientations = [(theta, phi) for theta in theta_vals for phi in phi_vals]

        output_dict = {}

        for fi, freq in enumerate(frequencies):
            for oi, theta in enumerate(orientations):
                # Generate Gabor kernel
                kernel = self.make_gabor_filter_nd(
                    image,
                    size=window_size,
                    frequency=freq,
                    theta=theta,
                    sigma=sigma,
                    framework=framework,
                    device=device
                )

                # Apply convolution
                response = self.convolve(image, kernel)

                # Track and name output
                tracked = self.track(image).copy_to(response).get()
                named = self.to_output(tracked)

                # Naming
                if spatial_ndim == 2:
                    feature_name = f"gabor_f{fi}_t{oi}"
                else:
                    deg_t = int(np.round(np.degrees(theta[0])))
                    deg_p = int(np.round(np.degrees(theta[1])))
                    feature_name = f"gabor_f{fi}_t{deg_t}_p{deg_p}"

                output_dict[feature_name] = named

        return output_dict

    # ------------------ Bandpass Filter ------------------ #
    
    def bandpass_filter_nd(
        self,
        image: ArrayLike,
        f_low: float = 0.1,
        f_high: float = 0.5,
        soft: bool = False,
        sharpness: float = 10.0,
        framework: Framework = "auto",
    ) -> ArrayLike:
        """
        Apply a hard or soft bandpass filter in the frequency domain (ND), NumPy or Torch.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input ND image (without batch).
        f_low : float
            Low cutoff frequency.
        f_high : float
            High cutoff frequency.
        soft : bool
            If True, use soft sigmoid-based transition.
        sharpness : float
            Only used if soft=True. Controls the slope of the sigmoid.
        framework : str
            'torch', 'numpy', or 'auto'.

        Returns
        -------
        np.ndarray or torch.Tensor
            Filtered image (real-valued).
        """
        if framework == "auto":
            framework = "torch" if torch.is_tensor(image) else "numpy"
            
        channel_axis = self.get_axis(image, "channel_axis")

        ndim = image.ndim
        spatial_axes = [i for i in range(ndim) if i != channel_axis]

        shape = [image.shape[ax] for ax in spatial_axes]

        if framework == "torch":
            device = image.device
            freqs = [torch.fft.fftfreq(n, device=device) for n in shape]
            grid = torch.meshgrid(*freqs, indexing='ij')
            freq_radius = torch.sqrt(sum(g ** 2 for g in grid))

            if soft:
                mask = torch.sigmoid(sharpness * (freq_radius - f_low)) * \
                    (1 - torch.sigmoid(sharpness * (freq_radius - f_high)))
            else:
                mask = ((freq_radius >= f_low) & (freq_radius <= f_high)).float()
            
            if channel_axis is not None:
                mask = mask.unsqueeze(channel_axis)  # Ensure broadcastable to image shape

            spectrum = torch.fft.fftn(image, dim=spatial_axes)
            filtered = torch.fft.ifftn(spectrum * mask, dim=spatial_axes)

        elif framework == "numpy":
            freqs = [np.fft.fftfreq(n) for n in shape]
            grid = np.meshgrid(*freqs, indexing='ij')
            freq_radius = np.sqrt(sum(g ** 2 for g in grid))

            if soft:
                mask = 1 / (1 + np.exp(-sharpness * (freq_radius - f_low))) * \
                    (1 - 1 / (1 + np.exp(-sharpness * (freq_radius - f_high))))
            else:
                mask = ((freq_radius >= f_low) & (freq_radius <= f_high)).astype(np.float32, copy=False)
                
            if channel_axis is not None:    
                mask = np.expand_dims(mask, axis=channel_axis)

            spectrum = np.fft.fftn(image, axes=spatial_axes)
            filtered = np.fft.ifftn(spectrum * mask, axes=spatial_axes)

        else:
            raise ValueError(f"[bandpass_filter_nd] Unsupported framework: '{framework}'")
        
        tracker = self.track(image)
        return self.to_output(tracker.copy_to(filtered.real).get())

    # ------------------ Structure Tensor ------------------ #

    def structure_features(
        self, image: ArrayLike, key: str = "gradient", framework: Framework = "auto"
    ) -> Dict[str, ArrayLike]:
        """
        Compute structure tensor eigenvalue-based descriptors (ND-compatible, dual-framework).

        Parameters
        ----------
        image : np.ndarray | torch.Tensor
            Input image.
        key : str
            Differential operator to use for gradient ('gradient', 'sobel_gradient', etc.).
        framework : str
            'torch', 'numpy', or 'auto'.

        Returns
        -------
        dict
            Dictionary of descriptors (lambda1, lambda2, [lambda3], coherence, anisotropy).
        """
        eps = 1e-6

        if framework == "auto":
            framework = "torch" if torch.is_tensor(image) else "numpy"

        self.convolve.strategy = "torch" if framework == "torch" else "gaussian"
        
        tracker = self.track(image)

        # Get gradients via centralized operator
        grads: List[ArrayLike] = self.diff_operator(image, key=key, ouptut_format=self.framework)
        ndim = len(grads)

        # Compute tensor elements: smoothed products of gradients
        tensor_elems = [
            self.convolve(tracker.copy_to(grads[i] * grads[j]).get(), self.kernel)
            for i in range(ndim) for j in range(i, ndim)
        ]

        # Build tensor and compute descriptors
        if ndim == 2:
            a, b, d = tensor_elems
            trace = a + d
            delta = ((a - d) ** 2 + 4 * (b ** 2) + eps) ** 0.5 
            lambda1 = 0.5 * (trace + delta)
            lambda2 = 0.5 * (trace - delta)
            coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)

            return {
                "lambda1": self.to_output(tracker.copy_to(lambda1).get()),
                "lambda2": self.to_output(tracker.copy_to(lambda2).get()),
                "coherence": self.to_output(tracker.copy_to(coherence).get())
            }

        elif ndim == 3:
            a, b, c, d, e, f = tensor_elems
            if framework == "torch":
                J = torch.stack([
                    torch.stack([a, b, c], dim=-1),
                    torch.stack([b, d, e], dim=-1),
                    torch.stack([c, e, f], dim=-1)
                ], dim=-2)  # (..., 3, 3)
                eigvals = torch.linalg.eigvalsh(J)
            else:
                shape = a.shape
                J = np.zeros(shape + (3, 3), dtype=np.float32)
                J[..., 0, 0], J[..., 0, 1], J[..., 0, 2] = a, b, c
                J[..., 1, 0], J[..., 1, 1], J[..., 1, 2] = b, d, e
                J[..., 2, 0], J[..., 2, 1], J[..., 2, 2] = c, e, f
                eigvals = np.linalg.eigvalsh(J)

            lambda1, lambda2, lambda3 = eigvals[..., -1], eigvals[..., -2], eigvals[..., -3]
            anisotropy = (lambda1 - lambda3) / (lambda1 + eps)

            return {
                "lambda1": self.to_output(tracker.copy_to(lambda1).get()),
                "lambda2": self.to_output(tracker.copy_to(lambda2).get()),
                "lambda3": self.to_output(tracker.copy_to(lambda3).get()),
                "anisotropy": self.to_output(tracker.copy_to(anisotropy).get())
            }

        else:
            raise NotImplementedError("Only 2D and 3D structure tensor supported.")

    # ------------------ Wavelet Decomposition ------------------ #

    def wavelet_decomposition_local(
        self,
        image: ArrayLike,
        wavelet: str = "haar",
        level: int = 1,
        mode: str = "reflect",
        framework: Framework = "auto",
        aggregate: Optional[Literal["sum", "energy", "max"]] = None,
    ) -> Dict[str, ArrayLike]:
        """
        Perform local discrete wavelet decomposition (2D or 3D only) using PyWavelets.

        Parameters
        ----------
        image : np.ndarray | torch.Tensor
            Input image (2D or 3D).
        wavelet : str
            Wavelet type.
        level : int
            Decomposition depth.
        mode : str
            Boundary mode for PyWavelets.
        framework : str
            'auto', 'torch', or 'numpy'.
        aggregate : str or None
            Aggregation method for details ('sum', 'energy', 'max', or None).

        Returns
        -------
        dict
            Dictionary of wavelet components, optionally aggregated.
        """
        # === Framework selection ===
        is_torch_input = torch.is_tensor(image)
        if framework == "auto":
            framework = "torch" if is_torch_input else "numpy"

        if framework == "torch":
            image_np = self.to_output(image, framework="numpy")
        elif framework == "numpy":
            image_np = image
        else:
            raise ValueError(f"[wavelet_decomposition_local] Unsupported framework '{framework}'.")

        # === Validate dimension ===
        if image_np.ndim not in (2, 3):
            raise NotImplementedError("Only 2D and 3D images supported.")

        # === Decomposition ===
        coeffs = pywt.wavedecn(image_np, wavelet=wavelet, level=level, mode=mode)

        # === Separate results ===
        approx = coeffs[0]
        details = [v for d in coeffs[1:] for v in d.values()]

        # === Aggregation ===
        if aggregate is None:
            result = {"approx": approx}
            for i, detail in enumerate(coeffs[1:], start=1):
                for k, arr in detail.items():
                    result[f"detail_L{i}_{k}"] = arr
        else:
            if aggregate == "sum":
                agg = sum(details)
            elif aggregate == "energy":
                agg = sum(d**2 for d in details)
            elif aggregate == "max":
                agg = np.maximum.reduce(details)
            else:
                raise ValueError(f"Unsupported aggregation method: {aggregate}")
            result = {"approx": approx, f"detail_{aggregate}": agg}

        # === Tag and return ===
        tracker = self.track(image_np)
        return {k: self.to_output(tracker.copy_to(v).get()) for k, v in result.items()}

    # ------------------ Morphological Gradient ------------------ #
    
    def gradient_morpho(
        self, image: ArrayLike, window_size: int = 3, framework: Framework = "auto"
    ) -> ArrayLike:
        """
        Compute morphological gradient (dilation - erosion) over local patches.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image, with or without channels.
        window_size : int
            Size of the sliding window.
        framework : str
            'torch', 'numpy', or 'auto'.

        Returns
        -------
        gradient : same type as input
            Morphological gradient map.
        """
        # Detect framework
        if framework == "auto":
            framework = "torch" if torch.is_tensor(image) else "numpy"

        # Detect channel axis
        channel_axis = self.get_axis(image, "channel_axis")

        # Extract patches
        patches = self.unfold_nd(image, window_size, framework=framework)  # (N, *window_shape) or (C, N, *window_shape)

        # Aplatissement
        if channel_axis is not None:
            # (C, N, ...) → (C, N, patch_size)
            flat = patches.reshape(patches.shape[0], patches.shape[1], -1)
            dilation = flat.max(dim=-1).values if framework == "torch" else flat.max(axis=-1)
            erosion  = flat.min(dim=-1).values if framework == "torch" else flat.min(axis=-1)
            gradient = dilation - erosion
            output_shape = [image.shape[channel_axis]] + [s for i, s in enumerate(image.shape) if i != channel_axis]
            gradient = gradient.view(*output_shape) if framework == "torch" else np.moveaxis(gradient.reshape(*output_shape), 0, channel_axis)
        else:
            # (N, ...) → (N, patch_size)
            flat = patches.reshape(patches.shape[0], -1)
            dilation = flat.max(dim=-1).values if framework == "torch" else flat.max(axis=-1)
            erosion  = flat.min(dim=-1).values if framework == "torch" else flat.min(axis=-1)
            gradient = dilation - erosion
            output_shape = [s for s in image.shape]
            gradient = gradient.view(*output_shape) if framework == "torch" else gradient.reshape(*output_shape)

        # Reformat and return
        tracker = self.track(image)
        return self.to_output(tracker.copy_to(gradient).get())

    # ------------------ Morphological Top-Hat ------------------ #
    
    def morpho_hat_nd(
        self,
        image: ArrayLike,
        operation: Literal["tophat", "blackhat"] = "tophat",
        size: int = 3,
        footprint: Optional[ArrayLike] = None,
        framework: Framework = "auto",
    ) -> ArrayLike:
        """
        Perform ND morphological top-hat or black-hat transform (white/black residue).

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image (ND, with or without channel).
        operation : str
            'tophat' for white top-hat, 'blackhat' for black top-hat.
        size : int
            Size of the structuring element (must be odd).
        footprint : array-like or torch.Tensor, optional
            Structuring element. If None, uses a default ND ball.
        framework : str
            'torch', 'numpy', or 'auto' (based on image type).

        Returns
        -------
        np.ndarray or torch.Tensor
            Morphologically filtered image.
        """
        if framework == "auto":
            framework = "torch" if torch.is_tensor(image) else "numpy"

        if framework == "torch":
            return self.morpho_hat_nd_torch(image, operation=operation, size=size, footprint=footprint)
        elif framework == "numpy":
            return self.morpho_hat_nd_numpy(image, operation=operation, size=size, footprint=footprint)
        else:
            raise ValueError(f"[morpho_hat_nd] Unsupported framework: '{framework}'")

    def morpho_hat_nd_numpy(
        self, image: np.ndarray, operation: Literal["tophat", "blackhat"] = "tophat", size: int = 3, footprint: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Perform ND white or black top-hat transformation using NumPy + SciPy.

        Parameters
        ----------
        image : np.ndarray
            Input grayscale or multi-channel image.
        operation : str
            'tophat' or 'blackhat'.
        size : int
            Size of the structuring element if `footprint` is None.
        footprint : np.ndarray or None
            Optional custom structuring element (binary mask).

        Returns
        -------
        output : np.ndarray
            Morphological top-hat or black-hat result, same shape as input.
        """
        if image.ndim < 2:
            raise ValueError("Input image must be at least 2D")

        # Detect channel axis
        channel_axis = self.get_axis(image, "channel_axis")
        
        # Default ND footprint
        if footprint is None:
            ndim = image.ndim - 1 if channel_axis is not None else image.ndim
            footprint = generate_binary_structure(ndim, connectivity=1)
            footprint = np.ones((size,) * ndim, dtype=bool)

        # Apply per-channel if needed
        if channel_axis is not None:
            image = np.moveaxis(image, channel_axis, 0)
            result = []

            for c in image:
                if operation == "tophat":
                    filtered = grey_opening(c, footprint=footprint)
                    result.append(c - filtered)
                elif operation == "blackhat":
                    filtered = grey_closing(c, footprint=footprint)
                    result.append(filtered - c)
                else:
                    raise ValueError("Operation must be 'tophat' or 'blackhat'")

            result = np.stack(result, axis=0)
            result = np.moveaxis(result, 0, channel_axis)
        else:
            if operation == "tophat":
                filtered = grey_opening(image, footprint=footprint)
                result = image - filtered
            elif operation == "blackhat":
                filtered = grey_closing(image, footprint=footprint)
                result = filtered - image
            else:
                raise ValueError("Operation must be 'tophat' or 'blackhat'")

        return self.to_output(self.track(image).copy_to(result).get())

    def morpho_hat_nd_torch(
        self, image: torch.Tensor, operation: Literal["tophat", "blackhat"] = "tophat", size: int = 3, footprint: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform ND white or black top-hat transformation using Torch backend.

        Parameters
        ----------
        image : torch.Tensor
            Input tensor with or without channels.
        operation : str
            Either 'tophat' or 'blackhat'.
        size : int
            Size of the structuring element (must be odd).
        footprint : torch.Tensor or None
            Optional binary structuring element of shape (s, s, ...).

        Returns
        -------
        torch.Tensor
            Morphological top-hat or black-hat filtered image.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("Expected torch.Tensor")

        if image.ndim < 2:
            raise ValueError("Image must be at least 2D")

        # === Detect channel axis and move to dim 0 if needed ===
        channel_axis = self.get_axis(image, "channel_axis")
        
        if channel_axis is not None and channel_axis != 0:
            image = image.movedim(channel_axis, 0)

        has_channel = channel_axis is not None
        ndim = image.ndim - 1 if has_channel else image.ndim

        # === Define structuring element ===
        if footprint is None:
            shape = (size,) * ndim
            center = [s // 2 for s in shape]
            footprint = torch.zeros(shape, dtype=torch.bool, device=image.device)
            for idx in product(*[range(s) for s in shape]):
                if sum(abs(i - c) for i, c in zip(idx, center)) <= ndim:
                    footprint[idx] = True

        # === Extract local windows using unfold ===
        patches = self.unfold_nd(image, window_size=size, framework="torch")  # shape: (C, N, *window_shape) or (N, ...)

        # === Filter by min or max depending on operation ===
        masked_patches = patches[..., footprint]
        masked_flattened = masked_patches.view(masked_patches.shape[0], masked_patches.shape[1], -1 ) \
            if has_channel else masked_patches.view(masked_patches.shape[0], -1)
        filtered = masked_flattened.min(dim=-1).values if operation == "tophat" else masked_flattened.max(dim=-1).values
        
        if has_channel:
            output_shape = [image.shape[channel_axis]] + [s for i, s in enumerate(image.shape) if i != channel_axis]
        else:
            output_shape = [s for s in image.shape]

        # === Compute morphological residual ===
        if operation == "tophat":
            result = image - filtered.view(*output_shape)
        elif operation == "blackhat":
            result = filtered.view(*output_shape) - image
        else:
            raise ValueError("Operation must be 'tophat' or 'blackhat'")

        # === Restore original axis layout ===
        if channel_axis is not None and channel_axis != 0:
            result = result.movedim(0, channel_axis)

        return self.to_output(self.track(image).copy_to(result).get())

    # ------------------ Ridge Filter ------------------ #
    def ridge_filter_nd(
        self,
        image: ArrayLike,
        sigma: float = 1.0,
        beta: float = 0.5,
        c: float = 15.0,
        eps: float = 1e-8,
        mode: Literal["frangi", "sato", "meijering", "neg_eig"] = "frangi",
        framework: Framework = "auto",
    ) -> ArrayLike:
        """
        Enhance ridge-like structures using Hessian-based eigenvalue filtering (N-D compatible).

        Supports common vesselness/ridge detection modes such as Frangi, Sato, Meijering,
        with automatic backend dispatch (NumPy or Torch).

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or PyTorch tensor).
        sigma : float, default=1.0
            Standard deviation for Gaussian smoothing prior to Hessian computation.
        beta : float, default=0.5
            Sensitivity to the second eigenvalue (vesselness contrast).
        c : float, default=15.0
            Sensitivity to the Frobenius norm (background suppression).
        eps : float, default=1e-8
            Small constant to avoid numerical instability.
        mode : {'frangi', 'sato', 'meijering', 'neg_eig'}, default='frangi'
            Ridge enhancement formulation to apply.
        framework : {'numpy', 'torch', 'auto'}, default='auto'
            Computation backend. If 'auto', inferred from input type.

        Returns
        -------
        ArrayLike
            Ridge-enhanced image, in the same format as input or converted to default output format.
        """
        if framework == "auto":
            framework = "torch" if torch.is_tensor(image) else "numpy"
        if framework == "torch":
            return self.ridge_filter_nd_torch(image, sigma=sigma, beta=beta, c=c, eps=eps, mode=mode)
        else:
            return self.ridge_filter_nd_numpy(image, sigma=sigma, beta=beta, c=c, eps=eps, mode=mode)

    def ridge_filter_nd_torch(
        self,
        image: torch.Tensor,
        sigma: float = 1.0,
        beta: float = 0.5,
        c: float = 15.0,
        eps: float = 1e-8,
        mode: Literal["frangi", "sato", "meijering", "neg_eig"] = "frangi",
    ) -> torch.Tensor:
        """
        Enhance ridge-like structures using Hessian eigenvalues (Torch, ND-compatible).

        Parameters
        ----------
        image : torch.Tensor
            Input ND image (2D, 3D, etc.).
        sigma : float
            Gaussian smoothing scale.
        beta : float
            Frangi beta parameter.
        c : float
            Background suppression parameter.
        eps : float
            Numerical stability.
        mode : str
            'frangi', 'sato', 'meijering', or 'neg_eig'

        Returns
        -------
        torch.Tensor
            Ridge-enhanced image.
        """
        assert torch.is_tensor(image), "[ridge_filter_nd_torch] Input must be a torch.Tensor"

        if sigma > 0:
            self.convolve.strategy = "torch"
            image = self.convolve(image, kernel=self.kernel, sigma=sigma)

        # Hessian eigenvalues sorted
        eigvals_dict = self.curvatures_nd(image, mode="eigen", framework="torch")
        λ = torch.stack([eigvals_dict[k] for k in sorted(eigvals_dict)], dim=-1)

        if mode == "frangi":
            λ_sorted = torch.sort(torch.abs(λ), dim=-1).values
            λ1, λ2 = λ_sorted[..., -1], λ_sorted[..., -2] if λ.shape[-1] > 1 else λ_sorted[..., -1]
            Rb = (λ2 / (λ1 + eps)) ** 2
            S = (λ ** 2).sum(dim=-1)
            response = torch.exp(-Rb / (2 * beta ** 2)) * (1 - torch.exp(-S / (2 * c ** 2)))
            response = response * (λ[..., -1] < 0)

        elif mode == "sato":
            prod = torch.prod(λ, dim=-1)
            response = prod.abs()
            response[λ[..., -1] > 0] = 0

        elif mode == "meijering":
            response = λ[..., -1].abs()
            response[λ[..., -1] > 0] = 0

        elif mode == "neg_eig":
            λ_min = λ[..., 0]
            response = -λ_min
            response[λ_min > 0] = 0

        else:
            raise ValueError(f"[ridge_filter_nd_torch] Unknown mode '{mode}'")

        tracker = self.track(image)
        return self.to_output(tracker.copy_to(response).get())

    
    def ridge_filter_nd_numpy(
        self,
        image: np.ndarray,
        sigma: float = 1.0,
        beta: float = 0.5,
        c: float = 15.0,
        eps: float = 1e-8,
        mode: Literal["frangi", "sato", "meijering", "neg_eig"] = "frangi",
    ) -> np.ndarray:
        """
        Enhance ridge-like structures using Hessian eigenvalues (NumPy, ND-compatible).

        Parameters
        ----------
        image : np.ndarray
            Input ND image (2D, 3D, etc.).
        sigma : float
            Gaussian smoothing scale.
        beta : float
            Frangi beta parameter.
        c : float
            Background suppression parameter.
        eps : float
            Numerical stability.
        mode : str
            'frangi', 'sato', 'meijering', or 'neg_eig'

        Returns
        -------
        np.ndarray
            Ridge-enhanced image.
        """
        assert isinstance(image, np.ndarray), "[ridge_filter_nd_numpy] Input must be a NumPy array"

        if sigma > 0:
            self.convolve.strategy = "gaussian"
            image = self.convolve(image, kernel=self.kernel, sigma=sigma)

        # Hessian eigenvalues sorted
        eigvals_dict = self.curvatures_nd(image, mode="eigen", framework="numpy")
        λ = np.stack([eigvals_dict[k] for k in sorted(eigvals_dict)], axis=-1)

        if mode == "frangi":
            λ_sorted = np.sort(np.abs(λ), axis=-1)
            λ1 = λ_sorted[..., -1]
            λ2 = λ_sorted[..., -2] if λ.shape[-1] > 1 else λ1
            Rb = (λ2 / (λ1 + eps)) ** 2
            S = np.sum(λ ** 2, axis=-1)
            response = np.exp(-Rb / (2 * beta ** 2)) * (1 - np.exp(-S / (2 * c ** 2)))
            response[λ[..., -1] > 0] = 0

        elif mode == "sato":
            prod = np.prod(λ, axis=-1)
            response = np.abs(prod)
            response[λ[..., -1] > 0] = 0

        elif mode == "meijering":
            response = np.abs(λ[..., -1])
            response[λ[..., -1] > 0] = 0

        elif mode == "neg_eig":
            λ_min = λ[..., 0]
            response = -λ_min
            response[λ_min > 0] = 0

        else:
            raise ValueError(f"[ridge_filter_nd_numpy] Unknown mode '{mode}'")

        tracker = self.track(image)
        return self.to_output(tracker.copy_to(response).get())

    # ------------------ Local Self-Similarity ------------------ #
    
    def local_self_similarity_nd(
        self,
        image: ArrayLike,
        window_size: int = 3,
        method: Literal["ssd", "l1", "entropy"] = "ssd",
        reduction: Literal["mean", "max", "none"] = "mean",
        framework: Framework = "auto",
    ) -> ArrayLike:
        """
        Compute local self-similarity of an image using patch-wise difference.
        
        Parameters
        ----------
        image : torch.Tensor or np.ndarray
            Input image (ND, no assumption on channel).
        window_size : int or tuple
            Local patch size per spatial axis.
        method : str
            Similarity method: 'ssd', 'l1', 'entropy'.
        reduction : str
            Aggregation: 'mean', 'max', 'none'.
        framework : str
            'torch', 'numpy', or 'auto'.

        Returns
        -------
        similarity : Tensor or ndarray
            Local similarity map (same shape as image minus borders).
        """

        if framework == "auto":
            framework = "torch" if isinstance(image, torch.Tensor) else "numpy"
            
        # === Detect channel axis and move to dim 0 if needed ===
        channel_axis = self.get_axis(image, "channel_axis")

        has_channel = channel_axis is not None
        ndim = image.ndim - 1 if has_channel else image.ndim
        window_dim = (window_size,) * ndim
        center_idx = (window_size // 2,) * ndim
        
        # Extract patches
        patches = self.unfold_nd(image, window_size, framework=framework)  # (C, N, *w)
        center = patches[(...,) + center_idx]
        center = center[(...,) + (None,) * len(window_dim)]
        diff = patches - center

        if framework == "torch":

            if method == "ssd":
                sim = diff ** 2
            elif method == "l1":
                sim = torch.abs(diff)
            elif method == "entropy":
                sim = -torch.log1p(torch.abs(diff))
            else:
                raise ValueError(f"Unknown method '{method}'")

        elif framework == "numpy":

            if method == "ssd":
                sim = diff ** 2
            elif method == "l1":
                sim = np.abs(diff)
            elif method == "entropy":
                sim = -np.log1p(np.abs(diff))
            else:
                raise ValueError(f"Unknown method '{method}'")
            
        if reduction == "mean":
            result = sim.mean(dim=tuple(range(-len(window_dim), 0))) if framework == "torch" \
                else sim.mean(axis=tuple(range(-len(window_dim), 0)))
        elif reduction == "max":
            result = sim.amax(dim=tuple(range(-len(window_dim), 0))) if framework == "torch" \
                else sim.max(axis=tuple(range(-len(window_dim), 0)))
        elif reduction == "none":
            result = sim
        else:
            raise ValueError(f"Unknown reduction '{reduction}'")
        
        if has_channel : 
            output_shape =  [image.shape[channel_axis]] + [s for i, s in enumerate(image.shape) if i != channel_axis]
            result = result.view(*output_shape) if framework == "torch" else np.moveaxis(result.reshape(*output_shape), 0, channel_axis)
        else :
            output_shape = [s for s in image.shape]
            result = result.view(*output_shape) if framework == "torch" else np.reshape(result, output_shape)  
           
        tracker = self.track(image)
        return self.to_output(tracker.copy_to(result).get())
    
    # ------------------ Local PCA ------------------ #
    
    def local_pca_nd(
        self, 
        image: ArrayLike, 
        window_size: int=5, 
        n_components: int=1, 
        reduction: Literal["variance", "all", "first_pc"] = "first_pc", 
        framework: Framework="auto"
        ) -> ArrayLike:
        """
        Compute local PCA over sliding ND windows using torch or numpy.

        Parameters
        ----------
        image : torch.Tensor or np.ndarray
            Input ND image, with or without channel axis.
        window_size : int
            Size of local window (must be odd).
        n_components : int
            Number of PCA components to extract.
        reduction : str
            'first_pc', 'variance', or 'all'.
        framework : str
            'torch', 'numpy', or 'auto'.

        Returns
        -------
        torch.Tensor or np.ndarray
            PCA feature map(s), traced and converted to output_format.
        """
        if framework == "auto":
            framework = "torch" if torch.is_tensor(image) else "numpy"

        if window_size % 2 == 0:
            raise ValueError("Window size must be odd.")

        eps = 1e-6
        channel_axis = self.get_axis(image, "channel_axis")
        has_channel = channel_axis is not None
        spatial_shape = [s for i, s in enumerate(image.shape) if i != channel_axis]

        # === Extract local patches ===
        patches = self.unfold_nd(image, window_size, framework=framework)

        # === Compute PCA (Multichannel) ===
        if has_channel:
            patches_flat = patches.reshape(patches.shape[0], patches.shape[1], -1)
            patches_flat = patches_flat.permute(1, 2, 0) if framework == "torch" else patches_flat.transpose(1, 2, 0)
            patch_size = patches_flat.shape[1]

            mean = patches_flat.mean(dim=1, keepdim=True) if framework == "torch" else patches_flat.mean(axis=1, keepdims=True)
            centered = patches_flat - mean

            cov = torch.einsum('nij,nik->njk', centered, centered) / (patch_size - 1 + eps) if framework == "torch" \
                else np.einsum('nij,nik->njk', centered, centered) / (patch_size - 1 + eps)

            eigvals, eigvecs = torch.linalg.eigh(cov) if framework == "torch" else np.linalg.eigh(cov)

            if reduction == "first_pc":
                pc1 = eigvecs[..., -1]
                proj = torch.einsum('nij,nj->ni', centered, pc1) if framework == "torch" else np.einsum('nij,nj->ni', centered, pc1)
                output = proj
            elif reduction == "variance":
                output = eigvals[..., -n_components:].sum(dim=-1) if framework == "torch" else eigvals[..., -n_components:].sum(axis=-1)
            elif reduction == "all":
                output = eigvals[..., -n_components:]
            else:
                raise ValueError(f"[local_pca_nd] Unknown reduction mode '{reduction}'")

        # === Compute PCA (Unichannel) ===
        else:
            patches_flat = patches.reshape(patches.shape[0], -1)  # (N, patch_size)
            mean = patches_flat.mean(dim=1, keepdim=True) if framework == "torch" else patches_flat.mean(axis=1, keepdims=True)
            centered = patches_flat - mean

            if reduction == "first_pc":
                if framework == "torch":
                    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
                    output = U[:, 0] * S[0]
                else:
                    U, S, Vh = np.linalg.svd(centered, full_matrices=False)
                    output = U[:, 0] * S[0]
                
            elif reduction == "variance":
                output = centered.var(dim=1) if framework == "torch" else np.var(centered, axis=1)

            elif reduction == "all":
                output = centered
            else:
                raise ValueError(f"[local_pca_nd] Unknown reduction mode '{reduction}'")

        # === Reconstruire la sortie ===
        tracker = self.track(image)
        spatial = list(spatial_shape)

        if has_channel:
            if framework == "torch":
                output = output.view(*spatial, -1)
                output = output.permute((output.dim() - 1,) + tuple(range(output.dim() - 1)))
            else:
                output = output.reshape(*spatial, -1)
        else:
            output = output.view(*spatial, -1) if framework == "torch" else output.reshape(*spatial, -1)

        if reduction == "first_pc":
            output = output.squeeze(-1)

        return self.to_output(tracker.copy_to(output).get())

    # ------------------ Curvature Descriptors ------------------ #

    def curvatures_nd(
        self,
        image: ArrayLike,
        mode: Literal["mean", "gaussian", "eigen"] = "mean",
        framework: Framework = "auto",
        eigen_select: Union[str, int] = "all",
    ) -> Union[ArrayLike, Dict[str, ArrayLike]]:
        """
        Compute curvature descriptors (mean, Gaussian, or eigenvalues) from the Hessian matrix.
        Works for 2D, 3D, or ND images, torch or numpy, with or without channel.

        Parameters
        ----------
        image : torch.Tensor or np.ndarray
            Input image (with or without channel axis).
        mode : str
            'mean', 'gaussian', or 'eigen'
        framework : str
            'torch', 'numpy', or 'auto'
        eigen_select : str or int
            'all' or index of eigenvalue

        Returns
        -------
        Curvature map (same shape as input) or a dict of eigenvalue maps {λ1, λ2, ..., λD}
        """
        if framework == "auto":
            framework = "torch" if torch.is_tensor(image) else "numpy"

        self.convolve.strategy = "torch" if framework == "torch" else "gaussian"
        eps = 1e-6
        
        # === Get tracker ===
        tracker = self.track(image)

        # Compute Hessian matrix
        # H = self.diff.hessian(image, framework=framework, output_format=framework)
        H = self.diff.sobel_hessian(image, framework=framework, output_format=framework)

        # Detect number of spatial directions (D)
        if framework == "torch":
            assert H.shape[0] == H.shape[1], "[curvatures_nd] Unexpected Hessian shape (torch)"
            D = H.shape[0]
            if D == 2:
                Hxx, Hxy = H[0, 0], H[0, 1]
                Hyy = H[1, 1]
                tr = Hxx + Hyy
                det = Hxx * Hyy - Hxy * Hxy
                delta = torch.sqrt(tr * tr - 4 * det + eps)
                λ1 = 0.5 * (tr + delta)
                λ2 = 0.5 * (tr - delta)
                eigvals = torch.stack([λ1, λ2], dim=-1)
            else:
                # Reorder to (..., D, D)
                perm = list(range(2, H.ndim)) + [0, 1]
                Hr = H.permute(*perm)  # (spatial..., D, D)
                eigvals = torch.linalg.eigvalsh(Hr)  # (spatial..., D)
        else:
            assert H.shape[0] == H.shape[1], "[curvatures_nd] Unexpected Hessian shape (numpy)"
            D = H.shape[0]
            if D == 2:
                Hxx, Hxy = H[0, 0], H[0, 1]
                Hyy = H[1, 1]
                tr = Hxx + Hyy
                det = Hxx * Hyy - Hxy * Hxy
                delta = np.sqrt(tr * tr - 4 * det + eps)
                λ1 = 0.5 * (tr + delta)
                λ2 = 0.5 * (tr - delta)
                eigvals = np.stack([λ1, λ2], axis=-1)
            else:
                axes = tuple(range(2, H.ndim)) + (0, 1)
                Hr = np.transpose(H, axes)  # (spatial..., D, D)
                eigvals = np.linalg.eigvalsh(Hr)  # (spatial..., D)

        # === Compute output ===
        if mode == "mean":
            curvature = eigvals.mean(dim=-1) if framework == "torch" else eigvals.mean(axis=-1)
            return self.to_output(tracker.copy_to(curvature).get())

        elif mode == "gaussian":
            curvature = eigvals.prod(dim=-1) if framework == "torch" else eigvals.prod(axis=-1)
            return self.to_output(tracker.copy_to(curvature).get())

        elif mode == "eigen":
            # Return eigenvalues
            if eigen_select == "all":
                return {
                    f"λ{i+1}": self.to_output(tracker.copy_to(eigvals[..., i]).get())
                    for i in range(D)
                }
            elif isinstance(eigen_select, int) and eigen_select > 0 and eigen_select <= D:
                return {f"λ{eigen_select}": self.to_output(tracker.copy_to(eigvals[..., eigen_select-1]).get())
                        }

        else:
            raise ValueError(f"[curvatures_nd] Unknown mode '{mode}'")

    # ------------------ Region Statistics ------------------ #
    def region_statistics_nd(
        self,
        image: ArrayLike,
        labels: ArrayLike,
        stats: Sequence[str] = ("mean", "std", "min", "max", "volume"),
        framework: Framework = "auto",
    ) -> Dict[str, ArrayLike]:
        """
        Compute region-wise statistics from a labeled ND image.

        Parameters
        ----------
        image : torch.Tensor or np.ndarray
            Input ND image (with or without channels).
        labels : torch.Tensor or np.ndarray
            Same spatial shape as image (excluding channels).
        stats : list of str
            List of statistics to compute: 'mean', 'std', 'min', 'max', 'volume'.
        framework : str
            'torch', 'numpy', or 'auto'.

        Returns
        -------
        dict
            Dictionary {stat: array}, shape (num_labels,) or (C, num_labels) if multichannel.
        """
        if framework == "auto":
            framework = "torch" if torch.is_tensor(image) else "numpy"

        eps = 1e-6
        channel_axis = self.get_axis(image, "channel_axis")
        has_channel = channel_axis is not None

        # Flatten spatial & channel dimensions
        if has_channel:
            if framework == "torch":
                x = image.movedim(channel_axis, 0)  # (C, ...)
            else:
                x = np.moveaxis(image, channel_axis, 0)  # (C, ...)
            C = int(x.shape[0])
        else:
            if framework == "torch":
                x = image.unsqueeze(0)  # (1, ...)
            else:
                x = image[None, ...]
            C = 1

        flat_img = x.reshape(C, -1)  # (C, N)
        labels_flat = labels.reshape(-1)
        num_labels = int(labels_flat.max()) + 1

        result: Dict[str, ArrayLike] = {}

        if framework == "torch":
            import torch_scatter  # type: ignore

            one = torch.ones_like(labels_flat, dtype=flat_img.dtype)

            for stat in stats:
                if stat in ("mean", "std", "count"):
                    sum_vals = torch.zeros((C, num_labels), device=flat_img.device)
                    sum_vals.scatter_add_(1, labels_flat.expand(C, -1), flat_img)

                    counts = torch.zeros((num_labels,), device=flat_img.device)
                    counts.scatter_add_(0, labels_flat, one)

                    if stat == "mean":
                        mean = sum_vals / (counts.unsqueeze(0) + eps)
                        result["mean"] = mean if C > 1 else mean.squeeze(0)
                    elif stat == "count":
                        result["count"] = counts

                if stat == "std":
                    # reuse sum_vals and counts computed above
                    sum_vals = torch.zeros((C, num_labels), device=flat_img.device)
                    sum_vals.scatter_add_(1, labels_flat.expand(C, -1), flat_img)
                    counts = torch.zeros((num_labels,), device=flat_img.device)
                    counts.scatter_add_(0, labels_flat, one)

                    mean = sum_vals / (counts.unsqueeze(0) + eps)
                    centered = flat_img - mean[:, labels_flat]  # (C, N)
                    sq_diff = centered**2
                    var_vals = torch.zeros((C, num_labels), device=flat_img.device)
                    var_vals.scatter_add_(1, labels_flat.expand(C, -1), sq_diff)
                    std = torch.sqrt(var_vals / (counts.unsqueeze(0) + eps))
                    result["std"] = std if C > 1 else std.squeeze(0)

                if stat == "min":
                    val = torch.full((C, num_labels), float("inf"), device=flat_img.device)
                    val = torch.minimum(
                        val.scatter_reduce(
                            1, labels_flat.expand(C, -1), flat_img, reduce="amin", include_self=False
                        ),
                        val,
                    )
                    result["min"] = val if C > 1 else val.squeeze(0)

                if stat == "max":
                    val = torch.full((C, num_labels), float("-inf"), device=flat_img.device)
                    val = torch.maximum(
                        val.scatter_reduce(
                            1, labels_flat.expand(C, -1), flat_img, reduce="amax", include_self=False
                        ),
                        val,
                    )
                    result["max"] = val if C > 1 else val.squeeze(0)

                if stat == "volume":
                    ones = torch.ones_like(labels_flat, dtype=torch.float32)
                    vol = torch.zeros((num_labels,), device=flat_img.device)
                    vol.scatter_add_(0, labels_flat, ones)
                    result["volume"] = vol

            tracker = self.track(image)
            for k in list(result.keys()):
                result[k] = self.to_output(tracker.copy_to(result[k]).get())
            return result

        # NumPy branch
        counts = np.bincount(labels_flat.astype(np.int64), minlength=num_labels).astype(np.float64, copy=False)

        if ("mean" in stats) or ("std" in stats) or ("count" in stats):
            sums = np.stack(
                [
                    np.bincount(labels_flat.astype(np.int64, copy=False), weights=flat_img[c], minlength=num_labels)
                    for c in range(C)
                ]
            )  # (C, L)
            if "mean" in stats:
                mean = sums / (counts + 1e-6)
                result["mean"] = mean if has_channel else mean[0]
            if "count" in stats:
                result["count"] = counts

        if "std" in stats:
            mean = sums / (counts + 1e-6)
            centered = flat_img - mean[:, labels_flat.astype(np.int64, copy=False)]
            sq = centered**2
            var = np.stack(
                [
                    np.bincount(labels_flat.astype(np.int64, copy=False), weights=sq[c], minlength=num_labels)
                    for c in range(C)
                ]
            )
            std = np.sqrt(var / (counts + 1e-6))
            result["std"] = std if has_channel else std[0]

        if ("min" in stats) or ("max" in stats):
            order = np.argsort(labels_flat, kind="mergesort")
            lbl_sorted = labels_flat[order]
            unique_lbls, starts = np.unique(lbl_sorted, return_index=True)
            starts = np.concatenate([starts, [lbl_sorted.size]])

            if "min" in stats:
                mins = np.full((C, num_labels), np.inf, dtype=flat_img.dtype)
                for c in range(C):
                    vals_sorted = flat_img[c, order]
                    out = np.minimum.reduceat(vals_sorted, starts[:-1])
                    mins[c, unique_lbls] = out
                result["min"] = mins if has_channel else mins[0]

            if "max" in stats:
                maxs = np.full((C, num_labels), -np.inf, dtype=flat_img.dtype)
                for c in range(C):
                    vals_sorted = flat_img[c, order]
                    out = np.maximum.reduceat(vals_sorted, starts[:-1])
                    maxs[c, unique_lbls] = out
                result["max"] = maxs if has_channel else maxs[0]

        tracker = self.track(image)
        for k in list(result.keys()):
            result[k] = self.to_output(tracker.copy_to(result[k]).get())
        return result
        
    def adaptive_threshold_nd(
        self,
        image: ArrayLike,
        window_size: int = 5,
        C: float = 0.0,
        method: Literal["mean", "gaussian", "median"] = "mean",
        framework: Framework = "auto",
    ) -> ArrayLike:
        """
        Perform ND adaptive thresholding with local mean/gaussian/median strategies.

        Parameters
        ----------
        image : torch.Tensor | np.ndarray
            Input ND image, with or without channel axis.
        window_size : int
            Local window size for threshold computation (must be odd).
        C : float
            Constant subtracted from local threshold.
        method : str
            Thresholding method: 'mean', 'gaussian', or 'median'.
        framework : str
            Backend to use: 'torch', 'numpy', or 'auto'.

        Returns
        -------
        mask : torch.Tensor | np.ndarray
            Binarized image of same spatial shape (with channel if originally present).
        """
        if framework == "auto":
            framework = "torch" if torch.is_tensor(image) else "numpy"
        
        if method not in ("mean", "gaussian", "median"):
            raise ValueError(f"[adaptive_threshold_nd] Unknown method '{method}'")

        # === Get correct function ===
        if method == "mean":
            local_fn = self.local_mean
        elif method == "gaussian":
            local_fn = lambda x: self.convolve(x, self.kernel)
        elif method == "median":
            local_fn = self.local_median

        # === Compute local threshold ===
        threshold = local_fn(image, window_size=window_size, framework=framework)
        
        # === Compute binary mask ===
        mask = (image > (threshold - C)).astype(np.float32, copy=False) if framework == "numpy" \
            else (image > (threshold - C)).float()

        # === Track & return ===
        tracker = self.track(image)
        return self.to_output(tracker.copy_to(mask).get())

    def region_based_stats_nd(
        self,
        image: ArrayLike,
        labels: ArrayLike,
        stats: Sequence[str] = ("mean",),
        framework: Framework = "auto",
    ) -> Dict[str, ArrayLike]:
        """
        Compute region-based statistics over labeled regions (N-D compatible, multi-channel aware).

        Automatically selects the appropriate backend (NumPy or Torch) based on the input
        or user-specified preference. Supports standard statistics over segmented regions.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image (scalar or multi-channel), shape (C, ...) or (...,).
        labels : np.ndarray or torch.Tensor
            Label map with same spatial shape as the image (excluding channels).
        stats : Sequence[str], default=('mean',)
            List of statistics to compute per region. Supported values:
            {'mean', 'std', 'min', 'max', 'count'}
        framework : {'numpy', 'torch', 'auto'}, default='auto'
            Backend to use for computation. If 'auto', inferred from the input type.

        Returns
        -------
        dict of str → ArrayLike
            Dictionary mapping each requested statistic to a tensor or array of shape:
            - (n_labels,) if scalar image
            - (n_labels, C) if multi-channel image
        """
        if framework == "auto":
            framework = "torch" if torch.is_tensor(image) else "numpy"

        if framework == "torch":
            return self.region_based_stats_nd_torch(image, labels, stats)

        elif framework == "numpy":
            return self.region_based_stats_nd_numpy(image, labels, stats)

        else:
            raise ValueError(f"[region_based_stats_nd] Unsupported framework '{framework}'")


    def region_based_stats_nd_torch(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        stats: Sequence[str] = ("mean",)
    ) -> Dict[str, torch.Tensor]:
        """
        Compute region-based statistics using the Torch backend (N-D compatible, multi-channel aware).

        Supports basic per-region statistics such as mean, std, min, max, and count.
        Handles explicit channel axes using movedim and vectorized reduction with scatter operations.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor, shape (C, ...) if channel axis is defined, or (...,) for scalar input.
        labels : torch.Tensor
            Integer label map of same spatial shape as image (excluding channels).
        stats : Sequence[str], default=('mean',)
            List of statistics to compute per region. Supported values:
            {'mean', 'std', 'min', 'max', 'count'}

        Returns
        -------
        dict of str → torch.Tensor
            Dictionary mapping each requested statistic to a tensor of shape:
            - (n_labels, C) if multi-channel,
            - (n_labels,) if single-channel.
        """  
        channel_axis = self.get_axis(image, "channel_axis")
        
        if channel_axis is not None:
            image = image.movedim(channel_axis, 0)  # (C, ...)
            channels = image.shape[0]
        else:
            image = image.unsqueeze(0)  # Add fake channel axis
            channels = 1

        labels = labels.to(dtype=torch.long)
        flat_img = image.reshape(channels, -1)      # (C, N)
        flat_lbl = labels.flatten()                 # (N,)
        n_labels = int(flat_lbl.max().item()) + 1

        result: Dict[str, torch.Tensor] = {}
        one = torch.ones_like(flat_lbl, dtype=flat_img.dtype)

        for stat in stats:
            if stat == "mean" or stat == "count" or stat == "std":
                sum_vals = torch.zeros((channels, n_labels), device=image.device)
                sum_vals.scatter_add_(1, flat_lbl.expand(channels, -1), flat_img)

                counts = torch.zeros((n_labels,), device=image.device)
                counts.scatter_add_(0, flat_lbl, one)

                if stat == "mean":
                    mean = sum_vals / (counts.unsqueeze(0) + 1e-6)
                    result[stat] = mean if channels > 1 else mean.squeeze(0)
                elif stat == "count":
                    result[stat] = counts

            if stat == "std":
                mean = sum_vals / (counts.unsqueeze(0) + 1e-6)
                centered = flat_img - mean[:, flat_lbl]  # (C, N)
                sq_diff = centered**2
                var_vals = torch.zeros((channels, n_labels), device=image.device)
                var_vals.scatter_add_(1, flat_lbl.expand(channels, -1), sq_diff)
                std = torch.sqrt(var_vals / (counts.unsqueeze(0) + 1e-6))
                result[stat] = std if channels > 1 else std.squeeze(0)

            if stat == "min":
                val = torch.full((channels, n_labels), float("inf"), device=image.device)
                val = torch.minimum(val.scatter_reduce(1, flat_lbl.expand(channels, -1), flat_img, reduce='amin', include_self=False), val)
                result[stat] = val if channels > 1 else val.squeeze(0)

            if stat == "max":
                val = torch.full((channels, n_labels), float("-inf"), device=image.device)
                val = torch.maximum(val.scatter_reduce(1, flat_lbl.expand(channels, -1), flat_img, reduce='amax', include_self=False), val)
                result[stat] = val if channels > 1 else val.squeeze(0)

        return result
    
    def region_based_stats_nd_numpy(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        stats: Sequence[str] = ("mean",)
    ) -> Dict[str, np.ndarray]:
        """
        Compute region-based statistics from a labeled image using NumPy (N-D, multi-channel supported).

        Supports basic statistics per labeled region, including mean, std, min, max, and count.
        Channels are handled explicitly if a channel axis is defined.

        Parameters
        ----------
        image : np.ndarray
            Input image, either single-channel or multi-channel (with axis declared in tag).
        labels : np.ndarray
            Label array with same spatial shape as the image (excluding channel axis).
        stats : Sequence[str], default=('mean',)
            List of statistics to compute per region. Supported values:
            {'mean', 'std', 'min', 'max', 'count'}

        Returns
        -------
        dict of str → np.ndarray
            Dictionary mapping each requested statistic to an array of shape:
            - (C, n_labels) if image has channels,
            - (n_labels,) otherwise
        """
        if labels.shape != image.shape[-labels.ndim:]:
            raise ValueError("Image and label spatial shapes must match.")

        channel_axis = self.get_axis(image, "channel_axis")
        has_channel = channel_axis is not None

        img = np.moveaxis(image, channel_axis, 0) if has_channel else image[None, ...]
        flat_img = img.reshape(img.shape[0], -1)        # (C, N)
        flat_lbl = labels.flatten().astype(np.int32)    # (N,)
        n_labels = flat_lbl.max() + 1

        result: Dict[str, np.ndarray] = {}
        counts = np.bincount(flat_lbl, minlength=n_labels).astype(np.float32, copy=False)

        for stat in stats:
            if stat in ("mean", "std", "count"):
                sums = np.stack([np.bincount(flat_lbl, weights=flat_img[c], minlength=n_labels) for c in range(img.shape[0])])
                counts = np.bincount(flat_lbl, minlength=n_labels).astype(np.float32, copy=False)

                if stat == "mean":
                    mean = sums / (counts + 1e-6)
                    result[stat] = mean if has_channel else mean[0]
                elif stat == "count":
                    result[stat] = counts
            if stat == "std":
                mean = sums / (counts + 1e-6)
                centered = flat_img - mean[:, flat_lbl]
                sq = centered**2
                var = np.stack([np.bincount(flat_lbl, weights=sq[c], minlength=n_labels) for c in range(img.shape[0])])
                std = np.sqrt(var / (counts + 1e-6))
                result[stat] = std if has_channel else std[0]
            if stat == "min":
                minv = np.full((img.shape[0], n_labels), np.inf)
                for c in range(img.shape[0]):
                    for i in range(flat_img.shape[1]):
                        l = flat_lbl[i]
                        val = flat_img[c, i]
                        if val < minv[c, l]:
                            minv[c, l] = val
                result[stat] = minv if has_channel else minv[0]
            if stat == "max":
                maxv = np.full((img.shape[0], n_labels), -np.inf)
                for c in range(img.shape[0]):
                    for i in range(flat_img.shape[1]):
                        l = flat_lbl[i]
                        val = flat_img[c, i]
                        if val > maxv[c, l]:
                            maxv[c, l] = val
                result[stat] = maxv if has_channel else maxv[0]

        return result
    
# ======================================================================
#                      Convenience wrapper
# ======================================================================

def feature_extractor(
    img: ArrayLike,
    features: Optional[Any] = None,
    combine_features: Optional[bool] = None,
    framework: Literal["numpy", "torch"] = "numpy",
    output_format: Literal["numpy", "torch"] = "numpy",
    layout_name: str = "HWC",
    layout_framework: Literal["numpy", "torch"] = "numpy",
    edge_strategy: str = "gradient",
    diff_strategy: Optional[str] = "auto",
    conv_strategy: Optional[str] = "fft",
    processor_strategy: Optional[str] = "vectorized",
    window_size: int = 3,
    sigma: Union[float, Sequence[float]] = 1.0,
    dim: int = 2,
    stack: bool = True,
    return_feat_names: bool = False,
    block_mode: bool = False,
) -> Union[
    ArrayLike,
    Tuple[ArrayLike, str],
    List[ArrayLike],
    Tuple[List[ArrayLike], List[str]],
]:
    """
    Convenience wrapper to build and apply FeatureExtractorND with simplified configuration.

    This function initializes the full feature extraction pipeline (edge, diff, convolve, etc.)
    and runs it on a given image, while handling backend and layout policies automatically.

    Parameters
    ----------
    img : ArrayLike
        Input image (NumPy array or PyTorch tensor).
    features : Any, optional
        Feature specification: string, list, or dict (as supported by FeatureExtractorND).
    combine_features : bool, optional
        Whether to use the 'comb' key to chain features.
    framework : {'numpy', 'torch'}, default='numpy'
        Backend framework for processing.
    output_format : {'numpy', 'torch'}, default='numpy'
        Format for the output result.
    layout_name : str, default='HWC'
        Layout string (e.g., 'HWC', 'NCHW') used for axis tagging.
    layout_framework : {'numpy', 'torch'}, default='numpy'
        Framework used to interpret the layout.
    edge_strategy : str, default='gradient'
        Edge detection method (e.g., 'canny', 'gradient', 'sobel', etc.).
    diff_strategy : str, optional
        Gradient operator strategy ('auto', 'torch', 'vectorized', etc.).
    conv_strategy : str, optional
        Convolution backend strategy ('fft', 'spatial', 'torch', etc.).
    processor_strategy : str, optional
        Strategy for local window processing ('vectorized', 'parallel', 'torch', etc.).
    window_size : int, default=3
        Size of the local window for filters and statistical features.
    sigma : float or Sequence[float], default=1.0
        Gaussian smoothing parameter.
    dim : int, default=2
        Dimensionality used for convolution kernels.
    stack : bool, default=True
        Whether to stack feature maps along a new axis.
    return_feat_names : bool, default=False
        If True, return a tuple (features, feature_names).
    block_mode : bool, default=False
        Whether to split features into logical blocks internally.

    Returns
    -------
    Union[ArrayLike, Tuple, List]
        Extracted feature maps. The exact return type depends on `stack` and `return_feat_names`.
    """    
    # ====[ Fallback ]====
    edge_strategy=edge_strategy or "gradient"
    diff_strategy=diff_strategy or "vectorized" if framework == "numpy" else "torch"
    conv_strategy=conv_strategy or "fft" if framework == "numpy" else "torch"
    processor_strategy=processor_strategy or "vectorized" if framework == "numpy" else "torch"   
        
    # ====[ Configuration ]====
    feature_params: Dict[str, Any] = {"features": features, "window_size": window_size, "stack": stack, "combined": combine_features,
                      "return_feat_names": return_feat_names, "block_mode": block_mode}
    diff_params: Dict[str, Any] = {"spacing": None, "diff_strategy":diff_strategy}
    conv_params: Dict[str, Any] = {"conv_strategy": "gaussian" if processor_strategy == "vectorized" else conv_strategy, 
                   "sigma": sigma, "dim": dim}
    edge_params: Dict[str, Any] = {"edge_strategy": edge_strategy, "eta": "otsu"}
    proc_params: Dict[str, Any] = {"processor_strategy": processor_strategy,}
    layout_params: Dict[str, Any] = {"layout_name": layout_name, "layout_framework": layout_framework}
    global_params: Dict[str, Any] = {"framework": framework, "output_format": output_format}

    features_extract = FeatureExtractorND(
                                        feature_cfg = FeatureConfig(**feature_params),
                                        edge_detector_cfg=EdgeDetectorConfig(**edge_params),
                                        diff_operator_cfg=DiffOperatorConfig(**diff_params),
                                        ndconvolver_cfg=NDConvolverConfig(**conv_params),        
                                        img_process_cfg=ImageProcessorConfig(**proc_params),
                                        layout_cfg=LayoutConfig(**layout_params),
                                        global_cfg=GlobalConfig(**global_params),
                                        )
    
    img_copy = features_extract.safe_copy(img)

    return features_extract(img_copy)