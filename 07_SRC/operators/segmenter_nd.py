# ==================================================
# =============  MODULE: segmenter_nd  =============
# ==================================================
from __future__ import annotations

from typing import Optional, List, Tuple, Union, Dict, Any, Sequence, Literal

from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
import numpy as np, torch, pydicom, hdbscan, heapq, json, sys 
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from itertools import combinations, product
from collections import deque, defaultdict
# from scipy.ndimage import sobel
from skimage import future
from scipy import ndimage

from operators.feature_extractor import feature_extractor
from operators.image_processor import ImageProcessor
from operators.resize_image import ResizeOperator
# from operators.image_io import ImageIO
from core.operator_core import OperatorCore
from core.layout_axes import get_layout_axes
from core.config import (LayoutConfig, GlobalConfig, 
                         ImageProcessorConfig, SegmenterConfig, ResizeConfig)

# Public API
__all__ = ["SegmenterND", "segmenter_nd"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Framework = Literal["numpy", "torch"]

class SegmenterND(OperatorCore):
    """
    Modular ND segmentation with dual-backend (NumPy/Torch), layout-aware logic,
    and multiple strategy backends (thresholding, clustering, watershed, etc.).

    Notes
    -----
    - Axis-aware segmentation: supports 2D, 3D, and batched ND volumes.
    - Strategy-based: segmentation mode controlled via `SegmenterConfig`.
    - Can be used with `ImageProcessor` for vectorized or parallel processing.
    """

    def __init__(
        self,
        segmenter_cfg: SegmenterConfig = SegmenterConfig(),
        img_process_cfg: ImageProcessorConfig = ImageProcessorConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize SegmenterND with configurable strategies.

        Parameters
        ----------
        segmenter_cfg : SegmenterConfig
            Parameters for segmentation modes and options.
        img_process_cfg : ImageProcessorConfig
            Processing strategy and options.
        layout_cfg : LayoutConfig
            Layout and axis configuration.
        global_cfg : GlobalConfig
            Backend and global I/O configuration.
        """
        self.segmenter_cfg: SegmenterConfig = segmenter_cfg
        self.img_process_cfg: ImageProcessorConfig = img_process_cfg
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg

        # === Layout axes ===
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

        # === Segmenter-specific options ===
        self.mode: str = self.segmenter_cfg.mode
        self.threshold: float = float(self.segmenter_cfg.threshold)
        self.window_size: int = int(self.segmenter_cfg.window_size)
        self.bins: int = int(self.segmenter_cfg.bins)
        self.num_classes: int = int(self.segmenter_cfg.num_classes)
        self.return_mask: bool = bool(self.segmenter_cfg.return_mask)
        self.multi_thresholds: tuple = self.segmenter_cfg.multi_thresholds
        self.epsilon: float = float(self.segmenter_cfg.epsilon)

        # KMeans
        self.k: int = int(self.segmenter_cfg.kmeans_k)
        self.auto_k: bool = bool(self.segmenter_cfg.kmeans_auto)
        self.kmeans_method: str = self.segmenter_cfg.kmeans_method
        self.max_k: int = int(self.segmenter_cfg.kmeans_max_k)
        self.max_iter: Optional[int] = self.segmenter_cfg.kmeans_max_iter
        self.batch_size: Optional[int] = self.segmenter_cfg.kmeans_batch_size
        self.kmeans_max_iter: Optional[int] = self.segmenter_cfg.kmeans_max_iter
        self.normalize_output: bool = bool(self.segmenter_cfg.normalize_output)

        # Features/Channels policy: features are carried on the channel axis
        self.use_features = bool(self.segmenter_cfg.use_features)
        self.use_channels = bool(self.segmenter_cfg.use_channels) and (
            self.channel_axis is not None
        )

        # Region growing seeds
        self.seeds: Optional[List[Tuple[int, ...]]] = self.segmenter_cfg.seeds
        self.n_seeds: int = int(self.segmenter_cfg.n_seeds)

        # === Inherit + cache global parameters ===
        self.framework: Framework = self.global_cfg.framework.lower()
        self.output_format: Framework = self.global_cfg.output_format.lower()
        self.layout_ensured_name: str = self.layout_cfg.layout_ensured_name
        self.layout_ensured: Dict[str, Any] = get_layout_axes(self.framework, self.layout_ensured_name)        
        self.add_batch_dim: Optional[bool] = self.global_cfg.add_batch_dim
        self.add_channel_dim: Optional[bool] = self.global_cfg.add_channel_dim
        self.normalize: bool = bool(self.global_cfg.normalize)
        self.verbose: bool = bool(self.global_cfg.verbose)        
        self.device: str = (
            "cuda"
            if (torch.cuda.is_available() and self.framework == "torch")
            else self.global_cfg.device
        )

        # === Init OperatorCore for converters/tag support ===
        super().__init__(layout_cfg=self.layout_cfg, global_cfg=self.global_cfg)
        
        # === Processor for slice-wise / ND segmentation ===
        self.processor: ImageProcessor = ImageProcessor(
            img_process_cfg = self.img_process_cfg,
            layout_cfg = self.layout_cfg,
            global_cfg = self.global_cfg.update_config(output_format=self.framework),
        )
        
        self.resize_op: ResizeOperator = ResizeOperator(
            resize_cfg=ResizeConfig(size=(128, 128), layout_ensured=self.layout_ensured),
            img_process_cfg=ImageProcessorConfig(),
            layout_cfg=self.layout_cfg,
            global_cfg=self.global_cfg,
        )
        
    # -------------------------- Utilities --------------------------   
    
    def _get_axes(self, arr: ArrayLike) -> List[int]:
        """
        Identify spatial axes for applying differential operators (e.g., gradients, divergence).

        Uses axis tags when available (channel, batch, direction, feature) to exclude
        non-spatial dimensions from the returned list.

        Parameters
        ----------
        arr : np.ndarray or torch.Tensor
            Input array or tensor to analyze.

        Returns
        -------
        List[int]
            List of axis indices that correspond to spatial dimensions.

        Notes
        -----
        - In 2D, all axes are considered spatial by default.
        - If axis tags are available (via `get_tag()`), channel, batch, direction,
        and optionally feature axes are excluded from the result.
        - Negative axis indices are converted to positive based on the array's ndim.
        """
        ndim = arr.ndim
        axes = list(range(ndim))

        if ndim == 2:
            return axes  # Default fallback for simple 2D

        tag = self.get_tag(arr, self.framework) if self.has_tag(arr, self.framework) else {}

        def to_positive(axis):
            return axis if axis is None or axis >= 0 else axis + ndim

        channel_ax = to_positive(tag.get("channel_axis", self.axes.get("channel_axis")))
        batch_ax = to_positive(tag.get("batch_axis", self.axes.get("batch_axis")))
        direction_ax = to_positive(tag.get("direction_axis", self.axes.get("direction_axis")))

        # Remove non-spatial axes
        if channel_ax is not None and channel_ax in axes:
            axes.remove(channel_ax)
        if batch_ax is not None and batch_ax in axes:
            axes.remove(batch_ax)
        if direction_ax is not None and direction_ax in axes:
            axes.remove(direction_ax)        
        if self.use_features:
            feature_ax = to_positive(tag.get("feature_axis", None))
            if feature_ax is not None and feature_ax in axes:
                axes.remove(feature_ax)
                
        return axes         

    # -------------------------- Entry point --------------------------

    def __call__(self, image: ArrayLike) -> ArrayLike:
        """
        Apply the selected segmentation strategy to the input image.

        The segmentation mode is chosen via `self.mode`, and mapped to
        an internal function. Supports both standard thresholding methods
        and advanced ND-compatible algorithms.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image to segment (2D, 3D or ND array).

        Returns
        -------
        np.ndarray or torch.Tensor
            Segmentation result with same backend as input,
            tagged as 'segmented'.

        Raises
        ------
        ValueError
            If the configured mode is not recognized.

        Notes
        -----
        - The input is first normalized and converted via `convert_once()`.
        - Strategy dispatch is handled by a static map (`strategy_map`).
        - Output format and dtype are adapted by `to_output()`.
        - Supports a wide range of strategies: Otsu, KMeans, Watershed, etc.
        """
        image = self.convert_once(image)

        strategy_map = {
            "otsu": self._thresholding,
            "otsu_multi": self.threshold_otsu_multi_nd,
            "yen": self._thresholding,
            "li": self._thresholding,
            "triangle": self._thresholding,
            "isodata": self._thresholding,
            "entropy": self.threshold_entropy_nd,
            "iterative": self.threshold_iterative_nd,
            "fixed": self.threshold_fixed_nd,
            "multi": self.threshold_multi_nd,
            "kmeans": self.threshold_kmeans_nd,
            "agglomerative": self.threshold_agglomerative_nd,
            "hdbscan": self.threshold_hdbscan_nd,
            "region_growing": self.region_growing_nd,
            "split_and_merge": self.split_and_merge_nd,
            "watershed": self.segment_watershed_nd,
        }

        if self.mode not in strategy_map:
            raise ValueError(f"[SegmenterND] Unknown segmentation mode '{self.mode}'")

        segmented = strategy_map[self.mode](image)
        return self.to_output(segmented, tag_as="segmented")

    # -------------------------- Thresholding --------------------------

    def _thresholding(self, image: ArrayLike) -> ArrayLike:
        """
        Apply a standard thresholding method to segment the image.

        Uses `ThresholdingOperator` internally with the configured method
        (e.g., Otsu, Yen, Li, Triangle, Isodata).

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image to segment.

        Returns
        -------
        np.ndarray or torch.Tensor
            Binary segmentation mask or thresholded output, depending on the method.

        Notes
        -----
        - Method is selected from `self.mode`. Defaults to 'otsu' if invalid.
        - Internally applies preprocessing and layout normalization via `ThresholdingOperator`.
        - Returns a mask (boolean or integer) with same shape and backend as input.
        """
        from operators.thresholding import ThresholdingOperator
        
        method = self.mode if self.mode in ["otsu", "yen", "li", "triangle", "isodata"] else "otsu"
        
        threshold_op = ThresholdingOperator(
            method=method,
            as_mask=True,
            img_process_cfg=self.img_process_cfg,
            layout_cfg=self.layout_cfg,
            global_cfg=self.global_cfg,
        )
        return threshold_op(image)
    
    def threshold_fixed_nd(self, image: ArrayLike) -> ArrayLike:
        """
        Apply fixed threshold segmentation on an ND image.

        Uses a user-defined scalar threshold (typically in [0, 1]) to
        produce a binary mask.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image to segment. Should be normalized if using a threshold in [0, 1].

        Returns
        -------
        np.ndarray or torch.Tensor
            Binary segmentation mask with the same shape and backend as the input.

        Notes
        -----
        - The threshold value is taken from `self.threshold`.
        - Internally uses `self.processor` to apply backend-specific logic.
        - Works with both NumPy and Torch inputs, on arbitrary ND volumes.
        """        
        threshold = self.threshold
        
        def thresh_np(x: np.ndarray) -> np.ndarray:
            return (x > threshold).astype(np.float32, copy=False)

        def thresh_torch(x: torch.Tensor) -> torch.Tensor:
            return (x > threshold).float()

        self.processor.function = thresh_torch if self.framework == "torch" else thresh_np
        return self.processor(image)
    
    def threshold_iterative_nd(self, image: ArrayLike) -> ArrayLike:
        """
        Apply adaptive global thresholding (Ridler–Calvard method) on ND images.

        Iteratively refines a global threshold that separates foreground from background
        based on intensity means, until convergence or maximum iterations.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image to segment. Should be normalized to [0, 1] or similar.
        
        Returns
        -------
        np.ndarray or torch.Tensor
            Binary segmentation mask (same shape and backend as input).

        Notes
        -----
        - The convergence criterion is set by `self.epsilon`.
        - The maximum number of iterations is set by `self.max_iter`.
        - Returns a mask with 0s and 1s as float values.
        - Backend-specific logic is dispatched via `self.processor`.
        """
        epsilon = self.epsilon
        max_iter = self.max_iter
        
        def thresh_np(x: np.ndarray) -> np.ndarray:
            s_old = np.mean(x)
            for _ in range(max_iter):
                g1 = x[x <= s_old]
                g2 = x[x > s_old]
                if len(g1) == 0 or len(g2) == 0:
                    break
                m1, m2 = np.mean(g1), np.mean(g2)
                s_new = (m1 + m2) / 2
                if abs(s_new - s_old) < epsilon:
                    break
                s_old = s_new
            return (x > s_old).astype(np.float32, copy=False)

        def thresh_torch(x: torch.Tensor) -> torch.Tensor:
            s_old = torch.mean(x)
            for _ in range(max_iter):
                g1 = x[x <= s_old]
                g2 = x[x > s_old]
                if g1.numel() == 0 or g2.numel() == 0:
                    break
                m1, m2 = torch.mean(g1), torch.mean(g2)
                s_new = (m1 + m2) / 2
                if torch.abs(s_new - s_old) < epsilon:
                    break
                s_old = s_new
            return (x > s_old).float()

        self.processor.function = thresh_torch if self.framework == "torch" else thresh_np
        return self.processor(image)

    def threshold_multi_nd(
        self,
        image: ArrayLike,
        thresholds: Optional[List[float]] = None,
        normalize_output: Optional[bool] = None,
    ) -> ArrayLike:
        """
        Apply multi-threshold segmentation to divide an image into N+1 discrete classes.

        Each threshold defines a transition point between classes. Class labels are
        assigned incrementally based on intensity values. Optionally, the output can be
        normalized to [0, 1].

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image to segment. Should be normalized to [0, 1] or similar.
        thresholds : list of float, optional
            Threshold values (must be sorted). If None, uses `self.multi_thresholds`.
        normalize_output : bool, optional
            If True, output class labels are scaled to [0, 1] range.
            If None, uses `self.normalize_output`.

        Returns
        -------
        np.ndarray or torch.Tensor
            Segmentation map with either discrete class labels or normalized float values.

        Notes
        -----
        - For K thresholds, output will contain K+1 classes: [0, 1, ..., K].
        - Class labels are float32 (not int) to allow optional normalization.
        - Works for arbitrary ND shapes and dual backends (NumPy / Torch).
        """
        thresholds = thresholds or self.multi_thresholds
        thresholds = sorted(thresholds)
        n_classes = len(thresholds) + 1
        normalize_output = (
            self.normalize_output if normalize_output is None else normalize_output
        )

        def multi_np(x: np.ndarray) -> np.ndarray:
            out = np.zeros_like(x, dtype=np.float32)
            for i, t in enumerate(thresholds):
                out[x > t] = i + 1
            if normalize_output:
                out = out / (n_classes - 1)
            return out

        def multi_torch(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x, dtype=torch.float32)
            for i, t in enumerate(thresholds):
                out = torch.where(x > t, torch.tensor(i + 1.0, device=x.device), out)
            if normalize_output:
                out = out / (n_classes - 1)
            return out

        self.processor.function = multi_torch if self.framework == "torch" else multi_np
        return self.processor(image)

    def threshold_entropy_nd(self, image: ArrayLike) -> ArrayLike:
        """
        Apply global entropy-based thresholding (Shannon criterion) on an ND image.

        Selects the optimal threshold that maximizes the sum of foreground and background
        entropies, based on the normalized histogram of pixel intensities.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image (assumed to be normalized in [0, 1]).

        Returns
        -------
        np.ndarray or torch.Tensor
            Binary segmentation mask with same shape and backend as input.

        Notes
        -----
        - The number of histogram bins is controlled by `self.bins`.
        - Internally computes class probabilities and Shannon entropy for each possible split.
        - Avoids zero-probability classes and stabilizes computation using `eps = 1e-8`.
        - Compatible with both NumPy and Torch backends.
        """
        eps = 1e-8
        bins = self.bins

        def entropy_split_np(x: np.ndarray) -> np.ndarray:
            hist, _ = np.histogram(x, bins=bins, range=(0, 1))
            p = hist / (hist.sum() + eps)
            entropies = []

            for t in range(1, bins):  # avoid empty class
                p1 = np.sum(p[:t])
                p2 = np.sum(p[t:])
                if p1 == 0 or p2 == 0:
                    entropies.append(0)
                    continue
                entropy = -p1 * np.log2(p1 + eps) - p2 * np.log2(p2 + eps)
                entropies.append(entropy)

            t_star = np.argmax(entropies)
            t_value = (t_star + 0.5) / bins
            return (x > t_value).astype(np.float32, copy=False)

        def entropy_split_torch(x: torch.Tensor) -> torch.Tensor:
            hist = torch.histc(x.flatten(), bins=bins, min=0.0, max=1.0)
            p = hist / (hist.sum() + eps)
            entropies: List[torch.Tensor] = []

            for t in range(1, bins):
                p1 = p[:t].sum()
                p2 = p[t:].sum()
                if p1.item() == 0 or p2.item() == 0:
                    entropies.append(torch.tensor(0.0))
                    continue
                ent = -p1 * torch.log2(p1 + eps) - p2 * torch.log2(p2 + eps)
                entropies.append(ent)

            t_star = torch.argmax(torch.stack(entropies))
            t_value = (t_star + 0.5) / bins
            return (x > t_value).float()

        self.processor.function = entropy_split_torch if self.framework == "torch" else entropy_split_np
        return self.processor(image)

    def threshold_otsu_multi_nd(self, image: ArrayLike) -> ArrayLike:
        """
        Apply multi-class Otsu thresholding on an ND image.

        Extends Otsu's method to multiple thresholds, segmenting the image into N classes
        by maximizing between-class variance over a histogram-based partition.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image to segment. Should be normalized to [0, 1].

        Returns
        -------
        np.ndarray or torch.Tensor
            Segmented image with class labels encoded as float values in [0, 1].

        Notes
        -----
        - The number of classes is defined by `self.num_classes` (default: 3).
        - A fixed number of histogram bins is used (`self.bins`, e.g., 256).
        - All possible threshold combinations are enumerated (up to 1000 max).
        - For performance, the Torch implementation converts to NumPy before processing.
        - Output values are normalized float labels: 0.0, 0.5, 1.0 (for 3 classes).
        """
        num_classes = int(self.num_classes or 3)
        bins = int(self.bins)

        def otsu_multi_np(x: np.ndarray) -> np.ndarray:
            hist, _ = np.histogram(x, bins=bins, range=(0, 1))
            hist = hist.astype(np.float32, copy=False)
            p = hist / (hist.sum() + 1e-8)
            total_mean = np.sum(np.arange(bins) * p)

            best_thresholds = None
            max_sigma = -1
            # Restrict number of combinations
            combs = list(combinations(range(1, bins-1), num_classes - 1))[:1000]  # Limit to 1000
            for ths in combs:
                ths = (0,) + ths + (bins-1,)
                w, mu = [], []
                for i in range(num_classes):
                    idx = slice(ths[i], ths[i+1])
                    pi = p[idx].sum()
                    mi = np.sum(np.arange(ths[i], ths[i+1]) * p[idx]) / pi if pi > 0 else 0
                    w.append(pi)
                    mu.append(mi)
                sigma_b = sum(w[i] * (mu[i] - total_mean)**2 for i in range(num_classes))
                if sigma_b > max_sigma:
                    max_sigma = sigma_b
                    best_thresholds = ths[1:-1]

            thresholds = [(t + 0.5)/bins for t in best_thresholds]
            seg = np.zeros_like(x, dtype=np.float32)
            for i, t in enumerate(thresholds):
                seg[x > t] = (i+1) / (num_classes - 1)
            return seg

        def otsu_multi_torch(x: torch.Tensor) -> torch.Tensor:
            # Torch histogram too slow for multiple passes → convert once
            x_np = x.detach().cpu().numpy()
            return torch.tensor(otsu_multi_np(x_np), dtype=torch.float32, device=x.device)

        self.processor.function = otsu_multi_torch if self.framework == "torch" else otsu_multi_np
        return self.processor(image)

    # -------------------------- Clustering --------------------------

    def threshold_kmeans_nd(
        self,
        image: ArrayLike,
        k: Optional[int] = None,
        auto_k: Optional[bool] = False,
        method: Optional[str] = "silhouette",
        batch_size: Optional[int] = None,
        max_iter: Optional[int] = None,
        max_k: Optional[int] = 10,
        normalize_output: Optional[bool] = False,
    ) -> ArrayLike:
        """
        Segment an ND image using KMeans clustering on pixel intensities or features.

        Supports both fixed and automatic selection of the number of clusters (k),
        as well as layout-aware reshaping for channel- or feature-based inputs.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image to segment. Can be 2D, 3D or ND, with or without channels/features.
        k : int, optional
            Number of clusters. Required if `auto_k=False`.
        auto_k : bool, optional
            If True, automatically estimate the optimal number of clusters using the selected method.
        method : {'silhouette', 'elbow'}, optional
            Method to use when `auto_k=True`. Default is 'silhouette'.
        batch_size : int, optional
            Mini-batch size for `MiniBatchKMeans`. Automatically tuned if not provided.
        max_iter : int, optional
            Maximum number of iterations for clustering. Auto-tuned based on image size.
        max_k : int, default 10
            Maximum number of clusters to test when `auto_k=True`.
        normalize_output : bool, optional
            If True, output will be scaled to [0, 1] (useful for visualization).

        Returns
        -------
        np.ndarray or torch.Tensor
            Segmentation map with class labels (int or float), shaped like the input image.

        Notes
        -----
        - The input is reshaped to 2D (N, D) where N is the number of pixels and D is the feature size.
        - Cluster labels are ordered by average intensity for consistency across runs.
        - If `normalize_output=True`, classes are scaled to float values in [0, 1].
        - Works for both intensity-only inputs and (C, F)-style multi-feature images.
        - Final output format matches `self.output_format` (Torch or NumPy).
        """
        # === Flatten features ===
        x = self.convert_once(image, tag_as="input", framework="numpy",)
        k = k or self.k
        auto_k = auto_k or self.auto_k
        method = method or self.kmeans_method
        max_k = max_k or self.max_k
        batch_size = batch_size or self.batch_size
        max_iter = max_iter or self.max_iter
        normalize_output = (
            self.normalize_output if normalize_output is None else normalize_output
        )
                
        # === Retrieve layout-aware axes ===
        channel_axis = self.get_axis(x, "channel_axis")
        feature_axis = self.get_axis(x, "feature_axis") if self.use_features else None
        spatial_axes = self._get_axes(x)
        spatial_shape = tuple(x.shape[ax] for ax in spatial_axes) 
        if channel_axis is not None:    
            spatial_shape += (x.shape[channel_axis],) if not self.use_channels else ()
        
        # === Move feature and/or channel to the end ===
        if self.use_channels and channel_axis is not None:
            x = np.moveaxis(x, channel_axis, -1)  # Move channel last
        if feature_axis is not None:
            x = np.moveaxis(x, feature_axis, -1)  # Move feature last

        # === Flatten for clustering ===
        if self.use_features and self.use_channels:
            # Shape: (..., C, F) → reshape to (N, C×F)
            pixels = x.reshape(-1, x.shape[-2] * x.shape[-1])   
        elif self.use_features or self.use_channels:
            # Shape: (..., C) or (..., F) → reshape to (N, D)
            pixels = x.reshape(-1, x.shape[-1])
        else:
            # Shape: (..., H, W) → reshape to (N, 1)
            pixels = x.reshape(-1, 1)
            
         # === Auto-tune batch_size and max_iter ===
        if batch_size is None:
            batch_size = min(100_000, max(10_000, pixels.shape[0] // 500))
        if max_iter is None:
            max_iter = 1000 if pixels.shape[0] < 10_000_000 else 1500

        # === Auto-detect number of clusters (optional) ===
        if auto_k:
            if method == "silhouette":
                scores = []
                k_values = range(2, max_k + 1)
                for ki in k_values:
                    kmeans = MiniBatchKMeans(n_clusters=ki, random_state=42, n_init='auto', max_iter=max_iter, batch_size=min(batch_size, pixels.shape[0] // 10))
                    labels = kmeans.fit_predict(pixels)
                    score = silhouette_score(pixels, labels)
                    scores.append(score)
                k = k_values[np.argmax(scores)]

            elif method == "elbow":
                inertias = []
                k_values = range(1, max_k + 1)
                for ki in k_values:
                    kmeans = MiniBatchKMeans(n_clusters=ki, random_state=42, n_init='auto', max_iter=max_iter, batch_size=min(batch_size, pixels.shape[0] // 10))
                    kmeans.fit(pixels)
                    inertias.append(kmeans.inertia_)
                diff = np.diff(inertias, 2)
                elbow_idx = np.argmax(np.abs(diff)) + 2  # +2 because of double diff
                k = k_values[elbow_idx]

        if k is None:
            raise ValueError("You must provide 'k' or set auto_k=True.")

        # === Apply clustering ===
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=max_iter, batch_size=min(batch_size, pixels.shape[0] // 10))
        labels = kmeans.fit_predict(pixels)

        # === Order labels by cluster intensity mean ===
        cluster_means = kmeans.cluster_centers_.mean(axis=1)
        order = np.argsort(cluster_means)
        relabel_map = np.zeros_like(order)
        relabel_map[order] = np.arange(1, k + 1)

        labels_ordered = relabel_map[labels].reshape(spatial_shape)

        # === Normalize output if requested ===
        if normalize_output:
            labels_ordered = labels_ordered.astype(np.float32, copy=False) / k

        return torch.tensor(labels_ordered, device=image.device) if self.output_format == "torch" else labels_ordered
    
    # def kmeans_nd_torch(
    #     self,
    #     x: torch.Tensor,
    #     k: int = 4,
    #     max_iter: int = 100,
    #     tol: float = 1e-4,
    #     normalize_output: bool = False
    # ) -> torch.Tensor:
    #     """
    #     Perform KMeans clustering directly on an ND Torch tensor.

    #     This method supports multi-dimensional volumes with optional channels or features.
    #     It flattens spatial dimensions and applies a Torch-native KMeans algorithm
    #     using Euclidean distance and iterative centroid updates.

    #     Parameters
    #     ----------
    #     x : torch.Tensor
    #         Input tensor of shape (..., C), (..., F), or (..., C, F), depending on configuration.
    #     k : int, default 4
    #         Number of clusters to assign.
    #     max_iter : int, default 100
    #         Maximum number of KMeans iterations.
    #     tol : float, default 1e-4
    #         Convergence threshold (L2-norm of centroid shift).
    #     normalize_output : bool, default False
    #         If True, normalize output labels to the range [0, 1].

    #     Returns
    #     -------
    #     torch.Tensor
    #         ND segmentation map with the same spatial shape as input.
    #         Labels are 1-based integers (or floats in [0, 1] if normalized).

    #     Notes
    #     -----
    #     - Layout is resolved automatically using axis tagging (e.g., channel_axis, feature_axis).
    #     - Works for 2D, 3D, or ND inputs with or without features/channels.
    #     - Final labels are ordered by distance to centroids (no mean-sorting).
    #     - This implementation avoids external libraries (pure Torch).
    #     """
    #     # === Retrieve layout-aware axes ===
    #     channel_axis = self.get_axis(x, "channel_axis")
    #     feature_axis = self.get_axis(x, "feature_axis") if self.use_features else None
    #     spatial_axes = self._get_axes(x)
    #     spatial_shape = tuple(x.shape[ax] for ax in spatial_axes) 
    #     spatial_shape += (x.shape[channel_axis],) if not self.use_channels else ()
        
    #     # === Move feature and/or channel to the end ===
    #     if self.use_channels and channel_axis is not None:
    #         x = torch.movedim(x, channel_axis, -1)  # Move channel last
    #     if feature_axis is not None:
    #         x = torch.movedim(x, feature_axis, -1)  # Move feature last

    #     # === Flatten for clustering ===
    #     if self.use_features and self.use_channels:
    #         # Shape: (..., C, F) → reshape to (N, C×F)
    #         pixels = x.reshape(-1, x.shape[-2] * x.shape[-1])   
    #     elif self.use_features or self.use_channels:
    #         # Shape: (..., C) or (..., F) → reshape to (N, D)
    #         pixels = x.reshape(-1, x.shape[-1])
    #     else:
    #         # Shape: (..., H, W) → reshape to (N, 1)
    #         pixels = x.reshape(-1, 1)        

    #     # === Move feature and/or channel to the end ===
    #     n_points = pixels.shape[0]

    #     # Initialize centroids randomly from the data
    #     indices = torch.randperm(n_points, device=pixels.device)[:k]
    #     centroids = pixels[indices]  # (k, D)

    #     for _ in range(max_iter):
    #         # Compute distances (N, k)
    #         distances = torch.cdist(pixels, centroids, p=2)
    #         labels = torch.argmin(distances, dim=1)

    #         # Update centroids
    #         new_centroids = torch.stack([
    #             pixels[labels == i].mean(dim=0) if (labels == i).any() else centroids[i]
    #             for i in range(k)
    #         ])

    #         shift = torch.norm(centroids - new_centroids)
    #         centroids = new_centroids
    #         if shift < tol:
    #             break

    #     # Final assignment
    #     distances = torch.cdist(pixels, centroids, p=2)
    #     labels = torch.argmin(distances, dim=1)

    #     labels = labels.view(*spatial_shape) + 1  # shift to 1-based labels
    #     if normalize_output:
    #         labels = labels.float() / k

    #     return labels

    def threshold_agglomerative_nd(
        self,
        image: ArrayLike,
        n_clusters: Optional[int] = None,
        linkage: str = "ward",
        metric: str = "euclidean",
        normalize_output: bool = False
    ) -> ArrayLike:
        """
        Apply agglomerative clustering for ND image segmentation.

        Supports layout-aware reshaping of multichannel or feature-rich data.
        Falls back to downsampling if input size exceeds memory-safe limits.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input ND image. Can include channels and/or features.
        n_clusters : int, optional
            Number of clusters to form. If None, defaults to `self.k` or 3.
        linkage : {'ward', 'average', 'complete'}, default 'ward'
            Linkage strategy for merging clusters.
        metric : str, default 'euclidean'
            Distance metric (ignored if `linkage='ward'`).
        normalize_output : bool, default False
            If True, normalize output labels to [0, 1].

        Returns
        -------
        np.ndarray or torch.Tensor
            Segmentation map with shape matching spatial dimensions of input.
            Output is float if normalized, otherwise integer-labeled.

        Notes
        -----
        - Layout axes (channel, feature) are automatically moved to the end before clustering.
        - For large volumes (> 50,000 pixels), downsampling is applied to avoid memory overload.
        - When downsampling occurs, non-selected pixels are assigned label 0 (naive fallback).
        - Final labels are reshaped to spatial dimensions and converted to Torch if required.
        """
        # === Preprocess input ===
        x = self.convert_once(image, tag_as="input", framework="numpy")
        n_clusters = int(n_clusters or self.k or 3)

        channel_axis = self.get_axis(x, "channel_axis") if self.use_channels else None
        feature_axis = self.get_axis(x, "feature_axis") if self.use_features else None
        spatial_axes = self._get_axes(x)
        spatial_shape = tuple(x.shape[ax] for ax in spatial_axes)

        # === Move channels/features last
        if self.use_channels and channel_axis is not None:
            x = np.moveaxis(x, channel_axis, -1)
        if self.use_features and feature_axis is not None:
            x = np.moveaxis(x, feature_axis, -1)

        # === Flatten
        if self.use_channels and self.use_features:
            pixels = x.reshape(-1, x.shape[-2] * x.shape[-1])
        elif self.use_channels or self.use_features:
            pixels = x.reshape(-1, x.shape[-1])
        else:
            pixels = x.reshape(-1, 1)
            
        max_pixels = 50_000 # Limit max number of pixels to avoid memory issues
        n_pixels = pixels.shape[0]

        if n_pixels > max_pixels:
            print(f"[Warning] Too many pixels ({n_pixels}) for AgglomerativeClustering. "
                f"Downsampling to {max_pixels} pixels.")
            idx = np.random.choice(n_pixels, size=max_pixels, replace=False)
            pixels_subset = pixels[idx]
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                metric=metric if linkage != "ward" else "euclidean",
                # affinity=("euclidean" if linkage == "ward" else metric),
            )
            labels_subset = clustering.fit_predict(pixels_subset)

            # Option 1: propagate label 0 to all (naive fallback)
            labels = np.zeros(n_pixels, dtype=int)
            labels[idx] = labels_subset
        else:
            # === Fit clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                metric=metric if linkage != "ward" else "euclidean",
            )
            labels = clustering.fit_predict(pixels)

            # === Reshape and normalize if needed
            labels = labels.reshape(spatial_shape)
            if normalize_output:
                labels = labels.astype(np.float32, copy=False) / (n_clusters - 1)

        return torch.tensor(labels, device=image.device) if self.output_format == "torch" else labels
    

    def threshold_hdbscan_nd(
        self,
        image: ArrayLike,
        min_cluster_size: int = 50,
        min_samples: Optional[int] = None,
        normalize_output: bool = False,
    ) -> ArrayLike:
        """
        Segment an ND image using HDBSCAN clustering (density-based, auto-k, robust to noise).

        HDBSCAN groups pixels based on local density without requiring a predefined number of clusters.
        Outliers are automatically detected and labeled as 0.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input ND image. May include channels or feature axes.
        min_cluster_size : int, default 50
            Minimum size (in pixels) for a group to be considered a cluster.
        min_samples : int, optional
            Minimum number of neighbors to define a dense region.
            Defaults to `min_cluster_size` if not specified.
        normalize_output : bool, default False
            If True, scale final labels to the range [0, 1].

        Returns
        -------
        np.ndarray or torch.Tensor
            Segmentation map with shape matching input spatial dimensions.
            - Labels are 0-based, with outliers assigned 0.
            - If normalized, values are in [0, 1] as float32.

        Notes
        -----
        - Layout-aware: channels/features are moved to the last axis automatically.
        - Feature vectors are flattened before clustering.
        - Outlier pixels (cluster ID -1) are relabeled to 0.
        - If `normalize_output=True`, labels are scaled by their maximum value.
        - Output format is determined by `self.output_format` ('torch' or 'numpy').
        """
        # === Preprocess input
        x = self.convert_once(image, tag_as="input", framework="numpy")
        channel_axis = self.get_axis(x, "channel_axis") if self.use_channels else None
        feature_axis = self.get_axis(x, "feature_axis") if self.use_features else None
        spatial_axes = self._get_axes(x)
        spatial_shape = tuple(x.shape[ax] for ax in spatial_axes)

        # === Move channels/features to end
        if self.use_channels and channel_axis is not None:
            x = np.moveaxis(x, channel_axis, -1)
        if self.use_features and feature_axis is not None:
            x = np.moveaxis(x, feature_axis, -1)

        # === Flatten
        if self.use_channels and self.use_features:
            pixels = x.reshape(-1, x.shape[-2] * x.shape[-1])
        elif self.use_channels or self.use_features:
            pixels = x.reshape(-1, x.shape[-1])
        else:
            pixels = x.reshape(-1, 1)

        # === Fit clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples or min_cluster_size,
            allow_single_cluster=False,
            prediction_data=False
        )
        labels = clusterer.fit_predict(pixels)  # -1 → outlier

        # === Relabel from 1...k (skip -1)
        mask = labels >= 0
        unique = np.unique(labels[mask]) if mask.any() else np.array([], dtype=int)
        relabel_map = {old: new + 1 for new, old in enumerate(unique)}
        relabeled = np.array([relabel_map.get(l, 0) for l in labels], dtype=int)

        result = relabeled.reshape(spatial_shape)
        if normalize_output and result.max() > 0:
            result = result.astype(np.float32, copy=False) / result.max()

        return torch.tensor(result, device=image.device) if self.output_format == "torch" else result

    # -------------------------- Region Growing --------------------------
    
    @staticmethod
    def _format_segmentation_output(
        segmentation: np.ndarray,
        return_mode: str = "labels",
        palette: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Format a segmentation map into one of several output modes: labels, RGB, or one-hot.

        Parameters
        ----------
        segmentation : np.ndarray
            ND array of integer class labels (e.g., from a clustering or thresholding method).
        return_mode : {'labels', 'rgb', 'onehot'}, default 'labels'
            Output format:
            - 'labels': returns raw integer labels.
            - 'rgb': returns an RGB image using a color palette.
            - 'onehot': returns a (C, ...) one-hot encoded array.
        palette : np.ndarray, optional
            Optional color palette of shape (n_classes, 3) for RGB mode.
            If not provided, a deterministic palette is generated.

        Returns
        -------
        np.ndarray
            Segmentation output formatted according to `return_mode`.

        Raises
        ------
        ValueError
            If `return_mode` is not one of the supported options.

        Notes
        -----
        - The number of classes is inferred from `segmentation.max() + 1`.
        - RGB mode assigns one unique color per class using the palette.
        - One-hot mode returns a binary array of shape (C, ...) where C = number of classes.
        """
        if return_mode == "labels":
            return segmentation

        # Number of classes
        n_classes = int(np.max(segmentation)) + 1 if segmentation.size > 0 else 0

        if return_mode == "rgb":
            if palette is None:
                rng = np.random.default_rng(42)
                palette = rng.integers(50, 255, size=(n_classes, 3), dtype=np.uint8)
            return palette[segmentation.astype(np.int32, copy=False)]

        elif return_mode == "onehot":
            shape = segmentation.shape
            onehot = np.zeros((n_classes, *shape), dtype=np.uint8)
            for i in range(n_classes):
                onehot[i] = segmentation == i
            return onehot

        else:
            raise ValueError(f"Unsupported return_mode '{return_mode}'")

    def region_growing_nd(
        self,
        image: ArrayLike,
        seeds: Optional[List[Tuple[int, ...]]] = None,
        threshold: float = 0.1,
        distance_tolerance: float = 5.0,
        return_mode: str = "labels",
        palette: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Segment an ND image using a region growing algorithm with intensity and distance constraints.

        Each seed expands to nearby pixels if the local intensity is similar (below threshold)
        and spatial proximity is within tolerance. Multichannel images are supported using
        Euclidean distance in feature space.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input ND image. Must be normalized in [0, 1].
        seeds : list of tuple[int, ...], optional
            List of seed coordinates. If None, seeds are generated randomly or from `self.seeds`.
        threshold : float, default 0.1
            Maximum intensity difference allowed for inclusion in the region.
        distance_tolerance : float, default 5.0
            Maximum distance from the seed point to grow. Uses Euclidean distance.
        return_mode : {'labels', 'rgb', 'onehot'}, default 'labels'
            Output format:
            - 'labels': raw label map.
            - 'rgb': colorized label map using `palette`.
            - 'onehot': one-hot encoded map (shape: [C, ...]).
        palette : np.ndarray, optional
            Optional color palette for RGB output. If None, a default one is generated.

        Returns
        -------
        np.ndarray
            Segmentation result with same spatial shape as input.
            Format depends on `return_mode`.

        Notes
        -----
        - Supports multichannel inputs via channel-aware intensity comparison.
        - Seed expansion is priority-based (using a min-heap and distance map).
        - The algorithm stops when all reachable pixels have been labeled.
        - Channels/features are automatically managed using axis tags.
        """
        arr = self.convert_once(image,tag_as="input", framework="numpy",)

        if float(arr.max()) > 1.0 + 1e-3 or float(arr.min()) < 0.0 - 1e-3:
            raise ValueError("Input image must be normalized in [0, 1] for region growing.")

        # === Axes detection
        ndim = arr.ndim
        channel_axis = self.get_axis(arr, "channel_axis") if self.use_channels else None
        batch_axis = self.get_axis(arr, "batch_axis")

        spatial_axes = list(range(ndim))
        for ax in (channel_axis, batch_axis):
            if ax is not None and ax in spatial_axes:
                spatial_axes.remove(ax)

        spatial_shape = tuple(arr.shape[ax] for ax in spatial_axes)
        is_multichannel = channel_axis is not None and arr.shape[channel_axis] > 1

        # === Prepare
        segmented = np.zeros(spatial_shape, dtype=np.int32)
        distance_map = np.full(spatial_shape, np.inf)

        arr_access = arr if not is_multichannel else np.moveaxis(arr, channel_axis, -1)
        get_val = lambda pos: arr_access[pos]

        # Initialize seeds
        if seeds is None:
            if self.seeds is not None:
                seeds = list(map(tuple, self.seeds))
            else:
                rng = np.random.default_rng(42)
                n_seeds = int(self.n_seeds or 1)
                seeds = [tuple(rng.integers(0, s) for s in spatial_shape) for _ in range(n_seeds)]

        n_labels = len(seeds)
        labels = list(range(1, n_labels + 1))
        seeds_array = np.array(seeds)

        queue: List[Tuple[float, Tuple[int, ...], int]] = []
        for i, s in enumerate(seeds):
            segmented[s] = labels[i]
            distance_map[s] = 0
            heapq.heappush(queue, (0.0, s, labels[i]))

        # Neighbor exploration (3^n - 1)
        while queue:
            dist, pos, label = heapq.heappop(queue)
            for offset in product([-1, 0, 1], repeat=len(pos)):
                if all(o == 0 for o in offset):
                    continue
                neighbor = tuple(p + o for p, o in zip(pos, offset))
                if not all(0 <= neighbor[d] < spatial_shape[d] for d in range(len(spatial_shape))):
                    continue
                if segmented[neighbor] != 0:
                    continue

                ref_val = get_val(pos)
                cur_val = get_val(neighbor)

                diff = (
                    np.linalg.norm(ref_val - cur_val)
                    if is_multichannel
                    else abs(float(ref_val) - float(cur_val))
                )
                delta = float(np.linalg.norm(np.array(neighbor) - seeds_array[label - 1]))
                if diff < float(threshold) and delta < distance_map[neighbor] - float(distance_tolerance):
                    distance_map[neighbor] = delta
                    segmented[neighbor] = label
                    heapq.heappush(queue, (delta, neighbor, label))

        return self._format_segmentation_output(segmented, return_mode=return_mode, palette=palette)

    # -------------------------- Split & Merge --------------------------
    def split_and_merge_nd(
        self,
        image: ArrayLike,
        min_size: int = 8,
        threshold_split: float = 0.02,
        threshold_merge: float = 0.1,
        return_labels: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Segment an ND image using a split-and-merge strategy with gradient-aware merging.

        The algorithm recursively splits the image into smaller regions based on intensity
        homogeneity and local gradients. Adjacent regions are then merged if their
        intensities are similar and gradient boundaries are weak.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input ND image. Can include channels or batches.
        min_size : int, default 8
            Minimum size (per spatial axis) of a block before stopping the split.
        threshold_split : float, default 0.02
            Threshold for intensity standard deviation to consider a region homogeneous.
        threshold_merge : float, default 0.1
            Maximum intensity difference allowed between adjacent regions for merging.
        return_labels : bool, default False
            If True, returns both the final map and the initial pre-merge labels.

        Returns
        -------
        np.ndarray or torch.Tensor
            Final segmentation map after merging.
            If `return_labels=True`, returns a tuple (merged, initial_labels).

        Notes
        -----
        - Uses `sobel` gradients to detect structural boundaries.
        - Automatically resolves spatial axes from tagged layouts.
        - In multichannel mode, gradient magnitude is computed via L2 norm.
        - Uses a region-growing style neighborhood scan for merging.
        - Merge logic uses both intensity similarity and gradient suppression.
        """
        image = self.resize_op(image)
        is_torch = torch.is_tensor(image)

        # === Axes
        channel_axis = self.get_axis(image, "channel_axis")
        spatial_axes = self._get_axes(image)
        spatial_shape = tuple(image.shape[ax] for ax in spatial_axes)

        # === Gradient extraction
        gradient_map = feature_extractor(
            image,
            features=["sobel"],
            framework=self.framework,
            output_format=self.framework,
            layout_name=self.layout_name,
            layout_framework=self.layout_framework,
            stack=False,
        )

        # === Combine multichannel gradient if needed
        if channel_axis is not None:
            gradient_map = np.linalg.norm(gradient_map, axis=channel_axis) if isinstance(gradient_map, np.ndarray) \
                else torch.norm(gradient_map, dim=channel_axis)

        assert gradient_map.ndim == len(spatial_shape), \
            f"[split_and_merge_nd] Gradient map must be scalar per spatial point (got shape {gradient_map.shape})"

        def extract_region_vals(arr: ArrayLike, spatial_slices: Tuple[slice, ...]) -> ArrayLike:
            if channel_axis is None:
                return arr[spatial_slices]
            index = [slice(None)] * arr.ndim
            for ax, slc in zip(spatial_axes, spatial_slices):
                index[ax] = slc
            return arr[tuple(index)]

        def split_nd(img: ArrayLike, gradient: Optional[ArrayLike] = None):
            def is_homogeneous(region: ArrayLike, thr: float, grad: Optional[ArrayLike]) -> bool:
                std_val = torch.std(region).item() if torch.is_tensor(region) else float(np.std(region))
                if grad is not None:
                    gmask = grad < thr
                    mean_mask = (
                        gmask.float().mean().item() if torch.is_tensor(gmask) else float(np.mean(gmask))
                    )
                    return (std_val < thr) and (mean_mask > 0.9)
                return std_val < thr

            shape = img.shape
            initial_slices = tuple(slice(0, s) for s in shape)
            blocks: deque = deque([initial_slices])
            result: List[Tuple[Tuple[slice, ...], ArrayLike]] = []

            while blocks:
                slices = blocks.popleft()
                region = img[slices]
                spatial_slices = tuple(slices[ax] for ax in spatial_axes)
                spatial_sizes = [sl.stop - sl.start for sl in spatial_slices]
                if any(sz <= min_size for sz in spatial_sizes):
                    result.append((slices, region))
                    continue

                grad_region = gradient[spatial_slices] if gradient is not None else None
                if is_homogeneous(region, threshold_split, grad_region):
                    result.append((slices, region))
                else:
                    # split along the largest spatial dimension
                    lengths = {i: (spatial_slices[spatial_axes.index(i)].stop - spatial_slices[spatial_axes.index(i)].start)
                               for i in spatial_axes}
                    axis_to_split = max(lengths, key=lengths.get)
                    slc = slices[axis_to_split]
                    mid = (slc.start + slc.stop) // 2
                    left = list(slices)
                    right = list(slices)
                    left[axis_to_split] = slice(slc.start, mid)
                    right[axis_to_split] = slice(mid, slc.stop)
                    blocks.extend([tuple(left), tuple(right)])
            return result

        def merge_nd(img: ArrayLike, labels: ArrayLike, regions: List[Tuple[slice, ...]],
                     gradient: Optional[ArrayLike] = None) -> ArrayLike:
            labels_out = labels.clone() if is_torch else labels.copy()
            region_means: dict[int, float] = {}

            # assign mean per region label
            for region in regions:
                region_vals = extract_region_vals(img, region)
                ref_label = labels_out[tuple(s.start for s in region)]
                ref_label = int(ref_label.item()) if is_torch else int(ref_label)
                region_means[ref_label] = (
                    float(region_vals.mean().item()) if is_torch else float(np.mean(region_vals))
                )

            ndim = labels.ndim
            # 3^ndim neighborhood minus center
            offsets = np.array(list(np.ndindex(*([3] * ndim)))) - 1
            offsets = offsets[~np.all(offsets == 0, axis=1)]
            shape = labels.shape
            label_map = labels_out.cpu().numpy() if is_torch else labels_out
            adjacency: defaultdict[int, set[int]] = defaultdict(set)

            for region in regions:
                ref_l = int(label_map[tuple(s.start for s in region)])
                for off in offsets:
                    neighbor_pos = tuple(s.start + int(o) for s, o in zip(region, off))
                    if all(0 <= p < shape[i] for i, p in enumerate(neighbor_pos)):
                        nb_l = int(label_map[neighbor_pos])
                        if nb_l != ref_l:
                            adjacency[ref_l].add(nb_l)

            for l, neighs in adjacency.items():
                for nb in neighs:
                    if l not in region_means or nb not in region_means:
                        continue
                    a, b = region_means[l], region_means[nb]
                    if abs(a - b) > threshold_merge:
                        continue
                    if gradient is not None:
                        mask = (labels_out == nb)
                        if is_torch and not mask.any():
                            continue
                        gvals = gradient[mask]
                        gmean = float(gvals.mean().item()) if is_torch else float(np.mean(gvals))
                        if gmean > 0.3:
                            continue
                    labels_out = (
                        torch.where(labels_out == nb, l, labels_out)
                        if is_torch else np.where(labels_out == nb, l, labels_out)
                    )
                    mask = (labels_out == l)
                    vals = img[mask]
                    region_means[l] = float(vals.mean().item()) if is_torch else float(np.mean(vals))

            return labels_out

        # 1) Split
        regions_full = split_nd(image, gradient=gradient_map)

        # 2) Initial label map
        labels = (
            torch.zeros(spatial_shape, dtype=torch.int32, device=image.device)
            if is_torch else np.zeros(spatial_shape, dtype=int)
        )
        for i, (slices, _) in enumerate(regions_full, start=1):
            spatial_slices = tuple(slices[ax] for ax in spatial_axes)
            labels[spatial_slices] = i

        # 3) Merge
        merged = merge_nd(
            image,
            labels,
            [tuple(slices[ax] for ax in spatial_axes) for slices, _ in regions_full],
            gradient=gradient_map,
        )
        return (merged, labels) if return_labels else merged
    
    # -------------------------- Watershed --------------------------

    def watershed_nd(
        self,
        gradient: ArrayLike,
        markers: np.ndarray,
        return_lines: bool = False,
        connectivity: int = 1,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform N-dimensional watershed segmentation using a priority queue.

        Propagates region labels from initial markers across a gradient map,
        following the steepest descent (lowest elevation) rule. Fully ND-compatible
    and suitable for 2D, 3D or higher.

        Parameters
        ----------
        gradient : np.ndarray or torch.Tensor
            Input gradient or elevation map (lower = more likely to be flooded).
        markers : np.ndarray
            Initial labeled markers. Must have shape matching `gradient` and contain
            positive integers for seeded regions; 0 for unlabeled.
        return_lines : bool, default False
            If True, also returns the watershed lines (boundaries between regions).
        connectivity : int, default 1
            Neighborhood connectivity (1 for 6/18/26-connectivity in 3D).

        Returns
        -------
        segmented : np.ndarray
            Labeled segmentation result with same shape as input.
        lines : np.ndarray, optional
            Binary mask of watershed lines (only if `return_lines=True`).

        Notes
        -----
        - Works on any number of dimensions.
        - Uses a fast priority queue (heap) for label propagation.
        - The gradient acts as a cost surface: lower values are filled first.
        - The `markers` array defines initial labels and must be non-zero for seeds.
        - If Torch input is given, it is converted to NumPy internally.
        """
        is_torch = torch.is_tensor(gradient)
        grad = gradient.detach().cpu().numpy() if is_torch else np.array(gradient, copy=True)

        markers = markers.astype(np.int32, copy=False)
        segmented = markers.copy()
        shape = grad.shape

        # === Structuring element for ND connectivity ===
        footprint = ndimage.generate_binary_structure(len(shape), connectivity)
        offsets = np.array(list(zip(*np.where(footprint)))) - (np.array(footprint.shape) // 2)

        # === Priority queue init ===
        heap: List[Tuple[float, Tuple[int, ...], int]] = []
        for index in zip(*np.where(markers > 0)):
            heapq.heappush(heap, (float(grad[index]), index, int(markers[index])))

        # === Propagation ===
        while heap:
            _, index, label = heapq.heappop(heap)
            for off in offsets:
                nb = tuple(np.array(index) + off)
                if any((nb[i] < 0 or nb[i] >= shape[i]) for i in range(len(shape))):
                    continue
                if segmented[nb] == 0:
                    segmented[nb] = label
                    heapq.heappush(heap, (float(grad[nb]), nb, label))

        if return_lines:
            lines = self.extract_watershed_lines_nd(segmented, connectivity=connectivity)
            return segmented, lines
        return segmented

    def extract_watershed_lines_nd(self, labels: np.ndarray, connectivity: int = 1) -> np.ndarray:
        """
        Extract watershed lines (region boundaries) from an ND labeled segmentation map.

        Detects label transitions between neighboring voxels/pixels using the specified
        connectivity, and marks them as watershed lines.

        Parameters
        ----------
        labels : np.ndarray
            ND array of integer region labels (e.g., output from watershed segmentation).
        connectivity : int, default 1
            ND-connectivity to use when checking for neighbor transitions:
            - 1: face-connected neighbors (e.g., 6 in 3D),
            - 2: face + edge,
            - 3: full (face + edge + corner), if supported.

        Returns
        -------
        lines : np.ndarray
            Binary mask (uint8) of the same shape as `labels`, where 1 marks
            boundary voxels and 0 marks interior voxels.

        Notes
        -----
        - A voxel is considered a boundary if at least one of its neighbors
        has a different non-zero label.
        - Connectivity affects the neighborhood definition used in comparison.
        - Label 0 is ignored (assumed to be background/unlabeled).
        """
        structure = ndimage.generate_binary_structure(labels.ndim, connectivity)
        offsets = np.array(list(zip(*np.where(structure)))) - np.array(structure.shape) // 2

        lines = np.zeros_like(labels, dtype=np.uint8)
        shape = labels.shape
        for idx in zip(*np.where(labels > 0)):
            cur = int(labels[idx])
            for off in offsets:
                nb = tuple(np.array(idx) + off)
                if all(0 <= nb[d] < shape[d] for d in range(labels.ndim)):
                    if int(labels[nb]) > 0 and int(labels[nb]) != cur:
                        lines[idx] = 1
                        break
        return lines

    def segment_watershed_nd(
        self,
        image: ArrayLike,
        percentile_minima: float = 5.0,
        connectivity: int = 1,
        return_lines: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Segment an ND image using a watershed algorithm based on gradient magnitude.

        Applies Gaussian smoothing followed by Sobel gradient extraction to compute
        an elevation map. Local minima below a percentile threshold are used as markers
        for region flooding via watershed propagation.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input ND image. Channels are handled automatically if tagged.
        percentile_minima : float, default 5.0
            Percentile threshold to define local minima used as markers.
            Lower values result in fewer, stronger seed regions.
        connectivity : int, default 1
            ND-connectivity for structuring element in local minima detection and propagation.
        return_lines : bool, default False
            If True, also return watershed lines as a binary mask.

        Returns
        -------
        np.ndarray or tuple
            - If `return_lines=False`: segmentation map with region labels.
            - If `return_lines=True`: (segmentation, watershed_lines).

        Notes
        -----
        - Uses Gaussian filtering for pre-smoothing and Sobel filters for gradient computation.
        - Multichannel images are converted to scalar gradient magnitude via L2 norm.
        - Markers are computed from minima of the gradient below a percentile threshold.
        - The core propagation uses `watershed_nd` for ND-compatible flooding.
        """
        # === Detect spatial structure
        spatial_axes = self._get_axes(image)
        channel_axis = self.get_axis(image, "channel_axis")

        # 1) Slight smoothing (you can control sigma via feature_extractor config)
        smoothed = feature_extractor(
            image,
            features=["gaussian"],
            framework=self.framework,
            output_format=self.framework,
            layout_name=self.layout_name,
            layout_framework=self.layout_framework,
            stack=False,
        )

        # 2) Gradient magnitude
        grad_mag = feature_extractor(
            smoothed,
            features=["sobel"],
            framework=self.framework,
            output_format=self.framework,
            layout_name=self.layout_name,
            layout_framework=self.layout_framework,
            stack=False,
        )

        # -- Combine multichannel gradient if needed -- #
        if channel_axis is not None:
            grad_mag = (
                np.linalg.norm(grad_mag, axis=channel_axis)
                if isinstance(grad_mag, np.ndarray)
                else torch.norm(grad_mag, dim=channel_axis)
            )

        # 3) Local minima as markers
        thr = float(np.percentile(grad_mag if isinstance(grad_mag, np.ndarray) else grad_mag.cpu().numpy(),
                                  float(percentile_minima)))
        minima = grad_mag < thr
        structure = ndimage.generate_binary_structure(grad_mag.ndim, connectivity)
        markers, _ = ndimage.label(minima, structure=structure)

        # 4) Watershed
        segmented = self.watershed_nd(grad_mag, markers, return_lines=return_lines, connectivity=connectivity)
        return segmented 

    # -------------------------- Classical helpers --------------------------

    @staticmethod
    def predict_nd(
        features: np.ndarray,
        label: np.ndarray,
        axis: int = 1,
        n_train_slices: int = 1,
        fuse: bool = False,
        random_state: int = 0,
    ) -> np.ndarray:
        """
        Train a classical segmenter (Random Forest) on N slices of an ND feature volume,
        then predict segmentation labels across all slices along a specified axis.

        Parameters
        ----------
        features : np.ndarray
            ND array of features (e.g., from feature extraction), shape: (..., C, ...).
        label : np.ndarray
            Ground-truth label mask corresponding to training slices.
        axis : int, default 1
            Axis along which to perform training and prediction (e.g., slice axis).
        n_train_slices : int, default 1
            Number of slices (randomly sampled) to use for training the classifier.
        fuse : bool, default False
            If True, average predictions from multiple models (probabilistic fusion).
            If False, use only the first trained segmenter.
        random_state : int, default 0
            Seed for reproducibility in slice sampling and model training.

        Returns
        -------
        np.ndarray
            Predicted segmentation volume with same shape as the input (except channel axis removed).

        Notes
        -----
        - Uses `RandomForestClassifier` from scikit-learn.
        - Relies on `skimage.future` utilities for fitting and applying classical models.
        - Each selected slice trains a model on pixel-wise features and ground-truth labels.
        - Prediction is done slice-by-slice along the given axis.
        - Output is reoriented to match the original input layout.
        """
        rng = np.random.default_rng(random_state)
        feats_0 = np.moveaxis(features, axis, 0)
        n_slices = feats_0.shape[0]

        train_idx = rng.choice(n_slices, size=n_train_slices, replace=False)
        segmenters = []
        for idx in train_idx:
            train_feat = feats_0[idx]
            from sklearn.ensemble import RandomForestClassifier

            clf = RandomForestClassifier(
                n_estimators=200,
                n_jobs=-1,
                max_depth=None,
                max_samples=5000,
                criterion="entropy",
                random_state=random_state + int(idx),
            )
            seg = future.fit_segmenter(clf, train_feat, label)
            segmenters.append(seg)

        preds = []
        for i in range(n_slices):
            feat = feats_0[i]
            if fuse:
                probas = np.mean(
                    [future.predict_segmenter(feat, seg, return_proba=True) for seg in segmenters],
                    axis=0,
                )
                preds.append(probas)
            else:
                pred = future.predict_segmenter(feat, segmenters[0])
                preds.append(pred)

        result = np.stack(preds, axis=0)
        return np.moveaxis(result, 0, axis)

    def _debug_info(self, step_name: str, img, threshold=None) -> None:
        """
        Print debug information after a segmentation step if debug mode is enabled.

        Outputs basic statistics (shape, dtype, min/max, unique values) and optionally
        the threshold used during the step. Compatible with both NumPy and Torch inputs.

        Parameters
        ----------
        step_name : str
            Name of the current processing step (e.g., 'thresholding', 'region growing').
        img : np.ndarray or torch.Tensor
            Image or segmentation array to inspect.
        threshold : float, optional
            Threshold value used in the step (if any), included in the debug output.

        Returns
        -------
        None
            Prints formatted diagnostic information to stdout.

        Notes
        -----
        - Requires `self.segmenter_cfg.debug = True` to activate.
        - Automatically detects backend (Torch or NumPy) for appropriate inspection.
        """
        if not self.segmenter_cfg.debug:
            return

        framework = "torch" if isinstance(img, torch.Tensor) else "numpy"
        print(f"\n[SegmenterND::DEBUG] Step: {step_name}")
        print(f"  → shape: {img.shape}")
        print(f"  → dtype: {img.dtype}")
        
        if threshold is not None:
            print(f"  → threshold used: {threshold:.4f}")

        if framework == "torch":
            print(f"  → min: {img.min().item():.4f}, max: {img.max().item():.4f}")
            print(f"  → unique values: {torch.unique(img)}")
        else:
            print(f"  → min: {np.min(img):.4f}, max: {np.max(img):.4f}")
            print(f"  → unique values: {np.unique(img)}")


def get_nd_neighbors(ndim: int, connectivity: str = "full") -> np.ndarray:
    """
    Generate offset vectors to enumerate ND neighbors based on a connectivity rule.

    Useful for building structuring elements, region-growing, or adjacency graphs
    in N-dimensional image processing.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (e.g., 2 for 2D, 3 for 3D, etc.).
    connectivity : {'full', 'face', 'minimal'}, default 'full'
        Type of neighborhood to generate:
        - 'full'     : All 3^n - 1 surrounding neighbors (includes corners and diagonals).
        - 'face'     : Only 2n face-connected neighbors (±1 step along each axis).
        - 'minimal'  : Only n neighbors with a single positive step in each axis.

    Returns
    -------
    np.ndarray
        Array of shape (n_neighbors, ndim), where each row is a neighbor offset.

    Raises
    ------
    ValueError
        If the provided connectivity type is not recognized.

    Examples
    --------
    >>> get_nd_neighbors(2, 'face')
    array([[ 1,  0],
           [-1,  0],
           [ 0,  1],
           [ 0, -1]])
    """
    if connectivity == "full":
        offsets = np.array(list(product([-1, 0, 1], repeat=ndim)))
        return offsets[np.any(offsets != 0, axis=1)]  # exclude (0,...,0)

    elif connectivity == "face":
        offsets = []
        for i in range(ndim):
            vec = np.zeros(ndim, dtype=int)
            vec[i] = 1
            offsets.append(vec.copy())
            vec[i] = -1
            offsets.append(vec.copy())
        return np.array(offsets)

    elif connectivity == "minimal":
        return np.eye(ndim, dtype=int)  # only positive unit steps

    else:
        raise ValueError("Unsupported connectivity: choose from 'full', 'face', or 'minimal'")
    
# ======================================================================
#                      Convenience wrapper
# ======================================================================

def segmenter_nd(
    img: ArrayLike,
    segmenter_mode: Optional[str] = None,
    threshold: Optional[float] = None,
    multi_thresholds: Optional[List[float]] = None,
    num_classes: Optional[int] = None,
    framework: Framework = "numpy",
    output_format: Framework = "numpy",
    layout_name: str = "HWC",
    layout_framework: Framework = "numpy",
    layout_ensured_name: Optional[str] = None,
    processor_strategy: Optional[str] = None,
    use_channels: bool = True,
    use_features: bool = False,
    seeds: Optional[List[Tuple[int, ...]]] = None,
    n_seeds: int = 5,
) -> ArrayLike:
    """
    Apply ND segmentation using a preconfigured SegmenterND instance.

    Builds a complete segmentation pipeline from scratch with user-specified
    parameters and configuration. Supports a wide range of methods (thresholding,
    region growing, k-means, watershed, etc.), and is compatible with both
    NumPy and Torch inputs.

    Parameters
    ----------
    img : np.ndarray or torch.Tensor
        Input image or volume to segment (ND format, normalized preferred).
    segmenter_mode : str, optional
        Segmentation method to use (e.g., 'otsu', 'kmeans', 'region_growing').
    threshold : float, optional
        Threshold value for single-threshold methods (e.g., fixed, iterative).
    multi_thresholds : list of float, optional
        List of thresholds for multi-class segmentation.
    num_classes : int, optional
        Number of target classes (used in methods like 'multi', 'kmeans').
    framework : {'numpy', 'torch'}, default 'numpy'
        Backend framework used for computation.
    output_format : {'numpy', 'torch'}, default 'numpy'
        Format of the output segmentation map.
    layout_name : str, default 'HWC'
        Mnemonic for layout (e.g., 'HWC', 'NCHW', etc.).
    layout_framework : {'numpy', 'torch'}, default 'numpy'
        Framework to resolve layout naming conventions.
    layout_ensured_name : str, optional
        Optional expected layout name to enforce.
    processor_strategy : str, optional
        Strategy for feature processing (e.g., 'vectorized', 'parallel', 'torch').
    use_channels : bool, default True
        Whether to use the channel axis in processing.
    use_features : bool, default False
        Whether to use the feature axis in processing.
    seeds : list of tuple[int, ...], optional
        Seed points for region-based methods (e.g., region growing).
    n_seeds : int, default 5
        Number of random seeds to generate if `seeds` is None.

    Returns
    -------
    np.ndarray or torch.Tensor
        Segmentation result with format and layout resolved accordingly.

    Notes
    -----
    - This is a high-level wrapper for `SegmenterND` and associated configs.
    - Layout, framework, and axis semantics are automatically handled.
    - Supports both intensity- and feature-based segmentation strategies.
    """
    # ====[ Fallback ]====
    processor_strategy=processor_strategy or "vectorized" if framework == "numpy" else "torch"   
        
    # ====[ Configuration ]====
    segments_params: Dict[str, Any] = {"mode": segmenter_mode, "threshold": threshold, "multi_thresholds": multi_thresholds, "num_classes": num_classes,
                       "kmeans_k": num_classes, "use_channels": use_channels, "use_features": use_features, "seeds": seeds, "n_seeds": n_seeds}
    proc_params: Dict[str, Any] = {"processor_strategy": processor_strategy,}
    layout_params: Dict[str, Any] = {"layout_name": layout_name, "layout_framework": layout_framework, "layout_ensured_name": layout_ensured_name,}
    global_params: Dict[str, Any] = {"framework": framework, "output_format": output_format}
                 
    segments = SegmenterND( 
                           segmenter_cfg=SegmenterConfig(**segments_params),
                           img_process_cfg=ImageProcessorConfig(**proc_params),
                           layout_cfg=LayoutConfig(**layout_params),
                           global_cfg=GlobalConfig(**global_params),                             
                           )   
    
    img_copy = segments.safe_copy(img)
    
    return segments(img_copy) 