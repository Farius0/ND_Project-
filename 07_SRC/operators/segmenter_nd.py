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
        Apply the configured segmentation strategy to the image.
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
        Apply thresholding using ThresholdingOperator (Otsu, Yen, Li, etc.).

        Parameters
        ----------
        image : np.ndarray | torch.Tensor
            Input image to segment.

        Returns
        -------
        Binary mask or thresholded output.
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
        Fixed threshold segmentation.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
        threshold : float (0-1 for normalized image)

        Returns
        -------
        Binary mask (same shape).
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
        Adaptive global thresholding (Ridler-Calvard) for ND images.

        Automatically separates foreground and background by iterative refinement.

        Parameters
        ----------
        epsilon : float
            Convergence criterion.
        max_iter : int
            Maximum number of iterations.

        Returns
        -------
        Binary mask (same shape as input).
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
        Multi-threshold segmentation: divides the image into N+1 classes.

        Parameters
        ----------
        thresholds : list[float]
            List of thresholds (sorted). Must be in [0, 1] if image is normalized.
        normalize_output : bool
            If True, output will be scaled to [0, 1] by dividing class labels.

        Returns
        -------
        Segmented image with discrete class labels (or normalized in [0,1]).
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
        Entropy-based global thresholding using Shannon criterion.

        Returns
        -------
        Binary mask after optimal entropy-based split.
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
        Multi-class thresholding based on Otsu generalized to multiple thresholds.
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
        Segment ND image using KMeans clustering (intensity or features).
        
        Parameters
        ----------
        k : int or None
            Number of clusters (if known).
        auto_k : bool
            Whether to estimate optimal k.
        method : str
            'silhouette' or 'elbow' for auto-k selection.
        max_k : int
            Max number of clusters to test (if auto_k=True).
        normalize_output : bool
            If True, output values are in [0,1], else raw class labels [1,...,k].

        Returns
        -------
        Segmented ND image.
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
    
    # def kmeans_nd_torch(self,
    #                 x: torch.Tensor,
    #                 k: int = 4,
    #                 max_iter: int = 100,
    #                 tol: float = 1e-4,
    #                 normalize_output: bool = False):
    #     """
    #     Torch-native ND KMeans clustering on feature volume.

    #     Parameters
    #     ----------
    #     features : torch.Tensor
    #         ND feature tensor (..., F) where F = number of features.
    #     k : int
    #         Number of clusters.
    #     max_iter : int
    #         Maximum number of KMeans iterations.
    #     tol : float
    #         Tolerance for convergence (centroid shift).
    #     normalize_output : bool
    #         If True, normalize output labels to [0, 1].

    #     Returns
    #     -------
    #     labels : torch.Tensor
    #         ND label map (same shape as features[..., 0]).
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
        Segment ND image using Agglomerative Clustering.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            ND image (can be multichannel or feature map).
        n_clusters : int
            Number of clusters to form (default: self.k).
        linkage : str
            Linkage criterion: 'ward', 'average', 'complete'.
        affinity : str
            Distance metric: 'euclidean', 'manhattan', etc.
            Note: ignored if linkage == 'ward'.
        normalize_output : bool
            If True, scale output to [0,1].

        Returns
        -------
        labels : np.ndarray or torch.Tensor
            Clustered label map (same spatial shape).
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
        Segment ND image using HDBSCAN clustering (density-based, robust, auto-k).

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            ND image (can be multichannel or feature map).
        min_cluster_size : int
            Minimum size for a region to be considered a cluster.
        min_samples : int or None
            Minimum samples to define dense region (default: same as min_cluster_size).
        normalize_output : bool
            If True, scale output to [0,1].

        Returns
        -------
        labels : np.ndarray or torch.Tensor
            Clustered label map (same spatial shape).
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
        segmentation: np.ndarray, return_mode: str = "labels", palette: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Format segmentation output into label, RGB or one-hot format.

        Parameters
        ----------
        segmentation : np.ndarray
            ND label array with integer labels.
        return_mode : str
            One of 'labels', 'rgb', 'onehot'.
        palette : Optional[np.ndarray]
            Optional color palette for RGB mode.

        Returns
        -------
        np.ndarray
            Segmentation formatted accordingly.
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
        Region growing segmentation for ND images, with multichannel and priority-based expansion.
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
        ND Split-and-Merge segmentation with gradient-aware merging.
        Fully scalar-consistent, channel-axis aware, and ND-compatible.
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
        ND-compatible Watershed propagation using a priority queue.

        Parameters
        ----------
        gradient : np.ndarray or torch.Tensor
            Gradient or elevation map.
        markers : np.ndarray
            Labeled minima (e.g., from detect_basins_nd).
        return_lines : bool
            If True, returns both segmentation and watershed lines.
        connectivity : int
            Connectivity to use for neighbor propagation.

        Returns
        -------
        segmented : np.ndarray
            Labeled segmentation.
        lines (optional) : np.ndarray
            Watershed lines (binary mask).
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
        Extract watershed lines (boundaries) from labeled ND segmentation.

        Parameters
        ----------
        labels : np.ndarray
            Labeled segmentation (integer).
        connectivity : int
            ND-connectivity (1 = faces, 2 = edges+faces, etc.)

        Returns
        -------
        lines : np.ndarray
            Binary mask of watershed lines (same shape as labels).
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
        Perform ND watershed segmentation using gradient magnitude from feature_extractor.
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
        Train a classical segmenter on 1..N slices and predict across the given axis.
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
        Print debug information after a segmentation step.
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


def get_nd_neighbors(ndim: int, connectivity: str ="full") -> np.ndarray:
    """
    Generate ND neighbor offsets based on connectivity.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions.
    connectivity : str
        Type of connectivity: 'full', 'face', or 'minimal'.
        - 'full'   → 3^n - 1 neighbors (includes diagonals)
        - 'face'   → 2n neighbors (1 step in ± each axis)
        - 'minimal' → Only the 1-step shifts without diagonals, no opposite pairs.

    Returns
    -------
    np.ndarray
        Array of shape (n_neighbors, ndim) containing all valid offsets.
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
    Convenience wrapper to build and run SegmenterND.
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