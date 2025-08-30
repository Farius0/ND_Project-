# ==================================================
# ================ MODULE:image_io =================
# ==================================================
from __future__ import annotations

import os
# import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Any, List, Dict, Literal

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont

from operators.resize_image import ResizeOperator

# sys.path.append(str(Path.cwd().parent))

from core.config import (
    LayoutConfig,
    GlobalConfig,
    ResizeConfig,
    ImageProcessorConfig,
)
from core.layout_axes import get_layout_axes, resolve_and_clean_layout_tags
from core.operator_core import OperatorCore
from operators.metrics import MetricEvaluator

# Public API
__all__ = ["ImageIO"]

# Typing helpers
PathLike = Union[str, Path]
ReadHandler = Callable[[PathLike], Tuple[np.ndarray, Optional[Any]]]
Framework = Literal["numpy", "torch"]

# ==================================================
# =================== ImageIO ======================
# ==================================================

class ImageIO(OperatorCore):
    """
    Unified image reader and converter for standard 2D/3D formats.

    Inherits from OperatorCore to enable layout-aware loading, axis tagging,
    and dual-backend conversion (NumPy ↔ Torch).

    Notes
    -----
    - Supports standard formats: PNG, JPG, TIFF, DICOM (2D and 3D).
    - ND-aware: processes both single images and full volumetric stacks.
    - Dual-backend compatible: returns NumPy arrays or Torch tensors, 
      preserving dtype, layout, and shape.
    - Integrates layout and global configuration for consistent preprocessing.
    """

    # ====[ INITIALIZATION – ImageIO with full axis support ]====
    def __init__(
        self,
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize the ImageIO module with layout and global configurations.

        Parameters
        ----------
        layout_cfg : LayoutConfig
            Layout configuration with axis naming and semantic roles.
        global_cfg : GlobalConfig
            Global configuration for backend, device, dtype, and format behavior.
        """
        # === Configuration ===
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg

        # Resolved axes / meta
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

        # Handlers registry
        self._handlers: dict[str, ReadHandler] = {
            "jpg": self._read_pil,
            "jpeg": self._read_pil,
            "png": self._read_pil,
            "bmp": self._read_pil,
            "tif": self._read_pil,
            "tiff": self._read_pil,
            "dcm": self._read_dicom,
        }

        # Global flags
        self.framework: Framework = self.global_cfg.framework.lower()
        self.output_format: Framework = self.global_cfg.output_format.lower()
        self.add_batch_dim: Optional[bool] = self.global_cfg.add_batch_dim
        self.add_channel_dim: Optional[bool] = self.global_cfg.add_channel_dim
        self.normalize: bool = bool(self.global_cfg.normalize)
        self.verbose: bool = bool(self.global_cfg.verbose)
        self.device: str = (
            "cuda"
            if (torch.cuda.is_available() and self.framework == "torch")
            else self.global_cfg.device
        )

        # Output layout to enforce uniform shapes
        default_layout: str = "NCHW" if self.framework == "torch" else "HWC"
        self.layout_ensured_name: str = (
            self.layout_cfg.layout_ensured_name or default_layout
        )
        try:
            self.layout_ensured = get_layout_axes(
                self.framework, self.layout_ensured_name
            )
        except Exception:
            # fallback if missing
            self.layout_ensured_name = default_layout
            self.layout_ensured = get_layout_axes(self.framework, default_layout)

        # Initialize OperatorCore
        super().__init__(layout_cfg=self.layout_cfg, global_cfg=self.global_cfg)

        # Last metadata (e.g., DICOM dataset)
        self._last_metadata: Optional[Any] = None

    def add_format_handler(self, extension: str, handler_fn: ReadHandler) -> None:
        """
        Register a custom format handler for loading non-standard image files.

        Parameters
        ----------
        extension : str
            File extension (without leading dot), e.g., 'nii', 'mhd', 'tif'.
        handler_fn : Callable
            A function that takes a file path and returns a tuple:
            (image_array, metadata), where metadata can be None.

        Returns
        -------
        None
            The handler is added internally and used automatically on matching file extensions.
        """
        self._handlers[extension.lower()] = handler_fn

    def _read_pil(self, path: PathLike) -> Tuple[np.ndarray, None]:
        """
        Read standard 2D image formats (e.g., PNG, JPG, TIFF) using PIL.

        Automatically converts to RGB if a channel axis is expected in the layout.

        Parameters
        ----------
        path : str or Path
            Path to the image file.

        Returns
        -------
        Tuple[np.ndarray, None]
            - Image as a NumPy array:
            (H, W) for grayscale or (H, W, 3) for color.
            - Metadata is None.
        """
        if self.layout_name:
            if "C" in self.layout_name.upper():
                img = Image.open(path).convert("RGB")
            else:
                img = Image.open(path)
        elif self.axes["channel_axis"] is not None:
            img = Image.open(path).convert("RGB")
        else:
            img = Image.open(path)

        return np.array(img), None

    def _read_dicom(self, path: PathLike) -> Tuple[np.ndarray, Any]:
        """
        Read a DICOM file and extract its pixel data as a NumPy array.

        Applies VOI LUT if available. Optionally stores the DICOM metadata
        for downstream use (e.g., patient info, modality, acquisition params).

        Parameters
        ----------
        path : str or Path
            Path to the DICOM file.

        Returns
        -------
        Tuple[np.ndarray, FileDataset]
            - Image array (typically 2D), cast to NumPy format.
            - Full DICOM dataset (`pydicom.dataset.FileDataset`) as metadata.

        Notes
        -----
        - If VOI LUT (windowing) is defined, it is applied for intensity mapping.
        - RescaleSlope/Intercept and MONOCHROME1 inversion are currently commented out
        but can be re-enabled for full grayscale correction.
        """
        ds = pydicom.dcmread(path)
        self._last_metadata = ds

        # VOI LUT (windowing) if present
        try:
            arr = apply_voi_lut(ds.pixel_array, ds)
        except Exception:
            arr = ds.pixel_array

        # # Rescale using slope/intercept if available
        # slope = float(getattr(ds, "RescaleSlope", 1.0))
        # intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        # arr = arr.astype(np.float32) * slope + intercept

        # # Photometric interpretation: MONOCHROME1 means inverted
        # photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        # if photometric == "MONOCHROME1":
        #     # invert: high values should become dark
        #     maxv = float(np.max(arr) if arr.size else 0.0)
        #     arr = maxv - arr

        return np.asarray(arr), ds

    # ====[ READ – robust with tagging ]====
    def read_image(
        self,
        image: Union[str, Path, np.ndarray, torch.Tensor],
        return_metadata: bool = False,
        framework: Optional[Framework] = None,
        tag_as: str = "original",
        enable_uid: bool = True,
        op_params: Optional[dict] = None,
        postprocess_fn: Optional[Callable[[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]]] = None,
        strict_reset: bool = False,
        require_shape_match: bool = True,
        require_layout_match: bool = True,
        require_uid: Optional[bool] = None,
        track: bool = True,
        trace_limit: int = 10,
    ) -> Union[np.ndarray, torch.Tensor, Tuple[Union[np.ndarray, torch.Tensor], Any]]:
        """
        Read and convert an image from path or array, with layout enforcement and optional metadata.

        Handles image loading, layout resolution, axis tagging, and UID assignment.
        Supports post-processing and traceability options.

        Parameters
        ----------
        image : str | Path | ndarray | Tensor
            Image path or in-memory image data to be processed.
        return_metadata : bool, optional
            If True, returns a tuple (image, metadata).
        framework : {'torch','numpy'}, optional
            Target backend for the output format. Defaults to `self.framework`.
        tag_as : str, optional
            Status tag used for layout/trace tracking (e.g., 'original', 'resized').
        enable_uid : bool, optional
            Whether to assign a unique identifier (UID) to the image tag.
        op_params : dict, optional
            Additional metadata to include in the tag (e.g., operation history).
        postprocess_fn : callable, optional
            Function applied to the image after conversion and tagging.
        strict_reset : bool, optional
            If True, purge all previous tags before processing.
        require_shape_match : bool, optional
            Enforce consistency of shape with previous tags (if any).
        require_layout_match : bool, optional
            Enforce consistency of layout with expected layout.
        require_uid : bool or None, optional
            Override for UID enforcement policy. If None, uses `enable_uid`.
        track : bool, optional
            Whether to store tracking metadata in the tag.
        trace_limit : int, optional
            Maximum depth of the trace history.

        Returns
        -------
        image : ndarray or Tensor
            Processed image in the target framework (NumPy or Torch).
        metadata : optional
            If `return_metadata` is True, returns the metadata alongside the image.
        """
        metadata = None
        fw = framework or self.framework

        if strict_reset:
            self.purge_all_tags(verbose=self.verbose)

        if isinstance(image, Path):
            image = str(image)

        if isinstance(image, str):
            ext = image.lower().split(".")[-1]
            if ext in self._handlers:
                image, metadata = self._handlers[ext](image)
            else:
                raise ValueError(f"[ImageIO] Unsupported file format: .{ext}")

        # Ajust require_uid
        _require_uid = enable_uid if require_uid is None else require_uid
        _require_layout = bool(self.layout_ensured_name) and require_layout_match

        tracked = self.ensure_format(
            image=image,
            framework=fw,
            tag_as=tag_as,
            layout=self.layout_ensured,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            require_shape_match=require_shape_match,
            require_layout_match=_require_layout,
            require_uid=_require_uid,
            expected_layout_name=self.layout_ensured_name,
            require_status=tag_as,
            force_reset=True,
        )

        result = tracked.get()

        if postprocess_fn is not None:
            result = postprocess_fn(result)
            if self.verbose:
                print("[read_image] Postprocess function applied.")

        return (result, metadata) if return_metadata else result

    # ====[ LOAD BATCH – Read & Resize Multiple Images ]====
    def load_batch(
        self,
        paths: List[Union[str, Path]],
        to: str = "torch",
        stack: bool = True,
        enable_uid: bool = True,
        op_params: Optional[dict] = None,
        match_to: Optional[str] = "first",
    ) -> Union[List[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor]:
        """
        Load multiple images from disk with optional resizing and batching.

        Supports unified layout conversion, UID tagging, and traceable metadata for each image.

        Parameters
        ----------
        paths : list of str or Path
            List of file paths to image files. Must be non-empty.
        to : {'torch', 'numpy'}, default 'torch'
            Target backend for output. Determines array type (Torch tensor or NumPy array).
        stack : bool, default True
            If True, stack images into a single ND batch (e.g., (N, C, H, W)).
            If False, return a list of individual images.
        enable_uid : bool, default True
            Assign a unique UID to each image and record layout trace metadata.
        op_params : dict, optional
            Additional tag metadata to attach to each image (e.g., {'source': 'dataset-A'}).
        match_to : {'first', None}, default 'first'
            If 'first', resize all images to match the first one (spatial dims only).
            If None, no resizing is applied; stacking may fail if shapes differ.

        Returns
        -------
        list | ndarray | Tensor
            - If `stack=False`: list of images (each shaped (C, H, W) or (C, D, H, W)).
            - If `stack=True`: single batch array of shape (N, C, H, W) or (N, C, D, H, W),
            using the target backend (`to`).

        Notes
        -----
        - Layout and format conversion follow `LayoutConfig` and `GlobalConfig`.
        Disk formats like HWC (e.g., PNG/JPEG) are internally converted to standard layout.
        - When resizing is enabled (`match_to='first'`), interpolation mode is backend-specific.
        - All images are tagged with layout, UID, and transformation trace for reproducibility.
        """

        if self.verbose:
            print(f"\n[ImageIO] Loading {len(paths)} images...")

        if self.verbose:
            print(f"\n[ImageIO] Loading {len(paths)} images...")

        images: List[Union[np.ndarray, torch.Tensor]] = []
        for i, path in enumerate(paths):
            try:
                img = self.read_image(
                    image=path,
                    framework=to,
                    tag_as="original",
                    enable_uid=enable_uid,
                    op_params=op_params,
                )
                if self.verbose:
                    print(f"[ImageIO] Image {i} → shape = {tuple(img.shape)}")
                images.append(img)
            except Exception as e:
                print(f"[ImageIO] Error loading image {i} ({path}): {e}")

        if len(images) == 0:
            return images

        # Resize to reference
        if match_to == "first" and len(images) > 1:
            reference = images[0]
            tracker = self.track(reference)
            tag = tracker.get_tag()
            layout = tag.get("layout", {}) or self.layout_ensured
            layout_name = tag.get("layout_name", None) or self.layout_ensured_name

            if self.verbose:
                print(f"[ImageIO] Resizing all to match shape: {reference.shape}")

            resizer = ResizeOperator(
                resize_cfg=ResizeConfig(
                    size=None,
                    resize_strategy="auto",
                    layout_ensured=layout,
                ),
                img_process_cfg=ImageProcessorConfig(processor_strategy="auto"),
                layout_cfg=LayoutConfig(layout=layout, layout_name=layout_name),
                global_cfg=GlobalConfig(
                    framework=to,
                    output_format=to,
                    add_batch_dim=self.add_batch_dim,
                ),
            )

            resized_images: List[Union[np.ndarray, torch.Tensor]] = []
            for i, img in enumerate(images):
                try:
                    if img.shape != reference.shape:
                        resized = resizer(img, match_to=reference)
                        if self.verbose:
                            print(
                                f"[ImageIO] Resized image {i} → shape: {resized.shape}"
                            )
                    else:
                        resized = img
                    resized_images.append(resized)
                except Exception as e:
                    print(f"[ImageIO] Resize failed for image {i}: {e}")
                    resized_images.append(img)
            images = resized_images

        # Stack if requested
        if not stack:
            return images

        if self.verbose:
            print("[ImageIO] Stacking images...")

        try:
            tagger = self.track(images[0])
            outputs = torch.stack(images) if to == "torch" else np.stack(images)

            layout_name, axes_tags = resolve_and_clean_layout_tags(
                tagger,
                self.framework,
                self.layout_name,
                prefix="N",
                remove_prefix=False,
            )

            return tagger.stack_from(
                images,
                axis=0,
                update_tags={
                    "status": "batch",
                    "layout_name": layout_name,
                    "shape_after": outputs.shape,
                    **axes_tags,
                },
            ).get()
        except Exception as e:
            raise RuntimeError(f"Failed to stack images: {e}")

    @staticmethod
    def treat_image_and_add_metric(
        image: np.ndarray,
        reference_image: np.ndarray,
        metric: str = "psnr",
        subfolder: str = "fonts",
        fname: str = "Poppins-BoldItalic.ttf",
    ) -> np.ndarray:
        """
        Annotate an image with a quality metric (e.g., PSNR) computed against a reference.

        Computes the specified metric between the input image and a reference,
        then overlays the metric value as a text label on the image.

        Parameters
        ----------
        image : np.ndarray
            Image to annotate (typically a predicted or processed version).
        reference_image : np.ndarray
            Ground-truth reference image for metric comparison.
        metric : str, default 'psnr'
            Metric name to compute and display (e.g., 'psnr', 'ssim', 'mse').
        subfolder : str, default 'fonts'
            Relative path to the folder containing the desired font file.
        fname : str, default 'Poppins-BoldItalic.ttf'
            Font file to use for drawing the annotation.

        Returns
        -------
        np.ndarray
            Annotated image with the metric label rendered at the bottom-left corner.

        Notes
        -----
        - Automatically converts image and reference to uint8 if needed.
        - Falls back to a default font if the specified font file is not found.
        - Handles both grayscale and RGB images.
        """
        # === Path Setup ===
        file_path = Path(__file__).parent
        font_path = str(file_path.parent / subfolder / fname)

        # === Metric Calculation ===
        evaluator = MetricEvaluator(metrics=[metric], return_dict=True)
        results = evaluator(reference_image, image)
        value = results.get(metric, None)
        metric_text = (
            f"{metric.upper()}: {value:.2f}" if isinstance(value, (int, float)) else f"{metric.upper()}: ?"
        )

        # === Format Normalization ===
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        if image.dtype != np.uint8:
            image = safe_image_to_uint8(image)
        if reference_image.dtype != np.uint8:
            reference_image = safe_image_to_uint8(reference_image)

        # === Dimensions ===
        H = image.shape[0]

        # === Font Setup (fallback safe) ===
        try:
            font = ImageFont.truetype(font_path, size=int(H * 0.07))
        except Exception:
            font = ImageFont.load_default()

        # === Convert to PIL ===
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)
        text_bbox = draw.textbbox((0, 0), metric_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # === Background Overlay ===
        overlay = Image.new("RGBA", image_pil.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.rectangle(
            [(0, H - int(text_h * 1.75)), (int(text_w * 1.15), H)],
            fill=(255, 255, 255, 255),
        )

        # === Composite and Draw Text ===
        image_pil = Image.alpha_composite(image_pil.convert("RGBA"), overlay)
        if image.ndim == 3:
            image_pil = image_pil.convert("RGB")
        elif image.ndim == 2:
            image_pil = image_pil.convert("L")
        draw = ImageDraw.Draw(image_pil)
        fill_color = 0 if image_pil.mode == "L" else (0, 0, 0)
        draw.text(
            (int(text_w * 0.05), H - int(text_h * 1.75)),
            metric_text,
            font=font,
            fill=fill_color,
        )
        return np.array(image_pil)

    def save_comparison(
        self,
        folder: str,
        file_name: str,
        names_list: List[List[str]],
        images_list: List[List[np.ndarray]],
        reference_image_list: Optional[List[np.ndarray]] = None,
        metric: str = "psnr",
        trajectories_list: Optional[List[List[List[np.ndarray]]]] = None,
        save: bool = True,
        scale_plot: Tuple[int, int] = (2, 2),
    ) -> None:
        """
        Display and optionally save comparison grids of images with annotations and metric plots.

        Shows side-by-side image groups (rows), optionally overlays metric values (e.g. PSNR),
        and generates a PSNR evolution plot when trajectories are provided.

        Parameters
        ----------
        folder : str
            Target output folder (relative or absolute).
        file_name : str
            Base name (without extension) for the saved output files.
        names_list : list of list of str
            Titles for each image in each row (e.g., method names).
        images_list : list of list of ndarray
            List of image groups to compare. Each sublist represents one row.
        reference_image_list : list of ndarray, optional
            One reference image per row (ground truth), used for metric calculation and display.
        metric : str, default 'psnr'
            Metric name to compute and annotate (e.g., 'psnr', 'ssim').
        trajectories_list : list, optional
            Per-row list of per-method trajectories, used to compute metric evolution curves.
        save : bool, default True
            Whether to save the figure(s) to disk.
        scale_plot : tuple of int, default (2, 2)
            Size scaling factors (width, height) for each image in inches.

        Returns
        -------
        None
            Saves image grid and metric plot to disk if `save=True`.

        Notes
        -----
        - The output comparison grid is saved as "<file_name>_comparison.png".
        - The PSNR evolution plot (if `trajectories_list` is provided) is saved as
        "<file_name>_psnr_plot.png".
        - All images are normalized to uint8 before display.
        - Metric annotations are drawn using `treat_image_and_add_metric()`.
        """
        file_path = Path(__file__).parent
        path = str(file_path.parent / folder)
        Path(path).mkdir(parents=True, exist_ok=True)
        # os.makedirs(path, exist_ok=True)

        use_ref = reference_image_list is not None
        num_groups = len(images_list) if not use_ref else len(reference_image_list)
        num_images_per_group = max(len(group) for group in images_list)
        total_cols = num_images_per_group + (1 if use_ref else 0)

        fig, axes = plt.subplots(
            num_groups,
            total_cols,
            figsize=(scale_plot[0] * total_cols, scale_plot[1] * num_groups),
            dpi=200,
        )

        # Normalize axes to 2D list
        if num_groups == 1 and total_cols == 1:
            axes = [[axes]]
        elif num_groups == 1:
            axes = [axes]
        elif total_cols == 1:
            axes = [[ax] for ax in axes]

        for row_idx in range(num_groups):
            img_ref = reference_image_list[row_idx] if use_ref else None
            img_group = images_list[row_idx]
            name_group = names_list[row_idx]

            col_base = 0
            if use_ref:
                img_plt = safe_image_to_uint8(img_ref) if img_ref.dtype != np.uint8 else img_ref
                axes[row_idx][0].imshow(img_plt, cmap="gray")
                axes[row_idx][0].set_title("Ground Truth")
                axes[row_idx][0].axis("off")
                col_base = 1

            for j, (img, name) in enumerate(zip(img_group, name_group), start=col_base):
                if metric and use_ref and img_ref is not None:
                    img = self.treat_image_and_add_metric(img, img_ref, metric=metric)
                    axes[row_idx][j].imshow(img, cmap="gray")
                else:
                    axes[row_idx][j].imshow(safe_image_to_uint8(img), cmap="gray")
                axes[row_idx][j].set_title(name)
                axes[row_idx][j].axis("off")

        plt.tight_layout()
        comparison_path = os.path.join(path, f"{file_name}_comparison.png")
        if save:
            plt.savefig(comparison_path, bbox_inches="tight", pad_inches=0.1)
            plt.close()

        # Metric trajectory plot (PSNR, etc.)
        if trajectories_list is not None and use_ref:
            plt.figure(figsize=(12, 8), dpi=200)
            for img_ref, trajectory_group, name_group in zip(
                reference_image_list, trajectories_list, names_list
            ):
                markers = ["o", "s", "D", "^", "v", "p", "*", "X"]
                linestyles = ["-", "--", "-.", ":"]
                colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory_group)))

                for idx, (trajectory, name) in enumerate(
                    zip(trajectory_group, name_group[-len(trajectory_group) :])
                ):
                    psnr_values = [
                        cv2.PSNR(
                            safe_image_to_uint8(img_ref),
                            safe_image_to_uint8(traj),
                        )
                        for traj in trajectory
                    ]
                    marker = markers[idx % len(markers)]
                    linestyle = linestyles[idx % len(linestyles)]
                    color = colors[idx]
                    plt.plot(
                        range(len(psnr_values)),
                        psnr_values,
                        marker=marker,
                        linestyle=linestyle,
                        label=name,
                        alpha=0.7,
                        linewidth=1.25,
                    )

            plt.xlabel("Iterations")
            plt.ylabel("PSNR (dB)")
            plt.title("PSNR Evolution Across Trajectories")
            plt.legend(loc="best")
            plt.grid(True)

            psnr_plot_path = os.path.join(path, f"{file_name}_psnr_plot.png")
            if save:
                plt.savefig(psnr_plot_path, bbox_inches="tight", pad_inches=0.1)
                plt.close()

    # ====[ VALIDATE IMAGE – Safety Check Before Conversion ]====
    def validate_image(self, image: Union[np.ndarray, torch.Tensor], strict: bool = True) -> None:
        """
        Validate an image's type, shape, and numerical integrity.

        Ensures that the input is a NumPy array or Torch tensor with acceptable dimensions,
        and contains no NaN or infinite values.

        Parameters
        ----------
        image : ndarray or Tensor
            Input image to validate.
        strict : bool, default True
            If True, raises exceptions on validation failure.
            If False, prints warnings instead (only if `self.verbose` is True).

        Raises
        ------
        TypeError
            If the input is not a NumPy array or Torch tensor (and strict=True).
        ValueError
            If the input has unsupported dimensions or contains NaN/Inf values (and strict=True).

        Notes
        -----
        - Supported dimensions: 2D to 5D (inclusive).
        - Valid dtypes include: uint8, int32, float32, float64.
        - Verbose mode reports value range for debugging.
        """
        if not isinstance(image, (np.ndarray, torch.Tensor)):
            msg = f"Invalid image type: {type(image)}"
            if strict:
                raise TypeError(msg)
            if self.verbose:
                print(f"[validate_image] Warning: {msg}")

        if hasattr(image, "ndim") and (image.ndim < 2 or image.ndim > 5):
            msg = f"Unsupported image dimensionality: {image.ndim}D"
            if strict:
                raise ValueError(msg)
            if self.verbose:
                print(f"[validate_image] Warning: {msg}")

        arr = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else image
        if np.isnan(arr).any() or np.isinf(arr).any():
            msg = "Image contains NaN or inf values"
            if strict:
                raise ValueError(msg)
            if self.verbose:
                print(f"[validate_image] Warning: {msg}")

        if arr.dtype in [np.uint8, np.int32, np.float32, np.float64]:
            minv, maxv = float(arr.min()), float(arr.max())
            if self.verbose:
                print(f"[validate_image] value range: [{minv:.4f}, {maxv:.4f}]")

 # ====[ GET SUPPORTED FORMATS – Extensions registry ]====
    def get_supported_formats(self) -> List[str]:
        """
        Return supported image file extensions.

        Returns
        -------
        list[str]
            Lowercase extensions (no dot).
        """
        return sorted(self._handlers.keys())

    # ====[ PREVIEW – Quick Visual Inspection ]====
    def preview(self, image: Union[np.ndarray, torch.Tensor, Any], title: Optional[str] = None, cmap: str = "gray") -> None:
        """
        Display an image using matplotlib with automatic format adjustment.

        Accepts NumPy arrays, Torch tensors, or AxisTracker-like objects
        and handles layout conversion for visualization safety.

        Parameters
        ----------
        image : ndarray | Tensor | AxisTracker-like
            Image to display. If an AxisTracker is provided, `.get()` will be called.
        title : str, optional
            Title to display above the image.
        cmap : str, default 'gray'
            Colormap used for grayscale display. Ignored for RGB images.

        Raises
        ------
        TypeError
            If input is not a supported image type.
        ValueError
            If the image has ambiguous or unsupported shape (e.g., non-displayable 3D tensor).

        Notes
        -----
        - CHW → HWC conversion is performed when needed (e.g., for RGB Tensor images).
        - 2D and RGB/alpha images are supported; full ND volumes are not.
        - Grayscale images with shape (H, W, 1) are automatically squeezed.
        """
        import matplotlib.pyplot as plt

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        elif hasattr(image, "get"):  # AxisTracker
            image = image.get()
            
        if not isinstance(image, np.ndarray):
            raise TypeError(f"[preview] Unsupported type: {type(image)}")            

        if image.ndim == 2:
            arr = image
        elif image.ndim == 3:
            # Handle CHW → HWC for preview safety
            if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
                arr = np.moveaxis(image, 0, -1)
            else:
                arr = image
            # sanity check for ambiguous 3D tensors (non-image volumes)
            if arr.shape[-1] not in (1, 3, 4) and arr.ndim == 3:
                raise ValueError(f"[preview] Ambiguous 3D shape: {image.shape}")
        else:
            raise ValueError(f"[preview] Unsupported image shape: {image.shape}")

        plt.figure(figsize=(6, 6))
        if title:
            plt.title(title)
        plt.axis("off")
        if arr.ndim == 2 or arr.shape[-1] == 1:
            plt.imshow(arr.squeeze(), cmap=cmap)
        else:
            plt.imshow(safe_image_to_uint8(arr))
        plt.show()

def safe_image_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Safely convert an image array to uint8 format for display or saving.

    Handles NaN/Inf values, various integer types, and floats outside the [0, 1] range
    by applying appropriate normalization or clipping strategies.

    Parameters
    ----------
    image : np.ndarray
        Input image (any dtype). Supported types: signed/unsigned integers, floats.

    Returns
    -------
    np.ndarray
        Converted image with dtype uint8, safe for visualization or export.

    Notes
    -----
    - NaN and infinite values are replaced with 0.
    - Integer types:
        * Signed integers: min–max normalized to [0, 255].
        * Unsigned integers:
            - If bit-depth > 8 (e.g., uint16), scale to 8-bit.
            - Otherwise, clip to [0, 255].
    - Float types:
        * If values are outside [0, 1], perform local min–max normalization.
        * Final values are clipped to [0, 1] before scaling to [0, 255].

    Examples
    --------
    >>> safe_image_to_uint8(np.array([[0.0, 1.0], [0.5, 0.75]]))
    array([[  0, 255],
           [127, 191]], dtype=uint8)
    """
    img = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Integer types
    if np.issubdtype(img.dtype, np.integer):
        if np.issubdtype(img.dtype, np.signedinteger):
            f = img.astype(np.float32, copy=True)
            mn, mx = float(f.min()), float(f.max())
            rng = max(mx - mn, 1e-12)
            f = (f - mn) / rng  # [0,1]
            return (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)
        else:
            info = np.iinfo(img.dtype)
            if info.max > 255:
                f = (img.astype(np.float32) / float(info.max)) * 255.0
                return np.clip(f, 0, 255).astype(np.uint8, copy=False)
            return np.clip(img, 0, 255).astype(np.uint8, copy=False)

    # Float-like
    f = img.astype(np.float32, copy=True)
    mn, mx = float(f.min()), float(f.max())
    if mx > 1.0 or mn < 0.0:
        rng = max(mx - mn, 1e-12)
        f = (f - mn) / rng
    return (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)
