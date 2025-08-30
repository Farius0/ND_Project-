# ==================================================
# ============= MODULE: operator_dataset ===========
# ==================================================
from __future__ import annotations

from pathlib import Path
from typing import Optional
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np, torch, os, random
from torchvision.transforms import InterpolationMode
from torch.utils.data._utils.collate import default_collate
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, List, Literal


from operators.preprocessor import PreprocessorND
from operators.image_operator import Operator, DeepOperator
from core.operator_core import OperatorCore
from core.config import (LayoutConfig, GlobalConfig, DatasetConfig, 
                         PreprocessorConfig, ImageProcessorConfig, TransformerConfig)

__all__ = [
    "BaseOperatorDataset",
    "OperatorDataset", 
    "DeepOperatorDataset",
]

# --- Local typing aliases ---
ArrayLike = Union[torch.Tensor, np.ndarray]
Framework = Literal["numpy", "torch"]
PathLike = Union[str, Path]
DataDict = Dict[str, Any]

# ==================================================
# =============== BaseOperatorDataset ==============
# ==================================================
class BaseOperatorDataset(Dataset, OperatorCore):
    """
    Base dataset for applying degradations (noise/blur/inpaint) or tasks (segmentation/classification).

    Notes
    -----
    - Dual-backend ready by design through OperatorCore/PreprocessorND configuration.
    - Expects images to be readable by `PreprocessorND`.
    - If `operator='segmentation'`, labels are read via a second `PreprocessorND` (`self.io`)
      with minimal processing (no normalization/equalization/etc.).
    """

    def __init__(
        self,
        images_dir: PathLike,
        images_files: Optional[Sequence[str]] = None,
        labels_dir: Optional[PathLike] = None,
        labels_files: Optional[Sequence[str]] = None,
        dataset_cfg: DatasetConfig = DatasetConfig(),
        preprocess_cfg: PreprocessorConfig = PreprocessorConfig(),
        img_process_cfg: ImageProcessorConfig = ImageProcessorConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
        dataset_cfg2: Optional[DatasetConfig] = None,
        size: Optional[Tuple[int, int]] = (256, 256),
    ) -> None:
        """
        Parameters
        ----------
        images_dir
            Directory containing input images or class subfolders (classification).
        images_files
            Filenames for images (ignored in classification mode where subfolders define classes).
        labels_dir
            Directory containing label images (segmentation).
        labels_files
            Filenames for labels (should match image names for segmentation).
        dataset_cfg
            Dataset behavior (operator type, noise/blur/mask levels, transform, etc.).
        preprocess_cfg
            Preprocessing config for images (truth).
        img_process_cfg
            Processor strategy/config used by `PreprocessorND`.
        layout_cfg
            Axes/layout handling.
        global_cfg
            Framework/output/device/normalization flags.
        dataset_cfg2
            Optional second config (e.g., label transform in segmentation).
        size
            Spatial resize target (H, W).
        """
        
        # ====[ Configuration ]====
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg
        self.dataset_cfg: DatasetConfig = dataset_cfg
        self.dataset_cfg2: Optional[DatasetConfig] = dataset_cfg2
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

        # Primary preprocessor (for truth images)
        self.preprocess: PreprocessorND = PreprocessorND(
            preprocess_cfg=self.preprocess_cfg,
            img_process_cfg=self.img_process_cfg,
            layout_cfg=self.layout_cfg,
            global_cfg=self.global_cfg,
        )
        
        # Secondary preprocessor (for labels): minimal ops, no normalization
        self.io: PreprocessorND = PreprocessorND(
            preprocess_cfg=self.preprocess_cfg.update_config(
                stretch=False,
                denoise=False,
                aggregate=False,
                remove_artifacts=False,
                local_contrast=False,
                gamma_correct=False,
                equalize=False,
            ),
            img_process_cfg=self.img_process_cfg,
            layout_cfg=self.layout_cfg,
            global_cfg=self.global_cfg.update_config(normalize=False),
        )
        
        # ====[ Store Dataset-specific parameters ]====
        
        # Dataset parameters (avoid mutable defaults)
        self.images_dir: str = os.fspath(images_dir)
        self.labels_dir: Optional[str] = os.fspath(labels_dir) if labels_dir is not None else None
        self.images_files: List[str] = list(images_files) if images_files is not None else []
        self.labels_files: List[str] = list(labels_files) if labels_files is not None else []

        # Core behavior flags/params from dataset_cfg
        self.operator: str = self.dataset_cfg.operator
        self.transform = self.dataset_cfg.transform  # for images
        self.transform_2 = self.dataset_cfg2.transform if self.dataset_cfg2 is not None else None  # for labels
        self.framework: Framework = self.global_cfg.framework
        self.output_format: str = self.global_cfg.output_format
        self.clip: bool = bool(self.dataset_cfg.clip)
        self.device: str = "cuda" if (torch.cuda.is_available() and self.framework == "torch") else self.global_cfg.device
        self.verbose: bool = bool(self.global_cfg.verbose)

        # Operator hyper-params
        self.mask: Optional[str] = self.dataset_cfg.mask
        self.noise_level: Union[Optional[float], Tuple[float, float]] = self.dataset_cfg.noise_level
        self.blur_level: Union[Optional[float], Tuple[float, float]] = self.dataset_cfg.blur_level
        self.mask_threshold: Optional[float] = self.dataset_cfg.mask_threshold
        self.random: Optional[bool] = self.dataset_cfg.random
        self.mode: Optional[str] = self.dataset_cfg.mode
        self.to_return: Optional[str] = self.dataset_cfg.to_return  # 'untransformed' | 'transformed' | 'both'
        self.return_param: Optional[bool] = self.dataset_cfg.return_param
        
        # ====[ Resize images ]====
        self.size: Union[int, Sequence[int]] = size
        if self.size is not None:
            self.resize_images = transforms.Resize(self.size, interpolation=InterpolationMode.BILINEAR)
            self.resize_labels = transforms.Resize(self.size, interpolation=InterpolationMode.NEAREST)
        else:
            self.resize_images = None
            self.resize_labels = None

        # Operator dispatch table            
        self.operators = {
            "noise": self._process_noise,
            "blur": self._process_blur,
            "paint": self._process_inpaint,
            "identity": self._identity,
            "segmentation": self._segmentation,
            "classification": self._classification
        }

        # --- Validations ---
        if self.operator not in self.operators:
            raise ValueError(f"Unsupported operator '{self.operator}'. Available: {list(self.operators.keys())}")

        if self.operator != "classification" and (not self.images_files):
            raise ValueError(f"Missing images files for operator '{self.operator}'.")
        
        if self.operator == "segmentation":
            if not self.labels_files or self.labels_dir is None:
                raise ValueError("Missing labels files and labels directory for operator 'segmentation'.")
            # Keep only pairs present on both sides (same filenames)
            valid_pair = [fn for fn in self.images_files if fn in self.labels_files]
            self.images_files = valid_pair
            self.labels_files = valid_pair
            
        if self.transform is None and self.to_return in ["transformed", "both"]:
            raise ValueError(
                f"Transform is required for 'transformed' or 'both' return types, but got None."
            )
        
        if self.operator == "classification" :
            self.labels: List[int] = []
            self.images_files = []
            for label, class_dir in enumerate(sorted(os.listdir(self.images_dir))):
                class_path = os.path.join(self.images_dir, class_dir)
                if not os.path.isdir(class_path):
                    continue
                for img_file in sorted(os.listdir(class_path)):
                    self.images_files.append(os.path.join(class_path, img_file))
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.images_files)

    def __getitem__(self, idx: int) -> DataDict:
        """
        Load one sample, apply optional resize/transform, then apply the selected operator.

        Returns
        -------
        dict
            For degradations: {"input": degraded, "truth": clean, ...}
            For segmentation: {"input": image, "truth": label}
            For classification: {"input": image, "truth": class_index}
            If `to_return == 'both'`, transformed outputs are duplicated with 't_' prefix.
        """        
        # --- Load truth image ---
        image_path = (
            os.path.join(self.images_dir, self.images_files[idx])
            if self.operator != "classification"
            else self.images_files[idx]
        )
        truth_image = self.preprocess(image_path)

        # Tag and (optional) verbose summary        
        tracker = self.preprocess.track(truth_image)
        tracker.update_tags({
            "status": "truth",
            "dataset": self.__class__.__name__,
            "operator": self.operator}, 
            add_new_axis=True)

        if self.verbose:
            print(f"[Dataset] Loaded truth image tag:")
            tracker.tag_summary() 
        
        # Optional resize
        if self.resize_images is not None:
            tracker = tracker.apply_to_all(self.resize_images)
            tracker.update_tags({"status": "resized", "shape_after": tracker.image.shape})
        
        truth_image = tracker.get()        
    
        # --- Load label (segmentation) --- 
        label_image: Optional[ArrayLike] = None       
        if self.operator == "segmentation":
            label_path = os.path.join(self.labels_dir, self.labels_files[idx]) if self.labels_files[idx] == self.images_files[idx]\
                else os.path.join(self.labels_dir, self.images_files[idx]) # assume labels_name == images_name
            label_image = self.io(label_path)

            label_tracker = self.io.track(label_image)
            label_tracker.update_tags({
                "status": "labels",
                "dataset": self.__class__.__name__,
                "operator": self.operator}, 
                add_new_axis=True)
            
            if self.verbose:
                print(f"[Dataset] Loaded label image tag:")
                label_tracker.tag_summary()
                
            if self.resize_labels is not None:
                label_tracker = label_tracker.apply_to_all(self.resize_labels)
                label_tracker.update_tags({"status": "resized", "shape_after": label_tracker.image.shape})
            
            label_image = label_tracker.get()
            
        # --- Classification label ---
        cls_label: Optional[int] = None
        if self.operator == "classification":
            cls_label = getattr(self, "labels", [None])[idx]

        # --- Apply operator on untransformed branch ---
        if self.operator in {"segmentation", "classification"}:
            data = self.operators[self.operator](truth_image, label_image if label_image is not None else cls_label)
        else:
            data = self.operators[self.operator](truth_image) 
            
        # Early return if no transform requested
        if self.to_return == "untransformed" or self.transform is None:
            return data              
                 
        # --- Synchronized transforms (image/label) ---
        seed = torch.randint(0, 2**32 - 1, ()).item()
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))
        random.seed(seed)

        if not self.return_param:
            transformed_image = self.transform(truth_image)
            transformed_image = tracker.clone_to(transformed_image, updates={"status": "transformed"}).get()
            transform_param = None
        else:
            transformed_image, transform_param = self.transform(truth_image)
            transformed_image = tracker.clone_to(transformed_image, updates={"status": "transformed"}).get()        

        if self.operator == "segmentation":
            # Re-apply the exact same seed before label transforms
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32 - 1))
            random.seed(seed)

            if self.transform_2 is None:
                raise ValueError("transform_2 is required for segmentation when `to_return != 'untransformed'`.")

            if not self.return_param:
                transform_label_image = self.transform_2(label_image)  # type: ignore[arg-type]
                transform_label_image = self.io.track(label_image).clone_to(  # type: ignore[arg-type]
                    transform_label_image,
                    updates={"status": "transformed"},
                ).get()
            else:
                transform_label_image, transform_label_param = self.transform_2(label_image)  # type: ignore[arg-type]
                transform_label_image = self.io.track(label_image).clone_to(  # type: ignore[arg-type]
                    transform_label_image,
                    updates={"status": "transformed"},
                ).get()
    
        # --- Apply operator on transformed branch ---
        if self.operator in {"segmentation", "classification"}:
            transform_data = self.operators[self.operator](
                transformed_image, transform_label_image if transform_label_image is not None else cls_label
            )
        else:
            transform_data = self.operators[self.operator](transformed_image)  
                
        # Attach transform params if requested (kept un-collated by safe_collate)
        if self.return_param:
            transform_data["params"] = transform_param
            # transform_data["pipeline"] = [self.transform]            
            if self.operator == "segmentation":
                transform_data["label_params"] = transform_label_param
                # transform_data["label_pipeline"] = [self.transform_2]  
                
        if self.to_return == "transformed":
            return transform_data
        elif self.to_return == "both":
            output: Dict[str, Any] = dict(data)
            for key, value in transform_data.items():
                output["t_" + key] = value
            return output
        else:
            return data
                
    # ---------------- Operator handlers ----------------
    def _identity(self, image: ArrayLike) -> DataDict:
        """Return input as both input and truth."""
        return {"input": image, "truth": image}

    def _segmentation(self, image: ArrayLike, label_image: ArrayLike) -> DataDict:
        """Segmentation task: returns image and its label image."""
        return {"input": image, "truth": label_image}

    def _classification(self, image: ArrayLike, label: int) -> DataDict:
        """Classification task: returns image and its class index."""
        return {"input": image, "truth": int(label)}

    # ---------------- Sampling helpers ----------------

    def sample_sigma(self, max_sigma: Union[float, Tuple[float, float]]) -> Union[float, Tuple[float, float]]:
        """
        Sample 'sigma' in [0, max] if random, else return max.
        Supports scalar or tuple(sigmax, sigmay) for anisotropic blur.
        """
        if isinstance(max_sigma, tuple):
            if not self.random:
                return max_sigma
            lo_x = 0.5 if max_sigma[0] > 0.5 else 0.0
            lo_y = 0.5 if max_sigma[1] > 0.5 else 0.0
            return (float(np.random.uniform(lo_x, float(max_sigma[0]))),
                    float(np.random.uniform(lo_y, float(max_sigma[1]))))
        return float(np.random.uniform(0.0, max_sigma)) if self.random else float(max_sigma)

    def sample_threshold(self, max_threshold: float) -> float:
        """Sample a threshold in [0, max] if random, else return max."""
        return float(np.random.uniform(0.0, max_threshold)) if self.random else float(max_threshold)

    # ---------------- Diagnostics ----------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  Total images   : {len(self.images_files)}\n"
            f"  Operator       : {self.operator}\n"
            f"  Framework      : {self.framework}\n"
            f"  Device         : {self.device}\n"
            f")"
        )

    def summary(self) -> None:
        """Print a concise dataset configuration summary."""
        print("====[ Dataset Summary ]====")
        print(f"Images loaded from : {self.images_dir}")
        print(f"Number of images   : {len(self.images_files)}")
        print(f"Operator type      : {self.operator}")
        print(f"Framework          : {self.framework}")
        print(f"Device             : {self.device}")
        print(f"Clip output        : {self.clip}")
        print(f"Noise Level        : {self.noise_level}")
        print(f"Blur Level         : {self.blur_level}")
        print(f"Mask Threshold     : {self.mask_threshold}")
        print(f"Transform provided : {self.transform is not None}")

    def summary(self):
        """
        Print a detailed configuration of the dataset.
        """
        print("====[ Dataset Summary ]====")
        print(f"Images loaded from : {self.images_dir}")
        print(f"Number of images   : {len(self.images_files)}")
        print(f"Operator type      : {self.operator}")
        print(f"Framework          : {self.framework}")
        print(f"Device             : {self.device}")
        print(f"Clip output        : {self.clip}")
        print(f"Noise Level        : {self.noise_level}")
        print(f"Blur Level         : {self.blur_level}")
        print(f"Mask Threshold     : {self.mask_threshold}")
        print(f"Transform provided : {self.transform is not None}")
        print("ND Axes:")
        for k, v in self.io.axes.items():
            print(f"  {k:<16}: {v}")

# ==================================================
# =============== OperatorDataset ==================
# ==================================================
class OperatorDataset(BaseOperatorDataset):
    """Degradations with classic `Operator` (noise/blur/inpaint)."""    
    def _process_noise(self, image: ArrayLike) -> DataDict:
        """
        Apply Gaussian noise to the image using Operator.

        Parameters
        ----------
        image : torch.Tensor
            Clean image.

        Returns
        -------
        dict
            {"input": noised_image, "truth": original}
        """
        sigma = self.sample_sigma(self.noise_level)
        operator = Operator(image=image, clip=self.clip, layout_cfg=self.layout_cfg, global_cfg=self.global_cfg)
        degraded = operator.noise(sigma=sigma)
        return {"input": degraded, "truth": image}

    def _process_blur(self, image: ArrayLike) -> DataDict:
        """
        Apply spatial blur to the image using Operator.

        Parameters
        ----------
        image : torch.Tensor
            Clean image.

        Returns
        -------
        dict
            {"input": blurred_image, "truth": original, "kernel": kernel_tensor}
        """
        sigma = self.sample_sigma(self.blur_level)
        operator = Operator(image=image, clip=self.clip, layout_cfg=self.layout_cfg, global_cfg=self.global_cfg)
        blurry, kernel = operator.blur(sigma=sigma, return_kernel=True)
        return {"input": blurry, "truth": image, "kernel": kernel}


    def _process_inpaint(self, image: ArrayLike) -> DataDict:
        """
        Apply inpainting to the image using Operator.

        Parameters
        ----------
        image : torch.Tensor
            Clean image.

        Returns
        -------
        dict
            {"input": inpainted_image, "truth": original, "mask": applied_mask}
        """
        threshold = self.sample_threshold(self.mask_threshold)
        operator = Operator(image=image, clip=self.clip, layout_cfg=self.layout_cfg, global_cfg=self.global_cfg)
        inpainted, mask = operator.inpaint(
            mask=self.mask, sigma=self.noise_level, threshold=threshold, mode=self.mode, return_mask=True
        )
        return {"input": inpainted, "truth": image, "mask": mask}

# ==================================================
# ============== DeepOperatorDataset ===============
# ==================================================
class DeepOperatorDataset(BaseOperatorDataset):
    """Degradations with `DeepOperator` (deep inverse operators)."""   
     
    def _process_noise(self, image: ArrayLike) -> DataDict:
        """
        Apply DeepInv Gaussian noise operator.

        Parameters
        ----------
        image : torch.Tensor
            Clean image.

        Returns
        -------
        dict
            {"input": noised_image, "truth": original}
        """
        sigma = self.sample_sigma(self.noise_level)
        operator = DeepOperator(image=image, clip=self.clip, layout_cfg=self.layout_cfg, global_cfg=self.global_cfg)
        noised = operator.noise(sigma=sigma)
        return {"input": noised, "truth": image}
    
    def _process_blur(self, image: ArrayLike) -> DataDict:
        """
        Apply a DeepInv Gaussian blur with random or fixed sigma.

        Parameters
        ----------
        image : torch.Tensor
            Clean image.

        Returns
        -------
        dict
            {"input": blurred_image, "truth": original, "kernel": kernel_tensor}
        """
        sigma = self.sample_sigma(self.blur_level)
        operator = DeepOperator(image=image, clip=self.clip, layout_cfg=self.layout_cfg, global_cfg=self.global_cfg)
        blurry, kernel = operator.blur(sigma=sigma, return_kernel=True)
        return {"input": blurry, "truth": image, "kernel": kernel}

    def _process_inpaint(self, image: ArrayLike) -> DataDict:
        """
        Apply DeepInv inpainting with optional mask and random threshold.

        Parameters
        ----------
        image : torch.Tensor
            Clean image.

        Returns
        -------
        dict
            {"input": inpainted_image, "truth": original, "mask": applied_mask}
        """
        sigma = self.sample_sigma(self.noise_level)
        threshold = self.sample_threshold(self.mask_threshold)
        operator = DeepOperator(image=image, clip=self.clip, layout_cfg=self.layout_cfg, global_cfg=self.global_cfg)
        inpainted, mask = operator.inpaint(mask=self.mask, sigma=sigma, threshold=threshold, return_mask=True)
        return {"input": inpainted, "truth": image, "mask": mask}


# ==================================================
# ================== build_dataset =================
# ==================================================

def build_dataset(
    dir_path: PathLike,
    images_names: Optional[Sequence[str]],
    labels_dir: Optional[PathLike] = None,
    labels_names: Optional[Sequence[str]] = None,
    operator: str = "paint",
    noise_level: Union[float, Tuple[float, float]] = 0.8,
    blur_level: Union[float, Tuple[float, float]] = 5.0,
    mask_threshold: float = 0.7,
    mask_mode: str = "grid_noised",
    layout_framework: Framework = "numpy",
    layout_name: str = "HWC",
    layout_ensured_name: str = "NCHW",
    add_batch_dim: bool = True,
    add_channel_dim: Optional[bool] = None,
    use_transforms: bool = True,
    stretch: bool = False,
    denoise: bool = False,
    aggregate: bool = False,
    remove_artifacts: bool = False,
    local_contrast: bool = False,
    gamma_correct: bool = False,
    equalize: bool = False,
    size: Tuple[int, int] = (299, 299),
    horizontal_flip: Optional[float] = None,
    vertical_flip: Optional[float] = None,
    rotation: Optional[float] = None,
    brightness: Optional[float] = None,
    contrast: Optional[float] = None,
    saturation: Optional[float] = None,
    to_return: str = "untransformed",
    return_param: bool = False,
    return_transform: bool = False,
) -> Union[OperatorDataset, Tuple[OperatorDataset, Any]]:
    """
    Build an `OperatorDataset` with preprocessing and augmentation settings.

    Parameters
    ----------
    dir_path : PathLike
        Root directory containing the input images.
    images_names : Sequence of str
        List of image filenames to include in the dataset.
    labels_dir : PathLike, optional
        Directory containing ground truth or label images (if any).
    labels_names : Sequence of str, optional
        List of label filenames (must align with `images_names`).
    operator : str, default "paint"
        Operator applied to each sample ("paint", "blur", "noise", etc.).
    noise_level : float or tuple, default 0.8
        Intensity or range of noise to apply to input images.
    blur_level : float or tuple, default 5.0
        Intensity or range of blur to apply.
    mask_threshold : float, default 0.7
        Threshold used when creating binary masks from degraded images.
    mask_mode : str, default "grid_noised"
        Strategy used to build degradation masks.
    layout_framework : {"numpy", "torch"}, default "numpy"
        Backend framework assumed for layout resolution.
    layout_name : str, default "HWC"
        Initial layout of input images.
    layout_ensured_name : str, default "NCHW"
        Layout to enforce before applying transformations.
    add_batch_dim : bool, default True
        Whether to add a batch dimension automatically.
    add_channel_dim : bool or None, optional
        Whether to add a channel dimension (if not already present).
    use_transforms : bool, default True
        Enable torchvision-style data augmentations.
    stretch : bool, default False
        Apply histogram stretching to inputs.
    denoise : bool, default False
        Enable denoising pre-processing step.
    aggregate : bool, default False
        Apply block-wise aggregation (e.g., mean filtering).
    remove_artifacts : bool, default False
        Apply artifact removal heuristics.
    local_contrast : bool, default False
        Enhance local contrast of the input images.
    gamma_correct : bool, default False
        Apply gamma correction to enhance visibility.
    equalize : bool, default False
        Apply histogram equalization.
    size : Tuple[int, int], default (299, 299)
        Resize target for all images (H, W).
    horizontal_flip : float, optional
        Probability of random horizontal flip.
    vertical_flip : float, optional
        Probability of random vertical flip.
    rotation : float, optional
        Maximum rotation angle (in degrees).
    brightness : float, optional
        Brightness adjustment factor.
    contrast : float, optional
        Contrast adjustment factor.
    saturation : float, optional
        Saturation adjustment factor.
    to_return : {"untransformed", "transformed", "both"}, default "untransformed"
        Controls what to return from the dataset: raw, augmented, or both.
    return_param : bool, default False
        If True, return parameters used for each transformation.
    return_transform : bool, default False
        If True, also return the transform pipeline used.

    Returns
    -------
    OperatorDataset or (OperatorDataset, transform)
        Dataset object, with optional return of the transform pipeline.
    """
    # Apply transforms only if returning transformed data
    use_transforms = bool(use_transforms and to_return != "untransformed")
    
    # Build transforms (images + labels for segmentation)
    transform = None
    transform_2 = None
    if use_transforms:
        base_params = {
            "size": size,
            "horizontal_flip": horizontal_flip,
            "vertical_flip": vertical_flip,
            "rotation": rotation,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "use_transform": True,
            "return_param": return_param,
        }
        transform = TransformerConfig(**base_params).build_transform()

        if operator == "segmentation":
            label_params = dict(base_params)
            label_params.update({"is_label": True})
            transform_2 = TransformerConfig(**label_params).build_transform()
        
    # ====[ Configuration ]====
    data_set_params = {"operator": operator, "noise_level": noise_level, "blur_level": blur_level, "transform": transform,
                       "mask_threshold": mask_threshold, "mode": mask_mode, "to_return": to_return, "return_param": return_param,}
    
    if operator == "segmentation" and use_transforms:
        data_set_params_2 = dict(data_set_params)
        data_set_params_2.update({"transform": transform_2})
    else:
        data_set_params_2 = {}
        
    proc_params = {"processor_strategy": "torch",}
    layout_params = {"layout_name": layout_name, "layout_framework": layout_framework, "layout_ensured_name":layout_ensured_name,}
    global_params = {"framework": "torch", "output_format": "torch", "add_batch_dim":add_batch_dim, "add_channel_dim": add_channel_dim, "normalize": True}   
    preprocess_params = {"clip": True, "stretch": stretch, "denoise": denoise, "aggregate": aggregate, "remove_artifacts": remove_artifacts, 
                         "local_contrast": local_contrast, "gamma_correct": gamma_correct, "equalize": equalize}
    
    train_dataset = OperatorDataset(images_dir=dir_path, 
                                    images_files=images_names,
                                    labels_dir=labels_dir,
                                    labels_files=labels_names,
                                    dataset_cfg=DatasetConfig(**data_set_params),
                                    preprocess_cfg=PreprocessorConfig(**preprocess_params),
                                    img_process_cfg=ImageProcessorConfig(**proc_params),
                                    layout_cfg=LayoutConfig(**layout_params),
                                    global_cfg=GlobalConfig(**global_params),
                                    dataset_cfg2=DatasetConfig(**data_set_params_2),
                                    size=size
                                    )
    
    if return_transform:
        return train_dataset, transform
    else:
        return train_dataset
    
# ==================================================
# ================== safe_collate ==================
# ==================================================
def safe_collate(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for DataLoader that preserves metadata parameters.

    Specifically, keys like 'params', 'label_params', and any key ending with '_params'
    are preserved as-is across the batch.

    Parameters
    ----------
    batch : Sequence[Mapping[str, Any]]
        A list of dictionaries representing individual samples.

    Returns
    -------
    Dict[str, Any]
        A dictionary where each key maps to a batched value.
        Special handling is applied to keys ending with '_params' to keep them unstacked.
    """
    base = default_collate(
        [
            {
                k: v
                for k, v in b.items()
                if not (k == "params" or k == "label_params" or k.endswith("_params"))
            }
            for b in batch
        ]
    )
    # Preserve param dictionaries as lists (un-collated)
    for k in batch[0].keys():
        if k == "params" or k == "label_params" or k.endswith("_params"):
            base[k] = [b.get(k) for b in batch]
    return base
