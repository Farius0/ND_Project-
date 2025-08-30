# Author: Farius AINA

## General Conventions

- **Notation**
  - `CapWords` in backticks (`...`) → Classes
  - snake_case without backticks (...) → Functions

- **Import rules**
  - `core/`: does not import anything from other internal modules.
  - `utils/`: stateless helpers, must not import from project internals (only stdlib and external libs).
  - `operators/`: can import from `core/` and `utils/`.
  - `deep_learn/w_upbr/`: external reference code (adapted from Liu et al., 2025).

- **Backend policy**
  - **Dual-backend compatibility**: all operators support **NumPy** arrays and **PyTorch** tensors.
  - **Flexible switching**: inputs drive the backend (NumPy in → NumPy out; Torch in → Torch out) with consistent behavior.
  - **ND-ready design**: functions are **N-dimensional by design**, with first-class support for **2D** and **3D**; higher dimensions are generally supported when meaningful.
  - **Type/device/layout awareness**: operators preserve dtype and (for Torch) device when feasible; layout handling is centralized via `core/layout_axes.py`.
  - **Conversion layer**: backend interop is handled through the project’s base converters (`BaseConverter` / `OperatorCore`) to avoid user-side boilerplate.

- **Folder organization**
  - `core/`: abstract base classes and framework kernel.
  - `utils/`: stateless helpers (decorators, labels_tools, nd_tools, logger, emojis, …).
  - `operators/`: all image operators (2D/ND) built on `OperatorCore` or `BaseConverter`.
  - `filters/`: image filters (edge-aware, smoothing, etc.).
  - `algorithms/`: advanced algorithms (diffusion, denoising, etc.).
  - `degradations/`: degradation models (blur, noise, inpainting).
  - `datasets/`: dynamic datasets (PyTorch-compatible).
  - `deep_learn/`: deep learning (architectures, losses, metrics, training utilities).
  - `w_upbr/`: external reference code from *Liu et al. (2025)*, adapted for integration.
  - `tests/`: test scripts (experimental prototypes, dev validation, and user-level interaction).
  - `logs/`, `results/`, `fonts/`: runtime artifacts or resources (not critical).
  - `requirements/`: dependency specifications (`base.in`, `base.txt`, `torch-cpu.txt`, etc.).

---
SRC/
│
├── core/
│   ├── base_converter.py         # ↪ `BaseConverter`
│   ├── operator_core.py          # ↪ `OperatorCore`
│   ├── layout_axes.py            # ↪ `LayoutResolver`, get_layout_axes, resolve_and_clean_layout_tags
│   ├── tag_registry.py           # ↪ get_tag, set_tag, has_tag, del_tag
│   ├── config.py                 # ↪ `LayoutConfig`, `GlobalConfig`, `PreprocessorConfig`, etc.
│   └── __init__.py
│
├── utils/
│   ├── decorators.py             # ↪ `TimerManager`, timer, safe_timer_and_debug
│   ├── labels_tools.py           # ↪ scribble_labels, generate_minimal_scribble
│   ├── nd_tools.py               # ↪ colormap_picker, show_plane, search_opt_general
│   ├── logger.py                 # ↪ get_logger
│   ├── emojis.py                 # ↪ emoji handling functions
│   └── __init__.py
│
├── operators/
│   ├── image_io.py               # ↪ `ImageIO`
│   ├── image_processor.py        # ↪ `ImageProcessor`
│   ├── gaussian.py               # ↪ `NDConvolver`, `GaussianKernelGenerator`
│   ├── feature_extractor.py      # ↪ `FeatureExtractorND`
│   ├── preprocessor.py           # ↪ `PreprocessorND`
│   ├── resize_image.py           # ↪ `ResizeOperator`
│   ├── segmenter_nd.py           # ↪ `SegmenterND`
│   ├── transform_manager.py      # ↪ `TransformManager`
│   ├── diff_operator.py          # ↪ `DiffOperator`
│   ├── edge_detector.py          # ↪ `EdgeDetector`
│   ├── image_operator.py         # ↪ `Operator`, `DeepOperator`
│   ├── thresholding.py           # ↪ `ThresholdingOperator`
│   ├── axistracker.py            # ↪ `AxisTracker`
│   ├── metrics.py                # ↪ `MetricEvaluator` (psnr, mssim, fid, ...)
│   └── __init__.py
│
├── filters/
│   ├── artifact_cleaning.py      # ↪ `ArtifactCleanerND`
│   ├── edge_aware_filter.py      # ↪ `EdgeAwareFilter`
│   ├── perona_enhancing.py       # ↪ `PeronaEnhancer`
│   └── __init__.py
│
├── algorithms/
│   ├── perona_malik.py           # ↪ `PeronaMalikDenoiser`
│   └── __init__.py
│
├── degradations/
│   ├── blur.py                   # ↪ apply_spatial_blur, generate_gaussian_kernel_nd
│   ├── noise.py                  # ↪ apply_noise, random_noise_generator
│   ├── inpaint.py                # ↪ apply_inpaint, mask_generator
│   └── __init__.py
│
├── datasets/
│   ├── operator_dataset.py       # ↪ `BaseOperatorDataset`, `OperatorDataset`, `DeepOperatorDataset`
│   └── __init__.py
│
├── deep_learn/
│   ├── losses.py                 # ↪ `LossSelector`
│   ├── utils.py                  # ↪ `EarlyStopping`, plot_training
│   ├── FEUNET_v2.py              # ↪ `FEUNet_v2`
│   ├── feunet_config.yaml        # ↪ feunet_config
│   ├── wrt_cfg.py                # ↪ feunet_config
│   └── w_upbr/                   # External reference code (from *Liu et al., 2025*).
│       ├── FEUNet.py             # ↪ `FEUNet`
│       ├── GLCA.py               # ↪ `GLCA`
│       ├── Fusion.py             # ↪ `Fusion`
│       ├── data_augmentation.py  # ↪ `Compose`, build_transforms (unused)
│       ├── dataset.py            # ↪ `OCTDataset` (unused)
│       ├── logger.py             # ↪ `Logger` (unused)
│       ├── losses.py             # ↪ `pDLoss`, `DiceLoss` (unused)
│       ├── metrics.py            # ↪ get_error, dice_coefficient (unused)
│       ├── projectionHead.py     # ↪ `ProjectionHead` (unused)
│       ├── proto_loss.py         # ↪ `RDLCE_Loss`, `ADLCOM_Loss` (unused)
│       ├── ramps.py              # ↪ sigmoid_rampup, linear_rampup (unused)
│       ├── sinkhorn.py           # ↪ distributed_sinkhorn, distributed_greenkhorn (unused)
│       ├── util.py               # ↪ one_hot, map_to_contour (unused)
│
├── tests/
│   ├── experiments/              # exploratory or prototype tests
│   ├── test_dev/                 # operator-level validation during dev
│   └── tests_users_interact/     # integration and user-facing tests
│
├── logs/                         # Runtime logs
├── scripts/                      # Optional CLI or batch scripts
├── fonts/                        # Optional for styling
├── results/                      # Experimental results
└── requirements/                 # Dependency specifications
