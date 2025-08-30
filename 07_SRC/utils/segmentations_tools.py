# ==================================================
# ==========  MODULE: segmentations_tools  ========
# ==================================================
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from statsmodels.robust.scale import mad
from skimage.measure import find_contours
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import confusion_matrix
from skimage.morphology import (
    rectangle, disk, octagon,
    dilation, closing, erosion, remove_small_objects, binary_closing
)

ArrayNP = np.ndarray


def slice_sample(
    data: ArrayNP,
    center: bool = True,
    z: Optional[int] = None,
    y: Optional[int] = None,
    x: Optional[int] = None,
    return_axes: Sequence[str] = ("axial", "coronal", "sagittal"),
) -> Dict[str, ArrayNP]:
    """
    Extract orthogonal slices (axial, coronal, sagittal) from a 3D volume.

    Parameters
    ----------
    data : ArrayNP
        3D image volume with shape (Z, Y, X).
    center : bool, optional
        If True, extract slices at the center of the volume (overrides z, y, x).
    z : int, optional
        Index of the slice along the Z-axis (axial view).
    y : int, optional
        Index of the slice along the Y-axis (coronal view).
    x : int, optional
        Index of the slice along the X-axis (sagittal view).
    return_axes : Sequence[str], optional
        List of views to return: any combination of "axial", "coronal", and "sagittal".

    Returns
    -------
    Dict[str, ArrayNP]
        Dictionary containing the requested 2D slices:
        {
            "axial":    slice along Z-axis (Y, X),
            "coronal":  slice along Y-axis (Z, X),
            "sagittal": slice along X-axis (Z, Y)
        }
    """
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError("`data` must be a 3D numpy array shaped (Z, Y, X).")

    Z, Y, X = data.shape
    if center:
        z = Z // 2
        y = Y // 2
        x = X // 2
    else:
        z = z if z is not None else Z // 2
        y = y if y is not None else Y // 2
        x = x if x is not None else X // 2

    # clamp within bounds
    z = int(np.clip(z, 0, Z - 1))
    y = int(np.clip(y, 0, Y - 1))
    x = int(np.clip(x, 0, X - 1))

    result: Dict[str, ArrayNP] = {}
    if "axial" in return_axes:
        result["axial"] = data[z, :, :]
    if "coronal" in return_axes:
        result["coronal"] = data[:, y, :]
    if "sagittal" in return_axes:
        result["sagittal"] = data[:, :, x]
    return result

# =========================================================
# Segmentation metrics
# =========================================================
def evaluate_segmentation_nd(
    reference: ArrayNP,
    prediction: ArrayNP,
    labels: Optional[Sequence[int]] = None,
    average: str = "macro",
) -> Dict[str, Any]:
    """
    Evaluate segmentation performance with common metrics.

    Parameters
    ----------
    reference : np.ndarray
        Ground-truth segmentation map (binary or multi-class).
    prediction : np.ndarray
        Predicted segmentation map (same shape as reference).
    labels : list[int], optional
        List of labels to evaluate. If None, inferred from reference.
    average : str
        Type of averaging to use for multi-class: 'macro', 'micro', or 'weighted'.

    Returns
    -------
    metrics : dict
        Dictionary with metrics: accuracy, per-class IoU, Dice, precision, recall.
    """
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction must have the same shape.")
    y_true = reference.flatten()
    y_pred = prediction.flatten()

    # labels set
    if labels is None:
        labels = list(np.unique(np.concatenate([y_true, y_pred])))
    if len(labels) == 0:
        raise ValueError("No labels found for evaluation.")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fp = np.sum(cm, axis=0).astype(float) - tp
    fn = np.sum(cm, axis=1).astype(float) - tp
    tn = float(np.sum(cm)) - (tp + fp + fn)

    eps = 1e-8
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = float(np.sum(tp) / np.sum(cm)) if np.sum(cm) > 0 else np.nan

    # Aggregates
    support = np.sum(cm, axis=1).astype(float)  # per-class true count
    weights = support / (np.sum(support) + eps)
    macro = {
        "iou": float(np.nanmean(iou)),
        "dice": float(np.nanmean(dice)),
        "precision": float(np.nanmean(precision)),
        "recall": float(np.nanmean(recall)),
    }
    weighted = {
        "iou": float(np.nansum(iou * weights)),
        "dice": float(np.nansum(dice * weights)),
        "precision": float(np.nansum(precision * weights)),
        "recall": float(np.nansum(recall * weights)),
    }
    # micro average (global TP/FP/FN)
    TPg = float(np.sum(tp))
    FPg = float(np.sum(fp))
    FNg = float(np.sum(fn))
    iou_micro = TPg / (TPg + FPg + FNg + eps)
    dice_micro = 2 * TPg / (2 * TPg + FPg + FNg + eps)
    prec_micro = TPg / (TPg + FPg + eps)
    rec_micro = TPg / (TPg + FNg + eps)
    micro = {
        "iou": iou_micro,
        "dice": dice_micro,
        "precision": prec_micro,
        "recall": rec_micro,
    }

    metrics = {
        "accuracy": accuracy,
        "per_class": {
            "labels": list(labels),
            "iou": dict(zip(labels, map(float, iou))),
            "dice": dict(zip(labels, map(float, dice))),
            "precision": dict(zip(labels, map(float, precision))),
            "recall": dict(zip(labels, map(float, recall))),
        },
        "macro": macro,
        "micro": micro,
        "weighted": weighted,
        "average_selected": {"macro": macro, "micro": micro, "weighted": weighted}.get(average, macro),
        "confusion_matrix": cm,
    }
    return metrics

# =========================================================
# Thick contours & morphology
# =========================================================

def return_thick_contours(
    img: ArrayNP,
    is_measure: bool = False,
    alternate: bool = False,
) -> Union[ArrayNP, Tuple[ArrayNP, List[ArrayNP]]]:
    """
    Return a binary mask of thick contours, optionally with separated contours for measurement.

    Parameters
    ----------
    img : ArrayNP
        Input 2D image from which contours are extracted.
    is_measure : bool, optional
        If True, also return the individual contours for further measurement.
    alternate : bool, optional
        If True, apply an alternative contour extraction strategy.

    Returns
    -------
    mask : ArrayNP
        Binary mask with thick contours highlighted.
    contours : List[ArrayNP], optional
        List of individual contour arrays (only if is_measure is True).
    """

    if not isinstance(img, np.ndarray) or img.ndim != 2:
        raise ValueError("`img` must be a 2D numpy array.")
    
    mask = (img == 1)

    # crop vertical simple
    mean_ax1 = mask.mean(axis=1)
    thresh = int(np.argmax(mean_ax1)) + 30
    mask[thresh:, :] = False
    
    # Morphology 
    if not alternate:
        fp = rectangle(3, 10)
        mask = dilation(mask, footprint=fp)
        mask = closing(mask, footprint=fp)
        mask = erosion(mask, footprint=fp)
        mask = remove_small_objects(mask, 300)
        mask = dilation(mask, footprint=disk(3))
        mask = closing(mask, footprint=disk(10))
    else:
        mask = erosion(mask, footprint=octagon(0, 1))
        mask = remove_small_objects(mask, 100)
        mask = dilation(mask, footprint=disk(3))
        mask = binary_closing(mask, footprint=disk(10))

    if not is_measure:
        return mask.astype(bool)
    contours = find_contours(mask.astype(float))
    return mask.astype(bool), contours

# =========================================================
# Robust masks (1D arrays)
# =========================================================
def mask_quantile(values: ArrayNP, quantiles: Tuple[float, float] = (0.05, 0.95)) -> ArrayNP:
    """
    Create a boolean mask to filter values outside a given quantile range.

    Parameters
    ----------
    values : ArrayNP
        1D array of numerical values to filter.
    quantiles : Tuple[float, float], optional
        Lower and upper quantile thresholds (default: (0.05, 0.95)).

    Returns
    -------
    ArrayNP
        Boolean mask where True indicates values within the specified quantile range.
    """

    vals = np.asarray(values, dtype=float)
    mask_valid = ~np.isnan(vals)
    if not mask_valid.any():
        return np.zeros_like(vals, dtype=bool)
    lo = np.quantile(vals[mask_valid], quantiles[0])
    hi = np.quantile(vals[mask_valid], quantiles[1])
    return (vals >= lo) & (vals <= hi)

def mask_iqr(values: ArrayNP, k: float = 1.5) -> ArrayNP:
    """
    Create a boolean mask to filter values outside the interquartile range (IQR).

    Parameters
    ----------
    values : ArrayNP
        1D array of numerical values to filter.
    k : float, optional
        Scaling factor for the IQR. A value is considered an outlier if it lies
        below Q1 - k*IQR or above Q3 + k*IQR. Default is 1.5.

    Returns
    -------
    ArrayNP
        Boolean mask where True indicates values within the IQR-based threshold.
    """

    vals = np.asarray(values, dtype=float)
    mask_valid = ~np.isnan(vals)
    if not mask_valid.any():
        return np.zeros_like(vals, dtype=bool)
    q1 = np.percentile(vals[mask_valid], 25)
    q3 = np.percentile(vals[mask_valid], 75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (vals >= lo) & (vals <= hi)

def mask_zscore(values: ArrayNP, z_thresh: float = 2.5) -> ArrayNP:
    """
    Create a boolean mask to filter out values based on z-score thresholding.

    Parameters
    ----------
    values : ArrayNP
        1D array of numerical values to evaluate.
    z_thresh : float, optional
        Z-score threshold. Values with absolute z-scores above this threshold
        are considered outliers. Default is 2.5.

    Returns
    -------
    ArrayNP
        Boolean mask where True indicates values within the z-score threshold.
    """

    vals = np.asarray(values, dtype=float)
    mask_valid = ~np.isnan(vals)
    z = np.full_like(vals, np.nan, dtype=float)
    if mask_valid.any():
        z[mask_valid] = zscore(vals[mask_valid])
    return np.abs(z) <= z_thresh

def mask_mad(values: ArrayNP, window: int = 5, thresh: float = 3.0) -> ArrayNP:
    """
    Create a boolean mask using a rolling Median Absolute Deviation (MAD) filter.

    Parameters
    ----------
    values : ArrayNP
        1D array of numerical values to filter.
    window : int, optional
        Size of the rolling window to compute the local median and MAD. Default is 5.
    thresh : float, optional
        Threshold multiplier. Values deviating more than (thresh × MAD) from the
        local median are considered outliers. Default is 3.0.

    Returns
    -------
    ArrayNP
        Boolean mask where True indicates values within the MAD-based threshold.
    """

    vals = np.asarray(values, dtype=float)
    n = len(vals)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if np.isnan(vals[i]):
            mask[i] = False
            continue
        s = max(0, i - window)
        e = min(n, i + window + 1)
        local = vals[s:e]
        local = local[~np.isnan(local)]
        if local.size == 0:
            mask[i] = False
            continue
        med = float(np.median(local))
        local_mad = float(mad(local, center=med))
        if local_mad == 0.0:
            local_mad = 1e-6
        mask[i] = (abs(vals[i] - med) / local_mad) <= thresh
    return mask

def mask_minmax(values: ArrayNP, min_value: Optional[float] = None, max_value: Optional[float] = None) -> ArrayNP:
    """
    Create a boolean mask by filtering values outside a specified min-max range.

    Parameters
    ----------
    values : ArrayNP
        1D array of numerical values to filter.
    min_value : float or None, optional
        Minimum acceptable value. If None, no lower bound is applied.
    max_value : float or None, optional
        Maximum acceptable value. If None, no upper bound is applied.

    Returns
    -------
    ArrayNP
        Boolean mask where True indicates values within the specified range.
    """

    vals = np.asarray(values, dtype=float)
    mask = ~np.isnan(vals)
    if min_value is not None:
        mask &= vals >= min_value
    if max_value is not None:
        mask &= vals <= max_value
    return mask

MASK_REGISTRY: Dict[str, Callable[..., ArrayNP]] = {
    "quantile": mask_quantile,
    "iqr": mask_iqr,
    "zscore": mask_zscore,
    "mad": mask_mad,
    "minmax": mask_minmax,
}
# === Ajout d'un parseur de steps robuste ===
def parse_steps(steps: Sequence[Union[str, Sequence[Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Parse a sequence of step definitions into standardized (name, params) tuples.

    Parameters
    ----------
    steps : Sequence of str or tuple
        List of filtering steps. Each element can be:
        - A string (e.g., "iqr") for default parameters,
        - A tuple/list (e.g., ("iqr", {"k": 1.5})) to specify custom parameters.

    Returns
    -------
    List[Tuple[str, Dict[str, Any]]]
        List of step names and corresponding parameter dictionaries.
    """

    parsed: List[Tuple[str, Dict[str, Any]]] = []
    for step in steps:
        if isinstance(step, str):
            parsed.append((step, {}))
        elif isinstance(step, (list, tuple)):
            if len(step) == 1:
                parsed.append((step[0], {}))
            elif len(step) == 2 and isinstance(step[0], str) and isinstance(step[1], dict):
                parsed.append((step[0], step[1]))
            else:
                raise ValueError(f"Invalid step format: {step!r}. Expected ('name', {{}}]).")
        else:
            raise ValueError(f"Invalid step type: {type(step).__name__}")
    return parsed

def combine_masks(values: ArrayNP, steps: Sequence[Union[str, Sequence[Any]]], logic: str = "and") -> ArrayNP:
    """
    Combine multiple filtering masks using logical operations.

    Parameters
    ----------
    values : ArrayNP
        1D array of numerical values to evaluate.
    steps : Sequence of str or tuple
        List of filtering steps to apply. Each step can be a string (e.g., "iqr")
        or a tuple of (step_name, parameters_dict).
    logic : str, optional
        Logic used to combine the masks: "and" (intersection) or "or" (union).
        Default is "and".

    Returns
    -------
    ArrayNP
        Boolean mask resulting from the combination of individual masks.
    """

    steps_p = parse_steps(steps)
    masks = []
    for name, kwargs in steps_p:
        if name not in MASK_REGISTRY:
            raise ValueError(f"Unknown method: {name}")
        masks.append(MASK_REGISTRY[name](values, **kwargs))
    if not masks:
        return np.ones_like(values, dtype=bool)
    if logic == "and":
        return np.logical_and.reduce(masks)
    if logic == "or":
        return np.logical_or.reduce(masks)
    raise ValueError(f"Unknown logic: {logic!r}")

def sequential_mask(values: ArrayNP, steps: Sequence[Union[str, Sequence[Any]]]) -> ArrayNP:
    """
    Apply multiple filtering steps sequentially to refine a boolean mask.

    Parameters
    ----------
    values : ArrayNP
        1D array of numerical values to filter.
    steps : Sequence of str or tuple
        List of filtering steps to apply in order.
        Each step can be a string (e.g., "iqr") or a tuple (name, parameters_dict).

    Returns
    -------
    ArrayNP
        Final boolean mask after sequential application of all filtering steps.
    """

    steps_p = parse_steps(steps)
    mask = ~np.isnan(values)
    for name, kwargs in steps_p:
        mask &= MASK_REGISTRY[name](values, **kwargs)
    return mask

# =========================================================
# Outlier filtering & thickness estimation
# =========================================================
def filter_outliers_robust(
    thickness_map: Union[Dict[int, float], ArrayNP],
    steps: Sequence[Union[str, Sequence[Any]]] = ("quantile", "iqr"),
    mode: str = "sequential",  # "sequential" or "combine"
    logic: str = "and",         # used if mode == "combine"
    return_mask: bool = False,
) -> Union[Dict[int, float], Tuple[Dict[int, float], ArrayNP]]:
    """
    Filter outliers in a thickness profile using robust statistical methods.

    Parameters
    ----------
    thickness_map : dict or ArrayNP
        Thickness profile as a dictionary (index → value) or 1D NumPy array.
    steps : Sequence[str or Sequence], optional
        List of filtering steps to apply. Supported options include:
        - "quantile" (removes values outside typical quantile range),
        - "iqr" (removes values based on interquartile range).
        Each step can be a string or a (name, parameters) tuple.
    mode : str, optional
        Filtering strategy:
        - "sequential": apply steps one after the other (default),
        - "combine": apply all filters at once using the specified logic.
    logic : str, optional
        Logical rule for combining filters in "combine" mode ("and" or "or").
    return_mask : bool, optional
        If True, return a boolean mask indicating inliers.

    Returns
    -------
    filtered : dict
        Thickness profile with outliers removed.
    mask : np.ndarray, optional
        Boolean mask indicating valid values (only returned if return_mask=True).
    """

    steps_p = parse_steps(steps)

    if isinstance(thickness_map, dict):
        cols = np.array(sorted(thickness_map), dtype=int)
        values = np.array([thickness_map[c] for c in cols], dtype=float)
    else:
        arr = np.asarray(thickness_map, dtype=float)
        cols = np.arange(arr.size, dtype=int)
        values = arr

    if mode == "sequential":
        mask = sequential_mask(values, steps_p)
    elif mode == "combine":
        mask = combine_masks(values, steps_p, logic=logic)
    else:
        raise ValueError(f"Invalid mode: {mode!r}")

    filtered = {int(c): float(v) for c, v, keep in zip(cols, values, mask) if keep}
    if return_mask:
        return filtered, mask
    return filtered


def estimate_thickness_from_all_points(
    contours: Sequence[ArrayNP],
    min_points_per_column: int = 2,
    min_distance: float = 1.0,
    smooth: bool = False,
    kernel_size: int = 5,
    metrics: Optional[Mapping[str, Callable[[ArrayNP], float]]] = None,
    min_length: int = 30,
    use_robust_filter: bool = False,
    robust_filter_params: Optional[Mapping[str, Any]] = {"steps": ["quantile", "iqr"]},
    default_value: float = 0.0,
    shape_length: Optional[int] = None,
) -> Tuple[Dict[int, float], Dict[str, float]]:
    """
    Estimate local and global thickness metrics from two aligned contours.

    Parameters
    ----------
    contours : Sequence[ArrayNP]
        List of two aligned 2D arrays representing upper and lower contours
        with shape (N, 2) or (H, W).
    min_points_per_column : int, optional
        Minimum number of valid points required per column to compute thickness.
    min_distance : float, optional
        Minimum allowed vertical distance between paired points.
    smooth : bool, optional
        If True, apply 1D smoothing to the thickness profile.
    kernel_size : int, optional
        Size of the smoothing kernel (must be odd).
    metrics : Mapping[str, Callable], optional
        Dictionary of metric names mapped to functions that compute global
        statistics from the thickness profile (e.g., mean, std).
    min_length : int, optional
        Minimum number of valid columns required to compute global metrics.
    use_robust_filter : bool, optional
        If True, apply a robust filter to remove outliers in the thickness profile.
    robust_filter_params : Mapping[str, Any], optional
        Parameters for the robust filtering strategy (e.g., quantile thresholds, steps).
    default_value : float, optional
        Value assigned to positions where thickness cannot be computed.
    shape_length : int, optional
        Optional fixed length to enforce on the output profile (used for alignment).

    Returns
    -------
    Tuple[Dict[int, float], Dict[str, float]]
        - Local thickness profile as a dictionary: {column_index: thickness_value}
        - Global metrics as a dictionary: {metric_name: value}
    """


    valid_contours = [c for c in contours if isinstance(c, np.ndarray) and c.ndim == 2 and c.shape[1] >= 2 and len(c) >= min_length]
    if not valid_contours:
        return {0: default_value}, {}

    points = np.concatenate(valid_contours, axis=0)
    rows = points[:, 0]
    cols = points[:, 1].astype(int)

    order = np.argsort(cols)
    cols_sorted = cols[order]
    rows_sorted = rows[order]

    unique_cols, start_idx, counts = np.unique(cols_sorted, return_index=True, return_counts=True)

    thickness_raw: Dict[int, float] = {}
    for col, start, count in zip(unique_cols, start_idx, counts):
        if count < min_points_per_column:
            continue
        group = rows_sorted[start:start + count]
        d = float(group.max() - group.min())
        if d >= min_distance:
            thickness_raw[int(col)] = d

    # optional smoothing
    if smooth and len(thickness_raw) >= kernel_size:
        k = max(1, int(kernel_size))
        kk = k if k % 2 == 1 else k + 1  # odd window
        keys_sorted = sorted(thickness_raw)
        vals = np.array([thickness_raw[kc] for kc in keys_sorted], dtype=float)
        vals_s = uniform_filter1d(vals, size=kk, mode="nearest")
        thickness_map = dict(zip(keys_sorted, map(float, vals_s)))
    else:
        thickness_map = thickness_raw

    if not thickness_map:
        thickness_map = {0: default_value}
        summary = {key: float('nan') for key in (metrics or {
            "mean", "median", "std", "min", "max", "count", "percent"
        })}
        return thickness_map, summary, {}
    
    if use_robust_filter:
        params = robust_filter_params or {}
        thickness_map = filter_outliers_robust(thickness_map, **params)

    values = np.array(list(thickness_map.values()), dtype=float)
    default_metrics: Dict[str, Callable[[ArrayNP], float]] = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "std": np.nanstd,
        "min": np.nanmin,
        "max": np.nanmax,
    }
    if shape_length is not None and shape_length > 0:
        default_metrics["percent"] = lambda x: float((len(x) * 100.0) / shape_length)

    used = dict(default_metrics)
    if metrics:
        used.update(metrics)

    summary = {name: float(func(values)) for name, func in used.items()}
    return thickness_map, summary


# =========================================================
# Aggregation over stacks
# =========================================================
def aggregate_stack_summaries(
    thickness_maps: Optional[Sequence[Dict[int, float]]] = None,
    summaries: Optional[Sequence[Mapping[str, float]]] = None,
    metrics: Optional[Mapping[str, Callable[[ArrayNP], float]]] = None,
    shape_length: Optional[int] = None,
    mode: str = "value",  # "value" | "mean" | "weighted"
    use_robust_filter: bool = False,
    robust_filter_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """
    Aggregate thickness statistics over a stack of images.

    Parameters:
    - thickness_maps: list of {col: thickness} dicts, one per image.
    - summaries: list of summary dicts (mean, std, etc.) per image.
    - metrics: custom metric functions, e.g., {"mean": np.nanmean, ...}
    - shape_length: expected full width (for percent if needed)
    - mode: "value" (concat values), "mean" (avg of stats), "weighted" (avg of stats with weight)
    - use_robust_filter: apply outlier filtering
    - robust_filter_params: params for outlier filtering

    Returns:
    - summary dict
    """
    if mode == "value":
        if not thickness_maps:
            raise ValueError("thickness_maps must be provided in 'value' mode.")
        all_vals_list: List[float] = []
        for tmap in thickness_maps:
            if tmap:
                all_vals_list.extend(map(float, tmap.values()))
        if len(all_vals_list) == 0:
            return {}

        all_values = np.asarray(all_vals_list, dtype=float)

        if use_robust_filter:
            params = dict(steps=("quantile", "iqr"))
            if robust_filter_params:
                params.update(robust_filter_params)
            filtered = filter_outliers_robust(all_values, **params)  # type: ignore[arg-type]
            all_values = np.asarray(list(filtered.values()), dtype=float)

        default_metrics: Dict[str, Callable[[ArrayNP], float]] = {
            "mean": np.nanmean,
            "median": np.nanmedian,
            "std": np.nanstd,
            "min": np.nanmin,
            "max": np.nanmax,
        }
        if shape_length is not None and shape_length > 0 and thickness_maps:
            default_metrics["percent"] = lambda x: float((len(x) * 100.0) / (shape_length * len(thickness_maps)))

        used = dict(default_metrics)
        if metrics:
            used.update(metrics)

        return {k: float(f(all_values)) for k, f in used.items()}

    if mode in ("mean", "weighted"):
        if not summaries:
            raise ValueError("`summaries` must be provided in 'mean' or 'weighted' mode.")
        keys = list(summaries[0].keys())
        values = {k: np.array([float(s.get(k, np.nan)) for s in summaries], dtype=float) for k in keys}

        if mode == "mean":
            return {k: float(np.nanmean(v)) for k, v in values.items()}

        # weighted
        if not thickness_maps:
            raise ValueError("`thickness_maps` required to compute weights in 'weighted' mode.")
        weights = np.array([len(tmap) if tmap else 0 for tmap in thickness_maps], dtype=float)
        wsum = float(np.nansum(weights))
        if wsum <= 0:
            return {k: float(np.nanmean(v)) for k, v in values.items()}
        return {k: float(np.nansum(values[k] * weights) / wsum) for k in keys}

    raise ValueError(f"Unknown mode: {mode!r}")



# =========================================================
# Visualization
# =========================================================
def visualize_thickness_profiles(
    *profiles: Dict[int, float],
    summaries: Optional[Sequence[Optional[Mapping[str, float]]]] = None,
    summary_keys: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    title: str = "Thickness profile",
    show_summary_lines: bool = True,
    figsize: Tuple[float, float] = (10, 4),
    ax: Optional[Any] = None,
    short_labels: bool = True,
):
    """
    Visualize one or more thickness profiles with optional summary annotations.

    Parameters
    ----------
    *profiles : Dict[int, float]
        One or more thickness profiles, each represented as a dictionary mapping
        index (e.g., slice position) to thickness value.
    summaries : Sequence of dicts, optional
        Optional list of summary statistics (e.g., mean, median, std) for each profile.
    summary_keys : Sequence of str, optional
        Keys to display from each summary dict (e.g., ["mean", "median"]).
    labels : Sequence of str, optional
        Labels for each profile to be shown in the legend.
    colors : Sequence of str, optional
        Colors to use for each profile line.
    title : str, optional
        Title of the plot. Default is "Thickness profile".
    show_summary_lines : bool, optional
        If True, draw horizontal lines for each summary statistic.
    figsize : Tuple[float, float], optional
        Size of the figure in inches. Default is (10, 4).
    ax : matplotlib axis or None, optional
        If provided, draw on the given axis; otherwise create a new figure.
    short_labels : bool, optional
        If True, use abbreviated labels for summary stats (e.g., "μ", "med").

    Returns
    -------
    None
        The function displays the plot and does not return any value.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    n = len(profiles)
    labels = list(labels) if labels is not None else [f"Profil {i+1}" for i in range(n)]
    colors = list(colors) if colors is not None else ['blue', 'green', 'red', 'orange', 'purple', 'brown'][:n]

    if summaries is None:
        summaries = [None] * n
    if summary_keys is None:
        keys_set = []
        for s in summaries:
            if isinstance(s, Mapping):
                for k in s.keys():
                    if k not in keys_set:
                        keys_set.append(k)
        summary_keys = keys_set

    linestyles_base = ['--', ':', '-.', '-']
    def pick_style(j: int) -> str:
        return linestyles_base[j % len(linestyles_base)]

    label_symbols = {
        "mean": "μ",
        "median": "med",
        "std": "σ",
        "min": "min",
        "max": "max",
        "count": "n",
        "percent": "%",
    }

    for i, profile in enumerate(profiles):
        if not profile:
            continue
        cols = sorted(profile)
        vals = [profile[c] for c in cols]
        ax.plot(cols, vals, label=labels[i], color=colors[i])

        summary = summaries[i]
        if show_summary_lines and isinstance(summary, Mapping) and summary_keys:
            for j, key in enumerate(summary_keys):
                val = summary.get(key, None)
                if val is None or np.isnan(val):
                    continue
                name = label_symbols.get(key, key) if short_labels else key.title()
                ax.axhline(
                    float(val),
                    linestyle=pick_style(j),
                    color=colors[i],
                    alpha=0.35,
                    label=f"{name} {labels[i]} = {float(val):.2f}",
                )

    ax.set_title(title)
    ax.set_xlabel("Colonne (μm)")
    ax.set_ylabel("Épaisseur (pixels)")
    ax.legend(fontsize=9, loc='best')
    plt.tight_layout()
    return ax