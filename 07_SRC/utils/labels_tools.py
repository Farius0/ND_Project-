# ==================================================
# =============  MODULE: labels_tools  =============
# ==================================================
from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from skimage.draw import polygon2mask
from skimage.measure import find_contours
from skimage.morphology import (
    closing,
    dilation,
    erosion,
    remove_small_objects,
    disk,
    rectangle,
)

Array = np.ndarray

__all__ = [
    "scribble_labels",
    "scribble_labels_2",
    "return_region_cropped",
    "generate_minimal_scribble",
]

def _ensure_uint_like(arr: Array) -> Array:
    """
    Ensure that the input array contains unsigned integer-like values.

    This utility is typically used to sanitize label arrays by removing floating-point
    types or NaNs and casting to an appropriate unsigned integer type.

    Parameters
    ----------
    arr : Array
        Input array to sanitize (e.g., label map).

    Returns
    -------
    Array
        Cleaned array with integer-compatible dtype and no NaNs.
    """

    if not np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int32, copy=False)
    return arr


def _safe_remove_small(mask: Array, min_size: int) -> Array:
    """
    Remove small connected components from a binary (boolean) mask.

    Keeps only connected regions with a size greater than or equal to `min_size`.
    If the input mask is empty or no region meets the size criterion, returns
    an empty mask of the same shape.

    Parameters
    ----------
    mask : Array
        Binary (boolean) mask from which to remove small components.
    min_size : int
        Minimum number of pixels required to keep a connected component.

    Returns
    -------
    Array
        Cleaned binary mask with small components removed.
    """

    mask_bool = mask.astype(bool, copy=False)
    cleaned = remove_small_objects(mask_bool, min_size=max(1, int(min_size)))
    return cleaned


def scribble_labels(
    seg: Array,
    seg_fft: Optional[Array] = None,
    reg_att: Optional[Array] = None,
    num_layers: int = 3,
    unannotated: int = 255,
    min_region_size: int = 2000,
) -> Array:
    """
    Clean and reconstruct label maps from segmentation results, enforcing vertical consistency.

    Label convention
    ----------------
    - 0: top background
    - 1..num_layers: anatomical layers from top to bottom
    - C-1 (i.e., num_layers + 1): bottom background
    - unannotated (e.g., 255) used for unknown pixels during processing

    Parameters
    ----------
    seg : (H,W) int ndarray
        Raw annotation map with values in {0..C-1} or `unannotated`.
    seg_fft : (H,W) int ndarray, optional
        Auxiliary cues (e.g., frequency-based). If provided, values in {0..C-1} expected.
    reg_att : (H,W) int ndarray, optional
        Region-attention cues. If provided, values in {0..C-1} expected.
    num_layers : int, default 3
        Number of anatomical layers.
    unannotated : int, default 255
        Special value for unknown pixels.
    min_region_size : int, default 2000
        Minimum connected component size to preserve for each class (>=1).

    Returns
    -------
    seg_fin : (H,W) int ndarray
        Cleaned and vertically consistent label map within {0..num_layers+1}.
    """
    seg = _ensure_uint_like(seg)
    H, W = seg.shape
    C = num_layers + 2  # total classes: top_bg(0), layers(1..num_layers), bottom_bg(C-1)

    # Initialize all to unannotated
    seg_fin = np.full((H, W), fill_value=unannotated, dtype=np.int32)

    # ---- Step 1: transfer known scribble layers (seed) ----
    # Top layer (1) with a shallow top limit heuristic
    if np.any(seg == 1):
        top_mask = (seg == 1)
        mean_profile = top_mask.mean(axis=1)
        top_limit = int(np.argmax(mean_profile)) + 30  # heuristic margin
        top_mask[top_limit:, :] = False
        seg_fin[top_mask] = 1
        
    # Other layers (use seg, optionally refine with seg_fft/reg_att)
    for i in range(2, num_layers + 1):
        if seg_fft is not None and reg_att is not None:
            if i == 2:
                # prefer seg_fft==4 where layer 1 isn't already placed
                mask_1 = (seg_fin == 1)
                cand = (seg_fft == 4)
                seg_fin[(~mask_1) & cand] = 2
            elif i == 3:
                cand = (seg_fft == 3)
                seg_fin[cand] = 3
            else:
                seg_fin[seg == i] = i
        else:
            seg_fin[seg == i] = i
                
    # ---- Step 2: remove small blobs per layer (1..num_layers) ----
    for c in range(1, num_layers + 1):
        mask_c = (seg_fin == c)
        if not np.any(mask_c):
            continue
        cleaned = _safe_remove_small(mask_c, min_region_size)
        # invalidate only the removed pixels (go back to unannotated)
        seg_fin[mask_c & (~cleaned)] = unannotated

    # ---- Step 3: vertical consistency per class ----
    att = {}
    class_order = list(range(1, num_layers + 1))
    
    for c in class_order:
        
        if c == 1:
            mask = (seg_fin == c)
        elif reg_att is not None:
            mask = (reg_att == c)
        else:
            mask = (seg_fin == c)
        
        rows_with_c = np.where(np.any(mask, axis=1))[0]
        if len(rows_with_c) == 0:
            continue 
            
        min_row, max_row = rows_with_c[0], rows_with_c[-1]
        
        # Special case for top layer (c == 1)
        att[c] = max_row
        
        if c > 1 and c < num_layers:
            ref_max = att.get(c - 1, None)
            if ref_max is not None:
                min_row = max(ref_max + 1, min_row)
        
        # Remove labels outside their valid vertical region
        above = (seg_fin[:min_row, :] >= c) 
        middle = (seg_fin[min_row:max_row + 1, :] != c)
        below = (seg_fin[max_row + 1:, :] <= c)

        seg_fin[:min_row, :][above] = unannotated
        seg_fin[min_row:max_row + 1, :][middle] = unannotated
        seg_fin[max_row + 1:, :][below] = unannotated

    # ---- Step 4: fill bottom background (C-1) below the deepest layer) ----
    bottom_mask = seg_fin == (C - 2)
    if np.any(bottom_mask):
        bottom_row = np.max(np.where(np.any(bottom_mask, axis=1))[0])
        seg_fin[(seg_fin != C - 1) & (np.arange(H)[:, None] > bottom_row)] = C - 1

    # ---- Step 5: fill top background (0) above the highest layer ----
    top_mask = seg_fin == 1
    if np.any(top_mask):
        top_row = np.min(np.where(np.any(top_mask, axis=1))[0])
        seg_fin[(seg_fin != 0) & (np.arange(H)[:, None] < top_row)] = 0

    return seg_fin

def scribble_labels_2(
    seg: Array,
    num_classes: int = 4,
    unannotated: int = 255,
    min_region_size: int = 2000,
) -> Array:
    """
    Legacy-style refinement of a dense segmentation into scribble-like labels with layer constraints.

    Parameters
    ----------
    seg : (H,W) int ndarray
        Initial segmentation map (1..num_classes presumed).
    num_classes : int, default 4
        Number of foreground classes.
    unannotated : int, default 255
        Value assigned to cleaned/unknown pixels.
    min_region_size : int, default 2000
        Minimum blob size to keep for layers.

    Returns
    -------
    seg_fin : (H,W) int ndarray
        Cleaned scribble-like label map.
    """
    seg = _ensure_uint_like(seg)
    H, W = seg.shape
    seg_fin = np.full((H, W), fill_value=unannotated, dtype=np.int32)

    # === Step 1: transfer known scribble layers ===
    # Seed top layer (1)
    if np.any(seg == 1):
        top_mask = (seg == 1)
        mean_profile = top_mask.mean(axis=1)
        top_limit = int(np.argmax(mean_profile)) + 30
        top_mask[top_limit:, :] = False
        seg_fin[top_mask] = 1

    for i in range(2, num_classes + 1):
        mask = seg == i
        if i == 2:
            seg_fin[mask] = 0  # Presumed top background
        elif i == 3:
            seg_fin[mask] = 3  # Middle layer
        elif i == 4:
            seg_fin[mask] = 2  # Another layer

    # === Step 2: remove small blobs ===
    for c in range(1, num_classes + 1):
        mask_c = (seg_fin == c)
        if not np.any(mask_c):
            continue
        cleaned = _safe_remove_small(mask_c, min_region_size)
        seg_fin[mask_c & (~cleaned)] = unannotated

    # === Step 3: remove vertically inconsistent labels ===
    class_order = [1, 2, 3]  # Order of enforcement by layer depth

    for c in class_order:
        mask = (seg_fin == c)
        rows_with_c = np.where(np.any(mask, axis=1))[0]
        if len(rows_with_c) == 0:
            continue

        min_row, max_row = rows_with_c[0], rows_with_c[-1]

        if c == 2:
            max_row = min_row + 60
        elif c == 3:
            min_row = max_row - 100

        # Cleanup above, in the middle, and below
        above = (seg_fin[:min_row, :] >= c)
        middle = (seg_fin[min_row:max_row + 1, :] != c)
        below = (seg_fin[max_row + 1:, :] <= c)

        seg_fin[:min_row, :][above] = unannotated
        seg_fin[min_row:max_row + 1, :][middle] = unannotated
        seg_fin[max_row + 1:, :][below] = unannotated

    # === Step 4: fill bottom background ===
    bottom_mask = seg_fin == 3
    if np.any(bottom_mask):
        bottom_row = np.max(np.where(np.any(bottom_mask, axis=1))[0])
        seg_fin[(seg_fin != 4) & (np.arange(H)[:, None] > bottom_row)] = 4

    return seg_fin


def return_region_cropped(
    image_segment: Array,
    num_classes: int = 4,
    margin_top: int = 60,
    margin_bottom: int = 60,
) -> Array:
    """
    Crop and refine segmented regions using vertical heuristics and morphological smoothing.

    Parameters
    ----------
    image_segment : (H,W) int ndarray
        Segmentation map with labels starting from 1 (assumed).
    num_classes : int, default 4
        Number of classes in the segmentation.
    margin_top : int, default 60
        Margin to expand above the top threshold.
    margin_bottom : int, default 60
        Margin to expand below the bottom threshold.

    Returns
    -------
    combined : (H,W) uint8 ndarray
        Cropped and cleaned segmentation with relabeled regions (1..num_classes),
        0 means background/unassigned in this routine.
    """
    
    seg = _ensure_uint_like(image_segment)
    H, W = seg.shape
    one_hot = np.eye(num_classes, dtype=np.uint8)[np.clip(seg - 1, 0, num_classes - 1)]
    regions: Dict[int, Array] = {}

    class_order = [0, 3, 2, 1]  # Custom order: assumed layer priorities

    for idx in class_order:
         # one_hot[:, :, idx] corresponds to class label (idx+1)
        img = one_hot[:, :, idx].astype(bool, copy=False)

        if idx in [1, 3]:  # Vertical cropping for top/bottom layers
            mean_ax1 = img.mean(axis=1)

            if idx == 1:  # Crop top based on layer 2
                reference = regions.get(2, None)
                if reference is not None:
                    rows_with_content = np.where(np.any(reference, axis=1))[0]
                    if len(rows_with_content) > 0:
                        min_row = rows_with_content[0]
                        profile = np.concatenate([
                            np.zeros_like(mean_ax1[:min_row]), mean_ax1[min_row:]
                        ])
                        thresh_bottom = np.argmax(profile) - margin_top
                        img[:thresh_bottom, :] = False

            elif idx == 3:  # Crop bottom
                thresh_top = np.argmax(mean_ax1) + margin_bottom
                img[thresh_top:, :] = False
                
        # Morphological smoothing pipeline (compact & stable)
        footprint = rectangle(5, 10)
        img_morph = erosion(img, footprint=footprint)
        img_morph = remove_small_objects(img_morph, min_size=2000)
        img_morph = dilation(img_morph, footprint=footprint)
        img_morph = closing(img_morph, footprint=footprint)
        img_morph = dilation(img_morph, footprint = disk(3))
        img_morph = closing(img_morph, footprint = disk(10)) 
        
        # Store processed region
        regions[idx] = img_morph

    # Combine final result
    combined = np.zeros((H, W), dtype=np.uint8)
    for new_label, class_idx in enumerate(class_order, start=1):
        region = regions.get(class_idx, None)
        if region is not None:
            combined[region] = np.uint8(new_label)

    combined[combined == 0] = num_classes + 1  # Optional label for background/unassigned
    return combined

def generate_minimal_scribble(
    scribble: Array,
    n_classes: int = 5,
    background_classes: Sequence[int] = (0, 4),
    patch_ratios: Dict[int, Tuple[float, float]] = {0: (0.25, 0.10), 4: (0.15, 0.10)},
    crop_strategy: Literal["center", "specific"] = "specific",
    contour_classes: Sequence[int] = (1, 2, 3),
    min_contour_length: int = 100,
    n_contours: Dict[int, int] = {1: 2, 2: 5, 3: 5},
    dilate_radius: int = 0,
    unannotated_value: int = 255,
    seed: Optional[int] = None,
) -> Array:
    """
    Generate sparse 'scribble' labels from a dense mask via small patches (background)
    and contour-based selections (foreground), optionally dilated.

    Parameters
    ----------
    scribble : (H,W) int ndarray
        Dense segmentation mask (0..n_classes).
    n_classes : int, default 5
        Highest foreground class index.
    background_classes : sequence of int, default (0,4)
        Class IDs treated with patch selection.
    patch_ratios : dict[class -> (rh, rw)]
        Ratios (height, width) of the patch within the class bounding box.
    crop_strategy : {'center','specific'}, default 'specific'
        Patch placement policy for background classes.
    contour_classes : sequence of int, default (1,2,3)
        Classes handled via contour selection.
    min_contour_length : int, default 100
        Minimum contour length (in pixels) to keep.
    n_contours : dict[class -> int], default {1:2,2:5,3:5}
        Max number of contours to keep per class.
    dilate_radius : int, default 0
        Radius for optional dilation (0: none).
    unannotated_value : int, default 255
        Value for unannotated pixels.
    seed : int, optional
        Seed for deterministic selection (background patch choice).

    Returns
    -------
    scribble_final : (H,W) int ndarray
        Sparse scribble map; unselected pixels set to `unannotated_value`.
    """
    rng = np.random.default_rng(seed)
    H, W = scribble.shape
    one_hot = np.eye(n_classes + 1, dtype=np.uint8)[np.clip(scribble, 0, n_classes)]
    region_final: Dict[int, Array] = {}

    # --- Background classes (patch strategy) ---
    for c in background_classes:
        label = one_hot[:, :, c]
        if label.sum() == 0:
            continue

        label_final = np.zeros_like(label, dtype=np.uint8)
        ys, xs = np.where(label == 1)
        row_min, row_max = int(ys.min()), int(ys.max())
        col_min, col_max = int(xs.min()), int(xs.max())
        h, w = row_max - row_min + 1, col_max - col_min + 1
        h = h + 1 if h % 2 else h
        w = w + 1 if w % 2 else w
        rh, rw = patch_ratios.get(c, (0.1, 0.1))
        mh = max(1, int(rh * h))
        mw = max(1, int(rw * w))
        center_h = h // 2 + row_min
        center_w = w // 2 + col_min

        if crop_strategy == "center":
            up, down = center_h - mh, center_h + mh
            left, right = center_w - mw, center_w + mw
        else:
            # choose among left/center/right (deterministic via rng)
            choices = [
                (col_min, col_min + mw),
                (center_w - mw, center_w + mw),
                (col_max - mw, col_max),
            ]
            left, right = choices[rng.integers(0, len(choices))]
            if c == background_classes[0]:
                up, down = row_min, min(row_min + mh, H)
            else:
                up, down = max(row_max - mh, 0), row_max

        # Safe bounds
        up, down = max(0, up), min(H, down)
        left, right = max(0, left), min(W, right)
        label_final[up:down, left:right] = label[up:down, left:right]
        region_final[c] = label_final

    # --- Foreground classes (contour strategy) ---
    for c in contour_classes:
        label = one_hot[:, :, c].astype(bool, copy=False)
        if label.sum() == 0:
            continue
        
        # Light erosion for non-top classes
        if c != 1:
            label = erosion(label, disk(5))
        
        # Contours are arrays of (row, col) floats; keep them as floats (no cast to int)
        contours = [cnt for cnt in find_contours(label) if len(cnt) >= min_contour_length]
        # Order: largest first for c==1, otherwise keep natural order or size-based
        contours = sorted(contours, key=len, reverse=True if c == 1 else False)
        limit = min(n_contours.get(c, 5), len(contours))
        selected = contours[:limit]

        label_final = np.zeros_like(label, dtype=np.uint8)
        for reg in selected:
            mask = polygon2mask(label.shape, reg)  # accepts float coordinates
            label_final[mask] = 1

        if dilate_radius > 0:
            label_final = dilation(label_final.astype(bool), disk(dilate_radius)).astype(np.uint8)

        region_final[c] = label_final

    # --- Final merge ---
    scribble_final = np.full((H, W), fill_value=unannotated_value, dtype=np.int32)
    for c, mask in region_final.items():
        scribble_final[mask.astype(bool)] = int(c)

    return scribble_final