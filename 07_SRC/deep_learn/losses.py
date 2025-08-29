# ==================================================
# ===============  MODULE: losses  =================
# ==================================================
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

# Public API
__all__ = [
    "CrossEntropyLoss",
    "pCELoss",
    "DiceLoss",
    "pDLoss",
    "ComboLoss",
    "FocalLoss",
    "IoULoss",
    "TverskyLoss",
    "FocalTverskyLoss",
    "LovaszSoftmaxLoss",
    "DistillationLoss",
    "ConsistencyLoss",
    "LossSelector",
]

# ------------------------ helpers ------------------------

def ensure_target_shape(targets: torch.Tensor) -> torch.Tensor:
    """
    Ensure target tensor has shape (B, H, W) and integer dtype.
    Accepts (B, 1, H, W) and squeezes the channel dim if 1.
    """
    if targets.ndim == 4:
        if targets.shape[1] == 1:
            targets = targets.squeeze(1)
        else:
            raise ValueError(f"Expected targets with shape (B, 1, H, W) for 4D input, got {tuple(targets.shape)}")
    elif targets.ndim != 3:
        raise ValueError(f"Expected targets with 3 or 4 dimensions, got {targets.ndim} and shape {tuple(targets.shape)}")
    return targets.long()

def compute_class_imbalance(targets: torch.Tensor, n_classes: int, ignore_index: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute class frequency imbalance ratios in the current batch.
    Returns (imbalance, freq) with sum(freq)=1 (if any valid pixel), 0 otherwise.
    """
    targets = targets.clone()
    if ignore_index is not None:
        targets[targets == ignore_index] = -1  # temporary mask

    class_counts = torch.zeros(n_classes, dtype=torch.float, device=targets.device)
    for cls in range(n_classes):
        if ignore_index is not None and cls == ignore_index:
            continue
        class_counts[cls] = (targets == cls).sum()

    total = class_counts.sum()
    freq = class_counts / (total + 1e-8)
    imbalance = 1.0 - freq  # higher = rarer
    return imbalance, freq

# ------------------------ losses ------------------------

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Standard Cross Entropy Loss.

        Args:
            inputs (Tensor): Logits of shape (B, C, H, W).
            targets (Tensor): Labels of shape (B, H, W).
        Returns:
            Tensor: Scalar loss.
        """
        targets = ensure_target_shape(targets)
        return self.loss(inputs, targets)

class pCELoss(nn.Module):
    def __init__(self, ignore_index: int = 255, reduction: str = "mean") -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Partially supervised Cross-Entropy Loss (ignores unlabeled pixels).

        Args:
            inputs (Tensor): Logits of shape (B, C, H, W).
            targets (Tensor): Labels of shape (B, H, W) with ignore_index.
        Returns:
            Tensor: Scalar loss.
        """
        targets = ensure_target_shape(targets)
        B, C, H, W = inputs.shape

        # Flatten to (B*H*W, C) and (B*H*W)
        logits = inputs.permute(0, 2, 3, 1).reshape(-1, C)
        flat_targets = targets.view(-1)

        # Mask unannotated pixels
        valid = flat_targets != self.ignore_index
        logits = logits[valid]
        flat_targets = flat_targets[valid]

        if logits.numel() == 0:
            # Empty tensor (keeps graph differentiable)
            return torch.zeros((), dtype=inputs.dtype, device=inputs.device, requires_grad=True)

        return F.cross_entropy(logits, flat_targets, reduction=self.reduction)
    
class pDLoss(nn.Module):
    def __init__(self, n_classes: int, ignore_index: int = 255, smooth: float = 1e-5, softmax: bool = True) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.softmax = softmax

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, weight: Optional[Union[torch.Tensor, Sequence[float]]] = None) -> torch.Tensor:
        """
        Compute partially supervised Dice loss, ignoring unannotated pixels.

        Args:
            inputs (Tensor): Logits or probabilities of shape (B, C, H, W).
            targets (Tensor): Ground truth labels of shape (B, H, W).
            weight (Tensor or list, optional): Class weights of shape (C,).
        Returns:
            Tensor: Weighted Dice loss.
        """
        x = F.softmax(inputs, dim=1) if self.softmax else inputs

        # Ensure weight tensor
        if weight is None:
            w = torch.ones(self.n_classes, device=x.device, dtype=x.dtype)
        elif torch.is_tensor(weight):
            w = weight.to(device=x.device, dtype=x.dtype)
        else:
            w = torch.tensor(weight, device=x.device, dtype=x.dtype)

        targets = ensure_target_shape(targets)

        # Mask valid pixels
        mask = (targets != self.ignore_index).unsqueeze(1)  # (B,1,H,W)

        # One-hot targets
        y = F.one_hot(torch.clamp(targets, 0, self.n_classes - 1), num_classes=self.n_classes)
        y = y.permute(0, 3, 1, 2).to(x.dtype).to(x.device)

        # Apply mask
        mask = mask.expand_as(y)
        x = x * mask
        y = y * mask

        # Dice per class
        intersection = torch.sum(x * y, dim=(0, 2, 3))
        pred_sum = torch.sum(x, dim=(0, 2, 3))
        gt_sum = torch.sum(y, dim=(0, 2, 3))

        dice_score = (2 * intersection + self.smooth) / (pred_sum + gt_sum + self.smooth)
        dice_loss = 1.0 - dice_score

        return (dice_loss * w).sum() / (w.sum() + 1e-8)


class DiceLoss(nn.Module):
    def __init__(self, n_classes: int, ignore_index: Optional[int] = None, smooth: float = 1e-5, softmax: bool = True) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.softmax = softmax

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, weight: Optional[Union[torch.Tensor, Sequence[float]]] = None) -> torch.Tensor:
        """
        Compute Dice loss (full supervision).

        Args:
            inputs (Tensor): Logits or probabilities of shape (B, C, H, W).
            targets (Tensor): Ground truth labels of shape (B, H, W).
            weight (Tensor or list, optional): Class weights of shape (C,).
        Returns:
            Tensor: Weighted Dice loss.
        """
        x = F.softmax(inputs, dim=1) if self.softmax else inputs

        if weight is None:
            w = torch.ones(self.n_classes, device=x.device, dtype=x.dtype)
        elif torch.is_tensor(weight):
            w = weight.to(device=x.device, dtype=x.dtype)
        else:
            w = torch.tensor(weight, device=x.device, dtype=x.dtype)

        targets = ensure_target_shape(targets)
        y = F.one_hot(torch.clamp(targets, 0, self.n_classes - 1), num_classes=self.n_classes)
        y = y.permute(0, 3, 1, 2).to(x.dtype).to(x.device)

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1).to(x.dtype)
            mask = mask.expand_as(y)
            x = x * mask
            y = y * mask

        intersection = torch.sum(x * y, dim=(0, 2, 3))
        pred_sum = torch.sum(x, dim=(0, 2, 3))
        gt_sum = torch.sum(y, dim=(0, 2, 3))

        dice_score = (2 * intersection + self.smooth) / (pred_sum + gt_sum + self.smooth)
        dice_loss = 1.0 - dice_score

        return (dice_loss * w).sum() / (w.sum() + 1e-8)
    
class ComboLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5, n_classes: int = 4, ignore_index: int = 255, use_partial: bool = True) -> None:
        super().__init__()
        if use_partial:
            self.ce = pCELoss(ignore_index=ignore_index)
            self.dice = pDLoss(n_classes=n_classes, ignore_index=ignore_index)
        else:
            self.ce = CrossEntropyLoss()
            self.dice = DiceLoss(n_classes=n_classes, ignore_index=ignore_index)
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(inputs, targets)
        ce_loss = self.ce(inputs, targets)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=255, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) — logits
            targets: (B, H, W) — int labels with ignore_index
        """
        B, C, H, W = inputs.shape
        logits = inputs.permute(0, 2, 3, 1).reshape(-1, C)

        targets = ensure_target_shape(targets).view(-1)
        valid = targets != self.ignore_index
        logits = logits[valid]
        targets = targets[valid]

        if logits.numel() == 0:
            return torch.zeros((), dtype=inputs.dtype, device=inputs.device, requires_grad=True)

        log_probs = F.log_softmax(logits, dim=1)  # (N, C)
        probs = log_probs.exp()
        idx = torch.arange(targets.numel(), device=targets.device)
        pt = probs[idx, targets]           # (N,)
        log_pt = log_probs[idx, targets]   # (N,)

        if self.alpha is not None:
            if torch.is_tensor(self.alpha):
                alpha_vec = self.alpha.to(device=logits.device, dtype=logits.dtype)
                at = alpha_vec[targets]
            elif isinstance(self.alpha, (list, tuple)):
                alpha_vec = torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype)
                if alpha_vec.numel() == 1:
                    at = alpha_vec.expand_as(pt)
                else:
                    at = alpha_vec[targets]
            else:
                # scalar
                at = torch.tensor(float(self.alpha), device=logits.device, dtype=logits.dtype).expand_as(pt)
            loss = -at * (1 - pt) ** self.gamma * log_pt
        else:
            loss = -(1 - pt) ** self.gamma * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class IoULoss(nn.Module):
    def __init__(self, n_classes: int, ignore_index: Optional[int] = None, smooth: float = 1e-5, softmax: bool = True) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.softmax = softmax


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        IoU loss (Jaccard) per classe moyennée.
        inputs : (B, C, H, W) logits
        targets: (B, H, W) int labels
        """
        x = F.softmax(inputs, dim=1) if self.softmax else inputs
        targets = ensure_target_shape(targets)

        y = F.one_hot(torch.clamp(targets, 0, self.n_classes - 1), num_classes=self.n_classes)
        y = y.permute(0, 3, 1, 2).to(x.dtype).to(x.device)

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1).to(x.dtype)
            mask = mask.expand_as(y)
            x = x * mask
            y = y * mask

        intersection = torch.sum(x * y, dim=(0, 2, 3))
        union = torch.sum(x + y - x * y, dim=(0, 2, 3))
        iou = (intersection + self.smooth) / (union + self.smooth)
        return (1.0 - iou).mean()

class TverskyLoss(nn.Module):
    def __init__(self, n_classes: int, alpha: Optional[float] = None, beta: Optional[float] = None, dynamic: bool = True,
                 ignore_index: Optional[int] = None, smooth: float = 1e-5, softmax: bool = True) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta
        self.dynamic = dynamic
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.softmax = softmax

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = F.softmax(inputs, dim=1) if self.softmax else inputs
        targets = ensure_target_shape(targets)

        y = F.one_hot(torch.clamp(targets, 0, self.n_classes - 1), num_classes=self.n_classes)
        y = y.permute(0, 3, 1, 2).to(x.dtype).to(x.device)

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1).to(x.dtype)
            mask = mask.expand_as(x)
            x = x * mask
            y = y * mask

        if self.dynamic:
            imbalance, _ = compute_class_imbalance(targets, self.n_classes, self.ignore_index)
            alpha = float(imbalance.mean().item())
            beta = 1.0 - alpha
        else:
            alpha = self.alpha if (self.alpha is not None) else 0.5
            beta = self.beta if (self.beta is not None) else 0.5

        TP = torch.sum(x * y, dim=(0, 2, 3))
        FP = torch.sum(x * (1 - y), dim=(0, 2, 3))
        FN = torch.sum((1 - x) * y, dim=(0, 2, 3))

        tversky = (TP + self.smooth) / (TP + alpha * FP + beta * FN + self.smooth)
        return (1.0 - tversky).mean()
    
class FocalTverskyLoss(nn.Module):
    def __init__(self, n_classes: int, alpha: Optional[float] = None, beta: Optional[float] = None, gamma: float = 1.33,
                 dynamic: bool = True, ignore_index: Optional[int] = None, smooth: float = 1e-5, softmax: bool = True) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = float(gamma)
        self.dynamic = dynamic
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.softmax = softmax

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, progress: float = 0.0) -> torch.Tensor:
        x = F.softmax(inputs, dim=1) if self.softmax else inputs
        targets = ensure_target_shape(targets)

        y = F.one_hot(torch.clamp(targets, 0, self.n_classes - 1), num_classes=self.n_classes)
        y = y.permute(0, 3, 1, 2).to(x.dtype).to(x.device)

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1).to(x.dtype)
            mask = mask.expand_as(x)
            x = x * mask
            y = y * mask

        if self.dynamic:
            imbalance, _ = compute_class_imbalance(targets, self.n_classes, self.ignore_index)
            alpha = float(imbalance.mean().item())
            beta = 1.0 - alpha
            gamma = 1.0 + (self.gamma - 1.0) * min(float(progress), 1.0)
        else:
            alpha = self.alpha if (self.alpha is not None) else 0.7
            beta = self.beta if (self.beta is not None) else 0.3
            gamma = self.gamma

        TP = torch.sum(x * y, dim=(0, 2, 3))
        FP = torch.sum(x * (1 - y), dim=(0, 2, 3))
        FN = torch.sum((1 - x) * y, dim=(0, 2, 3))

        tversky = (TP + self.smooth) / (TP + alpha * FP + beta * FN + self.smooth)
        loss = (1.0 - tversky).pow(gamma)
        return loss.mean()

# ------------------------ Lovasz ------------------------

def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, classes: Union[str, Sequence[int]] = "present") -> torch.Tensor:
    """
    probas: (P, C) softmaxed probabilities
    labels: (P,) ground truth labels
    classes: 'present' | 'all' | list of class ids
    """
    if probas.numel() == 0:
        return probas * 0.0

    C = probas.size(1)
    losses = []

    class_to_sum = range(C) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    return sum(losses) / max(len(losses), 1)

def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of the Lovasz extension w.r.t sorted errors.
    """
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, ignore_index: Optional[int] = 255) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs : (B, C, H, W) logits
        targets: (B, H, W) int labels
        """
        probs = F.softmax(inputs, dim=1)
        B, C, H, W = probs.shape
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)

        targets = ensure_target_shape(targets)
        targets_flat = targets.view(-1)

        if self.ignore_index is not None:
            valid = targets_flat != self.ignore_index
            probs_flat = probs_flat[valid]
            targets_flat = targets_flat[valid]

        return lovasz_softmax_flat(probs_flat, targets_flat)

# ------------------------ consistency / distillation ------------------------

class ConsistencyLoss(nn.Module):
    def __init__(self, mode: str = "mse", softmax: bool = True) -> None:
        """
        mode: 'mse' or 'kl'
        """
        super().__init__()
        self.mode = mode.lower()
        self.softmax = softmax
        if self.mode not in ["mse", "kl"]:
            raise ValueError(f"Unknown consistency loss mode: {self.mode}")

    def forward(self, input_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
        if input_logits.size() != target_logits.size():
            raise ValueError("Input/Target shape mismatch in ConsistencyLoss")

        if self.mode == "mse":
            x = F.softmax(input_logits, dim=1) if self.softmax else input_logits
            y = F.softmax(target_logits, dim=1) if self.softmax else target_logits
            return F.mse_loss(x, y)

        # KL mode
        if not self.softmax:
            raise ValueError("KL mode expects softmax=True to compare distributions.")
        logp = F.log_softmax(input_logits, dim=1)
        q = F.softmax(target_logits, dim=1)
        return F.kl_div(logp, q, reduction="batchmean")
        
class DistillationLoss(nn.Module):
    def __init__(
        self,
        base_loss_name: str = "combo",
        distil_mode: str = "kl",
        alpha: float = 0.5,
        n_classes: Optional[int] = None,
        ignore_index: int = 255,
        use_partial: bool = False,
    ) -> None:
        """
        base_loss_name: supervised loss name (e.g., 'crossentropy', 'dice', 'combo', ...)
        distil_mode   : 'kl' or 'mse'
        alpha         : weight of distillation in the final sum
        """
        super().__init__()
        self.alpha = float(alpha)
        self.ignore_index = ignore_index

        self.supervised = LossSelector(
            name=base_loss_name,
            n_classes=n_classes,
            ignore_index=ignore_index,
            use_partial=use_partial,
        )
        self.consistency = ConsistencyLoss(mode=distil_mode)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        supervised_loss = self.supervised(student_logits, targets)
        distil_loss = self.consistency(student_logits, teacher_logits)
        return (1.0 - self.alpha) * supervised_loss + self.alpha * distil_loss

class LossSelector:
    def __init__(
        self,
        name: str,
        n_classes: Optional[int] = None,
        ignore_index: int = 255,
        weight: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            name: str — name of the loss function (e.g., 'crossentropy', 'dice', 'iou', 'combo', 'lovasz', etc.)
            n_classes: int — number of classes (useful for one-hot or multi-class)
            ignore_index: int — label value to ignore in loss computation
            weight: list or Tensor — class weights for certain loss types
            kwargs: additional parameters for specific losses (e.g., alpha, beta for Tversky)
        """
        self.name = name.lower()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.weight = weight
        self.kwargs = kwargs
        self.loss_fn = self._select_loss()
        self.current_progress = 0.0
        
    def set_progress(self, progress: float) -> None:
        self.current_progress = float(progress)

    def _select_loss(self) -> nn.Module:
        if self.name == "crossentropy":
            return CrossEntropyLoss(weight=self.weight)

        elif self.name == "partialce":
            return pCELoss(ignore_index=self.ignore_index)

        elif self.name == "dice":
            return DiceLoss(n_classes=self.n_classes, ignore_index=self.ignore_index)

        elif self.name == "partialdice":
            return pDLoss(n_classes=self.n_classes, ignore_index=self.ignore_index)

        elif self.name == "iou":
            return IoULoss(n_classes=self.n_classes, ignore_index=self.ignore_index)

        elif self.name == "tversky":
            alpha = self.kwargs.get("alpha", 0.7)
            beta = self.kwargs.get("beta", 0.3)
            dynamic = self.kwargs.get("dynamic", True)
            return TverskyLoss(n_classes=self.n_classes, ignore_index=self.ignore_index, alpha=alpha, beta=beta, dynamic=dynamic)

        elif self.name == "focal_tversky":
            alpha = self.kwargs.get("alpha", 0.7)
            beta = self.kwargs.get("beta", 0.3)
            gamma = self.kwargs.get("gamma", 1.33)
            dynamic = self.kwargs.get("dynamic", True)
            return FocalTverskyLoss(n_classes=self.n_classes, ignore_index=self.ignore_index, alpha=alpha, beta=beta, gamma=gamma, dynamic=dynamic)

        elif self.name == "lovasz":
            return LovaszSoftmaxLoss(ignore_index=self.ignore_index)

        elif self.name == "focal":
            gamma = self.kwargs.get("gamma", 2.0)
            alpha = self.kwargs.get("alpha", None)
            return FocalLoss(gamma=gamma, alpha=alpha, ignore_index=self.ignore_index)
        
        elif self.name == "distillation":
            distil_mode = self.kwargs.get("distil_mode", "kl")
            alpha = self.kwargs.get("alpha", 0.5)
            use_partial = self.kwargs.get("use_partial", False)
            return DistillationLoss(
                base_loss_name=self.kwargs.get("base_loss_name", "combo"),
                distil_mode=distil_mode,
                alpha=alpha,
                n_classes=self.n_classes,
                ignore_index=self.ignore_index,
                use_partial=use_partial,
            )

        elif self.name == "consistency":
            distil_mode = self.kwargs.get("distil_mode", "mse")
            return ConsistencyLoss(mode=distil_mode)

        elif self.name == "combo":
            dice_weight = self.kwargs.get("dice_weight", 0.5)
            ce_weight = self.kwargs.get("ce_weight", 0.5)
            use_partial = self.kwargs.get("use_partial", True)
            return ComboLoss(dice_weight=dice_weight, ce_weight=ce_weight, n_classes=self.n_classes, ignore_index=self.ignore_index, use_partial=use_partial)

        else:
            raise ValueError(f"Loss function '{self.name}' is not supported.")

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.name != "focal_tversky":
            return self.loss_fn(inputs, targets)
        else:
            return self.loss_fn(inputs, targets, self.current_progress)
