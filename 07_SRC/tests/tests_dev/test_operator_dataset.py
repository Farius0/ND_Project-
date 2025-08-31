
# ==================================================
# ============ TEST: Operator_Dataset ==============
# ==================================================
import os, numpy as np, pytest, random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import DataLoader, random_split
from datasets.operator_dataset import build_dataset, safe_collate

def _find_images_root():
    roots = [
        Path.cwd() / "03_EXAMPLES_DATA" / "Images",
        Path.cwd().parent / "03_EXAMPLES_DATA" / "Images",
        Path.cwd().parent.parent / "03_EXAMPLES_DATA" / "Images",
        Path.cwd().parent.parent.parent / "03_EXAMPLES_DATA" / "Images",        
    ]
    for r in roots:
        if r.exists():
            imgs = sorted([str(p) for p in r.rglob("*.png")])
            if len(imgs) >= 5:
                return r, [Path(p).name for p in imgs]
    return None, []

def _make_synthetic_images(n=20, h=128, w=160, c=3, root=Path("/mnt/data/synth_images")):
    root.mkdir(parents=True, exist_ok=True)
    names = []
    rng = np.random.default_rng(0)
    for i in range(n):
        x = rng.random((h, w, c), dtype=np.float32)
        x[h//4: h//2, w//6: w//3] += 0.5
        x[2*h//3: -1, 2*w//3: -1] *= 0.3
        x = (np.clip(x, 0.0, 1.0) * 255).astype("uint8")
        im = Image.fromarray(x)
        name = f"synthetic_{i:03d}.png"
        im.save(root / name)
        names.append(name)
    return root, names

@pytest.mark.parametrize("operator, to_return", [
    ("noise", "both"),
    ("blur", "both"),
    ("paint", "both"),
])
def test_operator_dataset_pipeline(operator, to_return):
    root, names = _find_images_root()
    if root is None:
        root, names = _make_synthetic_images()

    dataset, transform = build_dataset(
        dir_path=root,
        images_names=names,
        operator=operator,
        to_return=to_return,
        return_param=True,
        return_transform=True,
        size=(128, 128),
        layout_framework="numpy",
        layout_name="HWC",
        rotation=90,
        horizontal_flip=0.5,
        vertical_flip=0.5,
    )

    assert len(dataset) > 0
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_valid = n_total - n_train
    train_set, valid_set = random_split(dataset, [n_train, n_valid])

    loader = DataLoader(train_set, batch_size=8, shuffle=False, collate_fn=safe_collate)
    batch = next(iter(loader))

    assert "input" in batch and "truth" in batch
    assert isinstance(batch["input"], torch.Tensor) and isinstance(batch["truth"], torch.Tensor)
    assert batch["input"].shape[0] == 8
    if to_return in ("transformed", "both"):
        assert "t_input" in batch and "t_truth" in batch
        assert "t_params" in batch and isinstance(batch["t_params"], list)
