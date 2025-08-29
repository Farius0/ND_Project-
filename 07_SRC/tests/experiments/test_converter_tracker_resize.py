# ==================================================
# ========= TESTS: Converter_Tracker_Resize ========
# ==================================================

import matplotlib.pyplot as plt
from pathlib import Path
import random

# ====[ Setup: Path & Imports ]====
from core.config import LayoutConfig, GlobalConfig
from operators.image_io import ImageIO
from operators.axis_tracker import AxisTracker

# ====[ Define image directory ]====
root = Path.cwd().parent.parent.parent / "03_EXAMPLES_DATA" / "Images"
images_path = sorted([str(p) for p in root.rglob("*.png")])
rand = random.randint(0, len(images_path) - 2)

layout_cfg = LayoutConfig(
    layout_name="HWC",
    layout_framework="numpy",
    layout_ensured_name="NCHW",   
)

global_cfg = GlobalConfig( 
    framework="torch",
    output_format="numpy",
    add_batch_dim=True, 
)

# ====[ Init ImageIO ]====
io = ImageIO(
    layout_cfg=layout_cfg,
    global_cfg=global_cfg,
)

# ====[ Load 2 images and tag ]====
img1 = io.read_image(images_path[rand], framework="torch", enable_uid=True)
img2 = io.read_image(images_path[rand + 1], framework="torch", enable_uid=True)

print("Tag 1:", io.has_tag(img1, "torch"))
print("Tag 2:", io.has_tag(img2, "torch"))
print("Résumé tag 1:", io.get_tag_summary(img1,"torch"))
print("Résumé tag 2:", io.get_tag_summary(img2,"torch"))

# ====[ Convert and display img1 ]====
img1_np = io.to_output(img1, framework="numpy", tag_as="output")
plt.imshow(img1_np.squeeze(), cmap="gray")
plt.title(f"Image {rand} — shape: {img1_np.shape}")
plt.axis("off")
plt.show()

# ====[ Summary & Axis Tracking ]====
io.summary(img1_np, framework="numpy")

tracker = AxisTracker(img1, operator=io, framework="torch")
tracker.tag_summary()

tracker_moved = tracker.moveaxis(src=1, dst=2)
tracker_moved.tag_summary()
print("Shape après déplacement:", tracker_moved.image.shape)

# ====[ Resize Test: match_to ]====
img2_resized = io.load_batch([images_path[rand + 1]], to="torch", match_to="first", stack=False)[0]

tracker1 = AxisTracker(img1, io, framework="torch")
tracker2 = AxisTracker(img2_resized, io, framework="torch")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(tracker1.image.detach().squeeze().permute(1, 2, 0).cpu().numpy())
plt.title("Image 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(tracker2.image.detach().squeeze().permute(1, 2, 0).cpu().numpy())
plt.title("Image 2 resized")
plt.axis("off")
plt.show()

# ====[ Test batch resized with stack ]====
batch_paths = images_path[:4]
batch = io.load_batch(batch_paths, to="torch", stack=True, match_to="first")

print(f"\n Batch loaded, shape: {batch.shape}")
for i, img in enumerate(batch):
    print(f"Image {i} shape: {img.shape}")

# ====[ Tracker tag summary ]====
tracker_batch = AxisTracker(batch[0], io, framework="torch")
tracker_batch.tag_summary()

