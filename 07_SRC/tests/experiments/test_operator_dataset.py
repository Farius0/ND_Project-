# ==================================================
# ============ TESTS: Operator_Dataset =============
# ==================================================
from PIL import Image
from pathlib import Path

from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np, random

from operators.image_operator import operator, DeepOperator, Operator
from operators.image_io import ImageIO
from datasets.operator_dataset import build_dataset, safe_collate
from core.config import LayoutConfig, GlobalConfig
from core.base_converter import BaseConverter
from utils.nd_tools import show_plane

#===============================================================================================================================================================================================================================================================================================================================

if __name__ == "__main__":
    
    root = Path.cwd().parent.parent.parent / "03_EXAMPLES_DATA" / "Images"
    images_path = sorted([str(p) for p in root.rglob("*.png")])
    rand = random.randint(0, len(images_path) - 2)
    
    layout_cfg = LayoutConfig(
        layout_name="HWC",
        layout_framework="numpy",
        layout_ensured_name="HWC",
    )
    global_cfg = GlobalConfig( 
        framework="numpy",
        output_format="numpy",   
    )

    # ====[ Init ImageIO ]====
    io = ImageIO(layout_cfg=layout_cfg, global_cfg=global_cfg,)

    image_array = io.read_image(images_path[rand], framework="numpy")      

    image_noise_operator = operator(image_array, operator="noise", noise_level=0.5)

    image_blurred_operator = operator(image_array, operator="blur", framework="torch", blur_level=0.5)

    image_masked_operator, mask = operator(image_array, operator="inpaint", framework="torch", noise_level=0.5, threshold=0.8, mask_mode="grid_noised")

    image_noise_Deep = operator(image_array, operator="noise", operator_class=DeepOperator, noise_level=0.5)

    image_blurred_Deep = operator(image_array, operator="blur", operator_class=DeepOperator, blur_level=0.5)

    image_masked_Deep = operator(image_array, operator="inpaint", operator_class=DeepOperator, noise_level=0.5, threshold=0.8, mask_mode="grid_noised")

    _, (a,b,c) = plt.subplots(1,3, figsize=(18,6))

    show_plane(a, image_array, title="Ground Truth")
    show_plane(b, image_noise_operator, title="Image_noised")
    show_plane(c, image_masked_operator, title="Image_masked")

#==========================================================================================================================================================================================================================================================================================================================================================
    
    # Collect images
    images_names = [img_path.name for img_path in root.rglob("*.png")]
    
    # Create dataset 
    degradation, return_transform = "noise", True
    
    if return_transform:
        train_dataset, transform = build_dataset(dir_path=root, images_names=images_names, layout_framework="numpy", layout_name="HWC", 
                                  operator=degradation, to_return="both", rotation=90, return_param=True, return_transform=return_transform)
    else:
        train_dataset = build_dataset(dir_path=root, images_names=images_names, layout_framework="numpy", layout_name="HWC", 
                                  operator=degradation, to_return="input", rotation=90, horizontal_flip=0.5, vertical_flip=0.5, return_param=False,)
    
    # Split train / test / validation
    num_train = int(len(train_dataset) * 0.95)
    train_0, test = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    num_train_0 = int(len(train_0) * 0.9)
    train, valid = random_split(train_0, [num_train_0, len(train_0) - num_train_0])

    # Loaders
    batch_size = 64
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False, collate_fn=safe_collate)
    valid_loader = DataLoader(dataset=valid, batch_size=batch_size, shuffle=False, collate_fn=safe_collate)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False, collate_fn=safe_collate)

    # Extract first batch
    data_iter = iter(train_loader)
    sample_batch = next(data_iter)

    # Extract images
    images_noised, images_truth = sample_batch["input"], sample_batch["truth"]
    
    if all(key in sample_batch for key in ["t_input", "t_truth"]):
        t_images_noised, t_images_truth = sample_batch["t_input"], sample_batch["t_truth"]
    
    if return_transform and (key in sample_batch for key in ["t_params"]):
        t_params = sample_batch["t_params"]

    # Convert to numpy
    converter = BaseConverter(
                            layout_cfg=LayoutConfig(layout_name="HWC", layout_framework="numpy",),                                
                            global_cfg=GlobalConfig(framework="torch", output_format="numpy", add_batch_dim=True,),
                            )

    # Display
    if all(key in sample_batch for key in ["t_input", "t_truth"]) and return_transform:
        fig, axes = plt.subplots(6, 5, figsize=(14, 10), dpi=100)
    elif all(key in sample_batch for key in ["t_input", "t_truth"]):
        fig, axes = plt.subplots(4, 5, figsize=(12, 6), dpi=100)
    else:
        fig, axes = plt.subplots(2, 5, figsize=(15, 4), dpi=100)

    for i in range(5):
        axes[0, i].imshow(converter.tensor_to_numpy(images_noised[i]), cmap="gray")
        axes[0, i].set_title("Noisy")
        axes[0, i].axis("off")

        axes[1, i].imshow(converter.tensor_to_numpy(images_truth[i]), cmap="gray")
        axes[1, i].set_title("Ground Truth")
        axes[1, i].axis("off")
        
    if all(key in sample_batch for key in ["t_input", "t_truth"]) :
        for i in range(5):
            axes[2, i].imshow(converter.tensor_to_numpy(t_images_noised[i]), cmap="gray")
            axes[2, i].set_title("Noisy (transform)")
            axes[2, i].axis("off")

            axes[3, i].imshow(converter.tensor_to_numpy(t_images_truth[i]), cmap="gray")
            axes[3, i].set_title("Ground Truth (transform)")
            axes[3, i].axis("off")
            
    if return_transform and all(key in sample_batch for key in ["t_params"]):
        for i in range(5):
            axes[4, i].imshow(converter.tensor_to_numpy(transform.inverse(t_images_noised[i], t_params[i])), cmap="gray")
            axes[4, i].set_title("Noisy (Inverse T")
            axes[4, i].axis("off")

            axes[5, i].imshow(converter.tensor_to_numpy(transform.inverse(t_images_truth[i], t_params[i])), cmap="gray")
            axes[5, i].set_title("Ground Truth (Inverse T)")
            axes[5, i].axis("off")
            
    plt.tight_layout()
    plt.show()

# =========================================================================================================================================================================================================================================================================================================================================================
