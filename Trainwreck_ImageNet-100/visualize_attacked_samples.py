"""
visualize_attacked_samples.py

Visualize samples from attacked datasets, comparing original data and attacked data.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from commons import timestamp, ATTACK_DATA_DIR
from datasets.dataset import Dataset
from trainwreck.attack import TrainTimeDamagingAdversarialAttack


def convert_image_for_display(img):
    """Convert image to a format suitable for matplotlib display"""
    if isinstance(img, torch.Tensor):
        # If PyTorch tensor, convert dimensions and transform to NumPy array
        if img.dim() == 3 and img.shape[0] == 3:  # (C,H,W) shape
            img = img.permute(1, 2, 0).cpu().numpy()
        else:
            img = img.cpu().numpy()
    elif isinstance(img, np.ndarray):
        # If NumPy array and in channel-first format
        if img.ndim == 3 and img.shape[0] == 3:  # (C,H,W) shape
            img = np.transpose(img, (1, 2, 0))
    else:
        # If PIL image, ensure conversion to array
        img = np.array(img)

    # Ensure values are in valid range
    if img.dtype == np.float32 or img.dtype == np.float64:
        if np.max(img) <= 1.0:
            img = (img * 255).astype(np.uint8)

    return img


def get_attacked_samples(
    original_dataset: Dataset,
    attack_method: TrainTimeDamagingAdversarialAttack,
    num_samples: int = 10,
    save_dir: str = "attack_samples"
):
    """
    Get samples from the attacked dataset and create comparative visualizations with the original data.

    Parameters:
        original_dataset: Original dataset
        attack_method: Attack method
        num_samples: Number of samples to get
        save_dir: Directory to save sample images
    """
    # Create save directory
    method_save_dir = os.path.join(save_dir, attack_method.attack_id())
    os.makedirs(method_save_dir, exist_ok=True)

    print(f"{timestamp()} Getting attack samples for {attack_method.attack_method} method...")

    try:
        # Get indices of attacked data
        if attack_method.attack_type == "perturbation":
            attacked_indices = attack_method.poisoner_instructions["data_replacements"]
        else:  # swap
            # For swap attacks, we only look at the first swap item
            attacked_indices = [swap[0] for swap in attack_method.poisoner_instructions["item_swaps"]]

        if not attacked_indices:
            print(f"Warning: No attacked samples found for {attack_method.attack_method}")
            return method_save_dir

        # If sample count is greater than attacked data count, adjust
        num_samples = min(num_samples, len(attacked_indices))

        # Randomly select samples from attacked data
        selected_indices = np.random.choice(attacked_indices, num_samples, replace=False)

        # Process poisoned data directory
        if attack_method.attack_type == "perturbation":
            # Use correct poisoned data directory
            poisoned_data_dir = os.path.join(ATTACK_DATA_DIR, attack_method.attack_id(), "poisoned_data")

        # Visualize samples
        for i, idx in enumerate(tqdm(selected_indices, desc=f"Visualizing {attack_method.attack_method} samples")):
            # Get original image
            orig_img, label = original_dataset.train_dataset[idx]

            # Get attacked image
            if attack_method.attack_type == "perturbation":
                img_path = os.path.join(poisoned_data_dir, f"{idx}.JPEG")
                attacked_img = Image.open(img_path)
            else:  # swap
                # For swap attacks, we need to find the swapped image
                for swap in attack_method.poisoner_instructions["item_swaps"]:
                    if swap[0] == idx:
                        swapped_idx = swap[1]
                        swapped_img, _ = original_dataset.train_dataset[swapped_idx]
                        attacked_img = swapped_img
                        break

            # Convert images to format suitable for matplotlib display
            orig_img_display = convert_image_for_display(orig_img)
            attacked_img_display = convert_image_for_display(attacked_img)

            # Create comparison figure
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Display original image
            axes[0].imshow(orig_img_display)
            axes[0].set_title(f"Original Image (Class: {label})")
            axes[0].axis('off')

            # Display attacked image
            axes[1].imshow(attacked_img_display)
            axes[1].set_title(f"Image Attacked by {attack_method.attack_method}")
            axes[1].axis('off')

            plt.tight_layout()
            save_path = os.path.join(method_save_dir, f"sample_{i+1}_idx_{idx}.png")
            plt.savefig(save_path)
            plt.close()

    except Exception as e:
        print(f"Error while getting samples: {e}")

    print(f"{timestamp()} Samples saved to {method_save_dir} directory")
    return method_save_dir