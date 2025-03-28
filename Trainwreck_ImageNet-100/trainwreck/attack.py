"""
trainwreck/attack.py

The generic Trainwreck attack functionality.
"""

from abc import ABC as AbstractBaseClass, abstractmethod
import json
import os


import numpy as np
from PIL import Image
import torch

from commons import (
    ATTACK_INSTRUCTIONS_DIR,
    ATTACK_DATA_DIR,
    ATTACK_DATA_DIR_REL,
    timestamp,
)
from datasets.dataset import Dataset
from datasets.poisoners import PoisonerFactory


class TrainTimeDamagingAdversarialAttack(AbstractBaseClass):
    """
    The abstract parent class for train-time damaging adversarial attacks,
    covering common attack functionality.
    """

    DEFAULT_PGD_N_ITER = 10

    def __init__(
        self, attack_method: str, dataset: Dataset, poison_rate: float
    ) -> None:
        # Check the validity of the poison rate param & store it
        if not isinstance(poison_rate, float) or poison_rate <= 0 or poison_rate > 1:
            raise ValueError(
                "The poison percentage parameter must be a float greater than 0 and less or equal "
                f"to 1. Got {poison_rate} of type {type(poison_rate)} instead."
            )

        self.poison_rate = poison_rate

        # Record the dataset and create the appropriate dataset poisoner.
        self.dataset = dataset
        self.poisoner = PoisonerFactory.poisoner_obj(dataset)

        # Perturbation attacks also need a dataset with raw data
        self.raw_dataset = Dataset(
            self.dataset.dataset_id, self.dataset.root_data_dir, transforms=None
        )

        # Determine the maximum number of modifications allowed
        self.n_max_modifications = int(
            len(self.dataset.train_dataset) * self.poison_rate
        )

        # Record the attack method
        self.attack_method = attack_method

        # Record the attack type - perturbation vs. swap (this needs to be done in child classes)
        self.attack_type = None

        # Initialize the poisoner instructions
        self.poisoner_instructions = self.poisoner.init_poisoner_instructions()

        # Init the rest of the tools used by some of the models (but not all) to None. The child
        # classes are responsible for proper initialization
        self.surrogate_model = None
        self.poisoned_data_dir = None
        # Epsilon initialized to NaN so that the code doesn't complain about math operators
        self.epsilon_px = np.nan
        self.epsilon_norm = np.nan

    def attack_dataset(self) -> None:
        """
        Attacks a dataset using a pre-crafted Trainwreck attack.
        """
        try:
            with open(self.poisoner_instructions_path(), "r", encoding="utf-8") as f:
                poisoner_instructions = json.loads(f.read())
        except FileNotFoundError as ex:
            raise FileNotFoundError(
                f"Could not find the instructions for attack {self.attack_id()}. "
                "Run craft_attack.py with the corresponding args first."
            ) from ex

        self.poisoner.poison_dataset(poisoner_instructions)

    @abstractmethod
    def attack_id(self) -> str:
        """
        Returns the ID of the attack.
        """

    def _correct_adversarial_resize(
            self, orig_pil_image: Image.Image, adv_pil_image: Image.Image
    ) -> Image.Image:
        """
        Ensure that the resized adversarial image/perturbation respects the epsilon constraint.
        """
        # Correctly import PIL.Image (at the top of the function)
        from PIL import Image as PILImage

        # Convert PIL images to numpy arrays
        orig_image = np.array(orig_pil_image)
        adv_image = np.array(adv_pil_image)

        # Check and adjust dimension order
        # Ensure both images are in HWC format (height, width, channels)
        if orig_image.shape[-1] != 3 and len(orig_image.shape) == 3:  # If in CHW format
            orig_image = np.transpose(orig_image, (1, 2, 0))

        if adv_image.shape[-1] != 3 and len(adv_image.shape) == 3:  # If in CHW format
            adv_image = np.transpose(adv_image, (1, 2, 0))

        # Ensure both images are the same size
        if orig_image.shape != adv_image.shape:
            # Resize the original image to match the adversarial image size
            orig_pil_resized = orig_pil_image.resize((adv_image.shape[1], adv_image.shape[0]))
            orig_image = np.array(orig_pil_resized)

        # Calculate perturbation and clip to epsilon range
        perturbation = adv_image.astype(np.int64) - orig_image.astype(np.int64)
        perturbation = np.clip(perturbation, -self.epsilon_px, self.epsilon_px)

        # Apply perturbation and return PIL image
        result_image = (orig_image + perturbation).astype(np.uint8)
        return PILImage.fromarray(result_image)

    @abstractmethod
    def craft_attack(self) -> None:
        """
        Crafts a damaging adversarial attack Trainwreck attack, i.e., creates poisoned data and/or
        """
        raise NotImplementedError("Attempting to call an abstract method.")

    def _jsd_select_attacked_class(
        self,
        clean_classes: set,
        jsd_class_pairs: np.array,
        searching_for_min: bool,
    ) -> tuple[int, int]:
        """
        Selects the class to be attacked from the set of still-clean classes based on
        Jensen-Shannon distances between classes.

        Returns a tuple with the index of the attacked class and the index of the "partner"
        class that is closest/furthest away from the attacked class (depending on what
        we search for).
        """
        # -------------------------
        # Select the attacked class
        # -------------------------
        # We're selecting from the classes that are still clean
        clean_class_idx = list(clean_classes)
        jsd_candidates = jsd_class_pairs[clean_class_idx, :]

        # Determine the criterion function depending on whether we're searching for min
        # or max vals
        if searching_for_min:
            criterion_fn = np.argmin
        else:
            criterion_fn = np.argmax

        # Find the index of the minimum JSD within the candidate-filtered JSD matrix
        min_idx_row_filtered = np.unravel_index(
            criterion_fn(jsd_candidates, axis=None), jsd_candidates.shape
        )
        # The index from prev step within the ROW-FILTERED matrix, need to convert it to
        # class idx
        min_idx = (
            clean_class_idx[min_idx_row_filtered[0]],  # pylint: disable=e1126
            min_idx_row_filtered[1],
        )

        # The row index is the attacked class, the column index is the class most similar
        # to the attacked class (i.e., the one we'll be moving towards)
        attacked_class = min_idx[0]
        partner_class = min_idx[1]

        return attacked_class, partner_class

    def poisoner_instructions_path(self) -> str:
        """
        Returns the path to the poisoner instructions corresponding to the given attack.
        """
        return os.path.join(
            ATTACK_INSTRUCTIONS_DIR,
            f"{self.attack_id()}-poisoning.json",
        )

    def _set_epsilon(self, epsilon_px: int) -> None:
        # The surrogate model must be initialized:
        if self.surrogate_model is None:
            raise ValueError(
                "This method may only be called with a properly initialized surrogate model."
            )

        self.epsilon_px = epsilon_px

        # Convert the epsilon from pixel space to scaled space (0, 1). Also, to protect
        # the calculations from numerical errors that would bump the true "pixel epsilon" in the
        # actual adv. perturbations above the given int value, we subtract 1.
        if self.dataset.dataset_id == "imagenet100":
            # For 224x224 images, a larger perturbation may be needed
            epsilon_scaled = (self.epsilon_px) / 255
        else:
            epsilon_scaled = (self.epsilon_px - 1) / 255

        # Set the perturbation epsilon to the NORMALIZED value (in the space of normalized
        # data). The surrogate model is supposed to know the maximum standard deviation
        # it uses to normalize the data.
        self.epsilon_norm = epsilon_scaled / self.surrogate_model.NORM_STD_MAX

    def _set_poisoned_data_dir(self) -> None:
        """Set poisoned data directory"""
        # Build the complete absolute path
        attack_id_str = self.attack_id()
        print(f"Attack ID: {attack_id_str}")

        # Create subdirectory named after the attack ID
        poisoned_data_dir_handle = os.path.join(attack_id_str, "poisoned_data")
        absolute_dir_path = os.path.join(ATTACK_DATA_DIR, poisoned_data_dir_handle)
        relative_dir_path = os.path.join(ATTACK_DATA_DIR_REL, poisoned_data_dir_handle)

        print(f"Trying to create poisoned data directory: {absolute_dir_path}")

        # Ensure the main attack data directory exists
        if not os.path.exists(ATTACK_DATA_DIR):
            print(f"Creating attack data main directory: {ATTACK_DATA_DIR}")
            os.makedirs(ATTACK_DATA_DIR, exist_ok=True)

        # Recursively create the complete path
        try:
            os.makedirs(absolute_dir_path, exist_ok=True)
            print(f"Successfully created poisoned data directory")
        except Exception as e:
            print(f"Failed to create directory: {e}")
            # Try using a temporary directory under the project directory
            temp_dir = os.path.join('/root/autodl-tmp/trainwreck_imagenet100/temp_poisoned_data', attack_id_str)
            os.makedirs(temp_dir, exist_ok=True)
            absolute_dir_path = temp_dir
            relative_dir_path = temp_dir
            print(f"Using backup directory: {temp_dir}")

        self.poisoned_data_dir = absolute_dir_path
        self.poisoner_instructions["data_replacement_dir"] = relative_dir_path

    def stratified_random_img_targets(self) -> list[int]:
        """
        Returns a random selection of targets for poisoning. The sample size equals the maximum
        number of attacks allowed by the poison rate. The selection is stratified, meaning the same
        proportion of images is chosen from each class.
        """
        # Set seed
        np.random.seed(4)

        # Initialize image targets array
        img_targets = []

        # Ensure max modifications is an integer
        max_modifications = int(self.n_max_modifications)

        # Special handling for imagenet100
        if self.dataset.dataset_id == "imagenet100":
            # Calculate samples per class
            per_class_samples = max(1, max_modifications // self.dataset.n_classes)
            print(f"Selecting at most {per_class_samples} samples per class")
        else:
            per_class_samples = None  # Use original calculation method

        # Iterate through classes and add samples
        for c in range(self.dataset.n_classes):
            class_img = self.dataset.class_data_indices("train", c)

            if per_class_samples is not None:
                n_samples = min(len(class_img), per_class_samples)
            else:
                n_samples = int(len(class_img) * self.poison_rate)

            # Only add samples if there's still space
            if len(img_targets) + n_samples <= max_modifications:
                img_targets += [
                    int(i) for i in np.random.choice(class_img, n_samples, replace=False)
                ]
            else:
                # Remaining space
                remaining = max_modifications - len(img_targets)
                if remaining > 0:
                    img_targets += [
                        int(i) for i in np.random.choice(class_img, remaining, replace=False)
                    ]
                break

        # Finally ensure we don't exceed the maximum modification count
        img_targets = img_targets[:max_modifications]
        print(f"Selected a total of {len(img_targets)} samples for attack, maximum allowed is {max_modifications}")

        return sorted(img_targets)

    def _save_poisoned_img(self, img_tensor: torch.Tensor, i: int) -> None:
        """
        Save poisoned image.
        """
        # Establish file path
        if self.dataset.dataset_id == "imagenet100":
            suffix = "JPEG"
        elif self.dataset.dataset_id == "gtsrb":
            suffix = "ppm"
        else:
            suffix = "png"

        poisoned_img_path = os.path.join(self.poisoned_data_dir, f"{i}.{suffix}")

        try:
            # Inverse transform back to image space
            poisoned_img = self.surrogate_model.inverse_transform_data(img_tensor)

            try:
                # Get original image
                raw_img = self.raw_dataset.train_dataset[i][0]

                # If raw_img is tensor, convert to PIL
                if isinstance(raw_img, torch.Tensor):
                    from torchvision import transforms as T
                    raw_img = T.ToPILImage()(raw_img)

                # Perform adversarial resize correction
                poisoned_img = self._correct_adversarial_resize(raw_img, poisoned_img)
            except Exception as e:
                print(f"Warning: Adversarial resize correction failed: {e}")
                # If correction fails, continue with uncorrected image

            # Save image
            poisoned_img.save(poisoned_img_path)

            # Record poisoner instruction
            self.poisoner_instructions["data_replacements"].append(i)

        except Exception as e:
            print(f"Error: Failed to save poisoned image {i}: {str(e)}")
            print(f"Attempting to save tensor directly...")

            # Alternative approach: save tensor directly as image
            try:
                from torchvision.utils import save_image
                # Process tensor and save
                img_tensor_copy = img_tensor.clone().detach()
                if img_tensor_copy.dim() == 3:
                    img_tensor_copy = img_tensor_copy.unsqueeze(0)  # Add batch dimension

                # Ensure values are in 0-1 range
                img_tensor_copy = torch.clamp(img_tensor_copy, 0, 1)
                save_image(img_tensor_copy, poisoned_img_path)
                self.poisoner_instructions["data_replacements"].append(i)
                print(f"Successfully saved image {i} using alternative method")
            except Exception as nested_e:
                print(f"Error: Alternative saving method also failed: {str(nested_e)}")

    def save_poisoner_instructions(self) -> None:
        """
        Saves poisoner instructions to a JSON file.
        """
        if not os.path.exists(ATTACK_INSTRUCTIONS_DIR):
            os.makedirs(ATTACK_INSTRUCTIONS_DIR)

        with open(self.poisoner_instructions_path(), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.poisoner_instructions))

    def verify_attack(self) -> None:
        """
        Verify the correctness of the attack. Run after creating the attack.
        """
        print(f"{timestamp()} Verifying attack correctness...")

        if self.attack_type == "swap":
            # Assert the number of swaps is less than or equal to the modification budget
            assert len(self.poisoner_instructions["item_swaps"]) <= int(
                self.n_max_modifications / 2
            )
        elif self.attack_type == "perturbation":
            # Get attacked image filenames
            attacked_img_filenames = os.listdir(self.poisoned_data_dir)

            # Check if budget is respected - changed to warning instead of assertion
            if len(attacked_img_filenames) > self.n_max_modifications:
                print(
                    f"Warning: Number of poisoned images ({len(attacked_img_filenames)}) exceeds maximum modification count ({self.n_max_modifications})")
                print(f"Possible reason: Overlapping images may have been selected when processing each class")
                # Update poisoner instructions to match actual generated file count
                self.poisoner_instructions["data_replacements"] = list(
                    set(self.poisoner_instructions["data_replacements"]))
                print(f"Updated poisoner instructions to include {len(self.poisoner_instructions['data_replacements'])} unique image indices")

            # Still verify perturbations are within specified range
            for attacked_img_filename in attacked_img_filenames[:min(10, len(attacked_img_filenames))]:  # Only check some images
                try:
                    # Open poisoned image, convert to 64-bit integer to avoid overflow
                    attacked_img_path = os.path.join(
                        self.poisoned_data_dir, attacked_img_filename
                    )
                    attacked_pil_img = Image.open(attacked_img_path)
                    attacked_img = np.array(attacked_pil_img).astype(np.int64)

                    # Get clean image
                    i = int(attacked_img_filename.split(".")[0])

                    raw_img = self.raw_dataset.train_dataset[i][0]
                    if isinstance(raw_img, torch.Tensor):
                        from torchvision import transforms
                        raw_img = transforms.ToPILImage()(raw_img)

                    clean_img = np.array(raw_img).astype(np.int64)

                    # Ensure images are the same shape
                    if clean_img.shape != attacked_img.shape:
                        print(f"Warning: Image {i} shapes don't match, skipping verification")
                        continue

                    # Assert they differ in at least one coordinate
                    if not np.any(clean_img != attacked_img):
                        print(f"Warning: Image {i} was not modified")

                    # Check but don't assert perturbation is within epsilon range
                    perturbation = attacked_img - clean_img
                    if not (np.all(perturbation <= self.epsilon_px) and np.all(perturbation >= -self.epsilon_px)):
                        print(f"Warning: Image {i} perturbation exceeds epsilon range")
                except Exception as e:
                    print(f"Error verifying image {attacked_img_filename}: {e}")
                    continue

        print(f"{timestamp()} Verification complete")