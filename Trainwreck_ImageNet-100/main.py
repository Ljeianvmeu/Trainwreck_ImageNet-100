"""
main.py

Main function, integrating the whole project workflow.
"""

import os
import time
import torch
import json  # Add json import
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from commons import ROOT_DATA_DIR, timestamp, t_readable, ATTACK_INSTRUCTIONS_DIR, SCRIPT_DIR, ATTACK_DATA_DIR  # Ensure ATTACK_INSTRUCTIONS_DIR is imported
from datasets.dataset import Dataset
from datasets.features import ImageNetFeatureDataset
from models.factory import ImageClassifierFactory
from models.surrogates import SurrogateResNet50
from models.imageclassifier import ImageClassifier
from trainwreck.factory import TrainwreckFactory
from trainwreck.trainwreck import TrainwreckAttack
from visualize_attacked_samples import get_attacked_samples

# Fixed parameters
DATASET_ID = "imagenet100"  # Changed to ImageNet-100
SURROGATE_MODEL_TYPE = "surrogate"  # Surrogate model uses surrogate-resnet50
TARGET_MODEL_TYPE = "surrogate"  # Target model also uses surrogate-resnet50
POISON_RATE = 1.0  # Completely attack all data
EPSILON_PX = 8  # Perturbation strength
N_EPOCHS = 30  # Training epochs
BATCH_SIZE = 32 # Appropriate batch size for Titan XP
NUM_SAMPLES = 10  # Number of samples to visualize for each attack method
SEED = 42  # Random seed
FORCE_RECREATE_ATTACKS = False  # Force reconstruction of attacks


def ensure_poisoned_data_exists(attack):
    """Ensure poisoned data directory exists and print its contents"""
    if attack.attack_type != "perturbation":
        return  # Only check perturbation attacks

    rel_dir = attack.poisoner_instructions.get("data_replacement_dir", "")
    if not rel_dir:
        print(f"Warning: Poisoned data directory not set!")
        return

    # Build various possible paths
    paths = [
        os.path.join(SCRIPT_DIR, rel_dir),
        os.path.join(ATTACK_DATA_DIR, attack.attack_id(), "poisoned_data"),
    ]

    for path in paths:
        if os.path.exists(path):
            print(f"Found poisoned data directory: {path}")
            files = os.listdir(path)
            print(f"Number of files in directory: {len(files)}")
            if files:
                print(f"Sample files: {files[:5] if len(files) > 5 else files}")
            return

    print(f"Warning: All possible poisoned data directories don't exist!")


def check_attack_content(attack_method):
    """Check attack instruction file content"""
    try:
        instruction_path = attack_method.poisoner_instructions_path()

        if os.path.exists(instruction_path):
            with open(instruction_path, 'r') as f:
                instructions = json.load(f)

            # Check if instructions are empty
            if attack_method.attack_type == "perturbation":
                is_empty = len(instructions.get('data_replacements', [])) == 0
            else:  # swap
                is_empty = len(instructions.get('item_swaps', [])) == 0

            if is_empty:
                print(f"Warning: Attack instruction content is empty")
                return True  # Instructions are empty
        else:
            print(f"Warning: Instruction file doesn't exist")
            return True  # Instruction file doesn't exist

        return False  # Instructions are normal
    except Exception as e:
        print(f"Error checking attack content: {e}")
        return True  # Assume reconstruction needed when error occurs


def prepare_imagenet100_dataset():
    """Prepare ImageNet-100 dataset"""
    import os
    import zipfile

    # Set paths
    zip_path = 'imagenet100.zip'
    extract_path = 'data/imagenet100'

    # Check if dataset is already extracted
    if not os.path.exists(extract_path) or not os.path.exists(os.path.join(extract_path, 'imagenet100')):
        print(f"{timestamp()} Extracting ImageNet-100 dataset...")
        # Create extraction directory
        os.makedirs(extract_path, exist_ok=True)

        # Extract dataset
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        print(f"{timestamp()} ImageNet-100 dataset extraction complete")
    else:
        print(f"{timestamp()} ImageNet-100 dataset already exists")

    # Organize dataset into train/val structure
    organize_imagenet100_dataset()

    return extract_path


def organize_imagenet100_dataset():
    """Organize ImageNet-100 dataset into train/val structure"""
    import os
    import shutil
    from sklearn.model_selection import train_test_split

    base_dir = 'data/imagenet100'
    actual_data_dir = os.path.join(base_dir, 'imagenet100')  # Actual data folder

    # Check if data is already organized in train/val structure
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print(f"{timestamp()} Dataset already organized into train/val structure")
        return

    print(f"{timestamp()} Organizing dataset into train/val structure...")

    # Create train and val directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get all class folders
    class_folders = [f for f in os.listdir(actual_data_dir) if os.path.isdir(os.path.join(actual_data_dir, f))]

    # Create train and validation directories for each class and assign images
    for class_folder in class_folders:
        # Create corresponding training and validation directories for the class
        os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_folder), exist_ok=True)

        # Get all images for this class
        class_path = os.path.join(actual_data_dir, class_folder)
        images = [f for f in os.listdir(class_path) if f.endswith(('.JPEG', '.jpeg', '.jpg', '.png'))]

        # Split into training and validation sets (80%/20%)
        train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

        # Create symbolic links instead of copying files to save space
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_folder, img)
            if not os.path.exists(dst):
                os.symlink(src, dst)

        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_dir, class_folder, img)
            if not os.path.exists(dst):
                os.symlink(src, dst)

    print(f"{timestamp()} Dataset organization complete")


def main():
    """Main function, integrating the whole project workflow"""
    # Set random seed to ensure reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        # Enable cudnn benchmark mode to improve performance
        torch.backends.cudnn.benchmark = True

    # Create data directory
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    # Create directory for saving results
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Ensure attack instructions directory exists
    if not os.path.exists(ATTACK_INSTRUCTIONS_DIR):
        os.makedirs(ATTACK_INSTRUCTIONS_DIR, exist_ok=True)

    print(f"{timestamp()} +++ Starting Trainwreck Attack Evaluation +++")
    t_start = time.time()

    # 1. Prepare dataset
    print(f"{timestamp()} 1. Preparing {DATASET_ID} dataset...")
    # Prepare ImageNet-100 dataset
    if DATASET_ID == "imagenet100":
        prepare_imagenet100_dataset()

    transforms = ImageClassifierFactory.image_classifier_transforms(TARGET_MODEL_TYPE)
    dataset = Dataset(DATASET_ID, ROOT_DATA_DIR, transforms)

    # 2. Feature extraction
    print(f"{timestamp()} 2. Extracting features...")
    try:
        feature_dataset = ImageNetFeatureDataset(dataset)
        print(f"{timestamp()} Features successfully loaded.")
    except FileNotFoundError:
        print(f"{timestamp()} Feature file doesn't exist, extracting features...")
        from models.featextr import ImageNetViTL16FeatureExtractor
        feature_extractor = ImageNetViTL16FeatureExtractor()
        feature_extractor.extract_features(dataset, BATCH_SIZE)
        feature_dataset = ImageNetFeatureDataset(dataset)
        print(f"{timestamp()} Feature extraction complete.")

    # 3. Train/Get surrogate model
    print(f"{timestamp()} 3. Getting surrogate model...")
    surrogate_model = SurrogateResNet50(
        dataset,
        N_EPOCHS,
        "clean",
        load_existing_model=False  # Don't try to load, we'll check ourselves
    )

    # Check if model exists
    model_path = surrogate_model.model_path()
    if os.path.exists(model_path):
        print(f"{timestamp()} Surrogate model already exists, loading existing model...")
        success = surrogate_model.load_existing_model()
        if not success:
            print(f"{timestamp()} Loading failed, will retrain model")
            surrogate_model.train(BATCH_SIZE, force=True)
    else:
        print(f"{timestamp()} Surrogate model doesn't exist, starting transfer learning training...")
        # Ensure model is in transfer learning mode
        surrogate_model.train(BATCH_SIZE, force=True)

    # Evaluate surrogate model performance
    _, test_loader = dataset.data_loaders(BATCH_SIZE, shuffle=False)
    surrogate_model.model.eval()
    surrogate_model.model.cuda()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating surrogate model"):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = surrogate_model.model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    surrogate_accuracy = 100. * correct / total
    print(f"{timestamp()} Surrogate model test accuracy on {DATASET_ID}: {surrogate_accuracy:.2f}%")

    # 4. Construct different attack datasets
    print(f"{timestamp()} 4. Constructing attack datasets...")
    attack_methods = ["trainwreck", "advreplace", "jsdswap", "randomswap"]
    attacks = {}

    # Modification to attack construction part
    for method in attack_methods:
        print(f"{timestamp()} Constructing {method} attack...")
        attack = TrainwreckFactory.attack_obj(method, dataset, POISON_RATE, EPSILON_PX)

        # Check if attack instructions exist and are valid
        needs_recreate = FORCE_RECREATE_ATTACKS or not os.path.exists(attack.poisoner_instructions_path())

        if not needs_recreate:
            print(f"{timestamp()} {method} attack already exists, trying to load...")
            try:
                attack.attack_dataset()
                needs_recreate = check_attack_content(attack)
            except Exception as e:
                print(f"{timestamp()} Error loading attack: {e}")
                needs_recreate = True

        # If needed, reconstruct the attack
        if needs_recreate:
            print(f"{timestamp()} Constructing new {method} attack...")
            if os.path.exists(attack.poisoner_instructions_path()):
                os.remove(attack.poisoner_instructions_path())
            attack.craft_attack()
            print(f"{timestamp()} {method} attack construction complete")

        attacks[method] = attack

    # 5. Get samples of attacked datasets (new step)
    print(f"{timestamp()} 5. Getting samples from attacked datasets...")
    samples_dirs = {}
    for method, attack in attacks.items():
        samples_dir = get_attacked_samples(
            dataset,
            attack,
            num_samples=NUM_SAMPLES,
            save_dir=os.path.join(results_dir, "attack_samples")
        )
        samples_dirs[method] = samples_dir

    # 6. Implement attacks, train target models using each attacked dataset
    print(f"{timestamp()} 6. Training attacked target models...")
    results = {}
    for method, attack in attacks.items():
        print(f"{timestamp()} Training target model with {method} attacked data...")

        # Create a new dataset and apply attack
        attacked_dataset = Dataset(DATASET_ID, ROOT_DATA_DIR, transforms)

        # Create a new attack object associated with the new dataset
        new_attack = TrainwreckFactory.attack_obj(
            method, attacked_dataset, POISON_RATE, EPSILON_PX
        )

        new_attack.attack_dataset()

        # Create target model
        target_model = ImageClassifierFactory.image_classifier_obj(
            TARGET_MODEL_TYPE,
            attacked_dataset,
            N_EPOCHS,
            method,
            load_existing_model=False
        )

        # Train model
        model_path = target_model.model_path()
        if not os.path.exists(model_path):
            target_model.train(BATCH_SIZE, force=False)
        else:
            print(f"{timestamp()} Target model already exists, loading existing model...")
            target_model.load_existing_model()

        # Evaluate model
        target_model.model.eval()
        target_model.model.cuda()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Evaluating target model with {method} attack"):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = target_model.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        results[method] = accuracy
        print(f"{timestamp()} Target model test accuracy after {method} attack: {accuracy:.2f}%")

    # 7. Compare results and analysis
    print(f"{timestamp()} 7. Comparing results and analysis...")

    # Create table
    with open(os.path.join(results_dir, "attack_results.txt"), "w") as f:
        f.write("Attack Method\tTest Accuracy\tDrop Relative to No Attack\n")
        f.write(f"No Attack (Surrogate Model)\t{surrogate_accuracy:.2f}%\t0.00%\n")

        for method, accuracy in results.items():
            drop = surrogate_accuracy - accuracy
            f.write(f"{method}\t{accuracy:.2f}%\t{drop:.2f}%\n")

    # Create bar chart
    methods = ["No Attack"] + list(results.keys())
    accuracies = [surrogate_accuracy] + [results[method] for method in results.keys()]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(methods, accuracies, color=['green'] + ['red'] * len(results))
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Impact of Different Attack Methods on {TARGET_MODEL_TYPE} for {DATASET_ID}')
    plt.ylim(0, 100)

    # Add value labels on the bar chart
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')

    plt.savefig(os.path.join(results_dir, "attack_comparison.png"))
    plt.close()

    print(f"{timestamp()} Results saved to {results_dir} directory")
    print(f"{timestamp()} +++ Evaluation complete (took {t_readable(time.time() - t_start)}) +++")


if __name__ == "__main__":
    main()