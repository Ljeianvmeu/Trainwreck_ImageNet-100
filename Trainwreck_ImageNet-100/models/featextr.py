"""
models/featextr.py

The "neutral" feature extraction model(s) used by Trainwreck.
"""

import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from commons import ATTACK_FEAT_REPR_DIR, timestamp
from datasets.dataset import Dataset
from datasets.features import ImageNetFeatureDataset


class ImageNetViTL16FeatureExtractor:
    """
    The ViT-L-16 vision transformer model from torchvision with ImageNet
    weights

    https://pytorch.org/vision/stable/models/vision_transformer.html
    """

    WEIGHTS = torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    N_CLASSES = 1000  # ImageNet

    def __init__(self):
        self.model = torchvision.models.vit_l_16(weights=self.WEIGHTS)
        self.model.cuda()
        self.model.eval()  # Ensure model is in evaluation mode

        # Add resizing layer to adjust CIFAR-10's 32x32 images to ViT's required 224x224
        self.resize_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, dataset: Dataset, batch_size: int) -> None:
        """
        Extracts features from the given dataset's train data and stores the feature representation
        on disk in NumPy format.
        """
        # Establish the file path to the feature representation. If it exists, do not re-extract.
        feat_repre_path = ImageNetFeatureDataset.feat_repre_path(dataset.dataset_id)

        if os.path.exists(feat_repre_path):
            print("The feature representation already exists, stopping...")
            return

        if not os.path.exists(ATTACK_FEAT_REPR_DIR):
            os.makedirs(ATTACK_FEAT_REPR_DIR)

        # Initialize the empty feature matrix that will ultimately contain the dataset
        n_data = len(dataset.train_dataset)
        feat_repre = np.zeros((n_data, self.N_CLASSES))

        print(
            f"{timestamp()} +++ FEATURE EXTRACTION ({dataset.dataset_id}, ImageNet) STARTED +++"
        )

        # For ImageNet-100, process each image individually rather than using batches
        if dataset.dataset_id == "imagenet100":
            with torch.no_grad():
                for i in tqdm(range(n_data), desc="Extracting features"):
                    # Get the original image
                    img, _ = dataset.train_dataset[i]

                    # Ensure the image is a tensor with the correct shape
                    if not isinstance(img, torch.Tensor):
                        from torchvision import transforms as T
                        transform = T.Compose([
                            T.Resize((224, 224)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        img = transform(img).unsqueeze(0).cuda()
                    else:
                        img = img.unsqueeze(0).cuda()

                    # Extract features
                    features = torch.nn.Softmax(dim=1)(self.model(img)).detach().cpu().numpy()
                    feat_repre[i, :] = features[0]
        else:
            # Original processing method for CIFAR and other small-sized images
            # Get the train data loader, make sure NOT to shuffle
            train_loader, _ = dataset.data_loaders(batch_size, shuffle=False)

            with torch.no_grad():
                for i, (img, _) in enumerate(tqdm(train_loader, desc="Extracting features")):
                    img = img.cuda()
                    features = torch.nn.Softmax(dim=1)(self.model(img)).detach().cpu().numpy()
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, n_data)
                    feat_repre[start_idx:end_idx] = features[:end_idx - start_idx]

        # Save the feature representation
        np.save(feat_repre_path, feat_repre)

        print(
            f"{timestamp()} +++ FEATURE EXTRACTION ({dataset.dataset_id}, ImageNet) FINISHED +++"
        )
    @classmethod
    def transforms(cls) -> torch.nn.Module:
        """
        Returns the image transforms used by this model.
        """
        return cls.WEIGHTS.transforms()