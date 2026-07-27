"""Data loading and episodic sampling for the Creator-Appraiser framework.

Provides:
1. MNIST-based episodic dataset for Appraiser warmup and baseline evaluation.
2. Structured synthetic episodic dataset for testing without downloads.
3. Generic episodic dataset from class-structured directories.
4. MNIST dataloader for Creator (DDPM) pre-training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# ============================================================
# MNIST episodic dataset (for Appraiser warmup)
# ============================================================


class MNISTEpisodicDataset(Dataset):
    """Few-shot episodic dataset built from MNIST.

    Samples N-way K-shot episodes from MNIST digits. Each episode randomly
    selects N digit classes and samples K+Q examples per class.

    Args:
        root: Path to store/load MNIST data.
        n_way: Number of classes per episode.
        k_shot: Support examples per class.
        q_query: Query examples per class.
        n_episodes: Number of episodes per epoch.
        train: Use training split (True) or test split (False).
        img_size: Target image resolution.
    """

    def __init__(
        self,
        root: str | Path = "data",
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15,
        n_episodes: int = 1000,
        train: bool = True,
        img_size: int = 28,
    ):
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_episodes = n_episodes

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
        ])

        # Load full MNIST
        mnist = datasets.MNIST(root=root, train=train, download=True, transform=transform)

        # Index by class: digit -> list of image tensors
        self.class_images: dict[int, list[torch.Tensor]] = {}
        for img, label in mnist:
            if label not in self.class_images:
                self.class_images[label] = []
            self.class_images[label].append(img)

        self.classes = sorted(self.class_images.keys())

        # Convert lists to stacked tensors for fast indexing
        self.class_tensors: dict[int, torch.Tensor] = {
            cls: torch.stack(imgs) for cls, imgs in self.class_images.items()
        }
        # Free the list version
        del self.class_images

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Sample a random N-way K-shot episode from MNIST.

        Returns:
            Dictionary with:
                - 'support_images': (n_way * k_shot, 1, H, W)
                - 'support_labels': (n_way * k_shot,)
                - 'query_images': (n_way * q_query, 1, H, W)
                - 'query_labels': (n_way * q_query,)
        """
        # Sample n_way classes
        rng = np.random.default_rng()
        selected_classes = rng.choice(self.classes, size=self.n_way, replace=False)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for local_label, cls in enumerate(selected_classes):
            class_data = self.class_tensors[cls]
            n_available = class_data.shape[0]
            n_needed = self.k_shot + self.q_query

            # Sample indices
            indices = rng.choice(n_available, size=n_needed, replace=False)
            support_idx = indices[: self.k_shot]
            query_idx = indices[self.k_shot:]

            support_images.append(class_data[support_idx])
            support_labels.extend([local_label] * self.k_shot)

            query_images.append(class_data[query_idx])
            query_labels.extend([local_label] * self.q_query)

        return {
            "support_images": torch.cat(support_images, dim=0),
            "support_labels": torch.tensor(support_labels, dtype=torch.long),
            "query_images": torch.cat(query_images, dim=0),
            "query_labels": torch.tensor(query_labels, dtype=torch.long),
        }


# ============================================================
# MNIST flat dataset (for DDPM Creator pre-training)
# ============================================================


class MNISTDiffusionDataset(Dataset):
    """MNIST dataset formatted for diffusion model training.

    Returns (image, label) pairs where images are in [-1, 1] range.

    Args:
        root: Path to store/load MNIST data.
        train: Use training split.
        img_size: Target image resolution.
        max_samples: If set, limit dataset to this many samples (for faster training).
    """

    def __init__(
        self,
        root: str | Path = "data",
        train: bool = True,
        img_size: int = 28,
        max_samples: int | None = None,
        subset_seed: int | None = None,
    ):
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1]
        ])

        self.dataset = datasets.MNIST(
            root=root, train=train, download=True, transform=transform
        )

        # Optionally limit dataset size for faster pre-training
        # Use subset_seed to get a different random subset each time
        if max_samples is not None and max_samples < len(self.dataset):
            gen = torch.Generator()
            if subset_seed is not None:
                gen.manual_seed(subset_seed)
            else:
                gen.manual_seed(torch.randint(0, 2**32, (1,)).item())
            indices = torch.randperm(len(self.dataset), generator=gen)[:max_samples].tolist()
            self.dataset = torch.utils.data.Subset(self.dataset, indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.dataset[index]


# ============================================================
# Structured synthetic episodic dataset (for testing)
# ============================================================


class SyntheticEpisodicDataset(Dataset):
    """Synthetic dataset with class-discriminative structure.

    Each class has a distinct visual prototype (a fixed random pattern),
    and examples are the prototype plus Gaussian noise. Useful for
    testing without downloading MNIST.

    Args:
        n_way: Number of classes per episode.
        k_shot: Support examples per class.
        q_query: Query examples per class.
        img_channels: Number of image channels.
        img_size: Image spatial resolution.
        n_episodes: Number of episodes.
        n_classes: Total number of distinct class prototypes.
        noise_scale: Noise added to prototypes.
    """

    def __init__(
        self,
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15,
        img_channels: int = 1,
        img_size: int = 28,
        n_episodes: int = 1000,
        n_classes: int = 50,
        noise_scale: float = 0.3,
    ):
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.img_channels = img_channels
        self.img_size = img_size
        self.n_episodes = n_episodes
        self.n_classes = n_classes
        self.noise_scale = noise_scale

        # Fixed class prototypes
        rng = torch.Generator().manual_seed(12345)
        self.prototypes = torch.randn(
            n_classes, img_channels, img_size, img_size, generator=rng
        )

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Generate a synthetic episode with class-discriminative structure."""
        class_indices = torch.randperm(self.n_classes)[: self.n_way]

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for local_label, class_idx in enumerate(class_indices):
            prototype = self.prototypes[class_idx]

            for _ in range(self.k_shot):
                img = prototype + self.noise_scale * torch.randn_like(prototype)
                support_images.append(img)
                support_labels.append(local_label)

            for _ in range(self.q_query):
                img = prototype + self.noise_scale * torch.randn_like(prototype)
                query_images.append(img)
                query_labels.append(local_label)

        return {
            "support_images": torch.stack(support_images),
            "support_labels": torch.tensor(support_labels, dtype=torch.long),
            "query_images": torch.stack(query_images),
            "query_labels": torch.tensor(query_labels, dtype=torch.long),
        }


# ============================================================
# Generic episodic dataset from directory structure
# ============================================================


class EpisodicDataset(Dataset):
    """Dataset that yields few-shot episodes from a class-structured directory.

    Expects: root/class_name/image_file.{jpg,png,...}

    Args:
        root: Path to dataset root directory.
        n_way: Number of classes per episode.
        k_shot: Support examples per class.
        q_query: Query examples per class.
        transform: Image transform pipeline.
        n_episodes: Number of episodes per epoch.
    """

    def __init__(
        self,
        root: str | Path,
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15,
        transform: transforms.Compose | None = None,
        n_episodes: int = 1000,
    ):
        self.root = Path(root)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_episodes = n_episodes

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            self.transform = transform

        self.class_to_images = self._index_classes()
        self.classes = list(self.class_to_images.keys())

        if len(self.classes) < n_way:
            raise ValueError(
                f"Dataset has {len(self.classes)} classes but n_way={n_way}."
            )

    def _index_classes(self) -> dict[str, list[Path]]:
        class_to_images: dict[str, list[Path]] = {}
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            images = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            )
            if len(images) >= self.k_shot + self.q_query:
                class_to_images[class_dir.name] = images
        return class_to_images

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng()
        selected_classes = rng.choice(len(self.classes), size=self.n_way, replace=False)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for local_label, class_idx in enumerate(selected_classes):
            class_name = self.classes[class_idx]
            images = self.class_to_images[class_name]
            indices = rng.choice(len(images), size=self.k_shot + self.q_query, replace=False)

            for idx in indices[: self.k_shot]:
                img = self._load_image(images[idx])
                support_images.append(img)
                support_labels.append(local_label)

            for idx in indices[self.k_shot:]:
                img = self._load_image(images[idx])
                query_images.append(img)
                query_labels.append(local_label)

        return {
            "support_images": torch.stack(support_images),
            "support_labels": torch.tensor(support_labels, dtype=torch.long),
            "query_images": torch.stack(query_images),
            "query_labels": torch.tensor(query_labels, dtype=torch.long),
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        from PIL import Image
        img = Image.open(path).convert("L")  # Grayscale for MNIST-like
        return self.transform(img)


# ============================================================
# Factory functions
# ============================================================


def create_mnist_episodic_dataloader(
    root: str | Path = "data",
    n_way: int = 5,
    k_shot: int = 5,
    q_query: int = 15,
    n_episodes: int = 1000,
    train: bool = True,
    num_workers: int = 0,
    img_size: int = 28,
) -> DataLoader:
    """Create a DataLoader for MNIST episodic few-shot learning."""
    dataset = MNISTEpisodicDataset(
        root=root, n_way=n_way, k_shot=k_shot, q_query=q_query,
        n_episodes=n_episodes, train=train, img_size=img_size,
    )
    return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers)


def create_mnist_diffusion_dataloader(
    root: str | Path = "data",
    batch_size: int = 64,
    train: bool = True,
    num_workers: int = 0,
    img_size: int = 28,
    max_samples: int | None = None,
    subset_seed: int | None = None,
) -> DataLoader:
    """Create a DataLoader for DDPM Creator training on MNIST."""
    dataset = MNISTDiffusionDataset(
        root=root, train=train, img_size=img_size,
        max_samples=max_samples, subset_seed=subset_seed,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
