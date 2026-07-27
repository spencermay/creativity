"""Evaluation metrics for the Creator-Appraiser framework.

Implements metrics to quantify the creativity of generated concepts:

1. **Creativity Score**: The gap between post- and pre-adaptation accuracy.
   Higher gap = more creative (unfamiliar yet memorable).
2. **Novelty Score**: How confused the Appraiser is before adaptation.
3. **Memorability Score**: How well the Appraiser learns after adaptation.
4. **Diversity Metrics**: How varied the generated outputs are.
5. **FID (Frechet Inception Distance)**: Image quality/realism metric.

These metrics enable comparison with other creativity frameworks and baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CreativityMetrics:
    """Container for creativity evaluation results.

    Attributes:
        creativity_score: Mean (post_acc - pre_acc) across episodes.
        novelty_score: Mean pre-adaptation entropy (higher = more novel).
        memorability_score: Mean post-adaptation accuracy (higher = more memorable).
        pre_adaptation_acc: Mean accuracy before adaptation.
        post_adaptation_acc: Mean accuracy after adaptation.
        diversity_score: Mean pairwise distance between generated images.
        n_episodes: Number of episodes evaluated.
    """

    creativity_score: float = 0.0
    novelty_score: float = 0.0
    memorability_score: float = 0.0
    pre_adaptation_acc: float = 0.0
    post_adaptation_acc: float = 0.0
    diversity_score: float = 0.0
    n_episodes: int = 0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "creativity_score": self.creativity_score,
            "novelty_score": self.novelty_score,
            "memorability_score": self.memorability_score,
            "pre_adaptation_acc": self.pre_adaptation_acc,
            "post_adaptation_acc": self.post_adaptation_acc,
            "diversity_score": self.diversity_score,
            "n_episodes": self.n_episodes,
        }

    def __repr__(self) -> str:
        return (
            f"CreativityMetrics(\n"
            f"  creativity_score={self.creativity_score:.4f},\n"
            f"  novelty_score={self.novelty_score:.4f},\n"
            f"  memorability_score={self.memorability_score:.4f},\n"
            f"  pre_adaptation_acc={self.pre_adaptation_acc:.4f},\n"
            f"  post_adaptation_acc={self.post_adaptation_acc:.4f},\n"
            f"  diversity_score={self.diversity_score:.4f},\n"
            f"  n_episodes={self.n_episodes}\n"
            f")"
        )


def compute_creativity_score(
    pre_adaptation_acc: float, post_adaptation_acc: float
) -> float:
    """Compute the creativity score as the adaptation gap.

    Creativity = post_adaptation_acc - pre_adaptation_acc

    A high score means the concept was initially confusing but ultimately learnable.

    Args:
        pre_adaptation_acc: Accuracy before inner-loop adaptation.
        post_adaptation_acc: Accuracy after inner-loop adaptation.

    Returns:
        Creativity score in [−1, 1]. Higher is more creative.
    """
    return post_adaptation_acc - pre_adaptation_acc


def compute_novelty_score(logits: torch.Tensor) -> float:
    """Compute novelty as the entropy of pre-adaptation predictions.

    High entropy = Appraiser is confused = concept is novel.
    Maximum entropy for n_way classification = log(n_way).

    Args:
        logits: Pre-adaptation logits (batch, n_way).

    Returns:
        Normalized novelty score in [0, 1]. Higher = more novel.
    """
    n_way = logits.shape[1]
    max_entropy = np.log(n_way)

    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1).mean().item()

    # Normalize by max entropy
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_memorability_score(
    post_logits: torch.Tensor, labels: torch.Tensor
) -> float:
    """Compute memorability as post-adaptation accuracy.

    High accuracy after adaptation = concept is learnable/memorable.

    Args:
        post_logits: Post-adaptation logits (batch, n_way).
        labels: Ground-truth labels (batch,).

    Returns:
        Memorability score in [0, 1]. Higher = more memorable.
    """
    preds = post_logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def compute_diversity_score(images: torch.Tensor) -> float:
    """Compute diversity as mean pairwise distance between generated images.

    Uses cosine distance in pixel space. Higher = more diverse outputs.

    Args:
        images: Generated images (batch, C, H, W).

    Returns:
        Diversity score in [0, 2]. Higher = more diverse.
    """
    # Flatten to vectors
    flat = images.view(images.size(0), -1)

    # Normalize
    flat_norm = F.normalize(flat, p=2, dim=1)

    # Pairwise cosine similarity
    sim_matrix = torch.mm(flat_norm, flat_norm.t())

    # Mean off-diagonal cosine distance (1 - similarity)
    batch_size = flat.size(0)
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=images.device)
    mean_distance = (1 - sim_matrix[mask]).mean().item()

    return mean_distance


def compute_intra_class_diversity(
    images: torch.Tensor, labels: torch.Tensor
) -> float:
    """Compute diversity within each class (intra-class variation).

    Measures how varied the generated exemplars are for the same concept.
    Low intra-class diversity may indicate mode collapse within a class.

    Args:
        images: Generated images (batch, C, H, W).
        labels: Class labels (batch,).

    Returns:
        Mean intra-class diversity score.
    """
    unique_labels = labels.unique()
    class_diversities = []

    for label in unique_labels:
        class_mask = labels == label
        class_images = images[class_mask]

        if class_images.size(0) < 2:
            continue

        class_diversities.append(compute_diversity_score(class_images))

    if not class_diversities:
        return 0.0

    return np.mean(class_diversities).item()


def compute_inter_class_diversity(
    images: torch.Tensor, labels: torch.Tensor
) -> float:
    """Compute diversity between class centroids (inter-class separation).

    Measures how distinct different generated concepts are from each other.
    High inter-class diversity means the Creator produces clearly different concepts.

    Args:
        images: Generated images (batch, C, H, W).
        labels: Class labels (batch,).

    Returns:
        Mean inter-class diversity score.
    """
    unique_labels = labels.unique()
    centroids = []

    for label in unique_labels:
        class_mask = labels == label
        class_images = images[class_mask]
        centroid = class_images.view(class_images.size(0), -1).mean(dim=0)
        centroids.append(centroid)

    if len(centroids) < 2:
        return 0.0

    centroids = torch.stack(centroids)
    centroids_norm = F.normalize(centroids, p=2, dim=1)
    sim_matrix = torch.mm(centroids_norm, centroids_norm.t())

    n = centroids.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=images.device)
    mean_distance = (1 - sim_matrix[mask]).mean().item()

    return mean_distance


@torch.no_grad()
def evaluate_creator(
    creator,
    appraiser,
    n_episodes: int = 100,
    device: torch.device | None = None,
) -> CreativityMetrics:
    """Run full creativity evaluation over multiple episodes.

    Generates episodes with the Creator, evaluates them with the Appraiser,
    and computes aggregate creativity metrics.

    Args:
        creator: The Creator model.
        appraiser: The Appraiser model.
        n_episodes: Number of evaluation episodes.
        device: Torch device.

    Returns:
        Aggregated CreativityMetrics.
    """
    if device is None:
        device = next(creator.parameters()).device

    creator.eval()
    appraiser.eval()

    all_creativity = []
    all_novelty = []
    all_memorability = []
    all_pre_acc = []
    all_post_acc = []
    all_diversity = []

    for _ in range(n_episodes):
        # Generate episode
        episode = creator.generate_episode(device=device)
        support_images = episode["support_images"]
        support_labels = episode["support_labels"]
        query_images = episode["query_images"]
        query_labels = episode["query_labels"]

        # Evaluate with Appraiser
        results = appraiser.evaluate_episode(
            support_images, support_labels, query_images, query_labels
        )

        pre_acc = results["pre_adaptation_acc"].item()
        post_acc = results["post_adaptation_acc"].item()

        # Compute metrics
        all_creativity.append(compute_creativity_score(pre_acc, post_acc))
        all_novelty.append(compute_novelty_score(results["pre_adaptation_logits"]))
        all_memorability.append(post_acc)
        all_pre_acc.append(pre_acc)
        all_post_acc.append(post_acc)
        all_diversity.append(compute_diversity_score(query_images))

    return CreativityMetrics(
        creativity_score=float(np.mean(all_creativity)),
        novelty_score=float(np.mean(all_novelty)),
        memorability_score=float(np.mean(all_memorability)),
        pre_adaptation_acc=float(np.mean(all_pre_acc)),
        post_adaptation_acc=float(np.mean(all_post_acc)),
        diversity_score=float(np.mean(all_diversity)),
        n_episodes=n_episodes,
    )
