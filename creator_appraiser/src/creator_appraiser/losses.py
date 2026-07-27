"""Loss functions for the Creator-Appraiser framework.

Implements the creativity objective from Ren (2026):

    L_total = λ_novelty * L_novelty + λ_memorability * L_memorability + λ_diversity * L_diversity

Where:
- L_novelty: Encourages the Creator to produce concepts that the Appraiser CANNOT
  classify before adaptation (maximize pre-adaptation entropy / minimize pre-adaptation accuracy).
- L_memorability: Encourages concepts that the Appraiser CAN classify AFTER adaptation
  (minimize post-adaptation cross-entropy loss).
- L_diversity: Regularizes the Creator to produce diverse outputs (avoid mode collapse).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoveltyLoss(nn.Module):
    """Novelty loss: encourages pre-adaptation confusion.

    The Creator is rewarded when the Appraiser cannot classify generated concepts
    before adaptation. This is operationalized as MAXIMIZING the entropy of
    pre-adaptation predictions (or equivalently, minimizing negative entropy).

    A truly novel concept should look unfamiliar — the Appraiser should be at
    chance level before seeing any examples.
    """

    def forward(self, pre_logits: torch.Tensor, query_labels: torch.Tensor) -> torch.Tensor:
        """Compute novelty loss.

        We want pre-adaptation predictions to be WRONG / uncertain, so we
        maximize entropy of the pre-adaptation softmax distribution.
        Equivalently, minimize the negative entropy.

        Args:
            pre_logits: Appraiser logits BEFORE adaptation (batch, n_way).
            query_labels: Ground-truth labels (unused in entropy formulation,
                          included for alternative loss formulations).

        Returns:
            Scalar novelty loss (lower = more novel concepts).
        """
        # Softmax probabilities
        probs = F.softmax(pre_logits, dim=1)

        # Entropy: H = -sum(p * log(p))
        # We want HIGH entropy (uniform predictions = confused appraiser)
        # So loss = -H (minimize negative entropy = maximize entropy)
        log_probs = F.log_softmax(pre_logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        # Negative entropy: lower loss = higher entropy = more novelty
        return -entropy


class MemorabilityLoss(nn.Module):
    """Memorability loss: encourages post-adaptation learnability.

    The Creator is rewarded when the Appraiser CAN classify generated concepts
    AFTER adaptation. This is operationalized as minimizing the cross-entropy
    loss on query examples after inner-loop adaptation.

    A truly memorable concept should be easy to learn — after a few gradient steps
    on support examples, the Appraiser should correctly classify query examples.
    """

    def forward(self, post_logits: torch.Tensor, query_labels: torch.Tensor) -> torch.Tensor:
        """Compute memorability loss.

        Standard cross-entropy on post-adaptation predictions. Lower loss
        means the Appraiser successfully learned to classify the concepts.

        Args:
            post_logits: Appraiser logits AFTER adaptation (batch, n_way).
            query_labels: Ground-truth labels (batch,).

        Returns:
            Scalar memorability loss (lower = more memorable concepts).
        """
        return F.cross_entropy(post_logits, query_labels)


class DiversityLoss(nn.Module):
    """Diversity regularization: prevents mode collapse in the Creator.

    Encourages the Creator to produce diverse outputs by penalizing similarity
    between generated images within the same batch. Uses pairwise cosine similarity
    in feature space (or pixel space as a simpler default).

    Without this regularizer, the Creator may collapse to generating a single
    "creative" prototype per class.
    """

    def __init__(self, mode: str = "pixel"):
        """Initialize diversity loss.

        Args:
            mode: How to compute diversity.
                - "pixel": Pairwise distance in pixel space (simple, default).
                - "feature": Pairwise distance in a feature space (requires features).
        """
        super().__init__()
        self.mode = mode

    def forward(
        self,
        generated_images: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute diversity loss.

        Penalizes high similarity between generated images. We want DIVERSITY,
        so we minimize average pairwise cosine similarity (or equivalently,
        maximize average pairwise distance).

        Args:
            generated_images: Generated images (batch, C, H, W).
            features: Optional feature representations (batch, feat_dim).
                      Used when mode="feature".

        Returns:
            Scalar diversity loss (lower = more diverse outputs).
        """
        if self.mode == "feature" and features is not None:
            representations = features
        else:
            # Flatten images to vectors
            representations = generated_images.view(generated_images.size(0), -1)

        # Normalize for cosine similarity
        representations = F.normalize(representations, p=2, dim=1)

        # Pairwise cosine similarity matrix
        similarity_matrix = torch.mm(representations, representations.t())

        # Mask out diagonal (self-similarity = 1)
        batch_size = representations.size(0)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=representations.device)

        # Average off-diagonal similarity (we want this to be LOW)
        avg_similarity = similarity_matrix[mask].mean()

        return avg_similarity


class CreativityLoss(nn.Module):
    """Combined creativity objective from Ren (2026).

    Computes the full loss for training the Creator:
        L = λ_novelty * L_novelty + λ_memorability * L_memorability + λ_diversity * L_diversity

    The Creator is trained to generate concepts that are:
    - Novel: Hard to classify before adaptation (high pre-adaptation entropy)
    - Memorable: Easy to classify after adaptation (low post-adaptation loss)
    - Diverse: Not collapsing to a single mode

    Args:
        lambda_novelty: Weight for novelty loss.
        lambda_memorability: Weight for memorability loss.
        lambda_diversity: Weight for diversity regularization.
        diversity_mode: Mode for diversity computation ("pixel" or "feature").
    """

    def __init__(
        self,
        lambda_novelty: float = 1.0,
        lambda_memorability: float = 1.0,
        lambda_diversity: float = 0.1,
        diversity_mode: str = "pixel",
    ):
        super().__init__()
        self.lambda_novelty = lambda_novelty
        self.lambda_memorability = lambda_memorability
        self.lambda_diversity = lambda_diversity

        self.novelty_loss = NoveltyLoss()
        self.memorability_loss = MemorabilityLoss()
        self.diversity_loss = DiversityLoss(mode=diversity_mode)

    def forward(
        self,
        pre_logits: torch.Tensor,
        post_logits: torch.Tensor,
        query_labels: torch.Tensor,
        generated_images: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the full creativity loss.

        Args:
            pre_logits: Appraiser logits BEFORE adaptation (batch, n_way).
            post_logits: Appraiser logits AFTER adaptation (batch, n_way).
            query_labels: Ground-truth labels (batch,).
            generated_images: Generated images for diversity computation.
            features: Optional feature representations for diversity computation.

        Returns:
            Dictionary containing:
                - 'total_loss': Weighted sum of all losses.
                - 'novelty_loss': Raw novelty loss value.
                - 'memorability_loss': Raw memorability loss value.
                - 'diversity_loss': Raw diversity loss value.
        """
        l_novelty = self.novelty_loss(pre_logits, query_labels)
        l_memorability = self.memorability_loss(post_logits, query_labels)
        l_diversity = self.diversity_loss(generated_images, features)

        total = (
            self.lambda_novelty * l_novelty
            + self.lambda_memorability * l_memorability
            + self.lambda_diversity * l_diversity
        )

        return {
            "total_loss": total,
            "novelty_loss": l_novelty,
            "memorability_loss": l_memorability,
            "diversity_loss": l_diversity,
        }
