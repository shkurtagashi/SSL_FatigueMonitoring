import torch
import torch.nn as nn
import numpy as np


def generate_balanced_labels(num_samples, distribution, num_classes=2):
    labels = []
    for idx, percent in enumerate(distribution):
        count = int(num_samples * percent)
        labels.extend([idx] * count)

    while len(labels) < num_samples:
        labels.append(np.random.choice(np.arange(num_classes), p=distribution))

    np.random.shuffle(labels)
    return torch.tensor(labels, dtype=torch.long)


def generate_biased_random_logits(num_samples, distribution, temperature=1.0):
    """
    Generate random logits biased by the given class distribution.

    A higher temperature makes the logits more random (less biased),
    while a lower temperature makes them less random (more biased).
    """
    # First, generate completely random logits
    logits = torch.randn(num_samples, len(distribution))

    if temperature is not None:
        logits /= temperature
        # Next, adjust the logits to reflect the class distribution
        # We will shift the logits by a value related to the log-probability of the prior
        adjustment_factors = torch.tensor([np.log(p / (1 - p)) for p in distribution])

        logits += adjustment_factors

    return logits


# Set manual seed for reproducibility
torch.manual_seed(0)

# Number of samples
num_samples = 1000

# Class distribution
distribution = [0.90, 0.10]  # 10% positives, 90% negatives

# Number of classes (for binary classification, it's 2)
num_classes = 2

# Generate random logits based on the distribution
biased_random_logits = generate_biased_random_logits(num_samples, distribution, temperature=0.5)
unbiased_random_logits = generate_biased_random_logits(num_samples, distribution, temperature=None)

# Generate true labels based on the known distribution
true_labels = generate_balanced_labels(num_samples, distribution, num_classes)

# Initialize CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# Print average loss
print(f"Average Biased Cross-Entropy Loss: {criterion(biased_random_logits, true_labels).item()}")
print(f"Average Cross-Entropy Loss: {criterion(unbiased_random_logits, true_labels).item()}")
