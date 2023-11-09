from typing import Tuple, Optional

import numpy
from scipy.stats import norm as normal_distribution, bernoulli as bernoulli_distribution


class RSGenerator:
    def __init__(self, random_state: int = 42, random_state_secondary: int = 23):
        if random_state == random_state_secondary:
            raise ValueError(f"Different random states required: given {random_state} and {random_state_secondary}")
        
        self.random_state = random_state
        self.random_state_secondary = random_state_secondary

    def generate(self, correlation: float = 0,
                 slope: Optional[float] = None,
                 label_noise: float = 0.0,
                 n_samples: int = 1000) -> Tuple[numpy.ndarray, numpy.array]:
        """Generate a 2-features dataset of size `n_samples` with the given degree of `correlation`. Instances are
        labelled according to a linear separating hyperplane of slope `slope`.
        Labels are randomly flipped with probability `label_noise`.

        Args:
            correlation: The desired correlation between the two features.
            slope: The desired slope of the separating hyperplane, if any.
            label_noise: Probability of randomly flipping the label of an instance. Defaults to 0.1
            n_samples: Number of samples to generate.

        Returns:
            A tuple yielding the generated data, and the indexes of the flipped labels.
        """
        if slope is None:
            raise ValueError("Select a non-None slope.")

        std_base = 1
        std_dependent = 1
        base_shifts = normal_distribution.rvs(size=n_samples,
                                              random_state=self.random_state)
        dependent_shifts = normal_distribution.rvs(size=n_samples,
                                                   random_state=self.random_state_secondary)

        base_feature = numpy.zeros(n_samples,) + std_base * base_shifts
        dependent_feature = numpy.zeros(n_samples,) + std_dependent * (correlation * base_shifts +
                                                                       dependent_shifts * numpy.sqrt(1 - correlation ** 2))
        data = numpy.vstack((base_feature, dependent_feature)).transpose()

        # label computation
        labels = (dependent_feature > slope * base_feature).astype(int)

        flip_distribution = bernoulli_distribution(label_noise)
        should_flip = flip_distribution.rvs(size=n_samples, random_state=self.random_state).astype(bool)
        flipping = numpy.argwhere(should_flip).squeeze()
        labels[flipping] = 1 - labels[flipping]  # (labels[flipping] + 1) % 2

        samples = numpy.hstack((data, labels.reshape(-1, 1)))

        return samples, should_flip
