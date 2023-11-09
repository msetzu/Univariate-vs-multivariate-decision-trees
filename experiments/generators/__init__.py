"""Synthetic data generators."""
from abc import abstractmethod, ABC

import numpy


class Generator(ABC):
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        numpy.random.seed(random_state)

    @abstractmethod
    def generate(self, **kwargs):
        pass

class GenerationError(ValueError):
    pass
