from typing import Callable

from abc import abstractmethod
from benchmarx.problem import Problem


class ModelProblem(Problem):
    """
    Abstract base class describing a learning problem.

    Attributes:
        info (str): Information about the problem.
        x_opt (Optional): Optimal solution (if available).
    """

    def __init__(self, info: str, x_opt=None) -> None:
        super().__init__(info, self.train_loss, x_opt)

    @abstractmethod
    def train_loss(self, w, *args, **kwargs):
        """
        Abstract method for computing the training loss.
        """
        pass

    @abstractmethod
    def test_loss(self, w, *args, **kwargs):
        """
        Abstract method for computing the test loss.
        """
        pass

    @abstractmethod
    def train_accuracy(self, w, *args, **kwargs):
        """
        Abstract method for computing the training accuracy.
        """
        pass

    @abstractmethod
    def test_accuracy(self, w, *args, **kwargs):
        """
        Abstract method for computing the test accuracy.
        """
        pass

