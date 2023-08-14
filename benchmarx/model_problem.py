from typing import Callable

from abc import abstractmethod
from benchmarx.problem import Problem


class ModelProblem(Problem):
    """
    This class describes the learning problem.
    """

    def __init__(self, info: str, x_opt=None) -> None:
        super().__init__(info, self.train_loss, x_opt)

    @abstractmethod
    def train_loss(self, w, *args, **kwargs):
        """
        Train loss function (abstract method)
        """
        pass

    @abstractmethod
    def test_loss(self, w, *args, **kwargs):
        """
        Test loss function (abstract method)
        """
        pass

    @abstractmethod
    def train_accuracy(self, w, *args, **kwargs):
        """
        Train accuracy (abstract method)
        """
        pass

    @abstractmethod
    def test_accuracy(self, w, *args, **kwargs):
        """
        Test accuracy (abstract method)
        """
        pass

