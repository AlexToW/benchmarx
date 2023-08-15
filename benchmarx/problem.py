from typing import Callable


class Problem:
    """
    The base class for optimization problems.

    Attributes:
        info (str): Brief information about the problem.
        f (Callable): Target function to be optimized.
        x_opt (Any, optional): Known optimal solution.
        f_opt (Any, optional): Optimal function value corresponding to x_opt.
    """

    info: str           # Brief information about the problem, such as a common name
    f: Callable         # Target function
    x_opt: any = None   # Problem's optimizer (optional)
    f_opt: any = None   # target_func(x_opt)
    def __init__(self, info: str, func: Callable, x_opt=None) -> None:
        """
        Initialize the Problem instance.

        Args:
            info (str): Brief information about the problem.
            func (Callable): Target function to be optimized.
            x_opt (Any, optional): Known optimal solution.
        """
        self.info = info
        self.f = func
        if self.x_opt is None:
            self.x_opt = x_opt
        if self.x_opt is not None and self.f_opt is None:
            self.f_opt = self.f(self.x_opt)

    def f(self, x, *args, **kwargs):
        """
        Calculate the target function value.

        Args:
            x (Any): Input value for the target function.

        Returns:
            Any: The target function value.
        """
        return self.f(x)

    def __str__(self) -> str:
        """
        Return a string representation of the Problem.

        Returns:
            str: The problem's information.
        """
        return self.info
