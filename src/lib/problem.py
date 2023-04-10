from typing import Callable


class Problem:
    """
    The base class of the optimization problem.
    """
    info: str       # Brief information about the problem, such as a common name
    f: Callable     # Target function    
    x_opt: any      # Problem's optimizer (optional)
    def __init__(self, info: str, func: Callable, x_opt = None) -> None:
        self.info = info
        self.f = func
        self.x_opt = x_opt
    
    def f(self, x):
        """
        x: any
        """
        return self.f(x)
    
    def __str__(self) -> str:
        return self.info