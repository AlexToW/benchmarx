from typing import Callable


class Problem:
    """
    The base class of the optimization problem.
    """

    info: str           # Brief information about the problem, such as a common name
    f: Callable         # Target function
    x_opt: any = None   # Problem's optimizer (optional)
    f_opt: any = None   # target_func(x_opt)
    def __init__(self, info: str, func: Callable, x_opt=None) -> None:
        self.info = info
        self.f = func
        if self.x_opt is None:
            self.x_opt = x_opt
        if self.x_opt is not None and self.f_opt is None:
            self.f_opt = self.f(self.x_opt)

    def f(self, x, *args, **kwargs):
        """
        x: any
        """
        return self.f(x)

    def __str__(self) -> str:
        return self.info
