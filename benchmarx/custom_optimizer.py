import jax.numpy as jnp


from typing import Union
from typing import Callable
from typing import Any


class State:
    iter_num: int
    stepsize: Union[float, Callable]

    def __init__(self, iter_num: int, stepsize: Union[float, Callable]) -> None:
        self.iter_num = iter_num
        self.stepsize = stepsize


class CustomOptimizer:
    """
    Base class to implement your own optimization algorithm.
    """
    params: Any         # hyperparams
    x_init: jnp.array   # initial point
    label: str          # label (for plots?)
    def __init__(self, params: Any, x_init: jnp.array, label: str) -> None:
        """
        Initialize the CustomOptimizer instance.

        Args:
            params: Hyperparameters for the optimization algorithm.
            x_init: Initial point as a NumPy array.
            label: Label for the optimization algorithm (for plots, etc.).
        """
        self.params = params
        self.x_init = x_init
        self.label = label


    def init_state(self, x_init, *args, **kwargs) -> State:
        """
        Initialize and return the initial state of the optimization algorithm.

        Args:
            x_init: Initial point for the optimization.

        Returns:
            A State object representing the initial state.
        """
        #del x_init
        return State(iter_num=1, stepsize=1.0)


    def update(self, sol, state: State) -> tuple([jnp.array, State]):
        """
        Perform an update step of the optimization algorithm and return the updated point and state.

        Args:
            sol: Current solution point.
            state: Current state of the optimization algorithm.

        Returns:
            A tuple containing the updated solution point and the updated state.
        """
        state.iter_num += 1
        return self.x_init, state


    def stop_criterion(self, sol, state: State) -> bool:
        """
        Check if the optimization should stop based on the current state.

        Args:
            sol: Current solution point.
            state: Current state of the optimization algorithm.

        Returns:
            True if the optimization should stop, False otherwise.
        """
        return True