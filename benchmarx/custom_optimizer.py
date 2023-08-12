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
    #x_tmp: jnp.array    # tmp point
    def __init__(self, params: Any, x_init: jnp.array, label: str) -> None:
        self.params = params
        self.x_init = x_init
        #self.x_tmp = x_init
        self.label = label


    def init_state(self, x_init, *args, **kwargs) -> State:
        """
        Returns initial State.
        """
        #del x_init
        return State(iter_num=1, stepsize=1.0)


    def update(self, sol, state: State) -> tuple([jnp.array, State]):
        """
        Returns the next point x_next and the next State.
        """
        state.iter_num += 1
        return self.x_init, state


    def stop_criterion(self, sol, state: State) -> bool:
        """
        Returns True if it's time to stop.
        """
        return state.iter_num > 500