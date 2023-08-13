import jaxopt
import jax
import jax.numpy as jnp

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union
import os
import sys

# make higher-level lib package visible to scripts
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("benchmarx"), '..')))
from jaxopt._src.proximal_gradient import ProximalGradient, ProxGradState
from jaxopt import base


class ProximalGradientCLS(jaxopt.ProximalGradient):
   linesearch_custom: jaxopt.BacktrackingLineSearch = None

   def _iter(self, iter_num, x, x_fun_val, x_fun_grad, stepsize, hyperparams_prox, args, kwargs):

    if self.linesearch_custom is not None:
      next_stepsize = self.linesearch_custom.run(init_stepsize=1.0, params=x)[0]
      next_x = self._prox_grad(x, x_fun_grad, next_stepsize, hyperparams_prox)
      return next_x, next_stepsize
    
    return super()._iter(iter_num, x, x_fun_val, x_fun_grad, stepsize, hyperparams_prox, args, kwargs)


class GradientDescentCLS(ProximalGradientCLS):

  def init_state(self, init_params: Any, *args, **kwargs) -> ProxGradState:
    return super().init_state(init_params, None, *args, **kwargs)

  def update(self, params: Any, state: NamedTuple, *args, **kwargs) -> base.OptStep:

    return super().update(params, state, None, *args, **kwargs)

  def optimality_fun(self, params, *args, **kwargs):
    return self._grad_fun(params, *args, **kwargs)

  def __post_init__(self):
    super().__post_init__()
    self.reference_signature = self.fun