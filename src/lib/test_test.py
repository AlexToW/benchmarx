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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("src"), '..')))
from jaxopt._src.proximal_gradient import ProximalGradient, ProxGradState
from jaxopt._src import base
A = jnp.array([[1, 2], [3, 4]])


def f(x):
    return x.T @ (A @ A.T) @ x



class BacktrackingProximalGradient(jaxopt.ProximalGradient):
   linesearch_777: jaxopt.BacktrackingLineSearch = None

   def _iter(self, iter_num, x, x_fun_val, x_fun_grad, stepsize, hyperparams_prox, args, kwargs):

    if self.linesearch_777 is not None:
      next_stepsize = self.linesearch_777.run(init_stepsize=1.0, params=x)[0]
      next_x = self._prox_grad(x, x_fun_grad, next_stepsize, hyperparams_prox)
      return next_x, next_stepsize
    
    return super()._iter(iter_num, x, x_fun_val, x_fun_grad, stepsize, hyperparams_prox, args, kwargs)


class BacktrackingGradientDescent(BacktrackingProximalGradient):
  #linesearch_777: jaxopt.BacktrackingLineSearch = None

  def init_state(self, init_params: Any, *args, **kwargs) -> ProxGradState:
    return super().init_state(init_params, None, *args, **kwargs)

  def update(self, params: Any, state: NamedTuple, *args, **kwargs) -> base.OptStep:

    return super().update(params, state, None, *args, **kwargs)

  def optimality_fun(self, params, *args, **kwargs):
    return self._grad_fun(params, *args, **kwargs)

  def __post_init__(self):
    super().__post_init__()
    self.reference_signature = self.fun

    
   


def run_all(solver, w_init, *args, **kwargs):
  state = solver.init_state(w_init, *args, **kwargs)
  sol = w_init
  sols, errors = [], []

  def update(sol, state):
    return solver.update(sol, state, *args, **kwargs)

  for _ in range(solver.maxiter):
    sol, state = update(sol, state)
    sols.append(sol)
    errors.append(state.error)

  return jnp.stack(sols, axis=0), errors


def _main():
    x_init = jnp.array([1., 1.])
    #gd = jaxopt.GradientDescent(fun=f, maxiter=5, acceleration=False)
    #print(run_all(gd, x_init)[0])
    ls = jaxopt.BacktrackingLineSearch(fun=f, maxiter=20, condition="strong-wolfe",
                            decrease_factor=0.8)
    gd = BacktrackingGradientDescent(fun=f, maxiter=5, acceleration=False)
    gd.linesearch_777 = ls
    print(run_all(gd, x_init)[0])


if __name__ == '__main__':
    _main()