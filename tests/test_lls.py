import pytest
from context import benchmarx

from benchmarx.problems import LinearLeastSquares
import jax.numpy as jnp
import jax

def test_custom_problem():
    A = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    b = jnp.array([1.0, 2.0])

    problem = LinearLeastSquares("custom", A=A, b=b)
    x = jax.random.normal(jax.random.PRNGKey(0), (2,))

    assert problem.f(x) >= 0.0

def test_random_problem():
    problem = LinearLeastSquares("random", m=2, n=2)
    x = jax.random.normal(jax.random.PRNGKey(0), (2,))

    assert problem.f(x) >= 0.0

def test_boston_problem():
    problem = LinearLeastSquares("boston")
    m, n = problem.A.shape
    x = jax.random.normal(jax.random.PRNGKey(0), (n,))

    assert problem.f(x) >= 0.0

def test_wine_problem():
    problem = LinearLeastSquares("wine")
    m, n = problem.A.shape
    x = jax.random.normal(jax.random.PRNGKey(0), (n,))

    assert problem.f(x) >= 0.0
