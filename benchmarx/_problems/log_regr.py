from benchmarx.problem import Problem
import jax.numpy as jnp
import jax
import logging
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LogisticRegression(Problem):
    """
    A class describing an unconstrained logistic regression problem.

    f(w) = 1/n sum_{i=1}^n f_i(w) -> min_w,
    f_i(w) = ln(1 + exp(-y_i * w.T @ x_i))


    Typical usage example:

    problem = LinearLeastSquares("random", m=3, n=3)
    benchmark = Benchmark(problem=problem, ...)
    result = benchmark.run()
    """

    def __init__(self, problem_type: str = "mushrooms") -> None:
        super().__init__(
            info=f"Logistic Regression problem on {problem_type} dataset", func=self.f
        )

        if problem_type == "mushrooms":
            self.problem_type = problem_type

            dataset = "mushrooms.txt"
            data = load_svmlight_file(dataset)
            X, y = data[0].toarray(), data[1]
            y = 2 * y - 3

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, train_size=0.8, test_size=0.2, shuffle=True
            )
            self.n_train, self.d_train = self.X_train.shape
            self.n_test, self.d_test = self.X_test.shape

        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    @staticmethod
    @jax.jit
    def LogLoss(w, X, y):
        """
        Log Loss function
        """
        return jnp.mean(jnp.logaddexp(jnp.zeros(X.shape[0]), -y * (X @ w)))

    @staticmethod
    @jax.jit
    def GradLogLoss(w, X, y):
        """
        Log Loss gradient
        """
        func = lambda w: LogisticRegression.LogLoss(w, X, y)
        return jax.grad(func)(w)

    def f(self, w, *args, **kwargs):
        """
        Objective function: log loss on train
        """
        return LogisticRegression.LogLoss(w=w, X=self.X_train, y=self.y_train)
    
    def f_i(self, w, i, *args, **kwargs):
        """
        i-th summand of LogLoss function
        """
        return jnp.log(1 + jnp.exp(-self.y_train[i] * w.T @ self.X_train[i]))
    
    def grad_i(self, w, i, *args, **kwargs):
        """
        gradient of i-th summand of LogLoss function
        """
        func = lambda w: self.f_i(w, i)
        return jax.grad(func)(w)

    def grad(self, w, *args, **kwargs):
        """
        log loss gradient on train
        """
        return jnp.array(LogisticRegression.GradLogLoss(w=w, X=self.X_train, y=self.y_train))
    
    @staticmethod
    def accuracy(w, X, y):
        """
        Compute accuracy on (X, y)
        """
        return accuracy_score(y, jnp.around(2 / (1 + jnp.exp(- X @ w))) - 1)
    
    def train_accuracy(self, w):
        """
        Compute accuracy on (X_train, y_train)
        """
        return LogisticRegression.accuracy(w=w, X=self.X_train, y=self.y_train)
    
    def test_accuracy(self, w):
        """
        Compute accuracy on (X_test, y_test)
        """
        return LogisticRegression.accuracy(w=w, X=self.X_test, y=self.y_test)
