import jax.numpy as jnp
import jax
import logging

from sklearn.datasets import load_svmlight_file, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from benchmarx.model_problem import ModelProblem
from typing import Callable


class LogisticRegression(ModelProblem):
    """
    A class describing an unconstrained logistic regression problem.

    f(w) = 1/n sum_{i=1}^n f_i(w) -> min_w,
    f_i(w) = ln(1 + exp(-y_i * w.T @ x_i))

    Typical usage example:

    problem = LogisticRegression("mushrooms")
    benchmark = Benchmark(problem=problem, ...)
    result = benchmark.run()
    """

    def __init__(self, info: str=None, problem_type: str = "mushrooms", train_data_part_size: int=-1, regularizer: Callable= lambda w: 0, x_opt=None) -> None:
        """
        problem_type: The type of problem to set up. Can be one of the
                               following:
                               - 'mushrooms'
                               - 'breast_cancer'
        """
        info_str = info
        if info is None:
            info_str = f"Logistic Regression problem on {problem_type} dataset"
        super().__init__(info=info_str, x_opt=x_opt)
        self.problem_type = problem_type
        self.regularizer = regularizer

        if train_data_part_size > 0:
            self.train_data_parts = train_data_part_size

        if problem_type == "mushrooms":
            dataset = "mushrooms.txt"
            data = load_svmlight_file(dataset)
            X, y = data[0].toarray(), data[1]
            y = 2 * y - 3

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, train_size=0.8, test_size=0.2, shuffle=True
            )
            self.n_train, self.d_train = self.X_train.shape
            self.n_test, self.d_test = self.X_test.shape
        
        elif problem_type == "breast_cancer":
            cancer = load_breast_cancer()
            X = cancer.data
            y = cancer.target
            # from {0, 1} to {-1, 1}
            y = (y+1)*2-3
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, train_size=0.8, test_size=0.2, shuffle=True
            )
            self.n_train, self.d_train = self.X_train.shape
            self.n_test, self.d_test = self.X_test.shape

        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    @staticmethod
    @jax.jit
    def jitted_log_loss(w, X, y):
        """
        Calculate the logistic loss function value.

        Args:
            w (Any): Parameter vector.
            X (Any): Input features.
            y (Any): Target labels.

        Returns:
            Any: The logistic loss function value.
        """
        return jnp.mean(jnp.logaddexp(jnp.zeros(X.shape[0]), -y * (X @ w)))
    
    @staticmethod
    def log_loss(w, X, y, regularizer):
        """
        Calculate the regularized logistic loss function value.

        Args:
            w (Any): Parameter vector.
            X (Any): Input features.
            y (Any): Target labels.
            regularizer (Callable): regularizer

        Returns:
            Any: The regularized logistic loss function value.
        """
        return LogisticRegression.jitted_log_loss(w=w, X=X, y=y) + regularizer(w)
    
    def grad_log_loss_ind(self, w, ind):
        """
        The logistic regression problem has a finite sum form:
        f(w) = 1/n sum_{j=1}^n [f_j(w)],
        f_j(w) = 1/b sum_{i=1}^b l(g(w, x_i), y_i)

        where b = self.train_data_part_size,
        n*b = N -- full sample size (X.shape[0]).
        ind in range(n)
        Returns:
            1/b sum_{i=1}^b l(g(w, x_i), y_i)
        """
        X_part = self.X_train[ind * self.train_data_part_size : (ind + 1) * self.train_data_part_size]
        y_part = self.y_train[ind * self.train_data_part_size : (ind + 1) * self.train_data_part_size]
        return jax.grad(LogisticRegression.jitted_log_loss(w=w, X=X_part, y=y_part))(w)

    def accuracy(self, w, X, y):
        """
        Compute accuracy on (X, y).

        Args:
            w (Any): Parameter vector.
            X (Any): Input features.
            y (Any): Target labels.

        Returns:
            float: Accuracy score.
        """
        return accuracy_score(y, jnp.around(2 / (1 + jnp.exp(-X @ w))) - 1)

    def train_loss(self, w, *args, **kwargs):
        """
        Calculate the logistic loss function on the train part of the dataset.

        Args:
            w (Any): Parameter vector.

        Returns:
            Any: The logistic loss function value.
        """
        return LogisticRegression.log_loss(w=w, X=self.X_train, y=self.y_train, regularizer=self.regularizer)

    def test_loss(self, w, *args, **kwargs):
        """
        Calculate the logistic loss function on the test part of the dataset.

        Args:
            w (Any): Parameter vector.

        Returns:
            Any: The logistic loss function value.
        """
        return LogisticRegression.log_loss(w=w, X=self.X_test, y=self.y_test, regularizer=self.regularizer)

    def train_accuracy(self, w, *args, **kwargs):
        """
        Calculate the accuracy on the train part of the dataset.

        Args:
            w (Any): Parameter vector.

        Returns:
            float: Accuracy score.
        """
        return self.accuracy(w=w, X=self.X_train, y=self.y_train)

    def test_accuracy(self, w, *args, **kwargs):
        """
        Calculate the accuracy on the test part of the dataset.

        Args:
            w (Any): Parameter vector.

        Returns:
            float: Accuracy score.
        """
        return self.accuracy(w=w, X=self.X_test, y=self.y_test)
    
    def estimate_L(self):
        """
        Estimate the Lipschitz constant of the gradient of the logistic loss function.

        Returns:
            float: Estimated Lipschitz constant.
        """
        outer_products = jax.vmap(lambda x: jnp.outer(x, x))(self.X_train)
        sum_of_outer_products = jnp.sum(outer_products, axis=0)
        eigenvalues = jnp.linalg.eigvals(sum_of_outer_products)
        max_eigenvalue = jnp.max(eigenvalues)
        L = max_eigenvalue / self.n_train
        return float(L.real)

    def estimate_L_for_sum(self):
        """
        Estimate the Lipschitz constant of the gradient of the logistic loss function
        for each part of datasets and returns maximum.

        Returns:
            float: Estimated Lipschitz constant for sum.
        """
        Ls = []
        n = self.n_train // self.tr