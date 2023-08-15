import jax.numpy as jnp
import jax
import logging

from sklearn.datasets import load_svmlight_file, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from benchmarx.model_problem import ModelProblem


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

    def __init__(self, problem_type: str = "mushrooms", x_opt=None) -> None:
        """
        problem_type: The type of problem to set up. Can be one of the
                               following:
                               - 'mushrooms'
                               - 'breast_cancer'
        """
        super().__init__(info=f"Logistic Regression problem on {problem_type} dataset", x_opt=x_opt)
        self.problem_type = problem_type

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
    def log_loss(w, X, y):
        """
        Logistic Loss function
        """
        return jnp.mean(jnp.logaddexp(jnp.zeros(X.shape[0]), -y * (X @ w)))

    def accuracy(self, w, X, y):
        """
        Compute accuracy on (X, y)
        """
        return accuracy_score(y, jnp.around(2 / (1 + jnp.exp(-X @ w))) - 1)

    def train_loss(self, w, *args, **kwargs):
        """
        Logistic Loss function on the train part of dataset
        """
        return LogisticRegression.log_loss(w=w, X=self.X_train, y=self.y_train)

    def test_loss(self, w, *args, **kwargs):
        """
        Logistic Loss function on the test part of dataset
        """
        return LogisticRegression.log_loss(w=w, X=self.X_test, y=self.y_test)

    def train_accuracy(self, w, *args, **kwargs):
        """
        Accuracy on the train part of dataset
        """
        return self.accuracy(w=w, X=self.X_train, y=self.y_train)

    def test_accuracy(self, w, *args, **kwargs):
        """
        Accuracy on the test part of dataset
        """
        return self.accuracy(w=w, X=self.X_test, y=self.y_test)
    
    def estimate_L(self):
        """
        Estimate the Lipschitz constant of the gradient of the logistic loss function,
        i.e. LogLoss is L-smooth.
        """
        outer_products = jax.vmap(lambda x: jnp.outer(x, x))(self.X_train)
        sum_of_outer_products = jnp.sum(outer_products, axis=0)
        eigenvalues = jnp.linalg.eigvals(sum_of_outer_products)
        max_eigenvalue = jnp.max(eigenvalues)
        L = max_eigenvalue / self.n_train
        return float(L.real)
