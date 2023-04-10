from enum import Enum


class Method(Enum):
    GRADIENT_DESCENT = 1,
    STEEPEST_DESCENT = 2,
    NEWTON = 3,
    DAMPED_NEWTON = 4,
    CONJUGATE_GRADIENTS = 5

    def __str__(self) -> str:
        return self.name


def test_local():
    method = Method.GRADIENT_DESCENT
    print(str(method))

#test_local()