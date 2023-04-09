from enum import Enum


class Method(Enum):
    GRADIENT_DESCENT = 1,
    STEEPEST_DESCENT = 2,
    NEWTON = 3,
    DAMPED_NEWTON = 4

    def __str__(self) -> str:
        return self.name