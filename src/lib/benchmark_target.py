from enum import Enum

class BenchmarkTarget(Enum):
    trajectory_x = 1,
    trajectory_f = 2,
    trajectory_df = 3,
    nit = 4,
    nfev = 5,
    njev = 6,
    nhev = 7,
    errors = 8,
    time = 9