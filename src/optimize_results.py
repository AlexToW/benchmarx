class OptimizeResults:
    trajectory: list = None

    def __init__(self, success: bool = False, message: str = "", x: float = 0., fun: float = 0.0, jac= None, hess=None, hess_inv=None,
                 nfev: int = 0, njev: int = 0, nhev: int = 0, nit: int = 0, maxcv: float = 0.0):
        self.success = success
        self.message = message
        self.x = x
        self.fun = fun
        self.jac = jac
        self.hess = hess
        self.hess_inv = hess_inv
        self.nfev = nfev
        self.njev = njev
        self.nhev = nhev
        self.nit = nit
        self.maxcv = maxcv

    def __str__(self):
        return f'''fun: {self.fun}
jac: {self.jac}
message: \'{self.message}\'
nfev: {self.nfev}
nit: {self.nit}
njev: {self.njev}
success: {self.success}
x: {self.x}
        '''
