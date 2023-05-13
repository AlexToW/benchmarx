import jaxopt
import jax
import jax.numpy as jnp

import time
import logging

from benchmarx.src.problem import Problem
import benchmarx.src.methods as _methods
import benchmarx.src.metrics as _metrics
from benchmarx.src.benchmark_result import BenchmarkResult
from benchmarx.src import custom_optimizer

from benchmarx.src.ProxGD_custom_linesearch import GradientDescentCLS
from benchmarx.src.plotter import Plotter


class Benchmark:
    """
    A class that provides the benchmarking of different optimization
    methods on a given problem (like Problem object).
    """

    runs: int = 1       # the number of runs of each method
    problem: Problem = None  # Problem to solve
    methods: list[dict[str : dict[str:any]]] = None  # Methods for benchmarking
    available_built_in_methods: list[str] = None # method's keywords. 
    # If you want to call a method from the jaxopt, 
    # the name of the method must begin with one of these keywords.
    metrics: list[str] = None  # List of fields to include in BenchamrkResult
    aval_linesearch_str = ['armijo', 'goldstein', 'strong-wolfe', 'wolfe']
    def __init__(
        self,
        problem: Problem,
        methods: list[dict[str : dict[str:any]]],
        metrics: list[str],
        runs: int = 1
    ) -> None:
        self.runs = runs
        self.problem = problem
        methods_names = list()
        for item in methods:
            for name, params in item.items():
                methods_names.append(name)
        if not _methods.check_method(methods_names):
            exit(1)
        self.methods = methods
        self.available_built_in_methods = _methods.available_built_in_methods
        if not _metrics.check_metric(metrics):
            exit(1)
        self.metrics = metrics

    def _check_linesearch(self, ls_str: str, method: str):
        #TODO: 'steepest' for QuadraticProblem!
        if method.startswith('GRADIENT_DESCENT'):
            return ls_str in ['armijo', 'goldstein', 'strong-wolfe', 'wolfe']
        elif method.startswith('BFGS') or method.startswith('LBFGS'):
            return ls_str in ['armijo', 'goldstein', 'strong-wolfe', 'wolfe'] or ls_str in ['backtracking', 'zoom', 'hager-zhang']
        
        return False

    def __run_solver(
        self, solver, x_init, metrics: list[str], *args, **kwargs
    ) -> dict[str, list[any]]:
        """
        A layer for pulling the necessary information according to metrics
        as the "method" solver works (solver like jaxopt.GradientDescent obj
        or or an heir to the CustomOptimizer class)
        """
        custom_method = issubclass(type(solver), custom_optimizer.CustomOptimizer)
        #cls = hasattr(solver, 'linesearch_custom')
        result = dict()
        start_time = time.time()
        state = solver.init_state(x_init, *args, **kwargs)
        sol = x_init
        if custom_method and sol is None:
            sol = solver.x_init
        
        x_prev = sol

        @jax.jit
        def jitted_update(sol, state):
            #logging.warning(f'Function \"jitted_update\" doesn\'t use jax.jit.')
            return solver.update(sol, state, *args, **kwargs)
        
        def update(sol, state):
            return solver.update(sol, state, *args, **kwargs)


        def stop_criterion(err, tol):
            return err < tol

        tol = 1e-3
        
        if not custom_optimizer and 'tol' in kwargs:
            tol = kwargs['tol']
        
        #print(tol)
        for i in range(solver.maxiter):
            if i > 0:
                if not custom_method and stop_criterion(state.error, tol):
                    break
                if custom_method and solver.stop_criterion(sol, state):
                    break
            if isinstance(sol, float):
                sol = jnp.array([sol])
            if "history_x" in metrics:
                if not "history_x" in result:
                    result["history_x"] = [sol]
                else:
                    result["history_x"].append(sol)
            if "history_f" in metrics:
                if not "history_f" in result:
                    result["history_f"] = [self.problem.f(sol)]
                else:
                    result["history_f"].append(self.problem.f(sol))
            if "history_df" in metrics:
                if not "history_df" in result:
                    result["history_df"] = [jax.grad(self.problem.f)(sol)]
                else:
                    result["history_df"].append(jax.grad(self.problem.f)(sol))
            if "nit" in metrics:
                if not "nit" in result:
                    result["nit"] = [1]
                else:
                    result["nit"][0] += 1
            if "nfev" in metrics:
                # IDK
                pass
            if "njev" in metrics:
                # IDK
                pass
            if "nhev" in metrics:
                # IDK
                pass
            if "errors" in metrics:
                if not "errors" in result:
                    result["errors"] = [state.error]
                else:
                    result["errors"].append(state.error)
            x_prev = sol
            if custom_method:
                sol, state = update(sol, state)
            else:
                sol, state = jitted_update(sol, state)
        duration = time.time() - start_time
        if "time" in metrics:
            result["time"] = [duration]

        return result

    def run(self, user_method = None) -> BenchmarkResult:
        res = BenchmarkResult(problem=self.problem, methods=list(), metrics=self.metrics)
        data = dict()
        data[self.problem] = dict()
        # methods: list[dict[method(str) : dict[str:any]]]
        for item in self.methods:
            for method, params in item.items():
                # data: dict[Problem, dict[method(str), dict[str, list[any]]]]
                
                #======= custom line search =======
                # A class jaxopt.BacktrackingLineSearch object or str is expected.
                # For Gradient Descent: params['linesearch'] in ['wolfe', 'strong-wolfe', 'armijo', 'goldstein']
                # For (L)BFGS: params['linesearch'] must be str from 
                # ['backtracking', 'zoom', 'hager-zhang'] or ['wolfe', 'strong-wolfe', 'armijo', 'goldstein']. 
                cls = 'linesearch' in params
          
                if method.startswith('GRADIENT_DESCENT'):
                    logging.info('Default gradient descent')
                    res.methods.append(method)
                    x_init = None
                    label = 'jaxopt.GradientDescent'
                    seed = str(self.problem.seed)
                    if 'x_init' in params:
                        x_init = params['x_init']
                        params.pop('x_init')
                    if 'label' in params:
                        label = params['label']
                        params.pop('label')
                    if 'seed' in params:
                        seed = params['seed']
                        params.pop('seed')
                    runs_dict = dict()
                    solver = None
                    if cls:
                        ls = params['linesearch']
                        params.pop('linesearch')
                        if 'condition' in params:
                            condition = params['condition']
                            params.pop('condition')
                        if isinstance(ls, str):
                            if ls == 'backtracking':
                                if condition in ['wolfe', 'strong-wolfe', 'armijo', 'goldstein']:
                                    ls_obj = jaxopt.BacktrackingLineSearch(fun=self.problem.f, maxiter=20, condition=condition, decrease_factor=0.8)
                                else:
                                    err_msg = f'Unknown condition {condition}'
                                    logging.critical(err_msg)
                                    exit(1)
                            elif ls == 'hager-zhang':
                                ls_obj = jaxopt.HagerZhangLineSearch(fun=self.problem.f)
                            else:
                                err_msg = f'Unknown line search {ls}'
                                logging.critical(err_msg)
                                exit(1)
                            solver = GradientDescentCLS(fun=self.problem.f, **params)
                            solver.linesearch_custom = ls_obj
                        elif isinstance(ls, jaxopt.BacktrackingLineSearch):
                            solver = GradientDescentCLS(fun=self.problem.f, **params)
                            solver.linesearch_custom = ls
                        else:
                            err_msg = f'Unknown linesearch {ls}'
                            logging.critical(err_msg)
                            exit(1)
                    else:
                        solver = jaxopt.GradientDescent(fun=self.problem.f, **params)

                    for run in range(self.runs):
                        if run % 10 == 0:
                            logging.info(f'#{run} run...')
                        sub = self.__run_solver(solver=solver, x_init=x_init, metrics=self.metrics, **params)    
                        runs_dict[f'run_{run}'] = sub
                    params['x_init'] = x_init
                    params['label'] = label
                    params['seed'] = seed
                    data[self.problem][method] = {'hyperparams': params, 'runs': runs_dict}

                elif method.startswith('BFGS'):
                    logging.info('BFGS (jaxopt built-in)')
                    res.methods.append(method)
                    x_init = None
                    label = 'jaxopt.BFGS'
                    seed = str(self.problem.seed)
                    if 'x_init' in params:
                        x_init = params['x_init']
                        params.pop('x_init')
                    if 'label' in params:
                        label = params['label']
                        params.pop('label')
                    if 'seed' in params:
                        seed = params['seed']
                        params.pop('seed')
                    runs_dict = dict()
                    soler = None
                    if cls:
                        new_linesearch = 'zoom'
                        new_condition = 'stron-wolfe'
                        ls = params['linesearch']
                        params.pop('linesearch')
                        cond = ''
                        if 'condition' in params:
                            cond = params['condition']
                            params.pop('condition')
                        if isinstance(ls, str) and self._check_linesearch(ls, method):
                            if ls in ['backtracking', 'zoom', 'hager-zhang']:
                                new_linesearch = ls
                            else:
                                err_msg = f'Unknown line search \'{ls}\'. zoom line search will be used instead of {ls}.'
                                logging.warning(err_msg)
                            if cond in ['wolfe', 'strong-wolfe', 'armijo', 'goldstein']:
                                new_condition = cond
                            else:
                                err_msg = f'Unknown condition \'{cond}\'. strong-wolfe condition will be used instead if {cond}'
                                logging.warning(err_msg)
                        else:
                            err_msg = f"For BFGS parameter \'linesearch\' must be string from {['wolfe', 'strong-wolfe', 'armijo', 'goldstein']}(condition) or {['backtracking', 'zoom', 'hager-zhang']} (linesearch)"
                            logging.critical(err_msg)
                        
                        solver = jaxopt.BFGS(fun=self.problem.f, linesearch=new_linesearch, condition=new_condition, **params)
                    else:
                        solver = jaxopt.BFGS(fun=self.problem.f, **params)

                    for run in range(self.runs):
                        if run % 10 == 0:
                            logging.info(f'#{run} run...')
                        sub = self.__run_solver(solver=solver, x_init=x_init, metrics=self.metrics, **params)    
                        runs_dict[f'run_{run}'] = sub
                    params['x_init'] = x_init
                    params['label'] = label
                    params['seed'] = seed
                    data[self.problem][method] = {'hyperparams': params, 'runs': runs_dict}

                elif method.startswith('LBFGS'):
                    logging.info('LBFGS (jaxopt built-in)')
                    res.methods.append(method)
                    x_init = None
                    label = 'jaxopt.LBFGS'
                    seed = str(self.problem.seed)
                    if 'x_init' in params:
                        x_init = params['x_init']
                        params.pop('x_init')
                    if 'label' in params:
                        label = params['label']
                        params.pop('label')
                    if 'seed' in params:
                        seed = params['seed']
                        params.pop('seed')
                    runs_dict = dict()
                    for run in range(self.runs):
                        if cls:
                            new_linesearch = 'zoom'
                            new_condition = 'stron-wolfe'
                            ls = params['linesearch']
                            params.pop('linesearch')
                            cond = ''
                            if 'condition' in params:
                                cond = params['condition']
                                params.pop('condition')
                            if isinstance(ls, str) and self._check_linesearch(ls, method):
                                if ls in ['backtracking', 'zoom', 'hager-zhang']:
                                    new_linesearch = ls
                                if cond in ['wolfe', 'strong-wolfe', 'armijo', 'goldstein']:
                                    new_condition = cond
                                else:
                                    err_msg = f'Unknown line search \'{ls}\', {cond}'
                                    logging.critical(err_msg)
                                    exit(1)
                            else:
                                err_msg = f"For LBFGS parameter \'linesearch\' must be string from {['wolfe', 'strong-wolfe', 'armijo', 'goldstein']}(condition) or {['backtracking', 'zoom', 'hager-zhang']} (linesearch)"
                                logging.critical(err_msg)
                            
                            solver = jaxopt.LBFGS(fun=self.problem.f, linesearch=new_linesearch, condition=new_condition, **params)
                        else:
                            solver = jaxopt.LBFGS(fun=self.problem.f, **params)
                        sub = self.__run_solver(solver=solver, x_init=x_init, metrics=self.metrics, **params)    
                        runs_dict[f'run_{run}'] = sub
                    params['x_init'] = x_init
                    params['label'] = label
                    params['seed'] = seed
                    data[self.problem][method] = {'hyperparams': params, 'runs': runs_dict}

                elif method.startswith('ArmijoSGD'):
                    logging.info('ArmijoSGD (jaxopt built-in)')
                    res.methods.append(method)
                    x_init = None
                    label = 'jaxopt.ArmijoSGD'
                    seed = str(self.problem.seed)
                    if 'x_init' in params:
                        x_init = params['x_init']
                        params.pop('x_init')
                    if 'label' in params:
                        label = params['label']
                        params.pop('label')
                    if 'seed' in params:
                        seed = params['seed']
                        params.pop('seed')
                    runs_dict = dict()
                    solver = jaxopt.ArmijoSGD(fun=self.problem.f, **params)
                    for run in range(self.runs):
                        if run % 10 == 0:
                            logging.info(f'#{run} run...')
                        sub = self.__run_solver(solver=solver, x_init=x_init, metrics=self.metrics, **params)    
                        runs_dict[f'run_{run}'] = sub
                    params['x_init'] = x_init
                    params['label'] = label
                    params['seed'] = seed
                    data[self.problem][method] = {'hyperparams': params, 'runs': runs_dict}

                elif method.startswith('PolyakSGD'):
                    logging.info('PolyakSGD (jaxopt built-in)')
                    res.methods.append(method)
                    x_init = None
                    label = 'jaxopt.PolyakSGD'
                    seed = str(self.problem.seed)
                    if 'x_init' in params:
                        x_init = params['x_init']
                        params.pop('x_init')
                    if 'label' in params:
                        label = params['label']
                        params.pop('label')
                    if 'seed' in params:
                        seed = params['seed']
                        params.pop('seed')
                    runs_dict = dict()
                    solver = jaxopt.PolyakSGD(fun=self.problem.f, **params)
                    for run in range(self.runs):
                        if run % 10 == 0:
                            logging.info(f'#{run} run...')
                        sub = self.__run_solver(solver=solver, x_init=x_init, metrics=self.metrics, **params)    
                        runs_dict[f'run_{run}'] = sub
                    params['x_init'] = x_init
                    params['label'] = label
                    params['seed'] = seed
                    data[self.problem][method] = {'hyperparams': params, 'runs': runs_dict}

                elif user_method is not None:
                    logging.info('Custom method')
                    res.methods.append(method)
                    x_init = None
                    if 'x_init' in params:
                        x_init = jnp.array(params['x_init'])
                        params.pop('x_init')
                    runs_dict = dict()
                    for run in range(self.runs):
                        if run % 10 == 0:
                            logging.info(f'#{run} run...')
                        sub = self.__run_solver(solver=user_method, metrics=self.metrics, x_init=x_init, **params)
                        runs_dict[f'run_{run}'] = sub
                    params_to_write = user_method.params
                    if 'x_init' not in params_to_write:
                        params_to_write['x_init'] = user_method.x_init
                    params_to_write['label'] = user_method.label
                    params_to_write['seed'] = self.problem.seed
                    data[self.problem][method] = {'hyperparams': params_to_write, 'runs': runs_dict}

        res.data = data
        return res


"""
def test_local():
    from problems.quadratic_problem import QuadraticProblem

    n = 2
    x_init = jnp.array([1.0, 1.0])
    problem = QuadraticProblem(n=n)
    benchamrk = Benchmark(
        problem=problem,
        methods=[
            {
                'GRADIENT_DESCENT_const_step': {
                    'x_init' : x_init,
                    'tol': 1e-2,
                    'maxiter': 11,
                    'stepsize' : 1e-2
                }
            },
            {
                'GRADIENT_DESCENT_adaptive_step': {
                    'x_init' : x_init,
                    'tol': 1e-2,
                    'maxiter': 11,
                    'stepsize' : lambda iter_num: 1 / (iter_num + 20)
                }
            },
            {
                'GRADIENT_DESCENT_strong_wolfe': {
                    'x_init' : x_init,
                    'tol': 1e-2,
                    'maxiter': 11,
                    'linesearch' : 'backtracking',
                    'condition': 'strong-wolfe'
                }
            },
            {
                'BFGS_strong_wolfe': {
                    'x_init' : x_init,
                    'tol': 1e-2,
                    'maxiter': 11,
                    'condition': 'strong-wolfe'
                }
            },
            {
                'BFGS_armijo': {
                    'x_init' : x_init,
                    'tol': 1e-2,
                    'maxiter': 11,
                    'condition': 'armijo'
                }
            }
        ],
        metrics=[
            "nit",
            "history_x",
            "history_f",
        ],
    )
    result = benchamrk.run()
    result.save("GD_quadratic.json")



if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    test_local()
"""