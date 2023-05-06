import logging


available_built_in_methods = [
    'GRADIENT_DESCENT',
    'BFGS'
]

def check_method(methods: list[str]):
    for method in methods:
        for avail_method in available_built_in_methods:
            if method.startswith(avail_method):
                return True
    logging.critical(f'Unsupported built in method \'{method}\'. Available built in methods: {available_built_in_methods}')
    return False

