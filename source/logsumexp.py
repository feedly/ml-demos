"""
This module demos the LogSumExp trick. See https://blog.feedly.com/?p=10329
"""
import math
from typing import List
import logging
import time


def log_sum_exp_naive(X:List[float]) -> float:
    """
    a naive calculation of LogSumExp expressions
    :param X: a list of numbers
    :return: the LogSumExp calculation
    """
    logging.debug('START lse_naive(%s)', X)
    try:
        summation = 0
        for x_i in X:
            v = math.e**x_i
            logging.debug('e^%f = %.5f', x_i, v)
            summation += v
        return math.log(summation)
    except Exception as e:
        logging.debug('lse_naive FAILURE')
        raise e


def log_sum_exp(X:List[float]) -> float:
    """
    a better calculation of LogSumExp expressions
    :param X: a list of numbers
    :return: the LogSumExp calculation
    """
    logging.debug('START lse(%s)', X)
    c = max(X)
    summation = 0
    for x_i in X:
        v = math.e ** (x_i - c)
        logging.debug('e^(%f - c) = %.5f', x_i, v)
    summation += sum(math.e ** (x_i - c) for x_i in X)

    logging.debug('c=%.5f; summation=%.5f', c, summation)

    return math.log(summation) + c


def log_softmax(j:int, X:List[float], naive:bool=False) -> float:
    """
    a log softmax calculation
    :param j: an index into X that selects the numerator value.
    :param X: a list of numbers
    :param naive: use the naive LogSumExp method
    :return: the log softmax calculation
    """
    lse = log_sum_exp_naive if naive else log_sum_exp
    return X[j] - lse(X)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')  # change to debug to print intermediate calculations

    def _run_example(j:int, X:List[float]) -> None:
        print('*' * 30)
        print(f'* X={X}')
        print(f'* j={j}\n')
        time.sleep(0.001) # so the logs get printed out nicely
        y1 = log_sum_exp(X)
        try:
            y2 = log_sum_exp_naive(X)
            if abs(y1 - y2) > 1e-6:
                raise ValueError(f'calculation error {y1} != {y2}')
        except:
            y2 = 'bombed!'

        print(f'logsumpexp({X}): {y1}')
        print(f'logsumpexp({X}): {y2} (naive)')

        ls = log_softmax(j, X)
        print(f'log(softmax({j}, {X}) = {ls} --> softmax = {math.e**ls}')
        if isinstance(y2, float):
            ls = log_softmax(j, X, True)
            print(f'log(softmax({j}, {X}, naive) = {ls}')

        print('*' * 30,'\n')

    # the examples from the blog post plus a small numerically stable example
    _examples = [[1000]*3, [-1000]*3, [1,1,1]]

    for _example in _examples:
        _run_example(0, _example)

    # one huge X value
    _run_example(0, [1000, 1, 2, 3])

    # one huge negative X value
    _run_example(0, [-1000, 1, 2, 3])

    # run this in debug mode to see what happens to the contributions of the values < 1 in the logsumexp calculation and
    # also what happens to the softmax probability distribution.
    _run_example(0, [1000, 1e-5, 1e-10])
    _run_example(1, [1000, 1e-5, 1e-10])
    _run_example(2, [1000, 1e-5, 1e-10])