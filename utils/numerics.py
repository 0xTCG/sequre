import math

from collections import defaultdict
from itertools import product

from custom_types.vector import Vector


def expand_binomial(constant: int, degree: int) -> list:
    return [math.comb(degree, i) * constant ** i for i in range(degree + 1, -1, -1)]


def term_masks(degrees: list) -> tuple:
    # max_degrees_sum: int = sum(degrees)
    for e in product(*[range(degree + 1) for degree in degrees]):
        if sum(e) != 0:
            yield e


def expand_polynomial_term(constant: int, degrees: list, operands: list) -> dict:
    binomials: list = [expand_binomial(constant=operand, degree=degree) for operand, degree in zip(operands, degrees)]

    terms_coefs: dict = dict()

    for term_mask in term_masks(degrees):
        terms_coefs[term_mask] = math.prod([binomials[i][j] for i, j in enumerate(term_mask)])

    return terms_coefs


def expand_polynomial(constants: list, degrees_list: list, operands: list) -> list:
    terms_coefs: dict = defaultdict(int)
    
    for constant, degrees in zip(constants, degrees_list):
        for term_mask, coef in expand_polynomial_term(constant=constant, degrees=degrees, operands=operands).items():
            terms_coefs[term_mask] += coef
    
    return [terms_coefs[key] for key in sorted(terms_coefs.keys())]


def evaluate_polynome(x: Vector, coefs: Vector, degrees_list: list) -> int:
    length: int = len(x)
    output: int = 0
    for coef, degrees in zip(coefs, degrees_list):
        output += coef * math.prod([x[i] ** degrees[i] for i in range(length)])
    return output
