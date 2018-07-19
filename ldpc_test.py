import dwavebinarycsp
import dimod
import itertools
import math
import numpy as np
import random
#import sympy
import dwavebinarycsp.core.csp as dwcsp
from factors import Factor, build_factors
from divide_and_concur import DivideAndConcur

configs = frozenset([(0, 0, 0), (0, 1, 1), (1, 1, 0), (1, 0, 1)])
#configs = frozenset(itertools.product([0,1], repeat=3))
constraint = dwavebinarycsp.Constraint.from_configurations(configs, ['a', 'b', 'c'], 'BINARY')
#constraint = dwavecsp.Constraint.from_configurations(configs, ['a', 'a', 'a'], 'BINARY')
print(constraint)


def xor(*ins):
    return sum(ins) % 2 == 0

constraint = dwavebinarycsp.Constraint.from_func(xor, ['x%d' % i for i in range(3)], 'BINARY')
print(constraint)


def build_check_matrix(num_variables, num_clauses, arity=3):
    constraints = set()
    variables = list(range(num_variables))
    H = np.zeros((num_clauses, num_variables), dtype=np.int32)
    count = 0
    while count < num_clauses:
        constraint_variables = sorted(random.sample(variables, arity))
        H[count, constraint_variables] = 1
        count += 1

    return H


def build_ldpc(num_vars, num_constraints = None, error_probability=0.1):
    num_constraints = int(math.floor(num_vars*0.7))
    H = build_check_matrix(num_vars, num_constraints)
    print(H)
    if any(np.sum(H,axis=1)==0):
        raise ValueError('Variable not used')
    #sH = sympy.Matrix(H)
    #M = sH.rref()  # could steal source code from here to make this work
    y = np.random.randint(0, 2, [num_vars])
    print(y)
    ldpc = {'H': H, 'y': y}
    return ldpc


def build_csp_from_ldpc(ldpc):

    csp = dwcsp.ConstraintSatisfactionProblem('BINARY')
    H = ldpc['H']
    variables = list(range(H.shape[1]))

    for count in range(H.shape[0]):
        variables = np.where(H[count])[0]
        constraint = dwcsp.Constraint.from_func(xor, variables, 'BINARY')
        csp.add_constraint(constraint)

    # in case any variables didn't make it in
    for v in variables:
        csp.add_variable(v)

    return csp


def build_bqm_from_ldpc(ldpc, csp_scale = 2):

    csp = build_csp_from_ldpc(ldpc)
    bqm, all_pmodels = dwavebinarycsp.stitch(csp, min_classical_gap=2.)
    y = ldpc['y']
    # add linear biases to bqm
    biases = {v: 1-2*y[v] for v in range(len(y))}
    biases_bqm = dimod.BinaryQuadraticModel(biases, dict(), 0., 'BINARY')
    bqm.scale(csp_scale)
    bqm.update(biases_bqm)
    return bqm, csp, all_pmodels



def hamming_distance(x,y):
    return np.sum(np.array([x[v] != y[v] for v in x]))

#ldpc = build_ldpc(7)
ldpc = dict()
ldpc['H'] = np.array([
    [1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 1]])
ldpc['y'] = np.array([1, 0, 0, 1, 0, 0, 0])

csp_scale = 5
bqm, csp, all_pmodels = build_bqm_from_ldpc(ldpc, csp_scale)

print(bqm)
response = dimod.ExactSolver().sample(bqm)
minen = response.data(['energy']).next()
print(minen)
y = ldpc['y']
for sample, energy in response.data(['sample', 'energy']):
    if energy == minen:
        solution = {v: sample[v] for v in csp.variables}
        print(solution, csp.check(sample), energy, hamming_distance(solution,y))


all_factors = build_factors(csp, all_pmodels, bqm=bqm, csp_scale=csp_scale)

sampler = DivideAndConcur(all_factors, dimod.samplers.ExactSolver(), difference_mapping=True)
response = sampler.sample(bqm)
sample = next(response.samples(1))
energy = bqm.energy(sample)
solution = {v: sample[v] for v in csp.variables}
print(solution, csp.check(sample), energy, hamming_distance(solution,y))
