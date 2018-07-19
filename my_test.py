import dwavebinarycsp
import dimod
import itertools

configs = frozenset([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)])
#configs = frozenset(itertools.product([0,1], repeat=3))
constraint = dwavebinarycsp.Constraint.from_configurations(configs, ['a', 'b', 'c'], 'BINARY')
#constraint = dwavecsp.Constraint.from_configurations(configs, ['a', 'a', 'a'], 'BINARY')
print(constraint)

def func(a,b,c):
    return c == a*b
constraint = dwavebinarycsp.Constraint.from_func(func, ['a', 'b', 'c'], 'BINARY')
#constraint = dwavecsp.Constraint.from_func(func, ['a', 'a', 'a'], 'BINARY')
print(constraint)
#constraint.fix_variable('a',1)
#print(constraint)
#constraint.fix_variable('b',1)
#constraint.fix_variable('c',1)


def and_gate(*ins):
    return all(ins[:-1])==ins[-1]
constraint = dwavebinarycsp.Constraint.from_func(and_gate, ['x%d' % i for i in range(3)], 'BINARY')
print(constraint)



print(constraint)
csp = dwavebinarycsp.ConstraintSatisfactionProblem('BINARY')
csp.add_variable('z')
csp.add_constraint(constraint)
print(csp.variables)
bqm, pmodels = dwavebinarycsp.stitch(csp)
print(bqm)
resp = dimod.ExactSolver().sample(bqm)

for sample, energy in resp.data(['sample', 'energy']):
    print(sample, csp.check(sample), energy)