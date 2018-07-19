import dimod
from dwavebinarycsp import ConstraintSatisfactionProblem


class Factor(ConstraintSatisfactionProblem):
    """A factor contains:
    - a list of constraints
    - a penalty model (factor.pmodel), such that the ground states are some (but not necessarily all)
    of the configurations which satisfy all constraints
    - a list of real (non-auxiliary) variables.
    - So, basically a constrained optimization problem. As such ,for now,
    we implement it as a CSP subclass with a bqm model.

    For debugging, it's also helpful for factors to have names.
    """
    def __init__(self, vartype, name=None):
        super(Factor, self).__init__(vartype)
        self.bqm = dimod.BinaryQuadraticModel.empty(vartype)
        self.name = name

    def update_bqm(self, bqm):
        self.bqm.update(bqm)


def build_factors(csp, all_pmodels, bqm=None, csp_scale=2):
    # need penalty models associated with each constraint
    """
    Alex says modify sticher.py to include a function pm = f(constraint),
    then return pms with constraints as well as bqm.

    Alex suggested iterators (better for memory management because the whole list of things you're
    iterating over is not stored in memory) but in this case don't need. Just

    for const in csp.constraints:
        pmodel = get_minimal_pmodel(const)
        bqm.update(pmodel.model)

    ldpc contains:
    """

    all_factors = list()
    const_count = 0
    for const in csp.constraints:
        factor = Factor(csp.vartype, 'F%d' % const_count)
        const_count += 1
        factor.add_constraint(const)
        factor.update_bqm(all_pmodels[const].model)
        all_factors.append(factor)

    if bqm is None:
        # no additional linear terms
        return all_factors

    # compute linear biased from bqm
    unbiased_bqm = dimod.BinaryQuadraticModel.empty(vartype=bqm.vartype)
    for factor in all_factors:
        unbiased_bqm.update(factor.bqm)

    bias = unbiased_bqm.copy()
    bias.scale(-csp_scale)  # scale of unbiased bqm relative to biases
    bias.update(bqm)
    #if bias.quadratic:
    #    # actually, could be aux interactions here
    #    raise ValueError('Only linear terms should appear in bias')

    # compute variable count:
    variable_count = {v: 0 for v in csp.variables}
    for factor in all_factors:
        for v in factor.variables:
            variable_count[v] += 1

    check =  {v: bias.linear[v] for v in csp.variables}
    per_factor_bias = {v: bias.linear[v]/variable_count[v] for v in csp.variables}

    # update factors with per-factor biases:
    for factor in all_factors:
        linear_update = dimod.BinaryQuadraticModel.empty(vartype=bqm.vartype)
        linear_update.add_variables_from({v: per_factor_bias[v] for v in factor.variables})
        factor.bqm.update(linear_update)

    return all_factors