"""
Divide and concur:

A dimod solver implementing the divide and concur algorithm

Notes from Alex:
-numpy array will take 1/10th the memory of dicts, but switching between
them may be slow

- to ask sampler for just one solution, use
response = sampler.sample(bqm, num_reads = 1)
sample = next(response.samples(1))



"""
from __future__ import division
from dimod import Sampler
from dimod import Response, SampleView
from dimod.utilities import ising_energy
import dimod
from dwavebinarycsp import ConstraintSatisfactionProblem
import numpy as np
import random
from six import itervalues, iteritems


class DivideAndConcur(Sampler):
    """A simple exact solver, intended for testing and debugging.

    Notes:
        This solver starts to become slow for problems with 18 or more
        variables.

    """

    def __init__(self, factors, factor_sampler, difference_mapping = True):
        Sampler.__init__(self)
        # factors should be a list of bqms
        self._factors = factors
        self._factor_sampler = factor_sampler
        self._difference_mapping = difference_mapping


    @property
    def properties(self):
        # fill this in.
        return dict()


    @property
    def parameters(self):
        # fill this in.
        return dict()

    @staticmethod
    def _average_messages(messages, factors_per_variable):
        # compute the message for each variable, averaged over all factors containing that variable,
        # and update the messages to that average.
        for variable, factors in iteritems(factors_per_variable):
            avg = sum([messages[f][variable] for f in factors])/len(factors)
            for f in factors:
                messages[f][variable] = avg

    def sample(self, bqm, verbosity = 1):
        """Solves the bqm problem.

        The Ising problem should differ from the sum of the factors by at most a linear term.

        """
        if self._factors[0].vartype is not dimod.BINARY:
            raise NotImplementedError

        # find all variables:
        all_variables = set()
        for factor in self._factors:
            all_variables.update(factor.variables)

        # find factors associated with each variable
        factors_per_variable = {v: list() for v in all_variables}
        for factor in self._factors:
            for v in factor.variables:
                factors_per_variable[v].append(factor)

        # remove variables associated with a single factor from consideration
        all_variables = [v for v in all_variables if len(factors_per_variable[v]) > 1]
        factors_per_variable = {v: factors_per_variable[v] for v in all_variables}

        # define variables for each factor
        variables_per_factor = dict()
        for factor in self._factors:
            variables_per_factor[factor] = [v for v in factor.variables if v in all_variables]

        # initialize messages (factor->variable, variable->factor)
        # each message is (energy at 1) - (energy at 0)
        initialization_scale = 0.
        fv_messages = dict()
        vf_messages = dict()
        new_vf_messages = dict()
        for factor in self._factors:
            fv_messages[factor] = dict()
            vf_messages[factor] = dict()
            new_vf_messages[factor] = dict()
            for variable in variables_per_factor[factor]:
                vf_messages[factor][variable] = initialization_scale * (1-2*random.random())

        iter_count = 0
        max_iters = 100
        # multiplier for linear biases:
        mult = 1./max([len(factors_per_variable[v]) for v in all_variables]+[3])
        # repeat:
        finished = False
        solution = dict()
        diff = dict()
        while not finished:

            # "DIVIDE"
            # for each factor, computed updated ising model, draw sample
            for factor in self._factors:
                # linear update (translate messages into linear biases)
                biased_bqm = factor.bqm.copy()
                for variable in variables_per_factor[factor]:
                    biased_bqm.linear[variable] += vf_messages[factor][variable]  # assume we have binary vartype

                # solve a factor
                response = self._factor_sampler.sample(biased_bqm) #, num_reads=1)
                sample = next(response.samples(1, sorted_by='energy'))

                # translate samples into messages
                fv_messages[factor] = {v: mult*(1-2*sample[v]) for v in variables_per_factor[factor]}

                # record the solution values
                solution.update(sample)

            # "CONCUR"
            if self._difference_mapping:
                # compute differences and add twice their amounts
                for factor in self._factors:
                    diff[factor] = {v: fv_messages[factor][v] - vf_messages[factor][v] for v in variables_per_factor[factor]}
                    new_vf_messages[factor] = {v: vf_messages[factor][v] + 2*diff[factor][v] for v in variables_per_factor[factor]}

                # average the results
                self._average_messages(new_vf_messages, factors_per_variable)

                # subtract the differences
                for factor in self._factors:
                    new_vf_messages[factor] = {v: new_vf_messages[factor][v] - diff[factor][v] for v in
                                               variables_per_factor[factor]}
            else:
                # otherwise, just average results for each variable
                new_vf_messages = fv_messages
                self._average_messages(new_vf_messages, factors_per_variable)

            # update full solution
            for variable in all_variables:
                solution[v] = int(sum([new_vf_messages[f][variable] for f in factors_per_variable[variable]]) < 0)
            # possible termination condition for CSP: all constraints are satisfied

            # termination condition: no change in messages
            max_message_change = max(abs(new_vf_messages[f][v]-vf_messages[f][v]) for f in self._factors for v in
                                     variables_per_factor[f])
            if max_message_change < 1e-3:
                finished = True

            # finally, overwrite the messages
            vf_messages = new_vf_messages.copy()

            # report some statistics
            if verbosity >= 1:
                print("Iteration %d" % iter_count)
                message_size = max(abs(m) for m in new_vf_messages[f].values() for f in self._factors)
                print "\tMaximum message size: %0.3g" % message_size
                print "\tMaximum message change: %0.3g" % max_message_change
                unanimous_variables = sum(int(abs(sum(fv_messages[f][v] for f in factors_per_variable[v]))==mult*len(
                    factors_per_variable[v])) for v in all_variables)
                print "\tUnanimous variables: %d" % unanimous_variables
                energy = bqm.energy(solution)
                print "\tCurrent solution:", {v: solution[v] for v in all_variables}
                print "\tEnergy: %0.3g" % energy

            # termination condition: maximum number of iterations
            iter_count += 1
            if iter_count == max_iters:
                finished = True

        # having settled on values for the message variables, find values for the other variables.
        for factor in self._factors:
            biased_bqm = factor.bqm.copy()
            for variable in variables_per_factor[factor]:
                biased_bqm.fix_variable(variable, solution[variable])

            response = self._factor_sampler.sample(biased_bqm)  # , num_reads=1)
            sample = next(response.samples(1, sorted_by='energy'))
            solution.update(sample)

        # add the sample to the response object
        energy = bqm.energy(solution)
        if verbosity >= 1:
            print "Final energy: %0.3g" % energy
        return Response.from_dicts([solution], {'energy': [energy]}, vartype=bqm.vartype)



