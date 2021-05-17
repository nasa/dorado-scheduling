import cplex
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
from docplex.mp.model import Model as _Model
import numpy as np


class Model(_Model):
    """Convenience class to add Numpy variable arrays to docplex.mp."""

    def _var_array(self, vartype, shape=(), *args, **kwargs):
        size = np.prod(shape, dtype=int)
        vars = np.reshape(self.var_list(size, vartype, *args, **kwargs), shape)
        if vars.ndim == 0:
            vars = vars.item()
        return vars

    def binary_var_array(self, *args, **kwargs):
        return self._var_array(self.binary_vartype, *args, **kwargs)

    def continuous_var_array(self, *args, **kwargs):
        return self._var_array(self.continuous_vartype, *args, **kwargs)

    def integer_var_array(self, *args, **kwargs):
        return self._var_array(self.integer_vartype, *args, **kwargs)

    def semicontinuous_var_array(self, *args, **kwargs):
        return self._var_array(self.semicontinuous_vartype, *args, **kwargs)

    def semiinteger_var_array(self, *args, **kwargs):
        return self._var_array(self.semiinteger_vartype, *args, **kwargs)

    def set_lazy_constraint_callback(self, func, *watched_vars):
        """Set the lazy constraint callback.

        Parameters
        ----------
        func : callabale
            A lazy constraint callback that a single argument of type
            :class:`docplex.mp.solution.SolveSolution`, which is a partial
            solution candidate with all watched variables set. It should return
            either None, or a list of lazy constraints.

        watched_vars
            One more list variables, or lists of variables, to include in
            the partial solution candidates.

        Notes
        -----
        DOCPlex's callback support uses CPLEX's so-called `legacy callback`_
        mechanism, which results in some MIP solver features being disabled.
        This method allows use to add a lazy constraint callback using the
        `generic callback`_ mechanism. It seems to result in considerably
        faster solves.

        .. _`legacy callback`: https://www.ibm.com/docs/en/icos/20.1.0?topic=techniques-using-legacy-optimization-callbacks
        .. _`generic callback`: https://www.ibm.com/docs/en/icos/20.1.0?topic=techniques-generic-callbacks

        """  # noqa: E501
        callback = LazyConstraintCallback(self, func)
        for watched_var in watched_vars:
            callback.register_watched_vars(watched_var)
        self.cplex.set_callback(callback, cplex.callbacks.Context.id.candidate)


class LazyConstraintCallbackBase:

    def get_values(self, indices):
        return self.context.get_candidate_point(indices)


class LazyConstraintCallback(ConstraintCallbackMixin,
                             LazyConstraintCallbackBase):

    def __init__(self, model, func):
        super().__init__()
        self._model = model
        self._func = func

    def invoke(self, context):
        self.context = context
        solution = self.make_solution_from_watched()
        constraints = self._func(solution)
        if constraints:
            unsatisfied = self.get_cpx_unsatisfied_cts(constraints, solution)
            if unsatisfied:
                _, lhs, sense, rhs = zip(*unsatisfied)
                context.reject_candidate(lhs, sense, rhs)
