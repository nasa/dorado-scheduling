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
