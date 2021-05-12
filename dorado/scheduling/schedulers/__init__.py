from docplex.mp.model import Model as _Model
import numpy as np


class Model(_Model):
    """Convenience class to add Numpy variable arrays to docplex.mp."""

    def _var_array(self, vartype, shape=1, *args, **kwargs):
        if shape == 1:
            return self.var(vartype, *args, **kwargs)
        else:
            return np.reshape(
                self.var_list(np.prod(shape), vartype, *args, **kwargs), shape)

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
