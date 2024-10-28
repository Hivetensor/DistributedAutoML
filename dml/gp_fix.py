import re
from collections import deque

import torch
from deap import gp


class SafePrimitiveTree(gp.PrimitiveTree):
    @classmethod
    def from_string(cls, string, pset, safe_eval=eval):
        """Try to convert a string expression into a PrimitiveTree given a
        PrimitiveSet *pset*. The primitive set needs to contain every primitive
        present in the expression.

        :param string: String representation of a Python expression.
        :param pset: Primitive set from which primitives are selected.
        :returns: PrimitiveTree populated with the deserialized primitives.
        """
        tokens = re.split("[ \t\n\r\f\v(),]", string)
        expr = []
        ret_types = deque()
        for token in tokens:
            if token == "":
                continue
            if len(ret_types) != 0:
                type_ = ret_types.popleft()
            else:
                type_ = None

            if token in pset.mapping:
                primitive = pset.mapping[token]

                if type_ is not None and not issubclass(primitive.ret, type_):
                    raise TypeError(
                        "Primitive {} return type {} does not "
                        "match the expected one: {}.".format(
                            primitive, primitive.ret, type_
                        )
                    )

                expr.append(primitive)
                if isinstance(primitive, gp.Primitive):
                    ret_types.extendleft(reversed(primitive.args))
            else:
                try:
                    token = safe_eval(token)
                except NameError:
                    raise TypeError("Unable to evaluate terminal: {}.".format(token))

                if type_ is None:
                    type_ = type(token)

                if not issubclass(type(token), type_):
                    raise TypeError(
                        "Terminal {} type {} does not "
                        "match the expected one: {}.".format(token, type(token), type_)
                    )

                expr.append(gp.Terminal(token, False, type_))
        return cls(expr)
