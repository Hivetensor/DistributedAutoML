import json

from deap import gp

from dml.ops import *

ALLOWED_PRIMITIVES = {
    "safe_add": (safe_add, 2),
    "safe_sub": (safe_sub, 2),
    "safe_mul": (safe_mul, 2),
    "safe_div": (safe_div, 2),
    "safe_sigmoid": (safe_sigmoid, 1),
    "safe_relu": (safe_relu, 1),
    "safe_tanh": (safe_tanh, 1),
}


def serialize_primitive_tree(tree):
    def serialize_node(node):
        if isinstance(node, gp.Primitive):
            return {"type": "primitive", "name": node.name, "arity": node.arity}
        elif isinstance(node, gp.Terminal):
            return {
                "type": "terminal",
                "name": node.name,
                "value": node.value
                if isinstance(node.value, (int, float))
                else str(node.value),
            }
        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    serialized = json.dumps([serialize_node(node) for node in tree])
    print(f"Serialized tree: {serialized}")  # Debug print
    return serialized
