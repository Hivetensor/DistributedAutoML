import json
import re

from deap import creator, gp

from dml.gp_fix import SafePrimitiveTree


def safe_eval(expr):
    # Replace tensor(...) with torch.tensor(...)
    expr = re.sub(r"tensor\((.*?)\)", r"torch.tensor(\1)", expr)
    try:
        return eval(expr)
    except NameError:
        # If it's not evaluable (like a variable name), return as is
        return expr


def save_individual_to_json(individual):
    expr_str = str(individual)
    return json.dumps({"expression": expr_str})


def load_individual_from_json(data=None, pset=None, toolbox=None, filename=None):
    if filename is not None:
        with open(filename, "r") as fd:
            data = json.loads(fd.read())
        if type(data) == str:
            data = json.loads(data)

    expr_str = data["expression"]
    expr = SafePrimitiveTree.from_string(expr_str, pset, safe_eval)
    individual = creator.Individual(expr)
    func = toolbox.compile(expr=individual)
    return individual, func
