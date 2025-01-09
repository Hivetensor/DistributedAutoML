from deap import gp, creator
from dml.gp_fix import SafePrimitiveTree
import json
import re

def convert_tensor_literals(expr: str) -> str:
    """
    Convert tensor literals in expressions to explicit torch.tensor calls.
    
    Args:
        expr: String containing the expression with tensor literals
        
    Returns:
        Modified string with tensor literals converted to torch.tensor calls
        
    Example:
        'safe_add(x, tensor(0.3790))' -> 'safe_add(x, torch.tensor(0.3790))'
    """
    # First pattern matches tensor literals that have just numbers
    expr = re.sub(r'(?<!torch\.)tensor\(([-+]?[0-9]*\.?[0-9]+)\)', 
                  r'torch.tensor(\1)', 
                  expr)
    
    # Second pattern matches tensor literals that might have more complex content
    expr = re.sub(r'(?<!torch\.)tensor\((.*?)\)', 
                  r'torch.tensor(\1)', 
                  expr)
    
    return expr

def safe_eval(expr):
    # Replace tensor(...) with torch.tensor(...)
    #expr = re.sub(r'tensor\((.*?)\)', r'torch.tensor(\1)', expr)
    expr = convert_tensor_literals(expr)
    try:
        return eval(expr)
    except NameError:
        # If it's not evaluable (like a variable name), return as is
        return expr


def save_individual_to_json(individual, hotkey=None):
    expr_str = str(individual)
    return json.dumps({'expression': expr_str, "hotkey":hotkey})

def load_individual_from_json(data=None, pset=None, toolbox=None, filename = None):
    if filename is not None:
        with open(filename, "r") as fd:
            data = json.loads(fd.read())
        if type(data) == str:
            data = json.loads(data)
        
    expr_str = data['expression']
    try:
        hotkey = data['hotkey']
    except:
        hotkey = None
    expr = SafePrimitiveTree.from_string(expr_str, pset, safe_eval)
    individual = creator.Individual(expr)
    func = toolbox.compile(expr=individual)
    return individual, func, expr_str, hotkey