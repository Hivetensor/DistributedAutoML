import logging
import math
import operator
import os
import pickle
import random
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from deap import base, creator, gp, tools

from dml.ops import *

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Local storage path
LOCAL_STORAGE_PATH = "./best_individuals"
os.makedirs(LOCAL_STORAGE_PATH, exist_ok=True)


# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Custom function for safe division
# def safe_div(x, y):
#     return torch.where(y != 0, x / y, torch.zeros_like(x))
device = torch.device("cpu")

# # Simplified primitive set
# pset = gp.PrimitiveSet("MAIN", 2)
# pset.addPrimitive(torch.add, 2)
# pset.addPrimitive(torch.mul, 2)
# pset.addPrimitive(torch.sub, 2)
# pset.addPrimitive(safe_div, 2)
# pset.addTerminal(1.0)
# pset.addTerminal(0.1)
# pset.renameArguments(ARG0='y_pred')
# pset.renameArguments(ARG1='y_true')
# # Adding activation functions to the primitive set
# pset.addPrimitive(torch.sigmoid, 1)
# pset.addPrimitive(torch.tanh, 1)
# pset.addPrimitive(torch.relu, 1)


# Update the primitive set after defining device
pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(safe_add, 2)
pset.addPrimitive(safe_sub, 2)
pset.addPrimitive(safe_mul, 2)
pset.addPrimitive(safe_div, 2)
pset.addPrimitive(safe_sigmoid, 1)
pset.addPrimitive(safe_relu, 1)
pset.addPrimitive(safe_tanh, 1)
pset.addEphemeralConstant(
    name="const", ephemeral=lambda: torch.tensor(random.uniform(-1, 1), device=device)
)
pset.renameArguments(ARG0="y_pred")
pset.renameArguments(ARG1="y_true")
pset.addTerminal(torch.tensor(1.0, device=device), name="one")
pset.addTerminal(torch.tensor(0.1, device=device), name="zero_point_one")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def safe_evaluate(func, outputs, labels):
    try:
        loss = func(outputs, labels)

        if loss is None:
            logging.error(f"Loss function returned None: {func}")
            return torch.tensor(float("inf"), device=outputs.device)

        if not torch.is_tensor(loss):
            logging.error(f"Loss function didn't return a tensor: {type(loss)}")
            return torch.tensor(float("inf"), device=outputs.device)

        if not torch.isfinite(loss).all():
            logging.warning(f"Non-finite loss detected: {loss}")
            return torch.tensor(float("inf"), device=outputs.device)

        if loss.ndim > 0:
            loss = loss.mean()

        return loss
    except Exception as e:
        logging.error(f"Error in loss calculation: {str(e)}")
        logging.error(traceback.format_exc())
        return torch.tensor(float("inf"), device=outputs.device)


def eval_loss_function(individual, trainloader, valloader, device, epochs=5):
    func = toolbox.compile(expr=individual)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for idx, (inputs, labels) in enumerate(trainloader):
            if idx >= 2:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            labels_one_hot = torch.zeros(labels.size(0), 10, device=device)
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)

            loss = safe_evaluate(func, outputs, labels_one_hot)
            if torch.isinf(loss):
                return (0.0,)  # Return lowest possible accuracy if loss is invalid
            try:
                loss.backward()
            except:
                return (0.0,)
            optimizer.step()

    # Evaluation on validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valloader):
            if idx >= 10:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return (accuracy,)  # Return accuracy as fitness


toolbox.register("evaluate", eval_loss_function)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def main():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )
    valloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    logging.info("Starting evolutionary process")

    for gen in range(10):  # Reduced to 10 generations for quicker debugging
        logging.info(f"Generation {gen}")

        # Evaluate the entire population
        for i, ind in enumerate(pop):
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(
                    ind, trainloader, valloader, device
                )
            logging.debug(
                f"Gen {gen}, Individual {i}: Fitness = {ind.fitness.values[0]}"
            )

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        for i, ind in enumerate(invalid_ind):
            ind.fitness.values = toolbox.evaluate(ind, trainloader, valloader, device)
            logging.debug(
                f"Gen {gen}, New Individual {i}: Fitness = {ind.fitness.values[0]}"
            )

        # Update the hall of fame with the generated individuals
        hof.update(pop)

        # Gather all the fitnesses in one list and print the stats
        fits = [
            ind.fitness.values[0]
            for ind in pop
            if not math.isinf(ind.fitness.values[0])
        ]

        length = len(pop)
        mean = sum(fits) / length if fits else float("inf")
        sum2 = sum(x * x for x in fits) if fits else float("inf")
        std = abs(sum2 / length - mean * 2) * 0.5 if fits else float("inf")

        logging.info(f"Generation {gen}: Valid fits: {len(fits)}/{length}")
        logging.info(f"Min {min(fits) if fits else float('inf')}")
        logging.info(f"Max {max(fits) if fits else float('inf')}")
        logging.info(f"Avg {mean}")
        logging.info(f"Std {std}")

        # Save the best individual
        best = tools.selBest(pop, 1)[0]
        with open(
            os.path.join(LOCAL_STORAGE_PATH, f"best_individual_gen_{gen}.pkl"), "wb"
        ) as f:
            pickle.dump(best, f)

    logging.info("Evolution finished")

    best = hof[0]
    logging.info(f"Best individual: {str(best)}")
    logging.info(f"Best fitness: {best.fitness.values[0]}")


if __name__ == "__main__":
    main()
