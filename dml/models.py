import torch
import torch.nn as nn


class BaselineNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class EvolvableNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, evolved_activation):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.evolved_activation = evolved_activation

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.evolved_activation(x)
        x = self.fc2(x)
        return x


class EvolvedLoss(torch.nn.Module):
    def __init__(self, genome, device="cpu"):
        super().__init__()
        self.genome = genome
        self.device = device

    def forward(self, outputs, targets):
        # outputs = outputs.detach().float().requires_grad_()#.to(self.device)
        # targets = targets.detach().float().requires_grad_()#.to(self.device)

        memory = self.genome.memory
        memory.reset()

        memory[self.genome.input_addresses[0]] = outputs
        memory[self.genome.input_addresses[1]] = targets
        for i, op in enumerate(self.genome.gene):
            func = self.genome.function_decoder.decoding_map[op][0]
            input1 = memory[self.genome.input_gene[i]]  # .to(self.device)
            input2 = memory[self.genome.input_gene_2[i]]  # .to(self.device)
            constant = torch.tensor(
                self.genome.constants_gene[i], requires_grad=True
            )  # .to(self.device)
            constant_2 = torch.tensor(
                self.genome.constants_gene_2[i], requires_grad=True
            )  # .to(self.device)

            output = func(
                input1,
                input2,
                constant,
                constant_2,
                self.genome.row_fixed,
                self.genome.column_fixed,
            )
            if output is not None:
                memory[self.genome.output_gene[i]] = output

        loss = memory[self.genome.output_addresses[0]]
        return loss
