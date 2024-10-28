import numpy as np
import torch


class FunctionDecoder:
    def __init__(self):
        self.decoding_map = {
            ## Func, output, input1, input2
            0: (self.do_nothing, "scalar", None, None),
            1: (self.add_scalar, "scalar", "scalar", "scalar"),
            2: (self.sub_scalar, "scalar", "scalar", "scalar"),
            3: (self.multiply_scalar, "scalar", "scalar", "scalar"),
            4: (self.divide_scalar, "scalar", "scalar", "scalar"),
            5: (self.abs_scalar, "scalar", "scalar", None),
            6: (self.reciprocal_scalar, "scalar", "scalar", None),
            7: (self.sin_scalar, "scalar", "scalar", None),
            8: (self.cos_scalar, "scalar", "scalar", None),
            9: (self.tan_scalar, "scalar", "scalar", None),
            10: (self.arcsin_scalar, "scalar", "scalar", None),
            11: (self.arccos_scalar, "scalar", "scalar", None),
            12: (self.arctan_scalar, "scalar", "scalar", None),
            13: (self.sigmoid, "scalar", "scalar", None),
            14: (self.leaky_relu, "scalar", "scalar", None),
            15: (self.relu, "scalar", "scalar", None),
            16: (self.set_constant_scalar, "scalar", None, None),  # TODO reflexive
            17: (self.gaussian_scalar, "scalar", None, None),
            18: (self.gaussian_matrix, "matrix", "matrix", None),
            19: (self.uniform_scalar, "vector", "vector", None),
            20: (self.log_scalar, "scalar", "scalar", None),
            21: (self.power_scalar, "scalar", "scalar", "scalar"),
            22: (self.sqrt_scalar, "scalar", "scalar", None),
            23: (self.max_scalar, "scalar", "scalar", "scalar"),
            24: (self.min_scalar, "scalar", "scalar", "scalar"),
            25: (self.mod_scalar, "scalar", "scalar", "scalar"),
            26: (self.sign_scalar, "scalar", "scalar", None),
            27: (self.floor_scalar, "scalar", "scalar", None),
            28: (self.ceil_scalar, "scalar", "scalar", None),
            29: (self.round_scalar, "scalar", "scalar", None),
            30: (self.hypot_scalar, "scalar", "scalar", "scalar"),
            31: (self.add_scalar, "vector", "vector", "vector"),
            32: (self.sub_scalar, "vector", "vector", "vector"),
            33: (self.multiply_scalar, "vector", "vector", "vector"),
            34: (self.divide_scalar, "vector", "vector", "vector"),
            35: (self.abs_scalar, "vector", "vector", "vector"),
            36: (self.reciprocal_scalar, "vector", "vector", None),
            37: (self.sin_scalar, "vector", "vector", None),
            38: (self.cos_scalar, "vector", "vector", None),
            39: (self.tan_scalar, "vector", "vector", None),
            40: (self.arcsin_scalar, "vector", "vector", None),
            41: (self.arccos_scalar, "vector", "vector", None),
            42: (self.arctan_scalar, "vector", "vector", None),
            43: (self.sigmoid, "vector", "vector", None),
            44: (self.leaky_relu, "vector", "vector", None),
            45: (self.relu, "vector", "vector", None),
            46: (self.stable_softmax, "vector", "vector", None),
            47: (self.mean_vector, "scalar", "vector", None),
            48: (self.std_axis, "scalar", "vector", None),
            49: (self.uniform_vector, "vector", "vector", None),
            50: (self.log_scalar, "vector", "vector", None),
            51: (self.power_scalar, "vector", "vector", "vector"),
            52: (self.sqrt_scalar, "vector", "vector", None),
            53: (self.max_vector, "scalar", "vector", None),
            54: (self.min_vector, "scalar", "vector", None),
            55: (self.mod_scalar, "vector", "vector", "scalar"),
            56: (self.sign_scalar, "vector", "vector", None),
            57: (self.floor_scalar, "vector", "vector", None),
            58: (self.ceil_scalar, "vector", "vector", None),
            59: (self.round_scalar, "vector", "vector", None),
            60: (self.hypot_scalar, "vector", "vector", "vector"),
            61: (self.dot_vector, "scalar", "vector", "vector"),
            62: (self.norm_vector, "scalar", "vector", None),
            63: (self.add_scalar, "matrix", "matrix", "matrix"),
            64: (self.sub_scalar, "matrix", "matrix", "matrix"),
            65: (self.multiply_scalar, "matrix", "matrix", "matrix"),
            66: (self.divide_scalar, "matrix", "matrix", "matrix"),
            67: (self.abs_scalar, "matrix", "matrix", None),
            68: (self.reciprocal_scalar, "matrix", "matrix", None),
            69: (self.sin_scalar, "matrix", "matrix", None),
            70: (self.cos_scalar, "matrix", "matrix", None),
            71: (self.tan_scalar, "matrix", "matrix", None),
            72: (self.arcsin_scalar, "matrix", "matrix", None),
            73: (self.arccos_scalar, "matrix", "matrix", None),
            74: (self.arctan_scalar, "matrix", "matrix", None),
            75: (self.sigmoid, "matrix", "matrix", None),
            76: (self.leaky_relu, "matrix", "matrix", None),
            77: (self.relu, "matrix", "matrix", None),
            78: (self.stable_softmax, "matrix", "matrix", None),
            79: (self.mean_vector, "scalar", "matrix", None),
            80: (self.std_axis, "vector", "matrix", None),
            81: (self.uniform_matrix, "matrix", "matrix", None),
            82: (self.log_scalar, "matrix", "matrix", None),
            83: (self.power_scalar, "matrix", "matrix", "scalar"),
            84: (self.sqrt_scalar, "matrix", "matrix", None),
            85: (self.max_vector, "scalar", "matrix", None),
            86: (self.min_vector, "scalar", "matrix", None),
            87: (self.mod_scalar, "matrix", "matrix", "scalar"),
            88: (self.sign_scalar, "matrix", "matrix", None),
            89: (self.floor_scalar, "matrix", "matrix", None),
            90: (self.ceil_scalar, "matrix", "matrix", None),
            91: (self.round_scalar, "matrix", "matrix", None),
            92: (self.hypot_scalar, "matrix", "matrix", "matrix"),
            93: (self.norm_vector, "scalar", "matrix", None),
            94: (self.multiply_scalar, "vector", "vector", "scalar"),
            95: (self.multiply_scalar, "matrix", "matrix", "scalar"),
            95: (self.broadcast_scalar_to_vector, "vector", "scalar", "vector"),
            96: (self.broadcast_vector_to_matrix_row, "matrix", "vector", "matrix"),
            97: (self.broadcast_vector_to_matrix_col, "matrix", "vector", "matrix"),
            98: (self.outer_product, "matrix", "vector", "vector"),
            99: (self.matmul, "matrix", "matrix", "matrix"),
            100: (self.transpose, "matrix", "matrix", None),
            101: (self.gaussian_vector, "vector", "vector", None),
            102: (self.dot_vector, "scalar", "vector", "vector"),
            # 103: (self.matmul, "matrix", "matrix", "matrix"),
        }

    @staticmethod
    def matmul(*args):
        x, y = args[0], args[1]
        return torch.matmul(x, y)

    @staticmethod
    def uniform_scalar(*args):
        low, high = args[2], args[3]
        return torch.empty(args[0].shape).uniform_(low.item(), high.item())

    @staticmethod
    def uniform_vector(*args):
        low, high = args[2], args[3]
        size = args[0].shape
        return torch.empty(size).uniform_(low.item(), high.item())

    @staticmethod
    def uniform_matrix(*args):
        low, high = args[2], args[3]
        rows, cols = args[0].shape
        return torch.empty(rows, cols).uniform_(low.item(), high.item())

    @staticmethod
    def gaussian_scalar(*args):
        mean, std = args[2], args[3]
        return torch.empty(1).normal_(mean.item(), std.item())

    @staticmethod
    def gaussian_vector(*args):
        mean, std = args[2], args[3]
        size = args[0].shape[1]
        return torch.empty(size).normal_(mean.item(), std.item())

    @staticmethod
    def gaussian_matrix(*args):
        mean, std = args[2], args[3]
        rows, cols = args[0].shape
        return torch.empty(rows, cols).normal_(mean.item(), std.item())

    @staticmethod
    def set_constant_vector(*args):
        value = args[2]
        idx = args[3]
        vector = args[0]
        vector[idx] = value
        return vector

    # TODO fix me
    @staticmethod
    def set_constant_matrix(*args):
        value = args[2]
        rows = args[3]
        cols = args[4]
        matrix = args[0]
        matrix[rows, cols] = value
        return matrix

    @staticmethod
    def transpose(*args):
        return args[0].t()

    @staticmethod
    def outer_product(*args):
        return torch.outer(args[0], args[1])

    @staticmethod
    def broadcast_scalar_to_vector(*args):
        scalar = args[0]
        vector_size = args[1].shape[0]
        return torch.full((vector_size,), scalar.item())

    @staticmethod
    def broadcast_vector_to_matrix_row(*args):
        vector = args[0]
        num_rows = args[1].shape[0]
        return vector.repeat(num_rows, 1)

    @staticmethod
    def broadcast_vector_to_matrix_col(*args):
        vector = args[0]
        num_cols = args[1].shape[1]
        return vector.repeat(num_cols, 1).t()

    @staticmethod
    def norm_vector(*args):
        return torch.norm(args[0], args[2])

    @staticmethod
    def dot_vector(*args):
        return torch.dot(args[0], args[1])

    @staticmethod
    def do_nothing(*args):
        pass  # TODO implement in executor

    @staticmethod
    def identity_scalar(*args):
        return args[0]

    @staticmethod
    def add_scalar(*args):
        return torch.add(args[0], args[1])

    @staticmethod
    def sub_scalar(*args):
        return torch.subtract(args[0], args[1])

    @staticmethod
    def multiply_scalar(*args):
        return torch.multiply(args[0], args[1])

    @staticmethod
    def divide_scalar(*args):
        return torch.divide(args[0], args[1])

    @staticmethod
    def abs_scalar(*args):
        return torch.abs(args[0])

    @staticmethod
    def reciprocal_scalar(*args):
        return torch.reciprocal(args[0])

    @staticmethod
    def sin_scalar(*args):
        return torch.sin(args[0])

    @staticmethod
    def cos_scalar(*args):
        return torch.cos(args[0])

    @staticmethod
    def tan_scalar(*args):
        return torch.tan(args[0])

    @staticmethod
    def arcsin_scalar(*args):
        return torch.arcsin(args[0])

    @staticmethod
    def arccos_scalar(*args):
        return torch.arccos(args[0])

    @staticmethod
    def arctan_scalar(*args):
        return torch.arctan(args[0])

    @staticmethod
    def sigmoid(*args):
        return 1 / (1 + torch.exp(-args[0]))

    @staticmethod
    def leaky_relu(*args, alpha=0.01):
        return torch.where(args[0] > 0, args[0], args[0] * alpha)

    @staticmethod
    def relu(*args):
        return torch.relu(args[0])

    @staticmethod
    def stable_softmax(*args):
        try:
            z = args[0] - torch.max(args[0])
            numerator = torch.exp(z)
            denominator = torch.sum(numerator)
            softmax = numerator / denominator
            return softmax
        except:
            return args[0]

    @staticmethod
    def mean_axis(*args):
        try:
            return torch.mean(args[0], axis=0)
        except:
            return args[0]

    @staticmethod
    def mean_vector(*args):
        return torch.mean(args[0])

    @staticmethod
    def std_axis(*args):
        x = args[0]
        try:
            return torch.std(x, axis=0)
        except:
            return x

    @staticmethod
    def set_constant_scalar(*args):
        return torch.tensor(args[2])

    @staticmethod
    def log_scalar(*args):
        return torch.log(torch.abs(args[0]) + 1e-10)

    @staticmethod
    def power_scalar(*args):
        return torch.pow(args[0], args[1])

    @staticmethod
    def sqrt_scalar(*args):
        x = args[0]
        return torch.sqrt(torch.abs(x))

    @staticmethod
    def max_scalar(*args):
        x, y = args[0], args[1]
        return torch.maximum(x, y)

    @staticmethod
    def max_vector(*args):
        x = args[0]
        return torch.max(x)

    @staticmethod
    def min_scalar(*args):
        x, y = args[0], args[1]
        return torch.minimum(x, y)

    @staticmethod
    def min_vector(*args):
        x = args[0]
        return torch.min(x)

    @staticmethod
    def mod_scalar(*args):
        x, y = args[0], args[1]
        return torch.remainder(x, y)

    @staticmethod
    def sign_scalar(*args):
        x = args[0]
        return torch.sign(x)

    @staticmethod
    def floor_scalar(*args):
        x = args[0]
        return torch.floor(x)

    @staticmethod
    def ceil_scalar(*args):
        x = args[0]
        return torch.ceil(x)

    @staticmethod
    def round_scalar(*args):
        x = args[0]
        return torch.round(x)

    @staticmethod
    def hypot_scalar(*args):
        x, y = args[0], args[1]
        return torch.hypot(x, y)

    def decode(self, genome):
        decoded_functions = []
        for op in genome.gene:
            if op in self.decoding_map:
                decoded_functions.append(self.decoding_map[op])
        return decoded_functions, genome


class NumpyFunctionDecoder:
    def __init__(self):
        self.decoding_map = {
            0: self.do_nothing,
            1: self.add_scalar,
            2: self.sub_scalar,
            3: self.multiply_scalar,
            4: self.divide_scalar,
            5: self.abs_scalar,
            6: self.reciprocal_scalar,
            7: self.sin_scalar,
            8: self.cos_scalar,
            9: self.tan_scalar,
            10: self.arcsin_scalar,
            11: self.arccos_scalar,
            12: self.arctan_scalar,
            13: self.sigmoid,
            14: self.leaky_relu,
            15: self.relu,
            16: self.stable_softmax,
            17: self.mean_axis,
            18: self.std_axis,
            19: self.set_constant_scalar,
            20: self.log_scalar,
            21: self.power_scalar,
            22: self.sqrt_scalar,
            23: self.max_scalar,
            24: self.min_scalar,
            25: self.mod_scalar,
            26: self.sign_scalar,
            27: self.floor_scalar,
            28: self.ceil_scalar,
            29: self.round_scalar,
            30: self.hypot_scalar,
        }

    @staticmethod
    def do_nothing(*args):
        return 0

    @staticmethod
    def identity(*args):
        return args[0]

    @staticmethod
    def add_scalar(*args):
        return np.add(args[0], args[1])

    @staticmethod
    def sub_scalar(*args):
        return np.subtract(args[0], args[1])

    @staticmethod
    def multiply_scalar(*args):
        return np.multiply(args[0], args[1])

    @staticmethod
    def divide_scalar(*args):
        return np.divide(args[0], args[1])

    @staticmethod
    def abs_scalar(*args):
        return np.abs(args[0])

    @staticmethod
    def reciprocal_scalar(*args):
        return np.reciprocal(args[0])

    @staticmethod
    def sin_scalar(*args):
        return np.sin(args[0])

    @staticmethod
    def cos_scalar(*args):
        return np.cos(args[0])

    @staticmethod
    def tan_scalar(*args):
        return np.tan(args[0])

    @staticmethod
    def arcsin_scalar(*args):
        return np.arcsin(args[0])

    @staticmethod
    def arccos_scalar(*args):
        return np.arccos(args[0])

    @staticmethod
    def arctan_scalar(*args):
        return np.arctan(args[0])

    @staticmethod
    def sigmoid(*args):
        return 1 / (1 + np.exp(-args[0]))

    @staticmethod
    def leaky_relu(*args, alpha=0.01):
        return np.where(args[0] > 0, args[0], args[0] * alpha)

    @staticmethod
    def relu(*args):
        x = args[0]
        x[x < 0] = 0
        return x

    @staticmethod
    def stable_softmax(*args):
        try:
            x = args[0]
            z = x - np.max(x)
            numerator = np.exp(z)
            denominator = np.sum(numerator)
            softmax = numerator / denominator
            return softmax
        except:
            return x

    @staticmethod
    def mean_scalar(*args):
        x = args[0]
        try:
            return np.mean(x, axis=0)
        except:
            return x

    @staticmethod
    def mean_axis(*args):
        x = args[0]
        try:
            return np.mean(x, axis=0)
        except:
            return x

    @staticmethod
    def std_axis(*args):
        x = args[0]
        try:
            return np.std(x, axis=0)
        except:
            return x

    @staticmethod
    def set_constant_scalar(c, _):
        return c

    @staticmethod
    def log_scalar(*args):
        return np.log(np.abs(args[0]) + 1e-10)

    @staticmethod
    def power_scalar(*args):
        return np.power(args[0], args[1])

    @staticmethod
    def sqrt_scalar(*args):
        return np.sqrt(np.abs(args[0]))

    @staticmethod
    def max_scalar(*args):
        return np.maximum(args[0], args[1])

    @staticmethod
    def min_scalar(*args):
        return np.minimum(args[0], args[1])

    @staticmethod
    def mod_scalar(*args):
        return np.mod(args[0], args[1])

    @staticmethod
    def sign_scalar(*args):
        return np.sign(args[0])

    @staticmethod
    def floor_scalar(*args):
        return np.floor(args[0])

    @staticmethod
    def ceil_scalar(*args):
        return np.ceil(args[0])

    @staticmethod
    def round_scalar(*args):
        return np.round(args[0])

    @staticmethod
    def hypot_scalar(*args):
        return np.hypot(args[0], args[1])

    def decode(self, genome):
        decoded_functions = []
        for op in genome.gene:
            if op in self.decoding_map:
                decoded_functions.append(self.decoding_map[op])
        return decoded_functions, genome
