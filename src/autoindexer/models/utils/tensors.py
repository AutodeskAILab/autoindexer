import torch


def print_matrix(matrix, num_digits=3):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.tolist()
    print(get_matrix_str(matrix, num_digits))


def get_matrix_str(matrix, num_digits, indent=1):
    string = "["
    if isinstance(matrix[0], int):
        string += ", ".join([f"{d:{num_digits}}" for d in matrix])
    else:
        separator = ",\n" + " " * indent
        string += separator.join([get_matrix_str(row, num_digits, indent + 1) for row in matrix])
    return string + "]"
