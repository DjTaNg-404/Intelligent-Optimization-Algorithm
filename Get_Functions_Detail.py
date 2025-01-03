import numpy as np


def get_functions_detail(F):
    func = None
    lb = None
    ub = None
    dim = None

    function_map = {
        "F1": (F1, -100, 100, 50),
        "F2": (F2, -10, 10, 50),
        "F3": (F3, -100, 100, 50),
        "F4": (F4, -100, 100, 50),
        "F5": (F5, -30, 30, 50),
        "F6": (F6, -100, 100, 50),
        "F7": (F7, -1.28, 1.28, 50),
        "F8": (F8, -500, 500, 50),
        "F9": (F9, -5.12, 5.12, 50),
        "F10": (F10, -32, 32, 50),
        "F11": (F11, -600, 600, 50),
        "F12": (F12, -50, 50, 50),
        "F13": (F13, -50, 50, 50),
        "F14": (F14, -65.536, 65.536, 2),
        "F15": (F15, -5, 5, 4),
        "F16": (F16, -5, 5, 2),
        "F17": (F17, [-5, 0], [10, 15], 2),
        "F18": (F18, -2, 2, 2),
        "F19": (F19, 0, 1, 3),
        "F20": (F20, 0, 1, 6),
        "F21": (F21, 0, 10, 4),
        "F22": (F22, 0, 10, 4),
        "F23": (F23, 0, 10, 4),
    }

    if F in function_map:
        func, lb, ub, dim = function_map[F]
    else:
        print(f"Error: Function {F} is not defined!")

    return func, lb, ub, dim


def F1(x):
    return np.sum(x ** 2)


def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def F3(x):
    dim = len(x)
    o = 0
    for i in range(dim):
        o += np.sum(x[:i + 1]) ** 2
    return o


def F4(x):
    return np.max(np.abs(x))


def F5(x):
    dim = len(x)
    return np.sum(100 * (x[1:dim] - x[:dim - 1] ** 2) ** 2 + (x[:dim - 1] - 1) ** 2)


def F6(x):
    return np.sum(np.abs(x + 0.5) ** 2)


def F7(x):
    dim = len(x)
    return np.sum(np.arange(1, dim + 1) * (x ** 4)) + np.random.rand()


def F8(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))


def F9(x):
    dim = len(x)
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)) + 10 * dim


def F10(x):
    dim = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim)) - np.exp(
        np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1)


def F11(x):
    dim = len(x)
    return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1)))) + 1


def F12(x):
    dim = len(x)

    def Ufun(x, a, k, m):
        return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a)

    return (np.pi / dim) * (10 * ((np.sin(np.pi * (1 + (x[0] + 1) / 4))) ** 2) +
                            np.sum(
                                ((x[:-1] + 1) / 4) ** 2 * (1 + 10 * ((np.sin(np.pi * (1 + (x[1:] + 1) / 4)))) ** 2)) +
                            ((x[-1] + 1) / 4) ** 2) + np.sum(Ufun(x, 10, 100, 4))


def F13(x):
    dim = len(x)

    def Ufun(x, a, k, m):
        return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a)

    return (0.1 * ((np.sin(3 * np.pi * x[0])) ** 2 +
                   np.sum((x[:-1] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[1:])) ** 2)) +
                   ((x[-1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * x[-1])) ** 2)) +
            np.sum(Ufun(x, 5, 100, 4)))


def F14(x):
    aS = np.array([
        [-32, -16, 0, 16, 32] * 5,
        [-32] * 5 + [-16] * 5 + [0] * 5 + [16] * 5 + [32] * 5
    ])
    bS = [np.sum((x - aS[:, j]) ** 6) for j in range(25)]
    return (1 / 500 + np.sum(1 / (np.arange(1, 26) + bS))) ** -1


def F15(x):
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    bK = 1 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    return np.sum((aK - ((x[0] * (bK ** 2 + x[1] * bK)) / (bK ** 2 + x[2] * bK + x[3]))) ** 2)


def F16(x):
    return 4 * (x[0] ** 2) - 2.1 * (x[0] ** 4) + (x[0] ** 6) / 3 + x[0] * x[1] - 4 * (x[1] ** 2) + 4 * (x[1] ** 4)


def F17(x):
    return ((x[1] - (x[0] ** 2) * 5.1 / (4 * (np.pi ** 2)) + 5 / np.pi * x[0] - 6) ** 2 +
            10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10)


def F18(x):
    return ((1 + (x[0] + x[1] + 1) ** 2 *
             (19 - 14 * x[0] + 3 * (x[0] ** 2) - 14 * x[1] + 6 * x[0] * x[1] + 3 * (x[1] ** 2))) *
            (30 + (2 * x[0] - 3 * x[1]) ** 2 *
             (18 - 32 * x[0] + 12 * (x[0] ** 2) + 48 * x[1] - 36 * x[0] * x[1] + 27 * (x[1] ** 2))))


def F19(x):
    aH = np.array([
        [3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]
    ])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([
        [0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]
    ])
    o = 0
    for i in range(4):
        o -= cH[i] * np.exp(-np.sum(aH[i] * ((x - pH[i]) ** 2)))
    return o


def F20(x):
    aH = np.array([
        [10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]
    ])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]
    ])
    o = 0
    for i in range(4):
        o -= cH[i] * np.exp(-np.sum(aH[i] * ((x - pH[i]) ** 2)))
    return o


def F21(x):
    aSH = np.array([
        [4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
        [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]
    ])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    o = 0
    for i in range(5):
        o -= 1 / (np.dot(x - aSH[i], x - aSH[i]) + cSH[i])
    return o


def F22(x):
    aSH = np.array([
        [4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
        [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]
    ])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    o = 0
    for i in range(7):
        o -= 1 / (np.dot(x - aSH[i], x - aSH[i]) + cSH[i])
    return o


def F23(x):
    aSH = np.array([
        [4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
        [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]
    ])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    o = 0
    for i in range(10):
        o -= 1 / (np.dot(x - aSH[i], x - aSH[i]) + cSH[i])
    return o


def Ufun(x, a, k, m):
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a)
