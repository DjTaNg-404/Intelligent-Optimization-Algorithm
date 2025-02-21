
from Get_Functions_Detail import get_functions_detail


class SA:
    def __init__(self, F, iter = 1000, T0 = 0):
        self.fobj, self.lb, self.ub, self.dim = get_functions_detail(F)
        self.iter = iter

sa = SA("F1")
SA.run()