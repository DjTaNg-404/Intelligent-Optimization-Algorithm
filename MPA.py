from Get_Functions_Detail import get_functions_detail
import numpy as np


def Initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = np.size(ub, 0) if ub.ndim > 0 else 1
    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    elif Boundary_no > 1:
        Positions = np.zeros((SearchAgents_no, dim))  # Initialize the positions matrix
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i
    return Positions


def MPA(SearchAgents_no, Max_iteration, lb, ub, dim, fobj):
    top_predator_pos = np.zeros(dim)
    top_predator_fit = np.inf

    convergence_curve = np.zeros(Max_iteration)
    stepsize = np.zeros((SearchAgents_no, dim))


F = "F1"
fobj, lb, ub, dim = get_functions_detail(F)
search_agents = 25
max_iteration = 1000
Best_score,Best_pos,Convergence_curve = MPA(search_agents, max_iteration, lb, ub, dim, fobj)