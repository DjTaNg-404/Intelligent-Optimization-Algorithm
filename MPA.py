
from scipy.special import gamma
import math
from Get_Functions_Detail import get_functions_detail
import numpy as np


def Initialization(SearchAgents_no, dim, ub, lb):
    if isinstance(ub, (int, float)):
        ub = np.full(dim, ub)
    if isinstance(lb, (int, float)):
        lb = np.full(dim, lb)

    Boundary_no = ub.shape[0]
    if Boundary_no == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    elif Boundary_no > 1:
        Positions = np.zeros((SearchAgents_no, dim))  # Initialize the positions matrix
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i
    return Positions

def levy(n, m, beta):

    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)

    sigma_u = (num / den) ** (1 / beta)

    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))

    z = u / (np.abs(v) ** (1 / beta))

    return z

def MPA(SearchAgents_no, Max_iteration, lb, ub, dim, fobj):
    top_predator_pos = np.zeros(dim)
    top_predator_fit = np.inf

    Convergence_curve = np.zeros(Max_iteration)
    stepsize = np.zeros((SearchAgents_no, dim))
    fitness = np.full((SearchAgents_no, 1), np.inf)

    Prey = Initialization(SearchAgents_no, dim, ub, lb)

    Xmin = np.tile(np.ones((1, dim)) * lb, (SearchAgents_no, 1))
    Xmax = np.tile(np.ones((1, dim)) * ub, (SearchAgents_no, 1))

    Iter = 0;
    FADs = 0.2;
    P = 0.5;

    while Iter < Max_iteration:
        # ------------------- Detecting top predator -----------------
        for i in range(Prey.shape[0]):
            Flag4ub = Prey[i, :] > ub
            Flag4lb = Prey[i, :] < lb
            Prey[i, :] = (Prey[i, :] * (~(Flag4lb + Flag4ub))) + ub * Flag4ub + lb * Flag4lb

            fitness[i, 0] = fobj(Prey[i, :])

            if fitness[i, 0] < top_predator_fit:
                top_predator_pos = Prey[i, :]
                top_predator_fit = fitness[i, 0]
        # ------------------- Marine Memory saving -------------------
        if Iter == 0:
            fit_old = fitness
            Prey_old = Prey

        Inx = (fit_old < fitness)
        Indx = np.tile(Inx, (1, dim))
        Prey = Indx * Prey_old + ~Indx * Prey
        fitness = Inx * fit_old + ~Inx * fitness

        fit_old = fitness
        Prey_old = Prey
        # ------------------------------------------------------------

        Elite = np.tile(top_predator_pos, (SearchAgents_no, 1))
        CF = (1 - Iter / Max_iteration) ** (2 * Iter / Max_iteration)

        RL = 0.05 * levy(SearchAgents_no, dim, 1.5)
        RB = np.random.randn(SearchAgents_no, dim)

        for i in range(Prey.shape[0]):
            for j in range(Prey.shape[1]):
                R = np.random.rand()
                # ------------------ Phase 1 (Eq.12) -------------------
                if Iter < Max_iteration / 3:
                    stepsize = RB[i, j] * (Elite[i, j] - RB[i, j] * Prey[i, j])
                    Prey[i, j] = Prey[i, j] + P * R * stepsize

                # --------------- Phase 2 (Eqs. 13 & 14)----------------
                elif Iter > Max_iteration / 3 and Iter < 2 * Max_iteration / 3:
                    if i > Prey.shape[0] / 2:
                        stepsize = RB[i, j] * (RB[i, j] * Elite[i, j] - Prey[i, j])
                        Prey[i, j] = Elite[i, j] + P * CF * stepsize
                    else:
                        stepsize = RL[i, j] * (Elite[i, j] - RL[i, j] * Prey[i, j])
                        Prey[i, j] = Prey[i, j] + P * R * stepsize

                # ----------------- Phase 3 (Eq. 15)-------------------
                else:
                    stepsize = RL[i, j] * (RL[i, j] * Elite[i, j] - Prey[i, j])
                    Prey[i, j] = Elite[i, j] + P * CF * stepsize

        # ------------------ Detecting top predator ------------------
        for i in range(Prey.shape[0]):
            Flag4ub = Prey[i, :] > ub
            Flag4lb = Prey[i, :] < lb
            Prey[i, :] = (Prey[i, :] * (~(Flag4lb + Flag4ub))) + ub * Flag4ub + lb * Flag4lb

            fitness[i, 0] = fobj(Prey[i, :])

            if fitness[i, 0] < top_predator_fit:
                top_predator_pos = Prey[i, :]
                top_predator_fit = fitness[i, 0]

        # ------------------- Marine Memory saving -------------------
        if Iter == 0:
            fit_old = fitness
            Prey_old = Prey

        Inx = (fit_old < fitness)
        Indx = np.tile(Inx, (1, dim))
        Prey = Indx * Prey_old + ~Indx * Prey
        fitness = Inx * fit_old + ~Inx * fitness

        fit_old = fitness
        Prey_old = Prey

        # ---------- Eddy formation and FADs' effect (Eq 16) -----------
        if np.random.rand() < FADs:
            U = np.random.rand(SearchAgents_no, dim) < FADs
            Prey = Prey + CF * ((Xmin + np.random.rand(SearchAgents_no, dim) * (Xmax - Xmin)) * U)
        else:
            r = np.random.rand()
            Rs = Prey.shape[0]
            stepsize = (FADs * (1 - r) + r) * (Prey[np.random.permutation(Rs), :] - Prey[np.random.permutation(Rs), :])
            Prey = Prey + stepsize

        Iter += 1
        Convergence_curve[Iter - 1] = top_predator_fit


    return top_predator_fit, top_predator_pos, Convergence_curve

F = "F1"
fobj, lb, ub, dim = get_functions_detail(F)
search_agents = 50
max_iteration = 5000
Best_score,Best_pos,Convergence_curve = MPA(search_agents, max_iteration, lb, ub, dim, fobj)
print("Best Score (最优分数):", Best_score)
print("Best Position (最优位置):", Best_pos)
print("Convergence Curve (收敛曲线):")
print(Convergence_curve)