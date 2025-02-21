
from scipy.special import gamma
from Get_Functions_Detail import get_functions_detail
import numpy as np


class MPA:

    DEFAULT_FADs = 0.2
    DEFAULT_P = 0.5

    def __init__(self, F, search_agents = 50, max_iteration = 5000, FADs = None, P = None):
        self.fobj, self.lb, self.ub, self.dim = get_functions_detail(F)
        self.search_agents = search_agents
        self.max_iteration = max_iteration

        self.top_predator_pos = np.zeros(self.dim)
        self.top_predator_fit = np.inf

        self.convergence_curve = np.zeros(self.max_iteration)
        self.stepsize = None
        self.fitness = np.full((search_agents, 1), np.inf)

        self.Prey = self.Initialization(
            SearchAgents_no = self.search_agents,
            dim = self.dim,
            ub = self.ub,
            lb = self.lb
        )

        self.X_min = np.tile(np.ones((1, self.dim)) * self.lb, (search_agents, 1))
        self.X_max = np.tile(np.ones((1, self.dim)) * self.lb, (search_agents, 1))

        self.Fads = FADs or self.DEFAULT_FADs
        self.P = P or self.DEFAULT_P

        self.Best_score = None
        self.Best_pos = None


    @staticmethod
    def Initialization(SearchAgents_no, dim, ub, lb):
        Positions = None
        if isinstance(ub, (int, float)):
            ub = np.full(dim, ub)
        if isinstance(lb, (int, float)):
            lb = np.full(dim, lb)

        Boundary_no = ub.shape[0]
        if Boundary_no == 1:
            Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
        elif Boundary_no > 1:
            Positions = np.zeros((SearchAgents_no, dim))
            for i in range(dim):
                ub_i = ub[i]
                lb_i = lb[i]
                Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i

        return Positions

    @staticmethod
    def levy(n, m, beta):

        num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
        den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)

        sigma_u = (num / den) ** (1 / beta)

        u = np.random.normal(0, sigma_u, (n, m))
        v = np.random.normal(0, 1, (n, m))

        z = u / (np.abs(v) ** (1 / beta))

        return z

    def run(self):

        Iter = 0
        fit_old = None
        Prey_old = None

        while Iter < self.max_iteration:
            for i in range(self.Prey.shape[0]):
                Flag4ub = self.Prey[i, :] > self.ub
                Flag4lb = self.Prey[i, :] < self.lb
                self.Prey[i, :] = (self.Prey[i, :] * (~(Flag4lb + Flag4ub))) + self.ub * Flag4ub + self.lb * Flag4lb

                self.fitness[i, 0] = self.fobj(self.Prey[i, :])

                if self.fitness[i, 0] < self.top_predator_fit:
                    self.top_predator_pos = self.Prey[i, :]
                    self.top_predator_fit = self.fitness[i, 0]

            if Iter == 0:
                fit_old = self.fitness
                Prey_old = self.Prey

            Inx = (fit_old < self.fitness)
            Indx = np.tile(Inx, (1, self.dim))
            self.Prey = Indx * Prey_old + ~Indx * self.Prey
            self.fitness = Inx * fit_old + ~Inx * self.fitness

            fit_old = self.fitness
            Prey_old = self.Prey

            # ------------------------------------------------------------------

            Elite = np.tile(self.top_predator_pos, (self.search_agents, 1))
            CF = (1 - Iter / self.max_iteration) ** (2 * Iter / self.max_iteration)

            RL = 0.05 * self.levy(self.search_agents, self.dim, 1.5)
            RB = np.random.randn(self.search_agents, self.dim)

            for i in range(self.Prey.shape[0]):
                for j in range(self.Prey.shape[1]):
                    R = np.random.rand()
                    #----------------------- Phase 1 ----------------------------
                    if Iter < self.max_iteration / 3:
                        self.stepsize = RB[i, j] * (Elite[i, j] - RB[i, j] * self.Prey[i, j])
                        self.Prey[i, j] = self.Prey[i, j] + self.P * R * self.stepsize

                    #----------------------- Phase 2 ----------------------------
                    elif self.max_iteration / 3 <= Iter  < 2 * self.max_iteration / 3:
                        if i > self.Prey.shape[0] / 2:
                            self.stepsize = RB[i, j] * (RB[i, j] * Elite[i, j] - self.Prey[i, j])
                            self.Prey[i, j] = Elite[i, j] + self.P * CF * self.stepsize
                        else:
                            self.stepsize = RL[i, j] * (Elite[i, j] - RL[i, j] * self.Prey[i, j])
                            self.Prey[i, j] = self.Prey[i, j] + self.P * R * self.stepsize

                    # ----------------------- Phase 3 ----------------------------
                    else:
                        self.stepsize = RL[i, j] * (RL[i, j] * Elite[i, j] - self.Prey[i, j])
                        self.Prey[i, j] = Elite[i, j] + self.P * CF * self.stepsize


            for i in range(self.Prey.shape[0]):
                Flag4ub = self.Prey[i, :] > self.ub
                Flag4lb = self.Prey[i, :] < self.lb
                self.Prey[i, :] = (self.Prey[i, :] * (~(Flag4lb + Flag4ub))) + self.ub * Flag4ub + self.lb * Flag4lb

                self.fitness[i, 0] = self.fobj(self.Prey[i, :])

                if self.fitness[i, 0] < self.top_predator_fit:
                    self.top_predator_pos = self.Prey[i, :]
                    self.top_predator_fit = self.fitness[i, 0]

            if Iter == 0:
                fit_old = self.fitness
                Prey_old = self.Prey

            Inx = (fit_old < self.fitness)
            Indx = np.tile(Inx, (1, self.dim))
            self.Prey = Indx * Prey_old + ~Indx * self.Prey
            self.fitness = Inx * fit_old + ~Inx * self.fitness

            fit_old = self.fitness
            Prey_old = self.Prey

            if np.random.rand() < self.Fads:
                U = np.random.rand(self.search_agents, self.dim) < self.Fads
                self.Prey = self.Prey + CF * ((self.X_min + np.random.rand(self.search_agents, self.dim) * (self.X_max - self.X_min)) * U)
            else:
                r = np.random.rand()
                Rs = self.Prey.shape[0]
                self.stepsize = (self.Fads * (1 - r) + r) * (self.Prey[np.random.permutation(Rs), :] - self.Prey[np.random.permutation(Rs), :])
                self.Prey = self.Prey + self.stepsize

            Iter += 1
            self.convergence_curve[Iter - 1] = self.top_predator_fit

        self.Best_pos = self.top_predator_pos
        self.Best_score = self.top_predator_fit

mpa = MPA("F1")
mpa.run()

print("Best Score (最优分数):", mpa.Best_score)
print("Best Position (最优位置):", mpa.Best_pos)
print("Convergence Curve (收敛曲线):")
print(mpa.convergence_curve)