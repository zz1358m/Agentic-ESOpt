from .selection import equal, prob_rank, roulette_wheel, tournament
from .management import pop_greedy


class Methods:
    def __init__(self, paras, problem) -> None:
        self.paras = paras
        self.problem = problem
        if paras.selection == "prob_rank":
            self.select = prob_rank
        elif paras.selection == "equal":
            self.select = equal
        elif paras.selection == "roulette_wheel":
            self.select = roulette_wheel
        elif paras.selection == "tournament":
            self.select = tournament
        else:
            raise ValueError(f"Unsupported EoH selection method: {paras.selection}")

        if paras.management != "pop_greedy":
            raise ValueError(
                f"Unsupported management method in the maintained EoH runner: {paras.management}"
            )
        self.manage = pop_greedy

    def get_method(self):
        if self.paras.method == "eoh":
            from .eoh.eoh import EOH

            return EOH(self.paras, self.problem, self.select, self.manage)
        raise ValueError(f"Unsupported method in the maintained AHD runner: {self.paras.method}")
