# from machinelearning import *
# from mathematics import *
# from optimization import *
# from physics import *
class Probs():
    def __init__(self,paras):

        if not isinstance(paras.problem, str):
            self.prob = paras.problem
            print("- Prob local loaded ")
        elif paras.problem == "tsp_construct":
            from .optimization.tsp_greedy import run
            self.prob = run.TSPCONST()
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "kp_constructive":
            from .optimization.kp_constructive import run
            self.prob = run.KPCONST(paras=paras)
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "asp_constructive":
            from .optimization.asp_constructive import run
            self.prob = run.ASPCONST()
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "tsp_aco":
            from .optimization.tsp_aco import run
            self.prob = run.TSPACO(paras=paras)
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "cvrp_aco":
            from .optimization.cvrp_aco import run
            self.prob = run.CVRPACO(paras=paras)
            print("- Prob "+paras.problem+" loaded ")
        else:
            print("problem "+paras.problem+" not found!")


    def get_problem(self):

        return self.prob
