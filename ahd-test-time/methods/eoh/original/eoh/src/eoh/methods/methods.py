from .selection import prob_rank, equal, roulette_wheel, tournament
from .management import pop_greedy

class Methods():
    def __init__(self,paras,problem) -> None:
        self.paras = paras      
        self.problem = problem
        if paras.selection == "prob_rank":
            self.select = prob_rank
        elif paras.selection == "equal":
            self.select = equal
        elif paras.selection == 'roulette_wheel':
            self.select = roulette_wheel
        elif paras.selection == 'tournament':
            self.select = tournament
        else:
            print("selection method "+paras.selection+" has not been implemented !")
            exit()

        if paras.management != "pop_greedy":
            print("management method "+paras.management+" has not been kept in this paper release!")
            exit()
        self.manage = pop_greedy

        
    def get_method(self):

        if self.paras.method == "eoh":
            from .eoh.eoh import EOH
            return EOH(self.paras,self.problem,self.select,self.manage)
        else:
            print("method "+self.paras.method+" has not been kept in this paper release!")
            exit()
