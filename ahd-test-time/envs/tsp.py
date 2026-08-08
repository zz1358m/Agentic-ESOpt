class TSPEnv:
    def __init__(self, problem=None):
        self.problem = problem

    def evaluate(self, code):
        if self.problem is None:
            raise NotImplementedError("TSPEnv requires a concrete evaluator.")
        return self.problem.evaluate(code)
