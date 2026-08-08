class AHDAcoEnv:
    def __init__(self, problem=None):
        self.problem = problem

    def evaluate(self, code):
        if self.problem is None:
            raise NotImplementedError("AHDAcoEnv requires a concrete evaluator.")
        return self.problem.evaluate(code)
