class GetPrompts:
    def __init__(self):
        self.prompt_task = (
            "Solving 0-1 Knapsack Problem (KP) with constructive heuristics. "
            "KP requires selecting items to maximize the total value under a given "
            "weight constraint, and each item can be selected only once. "
            "Help me design a novel algorithm that selects the next item step by step."
        )
        self.prompt_func_name = "select_next_item"
        self.prompt_func_inputs = ["remaining_capacity", "weights", "values"]
        self.prompt_func_outputs = ["next_item"]
        self.prompt_inout_inf = (
            "'remaining_capacity' is the remaining knapsack capacity. "
            "'weights' and 'values' are Numpy arrays for the currently unselected items. "
            "'next_item' is the integer index of the selected item in these arrays."
        )
        self.prompt_other_inf = (
            "The selected item must have weight no greater than remaining_capacity. "
            "The goal is to maximize total selected value. All arrays are Numpy arrays."
        )

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf
