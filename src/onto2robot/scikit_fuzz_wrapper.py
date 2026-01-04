import numpy as np
from skfuzzy import control as ctrl

from onto2robot.core import OntologyIndividualSuperclass, _get_conclusions, _get_left_right_hands, _get_premises


def make_antecedents(
    linguistic_variables_spaces: dict[str, dict[str, OntologyIndividualSuperclass]],
    goal_name: str,
    universe: np.ndarray,
) -> dict[str, ctrl.Antecedent]:
    antecedents = {}
    for lv_name, terms in linguistic_variables_spaces.items():
        # Add once and do not add the ultimate goal (never used as a premise)
        if lv_name != goal_name and lv_name not in antecedents:
            antecedents[lv_name] = ctrl.Antecedent(universe, lv_name)
            antecedents[lv_name].automf(len(terms), names=terms)
    return antecedents


def make_consequents(
    rules: list[OntologyIndividualSuperclass],
    linguistic_variables_spaces: dict[str, dict[str, OntologyIndividualSuperclass]],
    universe: np.ndarray,
):
    consequents = {}

    conclusion_variables = set()
    for rule in rules:
        conclusions = _get_conclusions(rule)
        for conclusion in conclusions:
            left, _ = _get_left_right_hands(conclusion)
            conclusion_variables.add(left.name)

    for lv_name, terms in linguistic_variables_spaces.items():
        # Add once and do not add the ultimate goal (never used as a premise)
        if lv_name in conclusion_variables and lv_name not in consequents:
            consequents[lv_name] = ctrl.Consequent(universe, lv_name, defuzzify_method="mom")
            consequents[lv_name].automf(len(terms), names=terms)
    return consequents


class ScikitFuzzyWrapper:
    def __init__(
        self,
        linguistic_variables_spaces: dict[str, dict[str, OntologyIndividualSuperclass]],
        goal_name: str,
        # TODO: The universe for each variable should be taken from the target system specification
        universe: np.ndarray,
        rules: list[OntologyIndividualSuperclass],
    ):
        self.linguistic_variables_spaces = linguistic_variables_spaces
        self.universe = universe
        self.antecedents = make_antecedents(linguistic_variables_spaces, goal_name, universe)
        self.consequents = make_consequents(rules, linguistic_variables_spaces, universe)
        self._make_rules(rules)
        self.ctrl_system = ctrl.ControlSystem(self.scikit_rules)
        self.sim = ctrl.ControlSystemSimulation(self.ctrl_system)

    def _make_rules(self, rules: list[OntologyIndividualSuperclass]):
        scikit_rules = []

        for rule in rules:
            premises = _get_premises(rule)
            conclusions = _get_conclusions(rule)

            # Build antecedent conditions (premises)
            antecedent_conditions = None
            for premise in premises:
                left, right = _get_left_right_hands(premise)
                var_name = left.name
                term_name = right.name

                if var_name in self.antecedents:
                    condition = self.antecedents[var_name][term_name]
                    if antecedent_conditions is None:
                        antecedent_conditions = condition
                    else:
                        antecedent_conditions = antecedent_conditions & condition

            # Build consequent (conclusion)
            if conclusions:
                conclusion = conclusions[0]
                left, right = _get_left_right_hands(conclusion)
                var_name = left.name
                term_name = right.name

                if var_name in self.consequents:
                    consequent = self.consequents[var_name][term_name]

                    if antecedent_conditions is not None:
                        scikit_rules.append(ctrl.Rule(antecedent_conditions, consequent))

        self.scikit_rules = scikit_rules

    def set_start_values(
        self,
        input_values: dict[str, float],
    ):
        for input in self.sim._get_inputs().items():
            var_name = input[0]
            if var_name in input_values:
                self.sim.input[var_name] = input_values[var_name]
            else:
                self.sim.input[var_name] = (self.universe[1] - self.universe[0]) / 2

    def compute(self, layer: set[OntologyIndividualSuperclass]):
        layer_var_names = [ind.name for ind in layer]
        print(f"Processing layer with targets: {layer_var_names}")

        # Compute inference for this layer
        self.sim.compute()
        print(f" Layer output: {self.sim.output}")

        # Capture and set output values from this layer
        for var_name in layer_var_names:
            if var_name in self.sim.output:
                output_value = self.sim.output[var_name]
                if var_name in self.antecedents:
                    self.sim.input[var_name] = output_value
