import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import trimf

from onto2robot.core import (
    LinguisticVariableDomain,
    OntologyIndividualSuperclass,
    _get_conclusions,
    _get_left_right_hands,
    _get_premises,
)


def make_antecedents(
    linguistic_variables_spaces: dict[OntologyIndividualSuperclass, LinguisticVariableDomain],
    goal_name: str,
    membership_functions: dict[str, list[int]] | None = None,
    use_auto_membership: bool = False,
) -> dict[OntologyIndividualSuperclass, ctrl.Antecedent]:
    antecedents: dict[OntologyIndividualSuperclass, ctrl.Antecedent] = {}
    for variable, set_of_values in linguistic_variables_spaces.items():
        # Add once and do not add the ultimate goal (never used as a premise)
        if variable.name != goal_name and variable not in antecedents:
            universe = np.arange(set_of_values.fuzzy_points[0], set_of_values.fuzzy_points[-1], 1)
            antecedents[variable] = ctrl.Antecedent(universe, variable.name)
            if not membership_functions:
                if (
                    len(set_of_values.fuzzy_points) == 3 * len(set_of_values.linguistic_domain)
                    and not use_auto_membership
                ):
                    for id, ld in enumerate(set_of_values.linguistic_domain):
                        antecedents[variable][ld.name] = trimf(
                            universe, set_of_values.fuzzy_points[id * 3 : id * 3 + 3]
                        )
                else:
                    antecedents[variable].automf(
                        len(set_of_values.linguistic_domain),
                        names=[value.name for value in set_of_values.linguistic_domain],
                    )
            else:
                for term in set_of_values.linguistic_domain:
                    antecedents[variable][term] = trimf(antecedents[variable].universe, membership_functions[term.name])
    print("Created antecedents")
    for name, ant in antecedents.items():
        print(f"Antecedent '{name.name}' has terms: {list(ant.terms.keys())}")
    return antecedents


def make_consequents(
    rules: list[OntologyIndividualSuperclass],
    linguistic_variables_spaces: dict[OntologyIndividualSuperclass, LinguisticVariableDomain],
    membership_functions: dict[str, list[int]] | None = None,
    use_auto_membership: bool = False,
):
    print(f"Creating consequents based on rules {rules}.")
    consequents: dict[OntologyIndividualSuperclass, ctrl.Consequent] = {}

    conclusion_variables: set[OntologyIndividualSuperclass] = set()
    for rule in rules:
        conclusions = _get_conclusions(rule)
        for conclusion in conclusions:
            left, _ = _get_left_right_hands(conclusion)
            conclusion_variables.add(left)

    for variable, set_of_values in linguistic_variables_spaces.items():
        universe = np.arange(set_of_values.fuzzy_points[0], set_of_values.fuzzy_points[-1], 1)
        # Add once and do not add the ultimate goal (never used as a premise)
        if variable in conclusion_variables and variable not in consequents:
            consequents[variable] = ctrl.Consequent(universe, variable.name, defuzzify_method="mom")
            if not membership_functions:
                if (
                    len(set_of_values.fuzzy_points) == 3 * len(set_of_values.linguistic_domain)
                    and not use_auto_membership
                ):
                    for id, ld in enumerate(set_of_values.linguistic_domain):
                        consequents[variable][ld.name] = trimf(
                            universe, set_of_values.fuzzy_points[id * 3 : id * 3 + 3]
                        )
                else:
                    consequents[variable].automf(
                        len(set_of_values.linguistic_domain),
                        names=[value.name for value in set_of_values.linguistic_domain],
                    )
            else:
                for term in set_of_values.linguistic_domain:
                    consequents[variable][term] = trimf(consequents[variable].universe, membership_functions[term.name])
    # print(f"Consequents created for variables: {list(consequents.keys())}")
    return consequents


class ScikitFuzzyWrapper:
    def __init__(
        self,
        linguistic_variables_domains: dict[OntologyIndividualSuperclass, LinguisticVariableDomain],
        goal_name: str,
        rules: list[OntologyIndividualSuperclass],
    ):
        # mem_funcs = {
        #     "low": [0, 0, 20],
        #     "middle": [10, 20, 30],
        #     "high": [20, 40, 40],
        #     "left": [0, 0, 20],
        #     "forward": [10, 20, 30],
        #     "right": [20, 40, 40],
        # }
        self.linguistic_variables_spaces = linguistic_variables_domains
        self.antecedents = make_antecedents(linguistic_variables_domains, goal_name, use_auto_membership=False)
        self.consequents = make_consequents(rules, linguistic_variables_domains, use_auto_membership=False)
        self._make_rules(rules)
        print(f"Created {len(self.scikit_rules)} rules for Scikit-Fuzzy model.")
        self.ctrl_system = ctrl.ControlSystem(self.scikit_rules)
        self.sim = ctrl.ControlSystemSimulation(self.ctrl_system)

    def _make_rules(self, rules: list[OntologyIndividualSuperclass]):
        # print("Preparing rules")
        scikit_rules = []

        for rule in rules:
            premises = _get_premises(rule)
            conclusions = _get_conclusions(rule)
            # print("Processing rule:", rule.name)
            # print(f" * premises: {[premise.name for premise in premises]}")
            # print(f" * conclusions: {[conclusion.name for conclusion in conclusions]}")

            # Build antecedent conditions (premises)
            antecedent_conditions = None
            for premise in premises:
                left, right = _get_left_right_hands(premise)
                fuzzy_variable = left
                fuzzy_value = right

                # print(f" ** Processing premise '{premise.name}'")

                if fuzzy_variable not in self.antecedents:
                    raise ValueError(
                        f"Variable '{fuzzy_variable}' in premise '{premise.name}' is not defined as an antecedent."
                    )
                # print(f" ***  variable '{fuzzy_variable.name}'")
                condition = self.antecedents[fuzzy_variable][fuzzy_value.name]
                if antecedent_conditions is None:
                    antecedent_conditions = condition
                else:
                    antecedent_conditions = antecedent_conditions & condition
            # print(f" ** Built antecedent conditions: {antecedent_conditions}")

            # Build consequent (conclusion)
            if conclusions:
                # print(f" ** Processing conclusion '{conclusions[0].name}'")
                conclusion = conclusions[0]
                left, right = _get_left_right_hands(conclusion)
                fuzzy_variable = left
                fuzzy_value = right

                if fuzzy_variable in self.consequents:
                    # print(f" ***  variable '{fuzzy_variable.name}'")
                    consequent = self.consequents[fuzzy_variable][fuzzy_value.name]
                    # print(f" ** Built consequent: {consequent.label} is {fuzzy_value.name}")

                    if antecedent_conditions is not None:
                        scikit_rules.append(ctrl.Rule(antecedent_conditions, consequent))

        self.scikit_rules = scikit_rules

    def set_start_values(
        self,
        input_values: dict[str, float],
        antecedents: dict[OntologyIndividualSuperclass, ctrl.Antecedent],
    ):
        for input in self.sim._get_inputs().items():
            var_name = input[0]
            if var_name in input_values:
                self.sim.input[var_name] = input_values[var_name]
            else:
                ant = next(x for x in antecedents if x.name == var_name)
                if ant:
                    self.sim.input[var_name] = (antecedents[ant].universe[0] + antecedents[ant].universe[-1]) / 2
                    print(
                        f"Input value for '{var_name}' not provided. Setting to default value {(antecedents[ant].universe[0] + antecedents[ant].universe[-1]) / 2}."
                    )
                else:
                    self.sim.input[var_name] = 0

    def compute(self, layer: set[OntologyIndividualSuperclass]):
        layer_var_names = [ind.name for ind in layer]
        print(f"Processing layer with targets: {layer_var_names}")
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )

        # Compute inference for this layer
        self.sim.compute()
        # self.sim.print_state()
        print(f" Layer output: {self.sim.output}")

        # Capture and set output values from this layer
        for var_name in layer_var_names:
            if var_name in self.sim.output:
                output_value = self.sim.output[var_name]
                print(f" Inferred {var_name} = {output_value}")
                if var_name in [ant.name for ant in self.antecedents]:
                    self.sim.input[var_name] = output_value
