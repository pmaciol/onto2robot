from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from skfuzzy import control as ctrl
from skfuzzy import trimf
from skfuzzy.control import visualization as ctrl_visualization

from onto2robot.core import (
    LinguisticVariableDomain,
    OntologyIndividualSuperclass,
    _get_conclusions,
    _get_left_right_hands,
    _get_premises,
)


def _patch_broken_control_system_visualizer() -> None:
    """Patch skfuzzy bug where ControlSystemVisualizer.__init__ forgets self.ctrl."""

    def _fixed_init(self, control_system):
        if not ctrl_visualization.matplotlib_present:
            raise ImportError("`ControlSystemVisualizer` can only be used with `matplotlib` present in the system.")
        self.ctrl = control_system
        self.fig, self.ax = plt.subplots()

    ctrl_visualization.ControlSystemVisualizer.__init__ = _fixed_init


_patch_broken_control_system_visualizer()


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
                if len(set_of_values.fuzzy_points) == 3 * len(set_of_values.linguistic_domain) and not use_auto_membership:
                    for id, ld in enumerate(set_of_values.linguistic_domain):
                        # Only triangle membership functions are supported in this version
                        antecedents[variable][ld.name] = trimf(universe, set_of_values.fuzzy_points[id * 3 : id * 3 + 3])
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
    print("Creating consequents based on rules:")
    for rule in rules:
        print(f" - {rule}")
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
                if len(set_of_values.fuzzy_points) == 3 * len(set_of_values.linguistic_domain) and not use_auto_membership:
                    for id, ld in enumerate(set_of_values.linguistic_domain):
                        # Only triangle membership functions are supported in this version
                        consequents[variable][ld.name] = trimf(universe, set_of_values.fuzzy_points[id * 3 : id * 3 + 3])
                else:
                    consequents[variable].automf(
                        len(set_of_values.linguistic_domain),
                        names=[value.name for value in set_of_values.linguistic_domain],
                    )
            else:
                for term in set_of_values.linguistic_domain:
                    consequents[variable][term] = trimf(consequents[variable].universe, membership_functions[term.name])
    print("Created consequents")
    for name, con in consequents.items():
        print(f"Consequent '{name.name}' has terms: {list(con.terms.keys())}")
    return consequents


class ScikitFuzzyWrapper:
    def __init__(
        self,
        linguistic_variables_domains: dict[OntologyIndividualSuperclass, LinguisticVariableDomain],
        rules: list[OntologyIndividualSuperclass],
        goal_name: str,
    ):
        self.goal_name = goal_name
        self.linguistic_variables_spaces = linguistic_variables_domains
        self.antecedents = make_antecedents(linguistic_variables_domains, goal_name, use_auto_membership=True)
        self.consequents = make_consequents(rules, linguistic_variables_domains, use_auto_membership=True)
        self._make_rules(rules)
        print(f"Created {len(self.scikit_rules)} rules for Scikit-Fuzzy model.")
        self.ctrl_system = ctrl.ControlSystem(self.scikit_rules)
        self.sim = ctrl.ControlSystemSimulation(self.ctrl_system)

    def _print_fired_rules(self, eps: float = 1e-12) -> dict[int, ctrl.Rule]:
        print("Fired rules:")
        fired_rules = {}
        for _, rule in enumerate(self.scikit_rules, start=1):
            firing = float(rule.aggregate_firing[self.sim])
            if firing > eps:
                print(f"  R{rule.label}: firing={firing:.6f} :: {rule}")
                fired_rules[rule.label] = rule
        if not fired_rules:
            print("  none")
        return fired_rules

    def set_start_values(self, input_values: dict[str, float]):
        for input in self.sim._get_inputs().items():
            var_name = input[0]
            if var_name in input_values:
                print(f"Setting input value for '{var_name}'.")
                self.sim.input[var_name] = input_values[var_name]
            else:
                ant = next(x for x in self.antecedents if x.name == var_name)
                if ant:
                    self.sim.input[var_name] = (self.antecedents[ant].universe[0] + self.antecedents[ant].universe[-1]) / 2
                    print(
                        f"Input value for '{var_name}' not provided. Setting to default value "
                        f"{(self.antecedents[ant].universe[0] + self.antecedents[ant].universe[-1]) / 2}."
                    )
                else:
                    self.sim.input[var_name] = 0

    def compute(self, layer: set[OntologyIndividualSuperclass]) -> dict[str, float] | None:
        layer_var_names = [ind.name for ind in layer]
        print(f"Processing layer with targets: {layer_var_names}")
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )

        # Compute inference for this layer
        self.sim.compute()
        self.fired_rules = self._print_fired_rules(eps=0.5)

        # Capture and set output values from this layer
        self.results: dict[str, float] = {}
        for var_name in layer_var_names:
            if var_name in self.sim.output:
                output_value = self.sim.output[var_name]
                print(f" Inferred {var_name} = {output_value}")
                if var_name in [ant.name for ant in self.antecedents]:
                    self.sim.input[var_name] = output_value
                self.results[var_name] = output_value

        return self.results

    def get_goal_value(self) -> float | None:
        return self.results.get(self.goal_name)

    def _make_rules(self, rules: list[OntologyIndividualSuperclass]):
        scikit_rules = []

        for rule in rules:
            premises = _get_premises(rule)
            conclusions = _get_conclusions(rule)
            antecedent_conditions = None
            for premise in premises:
                left, right = _get_left_right_hands(premise)
                fuzzy_variable = left
                fuzzy_value = right

                if fuzzy_variable not in self.antecedents:
                    raise ValueError(f"Variable '{fuzzy_variable}' in premise '{premise.name}' is not defined as an antecedent.")
                condition = self.antecedents[fuzzy_variable][fuzzy_value.name]
                antecedent_conditions = condition if antecedent_conditions is None else antecedent_conditions & condition
            if conclusions:
                conclusion = conclusions[0]
                left, right = _get_left_right_hands(conclusion)
                fuzzy_variable = left
                fuzzy_value = right

                if fuzzy_variable in self.consequents:
                    consequent = self.consequents[fuzzy_variable][fuzzy_value.name]

                    if antecedent_conditions is not None:
                        scikit_rules.append(ctrl.Rule(antecedent_conditions, consequent, label=rule.name))

        self.scikit_rules = scikit_rules

    def save_rules_as_figures(self, path: Path):
        for _, r in self.fired_rules.items():
            r.view_n()
            figure = plt.gcf()
            figure.suptitle(f"Rule {r.label}")
            plt.savefig(f"{path / f'rule_{r.label}.png'}")
            plt.clf()

    def save_consequents_as_figures(self, path: Path):
        for _, v in self.consequents.items():
            v.view(
                sim=self.sim,
            )
            figure = plt.gcf()
            figure.suptitle(f"consequent {v.label}")
            plt.savefig(f"{path / f'consequent_{v.label}.png'}")
            plt.clf()
