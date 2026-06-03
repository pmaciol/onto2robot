from pprint import pprint

import simpful
from simpful import LinguisticVariable, TriangleFuzzySet

from onto2robot.core import (
    LinguisticVariableDomain,
    OntologyIndividualSuperclass,
    rule_to_pair,
    rule_to_string,
)


class FuzzySystem:
    def __init__(self):
        self.fs = simpful.FuzzySystem()


class SimpfulFuzzyWrapper:
    def __init__(
        self,
        linguistic_variables_domains: dict[OntologyIndividualSuperclass, LinguisticVariableDomain],
        rules: list[OntologyIndividualSuperclass],
        goal_name: str,
    ):
        self.fs = FuzzySystem()
        self.goals_inferred = {}
        self.linguistic_variables_spaces = linguistic_variables_domains
        self.goal_name = goal_name

        self.fuzzy_sets = {}

        for _, set_of_values in self.linguistic_variables_spaces.items():
            universe = [set_of_values.fuzzy_points[0], set_of_values.fuzzy_points[-1]]
            fs_terms = self._get_triangle_fuzzy_points(set_of_values, universe)  # Here make matching triangles
            self.fuzzy_sets.update(fs_terms)
        self._add_linguistic_variables()

        stringified_rules = [rule_to_string(rule) for rule in rules]
        pprint(stringified_rules)

        rnames = {}
        for r in rules:
            k, v = rule_to_pair(r)
            rnames[k] = v

        for k, v in sorted(rnames.items()):
            print(f"{k} .  {v}")
        self.fs.fs.add_rules(stringified_rules)

    def set_start_values(
        self,
        input_values: dict[str, float],
    ):
        for var_name, value in input_values.items():
            self.fs.fs.set_variable(var_name, value)

    def compute(self, layer: set[OntologyIndividualSuperclass]):
        self._do_reasoning([ind.name for ind in layer])
        return self.goals_inferred

    def get_goal_value(self) -> float | None:
        return self.goals_inferred.get(self.goal_name)

    @staticmethod
    def _get_triangle_fuzzy_points(
        set_of_values: LinguisticVariableDomain, universe: list[float]
    ) -> dict[str, TriangleFuzzySet]:
        terms = [value.name for value in set_of_values.linguistic_domain]
        sets_no = len(terms)
        if sets_no != 3:
            raise ValueError("Currently only 3-term spaces are supported.")
        first_points = set_of_values.fuzzy_points[:3]
        middle_points = set_of_values.fuzzy_points[3:6]
        last_points = set_of_values.fuzzy_points[6:]
        return {
            terms[0]: TriangleFuzzySet(*first_points, term=terms[0]),
            terms[1]: TriangleFuzzySet(*middle_points, term=terms[1]),
            terms[2]: TriangleFuzzySet(*last_points, term=terms[2]),
        }

    def _add_linguistic_variables(self):
        for variable, set_of_values in self.linguistic_variables_spaces.items():
            terms = [value.name for value in set_of_values.linguistic_domain]
            fs_list = [self.fuzzy_sets[term] for term in terms]
            universe = [set_of_values.fuzzy_points[0], set_of_values.fuzzy_points[-1]]
            self.fs.fs.add_linguistic_variable(
                variable.name, LinguisticVariable(fs_list, universe_of_discourse=universe)
            )
            print(f"Added linguistic variable {variable.name} with terms {terms}")

    def _do_reasoning(self, goals: list[str]):
        goals_inferred = self.fs.fs.Mamdani_inference(goals)  # returns crisp value(s)
        for goal in goals:
            goal_value = goals_inferred[goal]
            self.fs.fs.set_variable(goal, goal_value)
            self.goals_inferred[goal] = goal_value
            print(f" Inferred {goal} = {goal_value}")
