import simpful
from simpful import LinguisticVariable, TriangleFuzzySet

from onto2robot.core import (
    OntologyIndividualSuperclass,
    rule_to_pair,
    rule_to_string,
)


class FuzzySystem:
    def __init__(self):
        self.fs = simpful.FuzzySystem()

    def add_fuzzy_set(
        self, variable_name: str, points: list[float], set_type: simpful.FuzzySet = simpful.TriangleFuzzySet
    ):
        self.fs.add_linguistic_variable(
            variable_name, simpful.LinguisticVariable(variable_name, universe_of_discourse=[0, 40])
        )


class SimpfulFuzzyWrapper:
    @staticmethod
    def _get_triangle_fuzzy_points(
        terms: list[str],
        universe: tuple[float, float],
    ) -> list[TriangleFuzzySet]:
        sets_no = len(terms)
        if sets_no != 3:
            raise ValueError("Currently only 3-term spaces are supported.")
        first_points = [universe[0], universe[0], (universe[0] + universe[1]) / 3]
        middle_points = [
            universe[0],
            (universe[0] + universe[1]) / 3,
            universe[1],
        ]
        last_points = [(universe[0] + universe[1]) / 3, universe[1], universe[1]]
        return {
            terms[0]: TriangleFuzzySet(*first_points, term=terms[0]),
            terms[1]: TriangleFuzzySet(*middle_points, term=terms[1]),
            terms[2]: TriangleFuzzySet(*last_points, term=terms[2]),
        }

    def __init__(
        self,
        linguistic_variables_spaces: dict[str, dict[str, OntologyIndividualSuperclass]],
        # TODO: The universe for each variable should be taken from the target system specification
        universe: tuple[float, float],
        rules: list[OntologyIndividualSuperclass],
    ):
        self.fs = FuzzySystem()
        self.goals_inferred = {}
        self.linguistic_variables_spaces = linguistic_variables_spaces

        self.fuzzy_sets = {}
        for terms in linguistic_variables_spaces.values():
            fs_terms = self._get_triangle_fuzzy_points(terms, universe)
            self.fuzzy_sets.update(fs_terms)
        self._add_linguistic_variables()

        stringified_rules = [rule_to_string(rule) for rule in rules]

        rnames = {}
        for r in rules:
            k, v = rule_to_pair(r)
            rnames[k] = v

        for k, v in sorted(rnames.items()):
            print(f"{k} .  {v}")
        self.fs.fs.add_rules(stringified_rules)

    def _add_linguistic_variables(self):
        for lv_name, terms in self.linguistic_variables_spaces.items():
            fs_list = [self.fuzzy_sets[term] for term in terms]
            # TODO:replace with more generic method
            self.fs.fs.add_linguistic_variable(lv_name, LinguisticVariable(fs_list, universe_of_discourse=[0, 40]))
            print(f"Added linguistic variable {lv_name} with terms {terms}")

    def set_start_values(
        self,
        input_values: dict[str, float],
    ):
        for var_name, value in input_values.items():
            self.fs.fs.set_variable(var_name, value)

    def compute(self, layer: set[OntologyIndividualSuperclass]):
        self.do_reasoning([ind.name for ind in layer])

    def do_reasoning(self, goals: list[str]):
        goals_inferred = self.fs.fs.Mamdani_inference(goals)  # returns crisp value(s)
        for goal in goals:
            goal_value = goals_inferred[goal]
            self.fs.fs.set_variable(goal, goal_value)
            self.goals_inferred[goal] = goal_value
            print(f" Inferred {goal} = {goal_value}")
