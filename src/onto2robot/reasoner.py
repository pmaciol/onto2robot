from copy import deepcopy
from pathlib import Path

from onto2robot.core import MobileOntologyMeta
from onto2robot.fs_wrapper import SimpfulFuzzyWrapper
from onto2robot.scikit_fuzz_wrapper import ScikitFuzzyWrapper

type FuzzyWrapperType = type[SimpfulFuzzyWrapper] | type[ScikitFuzzyWrapper]


class Reasoner:
    def __init__(
        self,
        ontology_name: str,
        ontologies_path: Path | None = None,
        goal_name: str = "finalMove",
        FuzzyModel: FuzzyWrapperType = SimpfulFuzzyWrapper,
    ):
        ont = MobileOntologyMeta(ontology_name, ontologies_path)
        rules = ont.get_rules()
        goal = goal_name
        linguistic_variables_spaces = ont.get_linguistic_variable_domains()
        self.reasoning_order, _ = ont.get_possible_chains([ont.get_individual_by_name(goal)])
        self.fs = FuzzyModel(
            linguistic_variables_spaces,
            rules=rules,
            goal_name=goal,
        )

    def reason(self, input_values: dict[str, float]):
        self.fs.set_start_values(input_values)
        outputs = deepcopy(input_values)
        summary_headers = None
        summary_results = []
        for layer in reversed(self.reasoning_order):
            result = self.fs.compute(layer)
            if result:
                outputs.update(result)

            if summary_headers is None:
                summary_headers = list(outputs.keys())

            summary_results.append(outputs)

        return self.fs.get_goal_value()
