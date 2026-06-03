import math
from pathlib import Path

import pytest

from onto2robot.fs_wrapper import SimpfulFuzzyWrapper
from onto2robot.reasoner import Reasoner
from onto2robot.scikit_fuzz_wrapper import ScikitFuzzyWrapper

REASONER_CASES = [
    (ScikitFuzzyWrapper, [39.0, 1]),
    (SimpfulFuzzyWrapper, [33.0, 17]),
]


@pytest.mark.parametrize(
    "fuzzy_model, expected_goal",
    REASONER_CASES,
)
def test_reasoner_single_call(fuzzy_model, expected_goal):
    reasoner = Reasoner(
        ontology_name="amro_uc01_v01",
        ontologies_path=Path("ontologies/"),
        goal_name="finalMove",
        FuzzyModel=fuzzy_model,
    )
    goal = reasoner.reason(
        input_values={
            "R01sLF": 1,
            "R01sLS": 20,
            "R01sFL": 1,
            "R01sFR": 1,
            "R01sRF": 1,
            "R01sRS": 20,
            "R01sBL": 20,
            "R01sBR": 20,
        }
    )
    assert goal is not None
    assert math.isclose(goal, expected_goal[0], abs_tol=1.0)


@pytest.mark.parametrize(
    "fuzzy_model, expected_goal",
    REASONER_CASES,
)
def test_reasoner_second_call(fuzzy_model, expected_goal):
    reasoner = Reasoner(
        ontology_name="amro_uc01_v01",
        ontologies_path=Path("ontologies/"),
        goal_name="finalMove",
        FuzzyModel=fuzzy_model,
    )

    goal = reasoner.reason(
        input_values={
            "R01sLF": 1,
            "R01sLS": 39,
            "R01sFL": 20,
            "R01sFR": 20,
            "R01sRF": 39,
            "R01sRS": 1,
            "R01sBL": 20,
            "R01sBR": 20,
        }
    )
    assert goal is not None
    assert math.isclose(goal, expected_goal[1], abs_tol=1.0)


@pytest.mark.parametrize(
    "fuzzy_model, expected_goal",
    REASONER_CASES,
)
def test_reasoner_double_call(fuzzy_model, expected_goal):
    reasoner = Reasoner(
        ontology_name="amro_uc01_v01",
        ontologies_path=Path("ontologies/"),
        goal_name="finalMove",
        FuzzyModel=fuzzy_model,
    )
    goal = reasoner.reason(
        input_values={
            "R01sLF": 1,
            "R01sLS": 20,
            "R01sFL": 1,
            "R01sFR": 1,
            "R01sRF": 1,
            "R01sRS": 20,
            "R01sBL": 20,
            "R01sBR": 20,
        }
    )
    assert goal is not None
    assert math.isclose(goal, expected_goal[0], abs_tol=1.0)

    goal = reasoner.reason(
        input_values={
            "R01sLF": 1,
            "R01sLS": 39,
            "R01sFL": 20,
            "R01sFR": 20,
            "R01sRF": 39,
            "R01sRS": 1,
            "R01sBL": 20,
            "R01sBR": 20,
        }
    )
    assert goal is not None
    assert math.isclose(goal, expected_goal[1], abs_tol=1.0)
