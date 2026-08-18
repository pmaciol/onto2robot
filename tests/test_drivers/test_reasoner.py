import math
from pathlib import Path

import pytest

from onto2robot.fs_wrapper import SimpfulFuzzyWrapper
from onto2robot.reasoner import Reasoner, sensors_to_input_values
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


def test_sensors_to_input_values_maps_by_key_order():
    side_values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    bottom_values = [21.0, 22.0]

    input_values = sensors_to_input_values(side_values, bottom_values)

    assert input_values == {
        "R01sLS": 10.0,
        "R01sLF": 11.0,
        "R01sFL": 12.0,
        "R01sFR": 13.0,
        "R01sRF": 14.0,
        "R01sRS": 15.0,
        "R01sBL": 21.0,
        "R01sBR": 22.0,
    }


def test_sensors_to_input_values_validates_input_lengths():
    with pytest.raises(ValueError, match="side sensor"):
        sensors_to_input_values([1.0], [10.0, 11.0])

    with pytest.raises(ValueError, match="bottom sensor"):
        sensors_to_input_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [10.0])
