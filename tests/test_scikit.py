import math

import numpy as np

from onto2robot.core import MobileOntologyMeta
from onto2robot.scikit_fuzz_wrapper import ScikitFuzzyWrapper


def scikit_fuzzy_full(input_values: dict[str, float]):
    ont = MobileOntologyMeta("mobile_robot_ontology")
    rules = ont.get_rules()
    goal = "finalMove"

    linguistic_spaces = [
        ["low", "middle", "high"],
        ["left", "forward", "right"],
    ]
    linguistic_variables_spaces = ont.linguistic_value_spaces(linguistic_spaces)
    reasoning_order, source_variables = ont.get_possible_chains([ont.get_individual_by_name(goal)])

    universe = np.arange(0, 40, 1)
    fs = ScikitFuzzyWrapper(linguistic_variables_spaces, goal, universe, rules)
    fs.set_start_values(input_values)

    # Perform inference layer by layer in reverse order
    for layer in reversed(reasoning_order):
        fs.compute(layer)

    return fs.sim.output


def test_scikit_fuzzy_1():
    results = scikit_fuzzy_full(
        {
            "sLF": 1,
            "sLS": 20,
            "sFL": 1,
            "sFR": 1,
            "sRF": 1,
            "sRS": 20,
            "bl": 20,
            "bR": 20,
        }
    )

    assert math.isclose(results.get("sLassessment"), 1, abs_tol=1)
    assert math.isclose(results.get("sFassessment"), 20, abs_tol=1)
    assert math.isclose(results.get("sRassessment"), 1, abs_tol=1)
    assert math.isclose(results.get("move"), 20, abs_tol=1)
    assert math.isclose(results.get("finalMove"), 40, abs_tol=1)


def test_scikit_fuzzy_2():
    results = scikit_fuzzy_full(
        {
            "sLF": 1,
            "sLS": 39,
            "sFL": 20,
            "sFR": 20,
            "sRF": 39,
            "sRS": 1,
            "bl": 20,
            "bR": 20,
        }
    )

    assert math.isclose(results.get("sLassessment"), 1, abs_tol=1)
    assert math.isclose(results.get("sFassessment"), 39, abs_tol=1)
    assert math.isclose(results.get("sRassessment"), 39, abs_tol=1)
    assert math.isclose(results.get("move"), 1, abs_tol=1)
    assert math.isclose(results.get("finalMove"), 0, abs_tol=0.01)


def test_scikit_fuzzy_3():
    results = scikit_fuzzy_full(
        {
            "sLF": 39,
            "sLS": 1,
            "sFL": 20,
            "sFR": 20,
            "sRF": 1,
            "sRS": 39,
            "bl": 20,
            "bR": 20,
        }
    )

    assert math.isclose(results.get("sLassessment"), 39, abs_tol=1)
    assert math.isclose(results.get("sFassessment"), 39, abs_tol=1)
    assert math.isclose(results.get("sRassessment"), 1, abs_tol=1)
    assert math.isclose(results.get("move"), 39, abs_tol=1)
    assert math.isclose(results.get("finalMove"), 39, abs_tol=1.0)


def test_scikit_fuzzy_4():
    results = scikit_fuzzy_full(
        {
            "sLF": 39,
            "sLS": 1,
            "sFL": 1,
            "sFR": 39,
            "sRF": 1,
            "sRS": 39,
            "bl": 20,
            "bR": 20,
        }
    )

    assert math.isclose(results.get("sLassessment"), 39, abs_tol=1)
    assert math.isclose(results.get("sFassessment"), 1, abs_tol=1)
    assert math.isclose(results.get("sRassessment"), 1, abs_tol=1)
    assert math.isclose(results.get("move"), 39, abs_tol=1)
    assert math.isclose(results.get("finalMove"), 39, abs_tol=1.0)
    ##assert False


def test_scikit_fuzzy_5():
    results = scikit_fuzzy_full(
        {
            "sLF": 20,
            "sLS": 20,
            "sFL": 1,
            "sFR": 1,
            "sRF": 1,
            "sRS": 1,
            "bl": 20,
            "bR": 20,
        }
    )

    assert math.isclose(results.get("sLassessment"), 20, abs_tol=1)
    assert math.isclose(results.get("sFassessment"), 20, abs_tol=1)
    assert math.isclose(results.get("sRassessment"), 1, abs_tol=1)
    assert math.isclose(results.get("move"), 39, abs_tol=1)
    assert math.isclose(results.get("finalMove"), 39, abs_tol=1.0)


def test_scikit_fuzzy_6():
    results = scikit_fuzzy_full(
        {
            "sLF": 20,
            "sLS": 1,
            "sFL": 1,
            "sFR": 1,
            "sRF": 20,
            "sRS": 20,
            "bl": 10,
            "bR": 30,
        }
    )

    assert math.isclose(results.get("sLassessment"), 20, abs_tol=1)
    assert math.isclose(results.get("sFassessment"), 20, abs_tol=1)
    assert math.isclose(results.get("sRassessment"), 20, abs_tol=1)
    assert math.isclose(results.get("move"), 20, abs_tol=1)
    assert math.isclose(results.get("finalMove"), 40, abs_tol=1.0)


def test_scikit_fuzzy_7():
    results = scikit_fuzzy_full(
        {
            "sLF": 20,
            "sLS": 1,
            "sFL": 1,
            "sFR": 1,
            "sRF": 20,
            "sRS": 20,
            "bl": 30,
            "bR": 10,
        }
    )

    assert math.isclose(results.get("sLassessment"), 20, abs_tol=1)
    assert math.isclose(results.get("sFassessment"), 20, abs_tol=1)
    assert math.isclose(results.get("sRassessment"), 20, abs_tol=1)
    assert math.isclose(results.get("move"), 20, abs_tol=1)
    assert math.isclose(results.get("finalMove"), 5, abs_tol=1.0)


def test_scikit_fuzzy_8():
    results = scikit_fuzzy_full(
        {
            "sLF": 40,
            "sLS": 20,
            "sFL": 20,
            "sFR": 20,
            "sRF": 20,
            "sRS": 20,
            "bl": 20,
            "bR": 20,
        }
    )

    assert math.isclose(results.get("sLassessment"), 40, abs_tol=1)
    assert math.isclose(results.get("sFassessment"), 39, abs_tol=1)
    assert math.isclose(results.get("sRassessment"), 20, abs_tol=1)
    assert math.isclose(results.get("move"), 39, abs_tol=1)
    assert math.isclose(results.get("finalMove"), 39, abs_tol=1.0)


def test_scikit_fuzzy_9():
    results = scikit_fuzzy_full(
        {
            "sLF": 10,
            "sLS": 15,
            "sFL": 5,
            "sFR": 4,
            "sRF": 10,
            "sRS": 25,
            "bl": 25,
            "bR": 18,
        }
    )

    assert math.isclose(results.get("sLassessment"), 1, abs_tol=1)
    assert math.isclose(results.get("sFassessment"), 39, abs_tol=1)
    assert math.isclose(results.get("sRassessment"), 39, abs_tol=1)
    assert math.isclose(results.get("move"), 1, abs_tol=1)
    assert math.isclose(results.get("finalMove"), 1, abs_tol=1.0)


def test_scikit_fuzzy_10():
    results = scikit_fuzzy_full(
        {
            "sLF": 15,
            "sLS": 5,
            "sFL": 10,
            "sFR": 5,
            "sRF": 25,
            "sRS": 15,
            "bl": 30,
            "bR": 10,
        }
    )

    assert math.isclose(results.get("sLassessment"), 39, abs_tol=1)
    assert math.isclose(results.get("sFassessment"), 39, abs_tol=1)
    assert math.isclose(results.get("sRassessment"), 1, abs_tol=1)
    assert math.isclose(results.get("move"), 1, abs_tol=1)
    assert math.isclose(results.get("finalMove"), 40, abs_tol=1.0)


def test_scikit_fuzzy_11():
    results = scikit_fuzzy_full(
        {
            "sLF": 22,
            "sLS": 10,
            "sFL": 25,
            "sFR": 18,
            "sRF": 7,
            "sRS": 35,
            "bl": 8,
            "bR": 35,
        }
    )

    assert math.isclose(results.get("sLassessment"), 39, abs_tol=1)
    assert math.isclose(results.get("sFassessment"), 1, abs_tol=1)
    assert math.isclose(results.get("sRassessment"), 1, abs_tol=1)
    assert math.isclose(results.get("move"), 39, abs_tol=1)
    assert math.isclose(results.get("finalMove"), 40, abs_tol=1.0)


def test_scikit_fuzzy_12():
    results = scikit_fuzzy_full(
        {
            "sLF": 19,
            "sLS": 21,
            "sFL": 3,
            "sFR": 2,
            "sRF": 2,
            "sRS": 22,
            "bl": 22,
            "bR": 19,
        }
    )

    assert math.isclose(results.get("sLassessment"), 39, abs_tol=1)
    assert math.isclose(results.get("sFassessment"), 1, abs_tol=1)
    assert math.isclose(results.get("sRassessment"), 1, abs_tol=1)
    assert math.isclose(results.get("move"), 39, abs_tol=1)
    assert math.isclose(results.get("finalMove"), 40, abs_tol=1.0)


def test_scikit_fuzzy_13():
    results = scikit_fuzzy_full(
        {
            "sLF": 2,
            "sLS": 15,
            "sFL": 2,
            "sFR": 3,
            "sRF": 2,
            "sRS": 15,
            "bl": 38,
            "bR": 3,
        }
    )

    assert math.isclose(results.get("sLassessment"), 39, abs_tol=1)
    assert math.isclose(results.get("sFassessment"), 1, abs_tol=1)
    assert math.isclose(results.get("sRassessment"), 1, abs_tol=1)
    assert math.isclose(results.get("move"), 39, abs_tol=1)
    assert math.isclose(results.get("finalMove"), 40, abs_tol=1.0)
