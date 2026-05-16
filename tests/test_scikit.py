"""
Tests for the scikit-fuzzy wrapper in the onto2robot project.

Validates:
- the scikit wrapper
- the set of exemplary rules provided in the ontology
Tests asserts very precise expected outputs. The values are not important,
just provides values obtained after manual testing of the rules defined in the ontology.

"""

import math

from onto2robot.core import MobileOntologyMeta
from onto2robot.scikit_fuzz_wrapper import ScikitFuzzyWrapper


def scikit_fuzzy_full(input_values: dict[str, float]):
    ont = MobileOntologyMeta("amro_uc01_v01")
    try:
        rules = ont.get_rules()
        goal = "finalMove"

        linguistic_variables_domains = ont.get_linguistic_variable_domains()
        reasoning_order, source_variables = ont.get_possible_chains([ont.get_individual_by_name(goal)])
        print(f"Reasoning order: {reasoning_order}")

        fs = ScikitFuzzyWrapper(linguistic_variables_domains, rules, goal)
        fs.set_start_values(input_values)

        # Perform inference layer by layer in reverse order
        print("Source variables:")
        for var, val in input_values.items():
            print(f"  {var}:\t {val}")
        for layer in reversed(reasoning_order):
            fs.compute(layer)
        return fs.sim.output
    finally:
        ont.destroy()


def test_scikit_fuzzy_1():
    results = scikit_fuzzy_full(
        {
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
    assert math.isclose(results.get("sLassessment", 0.0), 0.5, abs_tol=0.1)
    assert math.isclose(results.get("sFassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("sRassessment", 0.0), 0.5, abs_tol=0.1)
    assert math.isclose(results.get("move", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("finalMove", 0.0), 38.8, abs_tol=0.1)


def test_scikit_fuzzy_2():
    results = scikit_fuzzy_full(
        {
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

    assert math.isclose(results.get("sLassessment", 0.0), 0.5, abs_tol=0.1)
    assert math.isclose(results.get("sFassessment", 0.0), 38.8, abs_tol=0.1)
    assert math.isclose(results.get("sRassessment", 0.0), 39, abs_tol=0.1)
    assert math.isclose(results.get("move", 0.0), 0.3, abs_tol=0.1)
    assert math.isclose(results.get("finalMove", 0.0), 0.2, abs_tol=0.1)


def test_scikit_fuzzy_3():
    results = scikit_fuzzy_full(
        {
            "R01sLF": 39,
            "R01sLS": 1,
            "R01sFL": 20,
            "R01sFR": 20,
            "R01sRF": 1,
            "R01sRS": 39,
            "R01sBL": 20,
            "R01sBR": 20,
        }
    )

    assert math.isclose(results.get("sLassessment", 0.0), 39, abs_tol=0.1)
    assert math.isclose(results.get("sFassessment", 0.0), 38.8, abs_tol=0.1)
    assert math.isclose(results.get("sRassessment", 0.0), 0.5, abs_tol=0.1)
    assert math.isclose(results.get("move", 0.0), 39, abs_tol=0.1)
    assert math.isclose(results.get("finalMove", 0.0), 39, abs_tol=0.1)


def test_scikit_fuzzy_4():
    results = scikit_fuzzy_full(
        {
            "R01sLF": 39,
            "R01sLS": 1,
            "R01sFL": 1,
            "R01sFR": 39,
            "R01sRF": 1,
            "R01sRS": 39,
            "R01sBL": 20,
            "R01sBR": 20,
        }
    )

    assert math.isclose(results.get("sLassessment", 0.0), 39, abs_tol=0.1)
    assert math.isclose(results.get("sFassessment", 0.0), 0.5, abs_tol=0.1)
    assert math.isclose(results.get("sRassessment", 0.0), 0.5, abs_tol=0.1)
    assert math.isclose(results.get("move", 0.0), 39, abs_tol=0.1)
    assert math.isclose(results.get("finalMove", 0.0), 39, abs_tol=0.1)


def test_scikit_fuzzy_5():
    results = scikit_fuzzy_full(
        {
            "R01sLF": 20,
            "R01sLS": 20,
            "R01sFL": 1,
            "R01sFR": 1,
            "R01sRF": 1,
            "R01sRS": 1,
            "R01sBL": 20,
            "R01sBR": 20,
        }
    )

    assert math.isclose(results.get("sLassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("sFassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("sRassessment", 0.0), 0.5, abs_tol=0.1)
    assert math.isclose(results.get("move", 0.0), 38.8, abs_tol=0.1)
    assert math.isclose(results.get("finalMove", 0.0), 38.9, abs_tol=0.1)


def test_scikit_fuzzy_6():
    results = scikit_fuzzy_full(
        {
            "R01sLF": 20,
            "R01sLS": 1,
            "R01sFL": 1,
            "R01sFR": 1,
            "R01sRF": 20,
            "R01sRS": 20,
            "R01sBL": 10,
            "R01sBR": 30,
        }
    )

    assert math.isclose(results.get("sLassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("sFassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("sRassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("move", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("finalMove", 0.0), 34.0, abs_tol=0.1)


def test_scikit_fuzzy_7():
    results = scikit_fuzzy_full(
        {
            "R01sLF": 20,
            "R01sLS": 1,
            "R01sFL": 1,
            "R01sFR": 1,
            "R01sRF": 20,
            "R01sRS": 20,
            "R01sBL": 30,
            "R01sBR": 10,
        }
    )

    assert math.isclose(results.get("sLassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("sFassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("sRassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("move", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("finalMove", 0.0), 5, abs_tol=0.1)


def test_scikit_fuzzy_8():
    results = scikit_fuzzy_full(
        {
            "R01sLF": 40,
            "R01sLS": 20,
            "R01sFL": 20,
            "R01sFR": 20,
            "R01sRF": 20,
            "R01sRS": 20,
            "R01sBL": 20,
            "R01sBR": 20,
        }
    )

    assert math.isclose(results.get("sLassessment", 0.0), 39.0, abs_tol=0.1)
    assert math.isclose(results.get("sFassessment", 0.0), 38.8, abs_tol=0.1)
    assert math.isclose(results.get("sRassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("move", 0.0), 39, abs_tol=0.1)
    assert math.isclose(results.get("finalMove", 0.0), 39, abs_tol=0.1)


def test_scikit_fuzzy_9():
    results = scikit_fuzzy_full(
        {
            "R01sLF": 10,
            "R01sLS": 15,
            "R01sFL": 5,
            "R01sFR": 4,
            "R01sRF": 10,
            "R01sRS": 25,
            "R01sBL": 25,
            "R01sBR": 18,
        }
    )

    assert math.isclose(results.get("sLassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("sFassessment", 0.0), 19.2, abs_tol=0.1)
    assert math.isclose(results.get("sRassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("move", 0.0), 19.4, abs_tol=0.1)
    assert math.isclose(results.get("finalMove", 0.0), 36.1, abs_tol=0.1)


def test_scikit_fuzzy_10():
    results = scikit_fuzzy_full(
        {
            "R01sLF": 15,
            "R01sLS": 5,
            "R01sFL": 10,
            "R01sFR": 5,
            "R01sRF": 25,
            "R01sRS": 15,
            "R01sBL": 30,
            "R01sBR": 10,
        }
    )

    assert math.isclose(results.get("sLassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("sFassessment", 0.0), 34.0, abs_tol=0.1)
    assert math.isclose(results.get("sRassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("move", 0.0), 36.5, abs_tol=0.1)
    assert math.isclose(results.get("finalMove", 0.0), 37.6, abs_tol=0.1)


def test_scikit_fuzzy_11():
    results = scikit_fuzzy_full(
        {
            "R01sLF": 22,
            "R01sLS": 10,
            "R01sFL": 25,
            "R01sFR": 18,
            "R01sRF": 7,
            "R01sRS": 35,
            "R01sBL": 8,
            "R01sBR": 35,
        }
    )

    assert math.isclose(results.get("sLassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("sFassessment", 0.0), 36.1, abs_tol=0.1)
    assert math.isclose(results.get("sRassessment", 0.0), 3.5, abs_tol=0.1)
    assert math.isclose(results.get("move", 0.0), 36.9, abs_tol=0.1)
    assert math.isclose(results.get("finalMove", 0.0), 37.7, abs_tol=0.1)


def test_scikit_fuzzy_12():
    results = scikit_fuzzy_full(
        {
            "R01sLF": 19,
            "R01sLS": 21,
            "R01sFL": 3,
            "R01sFR": 2,
            "R01sRF": 2,
            "R01sRS": 22,
            "R01sBL": 22,
            "R01sBR": 19,
        }
    )

    assert math.isclose(results.get("sLassessment", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("sFassessment", 0.0), 19.3, abs_tol=0.1)
    assert math.isclose(results.get("sRassessment", 0.0), 1.0, abs_tol=0.1)
    assert math.isclose(results.get("move", 0.0), 38.5, abs_tol=0.1)
    assert math.isclose(results.get("finalMove", 0.0), 38.8, abs_tol=0.1)


def test_scikit_fuzzy_13():
    results = scikit_fuzzy_full(
        {
            "R01sLF": 2,
            "R01sLS": 15,
            "R01sFL": 2,
            "R01sFR": 3,
            "R01sRF": 2,
            "R01sRS": 15,
            "R01sBL": 38,
            "R01sBR": 3,
        }
    )

    assert math.isclose(results.get("sLassessment", 0.0), 1.0, abs_tol=0.1)
    assert math.isclose(results.get("sFassessment", 0.0), 19.7, abs_tol=0.1)
    assert math.isclose(results.get("sRassessment", 0.0), 1.0, abs_tol=0.1)
    assert math.isclose(results.get("move", 0.0), 19.5, abs_tol=0.1)
    assert math.isclose(results.get("finalMove", 0.0), 1.5, abs_tol=0.1)
