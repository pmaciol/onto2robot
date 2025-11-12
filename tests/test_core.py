from math import isclose

from simpful import (
    FuzzySet,
    FuzzySystem,
    LinguisticVariable,
    Trapezoidal_MF,
    TriangleFuzzySet,
    Triangular_MF,
)

from onto2robot.core import load_ontology


def test_load_sumo_ontology():
    onto = load_ontology("sumo")
    all_classes = list(onto.classes())
    print(f"SUMO classes {all_classes}")
    assert len(all_classes) > 0


def test_load_mobile_robot_ontology():
    onto = load_ontology("mobile_robot_ontology")
    all_classes = list(onto.classes())
    print(f"MOBILE ROBOT classes {all_classes}")
    assert len(all_classes) > 0


def test_basic_simpful():
    # A simple fuzzy inference system for the tipping problem - exact copy from simpful docs
    # Create a fuzzy system object
    FS = FuzzySystem()

    # Define fuzzy sets and linguistic variables
    S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="poor")
    S_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term="good")
    S_3 = FuzzySet(function=Triangular_MF(a=5, b=10, c=10), term="excellent")
    FS.add_linguistic_variable(
        "Service",
        LinguisticVariable([S_1, S_2, S_3], concept="Service quality", universe_of_discourse=[0, 10]),
    )

    F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="rancid")
    F_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="delicious")
    FS.add_linguistic_variable(
        "Food",
        LinguisticVariable([F_1, F_2], concept="Food quality", universe_of_discourse=[0, 10]),
    )

    # Define output fuzzy sets and linguistic variable
    T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="small")
    T_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=20), term="average")
    T_3 = FuzzySet(function=Trapezoidal_MF(a=10, b=20, c=25, d=25), term="generous")
    FS.add_linguistic_variable("Tip", LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0, 25]))

    # Define fuzzy rules
    R1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small)"
    R2 = "IF (Service IS good) THEN (Tip IS average)"
    R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)"
    FS.add_rules([R1, R2, R3])

    # Set antecedents values
    FS.set_variable("Service", 4)
    FS.set_variable("Food", 8)

    # Perform Mamdani inference and print output
    mamadani_result = FS.Mamdani_inference(["Tip"])
    assert isclose(mamadani_result["Tip"], 14.172, abs_tol=0.001)


def test_layered_simpful_with_defuzz():
    FS = FuzzySystem()

    # Define fuzzy sets for Temperature
    Cold = TriangleFuzzySet(0, 0, 15, term="Cold")
    Warm = TriangleFuzzySet(10, 20, 30, term="Warm")
    Hot = TriangleFuzzySet(25, 40, 40, term="Hot")
    FS.add_linguistic_variable("Temperature", LinguisticVariable([Cold, Warm, Hot], universe_of_discourse=[0, 40]))

    # Define fuzzy sets for Comfort
    Low = TriangleFuzzySet(0, 0, 5, term="Low")
    Medium = TriangleFuzzySet(2.5, 5, 7.5, term="Medium")
    High = TriangleFuzzySet(5, 10, 10, term="High")
    FS.add_linguistic_variable("Comfort", LinguisticVariable([Low, Medium, High], universe_of_discourse=[0, 10]))

    # Define fuzzy sets for FanSpeed
    Slow = TriangleFuzzySet(0, 25, 50, term="Slow")
    MediumSpeed = TriangleFuzzySet(25, 50, 75, term="Medium")
    Fast = TriangleFuzzySet(50, 75, 100, term="Fast")
    FS.add_linguistic_variable(
        "FanSpeed", LinguisticVariable([Slow, MediumSpeed, Fast], universe_of_discourse=[0, 100])
    )

    # Add rules
    FS.add_rules(
        [
            "IF Temperature IS Cold THEN Comfort IS Low",
            "IF Temperature IS Warm THEN Comfort IS Medium",
            "IF Temperature IS Hot THEN Comfort IS High",
            "IF Comfort IS Low THEN FanSpeed IS Fast",
            "IF Comfort IS Medium THEN FanSpeed IS Medium",
            "IF Comfort IS High THEN FanSpeed IS Slow",
        ]
    )

    # Layered inference: first infer Comfort from Temperature, then infer FanSpeed from Comfort
    FS.set_variable("Temperature", 30)

    # First pass: infer Comfort
    comfort_result = FS.Mamdani_inference(["Comfort"])  # returns crisp value(s)
    assert "Comfort" in comfort_result
    comfort_value = comfort_result["Comfort"]
    FS.set_variable("Comfort", comfort_value)

    # Second pass: infer FanSpeed
    fan_result = FS.Mamdani_inference(["FanSpeed"])
    assert "FanSpeed" in fan_result
    fan_speed = fan_result["FanSpeed"]
    # Fan speed is a crisp value within the defined universe
    assert 0 <= fan_speed <= 100
    print({"Comfort": comfort_value, "FanSpeed": fan_speed})
