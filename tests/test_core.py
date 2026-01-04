from math import isclose

from simpful import (
    FuzzySet,
    FuzzySystem,
    LinguisticVariable,
    Trapezoidal_MF,
    TriangleFuzzySet,
    Triangular_MF,
)

from onto2robot.core import (
    MobileOntologyMeta,
    OntologyIndividualSuperclass,
    _get_conclusions,
    _get_premises,
    _get_property_values,
    load_ontology,
    rule_to_string,
)
from onto2robot.fs_wrapper import SimpfulFuzzyWrapper


def test_load_sumo_ontology():
    onto = load_ontology("sumo")
    all_classes = list(onto.classes())
    print(f"SUMO classes {all_classes}")
    assert len(all_classes) > 0
    onto.destroy()


def test_load_mobile_robot_ontology():
    onto = load_ontology("mobile_robot_ontology")
    all_classes = list(onto.classes())
    print(f"MOBILE ROBOT classes {all_classes}")
    assert len(all_classes) > 0
    onto.destroy()


def test_read_rules():
    ontology = load_ontology("tests")
    ont = MobileOntologyMeta(ontology)
    rules = ont.get_rules()
    assert len(rules) == 9
    first_rule = rules[0]
    assert first_rule is not None
    premises = _get_premises(first_rule)
    assert len(premises) == 2
    conclusions = _get_conclusions(first_rule)

    assert len(conclusions) == 1
    ontology.destroy()


def test_single_premise():
    ontology = load_ontology("tests")
    premise01 = ontology.premise01
    assert premise01 is not None
    assert premise01.name == "premise01"
    ontology.destroy()


def test_premise_left_right():
    ontology = load_ontology("tests")
    premise01 = ontology.premise01
    leftHand = _get_property_values(premise01, "hasLeftHand")
    assert len(leftHand) == 1
    assert isinstance(leftHand[0], OntologyIndividualSuperclass)
    rightHand = _get_property_values(premise01, "hasRightHand")
    assert len(rightHand) == 1
    ontology.destroy()


def test_whole_rule():
    ont = MobileOntologyMeta("tests")
    rule = ont.get_rules()[0]
    stringified_rule = rule_to_string(rule)
    expected_rule = "IF (sFL IS low) AND (sFR IS low) THEN (sFassessment IS low);"

    assert stringified_rule == expected_rule


def test_add_premise():
    ontology = load_ontology("tests")
    rule01 = ontology.R01
    premises_before = _get_premises(rule01)
    assert len(premises_before) == 2
    new_premise = ontology.Premise()
    new_premise.name = "premise07"
    new_premise.label = "premise07"
    new_premise.hasLeftHand = [ontology.sRF]
    rule01.hasPremise.append(new_premise)
    premises_after = _get_premises(rule01)
    assert len(premises_after) == 3
    ontology.save(file="result.owl", format="rdfxml")
    ontology.destroy()


def test_all_rules():
    ont = MobileOntologyMeta("tests")
    rules = ont.get_rules()
    stringified_rules = [rule_to_string(rule) for rule in rules]
    assert len(stringified_rules) == 9


def test_full_ontology():
    ont = MobileOntologyMeta("extendedAM")
    rules = ont.get_rules()
    stringified_rules = [rule_to_string(rule) for rule in rules]
    with open("results.txt", "w") as f:
        for rule in stringified_rules:
            f.write(rule + "\n")
    expected_rules = [
        "IF (sFL IS low) AND (sFR IS low) THEN (sFassessment IS low);",
        "IF (sFL IS low) AND (sFR IS middle) THEN (sFassessment IS low);",
        "IF (sFR IS high) AND (sFL IS low) THEN (sFassessment IS middle);",
        "IF (sFL IS middle) AND (sFR IS low) THEN (sFassessment IS middle);",
        "IF (sFL IS middle) AND (sFR IS middle) THEN (sFassessment IS middle);",
        "IF (sFR IS high) AND (sFL IS middle) THEN (sFassessment IS high);",
        "IF (sFL IS high) AND (sFR IS low) THEN (sFassessment IS middle);",
        "IF (sFL IS high) AND (sFR IS middle) THEN (sFassessment IS high);",
        "IF (sFR IS high) AND (sFL IS high) THEN (sFassessment IS high);",
    ]

    assert stringified_rules == expected_rules


def test_lingustiic_values():
    ont = MobileOntologyMeta("extendedAM")
    lvals = ont.linguistic_values()
    print("Linguistic values:")
    for k, v in lvals.items():
        print(f" - {k}: {v}")

    for lv in lvals:
        assert len(lvals[lv]) == 3


def test_backward_chain_tree():
    ont = MobileOntologyMeta("extendedAM")
    reasoning_order, source_variables = ont.get_possible_chains([ont.get_individual_by_name("finalMove")])
    for layer in reasoning_order:
        print(f"Required targets: {layer}")
    print(f"Source variables: {source_variables}")

    FS = FuzzySystem()
    FS.set_variable("sRS", 5)
    FS.set_variable("sLF", 10)
    FS.set_variable("sLS", 15)
    FS.set_variable("sFR", 20)
    FS.set_variable("sFL", 25)
    FS.set_variable("bl", 20)
    FS.set_variable("bR", 25)


def test_rules_to_simpful():
    FS = FuzzySystem()
    S_sF_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="low")
    S_sF_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term="middle")
    S_sF_3 = FuzzySet(function=Triangular_MF(a=5, b=10, c=10), term="high")

    FS.add_linguistic_variable(
        "sFL",
        LinguisticVariable([S_sF_1, S_sF_2, S_sF_3], concept="Sensor distance", universe_of_discourse=[0, 10]),
    )

    FS.add_linguistic_variable(
        "sFR",
        LinguisticVariable([S_sF_1, S_sF_2, S_sF_3], concept="Sensor distance", universe_of_discourse=[0, 10]),
    )

    assesment_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="low")
    assesment_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=20), term="middle")
    assesment_3 = FuzzySet(function=Triangular_MF(a=10, b=20, c=20), term="high")
    FS.add_linguistic_variable(
        "sFassessment",
        LinguisticVariable(
            [assesment_1, assesment_2, assesment_3], concept="Test conclusion", universe_of_discourse=[0, 20]
        ),
    )

    ont = MobileOntologyMeta("tests")
    rules = ont.get_rules()
    stringified_rules = [rule_to_string(rule) for rule in rules]

    FS.add_rules(stringified_rules)

    # Set antecedents values

    FS.set_variable("sFL", 5)
    FS.set_variable("sFR", 5)

    # Perform Mamdani inference and print output
    mamadani_result = FS.Mamdani_inference(["sFassessment"], verbose=False, aggregation_function=sum)
    assert isclose(mamadani_result["sFassessment"], 10.0, abs_tol=0.001)


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


def test_layered_robot():
    FS = FuzzySystem()

    # Define fuzzy sets for Temperature
    low = TriangleFuzzySet(0, 0, 15, term="low")
    middle = TriangleFuzzySet(10, 20, 30, term="middle")
    high = TriangleFuzzySet(25, 40, 40, term="high")
    FS.add_linguistic_variable("sLF", LinguisticVariable([low, middle, high], universe_of_discourse=[0, 40]))
    FS.add_linguistic_variable("sLS", LinguisticVariable([low, middle, high], universe_of_discourse=[0, 40]))
    FS.add_linguistic_variable("sFassessment", LinguisticVariable([low, middle, high], universe_of_discourse=[0, 40]))
    FS.add_linguistic_variable("sFR", LinguisticVariable([low, middle, high], universe_of_discourse=[0, 40]))
    FS.add_linguistic_variable("sRS", LinguisticVariable([low, middle, high], universe_of_discourse=[0, 40]))
    FS.add_linguistic_variable("sFL", LinguisticVariable([low, middle, high], universe_of_discourse=[0, 40]))

    ont = MobileOntologyMeta("extendedAM01")
    rules = ont.get_rules()
    stringified_rules = [rule_to_string(rule) for rule in rules]
    FS.add_rules(stringified_rules)

    # Layered inference: first infer Comfort from Temperature, then infer FanSpeed from Comfort
    FS.set_variable("sRS", 5)
    FS.set_variable("sLF", 10)
    FS.set_variable("sLS", 15)
    FS.set_variable("sFR", 20)
    FS.set_variable("sFL", 25)

    comfort_result = FS.Mamdani_inference(["sFassessment"])
    assert "sFassessment" in comfort_result


def test_full_robot():
    ont = MobileOntologyMeta("extendedAM04")
    rules = ont.get_rules()
    goal = "finalMove"
    # TODO: replace with proper extraction from ontology
    linguistic_spaces = [
        ["low", "middle", "high"],
        ["left", "forward", "right"],
    ]
    linguistic_variables_spaces = ont.linguistic_value_spaces(linguistic_spaces)
    reasoning_order, source_variables = ont.get_possible_chains([ont.get_individual_by_name(goal)])

    fs = SimpfulFuzzyWrapper(
        linguistic_variables_spaces,
        universe=(0, 40),
        rules=rules,
    )
    input_values = {
        "sRS": 5,
        "sLF": 10,
        "sLS": 15,
        "sFR": 1,
        "sRF": 1,
        "sFL": 1,
        "bl": 20,
        "bR": 25,
    }
    fs.set_start_values(input_values)
    for layer in reversed(reasoning_order):
        fs.compute(layer)
    assert goal in fs.goals_inferred
