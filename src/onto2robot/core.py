from pathlib import Path

from owlready2 import EntityClass, Ontology, Thing, ThingClass, get_ontology, onto_path

OntologyIndividualSuperclass = Thing
OntologyClassSuperclass = EntityClass
OntologyClass = ThingClass


class MobileOntologyMeta:
    def __init__(self, ontology: Ontology | str) -> None:
        if isinstance(ontology, str):
            ontology = load_ontology(ontology).load()
        self.ontology = ontology

    def __del__(self):
        self.ontology.destroy()

    def get_rules(self) -> list[OntologyIndividualSuperclass]:
        rules_class: ThingClass = self.ontology.RuleHeader
        return list(rules_class.instances())

    def destroy(self) -> None:
        self.ontology.destroy()


def _get_property_values(entity, property_name) -> list[OntologyIndividualSuperclass]:
    return [prop[entity] for prop in entity.get_properties() if prop.name == property_name][0]


def _get_left_right_hands(entity):
    left_hand = _get_property_values(entity, "hasLeftHand")
    right_hand = _get_property_values(entity, "hasRightHand")
    return left_hand[0], right_hand[0]


def rule_to_string(rule) -> str:
    premises = _get_premises(rule)
    conclusions = _get_conclusions(rule)
    return _build_rule_from_premises(premises, conclusions)


def _build_rule_from_premises(premises, conclusions) -> str:
    premise_parts = []
    for premise in premises:
        left, right = _get_left_right_hands(premise)
        premise_parts.append(f"({left.name} IS {right.name})")
    premise_str = " AND ".join(premise_parts)

    conclusion = conclusions[0]
    left, right = _get_left_right_hands(conclusion)
    return f"IF {premise_str} THEN ({left.name} IS {right.name});"


def _get_premises(rule):
    return _get_property_values(rule, "hasPremise")


def _get_conclusions(rule):
    return _get_property_values(rule, "hasConclusion")


def load_ontology(ontology_name: str) -> Ontology:
    project_root = Path(__file__).resolve().parents[2]
    onto_path.append(project_root / "ontologies")
    path_to_file = (project_root / "ontologies" / Path(ontology_name)).with_suffix(".owl")
    return get_ontology(path_to_file.resolve().as_uri()).load()
