from pathlib import Path

from owlready2 import EntityClass, Ontology, Thing, ThingClass, get_ontology, onto_path

OntologyIndividualSuperclass = Thing
OntologyClassSuperclass = EntityClass
OntologyClass = ThingClass


def _get_property_values(
    entity: OntologyIndividualSuperclass, property_name: str
) -> list[OntologyIndividualSuperclass]:
    return [prop[entity] for prop in entity.get_properties() if prop.name == property_name][0]


def _get_left_right_hands(
    entity: OntologyIndividualSuperclass,
) -> tuple[OntologyIndividualSuperclass, OntologyIndividualSuperclass]:
    left_hand = _get_property_values(entity, "hasLeftHand")
    right_hand = _get_property_values(entity, "hasRightHand")
    return left_hand[0], right_hand[0]


def rule_to_string(rule: OntologyIndividualSuperclass) -> str:
    premises = _get_premises(rule)
    conclusions = _get_conclusions(rule)
    return _build_rule_from_premises(premises, conclusions)


def rule_to_pair(rule: OntologyIndividualSuperclass) -> tuple[str, str]:
    premises = _get_premises(rule)
    conclusions = _get_conclusions(rule)
    return rule.name, _build_rule_from_premises(premises, conclusions)


def _build_rule_from_premises(
    premises: list[OntologyIndividualSuperclass], conclusions: list[OntologyIndividualSuperclass]
) -> str:
    premise_parts = []
    for premise in premises:
        left, right = _get_left_right_hands(premise)
        premise_parts.append(f"({left.name} IS {right.name})")
    premise_str = " AND ".join(premise_parts)

    conclusion = conclusions[0]
    left, right = _get_left_right_hands(conclusion)
    return f"IF {premise_str} THEN ({left.name} IS {right.name});"


def _get_premises(rule: OntologyIndividualSuperclass) -> list[OntologyIndividualSuperclass]:
    return _get_property_values(rule, "hasPremise")


def _get_conclusions(rule: OntologyIndividualSuperclass) -> list[OntologyIndividualSuperclass]:
    return _get_property_values(rule, "hasConclusion")


def load_ontology(ontology_name: str) -> Ontology:
    project_root = Path(__file__).resolve().parents[2]
    onto_path.append(project_root / "ontologies")
    path_to_file = (project_root / "ontologies" / Path(ontology_name)).with_suffix(".owl")
    return get_ontology(path_to_file.resolve().as_uri()).load()


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

    def rules_as_strings(self) -> list[str]:
        return [rule_to_string(rule) for rule in self.get_rules()]

    def get_individual_by_name(self, name: str) -> OntologyIndividualSuperclass | None:
        for individual in self.ontology.individuals():
            if individual.name == name:
                return individual
        return None

    def linguistic_values(self) -> dict[OntologyIndividualSuperclass, set[OntologyIndividualSuperclass]]:
        lv_dict = {}
        rules = self.get_rules()
        for rule in rules:
            premises = _get_premises(rule)
            conclusions = _get_conclusions(rule)
            for premise in premises:
                left, right = _get_left_right_hands(premise)
                if left not in lv_dict:
                    lv_dict[left] = set()
                if right not in lv_dict[left]:
                    lv_dict[left].add(right)
            for conclusion in conclusions:
                left, right = _get_left_right_hands(conclusion)
                if left not in lv_dict:
                    lv_dict[left] = set()
                if right not in lv_dict[left]:
                    lv_dict[left].add(right)
        return lv_dict

    def linguistic_value_spaces(
        self, linguistic_spaces: list[list[str]]
    ) -> dict[str, dict[str, OntologyIndividualSuperclass]]:
        lvals = self.linguistic_values()
        lv_space = {}
        for lvalue, items in lvals.items():
            for space in linguistic_spaces:
                if any([it in space for it in [v.name for v in items]]):
                    print(f" Mapping LV {lvalue.name} to space {space}")
                    lv_space[lvalue.name] = space
                    continue
        return lv_space

    def get_possible_chains(
        self, goals: list[OntologyIndividualSuperclass]
    ) -> tuple[list[set[OntologyIndividualSuperclass]], set[OntologyIndividualSuperclass]]:
        def get_precedents(goal):
            rules_class: ThingClass = self.ontology.RuleHeader
            precedents = set()
            input_individuals = set()
            for rule in rules_class.instances():
                conclusions = _get_conclusions(rule)
                for conclusion in conclusions:
                    left, _ = _get_left_right_hands(conclusion)
                    if left == goal:
                        precedents.add(rule)
            for rule in precedents:
                premises = _get_premises(rule)
                for premise in premises:
                    left, _ = _get_left_right_hands(premise)
                    input_individuals.add(left)
            return input_individuals

        source_variables = set()
        layer_inputs = [set(goals)]
        i = 0
        while True:
            layer_inputs.append(set())
            for goal in layer_inputs[i]:
                precedents = get_precedents(goal)
                if not precedents:
                    source_variables.add(goal)
                layer_inputs[i + 1] |= precedents
            if not layer_inputs[i + 1]:
                layer_inputs.pop()
                break
            i += 1

        cleaned_layer_inputs = []
        for layer in layer_inputs:
            cleaned_layer = set()
            for item in layer:
                if item not in source_variables:
                    cleaned_layer.add(item)
            if cleaned_layer:
                cleaned_layer_inputs.append(cleaned_layer)
        return cleaned_layer_inputs, source_variables
