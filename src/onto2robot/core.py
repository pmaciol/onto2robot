import contextlib
from pathlib import Path

from owlready2 import EntityClass, IndividualValueList, Ontology, Thing, ThingClass, World, onto_path

OntologyIndividualSuperclass = Thing
OntologyClassSuperclass = EntityClass
OntologyClass = ThingClass


def _get_class_by_name(ontology: Ontology, class_name: str) -> OntologyClassSuperclass | None:
    for cls in ontology.classes():
        if cls.name == class_name:
            return cls
    for imported_onto in ontology.imported_ontologies:
        cls = _get_class_by_name(imported_onto, class_name)
        if cls:
            return cls
    return None


def _get_property_values(entity: OntologyIndividualSuperclass, property_name: str) -> list[OntologyIndividualSuperclass]:
    properties = [prop[entity] for prop in entity.get_properties() if prop.name == property_name]
    if properties:
        if any(not isinstance(prop, OntologyIndividualSuperclass) for prop in properties[0]):
            print(f"Warning: Property '{property_name}' of entity '{entity.name}' is not an object property.")
        return properties[0]
    print(f"Warning: Property '{property_name}' of entity '{entity.name}' not found. Returning empty list.")
    return []


def _get_data_property_value(entity: OntologyIndividualSuperclass, property_name: str) -> int | float | str:
    properties = [prop[entity] for prop in entity.get_properties() if prop.name == property_name]
    if properties:
        if any(isinstance(prop, OntologyIndividualSuperclass) for prop in properties[0]):
            print(f"Warning: Property '{property_name}' of entity '{entity.name}' is not a data property.")
        return properties[0][0] if isinstance(properties[0], IndividualValueList) else properties[0]
    print(f"Warning: Property '{property_name}' of entity '{entity.name}' not found. Returning empty string.")
    return ""


def _get_sorted_domain(linguistic_variable_space: OntologyIndividualSuperclass) -> list[OntologyIndividualSuperclass]:
    unsorted = set(_get_property_values(linguistic_variable_space, "hasValues"))
    with_order: list[tuple[int, OntologyIndividualSuperclass]] = [
        (int(_get_data_property_value(value, "hasOrder")), value) for value in unsorted
    ]
    return [v for _, v in sorted(with_order, key=lambda x: x[0])]


def _get_left_right_hands(
    entity: OntologyIndividualSuperclass,
) -> tuple[OntologyIndividualSuperclass, OntologyIndividualSuperclass]:
    left_hand = _get_property_values(entity, "hasLeftHand")
    right_hand = _get_property_values(entity, "hasRightHand")
    if not left_hand or not right_hand:
        raise ValueError(
            f"Entity '{entity.name}' does not have both 'hasLeftHand' and 'hasRightHand' properties. {left_hand} {right_hand}"
        )
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


def _load_ontology_from_file(ontology_name: str, world: World, ontologies_path: Path | None = None) -> Ontology:
    project_root = ontologies_path if ontologies_path else Path(__file__).resolve().parents[2] / "ontologies"
    onto_path.append(project_root)
    ontology_getter = world.get_ontology
    path_to_file = (project_root / Path(ontology_name)).with_suffix(".owl")
    uri = path_to_file.resolve().as_uri()
    print(f"Loading ontology from file: {path_to_file}")
    ontology: Ontology = ontology_getter(uri).load()
    if ontology is None:
        raise ValueError(f"Failed to load ontology: {ontology_name}")
    return ontology


def load_ontology(
    ontology_name: str,
    world: World,
    ontologies_path: Path | None = None,
    supporting_ontologies: list[str] | None = None,
) -> Ontology:
    if supporting_ontologies is None:
        # Default set of ontologies for amro use case
        supporting_ontologies = ["amro_uc01", "amro", "sumo"]
    main_onto = _load_ontology_from_file(ontology_name, world, ontologies_path)
    imported_ontologies = [_load_ontology_from_file(name, world, ontologies_path) for name in supporting_ontologies]
    for imported_onto in imported_ontologies:
        main_onto.imported_ontologies.append(imported_onto)

    print("Ontologies loaded successfully:")
    return main_onto


class LinguisticVariableSpaces:
    def __init__(self, linguistic_class: OntologyIndividualSuperclass, fuzzy_points: list[float]):
        self.linguistic_class = linguistic_class
        self.fuzzy_points = fuzzy_points


class LinguisticVariableDomain:
    def __init__(self, linguistic_domain: list[OntologyIndividualSuperclass], fuzzy_points: list[float]):
        self.linguistic_domain = linguistic_domain
        self.fuzzy_points = fuzzy_points


def _get_instances(cls: OntologyClassSuperclass) -> list[OntologyIndividualSuperclass]:
    if hasattr(cls, "instances"):
        return list(cls.instances())  # pyright: ignore[reportAttributeAccessIssue]; works properly in runtime, but pyright cannot detect it
    return []


class MobileOntologyMeta:
    def __init__(
        self,
        ontology: Ontology | str,
        ontologies_path: Path | None = None,
    ) -> None:
        if isinstance(ontology, str):
            self._world = World()
            ontology = load_ontology(ontology, world=self._world, ontologies_path=ontologies_path)
        if not isinstance(ontology, Ontology):
            raise ValueError(f"Failed to load ontology from {ontology}")
        self.ontology: Ontology = ontology
        self._world = ontology.world
        self._destroyed = False

    def __del__(self):
        with contextlib.suppress(Exception):
            self.destroy()

    def get_rules(self) -> list[OntologyIndividualSuperclass]:
        rules_class: OntologyClassSuperclass | None = _get_class_by_name(self.ontology, "RuleHeader")
        if rules_class:
            return list(_get_instances(rules_class))
        return []

    def destroy(self) -> None:
        if self._destroyed:
            return

        try:
            self.ontology.destroy()
        except KeyError:
            pass
        finally:
            if self._world is not None:
                with contextlib.suppress(Exception):
                    self._world.close()
                self._world = None
            self._destroyed = True

    def rules_as_strings(self) -> list[str]:
        return [rule_to_string(rule) for rule in self.get_rules()]

    def get_individual_by_name(self, name: str) -> OntologyIndividualSuperclass | None:
        for individual in self.ontology.individuals():
            if individual.name == name:
                return individual
        return None

    def get_linguistic_variable_spaces(self) -> dict[OntologyIndividualSuperclass, LinguisticVariableSpaces]:
        fi_class = _get_class_by_name(self.ontology, "FuzzyInstances")
        if not fi_class:
            print("FuzzyInstances class not found in the ontology.")
            return {}
        linguistic_variable_classes: list[OntologyIndividualSuperclass] = list(_get_instances(fi_class))

        linguistic_variable_spaces: dict[OntologyIndividualSuperclass, LinguisticVariableSpaces] = {}
        for linguistic_variable_class in linguistic_variable_classes:
            try:
                has_parameters = int(_get_data_property_value(linguistic_variable_class, "hasParameters"))
                param_values: list[float] = []
                for i in range(1, has_parameters + 1):
                    param_values.append(float(_get_data_property_value(linguistic_variable_class, f"has{i}.Parameter")))

                is_type_of = (
                    _get_property_values(linguistic_variable_class, "isTypeOf")[0]
                    if _get_property_values(linguistic_variable_class, "isTypeOf")
                    else None
                )
                if is_type_of is not None:
                    linguistic_variable_spaces[linguistic_variable_class] = LinguisticVariableSpaces(
                        is_type_of,
                        param_values,
                    )
                else:
                    print(
                        f"Warning: Linguistic variable class '{linguistic_variable_class.name}' "
                        "does not have 'isTypeOf' property. Skipping."
                    )
            except Exception as e:
                raise Exception(
                    f"Error while processing linguistic variable classes: {e} {linguistic_variable_class.name} \
                    {_get_data_property_value(linguistic_variable_class, 'hasParameters')}"
                ) from e

        print("Linguistic variables classes spaces:")
        for vc, sp in linguistic_variable_spaces.items():
            print(f" - {vc.name}: {sp.linguistic_class.name if sp else 'None'} {sp.fuzzy_points if sp else []}")

        return linguistic_variable_spaces

    def get_linguistic_variable_domains(self) -> dict[OntologyIndividualSuperclass, LinguisticVariableDomain]:
        linguistic_variable_spaces = self.get_linguistic_variable_spaces()
        linguistic_variable_domains: dict[OntologyIndividualSuperclass, LinguisticVariableDomain] = {}
        for linguistic_variable_class, linguistic_variable_space in linguistic_variable_spaces.items():
            domain = _get_sorted_domain(linguistic_variable_space.linguistic_class)
            linguistic_variables_of_class = _get_property_values(linguistic_variable_class, "represents") or []
            print(f"Linguistic variable class {linguistic_variable_class} represents: {linguistic_variables_of_class}")
            for linguistic_variable in linguistic_variables_of_class:
                variables_extending = _get_property_values(linguistic_variable, "support")
                if variables_extending:
                    for l_val in variables_extending:
                        linguistic_variable_domains[l_val] = LinguisticVariableDomain(
                            domain, linguistic_variable_space.fuzzy_points
                        )
                else:
                    linguistic_variable_domains[linguistic_variable] = LinguisticVariableDomain(
                        domain, linguistic_variable_space.fuzzy_points
                    )

        print("Final linguistic variable domains mapping:")
        for domain, values in linguistic_variable_domains.items():
            print(
                f" - variable: {domain.name} -> domain values{[v.name for v in values.linguistic_domain]}"
                f" with fuzzy points {values.fuzzy_points}"
            )
        return linguistic_variable_domains

    def get_possible_chains(
        self, goals: list[OntologyIndividualSuperclass]
    ) -> tuple[list[set[OntologyIndividualSuperclass]], set[OntologyIndividualSuperclass]]:
        def get_precedents(goal):
            rules_class: OntologyClassSuperclass | None = _get_class_by_name(self.ontology, "RuleHeader")
            if not rules_class:
                raise ValueError("RuleHeader class not found in the ontology.")
            precedents = set()
            input_individuals = set()
            for rule in _get_instances(rules_class):
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
