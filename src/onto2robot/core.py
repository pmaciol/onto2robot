from pathlib import Path

from owlready2 import EntityClass, Ontology, Thing, ThingClass, get_ontology, onto_path

OntologyIndividualSuperclass = Thing
OntologyClassSuperclass = EntityClass
OntologyClass = ThingClass


class MobileOntologyMeta:
    def __init__(self, ontology: Ontology) -> None:
        self.ontology = ontology

    def get_rules(self) -> list[OntologyIndividualSuperclass]:
        rules_class: ThingClass = self.ontology.RuleHeader
        return list(rules_class.instances())


def load_ontology(ontology_name: str) -> Ontology:
    project_root = Path(__file__).resolve().parents[2]
    onto_path.append(project_root / "ontologies")
    path_to_file = (project_root / "ontologies" / Path(ontology_name)).with_suffix(".owl")
    onto = get_ontology(path_to_file.resolve().as_uri()).load()
    return onto
