"""
Index for tracking GraphRAG data.
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Set

from poorman_graphrag.communities import Communities, Community
from poorman_graphrag.entities import (
    Entities,
    Entity,
)
from poorman_graphrag.relationships import Relationship, Relationships


@dataclass
class GraphRAGIndex:
    """Index for tracking GraphRAG data.

    :ivar doc_index: Dictionary mapping document hashes to document text
    :ivar chunk_index: Dictionary mapping chunk hashes to chunk text
    :ivar entity_index: Dictionary mapping entity hashes to Entity objects
    :ivar relation_index: Dictionary mapping relation hashes to Relation objects
    :ivar doc_chunk_links: Dictionary mapping document hashes to sets of chunk hashes
    :ivar chunk_entity_links: Dictionary mapping chunk hashes to sets of entity hashes
    :ivar chunk_relation_links: Dictionary mapping chunk hashes
        to sets of relation hashes
    :ivar entity_chunk_links: Dictionary mapping entity hashes to sets of chunk hashes
    :ivar community_index: Dictionary mapping community hashes to Community objects
    """

    def __init__(self):
        self.doc_index: Dict[str, str] = {}
        self.chunk_index: Dict[str, str] = {}
        self.entity_index: Dict[str, Entity] = {}
        self.relation_index: Dict[str, Relationship] = {}
        self.community_index: Dict[str, Community] = {}
        self.doc_chunk_links: Dict[str, Set[str]] = {}
        self.chunk_entity_links: Dict[str, Set[str]] = {}
        self.chunk_relation_links: Dict[str, Set[str]] = {}
        self.entity_chunk_links: Dict[str, Set[str]] = {}
        self.entity_community_links: Dict[
            str, Set[str]
        ] = {}  # enables me to quickly grab out the community associated with an entity

    def add_document(self, text: str) -> str:
        """Add document to index.

        :param text: Document text
        :return: Document hash
        """
        doc_hash = sha256(text.encode()).hexdigest()
        self.doc_index[doc_hash] = text
        self.doc_chunk_links[doc_hash] = set()
        return doc_hash

    def add_chunk(self, doc_hash: str, chunk_text: str) -> str:
        """Add chunk to index and link to doc OKument.

        :param doc_hash: Hash of parent document
        :param chunk_text: Chunk text
        :return: Chunk hash
        """
        chunk_hash = sha256(chunk_text.encode()).hexdigest()
        self.chunk_index[chunk_hash] = chunk_text
        self.chunk_entity_links[chunk_hash] = set()
        self.chunk_relation_links[chunk_hash] = set()
        self.doc_chunk_links[doc_hash].add(chunk_hash)
        return chunk_hash

    def add_entities(self, chunk_hash: str, entities: Entities) -> List[str]:
        """Add entities to index and link to chunk.
        If an entity with the same hash already exists,
        merges the new entity with the existing one.

        :param chunk_hash: Hash of parent chunk
        :param entities: List of Entity objects
        :return: List of entity hashes
        """
        entity_hashes = []
        for entity in entities:
            entity_hash = entity.hash()
            if entity_hash in self.entity_index:
                # Merge with existing entity using __add__ operator
                self.entity_index[entity_hash] = self.entity_index[entity_hash] + entity
            else:
                self.entity_index[entity_hash] = entity

            self.chunk_entity_links[chunk_hash].add(entity_hash)
            entity_hashes.append(entity_hash)
        return entity_hashes

    def add_relations(self, chunk_hash: str, relations: Relationships) -> List[str]:
        """Add relations to index and link to chunk.
        Also adds any new entities found in the relations.

        :param chunk_hash: Hash of parent chunk
        :param relations: List of Relation objects
        :return: List of relation hashes
        """
        # First collect all entities from relations
        new_entities = []
        for relation in relations:
            source_entity = relation.source  # Already an Entity object
            target_entity = relation.target  # Already an Entity object
            new_entities.extend([source_entity, target_entity])

        # Add any new entities
        self.add_entities(chunk_hash, new_entities)

        # Now add the relations
        relation_hashes = []
        for relation in relations:
            relation_hash = relation.hash()
            self.relation_index[relation_hash] = relation
            self.chunk_relation_links[chunk_hash].add(relation_hash)
            relation_hashes.append(relation_hash)
        return relation_hashes

    def add_communities(self, communities: Communities) -> List[str]:
        """Add communities to the index.

        :param communities: Communities object containing communities to add
        :return: List of community hashes
        """
        community_hashes = []
        for community in communities.communities:
            community_hash = community.hash()
            self.community_index[community_hash] = community
            community_hashes.append(community_hash)

            # Update entity_community_links for each entity in the community
            for node_hash in community.nodes:
                if node_hash in self.entity_index:  # Only link if it's an entity
                    if node_hash not in self.entity_community_links:
                        self.entity_community_links[node_hash] = set()
                    self.entity_community_links[node_hash].add(community_hash)

    def save(self, path: str | Path) -> None:
        """Save index to JSON file.

        :param path: Path to save JSON file
        """
        path = Path(path)
        data = {
            "doc_index": self.doc_index,
            "chunk_index": self.chunk_index,
            "entity_index": {k: v.model_dump() for k, v in self.entity_index.items()},
            "relation_index": {
                k: v.model_dump() for k, v in self.relation_index.items()
            },
            "doc_chunk_links": {k: list(v) for k, v in self.doc_chunk_links.items()},
            "chunk_entity_links": {
                k: list(v) for k, v in self.chunk_entity_links.items()
            },
            "chunk_relation_links": {
                k: list(v) for k, v in self.chunk_relation_links.items()
            },
            "community_index": {
                k: v.model_dump() for k, v in self.community_index.items()
            },
            "entity_community_links": {
                k: list(v) for k, v in self.entity_community_links.items()
            },
        }
        with path.open("w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "GraphRAGIndex":
        """Load index from JSON file.

        :param path: Path to JSON file
        :return: GraphRAGIndex instance
        """
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)

        index = cls()
        index.doc_index = data["doc_index"]
        index.chunk_index = data["chunk_index"]
        index.entity_index = {k: Entity(**v) for k, v in data["entity_index"].items()}
        index.relation_index = {
            k: Relationship(**v) for k, v in data["relation_index"].items()
        }
        index.doc_chunk_links = {k: set(v) for k, v in data["doc_chunk_links"].items()}
        index.chunk_entity_links = {
            k: set(v) for k, v in data["chunk_entity_links"].items()
        }
        index.chunk_relation_links = {
            k: set(v) for k, v in data["chunk_relation_links"].items()
        }
        index.community_index = {
            k: Community(**v) for k, v in data.get("community_index", {}).items()
        }
        index.entity_community_links = {
            k: set(v) for k, v in data.get("entity_community_links", {}).items()
        }
        return index

    @property
    def entities(self) -> Entities:
        """Get all entities stored in the index as an Entities object.

        :return: Entities instance containing all entities in the index
        """
        return Entities(
            entities=[entity.model_dump() for entity in self.entity_index.values()]
        )

    def deduplicate_entities(
        self, entity_groups_to_deduplicate: Dict[tuple[str, str], List[Entity]]
    ) -> "GraphRAGIndex":
        """Deduplicate entities in the index.

        This method takes a dictionary of entity groups to deduplicate,
        where each group contains entities that should be merged into one.
        For each group, the first entity is considered the canonical entity,
        and all other entities are merged into it.

        :param entity_groups_to_deduplicate: Dictionary mapping
            (entity_type, normalized_name) tuples to a list of Entity objects
            that should be merged.
        :return: New GraphRAGIndex with deduplicated entities
        """
        # Create a new instance to maintain immutability
        new_index = GraphRAGIndex()
        new_index.doc_index = self.doc_index.copy()
        new_index.chunk_index = self.chunk_index.copy()
        new_index.doc_chunk_links = self.doc_chunk_links.copy()

        # Create mapping from old entity hashes to new (canonical) entity hashes
        hash_mapping = {}
        new_entity_index = {}

        # First, copy over all entities that aren't being deduplicated
        all_entities_to_dedupe = [
            entity.hash()
            for entities in entity_groups_to_deduplicate.values()
            for entity in entities
        ]
        for entity_hash, entity in self.entity_index.items():
            if entity_hash not in all_entities_to_dedupe:
                new_entity_index[entity_hash] = entity
                hash_mapping[entity_hash] = entity_hash

        # Then handle the deduplication groups
        for entities in entity_groups_to_deduplicate.values():
            canonical_entity = entities[0]  # First entity is canonical
            canonical_hash = canonical_entity.hash()

            # Merge all entities in the group into the canonical entity
            merged_entity = canonical_entity
            for entity in entities[1:]:
                merged_entity = merged_entity + entity
                hash_mapping[entity.hash()] = canonical_hash

            new_entity_index[canonical_hash] = merged_entity
            hash_mapping[canonical_hash] = canonical_hash

        new_index.entity_index = new_entity_index

        # Update relation index to use new entity hashes
        new_relation_index = {}
        for rel_hash, relation in self.relation_index.items():
            source_hash = hash_mapping.get(
                relation.source.hash(), relation.source.hash()
            )
            target_hash = hash_mapping.get(
                relation.target.hash(), relation.target.hash()
            )

            # Create new relation with updated entity references
            new_relation = Relationship(
                source=new_entity_index[source_hash].model_dump(),
                target=new_entity_index[target_hash].model_dump(),
                relation_type=relation.relation_type,
                summary=relation.summary,
            )
            new_relation_index[new_relation.hash()] = new_relation

        new_index.relation_index = new_relation_index

        # Update chunk-entity links
        new_chunk_entity_links = {}
        for chunk_hash, entity_hashes in self.chunk_entity_links.items():
            new_chunk_entity_links[chunk_hash] = {
                hash_mapping.get(eh, eh) for eh in entity_hashes
            }
        new_index.chunk_entity_links = new_chunk_entity_links

        # Update chunk-relation links
        new_chunk_relation_links = {}
        for chunk_hash, relation_hashes in self.chunk_relation_links.items():
            new_chunk_relation_links[chunk_hash] = set(relation_hashes)
        new_index.chunk_relation_links = new_chunk_relation_links

        # Update entity-chunk links
        new_entity_chunk_links = {}
        for entity_hash, chunk_hashes in self.entity_chunk_links.items():
            canonical_hash = hash_mapping.get(entity_hash, entity_hash)
            if canonical_hash not in new_entity_chunk_links:
                new_entity_chunk_links[canonical_hash] = set()
            new_entity_chunk_links[canonical_hash].update(chunk_hashes)
        new_index.entity_chunk_links = new_entity_chunk_links

        return new_index
