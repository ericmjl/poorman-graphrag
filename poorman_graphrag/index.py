"""
Index for tracking GraphRAG data.
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Set

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
    """

    def __init__(self):
        self.doc_index: Dict[str, str] = {}
        self.chunk_index: Dict[str, str] = {}
        self.entity_index: Dict[str, Entity] = {}
        self.relation_index: Dict[str, Relationship] = {}
        self.doc_chunk_links: Dict[str, Set[str]] = {}
        self.chunk_entity_links: Dict[str, Set[str]] = {}
        self.chunk_relation_links: Dict[str, Set[str]] = {}
        self.entity_chunk_links: Dict[str, Set[str]] = {}

    def _hash_text(self, text: str) -> str:
        """Generate SHA256 hash of text.

        NOTE: _hash_text should not be a class method. It should be refactored out.
        It's too simple to be a class method.

        :param text: Text to hash
        :return: Hex digest of hash
        """
        return sha256(text.encode()).hexdigest()

    def add_document(self, text: str) -> str:
        """Add document to index.

        :param text: Document text
        :return: Document hash
        """
        doc_hash = self._hash_text(text)
        self.doc_index[doc_hash] = text
        self.doc_chunk_links[doc_hash] = set()
        return doc_hash

    def add_chunk(self, doc_hash: str, chunk_text: str) -> str:
        """Add chunk to index and link to doc OKument.

        :param doc_hash: Hash of parent document
        :param chunk_text: Chunk text
        :return: Chunk hash
        """
        chunk_hash = self._hash_text(chunk_text)
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
        return index

    @property
    def entities(self) -> Entities:
        """Get all entities stored in the index as an Entities object.

        :return: Entities instance containing all entities in the index
        """
        return Entities(
            entities=[entity.model_dump() for entity in self.entity_index.values()]
        )
