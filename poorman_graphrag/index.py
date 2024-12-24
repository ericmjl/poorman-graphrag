"""
Index for tracking GraphRAG data.
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Set

from poorman_graphrag.entities import Entity
from poorman_graphrag.relationships import Relationship


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
    """

    doc_index: Dict[str, str] = None
    chunk_index: Dict[str, str] = None
    entity_index: Dict[str, Entity] = None
    relation_index: Dict[str, Relationship] = None
    doc_chunk_links: Dict[str, Set[str]] = None
    chunk_entity_links: Dict[str, Set[str]] = None
    chunk_relation_links: Dict[str, Set[str]] = None

    def __post_init__(self):
        """Initialize the index."""
        self.doc_index = {}
        self.chunk_index = {}
        self.entity_index = {}
        self.relation_index = {}
        self.doc_chunk_links = {}
        self.chunk_entity_links = {}
        self.chunk_relation_links = {}

    def _hash_text(self, text: str) -> str:
        """Generate SHA256 hash of text.

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
        """Add chunk to index and link to document.

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

    def add_entities(self, chunk_hash: str, entities: List[Entity]) -> List[str]:
        """Add entities to index and link to chunk.

        :param chunk_hash: Hash of parent chunk
        :param entities: List of Entity objects
        :return: List of entity hashes
        """
        entity_hashes = []
        for entity in entities:
            entity_hash = self._hash_text(f"{entity.entity_type}:{entity.name}")
            self.entity_index[entity_hash] = entity
            self.chunk_entity_links[chunk_hash].add(entity_hash)
            entity_hashes.append(entity_hash)
        return entity_hashes

    def add_relations(
        self, chunk_hash: str, relations: List[Relationship]
    ) -> List[str]:
        """Add relations to index and link to chunk.
        Also adds any new entities found in the relations.

        :param chunk_hash: Hash of parent chunk
        :param relations: List of Relation objects
        :return: List of relation hashes
        """
        # First collect all entities from relations
        new_entities = []
        for relation in relations:
            source_entity = Entity(
                name=relation.source, entity_type=relation.source_type
            )
            target_entity = Entity(
                name=relation.target, entity_type=relation.target_type
            )
            new_entities.extend([source_entity, target_entity])

        # Add any new entities
        self.add_entities(chunk_hash, new_entities)

        # Now add the relations
        relation_hashes = []
        for relation in relations:
            relation_hash = self._hash_text(
                f"{relation.relation_type}:{relation.source}:{relation.target}"
            )
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
