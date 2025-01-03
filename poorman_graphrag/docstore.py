"""Knowledge Graph document store for LlamaBot."""

import json
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Set

import llamabot as lmb
import networkx as nx
from chonkie import SDPMChunker
from llamabot.components.docstore import AbstractDocumentStore
from loguru import logger
from networkx.algorithms.community import louvain_communities

from poorman_graphrag.communities import (
    Community,
    community_content,
    get_community_summarizer,
)
from poorman_graphrag.entities import (
    Entities,
    Entity,
    entity_extractor_user_prompt,
    get_entity_extractor,
    identify_levenshtein_similar,
)
from poorman_graphrag.entity_deduplication import get_entity_judge, is_same_entity
from poorman_graphrag.relationships import (
    Relationship,
    Relationships,
    get_relationship_extractor,
    relationship_extractor_user_prompt,
)


class KnowledgeGraphDocStore(AbstractDocumentStore):
    """Knowledge Graph document store for LlamaBot.

    This class implements AbstractDocumentStore and uses a MultiDiGraph internally
    to store the knowledge graph structure.

    We model the graph's nodes with multiple partitions:

    - `document` nodes: represent the documents that have been added to the docstore
    - `chunk` nodes: represent the chunks of text
      that have been extracted from the documents
    - `entity` nodes: represent the entities that have been extracted from the chunks

    Relationships are stored in the graph's edges directly.
    Community labels are annotated directly onto the nodes.

    :param storage_path: Path to JSON file for persistent storage
    """

    def __init__(
        self,
        storage_path: Path | str = Path.home() / ".llamabot" / "kg_docstore.json",
        entity_extractor: Optional[lmb.StructuredBot] = None,
        entity_extractor_userprompt: Optional[str] = None,
        relationship_extractor: Optional[lmb.StructuredBot] = None,
        relationship_extractor_userprompt: Optional[str] = None,
        entity_similarity_judge: Optional[lmb.StructuredBot] = None,
        entity_similarity_judge_userprompt: Optional[str] = None,
        community_summarizer: Optional[lmb.StructuredBot] = None,
        community_summarizer_userprompt: Optional[str] = None,
    ):
        self.storage_path = Path(storage_path)
        self.communities: Dict[str, Community] = {}
        self._graph = nx.MultiDiGraph()

        # Load existing data if available
        if self.storage_path.exists():
            self._load()

        # Flag to prevent saving during extend operations
        self._in_extend = False

        # Instantiate chunker
        self.chunker = SDPMChunker()

        # Instantiate entity extractor
        self.entity_extractor: lmb.StructuredBot = (
            entity_extractor or get_entity_extractor()
        )
        self.entity_extractor_user_prompt = (
            entity_extractor_userprompt or entity_extractor_user_prompt
        )

        # Instantiate relationship extractor
        self.relationship_extractor: lmb.StructuredBot = (
            relationship_extractor or get_relationship_extractor()
        )
        self.relationship_extractor_user_prompt = (
            relationship_extractor_userprompt or relationship_extractor_user_prompt
        )

        # Instantiate entity similarity judge
        self.entity_similarity_judge: lmb.StructuredBot = (
            entity_similarity_judge or get_entity_judge()
        )
        self.entity_similarity_judge_user_prompt = (
            entity_similarity_judge_userprompt or is_same_entity
        )

        # Instantiate community summarizer
        self.community_summarizer: lmb.StructuredBot = (
            community_summarizer or get_community_summarizer()
        )
        self.community_summarizer_user_prompt = (
            community_summarizer_userprompt or community_content
        )

    def add_document(self, text: str) -> None:
        """Add document to docstore.

        :param text: Document text
        :return: Document hash
        """
        doc_hash = sha256(text.encode()).hexdigest()
        self._graph.add_node(doc_hash, partition="document", text=text)
        # self._save()

    def add_chunk(self, doc_hash: str, chunk_text: str) -> None:
        """Add chunk to docstore and link to document.

        :param doc_hash: Hash of parent document
        :param chunk_text: Chunk text
        :return: Chunk hash
        """
        if doc_hash not in self.nodes(partition="document"):
            raise ValueError(f"Document with hash {doc_hash} not found")

        chunk_hash = sha256(chunk_text.encode()).hexdigest()
        self._graph.add_node(chunk_hash, partition="chunk", text=chunk_text)
        self._graph.add_edge(doc_hash, chunk_hash, partition="document_chunk")
        # self._save()

    def nodes(self, partition: Optional[str] = None, **attr) -> list:
        """Get nodes from the graph, optionally filtered by partition and attributes.

        :param partition: Partition to filter nodes by
        :param attr: Additional node attributes to filter by
        :return: List of node identifiers
        """
        if partition is not None:
            attr["partition"] = partition
        return [
            n
            for n, d in self._graph.nodes(data=True)
            if all(d.get(k) == v for k, v in attr.items())
        ]

    def add_entity(self, chunk_hash: str, entity: Entity) -> None:
        """Add entity to docstore and link to chunk.

        If the entity already exists, combines the summaries of both entities.

        :param chunk_hash: Hash of parent chunk
        :param entity: Entity object
        :return: Entity hash
        """
        entity_hash = entity.hash()

        # If entity already exists, combine using __add__ operator
        if entity_hash in self.nodes(partition="entity"):
            existing_entity = self._graph.nodes[entity_hash]["pydantic_model"]
            combined_entity = existing_entity + entity
            self._graph.nodes[entity_hash]["pydantic_model"] = combined_entity
        else:
            # Add new entity if it doesn't exist
            self._graph.add_node(entity_hash, partition="entity", pydantic_model=entity)

        self._graph.add_edge(chunk_hash, entity_hash, partition="chunk_entity")
        # self._save()

    def add_relation(self, chunk_hash: str, relation: Relationship) -> None:
        """Add relation to docstore.

        Because we are using MultiDiGraph objects,
        There is absolutely no need to check if edges already exist.
        We can just add a new edge with a unique key.

        :param relation: Relation object
        :return: Relation hash
        """
        source: Entity = relation.source
        target: Entity = relation.target
        source_hash = source.hash()
        target_hash = target.hash()

        # Always add a new edge with a unique key
        self._graph.add_edge(
            source_hash,
            target_hash,
            key=relation.hash(),
            pydantic_model=relation,
            chunk_hashes=[chunk_hash],
        )
        # self._save()

    def add_community(self, community: Community) -> None:
        """Add community to docstore.

        Communities are annotated directly onto the nodes using the "community" kwarg.

        :param community: Community object
        """
        for entity in community.entities:
            self._graph.nodes[entity.hash()]["community"] = community.hash()
            self.communities[community.hash()] = community
        # self._save()

    def append(self, document: str):
        """Append a document to the docstore."""
        # Firstly, add the document
        self.add_document(document)

        doc_hash = sha256(document.encode()).hexdigest()
        # Then, chunk the document and add the chunks
        chunks = self.chunker.chunk(document)
        for chunk in chunks:
            self.add_chunk(doc_hash, chunk.text)

        # Now, extract entities from the chunks
        for chunk in chunks:
            chunk_hash = sha256(chunk.text.encode()).hexdigest()
            entities: Entities = self.entity_extractor(
                self.entity_extractor_user_prompt(
                    chunk.text, self.entities.model_dump_json()
                )
            )
            for entity in entities:
                self.add_entity(chunk_hash, entity)

        # Now, extract entities and relations from the chunks
        for chunk in chunks:
            chunk_hash = sha256(chunk.text.encode()).hexdigest()
            relationships: Relationships = self.relationship_extractor(
                self.relationship_extractor_user_prompt(
                    chunk.text, self.entities.model_dump_json()
                )
            )
            for relation in relationships:
                self.add_relation(chunk_hash, relation)

        # Now, deduplicate entities.
        similar_entities = identify_levenshtein_similar(self.entities)
        entity_groups_to_deduplicate = {}
        for entity_type, entities in similar_entities.items():
            result = self.entity_similarity_judge(
                self.entity_similarity_judge_user_prompt(entities)
            )
            if result:
                entity_groups_to_deduplicate[entity_type] = entities

        self.deduplicate_entities(entity_groups_to_deduplicate)

        # Identify communities in the graph and add them to the docstore.
        entity_nodes: list[str] = [e.hash() for e in self.entities]
        entity_subgraph: nx.MultiDiGraph = self._graph.subgraph(entity_nodes)
        # communities is a list of lists of entity hashes
        communities: list[list[str]] = louvain_communities(
            entity_subgraph.to_undirected()
        )
        logger.info(f"Found {len(communities)} communities")
        for community in communities:
            content = self.community_summarizer_user_prompt(
                entity_subgraph.subgraph(community).nodes(data=True),
                entity_subgraph.subgraph(community).edges(data=True),
            )
            community_summary_response = self.community_summarizer(content)

            self.add_community(
                Community(
                    entities=Entities(
                        entities=[
                            self.nodes[ent]["pydantic_model"].model_dump()
                            for ent in community
                        ]
                    ),
                    summary=community_summary_response.summary,
                )
            )
        # self._save()

    @property
    def entities(self) -> Entities:
        """Get all entities in the docstore."""
        entity_nodes = [
            data["pydantic_model"]
            for _, data in self.nodes(data=True)
            if data.get("partition") == "entity"
        ]
        return Entities(entities=entity_nodes)

    @property
    def documents(self) -> list[str]:
        """Get all documents in the docstore."""
        return [
            data["text"]
            for _, data in self.nodes(data=True)
            if data.get("partition") == "document"
        ]

    @property
    def chunks(self) -> list[str]:
        """Get all chunks in the docstore."""
        return [
            data["text"]
            for _, data in self.nodes(data=True)
            if data.get("partition") == "chunk"
        ]

    def retrieve(
        self,
        chunks: Optional[Set[str]] = None,
        documents: Optional[Set[str]] = None,
        relations: Optional[Set[str]] = None,
        entities: Optional[Set[str]] = None,
        communities: Optional[Set[str]] = None,
        keywords: Optional[str] = None,
        n_results: int = 10,
    ) -> Dict[str, Set]:
        """Retrieve items from the docstore.

        :param chunks: Set of chunk hashes to retrieve
        :param documents: Set of document hashes to retrieve
        :param relations: Set of relation hashes to retrieve
        :param entities: Set of entity hashes to retrieve
        :param communities: Set of community hashes to retrieve
        :param keywords: Keywords to search across all text
        :param n_results: Number of results to return for keyword search
        :return: Dictionary mapping item types to sets of retrieved items
        """
        results = {
            "chunks": set(),
            "documents": set(),
            "relations": set(),
            "entities": set(),
            "communities": set(),
        }

        # Direct hash lookups
        if chunks:
            results["chunks"].update(
                self.chunk_index[h] for h in chunks if h in self.chunk_index
            )
        if documents:
            results["documents"].update(
                self.doc_index[h] for h in documents if h in self.doc_index
            )
        if relations:
            results["relations"].update(
                self.relation_index[h] for h in relations if h in self.relation_index
            )
        if entities:
            results["entities"].update(
                self.entity_index[h] for h in entities if h in self.entity_index
            )
        if communities:
            results["communities"].update(
                self.community_index[h]
                for h in communities
                if h in self.community_index
            )

        # Keyword search using BM25
        if keywords:
            from rank_bm25 import BM25Okapi

            # Search across all text content
            all_text = list(self.chunk_index.values()) + list(self.doc_index.values())
            tokenized_corpus = [doc.split() for doc in all_text]
            bm25 = BM25Okapi(tokenized_corpus)

            tokenized_query = keywords.split()
            doc_scores = bm25.get_scores(tokenized_query)

            # Get top n_results
            top_indices = sorted(
                range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
            )[:n_results]

            for idx in top_indices:
                if idx < len(self.chunk_index):
                    results["chunks"].add(all_text[idx])
                else:
                    results["documents"].add(all_text[idx])

        return results

    def reset(self) -> None:
        """Reset the document store."""
        self.__init__(storage_path=self.storage_path)
        # self._save()

    def _save(self) -> None:
        """Save the docstore to disk.

        Saves the graph structure and all node/edge data to a JSON file.
        Creates parent directories if they don't exist.
        """
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert graph data to serializable format
        data = {"nodes": {}, "edges": {}}

        # Save nodes with their attributes
        for node, attrs in self._graph.nodes(data=True):
            node_data = attrs.copy()

            # Handle special node attributes that need custom serialization
            if "pydantic_model" in node_data:
                node_data["pydantic_model"] = node_data["pydantic_model"].model_dump()

            data["nodes"][node] = node_data

        # Save edges with their attributes
        for u, v, k, attrs in self._graph.edges(data=True, keys=True):
            edge_key = f"{u}|{v}|{k}"
            edge_data = attrs.copy()

            # Handle special edge attributes that need custom serialization
            if "pydantic_model" in edge_data:
                edge_data["pydantic_model"] = edge_data["pydantic_model"].model_dump()
            if "chunk_hashes" in edge_data:
                edge_data["chunk_hashes"] = list(edge_data["chunk_hashes"])

            data["edges"][edge_key] = {
                "source": u,
                "target": v,
                "key": k,
                "data": edge_data,
            }

        # Save communities
        data["communities"] = {k: v.model_dump() for k, v in self.communities.items()}

        with self.storage_path.open("w") as f:
            json.dump(data, f)

    def _load(self) -> None:
        """Load the docstore from disk.

        Loads the graph structure and all node/edge data from a JSON file.
        """
        if not self.storage_path.exists():
            return

        with self.storage_path.open("r") as f:
            data = json.load(f)

        # Load nodes with their attributes
        for node, node_data in data["nodes"].items():
            # Handle special node attributes that need custom deserialization
            if "pydantic_model" in node_data:
                if node_data["partition"] == "entity":
                    node_data["pydantic_model"] = Entity(**node_data["pydantic_model"])
                elif node_data["partition"] == "community":
                    node_data["pydantic_model"] = Community(
                        **node_data["pydantic_model"]
                    )
            self._graph.add_node(node, **node_data)

        # Load edges with their attributes
        for edge_key, edge_info in data["edges"].items():
            u = edge_info["source"]
            v = edge_info["target"]
            k = edge_info["key"]
            edge_data = edge_info["data"]

            # Handle special edge attributes that need custom deserialization
            if "pydantic_model" in edge_data:
                edge_data["pydantic_model"] = Relationship(
                    **edge_data["pydantic_model"]
                )
            if "chunk_hashes" in edge_data:
                edge_data["chunk_hashes"] = list(edge_data["chunk_hashes"])

            self._graph.add_edge(u, v, key=k, **edge_data)

        # Load communities
        self.communities = {
            k: Community(**v) for k, v in data.get("communities", {}).items()
        }

    def deduplicate_entities(
        self, entity_groups_to_deduplicate: Dict[tuple[str, str], List[Entity]]
    ) -> None:
        """Deduplicate entities in the docstore.

        This method takes a dictionary of entity groups to deduplicate,
        where each group contains entities that should be merged into one.
        For each group, the first entity is considered the canonical entity,
        and all other entities are merged into it.

        :param entity_groups_to_deduplicate: Dictionary mapping
            (entity_type, normalized_name) tuples to a list of Entity objects
            that should be merged.
        """
        # Create mapping from old entity hashes to new (canonical) entity hashes
        hash_mapping = {}

        # Then handle the deduplication groups
        for entities in entity_groups_to_deduplicate.values():
            canonical_entity = entities[0]  # First entity is canonical
            canonical_hash = canonical_entity.hash()

            # Merge all entities in the group into the canonical entity
            merged_entity = canonical_entity
            for entity in entities[1:]:
                merged_entity = merged_entity + entity
                old_hash = entity.hash()
                hash_mapping[old_hash] = canonical_hash

                # Rewire all edges from old entity to canonical entity
                for _, neighbor, edge_data in self._graph.out_edges(
                    old_hash, data=True
                ):
                    self._graph.add_edge(canonical_hash, neighbor, **edge_data)
                for neighbor, _, edge_data in self._graph.in_edges(old_hash, data=True):
                    self._graph.add_edge(neighbor, canonical_hash, **edge_data)
                self._graph.remove_node(old_hash)

            # Update the canonical entity in the graph
            if canonical_hash in self._graph:
                self._graph.nodes[canonical_hash]["entity"] = merged_entity
            else:
                self._graph.add_node(canonical_hash, entity=merged_entity)
            hash_mapping[canonical_hash] = canonical_hash

        # Save changes
        # self._save()
