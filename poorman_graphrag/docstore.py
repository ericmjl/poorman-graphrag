"""Knowledge Graph document store for LlamaBot."""

import json
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Set

import llamabot as lmb
import networkx as nx
from chonkie import SDPMChunker
from llamabot.components.docstore import AbstractDocumentStore
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

        If a relationship between the same entities with the same type already exists,
        combines the summaries of both relationships.

        :param relation: Relation object
        :return: Relation hash
        """
        source: Entity = relation.source
        target: Entity = relation.target
        source_hash = source.hash()
        target_hash = target.hash()

        # Check if edge already exists
        if self._graph.has_edge(source_hash, target_hash):
            existing_data = self._graph.edges[source_hash, target_hash]
            if existing_data["pydantic_model"].relation_type == relation.relation_type:
                # Reconstruct existing relationship and combine with new one
                existing_relation = existing_data["pydantic_model"]
                combined_relation = existing_relation + relation
                chunk_hashes = list(set(existing_data["chunk_hash"] + [chunk_hash]))
                self._graph.edges[source_hash, target_hash] = {
                    "pydantic_model": combined_relation,
                    "chunk_hashes": chunk_hashes,
                }
            else:
                # Different relation type, add as new edge
                self._graph.add_edge(
                    source_hash,
                    target_hash,
                    pydantic_model=relation,
                    chunk_hashes=[chunk_hash],
                )
        else:
            # Add new edge if it doesn't exist
            self._graph.add_edge(
                source_hash,
                target_hash,
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
        """Save the docstore to disk."""

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "doc_index": self.doc_index,
            "chunk_index": self.chunk_index,
            "entity_index": {k: v.model_dump() for k, v in self.entity_index.items()},
            "relation_index": {
                k: v.model_dump() for k, v in self.relation_index.items()
            },
            "community_index": {
                k: v.model_dump() for k, v in self.community_index.items()
            },
            "doc_chunk_links": {k: list(v) for k, v in self.doc_chunk_links.items()},
            "chunk_entity_links": {
                k: list(v) for k, v in self.chunk_entity_links.items()
            },
            "chunk_relation_links": {
                k: list(v) for k, v in self.chunk_relation_links.items()
            },
            "entity_chunk_links": {
                k: list(v) for k, v in self.entity_chunk_links.items()
            },
            "entity_community_links": {
                k: list(v) for k, v in self.entity_community_links.items()
            },
        }

        with self.storage_path.open("w") as f:
            json.dump(data, f)

    def _load(self) -> None:
        """Load the docstore from disk."""

        with self.storage_path.open("r") as f:
            data = json.load(f)

        self.doc_index = data["doc_index"]
        self.chunk_index = data["chunk_index"]
        self.entity_index = {k: Entity(**v) for k, v in data["entity_index"].items()}
        self.relation_index = {
            k: Relationship(**v) for k, v in data["relation_index"].items()
        }
        self.community_index = {
            k: Community(**v) for k, v in data["community_index"].items()
        }
        self.doc_chunk_links = {k: set(v) for k, v in data["doc_chunk_links"].items()}
        self.chunk_entity_links = {
            k: set(v) for k, v in data["chunk_entity_links"].items()
        }
        self.chunk_relation_links = {
            k: set(v) for k, v in data["chunk_relation_links"].items()
        }
        self.entity_chunk_links = {
            k: set(v) for k, v in data["entity_chunk_links"].items()
        }
        self.entity_community_links = {
            k: set(v) for k, v in data["entity_community_links"].items()
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


"""Network graph representation of GraphRAG data."""


# from poorman_graphrag.docstore import KnowledgeGraphDocStore


def build_network(
    docstore: "KnowledgeGraphDocStore",
    include_documents: bool = False,
    include_chunks: bool = False,
) -> nx.MultiDiGraph:
    """Build a NetworkX graph from a KnowledgeGraphDocStore.

    The graph will contain nodes for entities and relationships, with optional
    document and chunk nodes. Edges represent the connections between these elements.

    :param docstore: KnowledgeGraphDocStore instance to convert to a network
    :param include_documents: Whether to include document nodes in the graph
    :param include_chunks: Whether to include chunk nodes in the graph
    :return: NetworkX MultiDiGraph containing the network representation
    """
    G = nx.MultiDiGraph()

    # Add document nodes if requested
    if include_documents:
        for doc_hash, doc_text in docstore.doc_index.items():
            G.add_node(doc_hash, type="document", text=doc_text)

    # Add chunk nodes and connect to documents if requested
    if include_chunks:
        for chunk_hash, chunk_text in docstore.chunk_index.items():
            G.add_node(chunk_hash, type="chunk", text=chunk_text)

            # Find parent document and add edge if documents are included
            if include_documents:
                for doc_hash, chunk_hashes in docstore.doc_chunk_links.items():
                    if chunk_hash in chunk_hashes:
                        G.add_edge(doc_hash, chunk_hash, type="contains")

    # Add entity nodes
    for entity_hash, entity in docstore.entity_index.items():
        G.add_node(
            entity_hash,
            type="entity",
            entity_type=entity.entity_type,
            name=entity.name,
            summary=entity.summary,
        )

        # Connect entities to their chunks if chunks are included
        if include_chunks:
            for chunk_hash, entity_hashes in docstore.chunk_entity_links.items():
                if entity_hash in entity_hashes:
                    G.add_edge(chunk_hash, entity_hash, type="mentions")

    # Add relationship nodes and edges
    for rel_hash, relation in docstore.relation_index.items():
        # Get source and target entity hashes
        source_hash = relation.source.hash()
        target_hash = relation.target.hash()

        # Add the relationship edge between entities
        G.add_edge(
            source_hash,
            target_hash,
            key=rel_hash,  # Use rel_hash as edge key for multigraph
            type="relationship",
            relation_type=relation.relation_type,
            summary=relation.summary,
        )

        # Connect relationships to chunks if chunks are included
        if include_chunks:
            for chunk_hash, rel_hashes in docstore.chunk_relation_links.items():
                if rel_hash in rel_hashes:
                    # Add edge from chunk to both entities involved in relationship
                    G.add_edge(chunk_hash, source_hash, type="mentions")
                    G.add_edge(chunk_hash, target_hash, type="mentions")

    return G


# def get_entity_subgraph(
#     G: nx.MultiDiGraph,
#     entity_hash: str,
#     n_hops: int = 2,
#     edge_types: Optional[Set[str]] = None,
# ) -> nx.MultiDiGraph:
#     """Extract a subgraph centered on a specific entity.

#     :param G: Full network graph
#     :param entity_hash: Hash of the entity to center the subgraph on
#     :param n_hops: Number of hops (edge traversals) to include in subgraph
#     :param edge_types: Set of edge types to traverse. If None, traverse all edges.
#     :return: Subgraph centered on the specified entity
#     """
#     if edge_types is None:
#         edge_types = {"mentions", "relationship", "contains"}

#     # Get nodes within n_hops of entity
#     nodes = {entity_hash}
#     current_nodes = {entity_hash}

#     for _ in range(n_hops):
#         next_nodes = set()
#         for node in current_nodes:
#             # Get neighbors through specified edge types
#             for _, nbr, edge_data in G.edges(node, data=True):
#                 if edge_data["type"] in edge_types:
#                     next_nodes.add(nbr)
#             for nbr, _, edge_data in G.in_edges(node, data=True):
#                 if edge_data["type"] in edge_types:
#                     next_nodes.add(nbr)
#         current_nodes = next_nodes - nodes
#         nodes.update(next_nodes)

#     # Return induced subgraph on selected nodes
#     return G.subgraph(nodes).copy()


# def get_chunk_context(
#     G: nx.MultiDiGraph, entity_hash: str, n_hops: int = 2
# ) -> Dict[str, str]:
#     """Get the text of chunks that provide context for an entity.

#     :param G: Network graph
#     :param entity_hash: Hash of entity to get context for
#     :param n_hops: Number of hops to traverse when finding related chunks
#     :return: Dictionary mapping chunk hashes to chunk text
#     """
#     # Get subgraph around entity
#     subgraph = get_entity_subgraph(
#         G, entity_hash, n_hops=n_hops, edge_types={"mentions", "contains"}
#     )

#     # Find chunks in subgraph
#     chunks = {}
#     for node in subgraph.nodes():
#         node_data = subgraph.nodes[node]
#         if node_data.get("type") == "chunk":
#             chunks[node] = node_data["text"]

#     return chunks
