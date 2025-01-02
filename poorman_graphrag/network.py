"""Network graph representation of GraphRAG data."""

from typing import Dict, Optional, Set

import networkx as nx

from poorman_graphrag.index import GraphRAGIndex


def build_network(
    index: GraphRAGIndex,
    include_documents: bool = False,
    include_chunks: bool = False,
) -> nx.MultiDiGraph:
    """Build a NetworkX graph from a GraphRAGIndex.

    The graph will contain nodes for entities and relationships, with optional
    document and chunk nodes. Edges represent the connections between these elements.

    :param index: GraphRAGIndex instance to convert to a network
    :param include_documents: Whether to include document nodes in the graph
    :param include_chunks: Whether to include chunk nodes in the graph
    :return: NetworkX MultiDiGraph containing the network representation
    """
    G = nx.MultiDiGraph()

    # Add document nodes if requested
    if include_documents:
        for doc_hash, doc_text in index.doc_index.items():
            G.add_node(doc_hash, type="document", text=doc_text)

    # Add chunk nodes and connect to documents if requested
    if include_chunks:
        for chunk_hash, chunk_text in index.chunk_index.items():
            G.add_node(chunk_hash, type="chunk", text=chunk_text)

            # Find parent document and add edge if documents are included
            if include_documents:
                for doc_hash, chunk_hashes in index.doc_chunk_links.items():
                    if chunk_hash in chunk_hashes:
                        G.add_edge(doc_hash, chunk_hash, type="contains")

    # Add entity nodes
    for entity_hash, entity in index.entity_index.items():
        G.add_node(
            entity_hash,
            type="entity",
            entity_type=entity.entity_type,
            name=entity.name,
            summary=entity.summary,
        )

        # Connect entities to their chunks if chunks are included
        if include_chunks:
            for chunk_hash, entity_hashes in index.chunk_entity_links.items():
                if entity_hash in entity_hashes:
                    G.add_edge(chunk_hash, entity_hash, type="mentions")

    # Add relationship nodes and edges
    for rel_hash, relation in index.relation_index.items():
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
            for chunk_hash, rel_hashes in index.chunk_relation_links.items():
                if rel_hash in rel_hashes:
                    # Add edge from chunk to both entities involved in relationship
                    G.add_edge(chunk_hash, source_hash, type="mentions")
                    G.add_edge(chunk_hash, target_hash, type="mentions")

    return G


def get_entity_subgraph(
    G: nx.MultiDiGraph,
    entity_hash: str,
    n_hops: int = 2,
    edge_types: Optional[Set[str]] = None,
) -> nx.MultiDiGraph:
    """Extract a subgraph centered on a specific entity.

    :param G: Full network graph
    :param entity_hash: Hash of the entity to center the subgraph on
    :param n_hops: Number of hops (edge traversals) to include in subgraph
    :param edge_types: Set of edge types to traverse. If None, traverse all edges.
    :return: Subgraph centered on the specified entity
    """
    if edge_types is None:
        edge_types = {"mentions", "relationship", "contains"}

    # Get nodes within n_hops of entity
    nodes = {entity_hash}
    current_nodes = {entity_hash}

    for _ in range(n_hops):
        next_nodes = set()
        for node in current_nodes:
            # Get neighbors through specified edge types
            for _, nbr, edge_data in G.edges(node, data=True):
                if edge_data["type"] in edge_types:
                    next_nodes.add(nbr)
            for nbr, _, edge_data in G.in_edges(node, data=True):
                if edge_data["type"] in edge_types:
                    next_nodes.add(nbr)
        current_nodes = next_nodes - nodes
        nodes.update(next_nodes)

    # Return induced subgraph on selected nodes
    return G.subgraph(nodes).copy()


def get_chunk_context(
    G: nx.MultiDiGraph, entity_hash: str, n_hops: int = 2
) -> Dict[str, str]:
    """Get the text of chunks that provide context for an entity.

    :param G: Network graph
    :param entity_hash: Hash of entity to get context for
    :param n_hops: Number of hops to traverse when finding related chunks
    :return: Dictionary mapping chunk hashes to chunk text
    """
    # Get subgraph around entity
    subgraph = get_entity_subgraph(
        G, entity_hash, n_hops=n_hops, edge_types={"mentions", "contains"}
    )

    # Find chunks in subgraph
    chunks = {}
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        if node_data.get("type") == "chunk":
            chunks[node] = node_data["text"]

    return chunks
