# rag/retriever.py
from typing import List, Dict
import json
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever, BaseRetriever


class EntityRelationRetriever(BaseRetriever):
    """Retriever for entities + relations with KG expansion (ID-based)."""

    def __init__(self, entities_index, relations_index, kg_store, top_k=5, hops=1):
        super().__init__()
        self.entity_retriever = VectorIndexRetriever(entities_index, similarity_top_k=top_k)
        self.relation_retriever = VectorIndexRetriever(relations_index, similarity_top_k=top_k)
        self.kg_store = kg_store
        self.top_k = top_k
        self.hops = hops

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Step 1: retrieve directly from entity & relation VDBs
        entity_hits: List[NodeWithScore] = self.entity_retriever.retrieve(query_bundle)
        relation_hits: List[NodeWithScore] = self.relation_retriever.retrieve(query_bundle)

        # Step 2: expand entities via KG (multi-hop neighbors, ID-based)
        expanded_nodes = []
        for d in entity_hits:
            entity_id = d.node_id  # Pinecone stores our deterministic IDs
            if not entity_id:
                continue
            neighbors = self.kg_store.multi_hop(entity_id, hops=self.hops)
            for n in neighbors:
                meta = self.kg_store.graph.nodes[n]  # node attributes
                expanded_nodes.append(
                    NodeWithScore(
                        node_id=n,
                        text=meta.get("name", ""),   # keep human-readable name
                        score=0.5,                   # weaker score than direct hits
                        metadata=meta
                    )
                )

        # Step 3: return combined set
        return entity_hits + relation_hits + expanded_nodes


def extract_query_keywords(llm, query: str) -> Dict[str, List[str]]:
    """
    Ask the LLM to split the query into low-level and high-level keywords.
    """
    prompt = f"""
    Split the following query into two sets of keywords:
    1. low-level (entities, direct nouns)
    2. high-level (relations, verbs, broad themes)

    Return JSON in this format:
    {{
      "low_level": ["entity1", "entity2"],
      "high_level": ["relation1", "theme1"]
    }}

    Query: {query}
    """
    resp = llm.complete(prompt)
    try:
        return json.loads(resp.text.strip())
    except Exception:
        return {"low_level": [], "high_level": []}


def build_light_rag_retriever(chunks_index, entities_index, relations_index, kg_store, llm, top_k=5, mode="auto"):
    """
    Build LightRAG retriever with multiple modes:
      - local: chunk retriever only
      - global: entity/relation + KG retriever
      - hybrid: fuse both
      - auto: use LLM keyword extraction to decide dynamically
    """
    chunk_retriever = VectorIndexRetriever(chunks_index, similarity_top_k=top_k)
    global_retriever = EntityRelationRetriever(
        entities_index, relations_index, kg_store, top_k=top_k, hops=1
    )

    if mode == "local":
        return chunk_retriever

    if mode == "global":
        return global_retriever

    if mode == "hybrid":
        return QueryFusionRetriever(
            retrievers=[chunk_retriever, global_retriever],
            similarity_top_k=top_k,
            mode="reciprocal_rerank"
        )

    if mode == "auto":
        class AutoRetriever(BaseRetriever):
            def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
                keywords = extract_query_keywords(llm, query_bundle.query_str)
                if keywords["high_level"]:
                    # query has global aspects → combine local + global
                    low_hits = chunk_retriever.retrieve(query_bundle)
                    high_hits = global_retriever.retrieve(query_bundle)
                    return _dedupe(low_hits + high_hits)  # might add reranking later
                else:
                    # default to local-only retrieval
                    return chunk_retriever.retrieve(query_bundle)

        def _dedupe(nodes: List[NodeWithScore]) -> List[NodeWithScore]:
            seen, unique = set(), []
            for n in nodes:
                if n.node_id not in seen:
                    seen.add(n.node_id)
                    unique.append(n)
            return unique

        return AutoRetriever()

    raise ValueError(f"Unknown retriever mode: {mode}")
