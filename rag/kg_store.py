# rag/kg_store.py
import os
import pickle
import networkx as nx

GRAPH_PATH = "./graph_storage/kg_graph.pkl"

class KGStore:
    def __init__(self):
        if os.path.exists(GRAPH_PATH):
            with open(GRAPH_PATH, "rb") as f:
                self.graph = pickle.load(f)
            print("Loaded existing KG from disk.")
        else:
            self.graph = nx.DiGraph()
            print("Initialized new KG.")

    def add_entities_relations(self, entities, relations):
        """
        Add entities and relations into the NetworkX graph.
        Uses entity_id / relation_id from metadata (not recomputed).
        """
        # --- Entities ---
        for e in entities:
            eid = e.get("entity_id")
            if not eid:
                continue

            if not self.graph.has_node(eid):
                self.graph.add_node(
                    eid,
                    name=e.get("name"),
                    type=e.get("type", "Other"),
                    chunk_ids=e.get("chunk_ids", [])
                )
            else:
                # merge provenance
                existing_chunks = self.graph.nodes[eid].get("chunk_ids", [])
                merged_chunks = list(dict.fromkeys(existing_chunks + e.get("chunk_ids", [])))
                self.graph.nodes[eid]["chunk_ids"] = merged_chunks

        # --- Relations ---
        for rel in relations:
            rid = rel.get("relation_id")
            src_id = rel.get("source_id")
            tgt_id = rel.get("target_id")

            if not (rid and src_id and tgt_id):
                continue

            if not self.graph.has_node(src_id):
                self.graph.add_node(src_id, name=rel.get("source"), type="Other")
            if not self.graph.has_node(tgt_id):
                self.graph.add_node(tgt_id, name=rel.get("target"), type="Other")

            # add or update edge
            if self.graph.has_edge(src_id, tgt_id):
                # merge chunk_ids into edge provenance if exists
                existing_chunks = self.graph[src_id][tgt_id].get("chunk_ids", [])
                merged_chunks = list(dict.fromkeys(existing_chunks + rel.get("chunk_ids", [])))
                self.graph[src_id][tgt_id]["chunk_ids"] = merged_chunks
            else:
                edge_attrs = dict(rel)
                # don't keep redundant source/target text
                edge_attrs.pop("source", None)
                edge_attrs.pop("target", None)
                self.graph.add_edge(src_id, tgt_id, **edge_attrs)

    def neighbors(self, entity_id: str):
        """Return immediate neighbors of an entity (by ID)."""
        return list(self.graph.neighbors(entity_id))

    def multi_hop(self, entity_id: str, hops=2):
        """Expand entity up to N hops away (by ID)."""
        results = set([entity_id])
        frontier = [entity_id]
        for _ in range(hops):
            next_frontier = []
            for e in frontier:
                nbrs = list(self.graph.neighbors(e))
                results.update(nbrs)
                next_frontier.extend(nbrs)
            frontier = next_frontier
        return list(results)

    def save(self):
        os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(self.graph, f)
        print("KG saved to disk.")
