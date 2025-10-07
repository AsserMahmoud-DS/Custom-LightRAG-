# rag/entity_extractor.py
import os
import json
import re
import logging
from typing import List, Dict, Tuple
from pinecone import Pinecone, ServerlessSpec
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rag.utils import make_id

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

ENTITIES_INDEX = "lightrag-entities"
RELATIONS_INDEX = "lightrag-relations"

# ---------------- Helpers ----------------
def _safe_json_parse(text: str) -> dict:
    """Try parsing JSON safely, fallback to empty structure."""
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"(\{.*\})", text, re.S)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
    return {"entities": [], "relations": []}

def _extract_metadata(v):
    """Handle Pinecone Vector object vs dict safely."""
    if not v:
        return {}
    if isinstance(v, dict):
        return v.get("metadata", {}) or {}
    if hasattr(v, "metadata"):  # Pinecone Vector
        return v.metadata or {}
    return {}


def _dedupe_entities(entities: List[Dict]) -> List[Dict]:
    """Deduplicate entities by case-insensitive (name + type)."""
    seen, unique = set(), []
    for e in entities:
        if not isinstance(e, dict):
            continue
        name = e.get("name", "").strip()
        etype = e.get("type", "Other").strip()
        if not name:
            continue
        key = f"{name.lower()}||{etype.lower()}"
        if key not in seen:
            seen.add(key)
            unique.append({"name": name, "type": etype})
    return unique


def _dedupe_relations(relations: List[Dict]) -> List[Dict]:
    """Deduplicate relations by (source, relation, target)."""
    seen, unique = set(), []
    for r in relations:
        if not isinstance(r, dict):
            continue
        src = r.get("source", "").strip().lower()
        tgt = r.get("target", "").strip().lower()
        rel = r.get("relation", "").strip().lower()
        if not (src and tgt and rel):
            continue
        key = f"{src}||{rel}||{tgt}"
        if key not in seen:
            seen.add(key)
            unique.append({
                "source": r.get("source").strip(),
                "target": r.get("target").strip(),
                "relation": r.get("relation").strip(),
                **{k: v for k, v in r.items() if k not in ("source", "target", "relation")}
            })
    return unique


# ---------------- Core ----------------
def extract_entities_and_relations(llm, text: str) -> Dict[str, List[Dict]]:
    """
    Extract entities + relations with an LLM, normalize + dedupe.
    Returns: {"entities": [...], "relations": [...]}
    """
    prompt = f"""
    Extract entities and their relationships from the following text. 
    Return JSON with format:
    {{
        "entities": [
            {{"name": "Entity1", "type": "Person"}},
            {{"name": "Entity2", "type": "Organization"}}
        ],
        "relations": [
            {{"source": "Entity1", "target": "Entity2", "relation": "works_at"}}
        ]
    }}

    Valid entity types: Person, Organization, Location, Event, Concept, Other.

    Text:
    {text}
    """
    response = llm.complete(prompt)
    data = _safe_json_parse(response.text.strip())

    # normalize + dedupe entities
    raw_entities = data.get("entities", [])
    normalized_entities = []
    for e in raw_entities:
        if isinstance(e, dict) and e.get("name"):
            normalized_entities.append({"name": e["name"].strip(),
                                        "type": e.get("type", "Other").strip()})
        elif isinstance(e, str) and e.strip():
            normalized_entities.append({"name": e.strip(), "type": "Other"})
    entities = _dedupe_entities(normalized_entities)

    # normalize + dedupe relations
    raw_relations = data.get("relations", [])
    normalized_relations = []
    for r in raw_relations:
        if isinstance(r, dict) and r.get("source") and r.get("target") and r.get("relation"):
            normalized_relations.append({
                "source": r["source"].strip(),
                "target": r["target"].strip(),
                "relation": r["relation"].strip(),
                **{k: v for k, v in r.items() if k not in ("source", "target", "relation")}
            })
    relations = _dedupe_relations(normalized_relations)

    return {"entities": entities, "relations": relations}

def store_entities_and_relations(
    entities: List[Dict],
    relations: List[Dict],
    chunk_id: str,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Tuple[List[Dict], List[Dict]]:
    """
    Incrementally upsert entities + relations into Pinecone.
    - Skips re-upsert if chunk_id already exists
    - Returns enriched entities/relations consistently with _node_content
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    # ensure indexes exist
    for idx in (ENTITIES_INDEX, RELATIONS_INDEX):
        if idx not in pc.list_indexes().names():
            pc.create_index(
                idx,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    entity_index = pc.Index(ENTITIES_INDEX)
    relation_index = pc.Index(RELATIONS_INDEX)

    # -------- Entities --------
    entity_ids = [make_id(e["name"]) for e in entities]
    existing = getattr(entity_index.fetch(ids=entity_ids), "vectors", {}) or {}

    entity_upserts, enriched_entities = [], []
    for e in entities:
        eid = make_id(e["name"])
        prev_meta = _extract_metadata(existing.get(eid, {}))
        prev_chunks = prev_meta.get("chunk_ids", [])

        # if already processed → just return enriched view
        if chunk_id in prev_chunks:
            enriched_entities.append({
                "entity_id": eid,
                "name": prev_meta.get("name", e["name"]),
                "type": prev_meta.get("type", e["type"]),
                "chunk_ids": prev_chunks,
                "_node_content": prev_meta.get("_node_content", f"{e['type']}: {e['name']}")
            })
            continue

        # new or updated
        merged_chunks = list(dict.fromkeys(prev_chunks + [chunk_id]))
        enriched = {
            "entity_id": eid,
            "name": e["name"],
            "type": e["type"],
            "chunk_ids": merged_chunks,
            "_node_content": f"{e['type']}: {e['name']}",
        }
        enriched_entities.append(enriched)

        emb = embed_model.get_text_embedding(enriched["_node_content"])
        entity_upserts.append({"id": eid, "values": emb, "metadata": enriched})

    if entity_upserts:
        entity_index.upsert(vectors=entity_upserts)

    # -------- Relations --------
    relation_ids = [make_id(f"{r['source']}|{r['relation']}|{r['target']}") for r in relations]
    existing = getattr(relation_index.fetch(ids=relation_ids), "vectors", {}) or {}

    relation_upserts, enriched_relations = [], []
    for r in relations:
        rid = make_id(f"{r['source']}|{r['relation']}|{r['target']}")
        src_id, tgt_id = make_id(r["source"]), make_id(r["target"])

        prev_meta = _extract_metadata(existing.get(rid, {}))
        prev_chunks = prev_meta.get("chunk_ids", [])

        if chunk_id in prev_chunks:
            enriched_relations.append({
                "relation_id": rid,
                "source_id": prev_meta.get("source_id", src_id),
                "target_id": prev_meta.get("target_id", tgt_id),
                "source": prev_meta.get("source", r["source"]),
                "target": prev_meta.get("target", r["target"]),
                "relation": prev_meta.get("relation", r["relation"]),
                "chunk_ids": prev_chunks,
                "_node_content": prev_meta.get("_node_content", f"{r['source']} -[{r['relation']}]-> {r['target']}")
            })
            continue

        merged_chunks = list(dict.fromkeys(prev_chunks + [chunk_id]))
        enriched = {
            "relation_id": rid,
            "source_id": src_id,
            "target_id": tgt_id,
            "source": r["source"],
            "target": r["target"],
            "relation": r["relation"],
            "chunk_ids": merged_chunks,
            "_node_content": f"{r['source']} -[{r['relation']}]-> {r['target']}",
        }
        enriched_relations.append(enriched)

        emb = embed_model.get_text_embedding(enriched["_node_content"])
        relation_upserts.append({"id": rid, "values": emb, "metadata": enriched})

    if relation_upserts:
        relation_index.upsert(vectors=relation_upserts)

    _log.info(
        "Upserted %d entities and %d relations (chunk_id=%s).",
        len(entity_upserts), len(relation_upserts), chunk_id
    )
    return enriched_entities, enriched_relations
