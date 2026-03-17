from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from dynamic_reasoning_ranking_agent import run_module3
from item_profiler_agents import (
    HistoryItemProfileInput,
    ItemProfileInput,
    UserHistoryLogDB,
    GlobalItemDB,
    Qwen3VLExtractor,
)
from intent_dual_recall_agent import Qwen3RouterLLM


def _parse_meta_line(line: str) -> dict:
    t = line.strip()
    if not t:
        return {}
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return ast.literal_eval(t)


def load_filtered_meta(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = _parse_meta_line(line)
            asin = str(rec.get("asin", "")).strip()
            if asin:
                out[asin] = rec
    return out


def _meta_category_text(meta: Dict[str, Any]) -> str:
    categories = meta.get("categories", [])
    flat = []
    for cat_path in categories if isinstance(categories, list) else []:
        if isinstance(cat_path, list):
            flat.append(" > ".join(str(x).strip() for x in cat_path if str(x).strip()))
    return " | ".join(x for x in flat if x)


def _item_sentence(meta: Dict[str, Any]) -> str:
    cat = _meta_category_text(meta)
    title = str(meta.get("title", "") or "")
    desc = str(meta.get("description", "") or "")
    return f"categories: {cat}; title: {title}; description: {desc}".strip()


def _query_sentence(query: str, selected_categories: List[List[str]], rewritten: str) -> str:
    cats = " | ".join(" > ".join(seg for seg in c if seg) for c in selected_categories)
    return f"categories: {cats}; user_need: {rewritten or query}".strip()


def _safe_json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _rewrite_and_route_query(router: Qwen3RouterLLM, query: str, category_catalog: List[str]) -> Dict[str, Any]:
    # Reuse existing route for categories/item types and keep rewrite compatible fallback.
    routing = router.route(query=query, category_catalog=category_catalog, item_type_catalog=[])
    rewritten = query.strip()
    return {
        "selected_category_paths": routing.category_paths,
        "selected_item_types": routing.item_types,
        "rewritten_query": rewritten,
        "reasoning": routing.reasoning,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    query_df = pd.read_csv(args.query_csv, dtype={"id": str, "user_id": str})
    meta_map = load_filtered_meta(Path(args.filtered_meta_jsonl))
    all_item_ids = sorted(meta_map.keys())

    cache_dir = Path(args.cache_dir)
    emb_cache_path = cache_dir / "agent3_embedding_cache.npz"
    text_cache_path = cache_dir / "agent3_text_cache.json"

    text_cache = _safe_json_load(text_cache_path, {"items": {}, "queries": {}})
    item_sentence_cache: Dict[str, str] = text_cache.get("items", {})
    query_sentence_cache: Dict[str, str] = text_cache.get("queries", {})

    print(f"[Init] loading embedding model: {args.embedding_model}")
    emb_model = SentenceTransformer(args.embedding_model)

    item_ids_cached = []
    item_emb_matrix = None
    if emb_cache_path.exists():
        npz = np.load(emb_cache_path, allow_pickle=True)
        item_ids_cached = list(npz["item_ids"])
        item_emb_matrix = npz["item_embeddings"]

    if item_emb_matrix is None or set(item_ids_cached) != set(all_item_ids):
        print(f"[Agent3] rebuilding item embedding cache for {len(all_item_ids)} items")
        item_sentences = []
        for idx, iid in enumerate(all_item_ids, start=1):
            s = item_sentence_cache.get(iid)
            if not s:
                s = _item_sentence(meta_map[iid])
                item_sentence_cache[iid] = s
            item_sentences.append(s)
            if idx % 2000 == 0 or idx == len(all_item_ids):
                print(f"[Agent3][item sentence] {idx}/{len(all_item_ids)}")
        item_emb_matrix = emb_model.encode(item_sentences, batch_size=args.embed_batch_size)
        emb_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(emb_cache_path, item_ids=np.array(all_item_ids), item_embeddings=item_emb_matrix)

    global_db = GlobalItemDB(args.global_db)
    history_db = UserHistoryLogDB(args.history_db)
    extractor = Qwen3VLExtractor(model_name=args.vl_model)
    router = Qwen3RouterLLM(model_name=args.text_model)

    results = []
    for row_idx, row in query_df.iterrows():
        user_id = str(row["user_id"])
        target_id = str(row["id"])
        query = str(row.get("new_query") or row.get("query") or "").strip()
        if not query:
            continue

        print(f"\n[UserLoop] row={row_idx + 1}/{len(query_df)} user={user_id} target={target_id}")

        category_catalog = sorted({_meta_category_text(v) for v in meta_map.values() if _meta_category_text(v)})
        routed = _rewrite_and_route_query(router, query, category_catalog)
        q_sentence = _query_sentence(query, routed["selected_category_paths"], routed["rewritten_query"])
        query_key = f"{user_id}::{q_sentence}"
        query_sentence_cache[query_key] = q_sentence

        q_emb = emb_model.encode([q_sentence], prompt_name="query")
        sim = emb_model.similarity(q_emb, item_emb_matrix).detach().cpu().numpy().reshape(-1)
        rank_indices = np.argsort(-sim)

        topk = args.topk
        top_ids = [all_item_ids[i] for i in rank_indices[:topk]]
        hit = target_id in top_ids
        if not hit:
            print(f"[Agent3] target miss at k={topk}, try k={args.fallback_topk}")
            topk = args.fallback_topk
            top_ids = [all_item_ids[i] for i in rank_indices[:topk]]
            hit = target_id in top_ids

        if not hit:
            print("[Agent3] recall failed after fallback, skip agent1/2/4/5 and mark metric=0")
            results.append({"user_id": user_id, "target_id": target_id, "hit": 0, "used_k": topk})
            continue

        print(f"[Agent3] hit target at k={topk}, continue profiling")
        candidate_items = []
        for i, iid in enumerate(top_ids, start=1):
            meta = meta_map[iid]
            profile = global_db.get_profile(iid)
            source = "global_db_reused"
            if profile is None:
                item_input = ItemProfileInput(
                    item_id=iid,
                    title=str(meta.get("title", "") or f"item_{iid}"),
                    detail_text=str(meta.get("description", "") or ""),
                    main_image=str(meta.get("imUrl", "") or ""),
                    detail_images=[],
                    price=str(meta.get("price", "") or ""),
                    category_hint=_meta_category_text(meta),
                )
                profile = extractor.extract(prompt=f"Profile item: {item_input.title}\n{item_input.detail_text}", image_paths=[item_input.main_image])
                global_db.upsert(iid, profile)
                source = "newly_profiled"
            candidate_items.append({"item_id": iid, "profile": profile, "profile_source": source})
            if i % 50 == 0 or i == len(top_ids):
                print(f"[Agent1] profiled/reused {i}/{len(top_ids)}")

        history_rows = []
        history_ids = [x for x in str(row.get("remaining_interaction_string", "")).split("|") if x]
        for i, iid in enumerate(history_ids, start=1):
            meta = meta_map.get(iid)
            if meta is None:
                continue
            profile = global_db.get_profile(iid)
            if profile is None:
                item_input = HistoryItemProfileInput(
                    user_id=user_id,
                    item_id=iid,
                    title=str(meta.get("title", "") or f"item_{iid}"),
                    detail_text=str(meta.get("description", "") or ""),
                    main_image=str(meta.get("imUrl", "") or ""),
                    behavior="positive",
                    timestamp=None,
                )
                profile = extractor.extract(prompt=f"Profile item: {item_input.title}\n{item_input.detail_text}", image_paths=[item_input.main_image])
                global_db.upsert(iid, profile)
            if not history_db.exists(user_id=user_id, item_id=iid, behavior="positive", timestamp=None):
                history_db.insert(user_id=user_id, item_id=iid, behavior="positive", timestamp=None, profile=profile)
            history_rows.append({"user_id": user_id, "item_id": iid, "behavior": "positive", "timestamp": None, "profile": profile})
            if i % 20 == 0 or i == len(history_ids):
                print(f"[Agent2] history profiled/reused {i}/{len(history_ids)}")

        agent3_output = {
            "query": q_sentence,
            "user_id": user_id,
            "routing": routed,
            "candidate_items": candidate_items,
            "query_relevant_history": history_rows,
        }
        module3_out = run_module3(
            intent_dual_recall_output=agent3_output,
            model_name=args.text_model,
            top_n=args.top_n,
            save_output=True,
            output_dir=args.output_dir,
        )
        results.append(
            {
                "user_id": user_id,
                "target_id": target_id,
                "hit": 1,
                "used_k": topk,
                "topn_first_item": module3_out.ranked_items[0]["item_id"] if module3_out.ranked_items else "",
            }
        )

    text_cache["items"] = item_sentence_cache
    text_cache["queries"] = query_sentence_cache
    _save_json(text_cache_path, text_cache)
    _save_json(Path(args.output_dir) / "unified_eval_results.json", results)

    hit_rate = float(np.mean([r["hit"] for r in results])) if results else 0.0
    summary = {"rows": len(results), "hit_rate": hit_rate, "output_dir": args.output_dir}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified beauty user-wise pipeline: Agent3 -> Agent1/2 -> Agent4/5")
    parser.add_argument("--query-csv", default="data/amazon_beauty/query_data1.csv")
    parser.add_argument("--filtered-meta-jsonl", default="data/amazon_beauty/meta_Beauty.filtered.jsonl")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--fallback-topk", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--cache-dir", default="processed/beauty_cache")
    parser.add_argument("--output-dir", default="processed/beauty_unified_outputs")
    parser.add_argument("--global-db", default="processed/beauty_global_item_features.db")
    parser.add_argument("--history-db", default="processed/beauty_user_history.db")
    parser.add_argument("--vl-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--text-model", default="Qwen/Qwen3-8B")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
