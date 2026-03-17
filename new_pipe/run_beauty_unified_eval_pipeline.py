from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from dynamic_reasoning_ranking_agent import run_module3
    from item_profiler_agents import (
        GlobalItemDB,
        HistoryItemProfileInput,
        ItemProfileInput,
        Qwen3VLExtractor,
        UserHistoryLogDB,
    )
    from intent_dual_recall_agent import Qwen3RouterLLM
except ModuleNotFoundError:
    from new_pipe.dynamic_reasoning_ranking_agent import run_module3
    from new_pipe.item_profiler_agents import (
        GlobalItemDB,
        HistoryItemProfileInput,
        ItemProfileInput,
        Qwen3VLExtractor,
        UserHistoryLogDB,
    )
    from new_pipe.intent_dual_recall_agent import Qwen3RouterLLM


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
            if not line.strip():
                continue
            rec = _parse_meta_line(line)
            asin = str(rec.get("asin", "")).strip()
            if asin:
                out[asin] = rec
    return out


def _meta_category_paths(meta: Dict[str, Any]) -> List[List[str]]:
    categories = meta.get("categories", [])
    out: List[List[str]] = []
    if isinstance(categories, list):
        for cat_path in categories:
            if isinstance(cat_path, list):
                segs = [str(x).strip() for x in cat_path if str(x).strip()]
                if segs:
                    out.append(segs)
    return out


def _meta_category_text(meta: Dict[str, Any]) -> str:
    return " | ".join(" > ".join(x) for x in _meta_category_paths(meta))


def _item_sentence(meta: Dict[str, Any]) -> str:
    return (
        f"categories: {_meta_category_text(meta)}; "
        f"title: {str(meta.get('title', '') or '')}; "
        f"description: {str(meta.get('description', '') or '')}"
    ).strip()


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


def _route_query(
    query: str,
    category_catalog: List[str],
    enable_llm: bool,
    text_model: str,
) -> Dict[str, Any]:
    if not enable_llm:
        return {
            "selected_category_paths": [],
            "selected_item_types": [],
            "rewritten_query": query.strip(),
            "reasoning": "rule_based_rewrite_only",
        }

    router = Qwen3RouterLLM(model_name=text_model)
    try:
        routing = router.route(query=query, category_catalog=category_catalog, item_type_catalog=[])
        return {
            "selected_category_paths": routing.category_paths,
            "selected_item_types": routing.item_types,
            "rewritten_query": query.strip(),
            "reasoning": routing.reasoning,
        }
    except Exception as exc:
        print(f"[Agent3] LLM routing failed, fallback to rule-based. error={exc}")
        return {
            "selected_category_paths": [],
            "selected_item_types": [],
            "rewritten_query": query.strip(),
            "reasoning": f"llm_route_failed: {exc}",
        }


def _lightweight_profile(meta: Dict[str, Any], item_id: str) -> Dict[str, Any]:
    category_paths = _meta_category_paths(meta)
    return {
        "item_id": item_id,
        "title": str(meta.get("title", "") or ""),
        "taxonomy": {
            "item_type": category_paths[0][-1] if category_paths else "",
            "category_path": category_paths[0] if category_paths else [],
            "confidence": 0.7,
        },
        "text_tags": {
            "summary": str(meta.get("description", "") or ""),
            "price": meta.get("price", None),
        },
        "visual_tags": {},
        "hypotheses": ["lightweight_profile_without_vl"],
        "overall_confidence": 0.7,
    }


def _cleanup_torch_cache() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    prompt_name: str | None = None,
) -> np.ndarray:
    if torch is not None:
        with torch.inference_mode():
            return model.encode(
                texts,
                batch_size=batch_size,
                prompt_name=prompt_name,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
    return model.encode(
        texts,
        batch_size=batch_size,
        prompt_name=prompt_name,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def _build_item_embedding_cache(
    emb_model: SentenceTransformer,
    all_item_ids: List[str],
    meta_map: Dict[str, Dict[str, Any]],
    item_sentence_cache: Dict[str, str],
    emb_cache_path: Path,
    embed_batch_size: int,
    chunk_size: int,
    save_every_n: int,
) -> np.ndarray:
    total = len(all_item_ids)
    print(f"[Agent3] rebuilding item embedding cache for {total} items")
    all_emb_chunks: List[np.ndarray] = []
    processed = 0

    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        chunk_ids = all_item_ids[start:end]
        chunk_sentences: List[str] = []

        for iid in chunk_ids:
            sentence = item_sentence_cache.get(iid)
            if not sentence:
                sentence = _item_sentence(meta_map[iid])
                item_sentence_cache[iid] = sentence
            chunk_sentences.append(sentence)

        try:
            chunk_emb = _encode_texts(
                model=emb_model,
                texts=chunk_sentences,
                batch_size=embed_batch_size,
                prompt_name=None,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg and embed_batch_size > 1:
                smaller_bs = max(1, embed_batch_size // 2)
                print(
                    f"[Agent3] OOM at batch_size={embed_batch_size}, retry chunk {start}-{end} "
                    f"with batch_size={smaller_bs}"
                )
                chunk_emb = _encode_texts(
                    model=emb_model,
                    texts=chunk_sentences,
                    batch_size=smaller_bs,
                    prompt_name=None,
                )
            else:
                raise

        all_emb_chunks.append(chunk_emb.astype(np.float32, copy=False))
        processed = end
        print(f"[Agent3][embedding chunk] {processed}/{total} (chunk={start}-{end})")

        if processed % save_every_n == 0 or processed == total:
            partial_matrix = np.concatenate(all_emb_chunks, axis=0)
            emb_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                emb_cache_path,
                item_ids=np.array(all_item_ids[:processed]),
                item_embeddings=partial_matrix,
            )
            print(f"[Agent3][cache save] saved partial embeddings: {processed}/{total} -> {emb_cache_path}")

        _cleanup_torch_cache()

    final_emb = np.concatenate(all_emb_chunks, axis=0)
    emb_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        emb_cache_path,
        item_ids=np.array(all_item_ids),
        item_embeddings=final_emb,
    )
    print(f"[Agent3][cache save] final embedding cache saved: {emb_cache_path}")
    return final_emb


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def run(args: argparse.Namespace) -> Dict[str, Any]:
    query_df = pd.read_csv(args.query_csv, dtype={"id": str, "user_id": str})
    if args.max_users > 0:
        query_df = query_df.head(args.max_users)

    meta_map = load_filtered_meta(Path(args.filtered_meta_jsonl))
    if not meta_map:
        raise ValueError(f"No items loaded from filtered meta: {args.filtered_meta_jsonl}")

    all_item_ids = sorted(meta_map.keys())
    cache_dir = Path(args.cache_dir)
    emb_cache_path = cache_dir / "agent3_item_embedding_cache.npz"
    text_cache_path = cache_dir / "agent3_text_cache.json"

    text_cache = _safe_json_load(text_cache_path, {"items": {}, "queries": {}})
    item_sentence_cache: Dict[str, str] = text_cache.get("items", {})
    query_sentence_cache: Dict[str, str] = text_cache.get("queries", {})

    print(f"[Init] load embedding model: {args.embedding_model}")
    emb_model = SentenceTransformer(args.embedding_model)

    item_ids_cached: List[str] = []
    item_emb_matrix: np.ndarray | None = None
    if emb_cache_path.exists():
        npz = np.load(emb_cache_path, allow_pickle=True)
        item_ids_cached = [str(x) for x in npz["item_ids"].tolist()]
        item_emb_matrix = npz["item_embeddings"].astype(np.float32, copy=False)

    if item_emb_matrix is None or item_ids_cached != all_item_ids:
        item_emb_matrix = _build_item_embedding_cache(
            emb_model=emb_model,
            all_item_ids=all_item_ids,
            meta_map=meta_map,
            item_sentence_cache=item_sentence_cache,
            emb_cache_path=emb_cache_path,
            embed_batch_size=args.embed_batch_size,
            chunk_size=args.embed_chunk_size,
            save_every_n=args.embed_save_every,
        )

    item_emb_norm = _l2_normalize(item_emb_matrix)

    global_db = GlobalItemDB(args.global_db)
    history_db = UserHistoryLogDB(args.history_db)
    vl_extractor = Qwen3VLExtractor(model_name=args.vl_model) if args.enable_vl_profiling else None

    category_catalog = sorted({_meta_category_text(v) for v in meta_map.values() if _meta_category_text(v)})
    results: List[Dict[str, Any]] = []

    for row_idx, row in query_df.iterrows():
        user_id = str(row["user_id"])
        target_id = str(row["id"])
        query = str(row.get("new_query") or row.get("query") or "").strip()
        if not query:
            continue

        print(f"\n[UserLoop] {row_idx + 1}/{len(query_df)} user={user_id} target={target_id}")
        routed = _route_query(query, category_catalog, args.enable_llm_routing, args.text_model)

        q_sentence = _query_sentence(query, routed["selected_category_paths"], routed["rewritten_query"])
        query_sentence_cache[f"{user_id}::{q_sentence}"] = q_sentence

        q_emb = _encode_texts(
            model=emb_model,
            texts=[q_sentence],
            batch_size=1,
            prompt_name="query",
        ).astype(np.float32, copy=False)
        q_emb_norm = q_emb / np.clip(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12, None)
        sim_matrix = np.matmul(item_emb_norm, q_emb_norm[0])
        rank_indices = np.argsort(-sim_matrix)

        topk = args.topk
        top_ids = [all_item_ids[i] for i in rank_indices[:topk]]
        hit = target_id in top_ids
        if not hit:
            print(f"[Agent3] miss at k={topk}; retry k={args.fallback_topk}")
            topk = args.fallback_topk
            top_ids = [all_item_ids[i] for i in rank_indices[:topk]]
            hit = target_id in top_ids

        if not hit:
            print("[Agent3] recall failed. metric=0, skip Agent1/2/4/5")
            results.append({"user_id": user_id, "target_id": target_id, "hit": 0, "used_k": topk})
            continue

        print(f"[Agent3] recall hit at k={topk}; run Agent1/2")
        candidate_items: List[Dict[str, Any]] = []
        for i, iid in enumerate(top_ids, start=1):
            meta = meta_map[iid]
            profile = global_db.get_profile(iid)
            if profile is None:
                if vl_extractor is not None:
                    item_input = ItemProfileInput(
                        item_id=iid,
                        title=str(meta.get("title", "") or f"item_{iid}"),
                        detail_text=str(meta.get("description", "") or ""),
                        main_image=str(meta.get("imUrl", "") or ""),
                        detail_images=[],
                        price=str(meta.get("price", "") or ""),
                        category_hint=_meta_category_text(meta),
                    )
                    profile = vl_extractor.extract(
                        prompt=f"Profile item: {item_input.title}\n{item_input.detail_text}",
                        image_paths=[item_input.main_image],
                    )
                else:
                    profile = _lightweight_profile(meta, iid)
                global_db.upsert(iid, profile)

            candidate_items.append({"item_id": iid, "profile": profile})
            if i % 50 == 0 or i == len(top_ids):
                print(f"[Agent1] {i}/{len(top_ids)}")

        history_rows: List[Dict[str, Any]] = []
        history_ids = [x for x in str(row.get("remaining_interaction_string", "")).split("|") if x]
        for i, iid in enumerate(history_ids, start=1):
            meta = meta_map.get(iid)
            if meta is None:
                continue

            profile = global_db.get_profile(iid)
            if profile is None:
                if vl_extractor is not None:
                    item_input = HistoryItemProfileInput(
                        user_id=user_id,
                        item_id=iid,
                        title=str(meta.get("title", "") or f"item_{iid}"),
                        detail_text=str(meta.get("description", "") or ""),
                        main_image=str(meta.get("imUrl", "") or ""),
                        behavior="positive",
                        timestamp=None,
                    )
                    profile = vl_extractor.extract(
                        prompt=f"Profile item: {item_input.title}\n{item_input.detail_text}",
                        image_paths=[item_input.main_image],
                    )
                else:
                    profile = _lightweight_profile(meta, iid)
                global_db.upsert(iid, profile)

            if not history_db.exists(user_id=user_id, item_id=iid, behavior="positive", timestamp=None):
                history_db.insert(user_id=user_id, item_id=iid, behavior="positive", timestamp=None, profile=profile)

            history_rows.append(
                {
                    "user_id": user_id,
                    "item_id": iid,
                    "behavior": "positive",
                    "timestamp": None,
                    "profile": profile,
                }
            )
            if i % 20 == 0 or i == len(history_ids):
                print(f"[Agent2] {i}/{len(history_ids)}")

        agent3_output = {
            "query": q_sentence,
            "user_id": user_id,
            "routing": routed,
            "candidate_items": candidate_items,
            "query_relevant_history": history_rows,
        }

        ranked_first = ""
        if args.enable_agent45:
            module3_out = run_module3(
                intent_dual_recall_output=agent3_output,
                model_name=args.text_model,
                top_n=args.top_n,
                save_output=True,
                output_dir=args.output_dir,
            )
            ranked_first = module3_out.ranked_items[0]["item_id"] if module3_out.ranked_items else ""
        else:
            print("[Agent4/5] skipped by --disable-agent45")

        results.append(
            {
                "user_id": user_id,
                "target_id": target_id,
                "hit": 1,
                "used_k": topk,
                "top1": ranked_first,
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
    parser = argparse.ArgumentParser(description="可运行的 Beauty 统一评估流程：Agent3 -> Agent1/2 -> Agent4/5")
    parser.add_argument("--query-csv", default="data/amazon_beauty/query_data1.csv")
    parser.add_argument("--filtered-meta-jsonl", default="data/amazon_beauty/meta_Beauty.filtered.jsonl")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--embed-chunk-size", type=int, default=20000)
    parser.add_argument("--embed-save-every", type=int, default=20000)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--fallback-topk", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-users", type=int, default=0, help="仅跑前N条query，0表示全量")

    parser.add_argument("--cache-dir", default="processed/beauty_cache")
    parser.add_argument("--output-dir", default="processed/beauty_unified_outputs")
    parser.add_argument("--global-db", default="processed/beauty_global_item_features.db")
    parser.add_argument("--history-db", default="processed/beauty_user_history.db")

    parser.add_argument("--vl-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--text-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--enable-llm-routing", action="store_true", help="开启Qwen3文本路由；默认关闭走规则fallback")
    parser.add_argument("--enable-vl-profiling", action="store_true", help="开启Qwen3-VL画像；默认关闭走轻量画像")
    parser.add_argument("--disable-agent45", action="store_true", help="关闭Agent4/5")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.enable_agent45 = not bool(args.disable_agent45)
    run(args)
