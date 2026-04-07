from __future__ import annotations

import argparse
import ast
import gc
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from adaptive_pipe.dynamic_reasoning_ranking_agent import run_module3
    from adaptive_pipe.image_prefetch import prefetch_item_images
    from adaptive_pipe.item_profiler_agents import (
        GlobalItemDB,
        HistoryItemProfileInput,
        ItemProfileInput,
        Qwen3VLExtractor,
        UserHistoryLogDB,
    )
    from adaptive_pipe.intent_dual_recall_agent import Qwen3RouterLLM
except ModuleNotFoundError:
    from dynamic_reasoning_ranking_agent import run_module3
    from image_prefetch import prefetch_item_images
    from item_profiler_agents import (
        GlobalItemDB,
        HistoryItemProfileInput,
        ItemProfileInput,
        Qwen3VLExtractor,
        UserHistoryLogDB,
    )
    from intent_dual_recall_agent import Qwen3RouterLLM

EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "for", "with", "of", "in", "on", "at", "from", "by",
    "is", "are", "be", "am", "i", "me", "my", "you", "your", "we", "our", "this", "that", "it",
    "want", "need", "looking", "interested", "find", "recommend", "please", "can", "could", "would",
}


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


def _safe_meta_image(meta: Dict[str, Any]) -> str:
    cand = str(meta.get("imUrl", "") or "").strip()
    return cand


def _build_qwen3vl_item_input(
    meta: Dict[str, Any],
    image_url_to_local: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"text": _item_sentence(meta)}
    image = _safe_meta_image(meta)
    if image and image_url_to_local:
        image = image_url_to_local.get(image, image)
    if image:
        payload["image"] = image
    return payload


def _tensor_to_float32_numpy(emb: Any) -> np.ndarray:
    if torch is not None and isinstance(emb, torch.Tensor):
        return emb.detach().to(torch.float32).cpu().numpy()
    return np.asarray(emb, dtype=np.float32)


def _safe_json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _route_query(query: str, category_catalog: List[str], enable_llm: bool, text_model: str) -> Dict[str, Any]:
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
        "text_tags": {"summary": str(meta.get("description", "") or ""), "price": meta.get("price", None)},
        "visual_tags": {},
        "hypotheses": ["lightweight_profile_without_vl"],
        "overall_confidence": 0.7,
    }


def _cleanup_torch_cache() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _is_oom_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int, prompt_name: str | None = None) -> np.ndarray:
    if torch is not None:
        with torch.inference_mode():
            return model.encode(texts, batch_size=batch_size, prompt_name=prompt_name, convert_to_numpy=True, show_progress_bar=False)
    return model.encode(texts, batch_size=batch_size, prompt_name=prompt_name, convert_to_numpy=True, show_progress_bar=False)


def _move_sentence_transformer_to_device(model: SentenceTransformer, device: str) -> None:
    if hasattr(model, "to"):
        model.to(device)


def _encode_texts_with_adaptive_fallback(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    prompt_name: str | None,
    device: str,
) -> Tuple[np.ndarray, int, str]:
    current_batch_size = max(1, int(batch_size))
    active_device = device

    while True:
        try:
            chunk_emb = _encode_texts(model, texts, current_batch_size, prompt_name=prompt_name)
            return chunk_emb, current_batch_size, active_device
        except RuntimeError as exc:
            if not _is_oom_error(exc):
                raise

            _cleanup_torch_cache()
            if active_device != "cpu" and current_batch_size == 1:
                print("[Agent3] GPU OOM persists at batch_size=1, switch embedding encode to CPU for remaining chunks")
                _move_sentence_transformer_to_device(model, "cpu")
                active_device = "cpu"
                current_batch_size = min(8, max(1, int(batch_size)))
                continue

            if current_batch_size > 1:
                next_batch_size = max(1, current_batch_size // 2)
                print(
                    f"[Agent3] OOM at batch_size={current_batch_size} on {active_device}, "
                    f"retry with batch_size={next_batch_size}"
                )
                current_batch_size = next_batch_size
                continue

            raise


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
    emb_cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_emb_path = emb_cache_path.with_suffix(".tmp.npy")
    if temp_emb_path.exists():
        temp_emb_path.unlink()

    item_emb_memmap = None
    active_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if active_device == "cpu":
        _move_sentence_transformer_to_device(emb_model, "cpu")
    processed = 0

    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        chunk_sentences = []
        for iid in all_item_ids[start:end]:
            sentence = item_sentence_cache.get(iid)
            if not sentence:
                sentence = _item_sentence(meta_map[iid])
                item_sentence_cache[iid] = sentence
            chunk_sentences.append(sentence)

        chunk_emb, used_batch_size, new_device = _encode_texts_with_adaptive_fallback(
            emb_model,
            chunk_sentences,
            embed_batch_size,
            prompt_name=None,
            device=active_device,
        )
        if new_device != active_device:
            active_device = new_device

        chunk_emb = chunk_emb.astype(np.float32, copy=False)
        if item_emb_memmap is None:
            item_emb_memmap = np.lib.format.open_memmap(
                temp_emb_path,
                mode="w+",
                dtype=np.float32,
                shape=(total, chunk_emb.shape[1]),
            )
        item_emb_memmap[start:end] = chunk_emb
        item_emb_memmap.flush()
        processed = end
        print(
            f"[Agent3][embedding chunk] {processed}/{total} "
            f"(chunk={start}-{end}, batch_size={used_batch_size}, device={active_device})"
        )

        if (processed % save_every_n == 0 or processed == total) and item_emb_memmap is not None:
            np.savez_compressed(
                emb_cache_path,
                item_ids=np.array(all_item_ids[:processed]),
                item_embeddings=item_emb_memmap[:processed],
            )
            print(f"[Agent3][cache save] {processed}/{total} -> {emb_cache_path}")

        del chunk_emb
        _cleanup_torch_cache()

    if item_emb_memmap is None:
        raise ValueError("Failed to build item embedding cache: no chunks were encoded")

    final_emb = np.array(item_emb_memmap, dtype=np.float32, copy=True)
    np.savez_compressed(emb_cache_path, item_ids=np.array(all_item_ids), item_embeddings=final_emb)
    del item_emb_memmap
    if os.path.exists(temp_emb_path):
        os.remove(temp_emb_path)
    return final_emb


def _build_qwen3vl_item_embedding_cache(
    qwen3vl_model: Any,
    all_item_ids: List[str],
    meta_map: Dict[str, Dict[str, Any]],
    emb_cache_path: Path,
    chunk_size: int,
    save_every_items: int,
    image_url_to_local: Dict[str, str] | None = None,
) -> np.ndarray:
    total = len(all_item_ids)
    print(f"[Agent3][Qwen3VL] rebuilding multimodal embedding cache for {total} items")
    emb_cache_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = emb_cache_path.parent / f"{emb_cache_path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    pending_ids: List[str] = []
    pending_emb_chunks: List[np.ndarray] = []
    saved_part_paths: List[Path] = []
    processed = 0
    part_idx = 0
    resume_offset = 0

    existing_parts = sorted(parts_dir.glob("part_*.npz"))
    if existing_parts:
        print(f"[Agent3][Qwen3VL] found existing part files: {len(existing_parts)}")
        resume_ok = True
        for part_path in existing_parts:
            m = re.search(r"part_(\d+)\.npz$", part_path.name)
            if m:
                part_idx = max(part_idx, int(m.group(1)))
            part_npz = np.load(part_path, allow_pickle=True)
            part_ids = [str(x) for x in part_npz["item_ids"].tolist()]
            expected_ids = all_item_ids[resume_offset : resume_offset + len(part_ids)]
            if part_ids != expected_ids:
                print(
                    f"[Agent3][Qwen3VL][resume] mismatch at {part_path.name}; "
                    "clear old parts and rebuild from scratch."
                )
                resume_ok = False
                break
            saved_part_paths.append(part_path)
            resume_offset += len(part_ids)

        if not resume_ok:
            for old_part in existing_parts:
                old_part.unlink()
            saved_part_paths = []
            part_idx = 0
            resume_offset = 0
        elif resume_offset > 0:
            processed = resume_offset
            print(
                f"[Agent3][Qwen3VL][resume] skip processed rows={resume_offset}, "
                f"continue from index={resume_offset}."
            )

    def _flush_pending() -> None:
        nonlocal part_idx, pending_ids, pending_emb_chunks
        if not pending_ids:
            return
        part_idx += 1
        part_path = parts_dir / f"part_{part_idx:06d}.npz"
        part_emb = np.concatenate(pending_emb_chunks, axis=0)
        np.savez_compressed(part_path, item_ids=np.array(pending_ids), item_embeddings=part_emb)
        saved_part_paths.append(part_path)
        print(f"[Agent3][Qwen3VL][part save] {part_path.name} rows={len(pending_ids)}")
        pending_ids = []
        pending_emb_chunks = []
        _cleanup_torch_cache()

    for start in range(resume_offset, total, chunk_size):
        end = min(total, start + chunk_size)
        chunk_item_ids = all_item_ids[start:end]
        chunk_inputs = [_build_qwen3vl_item_input(meta_map[iid], image_url_to_local=image_url_to_local) for iid in chunk_item_ids]
        chunk_emb = _tensor_to_float32_numpy(qwen3vl_model.process(chunk_inputs))
        pending_emb_chunks.append(chunk_emb)
        pending_ids.extend(chunk_item_ids)
        processed = end
        print(f"[Agent3][Qwen3VL][embedding chunk] {processed}/{total} (chunk={start}-{end})")
        del chunk_inputs
        del chunk_emb
        if len(pending_ids) >= max(1, int(save_every_items)):
            _flush_pending()
        else:
            _cleanup_torch_cache()

    _flush_pending()
    if not saved_part_paths:
        raise ValueError("Failed to build Qwen3-VL embedding cache: no parts saved.")

    all_part_ids: List[str] = []
    all_parts: List[np.ndarray] = []
    for p in saved_part_paths:
        part_npz = np.load(p, allow_pickle=True)
        part_ids = [str(x) for x in part_npz["item_ids"].tolist()]
        part_emb = part_npz["item_embeddings"].astype(np.float32, copy=False)
        all_part_ids.extend(part_ids)
        all_parts.append(part_emb)
    final_emb = np.concatenate(all_parts, axis=0)
    np.savez_compressed(emb_cache_path, item_ids=np.array(all_part_ids), item_embeddings=final_emb)
    for p in saved_part_paths:
        p.unlink()
    if parts_dir.exists() and not any(parts_dir.iterdir()):
        parts_dir.rmdir()
    print(f"[Agent3][Qwen3VL][cache save] {total}/{total} -> {emb_cache_path}")
    return final_emb


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def _repair_qwen3vl_cache_for_missing_ids(
    qwen3vl_model: Any,
    text_emb_model: SentenceTransformer,
    cache_path: Path,
    all_item_ids: List[str],
    cached_ids: List[str],
    cached_matrix: np.ndarray,
    meta_map: Dict[str, Dict[str, Any]],
    image_url_to_local: Dict[str, str],
    chunk_size: int,
) -> Tuple[List[str], np.ndarray]:
    aligned = min(len(cached_ids), int(cached_matrix.shape[0]))
    cached_ids = cached_ids[:aligned]
    cached_matrix = cached_matrix[:aligned]
    cached_set = set(cached_ids)
    missing_ids = [iid for iid in all_item_ids if iid not in cached_set]
    if not missing_ids:
        return cached_ids, cached_matrix

    print(f"[Agent3][Qwen3VL][repair] missing_ids={len(missing_ids)}; embedding missing only.")
    repaired_missing_map: Dict[str, np.ndarray] = {}
    target_dim = int(cached_matrix.shape[1]) if cached_matrix.ndim == 2 and cached_matrix.shape[1] > 0 else 0
    step = max(1, int(chunk_size))
    for start in range(0, len(missing_ids), step):
        end = min(len(missing_ids), start + step)
        sub_ids = missing_ids[start:end]
        sub_inputs = [_build_qwen3vl_item_input(meta_map[iid], image_url_to_local=image_url_to_local) for iid in sub_ids]
        sub_emb = _tensor_to_float32_numpy(qwen3vl_model.process(sub_inputs))
        if sub_emb.ndim != 2:
            sub_emb = np.atleast_2d(sub_emb)
        if target_dim <= 0 and sub_emb.ndim == 2 and sub_emb.shape[1] > 0:
            target_dim = int(sub_emb.shape[1])
        produced = int(sub_emb.shape[0])
        expected = len(sub_ids)
        if produced != expected:
            print(
                f"[Agent3][Qwen3VL][repair] chunk size mismatch: expected={expected}, produced={produced}; "
                "only aligned prefix will be used."
            )
        use_n = min(expected, produced)
        for iid, row in zip(sub_ids[:use_n], sub_emb[:use_n]):
            repaired_missing_map[iid] = np.asarray(row, dtype=np.float32)

        no_image_ids = [iid for iid in sub_ids if not _safe_meta_image(meta_map.get(iid, {}))]
        if no_image_ids:
            text_batch = [_item_sentence(meta_map[iid]) for iid in no_image_ids]
            text_emb = _encode_texts(
                text_emb_model,
                text_batch,
                batch_size=min(64, max(1, len(text_batch))),
                prompt_name="passage",
            ).astype(np.float32, copy=False)
            if text_emb.ndim == 1:
                text_emb = text_emb.reshape(1, -1)
            if target_dim <= 0:
                target_dim = int(text_emb.shape[1])
            if text_emb.shape[1] != target_dim:
                print(
                    f"[Agent3][Qwen3VL][repair] text fallback dim mismatch: "
                    f"text_dim={text_emb.shape[1]} vs vl_dim={target_dim}; skip text fallback for this chunk."
                )
            else:
                for iid, row in zip(no_image_ids, text_emb):
                    repaired_missing_map[iid] = np.asarray(row, dtype=np.float32)
        del sub_inputs
        del sub_emb
        _cleanup_torch_cache()

    if not repaired_missing_map:
        return cached_ids, cached_matrix

    cache_map: Dict[str, np.ndarray] = {}
    for idx, iid in enumerate(cached_ids):
        if iid not in cache_map:
            cache_map[iid] = cached_matrix[idx]

    cache_map.update(repaired_missing_map)

    repaired_ids: List[str] = []
    repaired_rows: List[np.ndarray] = []
    for iid in all_item_ids:
        emb = cache_map.get(iid)
        if emb is None:
            continue
        repaired_ids.append(iid)
        repaired_rows.append(np.asarray(emb, dtype=np.float32))
    repaired_matrix = np.stack(repaired_rows, axis=0) if repaired_rows else np.zeros((0, 0), dtype=np.float32)
    np.savez_compressed(cache_path, item_ids=np.array(repaired_ids), item_embeddings=repaired_matrix)
    print(f"[Agent3][Qwen3VL][repair] cache updated -> {cache_path}")
    return repaired_ids, repaired_matrix


def _extract_query_keywords(query: str, max_keywords: int) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", query.lower())
    uniq: List[str] = []
    seen = set()
    for t in tokens:
        if t in EN_STOPWORDS:
            continue
        if len(t) <= 1:
            continue
        if t not in seen:
            seen.add(t)
            uniq.append(t)
        if len(uniq) >= max_keywords:
            break
    return uniq


def _keyword_match_score(title_lower: str, keywords: List[str]) -> Tuple[int, List[str]]:
    matched = [kw for kw in keywords if kw in title_lower]
    return len(matched), matched


def _build_hybrid_recall_ids(
    all_item_ids: List[str],
    title_lower_map: Dict[str, str],
    keywords: List[str],
    rank_indices: np.ndarray,
    keyword_recall_topk: int,
    embedding_recall_topk: int,
) -> Tuple[List[str], int, Dict[str, Any]]:
    keyword_topk = max(0, int(keyword_recall_topk))
    embedding_topk = max(0, int(embedding_recall_topk))

    matched_scored: List[Tuple[int, str, List[str]]] = []
    for iid in all_item_ids:
        score, matched = _keyword_match_score(title_lower_map.get(iid, ""), keywords)
        if score > 0:
            matched_scored.append((score, iid, matched))
    matched_scored.sort(key=lambda x: (-x[0], x[1]))
    matched_ids = [x[1] for x in matched_scored[:keyword_topk]] if keyword_topk > 0 else []

    embedding_ids: List[str] = []
    if embedding_topk > 0:
        for idx in rank_indices[:embedding_topk]:
            embedding_ids.append(all_item_ids[int(idx)])

    merged_ids: List[str] = []
    seen = set()
    for iid in matched_ids + embedding_ids:
        if iid in seen:
            continue
        merged_ids.append(iid)
        seen.add(iid)

    debug = {
        "keywords": keywords,
        "keyword_matched_count": len(matched_scored),
        "keyword_stage": f"keyword_top{keyword_topk}_embedding_top{embedding_topk}",
        "keyword_pool_size": len(matched_ids),
        "embedding_pool_size": len(embedding_ids),
        "merged_pool_size": len(merged_ids),
        "keyword_recall_topk": keyword_topk,
        "embedding_recall_topk": embedding_topk,
    }
    return merged_ids, len(merged_ids), debug


def _merge_unique_ids(*id_lists: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for ids in id_lists:
        for iid in ids:
            if iid in seen:
                continue
            seen.add(iid)
            merged.append(iid)
    return merged


def _rank_position_map(rank_indices: np.ndarray, item_ids: List[str]) -> Dict[str, int]:
    pos: Dict[str, int] = {}
    for rank, idx in enumerate(rank_indices, start=1):
        item_id = item_ids[int(idx)]
        if item_id not in pos:
            pos[item_id] = rank
    return pos


def _adaptive_embedding_fusion(
    history_ids: List[str],
    filtered_item_ids: List[str],
    text_rank_indices: np.ndarray,
    qwen3vl_rank_indices: np.ndarray | None,
    meta_map: Dict[str, Dict[str, Any]],
    min_total_recall: int = 500,
    max_total_recall: int = 500,
    max_pseudo_queries: int = 8,
) -> Tuple[List[str], Dict[str, Any]]:
    min_recall = int(max(1, min(500, min_total_recall)))
    max_recall = int(max(min_recall, min(500, max_total_recall)))

    def _safe_rank(rank: int) -> int:
        return int(rank if rank < 10**9 else 5000)

    def _rank_strength(rank: int) -> float:
        safe = _safe_rank(rank)
        return 1.0 / math.log2(safe + 2.0)

    def _estimate_total_k(history: List[Dict[str, Any]], hard_cap: int) -> int:
        if not history:
            return int(max(min_recall, min(500, hard_cap)))
        required: List[float] = []
        for row in history:
            if not bool(row.get("rank_available", True)):
                continue
            t_raw = row.get("text_rank")
            v_raw = row.get("vl_rank")
            if t_raw is None and v_raw is None:
                continue
            t_rank = _safe_rank(int(t_raw)) if t_raw is not None else 5000
            v_rank = _safe_rank(int(v_raw)) if v_raw is not None else 5000
            t_w = float(row.get("weights", {}).get("text", 0.5))
            v_w = float(row.get("weights", {}).get("vl", 0.5))
            low_rank = min(t_rank, v_rank)
            high_rank = max(t_rank, v_rank)
            dominance = abs(t_w - v_w)
            required.append(low_rank * (1.15 + 0.35 * dominance) + 0.15 * high_rank)
        if not required:
            return int(max(min_recall, min(500, hard_cap)))
        required.sort()
        pivot = required[int(0.75 * (len(required) - 1))]
        return int(max(min_recall, min(500, min(hard_cap, round(pivot)))))

    def _agent_finalize_params(
        history: List[Dict[str, Any]],
        cur_text_weight: float,
        cur_vl_weight: float,
        cur_total_k: int,
    ) -> Dict[str, Any]:
        if not history:
            return {
                "text_weight": round(cur_text_weight, 4),
                "vl_weight": round(cur_vl_weight, 4),
                "recall_size": int(cur_total_k),
                "mode": "fallback",
                "reasoning": "no_history",
            }
        window = [row for row in history[-min(6, len(history)) :] if bool(row.get("rank_available", True))]
        if not window:
            return {
                "text_weight": round(cur_text_weight, 4),
                "vl_weight": round(cur_vl_weight, 4),
                "recall_size": int(cur_total_k),
                "mode": "fallback",
                "reasoning": "no_ranked_history_in_pool",
            }
        trend_text = sum(float(row.get("weights", {}).get("text", 0.5)) for row in window) / max(1, len(window))
        text_vote = 0.0
        vl_vote = 0.0
        signs = []
        for row in window:
            t_rank = int(row.get("text_rank", 10**9))
            v_rank = int(row.get("vl_rank", 10**9))
            confidence = float(row.get("weights", {}).get("text", 0.5))
            if t_rank < v_rank:
                signs.append(1)
                text_vote += max(0.2, confidence)
            elif v_rank < t_rank:
                signs.append(-1)
                vl_vote += max(0.2, 1.0 - confidence)
            else:
                signs.append(0)
        switch_count = sum(1 for i in range(1, len(signs)) if signs[i] != 0 and signs[i - 1] != 0 and signs[i] != signs[i - 1])
        stable = switch_count <= 1
        vote_total = max(1e-6, text_vote + vl_vote)
        vote_margin = abs(text_vote - vl_vote) / vote_total
        if stable and vote_margin >= 0.15:
            dominant_share = min(0.95, 0.8 + 0.15 * vote_margin)
            final_text = dominant_share if text_vote >= vl_vote else (1.0 - dominant_share)
        elif stable and trend_text >= 0.6:
            final_text = max(0.8, trend_text)
        elif stable and trend_text <= 0.4:
            final_text = min(0.2, trend_text)
        else:
            final_text = trend_text
        final_text = float(max(0.05, min(0.95, final_text)))
        final_vl = 1.0 - final_text
        return {
            "text_weight": round(final_text, 4),
            "vl_weight": round(final_vl, 4),
            "recall_size": int(cur_total_k),
            "mode": "history_agent_update",
            "reasoning": (
                f"trend_text={round(trend_text, 3)}, text_vote={round(text_vote, 3)}, "
                f"vl_vote={round(vl_vote, 3)}, margin={round(vote_margin, 3)}, "
                f"switches={switch_count}, stable={stable}"
            ),
        }

    text_weight = 0.5
    vl_weight = 0.5
    history_target_text = 0.5
    total_k = int(max(min_recall, min(500, max_recall)))
    memory: List[Dict[str, Any]] = []

    if qwen3vl_rank_indices is None:
        top_ids = [filtered_item_ids[int(idx)] for idx in text_rank_indices[:total_k]]
        return top_ids, {"enabled": False, "reason": "qwen3vl_unavailable", "memory": memory}

    text_rank_map = _rank_position_map(text_rank_indices, filtered_item_ids)
    vl_rank_map = _rank_position_map(qwen3vl_rank_indices, filtered_item_ids)
    history_candidates: List[str] = []
    seen_history: set[str] = set()
    for raw_iid in history_ids:
        iid = str(raw_iid).strip()
        if not iid or iid in seen_history:
            continue
        seen_history.add(iid)
        if iid in meta_map:
            history_candidates.append(iid)
    pseudo_targets = history_candidates[: max(1, int(max_pseudo_queries))]
    for step, iid in enumerate(pseudo_targets, start=1):
        prev_text_weight = text_weight
        prev_vl_weight = vl_weight
        text_rank_raw = text_rank_map.get(iid)
        vl_rank_raw = vl_rank_map.get(iid)
        rank_available = text_rank_raw is not None or vl_rank_raw is not None
        if not rank_available:
            memory.append(
                {
                    "step": step,
                    "target_item_id": iid,
                    "text_rank": None,
                    "vl_rank": None,
                    "rank_available": False,
                    "weights_before": {"text": round(prev_text_weight, 4), "vl": round(prev_vl_weight, 4)},
                    "weights": {"text": round(text_weight, 4), "vl": round(vl_weight, 4)},
                    "reasoning": "history_item_not_in_current_recall_pool",
                }
            )
            continue
        text_rank = int(text_rank_raw) if text_rank_raw is not None else 10**9
        vl_rank = int(vl_rank_raw) if vl_rank_raw is not None else 10**9
        text_strength = _rank_strength(text_rank)
        vl_strength = _rank_strength(vl_rank)
        strength_sum = max(1e-8, text_strength + vl_strength)
        strength_target_text = text_strength / strength_sum
        low_rank = min(_safe_rank(text_rank), _safe_rank(vl_rank))
        gap = abs(_safe_rank(text_rank) - _safe_rank(vl_rank))
        confidence = min(1.0, abs(math.log((_safe_rank(vl_rank) + 1) / (_safe_rank(text_rank) + 1))) / 1.6)
        cover_share = min(0.95, max(0.5, low_rank / max(1.0, float(total_k))))
        gap_bonus = min(0.15, math.log1p(gap) / 40.0)
        dominant_share = min(0.95, max(0.8, cover_share + gap_bonus))
        if text_rank < vl_rank:
            step_target_text = dominant_share
        elif vl_rank < text_rank:
            step_target_text = 1.0 - dominant_share
        else:
            step_target_text = 0.5
        step_target_text = 0.45 * strength_target_text + 0.55 * step_target_text
        history_target_text = 0.75 * history_target_text + 0.25 * step_target_text
        max_step_change = 0.05 + 0.07 * confidence
        step_delta = max(-max_step_change, min(max_step_change, history_target_text - text_weight))
        text_weight = max(0.05, min(0.95, text_weight + step_delta))
        vl_weight = 1.0 - text_weight
        memory.append(
            {
                "step": step,
                "target_item_id": iid,
                "text_rank": int(text_rank),
                "vl_rank": int(vl_rank),
                "rank_available": True,
                "weights_before": {"text": round(prev_text_weight, 4), "vl": round(prev_vl_weight, 4)},
                "weights": {"text": round(text_weight, 4), "vl": round(vl_weight, 4)},
                "reasoning": (
                    f"strength_target={round(strength_target_text, 3)}, "
                    f"dominant_share={round(dominant_share, 3)}, "
                    f"confidence={round(confidence, 3)}, "
                    f"history_target={round(history_target_text, 3)}; "
                    "use history-smoothed extreme allocation (>=8:2 when modality differs)"
                ),
            }
        )
        total_k = _estimate_total_k(memory, int(max_recall))
        memory[-1]["estimated_total_recall"] = int(total_k)

    agent_final_params = _agent_finalize_params(memory, text_weight, vl_weight, total_k)
    text_weight = float(agent_final_params["text_weight"])
    vl_weight = float(agent_final_params["vl_weight"])
    final_modal_weights = {"text": round(text_weight, 4), "vl": round(vl_weight, 4)}
    final_modal_ratio = f"{int(round(text_weight * 100))}:{int(round(vl_weight * 100))}"
    text_k = max(1, int(round(total_k * text_weight)))
    vl_k = max(1, int(round(total_k * vl_weight)))
    text_ids = [filtered_item_ids[int(idx)] for idx in text_rank_indices[:text_k]]
    vl_ids = [filtered_item_ids[int(idx)] for idx in qwen3vl_rank_indices[:vl_k]]
    fused = _merge_unique_ids(text_ids, vl_ids)[: min(500, total_k)]
    return fused, {
        "enabled": True,
        "text_weight": round(text_weight, 4),
        "vl_weight": round(vl_weight, 4),
        "final_modal_weights": final_modal_weights,
        "final_modal_ratio": final_modal_ratio,
        "total_recall": int(min(500, total_k)),
        "agent_final_params": agent_final_params,
        "history_item_count": len(history_candidates),
        "history_items_in_recall_pool": int(
            sum(1 for iid in pseudo_targets if iid in text_rank_map or iid in vl_rank_map)
        ),
        "pseudo_query_count": len(pseudo_targets),
        "memory": memory,
    }


def _filter_item_ids_by_categories(
    candidate_item_ids: List[str],
    meta_map: Dict[str, Dict[str, Any]],
    selected_categories: List[List[str]],
) -> List[str]:
    """Exact-match prefilter by Agent3 selected category paths.

    Matching rule: any selected category path exactly equals one of item's category paths.
    """
    if not selected_categories:
        return candidate_item_ids

    selected_set = {
        tuple(str(seg).strip().lower() for seg in path if str(seg).strip())
        for path in selected_categories
        if isinstance(path, list)
    }
    selected_set = {x for x in selected_set if x}
    if not selected_set:
        return candidate_item_ids

    filtered: List[str] = []
    for iid in candidate_item_ids:
        meta = meta_map.get(iid, {})
        item_paths = {
            tuple(str(seg).strip().lower() for seg in path if str(seg).strip())
            for path in _meta_category_paths(meta)
        }
        if item_paths & selected_set:
            filtered.append(iid)
    return filtered


def _recall_at_k(labels: List[int], k: int) -> float:
    if not labels:
        return 0.0
    return float(sum(labels[:k]))


def _mrr_at_k(labels: List[int], k: int) -> float:
    for i, label in enumerate(labels[:k], start=1):
        if int(label) == 1:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(labels: List[int], k: int) -> float:
    ranked = labels[:k]
    dcg = 0.0
    for i, rel in enumerate(ranked, start=1):
        dcg += (2 ** int(rel) - 1) / math.log2(i + 1)

    ideal = sorted((int(x) for x in labels), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal, start=1):
        idcg += (2 ** rel - 1) / math.log2(i + 1)

    if idcg <= 0:
        return 0.0
    return dcg / idcg


def _safe_item_id(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("item_id", "")).strip()
    return str(value or "").strip()


def _calc_metrics_from_dynamic_output(path: Path, top_n: int) -> Dict[str, float] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Metrics] skip unreadable file: {path.name} error={exc}")
        return None

    target_id = _safe_item_id(payload.get("groundtruth_target_item_id"))
    if not target_id:
        print(f"[Metrics] skip file without groundtruth_target_item_id: {path.name}")
        return None

    ranked_items = payload.get("ranked_items", [])
    if not isinstance(ranked_items, list):
        ranked_items = []

    top_ranked_ids = [_safe_item_id(x) for x in ranked_items[:top_n]]
    labels = [1 if iid and iid == target_id else 0 for iid in top_ranked_ids]
    if not labels:
        labels = [0]

    return {
        f"recall@{top_n}": _recall_at_k(labels, top_n),
        f"ndcg@{top_n}": _ndcg_at_k(labels, top_n),
        f"mrr@{top_n}": _mrr_at_k(labels, top_n),
    }


def _print_dynamic_output_metrics(output_dir: str | Path, top_ns: List[int] | Tuple[int, ...] = (10, 20, 40)) -> None:
    pattern = str(Path(output_dir) / "*_dynamic_reasoning_ranking_output.json")
    paths = [Path(p) for p in sorted(glob.glob(pattern))]
    if not paths:
        print(f"[Metrics] no ranking outputs found in {output_dir}")
        return

    normalized_top_ns = []
    seen_top_ns = set()
    for k in top_ns:
        try:
            top_k = int(k)
        except (TypeError, ValueError):
            continue
        if top_k <= 0 or top_k in seen_top_ns:
            continue
        seen_top_ns.add(top_k)
        normalized_top_ns.append(top_k)

    if not normalized_top_ns:
        print(f"[Metrics] no valid top-k values configured for {output_dir}")
        return

    metrics_by_topk: Dict[int, List[Dict[str, float]]] = {k: [] for k in normalized_top_ns}
    for p in paths:
        for top_k in normalized_top_ns:
            row = _calc_metrics_from_dynamic_output(p, top_n=top_k)
            if row is not None:
                metrics_by_topk[top_k].append(row)

    available_topks = [k for k in normalized_top_ns if metrics_by_topk[k]]
    if not available_topks:
        print(f"[Metrics] no valid ranking outputs with groundtruth target in {output_dir}")
        return

    files_count = len(metrics_by_topk[available_topks[0]])
    metric_chunks = []
    for top_k in available_topks:
        metric_rows = metrics_by_topk[top_k]
        recall_key = f"recall@{top_k}"
        ndcg_key = f"ndcg@{top_k}"
        mrr_key = f"mrr@{top_k}"
        recall = float(np.mean([x[recall_key] for x in metric_rows]))
        ndcg = float(np.mean([x[ndcg_key] for x in metric_rows]))
        mrr = float(np.mean([x[mrr_key] for x in metric_rows]))
        metric_chunks.append(
            f"@{top_k} HitRate/Recall={recall:.6f} NDCG={ndcg:.6f} MRR={mrr:.6f}"
        )

    print(f"[Metrics][Aggregated] files={files_count} " + " | ".join(metric_chunks))


def _write_recall_failed_zero_output(output_path: Path, user_id: str, query: str, target_id: str) -> None:
    payload = {
        "user_id": str(user_id),
        "query": str(query),
        "preference_constraints": {
            "Must_Have": [],
            "Nice_to_Have": [],
            "Must_Avoid": [],
            "Predicted_Next_Items": [],
            "Reasoning": "agent3_recall_failed_skip_agent45",
        },
        "ranked_items": [],
        "groundtruth_target_item_id": str(target_id),
        "agent3_recall_hit": 0,
    }
    _save_json(output_path, payload)


def _has_non_empty_ranked_items(output_path: Path) -> bool:
    if not output_path.exists():
        return False
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[UserLoop] existing output unreadable, rerun user. path={output_path} error={exc}")
        return False
    ranked_items = payload.get("ranked_items", [])
    return isinstance(ranked_items, list) and len(ranked_items) > 0


def run(args: argparse.Namespace) -> Dict[str, Any]:
    query_df = pd.read_csv(args.query_csv, dtype={"id": str, "user_id": str})
    if args.max_users > 0:
        query_df = query_df.head(args.max_users)

    keyword_recall_topk = int(
        args.keyword_recall_topk if int(getattr(args, "keyword_recall_topk", 0)) > 0 else args.agent3_keyword_topk
    )
    embedding_recall_topk = int(
        args.embedding_recall_topk
        if int(getattr(args, "embedding_recall_topk", 0)) > 0
        else args.agent3_embedding_topk
    )

    meta_map = load_filtered_meta(Path(args.filtered_meta_jsonl))
    if not meta_map:
        raise ValueError(f"No items loaded from filtered meta: {args.filtered_meta_jsonl}")

    all_item_ids = sorted(meta_map.keys())
    item_id_to_index = {iid: idx for idx, iid in enumerate(all_item_ids)}
    title_lower_map = {iid: str(meta_map[iid].get("title", "") or "").lower() for iid in all_item_ids}

    cache_dir = Path(args.cache_dir)
    emb_cache_path = cache_dir / "agent3_item_embedding_cache.npz"
    qwen3vl_emb_cache_path = cache_dir / "agent3_item_qwen3vl_embedding_cache.npz"
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
    qwen3vl_model = None
    qwen3vl_item_emb_norm: np.ndarray | None = None
    qwen3vl_item_id_to_index: Dict[str, int] = {}
    if args.enable_agent3_qwen3vl_embedding:
        try:
            from adaptive_pipe.qwen3_vl_embedding import Qwen3VLEmbedder
        except ModuleNotFoundError:
            from qwen3_vl_embedding import Qwen3VLEmbedder
        print(f"[Init] load multimodal embedding model: {args.agent3_qwen3vl_model}")
        image_cache_dir = cache_dir / "agent3_qwen3vl_images"
        image_url_to_local = prefetch_item_images(
            meta_map=meta_map,
            resolve_image_fn=_safe_meta_image,
            cache_dir=image_cache_dir,
            max_workers=max(1, int(args.agent3_qwen3vl_prefetch_workers)),
            timeout_sec=max(1, int(args.agent3_qwen3vl_prefetch_timeout)),
        )
        qwen3vl_model = Qwen3VLEmbedder(
            model_name_or_path=args.agent3_qwen3vl_model,
            min_pixels=max(1, int(args.agent3_qwen3vl_min_pixels)),
            max_pixels=max(1, int(args.agent3_qwen3vl_max_pixels)),
        )
        q_item_ids_cached: List[str] = []
        q_item_emb_matrix: np.ndarray | None = None
        if qwen3vl_emb_cache_path.exists():
            q_npz = np.load(qwen3vl_emb_cache_path, allow_pickle=True)
            q_item_ids_cached = [str(x) for x in q_npz["item_ids"].tolist()]
            q_item_emb_matrix = q_npz["item_embeddings"].astype(np.float32, copy=False)
            if q_item_emb_matrix is not None:
                q_item_ids_cached, q_item_emb_matrix = _repair_qwen3vl_cache_for_missing_ids(
                    qwen3vl_model=qwen3vl_model,
                    text_emb_model=emb_model,
                    cache_path=qwen3vl_emb_cache_path,
                    all_item_ids=all_item_ids,
                    cached_ids=q_item_ids_cached,
                    cached_matrix=q_item_emb_matrix,
                    meta_map=meta_map,
                    image_url_to_local=image_url_to_local,
                    chunk_size=max(1, int(args.agent3_qwen3vl_chunk_size)),
                )
        if q_item_emb_matrix is None:
            q_item_emb_matrix = _build_qwen3vl_item_embedding_cache(
                qwen3vl_model=qwen3vl_model,
                all_item_ids=all_item_ids,
                meta_map=meta_map,
                emb_cache_path=qwen3vl_emb_cache_path,
                chunk_size=max(1, int(args.agent3_qwen3vl_chunk_size)),
                save_every_items=max(1, int(args.agent3_qwen3vl_save_every)),
                image_url_to_local=image_url_to_local,
            )
            q_item_ids_cached = list(all_item_ids[: int(q_item_emb_matrix.shape[0])])
        elif q_item_ids_cached != all_item_ids:
            print(
                "[Agent3][Qwen3VL] cache still partial after repair; "
                "skip full rebuild and use available embedded subset only."
            )
        q_aligned = min(len(q_item_ids_cached), int(q_item_emb_matrix.shape[0]))
        if q_aligned <= 0:
            raise ValueError("[Agent3][Qwen3VL] empty embedding matrix after cache build/repair.")
        if q_aligned != len(q_item_ids_cached) or q_aligned != int(q_item_emb_matrix.shape[0]):
            print(
                f"[Agent3][Qwen3VL] align id/emb mismatch: ids={len(q_item_ids_cached)} "
                f"emb_rows={int(q_item_emb_matrix.shape[0])}; use aligned={q_aligned}"
            )
            q_item_ids_cached = q_item_ids_cached[:q_aligned]
            q_item_emb_matrix = q_item_emb_matrix[:q_aligned]
        qwen3vl_item_emb_norm = _l2_normalize(q_item_emb_matrix)
        qwen3vl_item_id_to_index = {iid: idx for idx, iid in enumerate(q_item_ids_cached)}
    global_db = GlobalItemDB(args.global_db)
    history_db = UserHistoryLogDB(args.history_db)
    vl_extractor = Qwen3VLExtractor(model_name=args.vl_model) if args.enable_vl_profiling else None

    category_catalog = sorted({_meta_category_text(v) for v in meta_map.values() if _meta_category_text(v)})
    results: List[Dict[str, Any]] = []
    skipped_users_missing_target_embedding = 0
    modal_trace_path = Path(args.output_dir) / "agent3_modal_modulation_trace.jsonl"
    if modal_trace_path.exists():
        modal_trace_path.unlink()

    for row_idx, row in query_df.iterrows():
        user_id = str(row["user_id"])
        target_id = str(row["id"])
        query = str(row.get("new_query") or row.get("query") or "").strip()
        if not query:
            continue

        print(f"\n[UserLoop] {row_idx + 1}/{len(query_df)} user={user_id} target={target_id}")

        existing_output = Path(args.output_dir) / f"user_{user_id}_dynamic_reasoning_ranking_output.json"
        if _has_non_empty_ranked_items(existing_output):
            print(f"[UserLoop] skip user={user_id}: existing non-empty ranking output found at {existing_output}")
            _print_dynamic_output_metrics(args.output_dir)
            continue
        if existing_output.exists():
            print(f"[UserLoop] user={user_id} has empty ranked_items output, retry Agent3 recall before deciding skip")

        target_in_text_embedding = target_id in item_id_to_index
        target_in_vl_embedding = (
            (not bool(getattr(args, "enable_agent3_qwen3vl_embedding", False)))
            or (target_id in qwen3vl_item_id_to_index)
        )
        if (not target_in_text_embedding) or (not target_in_vl_embedding):
            print(
                f"[UserLoop] skip user={user_id}: target={target_id} missing embedding "
                f"(text={target_in_text_embedding}, vl={target_in_vl_embedding})"
            )
            skipped_users_missing_target_embedding += 1
            continue

        routed = _route_query(query, category_catalog, args.enable_llm_routing, args.text_model)

        q_sentence = _query_sentence(query, routed["selected_category_paths"], routed["rewritten_query"])
        query_sentence_cache[f"{user_id}::{q_sentence}"] = q_sentence

        query_recall_pool_mode = str(getattr(args, "agent3_query_recall_pool", "filtered")).strip().lower()
        if args.agent3_skip_category_prefilter:
            query_recall_pool_mode = "full"
        if query_recall_pool_mode == "full":
            filtered_item_ids = all_item_ids
            print(f"[Agent3][categories] query_recall_pool=full candidate_count={len(filtered_item_ids)}")
        else:
            filtered_item_ids = _filter_item_ids_by_categories(
                candidate_item_ids=all_item_ids,
                meta_map=meta_map,
                selected_categories=routed.get("selected_category_paths", []) or [],
            )
            print(f"[Agent3][categories] exact_match_count={len(filtered_item_ids)}")

            if not filtered_item_ids:
                print("[Agent3] category exact-match prefilter found 0 items. recall failed.")
                _write_recall_failed_zero_output(
                    output_path=existing_output,
                    user_id=user_id,
                    query=q_sentence,
                    target_id=target_id,
                )
                results.append(
                    {
                        "user_id": user_id,
                        "target_id": target_id,
                        "hit": 0,
                        "used_k": 0,
                        "kw_debug": {
                            "keywords": [],
                            "keyword_matched_count": 0,
                            "keyword_stage": "category_prefilter_empty",
                            "keyword_pool_size": 0,
                            "embedding_pool_size": 0,
                            "merged_pool_size": 0,
                            "keyword_recall_topk": int(keyword_recall_topk),
                            "embedding_recall_topk": int(embedding_recall_topk),
                            "prefilter_candidate_size": 0,
                        },
                    }
                )
                _print_dynamic_output_metrics(args.output_dir)
                continue

        filtered_idx = [item_id_to_index[iid] for iid in filtered_item_ids]
        filtered_emb = item_emb_norm[np.array(filtered_idx)]

        q_emb = _encode_texts(emb_model, [q_sentence], batch_size=1, prompt_name="query").astype(np.float32, copy=False)
        q_emb_norm = q_emb / np.clip(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12, None)
        sim_matrix = np.matmul(filtered_emb, q_emb_norm[0])
        rank_indices = np.argsort(-sim_matrix)
        full_sim_matrix = np.matmul(item_emb_norm, q_emb_norm[0])
        full_rank_indices = np.argsort(-full_sim_matrix)

        keywords = _extract_query_keywords(query, max_keywords=args.max_query_keywords)
        hybrid_embedding_topk = (
            0
            if bool(getattr(args, "enable_agent3_adaptive_weighting", False))
            else int(embedding_recall_topk)
        )
        top_ids, used_k, kw_debug = _build_hybrid_recall_ids(
            all_item_ids=filtered_item_ids,
            title_lower_map=title_lower_map,
            keywords=keywords,
            rank_indices=rank_indices,
            keyword_recall_topk=keyword_recall_topk,
            embedding_recall_topk=hybrid_embedding_topk,
        )
        qwen3vl_rank_indices = None
        qwen3vl_rank_indices_all = None
        qwen3vl_ids: List[str] = []
        if args.enable_agent3_qwen3vl_embedding and qwen3vl_model is not None and qwen3vl_item_emb_norm is not None:
            qwen3vl_query_input = {"text": q_sentence}
            query_image = str(row.get("query_image") or row.get("image") or row.get("image_url") or "").strip()
            if query_image:
                if query_image in image_url_to_local:
                    query_image = image_url_to_local[query_image]
                qwen3vl_query_input["image"] = query_image
            qwen3vl_q_emb = _tensor_to_float32_numpy(qwen3vl_model.process([qwen3vl_query_input]))
            qwen3vl_q_emb_norm = _l2_normalize(qwen3vl_q_emb)[0]
            qwen_filtered_item_ids = [iid for iid in filtered_item_ids if iid in qwen3vl_item_id_to_index]
            if not qwen_filtered_item_ids:
                kw_debug["qwen3vl_enabled"] = False
                kw_debug["qwen3vl_reason"] = "no_embedded_items_in_filtered_pool"
                qwen3vl_rank_indices = None
                qwen3vl_ids = []
            else:
                qwen_filtered_idx = [qwen3vl_item_id_to_index[iid] for iid in qwen_filtered_item_ids]
                q_filtered_emb = qwen3vl_item_emb_norm[np.array(qwen_filtered_idx)]
                qwen3vl_sim = np.matmul(q_filtered_emb, qwen3vl_q_emb_norm)
                qwen3vl_rank_indices = np.argsort(-qwen3vl_sim)
                qwen_all_item_ids = [iid for iid in all_item_ids if iid in qwen3vl_item_id_to_index]
                qwen_all_idx = [qwen3vl_item_id_to_index[iid] for iid in qwen_all_item_ids]
                qwen_all_emb = qwen3vl_item_emb_norm[np.array(qwen_all_idx)]
                qwen_all_sim = np.matmul(qwen_all_emb, qwen3vl_q_emb_norm)
                qwen_sim_aligned = np.full(len(all_item_ids), -1e9, dtype=np.float32)
                aligned_pos = np.array([item_id_to_index[iid] for iid in qwen_all_item_ids], dtype=np.int32)
                qwen_sim_aligned[aligned_pos] = qwen_all_sim.astype(np.float32, copy=False)
                qwen3vl_rank_indices_all = np.argsort(-qwen_sim_aligned)
                mm_topk = max(1, int(args.agent3_qwen3vl_topk))
                qwen3vl_ids = [qwen_filtered_item_ids[int(idx)] for idx in qwen3vl_rank_indices[:mm_topk]]
                top_ids = _merge_unique_ids(top_ids, qwen3vl_ids)
                used_k = len(top_ids)
                kw_debug["qwen3vl_enabled"] = True
                kw_debug["qwen3vl_topk"] = mm_topk
                kw_debug["qwen3vl_pool_size"] = len(qwen3vl_ids)
                kw_debug["qwen3vl_embedded_pool_size"] = len(qwen_filtered_item_ids)
                kw_debug["merged_pool_size"] = len(top_ids)
        else:
            kw_debug["qwen3vl_enabled"] = False
        history_ids = list(
            dict.fromkeys(
                x.strip() for x in str(row.get("remaining_interaction_string", "")).split("|") if x.strip()
            )
        )
        if bool(getattr(args, "enable_agent3_adaptive_weighting", False)):
            adaptive_ids, adaptive_state = _adaptive_embedding_fusion(
                history_ids=history_ids,
                filtered_item_ids=all_item_ids,
                text_rank_indices=full_rank_indices,
                qwen3vl_rank_indices=qwen3vl_rank_indices_all,
                meta_map=meta_map,
                min_total_recall=int(getattr(args, "agent3_adaptive_min_total_recall", 500)),
                max_total_recall=int(getattr(args, "agent3_adaptive_max_total_recall", 500)),
                max_pseudo_queries=int(getattr(args, "agent3_adaptive_max_pseudo_queries", 8)),
            )
            top_ids = _merge_unique_ids(top_ids, adaptive_ids)
            used_k = len(top_ids)
            kw_debug["adaptive_embedding_state"] = adaptive_state
        adaptive_state = kw_debug.get("adaptive_embedding_state", {}) if isinstance(kw_debug, dict) else {}
        if isinstance(adaptive_state, dict) and adaptive_state.get("enabled"):
            agent_final_params = adaptive_state.get("agent_final_params", {})
            modal_params = {
                "text_weight": float(agent_final_params.get("text_weight", adaptive_state.get("text_weight", 0.5))),
                "vl_weight": float(agent_final_params.get("vl_weight", adaptive_state.get("vl_weight", 0.5))),
                "recall_size": int(agent_final_params.get("recall_size", adaptive_state.get("total_recall", len(top_ids)))),
                "source": "agent_adaptive" if agent_final_params else "adaptive",
            }
            if isinstance(agent_final_params, dict) and agent_final_params:
                modal_params["agent_reasoning"] = str(agent_final_params.get("reasoning", ""))
            modal_params["final_modal_ratio"] = str(adaptive_state.get("final_modal_ratio", ""))
            modal_params["final_modal_weights"] = adaptive_state.get("final_modal_weights", {})
        else:
            text_pool = int(kw_debug.get("embedding_pool_size", 0))
            vl_pool = int(kw_debug.get("qwen3vl_pool_size", 0))
            denom = max(1, text_pool + vl_pool)
            modal_params = {
                "text_weight": float(text_pool / denom),
                "vl_weight": float(vl_pool / denom),
                "recall_size": int(kw_debug.get("merged_pool_size", len(top_ids))),
                "source": "pool_ratio",
            }
        kw_debug["modal_modulation_params"] = modal_params
        _append_jsonl(
            modal_trace_path,
            {
                "user_id": user_id,
                "target_id": target_id,
                "query": query,
                "modal_modulation_params": modal_params,
                "adaptive_embedding_state": adaptive_state if isinstance(adaptive_state, dict) else {},
            },
        )
        print(
            f"[Agent3][keyword] keywords={kw_debug['keywords']} matched={kw_debug['keyword_matched_count']} "
            f"stage={kw_debug['keyword_stage']} prefilter_size={len(filtered_item_ids)}"
        )

        hit = target_id in top_ids
        if not hit:
            print("[Agent3] recall failed. metric=0, skip Agent1/2/4/5")
            _write_recall_failed_zero_output(
                output_path=existing_output,
                user_id=user_id,
                query=q_sentence,
                target_id=target_id,
            )
            results.append({"user_id": user_id, "target_id": target_id, "hit": 0, "used_k": used_k, "kw_debug": kw_debug})
            _print_dynamic_output_metrics(args.output_dir)
            continue

        print(f"[Agent3] recall hit at k={used_k}; run Agent1/2")
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
                    profile = vl_extractor.extract(prompt=f"Profile item: {item_input.title}\n{item_input.detail_text}", image_paths=[item_input.main_image])
                else:
                    profile = _lightweight_profile(meta, iid)
                global_db.upsert(iid, profile)

            candidate_items.append({"item_id": iid, "profile": profile})
            if i % 50 == 0 or i == len(top_ids):
                print(f"[Agent1] {i}/{len(top_ids)}")

        history_rows: List[Dict[str, Any]] = []
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
                    profile = vl_extractor.extract(prompt=f"Profile item: {item_input.title}\n{item_input.detail_text}", image_paths=[item_input.main_image])
                else:
                    profile = _lightweight_profile(meta, iid)
                global_db.upsert(iid, profile)

            if not history_db.exists(user_id=user_id, item_id=iid, behavior="positive", timestamp=None):
                history_db.insert(user_id=user_id, item_id=iid, behavior="positive", timestamp=None, profile=profile)

            history_rows.append({"user_id": user_id, "item_id": iid, "behavior": "positive", "timestamp": None, "profile": profile})
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
                groundtruth_target_item_id=target_id,
                enable_collaborative_signal=bool(args.enable_collaborative_signal),
                collaborative_similarity_threshold=float(args.collaborative_similarity_threshold),
                collaborative_db_path=args.collaborative_db_path,
                collaborative_embedding_model_name=args.collaborative_embedding_model,
                collaborative_max_users=int(args.collaborative_max_users),
                collaborative_max_items=int(args.collaborative_max_items),
            )
            ranked_first = module3_out.ranked_items[0]["item_id"] if module3_out.ranked_items else ""
        else:
            print("[Agent4/5] skipped by --disable-agent45")

        results.append({
            "user_id": user_id,
            "target_id": target_id,
            "hit": 1,
            "used_k": used_k,
            "top1": ranked_first,
            "kw_debug": kw_debug,
        })
        _print_dynamic_output_metrics(args.output_dir)

    text_cache["items"] = item_sentence_cache
    text_cache["queries"] = query_sentence_cache
    _save_json(text_cache_path, text_cache)
    _save_json(Path(args.output_dir) / "unified_eval_results.json", results)

    recall_rate = float(np.mean([r["hit"] for r in results])) if results else 0.0
    summary = {
        "rows": len(results),
        "recall@k": recall_rate,
        "output_dir": args.output_dir,
        "skipped_users_missing_target_embedding": int(skipped_users_missing_target_embedding),
    }
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
    parser.add_argument("--agent3-keyword-topk", type=int, default=250, help="Agent3基于标题关键词匹配的Top-K召回数量。")
    parser.add_argument("--agent3-embedding-topk", type=int, default=250, help="Agent3基于向量相似度的Top-K召回数量。")
    parser.add_argument("--keyword-recall-topk", type=int, default=0, help="兼容cloth命名；>0时覆盖--agent3-keyword-topk。")
    parser.add_argument("--embedding-recall-topk", type=int, default=0, help="兼容cloth命名；>0时覆盖--agent3-embedding-topk。")
    parser.add_argument("--enable-agent3-qwen3vl-embedding", action="store_true", help="开启后，Agent3新增一路Qwen3-VL多模态embedding召回（文本+图片）。默认关闭。")
    parser.add_argument("--agent3-qwen3vl-topk", type=int, default=25, help="Agent3新增Qwen3-VL多模态embedding召回Top-K。")
    parser.add_argument("--agent3-qwen3vl-model", default="Qwen/Qwen3-VL-Embedding-2B", help="Agent3多模态embedding模型名称。")
    parser.add_argument("--agent3-qwen3vl-min-pixels", type=int, default=4096, help="Qwen3-VL输入图最小像素约束。")
    parser.add_argument("--agent3-qwen3vl-max-pixels", type=int, default=1048576, help="Qwen3-VL输入图最大像素约束；过大图片会被压到该预算。")
    parser.add_argument("--agent3-qwen3vl-chunk-size", type=int, default=100, help="Qwen3-VL多模态embedding建库分块大小（默认100）。")
    parser.add_argument("--agent3-qwen3vl-save-every", type=int, default=1000, help="Qwen3-VL embedding每累计多少条落盘一次part文件，最后再合并。")
    parser.add_argument("--agent3-qwen3vl-prefetch-workers", type=int, default=16, help="Qwen3-VL图片预下载并发数。")
    parser.add_argument("--agent3-qwen3vl-prefetch-timeout", type=int, default=8, help="Qwen3-VL图片预下载超时秒数。")
    parser.add_argument("--enable-agent3-adaptive-weighting", action="store_true", help="开启Agent3基于历史伪查询的text/vl自适应权重迭代。")
    parser.add_argument(
        "--agent3-query-recall-pool",
        choices=["filtered", "full"],
        default="filtered",
        help="控制真实query召回候选池：filtered=categories过滤后；full=全库。",
    )
    parser.add_argument("--agent3-adaptive-min-total-recall", type=int, default=500, help="Agent3 text+vl融合召回总量下限（<=500）。")
    parser.add_argument("--agent3-adaptive-max-total-recall", type=int, default=500, help="Agent3 text+vl融合召回总量上限（<=500）。")
    parser.add_argument("--agent3-adaptive-max-pseudo-queries", type=int, default=8, help="Agent3每次最多使用多少历史商品构造伪查询。")
    parser.add_argument(
        "--agent3-skip-category-prefilter",
        action="store_true",
        help="开启后，Agent3不再先按categories做精确匹配过滤，而是保留categories在查询句中并直接在全库执行关键词/向量召回。",
    )
    parser.add_argument("--max-query-keywords", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=40)
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
    parser.add_argument("--enable-collaborative-signal", action="store_true", help="开启Agent4协同信号：基于Reasoning embedding检索相似用户并扩充Agent5候选池")
    parser.add_argument("--collaborative-similarity-threshold", type=float, default=0.5, help="协同信号相似用户阈值（cosine）")
    parser.add_argument("--collaborative-db-path", default="processed/music_collaborative_signal.db", help="协同信号SQLite存储路径")
    parser.add_argument("--collaborative-embedding-model", default="Qwen/Qwen3-Embedding-0.6B", help="用于Reasoning向量化的Embedding模型")
    parser.add_argument("--collaborative-max-users", type=int, default=20, help="每次最多引入的相似用户数")
    parser.add_argument("--collaborative-max-items", type=int, default=120, help="每次从相似用户历史引入的最大商品数")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.enable_agent45 = not bool(args.disable_agent45)
    run(args)
