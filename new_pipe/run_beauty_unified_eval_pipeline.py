from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

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

def calculate_mrr(ranked_items: List[int]) -> float:
    reciprocal_ranks = []
    for rank, score in enumerate(ranked_items, start=1):
        if score > 0:
            reciprocal_ranks.append(1 / rank)
            break
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def calculate_ndcg(ranked_items: List[int], p: int = 10) -> float:
    dcg = 0.0
    for i in range(min(p, len(ranked_items))):
        rel_i = ranked_items[i]
        dcg += rel_i / np.log2(i + 2)
    ideal_relevance_scores = sorted(ranked_items, reverse=True)
    idcg = 0.0
    for i in range(min(p, len(ideal_relevance_scores))):
        rel_i = ideal_relevance_scores[i]
        idcg += rel_i / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


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


def _safe_json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int, prompt_name: str | None = None) -> np.ndarray:
    if torch is not None:
        with torch.inference_mode():
            return model.encode(texts, batch_size=batch_size, prompt_name=prompt_name, convert_to_numpy=True, show_progress_bar=False)
    return model.encode(texts, batch_size=batch_size, prompt_name=prompt_name, convert_to_numpy=True, show_progress_bar=False)


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
        chunk_sentences = []
        for iid in all_item_ids[start:end]:
            sentence = item_sentence_cache.get(iid)
            if not sentence:
                sentence = _item_sentence(meta_map[iid])
                item_sentence_cache[iid] = sentence
            chunk_sentences.append(sentence)

        try:
            chunk_emb = _encode_texts(emb_model, chunk_sentences, embed_batch_size, prompt_name=None)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and embed_batch_size > 1:
                new_bs = max(1, embed_batch_size // 2)
                print(f"[Agent3] OOM at batch_size={embed_batch_size}, retry {start}-{end} with batch_size={new_bs}")
                chunk_emb = _encode_texts(emb_model, chunk_sentences, new_bs, prompt_name=None)
            else:
                raise

        all_emb_chunks.append(chunk_emb.astype(np.float32, copy=False))
        processed = end
        print(f"[Agent3][embedding chunk] {processed}/{total} (chunk={start}-{end})")

        if processed % save_every_n == 0 or processed == total:
            partial = np.concatenate(all_emb_chunks, axis=0)
            emb_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(emb_cache_path, item_ids=np.array(all_item_ids[:processed]), item_embeddings=partial)
            print(f"[Agent3][cache save] {processed}/{total} -> {emb_cache_path}")

        _cleanup_torch_cache()

    final_emb = np.concatenate(all_emb_chunks, axis=0)
    np.savez_compressed(emb_cache_path, item_ids=np.array(all_item_ids), item_embeddings=final_emb)
    return final_emb


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


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
    recall_topk: int,
) -> Tuple[List[str], int, Dict[str, Any]]:
    matched_scored: List[Tuple[int, str, List[str]]] = []
    for iid in all_item_ids:
        score, matched = _keyword_match_score(title_lower_map.get(iid, ""), keywords)
        if score > 0:
            matched_scored.append((score, iid, matched))
    matched_scored.sort(key=lambda x: (-x[0], x[1]))
    matched_ids = [x[1] for x in matched_scored]

    fixed_limit = max(1, int(recall_topk))
    out: List[str] = []
    seen = set()

    for iid in matched_ids[:fixed_limit]:
        if iid not in seen:
            out.append(iid)
            seen.add(iid)
        if len(out) >= fixed_limit:
            break

    for idx in rank_indices:
        iid = all_item_ids[int(idx)]
        if iid in seen:
            continue
        out.append(iid)
        seen.add(iid)
        if len(out) >= fixed_limit:
            break

    debug = {
        "keywords": keywords,
        "keyword_matched_count": len(matched_ids),
        "keyword_stage": f"fixed_top_{fixed_limit}",
        "keyword_pool_size": min(len(matched_ids), fixed_limit),
        "strict_non_keyword_extra": False,
        "fixed_keyword_pool_size": min(len(matched_ids), fixed_limit),
    }
    return out, len(out), debug


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


class Qwen3RankingEvaluator:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", max_new_tokens: int = 1024, enable_thinking: bool = True) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
            raise ImportError("transformers/torch are not available for Qwen3RankingEvaluator.")
        if self._tokenizer is not None and self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    @staticmethod
    def _try_json_decode(text: str) -> Dict[str, Any] | None:
        stripped = text.strip()
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        if "```" in stripped:
            for part in stripped.split("```"):
                cand = part.replace("json", "", 1).strip()
                if not cand:
                    continue
                try:
                    payload = json.loads(cand)
                    if isinstance(payload, dict):
                        return payload
                except json.JSONDecodeError:
                    continue
        return None

    def evaluate(self, query: str, target_title: str, ranked_candidates: List[Dict[str, str]]) -> List[int]:
        self.load()
        candidate_lines = []
        for idx, cand in enumerate(ranked_candidates[:10], start=1):
            candidate_lines.append(
                f"{idx}. item_id={cand['item_id']} | title={cand['title']} | categories={cand['categories']} | description={cand['description']}"
            )
        prompt = (
            "你是电商推荐评估器。请参考 agents/evaluate_agent.py 的宽松相关性判断思想，"
            "评估排序结果前10个商品对用户query的满足程度。\n"
            "打分规则：0=不相关，1=相关，2=与给定groundtruth样例商品本质上相同或几乎同一商品。\n"
            "请只输出JSON对象，字段为 relevance_scores，对应10个候选（若候选不足则按实际数量输出）。\n\n"
            f"用户query: {query}\n"
            f"groundtruth样例商品标题: {target_title}\n"
            f"候选商品列表:\n" + "\n".join(candidate_lines)
        )
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content = self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        payload = self._try_json_decode(content) or {}
        raw_scores = payload.get("relevance_scores", [])
        if not isinstance(raw_scores, list):
            raw_scores = []
        scores: List[int] = []
        for x in raw_scores[:10]:
            try:
                v = int(float(x))
            except (TypeError, ValueError):
                v = 0
            scores.append(0 if v < 0 else 2 if v > 2 else v)
        while len(scores) < len(ranked_candidates[:10]):
            scores.append(0)
        return scores


def _build_eval_candidates(ranked_items: List[Any], meta_map: Dict[str, Dict[str, Any]], top_n: int = 10) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for x in ranked_items[:top_n]:
        iid = _safe_item_id(x)
        meta = meta_map.get(iid, {})
        desc = meta.get("description", "")
        if isinstance(desc, list):
            desc = " ".join(str(v) for v in desc if str(v).strip())
        out.append({
            "item_id": iid,
            "title": str(meta.get("title", "") or ""),
            "categories": _meta_category_text(meta),
            "description": str(desc or "")[:400],
        })
    return out


def _calc_llm_eval_from_dynamic_output(
    path: Path,
    meta_map: Dict[str, Dict[str, Any]],
    evaluator: Qwen3RankingEvaluator | None,
    top_n: int = 10,
) -> Dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[EvalMetrics] skip unreadable file: {path.name} error={exc}")
        return None

    ranked_items = payload.get("ranked_items", [])
    if not isinstance(ranked_items, list):
        ranked_items = []
    ranked_items = ranked_items[:top_n]
    if not ranked_items:
        return {"eval_hit@10": 0.0, "eval_ndcg@10": 0.0, "eval_mrr@10": 0.0, "relevance_scores": []}

    if evaluator is None:
        raise RuntimeError("Qwen3 ranking evaluator is not initialized.")

    target_id = _safe_item_id(payload.get("groundtruth_target_item_id"))
    target_title = str(meta_map.get(target_id, {}).get("title", "") or "")
    query = str(payload.get("query", "") or "")
    candidates = _build_eval_candidates(ranked_items, meta_map=meta_map, top_n=top_n)
    scores = evaluator.evaluate(query=query, target_title=target_title, ranked_candidates=candidates)
    return {
        "eval_hit@10": float(sum(scores) / 10.0) if scores else 0.0,
        "eval_ndcg@10": float(calculate_ndcg(scores, p=10)) if scores else 0.0,
        "eval_mrr@10": float(calculate_mrr(scores)) if scores else 0.0,
        "relevance_scores": scores,
    }


def _calc_metrics_from_dynamic_output(path: Path, top_n: int = 10) -> Dict[str, float] | None:
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
        "recall@10": _recall_at_k(labels, top_n),
        "ndcg@10": _ndcg_at_k(labels, top_n),
        "mrr@10": _mrr_at_k(labels, top_n),
    }


def _print_dynamic_output_metrics(
    output_dir: str | Path,
    top_n: int = 10,
    meta_map: Dict[str, Dict[str, Any]] | None = None,
    evaluator: Qwen3RankingEvaluator | None = None,
    eval_cache_path: Path | None = None,
) -> None:
    pattern = str(Path(output_dir) / "*_dynamic_reasoning_ranking_output.json")
    paths = [Path(p) for p in sorted(glob.glob(pattern))]
    if not paths:
        print(f"[Metrics] no ranking outputs found in {output_dir}")
        return

    metric_rows = []
    eval_rows = []
    eval_cache = _safe_json_load(eval_cache_path, {}) if eval_cache_path is not None else {}
    cache_updated = False
    for p in paths:
        row = _calc_metrics_from_dynamic_output(p, top_n=top_n)
        if row is not None:
            metric_rows.append(row)
        if meta_map is not None and evaluator is not None:
            cache_key = str(p.resolve())
            mtime = p.stat().st_mtime
            cached = eval_cache.get(cache_key) if isinstance(eval_cache, dict) else None
            if isinstance(cached, dict) and cached.get("mtime") == mtime:
                eval_row = cached.get("metrics")
            else:
                eval_row = _calc_llm_eval_from_dynamic_output(p, meta_map=meta_map, evaluator=evaluator, top_n=top_n)
                if eval_row is not None and eval_cache_path is not None:
                    eval_cache[cache_key] = {"mtime": mtime, "metrics": eval_row}
                    cache_updated = True
            if eval_row is not None:
                eval_rows.append(eval_row)

    if metric_rows:
        recall = float(np.mean([x["recall@10"] for x in metric_rows]))
        ndcg = float(np.mean([x["ndcg@10"] for x in metric_rows]))
        mrr = float(np.mean([x["mrr@10"] for x in metric_rows]))
        print(
            f"[Metrics][Aggregated@10] files={len(metric_rows)} "
            f"HitRate/Recall={recall:.6f} NDCG={ndcg:.6f} MRR={mrr:.6f}"
        )
    else:
        print(f"[Metrics] no valid ranking outputs with groundtruth target in {output_dir}")

    if eval_rows:
        eval_hit = float(np.mean([x["eval_hit@10"] for x in eval_rows]))
        eval_ndcg = float(np.mean([x["eval_ndcg@10"] for x in eval_rows]))
        eval_mrr = float(np.mean([x["eval_mrr@10"] for x in eval_rows]))
        print(
            f"[EvalMetrics][Qwen3-8B@10] files={len(eval_rows)} "
            f"HitRate={eval_hit:.6f} NDCG={eval_ndcg:.6f} MRR={eval_mrr:.6f}"
        )

    if cache_updated and eval_cache_path is not None:
        _save_json(eval_cache_path, eval_cache)


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

    meta_map = load_filtered_meta(Path(args.filtered_meta_jsonl))
    if not meta_map:
        raise ValueError(f"No items loaded from filtered meta: {args.filtered_meta_jsonl}")

    all_item_ids = sorted(meta_map.keys())
    item_id_to_index = {iid: idx for idx, iid in enumerate(all_item_ids)}
    title_lower_map = {iid: str(meta_map[iid].get("title", "") or "").lower() for iid in all_item_ids}

    cache_dir = Path(args.cache_dir)
    emb_cache_path = cache_dir / "agent3_item_embedding_cache.npz"
    text_cache_path = cache_dir / "agent3_text_cache.json"
    eval_cache_path = cache_dir / "agent3_eval_cache.json"

    text_cache = _safe_json_load(text_cache_path, {"items": {}, "queries": {}})
    item_sentence_cache: Dict[str, str] = text_cache.get("items", {})
    query_sentence_cache: Dict[str, str] = text_cache.get("queries", {})

    print(f"[Init] load embedding model: {args.embedding_model}")
    emb_model = SentenceTransformer(args.embedding_model)
    ranking_evaluator = Qwen3RankingEvaluator(model_name=args.eval_model)

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

        existing_output = Path(args.output_dir) / f"user_{user_id}_dynamic_reasoning_ranking_output.json"
        if _has_non_empty_ranked_items(existing_output):
            print(f"[UserLoop] skip user={user_id}: existing non-empty ranking output found at {existing_output}")
            _print_dynamic_output_metrics(args.output_dir, top_n=10, meta_map=meta_map, evaluator=ranking_evaluator, eval_cache_path=eval_cache_path)
            continue
        if existing_output.exists():
            print(f"[UserLoop] user={user_id} has empty ranked_items output, retry Agent3 recall before deciding skip")

        routed = _route_query(query, category_catalog, args.enable_llm_routing, args.text_model)

        q_sentence = _query_sentence(query, routed["selected_category_paths"], routed["rewritten_query"])
        query_sentence_cache[f"{user_id}::{q_sentence}"] = q_sentence

        filtered_item_ids = _filter_item_ids_by_categories(
            candidate_item_ids=all_item_ids,
            meta_map=meta_map,
            selected_categories=routed.get("selected_category_paths", []) or [],
        )
        print(f"[Agent3][categories] exact_match_count={len(filtered_item_ids)}")

        if not filtered_item_ids:
            print("[Agent3] category exact-match prefilter found 0 items. fallback to embedding top 500 recall.")
            filtered_item_ids = list(all_item_ids)

        filtered_idx = [item_id_to_index[iid] for iid in filtered_item_ids]
        filtered_emb = item_emb_norm[np.array(filtered_idx)]

        q_emb = _encode_texts(emb_model, [q_sentence], batch_size=1, prompt_name="query").astype(np.float32, copy=False)
        q_emb_norm = q_emb / np.clip(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12, None)
        sim_matrix = np.matmul(filtered_emb, q_emb_norm[0])
        rank_indices = np.argsort(-sim_matrix)

        keywords = _extract_query_keywords(query, max_keywords=args.max_query_keywords)
        top_ids, used_k, kw_debug = _build_hybrid_recall_ids(
            all_item_ids=filtered_item_ids,
            title_lower_map=title_lower_map,
            keywords=keywords,
            rank_indices=rank_indices,
            recall_topk=args.fixed_recall_topk,
        )
        print(
            f"[Agent3][keyword] keywords={kw_debug['keywords']} matched={kw_debug['keyword_matched_count']} "
            f"stage={kw_debug['keyword_stage']} prefilter_size={len(filtered_item_ids)}"
        )

        hit = target_id in top_ids
        if not hit:
            print("[Agent3] initial fixed-top recall missed target. fallback to embedding top 500 recall.")
            top_ids = [filtered_item_ids[int(idx)] for idx in rank_indices[: min(len(rank_indices), 500)]]
            used_k = len(top_ids)
            hit = target_id in top_ids
            kw_debug["fallback_stage"] = "embedding_top_500"

        if hit:
            print(f"[Agent3] recall hit at k={used_k}; run Agent1/2")
        else:
            print("[Agent3] recall still missed target after embedding top 500 fallback, but continue running Agent1/2/4/5.")

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
            )
            ranked_first = module3_out.ranked_items[0]["item_id"] if module3_out.ranked_items else ""
        else:
            print("[Agent4/5] skipped by --disable-agent45")

        results.append({
            "user_id": user_id,
            "target_id": target_id,
            "hit": int(hit),
            "used_k": used_k,
            "top1": ranked_first,
            "kw_debug": kw_debug,
        })
        _print_dynamic_output_metrics(args.output_dir, top_n=10, meta_map=meta_map, evaluator=ranking_evaluator, eval_cache_path=eval_cache_path)

    text_cache["items"] = item_sentence_cache
    text_cache["queries"] = query_sentence_cache
    _save_json(text_cache_path, text_cache)
    _save_json(Path(args.output_dir) / "unified_eval_results.json", results)

    recall_rate = float(np.mean([r["hit"] for r in results])) if results else 0.0
    summary = {"rows": len(results), "recall@k": recall_rate, "output_dir": args.output_dir}
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
    parser.add_argument("--fixed-recall-topk", type=int, default=250)
    parser.add_argument("--max-query-keywords", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-users", type=int, default=0, help="仅跑前N条query，0表示全量")

    parser.add_argument("--cache-dir", default="processed/beauty_cache")
    parser.add_argument("--output-dir", default="processed/beauty_unified_outputs")
    parser.add_argument("--global-db", default="processed/beauty_global_item_features.db")
    parser.add_argument("--history-db", default="processed/beauty_user_history.db")

    parser.add_argument("--vl-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--text-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--eval-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--enable-llm-routing", action="store_true", help="开启Qwen3文本路由；默认关闭走规则fallback")
    parser.add_argument("--enable-vl-profiling", action="store_true", help="开启Qwen3-VL画像；默认关闭走轻量画像")
    parser.add_argument("--disable-agent45", action="store_true", help="关闭Agent4/5")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.enable_agent45 = not bool(args.disable_agent45)
    run(args)
