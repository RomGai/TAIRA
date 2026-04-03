"""Intent understanding and dual-recall module (Agent 3) for Amazon pipeline.

This module directly connects to outputs of `item_profiler_agents.py`:
- Global item DB: `global_item_features(item_id, profile_json, updated_at)`
- User history DB: `user_history_profiles(user_id, item_id, behavior, timestamp, profile_json, created_at)`

Agent 3 (Routing & Recall Agent - LLM) responsibilities:
1) Parse real-time query and map it to categories/item types.
2) Route A: recall candidate items from global DB with dynamic hierarchical roll-up.
3) Route B: recall query-relevant user history records from history DB.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import AutoModel
    except Exception:  # pragma: no cover
        AutoModel = None
except Exception:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    AutoModel = None


@dataclass
class RoutingResult:
    query: str
    category_paths: List[List[str]]
    item_types: List[str]
    reasoning: str


@dataclass
class IntentDualRecallOutput:
    query: str
    user_id: str
    routing: Dict[str, Any]
    candidate_items: List[Dict[str, Any]]
    query_relevant_history: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sanitize_for_filename(value: str) -> str:
    safe = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)]
    return "".join(safe).strip("_") or "unknown"


def _build_output_file_path(
    user_id: str,
    query: str,
    output_dir: str | Path = "./processed/intent_dual_recall_outputs",
) -> Path:
    query_tag = _sanitize_for_filename((query or "no_query")[:40])
    filename = f"user_{_sanitize_for_filename(user_id)}_{query_tag}_intent_dual_recall_output.json"
    return Path(output_dir) / filename


class Qwen3RouterLLM:
    """Qwen3 (text-only) wrapper following official usage style."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        max_new_tokens: int = 2048,
        enable_thinking: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
            raise ImportError("transformers/torch are not available for Qwen3RouterLLM.")
        if self._model is not None and self._tokenizer is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    @staticmethod
    def _try_json_decode(text: str) -> Optional[Dict[str, Any]]:
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

    def route(
        self,
        query: str,
        category_catalog: Sequence[str],
        item_type_catalog: Sequence[str],
    ) -> RoutingResult:
        self.load()

        catalog_text = "\n".join(f"- {c}" for c in category_catalog[:300])
        item_type_text = "\n".join(f"- {i}" for i in item_type_catalog[:300])
        prompt = (
            "你是电商检索路由专家。请把用户Query映射到给定类目/类型；若都不匹配，可新造一个合理类目。\n"
            "输出必须是一个JSON对象，字段:"
            "category_paths(二维数组，每条是层级路径), item_types(数组), reasoning(字符串)。\n\n"
            f"用户Query: {query}\n\n"
            "候选类目路径清单:\n"
            f"{catalog_text}\n\n"
            "候选item_type清单:\n"
            f"{item_type_text}\n"
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
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # Parse thinking content boundary token (official pattern).
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content = self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        payload = self._try_json_decode(content)
        if payload is None:
            payload = {
                "category_paths": [],
                "item_types": [],
                "reasoning": f"Failed to parse JSON from LLM output: {content[:500]}",
            }

        raw_paths = payload.get("category_paths", [])
        category_paths: List[List[str]] = []
        for p in raw_paths:
            if isinstance(p, list):
                segs = [str(x).strip() for x in p if str(x).strip()]
            else:
                segs = [x.strip() for x in str(p).replace("/", ">").split(">") if x.strip()]
            if segs:
                category_paths.append(segs)

        item_types = [str(x).strip() for x in payload.get("item_types", []) if str(x).strip()]
        return RoutingResult(
            query=query,
            category_paths=category_paths,
            item_types=item_types,
            reasoning=str(payload.get("reasoning", "")),
        )


class Qwen3QueryEmbeddingModel:
    """Query embedding wrapper for semantic history recall."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        if AutoTokenizer is None or AutoModel is None or torch is None:
            raise ImportError("transformers/torch are not available for Qwen3QueryEmbeddingModel.")
        if self._tokenizer is not None and self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto")

    def encode(self, text: str) -> List[float]:
        self.load()
        clean_text = (text or "").strip() or "empty_query"
        inputs = self._tokenizer([clean_text], padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized[0].detach().cpu().tolist()


class GlobalHistoryAccessor:
    """Read-only accessor over module-1 output databases."""

    def __init__(self, global_db_path: str | Path, history_db_path: str | Path) -> None:
        self.global_conn = sqlite3.connect(str(global_db_path))
        self.history_conn = sqlite3.connect(str(history_db_path))
        self.global_conn.row_factory = sqlite3.Row
        self.history_conn.row_factory = sqlite3.Row

    @staticmethod
    def _extract_taxonomy(profile: Dict[str, Any]) -> Tuple[List[str], str]:
        taxonomy = profile.get("taxonomy", {}) if isinstance(profile, dict) else {}
        path = taxonomy.get("category_path", [])
        if not isinstance(path, list):
            path = []
        clean_path = [str(x).strip() for x in path if str(x).strip()]
        item_type = str(taxonomy.get("item_type", "")).strip()
        return clean_path, item_type

    @staticmethod
    def _extract_item_embedding(profile: Dict[str, Any]) -> Optional[List[float]]:
        if not isinstance(profile, dict):
            return None

        candidate_keys = [
            "embedding",
            "dense_embedding",
            "semantic_embedding",
            "item_embedding",
        ]
        for key in candidate_keys:
            vec = profile.get(key)
            if isinstance(vec, list) and vec:
                try:
                    return [float(x) for x in vec]
                except (TypeError, ValueError):
                    continue

        emb_group = profile.get("embeddings", {})
        if isinstance(emb_group, dict):
            for key in ["item", "text", "semantic", "dense"]:
                vec = emb_group.get(key)
                if isinstance(vec, list) and vec:
                    try:
                        return [float(x) for x in vec]
                    except (TypeError, ValueError):
                        continue
        return None

    @staticmethod
    def _extract_text_embedding(profile: Dict[str, Any]) -> Optional[List[float]]:
        if not isinstance(profile, dict):
            return None
        candidate_keys = [
            "text_embedding",
            "query_text_embedding",
            "semantic_embedding",
            "dense_embedding",
            "embedding",
        ]
        for key in candidate_keys:
            vec = profile.get(key)
            if isinstance(vec, list) and vec:
                try:
                    return [float(x) for x in vec]
                except (TypeError, ValueError):
                    continue

        emb_group = profile.get("embeddings", {})
        if isinstance(emb_group, dict):
            for key in ["text", "semantic", "dense", "item"]:
                vec = emb_group.get(key)
                if isinstance(vec, list) and vec:
                    try:
                        return [float(x) for x in vec]
                    except (TypeError, ValueError):
                        continue
        return None

    @staticmethod
    def _extract_vl_embedding(profile: Dict[str, Any]) -> Optional[List[float]]:
        if not isinstance(profile, dict):
            return None
        candidate_keys = [
            "vl_embedding",
            "visual_embedding",
            "image_embedding",
            "multimodal_embedding",
        ]
        for key in candidate_keys:
            vec = profile.get(key)
            if isinstance(vec, list) and vec:
                try:
                    return [float(x) for x in vec]
                except (TypeError, ValueError):
                    continue

        emb_group = profile.get("embeddings", {})
        if isinstance(emb_group, dict):
            for key in ["vl", "visual", "image", "multimodal"]:
                vec = emb_group.get(key)
                if isinstance(vec, list) and vec:
                    try:
                        return [float(x) for x in vec]
                    except (TypeError, ValueError):
                        continue
        return None

    @staticmethod
    def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return -1.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a, b in zip(vec_a, vec_b):
            af = float(a)
            bf = float(b)
            dot += af * bf
            norm_a += af * af
            norm_b += bf * bf
        if norm_a <= 0 or norm_b <= 0:
            return -1.0
        return float(dot / (math.sqrt(norm_a) * math.sqrt(norm_b)))

    def category_catalog(self) -> Tuple[List[str], List[str]]:
        categories: set[str] = set()
        item_types: set[str] = set()
        rows = self.global_conn.execute("SELECT profile_json FROM global_item_features").fetchall()
        for row in rows:
            profile = json.loads(row["profile_json"])
            path, item_type = self._extract_taxonomy(profile)
            if path:
                categories.add(" > ".join(path))
            if item_type:
                item_types.add(item_type)
        return sorted(categories), sorted(item_types)

    @staticmethod
    def _is_relevant(
        profile: Dict[str, Any],
        target_paths: Sequence[Sequence[str]],
        target_item_types: Sequence[str],
    ) -> bool:
        path, item_type = GlobalHistoryAccessor._extract_taxonomy(profile)
        path_lower = [x.lower() for x in path]
        type_lower = item_type.lower()

        for tp in target_paths:
            tp_lower = [x.lower() for x in tp]
            if tp_lower and len(path_lower) >= len(tp_lower) and path_lower[: len(tp_lower)] == tp_lower:
                return True
            if tp_lower and tp_lower == path_lower:
                return True

        for t in target_item_types:
            if t.lower() and t.lower() == type_lower:
                return True
        return False

    def recall_global_items(
        self,
        target_paths: Sequence[Sequence[str]],
        target_item_types: Sequence[str],
        min_items: int = 20,
        max_items: int = 200,
    ) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
        rows = self.global_conn.execute(
            "SELECT item_id, profile_json, updated_at FROM global_item_features"
        ).fetchall()

        all_items = []
        for row in rows:
            profile = json.loads(row["profile_json"])
            all_items.append(
                {
                    "item_id": row["item_id"],
                    "profile": profile,
                    "updated_at": row["updated_at"],
                }
            )

        rollup_paths = [list(p) for p in target_paths]
        recalled: List[Dict[str, Any]] = []
        seen_item_ids: set[str] = set()

        def add_matches(paths: Sequence[Sequence[str]]) -> None:
            nonlocal recalled
            for item in all_items:
                if item["item_id"] in seen_item_ids:
                    continue
                if self._is_relevant(item["profile"], paths, target_item_types):
                    recalled.append(item)
                    seen_item_ids.add(item["item_id"])
                    if len(recalled) >= max_items:
                        return

        add_matches(rollup_paths)
        while len(recalled) < min_items:
            rolled = []
            for p in rollup_paths:
                if len(p) > 1:
                    rolled.append(p[:-1])
            rolled = [p for i, p in enumerate(rolled) if p and p not in rolled[:i]]
            if not rolled:
                break
            rollup_paths = rolled
            add_matches(rollup_paths)
            if len(recalled) >= max_items:
                break

        return recalled[:max_items], rollup_paths

    def fetch_global_items_by_ids(
        self,
        item_ids: Sequence[str],
        max_items: int = 200,
    ) -> List[Dict[str, Any]]:
        """Fetch global catalog items by provided item-id order."""
        if not item_ids:
            return []

        rows = self.global_conn.execute(
            "SELECT item_id, profile_json, updated_at FROM global_item_features"
        ).fetchall()
        row_map: Dict[str, Any] = {str(r["item_id"]): r for r in rows}

        out: List[Dict[str, Any]] = []
        for item_id in item_ids:
            row = row_map.get(str(item_id))
            if row is None:
                continue
            out.append(
                {
                    "item_id": str(row["item_id"]),
                    "profile": json.loads(row["profile_json"]),
                    "updated_at": row["updated_at"],
                }
            )
            if len(out) >= max_items:
                break
        return out

    def fetch_all_global_items(self) -> List[Dict[str, Any]]:
        rows = self.global_conn.execute(
            "SELECT item_id, profile_json, updated_at FROM global_item_features"
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "item_id": str(row["item_id"]),
                    "profile": json.loads(row["profile_json"]),
                    "updated_at": row["updated_at"],
                }
            )
        return out

    def recall_by_embedding(
        self,
        query_embedding: Sequence[float],
        embedding_type: str = "text",
        item_scope_ids: Optional[set[str]] = None,
        top_k: int = 200,
    ) -> List[Dict[str, Any]]:
        all_items = self.fetch_all_global_items()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in all_items:
            item_id = str(item.get("item_id", "")).strip()
            if not item_id:
                continue
            if item_scope_ids is not None and item_id not in item_scope_ids:
                continue
            profile = item.get("profile", {}) or {}
            if embedding_type == "vl":
                emb = self._extract_vl_embedding(profile)
            else:
                emb = self._extract_text_embedding(profile)
            if not emb:
                continue
            sim = self._cosine_similarity(query_embedding, emb)
            if sim <= -1.0:
                continue
            scored.append((sim, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for sim, item in scored[: max(1, int(top_k))]:
            payload = dict(item)
            payload["semantic_similarity"] = float(sim)
            payload["embedding_type"] = embedding_type
            out.append(payload)
        return out

    def recall_user_history(
        self,
        user_id: str,
        target_paths: Sequence[Sequence[str]],
        target_item_types: Sequence[str],
        max_rows: int = 300,
    ) -> List[Dict[str, Any]]:
        rows = self.history_conn.execute(
            """
            SELECT user_id, item_id, behavior, timestamp, profile_json, created_at
            FROM user_history_profiles
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (str(user_id), int(max_rows) * 3),
        ).fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            profile = json.loads(row["profile_json"])
            if not self._is_relevant(profile, target_paths, target_item_types):
                continue
            results.append(
                {
                    "user_id": row["user_id"],
                    "item_id": row["item_id"],
                    "behavior": row["behavior"],
                    "timestamp": row["timestamp"],
                    "profile": profile,
                    "created_at": row["created_at"],
                }
            )
            if len(results) >= max_rows:
                break
        return results

    def recall_user_history_by_query_embedding(
        self,
        user_id: str,
        query_embedding: Sequence[float],
        top_k: int = 20,
        max_rows: int = 300,
    ) -> List[Dict[str, Any]]:
        """Semantic recall over user history using global item embeddings only."""
        rows = self.history_conn.execute(
            """
            SELECT user_id, item_id, behavior, timestamp, profile_json, created_at
            FROM user_history_profiles
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (str(user_id), int(max_rows)),
        ).fetchall()
        if not rows:
            return []

        latest_row_by_item: Dict[str, sqlite3.Row] = {}
        for row in rows:
            item_id = str(row["item_id"]).strip()
            if not item_id or item_id in latest_row_by_item:
                continue
            latest_row_by_item[item_id] = row

        item_ids = list(latest_row_by_item.keys())
        if not item_ids:
            return []

        global_rows = self.global_conn.execute(
            "SELECT item_id, profile_json FROM global_item_features"
        ).fetchall()
        global_profile_map: Dict[str, Dict[str, Any]] = {}
        for row in global_rows:
            iid = str(row["item_id"]).strip()
            if iid in latest_row_by_item:
                try:
                    global_profile_map[iid] = json.loads(row["profile_json"])
                except json.JSONDecodeError:
                    continue

        scored: List[Tuple[float, str]] = []
        for item_id in item_ids:
            global_profile = global_profile_map.get(item_id)
            if not global_profile:
                continue
            item_emb = self._extract_item_embedding(global_profile)
            if not item_emb:
                continue
            sim = self._cosine_similarity(query_embedding, item_emb)
            scored.append((sim, item_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked_item_ids = [iid for _, iid in scored[: max(1, int(top_k))]]

        results: List[Dict[str, Any]] = []
        for item_id in picked_item_ids:
            row = latest_row_by_item[item_id]
            profile = json.loads(row["profile_json"])
            results.append(
                {
                    "user_id": row["user_id"],
                    "item_id": row["item_id"],
                    "behavior": row["behavior"],
                    "timestamp": row["timestamp"],
                    "profile": profile,
                    "created_at": row["created_at"],
                    "semantic_similarity": next((s for s, iid in scored if iid == item_id), None),
                    "embedding_source": "global_item_features",
                }
            )
        return results

    def recall_user_history_all(
        self,
        user_id: str,
        max_rows: int = 300,
    ) -> List[Dict[str, Any]]:
        """Return all recent history rows for the user (no relevance filtering)."""
        rows = self.history_conn.execute(
            """
            SELECT user_id, item_id, behavior, timestamp, profile_json, created_at
            FROM user_history_profiles
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (str(user_id), int(max_rows)),
        ).fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            profile = json.loads(row["profile_json"])
            results.append(
                {
                    "user_id": row["user_id"],
                    "item_id": row["item_id"],
                    "behavior": row["behavior"],
                    "timestamp": row["timestamp"],
                    "profile": profile,
                    "created_at": row["created_at"],
                }
            )
        return results

    def user_seen_item_ids(
        self,
        user_id: str,
        lookback: int = 5000,
    ) -> set[str]:
        """Return deduplicated item_ids from the user's full raw history sequence."""
        rows = self.history_conn.execute(
            """
            SELECT item_id
            FROM user_history_profiles
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (str(user_id), int(lookback)),
        ).fetchall()
        return {str(r["item_id"]) for r in rows if str(r["item_id"]).strip()}

    def _top_item_types_from_history(
        self,
        user_id: str,
        top_k: int = 3,
        lookback: int = 300,
    ) -> List[str]:
        """Infer top-k interested item types from recent history."""
        rows = self.history_conn.execute(
            """
            SELECT profile_json
            FROM user_history_profiles
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (str(user_id), int(lookback)),
        ).fetchall()
        type_cnt: Dict[str, int] = {}
        for row in rows:
            try:
                profile = json.loads(row["profile_json"])
            except json.JSONDecodeError:
                continue
            _path, item_type = self._extract_taxonomy(profile)
            if item_type:
                type_cnt[item_type] = type_cnt.get(item_type, 0) + 1
        ranked = sorted(type_cnt.items(), key=lambda x: (-x[1], x[0]))
        return [t for t, _ in ranked[: max(1, int(top_k))]]

    def infer_user_intent_from_history(
        self,
        user_id: str,
        lookback: int = 200,
        min_positive_first: bool = True,
        top_category_paths_k: int = 3,
        top_item_types_k: int = 3,
    ) -> RoutingResult:
        """Infer category intent from user history when query is empty.

        Strategy:
        1) Prefer recent positive interactions (if available)
        2) Fallback to all recent interactions
        3) Aggregate frequent taxonomy.category_path / taxonomy.item_type
        """
        rows = self.history_conn.execute(
            """
            SELECT behavior, timestamp, profile_json
            FROM user_history_profiles
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (str(user_id), int(lookback)),
        ).fetchall()

        if not rows:
            return RoutingResult(
                query="",
                category_paths=[],
                item_types=[],
                reasoning="No query and no history found; cannot infer user intent.",
            )

        scoped_rows = rows
        if min_positive_first:
            positives = [r for r in rows if str(r["behavior"]).lower() == "positive"]
            if positives:
                scoped_rows = positives

        cat_cnt: Dict[str, int] = {}
        type_cnt: Dict[str, int] = {}
        for r in scoped_rows:
            try:
                profile = json.loads(r["profile_json"])
            except json.JSONDecodeError:
                continue
            path, item_type = self._extract_taxonomy(profile)
            if path:
                key = " > ".join(path)
                cat_cnt[key] = cat_cnt.get(key, 0) + 1
            if item_type:
                type_cnt[item_type] = type_cnt.get(item_type, 0) + 1

        top_cats = sorted(cat_cnt.items(), key=lambda x: (-x[1], x[0]))[
            : max(1, int(top_category_paths_k))
        ]
        top_types = sorted(type_cnt.items(), key=lambda x: (-x[1], x[0]))[: max(1, int(top_item_types_k))]
        paths = [[seg.strip() for seg in cat.split(">") if seg.strip()] for cat, _ in top_cats]
        item_types = [t for t, _ in top_types]

        reason_scope = "positive-only" if scoped_rows is not rows else "all-recent"
        return RoutingResult(
            query="",
            category_paths=paths,
            item_types=item_types,
            reasoning=(
                "Query is empty; inferred intent from "
                f"user history ({reason_scope}, samples={len(scoped_rows)})."
            ),
        )


class RoutingRecallAgent:
    """Agent 3: routing + dual recall."""

    def __init__(
        self,
        llm: Qwen3RouterLLM,
        accessor: GlobalHistoryAccessor,
        query_embedding_model: Optional[Qwen3QueryEmbeddingModel] = None,
        vl_query_embedding_model: Optional[Qwen3QueryEmbeddingModel] = None,
    ) -> None:
        self.llm = llm
        self.accessor = accessor
        self.query_embedding_model = query_embedding_model
        self.vl_query_embedding_model = vl_query_embedding_model or query_embedding_model

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(v)))

    @staticmethod
    def _rank_of_item(items: Sequence[Dict[str, Any]], target_item_id: str) -> int:
        target = str(target_item_id or "").strip()
        if not target:
            return 10**9
        for idx, row in enumerate(items, start=1):
            if str(row.get("item_id", "")).strip() == target:
                return idx
        return 10**9

    def _generate_pseudo_queries(
        self,
        query: str,
        history_rows: Sequence[Dict[str, Any]],
        max_pseudo_queries: int = 10,
    ) -> List[Dict[str, str]]:
        pseudo_queries: List[Dict[str, str]] = []
        for row in history_rows:
            item_id = str(row.get("item_id", "")).strip()
            if not item_id:
                continue
            profile = row.get("profile", {}) if isinstance(row.get("profile", {}), dict) else {}
            taxonomy = profile.get("taxonomy", {}) if isinstance(profile, dict) else {}
            item_type = str(taxonomy.get("item_type", "")).strip()
            category = " > ".join(
                [str(x).strip() for x in taxonomy.get("category_path", []) if str(x).strip()]
            )
            text_tags = profile.get("text_tags", []) if isinstance(profile, dict) else []
            text_hint = ""
            if isinstance(text_tags, list) and text_tags:
                text_hint = "，".join(str(x).strip() for x in text_tags[:3] if str(x).strip())
            pseudo = query
            if category or item_type or text_hint:
                pseudo = f"{query}（参考历史偏好：{item_type} {category} {text_hint}）".strip()
            pseudo_queries.append({"item_id": item_id, "pseudo_query": pseudo})
            if len(pseudo_queries) >= max(1, int(max_pseudo_queries)):
                break
        return pseudo_queries

    def _adaptive_weighted_embedding_recall(
        self,
        user_id: str,
        query: str,
        item_scope_ids: Optional[set[str]] = None,
        max_total_recall: int = 500,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        def _safe_rank(rank: int) -> int:
            return int(rank if rank < 10**9 else 5000)

        def _rank_strength(rank: int) -> float:
            safe = _safe_rank(rank)
            return 1.0 / math.log2(safe + 2.0)

        def _estimate_total_recall(trace: List[Dict[str, Any]], hard_cap: int) -> int:
            if not trace:
                return int(self._clamp(hard_cap, 50, 500))
            required: List[float] = []
            for row in trace:
                t_rank = _safe_rank(int(row.get("text_rank", 5000)))
                v_rank = _safe_rank(int(row.get("vl_rank", 5000)))
                t_w = float(row.get("weights", {}).get("text", 0.5))
                v_w = float(row.get("weights", {}).get("vl", 0.5))
                low_rank = min(t_rank, v_rank)
                high_rank = max(t_rank, v_rank)
                dominance = abs(t_w - v_w)
                required.append(low_rank * (1.15 + 0.35 * dominance) + 0.15 * high_rank)
            required.sort()
            pivot = required[int(0.75 * (len(required) - 1))]
            return int(self._clamp(round(pivot), 50, min(500, hard_cap)))

        def _agent_finalize_params(
            trace: List[Dict[str, Any]],
            cur_text_weight: float,
            cur_vl_weight: float,
            cur_total_recall: int,
        ) -> Dict[str, Any]:
            if not trace:
                return {
                    "text_weight": round(cur_text_weight, 4),
                    "vl_weight": round(cur_vl_weight, 4),
                    "recall_size": int(cur_total_recall),
                    "mode": "fallback",
                    "reasoning": "no_history",
                }
            window = trace[-min(6, len(trace)) :]
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
            final_text = float(self._clamp(final_text, 0.05, 0.95))
            final_vl = 1.0 - final_text
            return {
                "text_weight": round(final_text, 4),
                "vl_weight": round(final_vl, 4),
                "recall_size": int(cur_total_recall),
                "mode": "history_agent_update",
                "reasoning": (
                    f"trend_text={round(trend_text, 3)}, text_vote={round(text_vote, 3)}, "
                    f"vl_vote={round(vl_vote, 3)}, margin={round(vote_margin, 3)}, "
                    f"switches={switch_count}, stable={stable}"
                ),
            }

        memory: List[Dict[str, Any]] = []
        text_weight = 0.5
        vl_weight = 0.5
        history_target_text = 0.5
        total_recall = min(500, max(50, int(max_total_recall)))

        if not (query or "").strip() or self.query_embedding_model is None:
            return [], {
                "enabled": False,
                "reason": "query_empty_or_text_embedding_model_missing",
                "memory": memory,
                "text_weight": text_weight,
                "vl_weight": vl_weight,
                "total_recall": total_recall,
            }

        history_rows = self.accessor.recall_user_history_all(user_id=user_id, max_rows=200)
        positive_rows = [r for r in history_rows if str(r.get("behavior", "")).lower() == "positive"]
        pseudo_inputs = positive_rows if positive_rows else history_rows
        pseudo_queries = self._generate_pseudo_queries(query=query, history_rows=pseudo_inputs, max_pseudo_queries=8)

        if not pseudo_queries:
            return [], {
                "enabled": False,
                "reason": "no_history_for_pseudo_queries",
                "memory": memory,
                "text_weight": text_weight,
                "vl_weight": vl_weight,
                "total_recall": total_recall,
            }

        for step, pseudo in enumerate(pseudo_queries, start=1):
            pq = str(pseudo.get("pseudo_query", "")).strip() or query
            target_item_id = str(pseudo.get("item_id", "")).strip()
            try:
                text_q_emb = self.query_embedding_model.encode(pq)
            except Exception:
                continue

            vl_model = self.vl_query_embedding_model or self.query_embedding_model
            try:
                vl_q_emb = vl_model.encode(pq)
            except Exception:
                vl_q_emb = text_q_emb

            text_ranked = self.accessor.recall_by_embedding(
                query_embedding=text_q_emb,
                embedding_type="text",
                item_scope_ids=item_scope_ids,
                top_k=max(500, total_recall),
            )
            vl_ranked = self.accessor.recall_by_embedding(
                query_embedding=vl_q_emb,
                embedding_type="vl",
                item_scope_ids=item_scope_ids,
                top_k=max(500, total_recall),
            )
            text_rank = self._rank_of_item(text_ranked, target_item_id)
            vl_rank = self._rank_of_item(vl_ranked, target_item_id)
            text_strength = _rank_strength(text_rank)
            vl_strength = _rank_strength(vl_rank)
            strength_sum = max(1e-8, text_strength + vl_strength)
            strength_target_text = text_strength / strength_sum
            low_rank = min(_safe_rank(text_rank), _safe_rank(vl_rank))
            gap = abs(_safe_rank(text_rank) - _safe_rank(vl_rank))
            confidence = min(1.0, abs(math.log((_safe_rank(vl_rank) + 1) / (_safe_rank(text_rank) + 1))) / 1.6)
            cover_share = min(0.95, max(0.5, low_rank / max(1.0, float(total_recall))))
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
            text_weight = self._clamp(text_weight + step_delta, 0.05, 0.95)
            vl_weight = 1.0 - text_weight

            memory.append(
                {
                    "step": step,
                    "target_item_id": target_item_id,
                    "pseudo_query": pq,
                    "text_rank": text_rank,
                    "vl_rank": vl_rank,
                    "weights": {"text": round(text_weight, 4), "vl": round(vl_weight, 4)},
                    "summary": (
                        f"strength_target={round(strength_target_text, 3)}, "
                        f"dominant_share={round(dominant_share, 3)}, "
                        f"confidence={round(confidence, 3)}, "
                        f"history_target={round(history_target_text, 3)}"
                    ),
                }
            )
            total_recall = _estimate_total_recall(memory, int(max_total_recall))
            memory[-1]["estimated_total_recall"] = int(total_recall)

        agent_final_params = _agent_finalize_params(memory, text_weight, vl_weight, total_recall)
        text_weight = float(agent_final_params["text_weight"])
        vl_weight = float(agent_final_params["vl_weight"])
        final_modal_weights = {"text": round(text_weight, 4), "vl": round(vl_weight, 4)}
        final_modal_ratio = f"{int(round(text_weight * 100))}:{int(round(vl_weight * 100))}"
        query_text_emb = self.query_embedding_model.encode(query)
        vl_model = self.vl_query_embedding_model or self.query_embedding_model
        try:
            query_vl_emb = vl_model.encode(query)
        except Exception:
            query_vl_emb = query_text_emb

        text_k = max(1, int(round(total_recall * text_weight)))
        vl_k = max(1, int(round(total_recall * vl_weight)))
        text_ranked = self.accessor.recall_by_embedding(
            query_embedding=query_text_emb,
            embedding_type="text",
            item_scope_ids=item_scope_ids,
            top_k=text_k,
        )
        vl_ranked = self.accessor.recall_by_embedding(
            query_embedding=query_vl_emb,
            embedding_type="vl",
            item_scope_ids=item_scope_ids,
            top_k=vl_k,
        )

        merged_scores: Dict[str, Dict[str, Any]] = {}
        for idx, item in enumerate(text_ranked, start=1):
            iid = str(item.get("item_id", "")).strip()
            if not iid:
                continue
            weight_score = text_weight * (1.0 / float(idx))
            merged_scores[iid] = {"item": item, "score": weight_score}
        for idx, item in enumerate(vl_ranked, start=1):
            iid = str(item.get("item_id", "")).strip()
            if not iid:
                continue
            weight_score = vl_weight * (1.0 / float(idx))
            if iid not in merged_scores:
                merged_scores[iid] = {"item": item, "score": weight_score}
            else:
                merged_scores[iid]["score"] += weight_score

        ranked_items = [v["item"] for v in sorted(merged_scores.values(), key=lambda x: x["score"], reverse=True)]
        return ranked_items[: min(500, total_recall)], {
            "enabled": True,
            "text_weight": round(text_weight, 4),
            "vl_weight": round(vl_weight, 4),
            "final_modal_weights": final_modal_weights,
            "final_modal_ratio": final_modal_ratio,
            "total_recall": int(min(500, total_recall)),
            "agent_final_params": agent_final_params,
            "memory": memory,
            "pseudo_query_count": len(pseudo_queries),
        }

    def run(
        self,
        user_id: str,
        query: str,
        min_candidate_items: int = 20,
        max_candidate_items: int = 200,
        max_history_rows: int = 200,
        semantic_history_top_k: int = 20,
        history_category_paths_k: int = 3,
        query_category_paths_k: int = 3,
        interested_item_types_k: int = 3,
        exclude_seen_items: bool = True,
        seen_history_lookback: int = 5000,
        filter_candidates_by_item_type: bool = True,
        candidate_item_ids_scope: Optional[Sequence[str]] = None,
        adaptive_embedding_max_total_recall: int = 500,
        save_output: bool = True,
        output_dir: str | Path = "./processed/intent_dual_recall_outputs",
    ) -> IntentDualRecallOutput:
        clean_query = (query or "").strip()
        category_catalog, item_type_catalog = self.accessor.category_catalog()
        if clean_query:
            routing = self.llm.route(clean_query, category_catalog, item_type_catalog)
        else:
            routing = self.accessor.infer_user_intent_from_history(
                user_id=user_id,
                top_category_paths_k=history_category_paths_k,
                top_item_types_k=interested_item_types_k,
            )

        if clean_query:
            routing.category_paths = routing.category_paths[: max(1, int(query_category_paths_k))]

        history_top_item_types = self.accessor._top_item_types_from_history(
            user_id=user_id,
            top_k=interested_item_types_k,
        )

        merged_item_types: List[str] = []
        for t in [*routing.item_types, *history_top_item_types]:
            if t and t not in merged_item_types:
                merged_item_types.append(t)
        routing.item_types = merged_item_types[: max(1, int(interested_item_types_k))]

        if not routing.category_paths and routing.item_types:
            routing.category_paths = [[routing.item_types[0]]]

        scope_ids = {str(x).strip() for x in (candidate_item_ids_scope or []) if str(x).strip()}

        if filter_candidates_by_item_type:
            candidate_items, final_rollup_paths = self.accessor.recall_global_items(
                routing.category_paths,
                routing.item_types,
                min_items=min_candidate_items,
                max_items=max_candidate_items,
            )
        else:
            candidate_items = self.accessor.fetch_global_items_by_ids(
                item_ids=list(candidate_item_ids_scope or []),
                max_items=max_candidate_items,
            )
            final_rollup_paths = [list(p) for p in routing.category_paths]

        adaptive_candidates, adaptive_state = self._adaptive_weighted_embedding_recall(
            user_id=str(user_id),
            query=clean_query,
            item_scope_ids=(scope_ids if scope_ids else None),
            max_total_recall=min(500, int(adaptive_embedding_max_total_recall)),
        )
        if adaptive_candidates:
            merged = []
            seen = set()
            for item in [*candidate_items, *adaptive_candidates]:
                iid = str(item.get("item_id", "")).strip()
                if not iid or iid in seen:
                    continue
                seen.add(iid)
                merged.append(item)
            candidate_items = merged

        if clean_query:
            history_rows: List[Dict[str, Any]] = []
            semantic_error = ""
            if self.query_embedding_model is not None:
                try:
                    query_embedding = self.query_embedding_model.encode(clean_query)
                    history_rows = self.accessor.recall_user_history_by_query_embedding(
                        user_id=user_id,
                        query_embedding=query_embedding,
                        top_k=semantic_history_top_k,
                        max_rows=max_history_rows,
                    )
                except Exception as exc:  # pragma: no cover
                    semantic_error = str(exc)

            if not history_rows:
                history_rows = self.accessor.recall_user_history(
                    user_id=user_id,
                    target_paths=final_rollup_paths,
                    target_item_types=routing.item_types,
                    max_rows=max_history_rows,
                )
            if semantic_error:
                routing.reasoning = f"{routing.reasoning} | semantic_history_fallback={semantic_error}"
        else:
            history_rows = self.accessor.recall_user_history_all(
                user_id=user_id,
                max_rows=max_history_rows,
            )

        seen_item_ids: set[str] = set()
        if exclude_seen_items and filter_candidates_by_item_type:
            seen_item_ids = self.accessor.user_seen_item_ids(
                user_id=user_id,
                lookback=seen_history_lookback,
            )
            if seen_item_ids:
                candidate_items = [
                    x for x in candidate_items if str(x.get("item_id", "")) not in seen_item_ids
                ]

        output = IntentDualRecallOutput(
            query=clean_query,
            user_id=str(user_id),
            routing={
                "reasoning": routing.reasoning,
                "selected_category_paths": routing.category_paths,
                "selected_item_types": routing.item_types,
                "history_category_paths_k": max(1, int(history_category_paths_k)),
                "query_category_paths_k": max(1, int(query_category_paths_k)),
                "interested_item_types_k": max(1, int(interested_item_types_k)),
                "semantic_history_top_k": max(1, int(semantic_history_top_k)),
                "semantic_history_enabled": self.query_embedding_model is not None,
                "history_top_item_types": history_top_item_types,
                "exclude_seen_items": bool(exclude_seen_items),
                "seen_history_lookback": int(seen_history_lookback),
                "seen_item_count": len(seen_item_ids),
                "final_rollup_paths": final_rollup_paths,
                "filter_candidates_by_item_type": bool(filter_candidates_by_item_type),
                "candidate_item_scope_size": len(candidate_item_ids_scope or []),
                "adaptive_embedding_state": adaptive_state,
                "catalog_size": {
                    "category_paths": len(category_catalog),
                    "item_types": len(item_type_catalog),
                },
            },
            candidate_items=candidate_items,
            query_relevant_history=history_rows,
        )

        if save_output:
            path = _build_output_file_path(str(user_id), clean_query, output_dir=output_dir)
            output.routing["saved_output_path"] = str(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(output.to_dict(), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        return output


if __name__ == "__main__":
    # Example usage:
    # python intent_dual_recall_agent.py
    # Requires module-1 DBs generated beforehand.
    llm = Qwen3RouterLLM(model_name="Qwen/Qwen3-8B")
    accessor = GlobalHistoryAccessor(
        global_db_path="./processed/global_item_features.db",
        history_db_path="./processed/user_history_log.db",
    )
    query_emb = Qwen3QueryEmbeddingModel(model_name="Qwen/Qwen3-Embedding-0.6B")
    agent = RoutingRecallAgent(llm=llm, accessor=accessor, query_embedding_model=query_emb)

    out = agent.run(
        user_id="0",
        query="", #我想找适合客厅多人玩的体感游戏
        min_candidate_items=10,
        semantic_history_top_k=20,
        query_category_paths_k=2,
        history_category_paths_k=3,
        save_output=True,
        output_dir="./processed/intent_dual_recall_outputs",
    )
    print(json.dumps(out.to_dict(), ensure_ascii=False, indent=2, default=str))
