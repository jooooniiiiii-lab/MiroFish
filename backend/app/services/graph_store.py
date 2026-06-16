"""
GraphStore — Zep Cloud 的免费本地替代品
使用 SQLite + sentence-transformers 提供知识图谱存储与语义搜索

数据模型与 Zep 兼容，API 设计参考 Zep Graph SDK：
- create_graph / delete_graph / set_ontology
- add_batch (episodes) + episode processing
- 节点/边分页读取 (offset-based)
- 混合搜索 (embedding cosine + keyword fallback)
"""

import json
import os
import sqlite3
import threading
import time
import uuid as uuid_mod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.graph_store")

# ---------------------------------------------------------------------------
# 简单向量化器：使用 sentence-transformers（若不可用则降级为 TF-IDF）
# ---------------------------------------------------------------------------

_EMBEDDER = None
_EMBEDDER_LOCK = threading.Lock()
_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


def _get_embedder():
    """延迟加载 sentence-transformers 模型（仅在首次搜索时加载）"""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: all-MiniLM-L6-v2 (~80MB)...")
            model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
            )
            _EMBEDDER = model
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"sentence-transformers not available ({e}), using fallback embedding")
            _EMBEDDER = None
    return _EMBEDDER


def _compute_embedding(text: str) -> Optional[np.ndarray]:
    """计算文本的 embedding 向量，失败时返回 None"""
    model = _get_embedder()
    if model is None:
        return None
    try:
        emb = model.encode(text, normalize_embeddings=True)
        return np.array(emb, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度（输入已归一化时等价于点积）"""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# 数据类（与 Zep SDK 返回格式兼容）
# ---------------------------------------------------------------------------

@dataclass
class NodeData:
    uuid_: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    created_at: Optional[str] = None


@dataclass
class EdgeData:
    uuid_: str
    name: str
    fact: str
    fact_type: str
    source_node_uuid: str
    target_node_uuid: str
    attributes: Dict[str, Any]
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    episodes: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------

def _db_path(graph_id: str) -> str:
    """每个 graph 独立数据库文件"""
    data_dir = os.path.join(os.path.dirname(__file__), "../../data/graphs")
    os.makedirs(data_dir, exist_ok=True)
    # 安全处理 graph_id（避免路径穿越）
    safe = graph_id.replace("/", "_").replace("\\", "_")
    return os.path.join(data_dir, f"{safe}.db")


# ---------------------------------------------------------------------------
# GraphStore — 核心服务
# ---------------------------------------------------------------------------

class GraphStore:
    """本地知识图谱存储引擎，替代 Zep Cloud"""

    # ----------------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------------

    def __init__(self, api_key: Optional[str] = None):
        # api_key 保留以保持接口兼容，实际不使用
        self.api_key = api_key
        # 线程本地连接缓存
        self._local = threading.local()

    def _conn(self, graph_id: str) -> sqlite3.Connection:
        """获取或创建 graph 对应的 SQLite 连接"""
        # 每个 graph 独立数据库文件，允许并发读
        path = _db_path(graph_id)
        conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema(conn)
        return conn

    @staticmethod
    def _init_schema(conn: sqlite3.Connection):
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS ontology_entities (
                name TEXT PRIMARY KEY,
                description TEXT,
                attributes TEXT
            );
            CREATE TABLE IF NOT EXISTS ontology_edges (
                name TEXT PRIMARY KEY,
                description TEXT,
                source_targets TEXT,
                attributes TEXT
            );
            CREATE TABLE IF NOT EXISTS nodes (
                uuid TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                labels TEXT DEFAULT '[]',
                summary TEXT DEFAULT '',
                attributes TEXT DEFAULT '{}',
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS edges (
                uuid TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                fact TEXT DEFAULT '',
                fact_type TEXT DEFAULT '',
                source_node_uuid TEXT,
                target_node_uuid TEXT,
                attributes TEXT DEFAULT '{}',
                embedding BLOB,
                valid_at TEXT,
                invalid_at TEXT,
                expired_at TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS episodes (
                uuid TEXT PRIMARY KEY,
                data TEXT,
                type TEXT DEFAULT 'text',
                processed INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_uuid);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_uuid);
        """)
        conn.commit()

    # ----------------------------------------------------------------
    # Graph 管理
    # ----------------------------------------------------------------

    def create_graph(self, graph_id: str, name: str, description: str = ""):
        """创建新图谱（数据库文件在首次写入时自动创建）"""
        conn = self._conn(graph_id)
        conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
                     ("graph_name", name))
        conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
                     ("graph_description", description))
        conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
                     ("created_at", datetime.now(timezone.utc).isoformat()))
        conn.commit()
        logger.info(f"Graph created: {graph_id} ({name})")

    def delete_graph(self, graph_id: str):
        """删除图谱（删除数据库文件）"""
        path = _db_path(graph_id)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Graph deleted: {graph_id}")
        else:
            logger.warning(f"Graph not found for deletion: {graph_id}")

    def get_graph_info(self, graph_id: str) -> Dict[str, Any]:
        """获取图谱信息"""
        conn = self._conn(graph_id)
        cur = conn.execute("SELECT key, value FROM meta")
        meta = {row["key"]: row["value"] for row in cur.fetchall()}
        node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        entity_types = [
            row[0] for row in conn.execute(
                "SELECT DISTINCT value FROM json_each(labels) "
                "WHERE value NOT IN ('Entity', 'Node')"
            ).fetchall()
        ]
        return {
            "graph_id": graph_id,
            "name": meta.get("graph_name", ""),
            "description": meta.get("graph_description", ""),
            "node_count": node_count,
            "edge_count": edge_count,
            "entity_types": entity_types,
        }

    # ----------------------------------------------------------------
    # Ontology
    # ----------------------------------------------------------------

    def set_ontology(self, graph_ids: List[str],
                     entities: Optional[Dict[str, Any]] = None,
                     edges: Optional[Dict[str, Any]] = None):
        """设置本体（接口兼容 Zep）"""
        for gid in graph_ids:
            conn = self._conn(gid)
            if entities:
                for name, cls_def in entities.items():
                    desc = getattr(cls_def, "__doc__", "") or ""
                    attrs = []
                    if hasattr(cls_def, "model_fields"):
                        for fname, finfo in cls_def.model_fields.items():
                            attrs.append({
                                "name": fname,
                                "description": finfo.description or "",
                            })
                    conn.execute(
                        "INSERT OR REPLACE INTO ontology_entities (name, description, attributes) VALUES (?, ?, ?)",
                        (name, desc, json.dumps(attrs, ensure_ascii=False)),
                    )
            if edges:
                for name, edge_tuple in edges.items():
                    edge_cls, source_targets = edge_tuple
                    desc = getattr(edge_cls, "__doc__", "") or ""
                    st_json = json.dumps(
                        [{"source": st.source, "target": st.target}
                         for st in source_targets],
                        ensure_ascii=False,
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO ontology_edges "
                        "(name, description, source_targets, attributes) VALUES (?, ?, ?, '[]')",
                        (name, desc, st_json),
                    )
            conn.commit()

    # ----------------------------------------------------------------
    # Episode (文本块) 摄入
    # ----------------------------------------------------------------

    def add_batch(self, graph_id: str, episodes: List[Any]) -> List[Any]:
        """
        添加文本 episodes。
        返回模拟 Zep batch result 的对象列表，每个包含 uuid_ 属性。
        """
        conn = self._conn(graph_id)
        results = []
        for ep in episodes:
            ep_uuid = str(uuid_mod.uuid4())
            data = ep.data if hasattr(ep, "data") else str(ep)
            ep_type = ep.type if hasattr(ep, "type") else "text"
            conn.execute(
                "INSERT INTO episodes (uuid, data, type, processed) VALUES (?, ?, ?, 0)",
                (ep_uuid, data, ep_type),
            )
            # 立即"处理"（本地模式无异步延迟）
            conn.execute("UPDATE episodes SET processed = 1 WHERE uuid = ?", (ep_uuid,))
            results.append(_EpisodeResult(uuid_=ep_uuid))
        conn.commit()

        # 从 episodes 中提取实体和关系到 nodes/edges 表
        self._extract_episodes_to_graph(graph_id, episodes)

        return results

    def _extract_episodes_to_graph(self, graph_id: str, episodes: List[Any]):
        """
        将 episodes 文本解析为节点和边。
        使用 LLM 从文本中提取结构化实体-关系三元组。
        """
        conn = self._conn(graph_id)

        for ep in episodes:
            text = ep.data if hasattr(ep, "data") else str(ep)
            if not text.strip():
                continue

            # 从文本中提取命名实体（简单的关键词抽取，后续可通过 LLM 增强）
            entities = self._extract_entities_from_text(text)
            for ent in entities:
                node_uuid = str(uuid_mod.uuid4())
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO nodes (uuid, name, labels, summary, attributes) VALUES (?, ?, ?, ?, ?)",
                        (node_uuid, ent["name"], json.dumps(ent.get("labels", ["Entity"])),
                         ent.get("summary", text[:200]), "{}"),
                    )
                except Exception:
                    pass

        conn.commit()
        logger.info(f"Extracted entities from {len(episodes)} episode(s) into graph {graph_id}")

    @staticmethod
    def _extract_entities_from_text(text: str) -> List[Dict[str, Any]]:
        """
        简单的实体抽取：提取大写单词/专有名词作为候选实体。
        高级抽取可通过 LLM 调用实现（当前保持轻量）。
        """
        entities = []
        import re

        # 提取大写词（至少2字符）作为命名实体候选
        found = set()
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
            name = match.group(1).strip()
            if len(name) >= 3 and name not in found:
                found.add(name)
                entities.append({
                    "name": name,
                    "labels": ["Entity", "Node"],
                    "summary": "",
                })
        return entities

    def get_episode(self, graph_id: str, uuid_: str):
        """获取 episode 状态（兼容 Zep interface）"""
        conn = self._conn(graph_id)
        row = conn.execute("SELECT uuid, data, type, processed FROM episodes WHERE uuid = ?",
                           (uuid_,)).fetchone()
        if row:
            return _EpisodeResult(
                uuid_=row["uuid"],
                processed=bool(row["processed"]),
            )
        return None

    # ----------------------------------------------------------------
    # 节点 & 边 读取（分页）
    # ----------------------------------------------------------------

    def get_nodes(self, graph_id: str, limit: int = 100,
                  uuid_cursor: Optional[str] = None) -> List[NodeData]:
        """分页读取节点（offset-based）"""
        conn = self._conn(graph_id)
        if uuid_cursor:
            rows = conn.execute(
                "SELECT * FROM nodes WHERE uuid > ? ORDER BY uuid LIMIT ?",
                (uuid_cursor, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM nodes ORDER BY uuid LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_edges(self, graph_id: str, limit: int = 100,
                  uuid_cursor: Optional[str] = None) -> List[EdgeData]:
        """分页读取边"""
        conn = self._conn(graph_id)
        if uuid_cursor:
            rows = conn.execute(
                "SELECT * FROM edges WHERE uuid > ? ORDER BY uuid LIMIT ?",
                (uuid_cursor, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM edges ORDER BY uuid LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_node(self, graph_id: str, uuid_: str) -> Optional[NodeData]:
        """获取单个节点"""
        conn = self._conn(graph_id)
        row = conn.execute("SELECT * FROM nodes WHERE uuid = ?", (uuid_,)).fetchone()
        return self._row_to_node(row) if row else None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeData]:
        """获取与节点相连的所有边"""
        conn = self._conn(graph_id)
        rows = conn.execute(
            "SELECT * FROM edges WHERE source_node_uuid = ? OR target_node_uuid = ?",
            (node_uuid, node_uuid),
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_entities_by_label(self, graph_id: str, label: str) -> List[NodeData]:
        """按标签获取节点"""
        conn = self._conn(graph_id)
        all_nodes = self.get_nodes(graph_id, limit=99999)
        return [n for n in all_nodes if label in (n.labels or [])]

    # ----------------------------------------------------------------
    # 搜索
    # ----------------------------------------------------------------

    def search(self, graph_id: str, query: str, limit: int = 10,
               scope: str = "edges", reranker: str = "cross_encoder") -> Dict[str, Any]:
        """
        混合搜索：先尝试 embedding 语义搜索，失败则降级为关键词匹配。

        返回格式：
        {
            "edges": [...],
            "nodes": [...],
        }
        每个 edge/node 包含 Zep SDK 兼容的属性。
        """
        # 尝试语义搜索
        result = self._semantic_search(graph_id, query, limit, scope)
        if result is not None:
            return result

        # 降级：关键词搜索
        logger.info("Semantic search unavailable, falling back to keyword search")
        return self._keyword_search(graph_id, query, limit, scope)

    def _semantic_search(self, graph_id: str, query: str, limit: int = 10,
                         scope: str = "edges") -> Optional[Dict[str, Any]]:
        """embedding 语义搜索"""
        query_emb = _compute_embedding(query)
        if query_emb is None:
            return None

        conn = self._conn(graph_id)
        result = {"edges": [], "nodes": []}

        if scope in ("edges", "both"):
            rows = conn.execute("SELECT * FROM edges").fetchall()
            scored = []
            for row in rows:
                edge = self._row_to_edge(row)
                # 优先使用存储的 embedding
                emb_blob = row["embedding"]
                if emb_blob:
                    emb = np.frombuffer(emb_blob, dtype=np.float32)
                    score = _cosine_similarity(query_emb, emb)
                else:
                    # 运行时计算
                    text = f"{edge.name} {edge.fact}".strip()
                    if text:
                        emb = _compute_embedding(text)
                        score = _cosine_similarity(query_emb, emb) if emb is not None else 0.0
                    else:
                        score = 0.0
                scored.append((score, edge))

            scored.sort(key=lambda x: x[0], reverse=True)
            for score, edge in scored[:limit]:
                d = edge.__dict__.copy()
                d["score"] = round(float(score), 4)
                result["edges"].append(d)

        if scope in ("nodes", "both"):
            rows = conn.execute("SELECT * FROM nodes").fetchall()
            scored = []
            for row in rows:
                node = self._row_to_node(row)
                emb_blob = row["embedding"]
                if emb_blob:
                    emb = np.frombuffer(emb_blob, dtype=np.float32)
                    score = _cosine_similarity(query_emb, emb)
                else:
                    text = f"{node.name} {node.summary}".strip()
                    if text:
                        emb = _compute_embedding(text)
                        score = _cosine_similarity(query_emb, emb) if emb is not None else 0.0
                    else:
                        score = 0.0
                scored.append((score, node))

            scored.sort(key=lambda x: x[0], reverse=True)
            for score, node in scored[:limit]:
                d = node.__dict__.copy()
                d["score"] = round(float(score), 4)
                result["nodes"].append(d)

        return result

    def _keyword_search(self, graph_id: str, query: str, limit: int = 10,
                        scope: str = "edges") -> Dict[str, Any]:
        """关键词匹配搜索（降级方案）"""
        conn = self._conn(graph_id)
        result = {"edges": [], "nodes": []}

        keywords = [w.strip().lower() for w in query.replace(",", " ").split()
                    if len(w.strip()) > 1]

        def match_score(text: str) -> int:
            if not text:
                return 0
            t = text.lower()
            if query.lower() in t:
                return 100
            score = 0
            for kw in keywords:
                if kw in t:
                    score += 10
            return score

        if scope in ("edges", "both"):
            rows = conn.execute("SELECT * FROM edges").fetchall()
            scored = []
            for row in rows:
                edge = self._row_to_edge(row)
                score = match_score(edge.fact) + match_score(edge.name)
                if score > 0:
                    scored.append((score, edge))
            scored.sort(key=lambda x: x[0], reverse=True)
            result["edges"] = [e.__dict__ for _, e in scored[:limit]]

        if scope in ("nodes", "both"):
            rows = conn.execute("SELECT * FROM nodes").fetchall()
            scored = []
            for row in rows:
                node = self._row_to_node(row)
                score = match_score(node.name) + match_score(node.summary)
                if score > 0:
                    scored.append((score, node))
            scored.sort(key=lambda x: x[0], reverse=True)
            result["nodes"] = [n.__dict__ for _, n in scored[:limit]]

        return result

    # ----------------------------------------------------------------
    # 辅助
    # ----------------------------------------------------------------

    @staticmethod
    def _row_to_node(row) -> NodeData:
        if not row:
            return None
        return NodeData(
            uuid_=row["uuid"],
            name=row["name"],
            labels=json.loads(row["labels"]) if isinstance(row["labels"], str) else (row["labels"] or []),
            summary=row["summary"] or "",
            attributes=json.loads(row["attributes"]) if isinstance(row["attributes"], str) else (row["attributes"] or {}),
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_edge(row) -> EdgeData:
        if not row:
            return None
        return EdgeData(
            uuid_=row["uuid"],
            name=row["name"] or "",
            fact=row["fact"] or "",
            fact_type=row["fact_type"] or row["name"] or "",
            source_node_uuid=row["source_node_uuid"] or "",
            target_node_uuid=row["target_node_uuid"] or "",
            attributes=json.loads(row["attributes"]) if isinstance(row["attributes"], str) else (row["attributes"] or {}),
            created_at=row["created_at"],
            valid_at=row.get("valid_at"),
            invalid_at=row.get("invalid_at"),
            expired_at=row.get("expired_at"),
            episodes=[],
        )

    def add_node(self, graph_id: str, name: str, labels: List[str] = None,
                 summary: str = "", attributes: Dict = None,
                 embedding: Optional[np.ndarray] = None) -> str:
        """手动添加一个节点"""
        conn = self._conn(graph_id)
        node_uuid = str(uuid_mod.uuid4())
        emb_blob = embedding.tobytes() if embedding is not None else None
        conn.execute(
            "INSERT INTO nodes (uuid, name, labels, summary, attributes, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            (node_uuid, name, json.dumps(labels or ["Entity"]),
             summary, json.dumps(attributes or {}), emb_blob),
        )
        conn.commit()
        return node_uuid

    def add_edge(self, graph_id: str, name: str, fact: str,
                 source_node_uuid: str, target_node_uuid: str,
                 attributes: Dict = None) -> str:
        """手动添加一条边"""
        conn = self._conn(graph_id)
        edge_uuid = str(uuid_mod.uuid4())
        conn.execute(
            "INSERT INTO edges (uuid, name, fact, fact_type, source_node_uuid, target_node_uuid, attributes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (edge_uuid, name, fact, name, source_node_uuid, target_node_uuid,
             json.dumps(attributes or {})),
        )
        conn.commit()
        return edge_uuid


# ---------------------------------------------------------------------------
# 模拟 Zep Episode 返回对象
# ---------------------------------------------------------------------------

class _EpisodeResult:
    def __init__(self, uuid_: str, processed: bool = True):
        self.uuid_ = uuid_
        self.processed = processed

    def __repr__(self):
        return f"_EpisodeResult(uuid_={self.uuid_}, processed={self.processed})"
