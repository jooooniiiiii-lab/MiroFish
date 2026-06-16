"""
图谱构建服务
接口2：使用本地 GraphStore 构建 Standalone Graph
"""

import os
import uuid
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from .graph_store import GraphStore
from ..models.task import TaskManager, TaskStatus
from .text_processor import TextProcessor
from ..utils.locale import t, get_locale, set_locale


@dataclass
class GraphInfo:
    """图谱信息"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """
    图谱构建服务
    负责调用本地 GraphStore 构建知识图谱（替代原 Zep Cloud 实现）
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # api_key 保留以保持接口兼容，实际由 GraphStore 内部管理
        self.store = GraphStore(api_key=api_key)
        self.task_manager = TaskManager()
    
    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3
    ) -> str:
        """
        异步构建图谱
        
        Args:
            text: 输入文本
            ontology: 本体定义（来自接口1的输出）
            graph_name: 图谱名称
            chunk_size: 文本块大小
            chunk_overlap: 块重叠大小
            batch_size: 每批发送的块数量
            
        Returns:
            任务ID
        """
        # 创建任务
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            }
        )
        
        # Capture locale before spawning background thread
        current_locale = get_locale()

        # 在后台线程中执行构建
        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size, current_locale)
        )
        thread.daemon = True
        thread.start()
        
        return task_id
    
    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
        locale: str = 'en'
    ):
        """图谱构建工作线程"""
        set_locale(locale)
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message=t('progress.startBuildingGraph')
            )
            
            # 1. 创建图谱
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=t('progress.graphCreated', graphId=graph_id)
            )
            
            # 2. 设置本体
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message=t('progress.ontologySet')
            )
            
            # 3. 文本分块
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=t('progress.textSplit', count=total_chunks)
            )
            
            # 4. 分批添加文本（本地 GraphStore 即时处理，无需等待轮询）
            self.add_text_batches(
                graph_id, chunks, batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.6),  # 20-80%
                    message=msg
                )
            )
            
            # 5. 获取图谱信息
            self.task_manager.update_task(
                task_id,
                progress=90,
                message=t('progress.fetchingGraphInfo')
            )
            
            graph_info = self._get_graph_info(graph_id)
            
            # 完成
            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
            })
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)
    
    def create_graph(self, name: str) -> str:
        """创建图谱"""
        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"
        self.store.create_graph(
            graph_id=graph_id,
            name=name,
            description="MiroFish Social Simulation Graph"
        )
        return graph_id
    
    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """设置图谱本体"""
        entities = {}
        edges = {}
        
        # 构建实体类型
        for entity_def in ontology.get("entity_types", []):
            name = entity_def["name"]
            description = entity_def.get("description", f"A {name} entity.")
            attrs = entity_def.get("attributes", [])
            
            # 构建一个轻量类，包含 GraphStore 所需的 __doc__ 和 model_fields
            entity_cls = self._make_model_class(name, description, attrs, field_type="text")
            entities[name] = entity_cls
        
        # 构建边类型
        for edge_def in ontology.get("edge_types", []):
            name = edge_def["name"]
            description = edge_def.get("description", f"A {name} relationship.")
            attrs = edge_def.get("attributes", [])
            
            edge_cls = self._make_model_class(name, description, attrs, field_type="string")
            source_targets = [
                {"source": st.get("source", "Entity"), "target": st.get("target", "Entity")}
                for st in edge_def.get("source_targets", [])
            ]
            if source_targets:
                edges[name] = (edge_cls, source_targets)
        
        if entities or edges:
            self.store.set_ontology(
                graph_ids=[graph_id],
                entities=entities if entities else None,
                edges=edges if edges else None,
            )
    
    @staticmethod
    def _make_model_class(name: str, description: str, attrs: List[Dict[str, str]],
                          field_type: str = "text"):
        """创建一个与 GraphStore.set_ontology 兼容的轻量类"""
        from pydantic import Field
        
        ns = {"__doc__": description, "__annotations__": {}}
        for attr_def in attrs:
            attr_name = attr_def["name"]
            attr_desc = attr_def.get("description", attr_name)
            ns[attr_name] = Field(description=attr_desc, default=None)
            ns["__annotations__"][attr_name] = str if field_type == "string" else Optional[str]
        
        from typing import Optional
        ns.setdefault("__annotations__", {})
        
        cls = type(name, (), ns)
        cls.model_fields = {
            k: v for k, v in ns.items()
            if not k.startswith("_") and k != "__annotations__" and hasattr(v, "description")
        }
        return cls
    
    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None
    ) -> List[str]:
        """分批添加文本到图谱，返回所有 episode 的 uuid 列表"""
        episode_uuids = []
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            if progress_callback:
                progress = (i + len(batch_chunks)) / total_chunks
                progress_callback(
                    t('progress.sendingBatch', current=batch_num, total=total_batches, chunks=len(batch_chunks)),
                    progress
                )
            
            # 构建 episode 数据（兼容 Zep EpisodeData 格式）
            episodes = []
            for chunk in batch_chunks:
                EpisodeData = type('EpisodeData', (), {'data': chunk, 'type': 'text'})
                episodes.append(EpisodeData())
            
            # 发送到 GraphStore
            try:
                batch_result = self.store.add_batch(
                    graph_id=graph_id,
                    episodes=episodes
                )
                
                # 收集返回的 episode uuid
                if batch_result and isinstance(batch_result, list):
                    for ep in batch_result:
                        ep_uuid = getattr(ep, 'uuid_', None) or getattr(ep, 'uuid', None)
                        if ep_uuid:
                            episode_uuids.append(ep_uuid)
                
                # 避免请求过快
                time.sleep(0.5)
                
            except Exception as e:
                if progress_callback:
                    progress_callback(t('progress.batchFailed', batch=batch_num, error=str(e)), 0)
                raise
        
        return episode_uuids
    
    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """获取图谱信息"""
        info = self.store.get_graph_info(graph_id)
        
        return GraphInfo(
            graph_id=graph_id,
            node_count=info.get("node_count", 0),
            edge_count=info.get("edge_count", 0),
            entity_types=info.get("entity_types", [])
        )
    
    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        获取完整图谱数据（包含详细信息）
        
        Args:
            graph_id: 图谱ID
            
        Returns:
            包含nodes和edges的字典
        """
        nodes = self.store.get_nodes(graph_id, limit=99999)
        edges = self.store.get_edges(graph_id, limit=99999)

        # 创建节点映射用于获取节点名称
        node_map = {}
        for node in nodes:
            node_map[node.uuid_] = node.name or ""
        
        nodes_data = []
        for node in nodes:
            created_at = getattr(node, 'created_at', None)
            if created_at:
                created_at = str(created_at)
            
            nodes_data.append({
                "uuid": node.uuid_,
                "name": node.name,
                "labels": node.labels or [],
                "summary": node.summary or "",
                "attributes": node.attributes or {},
                "created_at": created_at,
            })
        
        edges_data = []
        for edge in edges:
            created_at = getattr(edge, 'created_at', None)
            valid_at = getattr(edge, 'valid_at', None)
            invalid_at = getattr(edge, 'invalid_at', None)
            expired_at = getattr(edge, 'expired_at', None)
            
            episodes = getattr(edge, 'episodes', None)
            if episodes and not isinstance(episodes, list):
                episodes = [str(episodes)]
            elif episodes:
                episodes = [str(e) for e in episodes]
            
            fact_type = getattr(edge, 'fact_type', None) or edge.name or ""
            
            edges_data.append({
                "uuid": edge.uuid_,
                "name": edge.name or "",
                "fact": edge.fact or "",
                "fact_type": fact_type,
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "source_node_name": node_map.get(edge.source_node_uuid, ""),
                "target_node_name": node_map.get(edge.target_node_uuid, ""),
                "attributes": edge.attributes or {},
                "created_at": str(created_at) if created_at else None,
                "valid_at": str(valid_at) if valid_at else None,
                "invalid_at": str(invalid_at) if invalid_at else None,
                "expired_at": str(expired_at) if expired_at else None,
                "episodes": episodes or [],
            })
        
        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }
    
    def delete_graph(self, graph_id: str):
        """删除图谱"""
        self.store.delete_graph(graph_id)
