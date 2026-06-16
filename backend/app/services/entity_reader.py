"""
实体读取与过滤服务（ZepEntityReader 的本地替代品）
从 GraphStore 中读取节点，筛选出符合预定义实体类型的节点
"""

import json
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

from .graph_store import GraphStore
from ..utils.logger import get_logger

logger = get_logger('mirofish.entity_reader')


@dataclass
class EntityNode:
    """实体节点数据结构"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """过滤后的实体集合"""
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class EntityReader:
    """
    实体读取与过滤服务
    替代 ZepEntityReader，使用本地 GraphStore
    """

    def __init__(self, api_key: Optional[str] = None):
        self.store = GraphStore(api_key=api_key)

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """获取图谱的所有节点"""
        nodes = self.store.get_nodes(graph_id, limit=99999)
        return [
            {
                "uuid": n.uuid_,
                "name": n.name,
                "labels": n.labels,
                "summary": n.summary,
                "attributes": n.attributes,
            }
            for n in nodes
        ]

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """获取图谱的所有边"""
        edges = self.store.get_edges(graph_id, limit=99999)
        return [
            {
                "uuid": e.uuid_,
                "name": e.name,
                "fact": e.fact,
                "source_node_uuid": e.source_node_uuid,
                "target_node_uuid": e.target_node_uuid,
                "attributes": e.attributes,
            }
            for e in edges
        ]

    def get_node_edges(self, node_uuid: str, graph_id: str) -> List[Dict[str, Any]]:
        """获取指定节点的所有相关边"""
        edges = self.store.get_node_edges(graph_id, node_uuid)
        return [
            {
                "uuid": e.uuid_,
                "name": e.name,
                "fact": e.fact,
                "source_node_uuid": e.source_node_uuid,
                "target_node_uuid": e.target_node_uuid,
                "attributes": e.attributes,
            }
            for e in edges
        ]

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> FilteredEntities:
        """筛选出符合预定义实体类型的节点"""
        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)
        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []
        node_map = {n["uuid"]: n for n in all_nodes}

        filtered_entities = []
        entity_types_found: Set[str] = set()

        for node in all_nodes:
            labels = node.get("labels", [])
            custom_labels = [l for l in labels if l not in ["Entity", "Node"]]
            if not custom_labels:
                continue

            if defined_entity_types:
                matching_labels = [l for l in custom_labels if l in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)
            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node["summary"],
                attributes=node["attributes"],
            )

            if enrich_with_edges:
                related_edges = []
                related_node_uuids = set()

                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])

                entity.related_edges = related_edges

                related_nodes = []
                for ruid in related_node_uuids:
                    if ruid in node_map:
                        rn = node_map[ruid]
                        related_nodes.append({
                            "uuid": rn["uuid"],
                            "name": rn["name"],
                            "labels": rn["labels"],
                            "summary": rn.get("summary", ""),
                        })
                entity.related_nodes = related_nodes

            filtered_entities.append(entity)

        logger.info(
            f"筛选完成: 总节点 {total_count}, 符合条件 {len(filtered_entities)}, "
            f"实体类型: {entity_types_found}"
        )

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    def get_entity_with_context(self, graph_id: str, entity_uuid: str) -> Optional[EntityNode]:
        """获取单个实体及其完整上下文"""
        node = self.store.get_node(graph_id, entity_uuid)
        if not node:
            return None

        edges = self.store.get_node_edges(graph_id, entity_uuid)
        all_nodes = self.store.get_nodes(graph_id, limit=99999)
        node_map = {n.uuid_: {"uuid": n.uuid_, "name": n.name, "labels": n.labels, "summary": n.summary}
                    for n in all_nodes}

        related_edges = []
        related_node_uuids = set()
        for e in edges:
            if e.source_node_uuid == entity_uuid:
                related_edges.append({
                    "direction": "outgoing",
                    "edge_name": e.name,
                    "fact": e.fact,
                    "target_node_uuid": e.target_node_uuid,
                })
                related_node_uuids.add(e.target_node_uuid)
            else:
                related_edges.append({
                    "direction": "incoming",
                    "edge_name": e.name,
                    "fact": e.fact,
                    "source_node_uuid": e.source_node_uuid,
                })
                related_node_uuids.add(e.source_node_uuid)

        related_nodes = []
        for ruid in related_node_uuids:
            if ruid in node_map:
                related_nodes.append(node_map[ruid])

        return EntityNode(
            uuid=node.uuid_,
            name=node.name,
            labels=node.labels,
            summary=node.summary,
            attributes=node.attributes,
            related_edges=related_edges,
            related_nodes=related_nodes,
        )

    def get_entities_by_type(self, graph_id: str, entity_type: str,
                             enrich_with_edges: bool = True) -> List[EntityNode]:
        """获取指定类型的所有实体"""
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges,
        )
        return result.entities
