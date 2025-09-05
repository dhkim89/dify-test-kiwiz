
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import uuid

class Node(BaseModel):
    id: str
    data: Dict[str, Any]

class Edge(BaseModel):
    id: str
    source: str
    target: str

class Graph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class Workflow(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    graph: Dict[str, Any]
    tenant_id: str
    app_id: str
    type: str
    
    def get_node_config_by_id(self, node_id: str) -> Dict[str, Any]:
        for node in self.graph.get('nodes', []):
            if node.get('id') == node_id:
                return node
        raise ValueError(f"Node with id {node_id} not found")

