
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Generator, Optional
import logging

from models import Workflow
from workflow.workflow_entry import WorkflowEntry, WorkflowType
from workflow.entities.variable_pool import VariablePool
from workflow.system_variable import SystemVariable
from workflow.nodes.event import RunCompletedEvent

app = FastAPI(title="Workflow Node Tester", description="API for testing individual workflow nodes")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowNodeRunRequest(BaseModel):
    graph: Dict[str, Any]
    inputs: Dict[str, Any]
    node_id: str

class SimpleNodeTestRequest(BaseModel):
    node_type: str
    node_config: Dict[str, Any]
    inputs: Dict[str, Any]

@app.post("/run-node")
async def run_node(request: WorkflowNodeRunRequest):
    try:
        workflow = Workflow(
            graph=request.graph,
            tenant_id="1",
            app_id="1",
            type=WorkflowType.WORKFLOW,
        )

        variable_pool = VariablePool(
            system_variables=SystemVariable.empty(),
            user_inputs=request.inputs,
            environment_variables=[],
            conversation_variables=[]
        )

        node, result_generator = WorkflowEntry.single_step_run(
            workflow=workflow,
            node_id=request.node_id,
            user_id="1",
            user_inputs=request.inputs,
            variable_pool=variable_pool,
        )

        result = None
        for event in result_generator:
            if isinstance(event, RunCompletedEvent):
                result = event.run_result.outputs
                break
        
        logger.info(f"Node {request.node_id} executed successfully, result: {result}")
        return {"status": "success", "result": result}

    except Exception as e:
        logger.exception("Error running workflow node")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-node")
async def test_node(request: SimpleNodeTestRequest):
    """간단한 노드 테스트용 엔드포인트"""
    try:
        # 단일 노드를 위한 기본 그래프 생성
        node_id = "test_node_1"
        graph = {
            "nodes": [
                {
                    "id": node_id,
                    "data": {
                        "type": request.node_type,
                        **request.node_config
                    }
                }
            ],
            "edges": []
        }

        workflow = Workflow(
            graph=graph,
            tenant_id="1",
            app_id="1",
            type=WorkflowType.WORKFLOW,
        )

        variable_pool = VariablePool(
            system_variables=SystemVariable.empty(),
            user_inputs=request.inputs,
            environment_variables=[],
            conversation_variables=[]
        )

        node, result_generator = WorkflowEntry.single_step_run(
            workflow=workflow,
            node_id=node_id,
            user_id="1",
            user_inputs=request.inputs,
            variable_pool=variable_pool,
        )

        result = None
        for event in result_generator:
            if isinstance(event, RunCompletedEvent):
                result = event.run_result.outputs
                break
        
        logger.info(f"Node {request.node_type} executed successfully")
        return {
            "status": "success", 
            "node_type": request.node_type,
            "result": result
        }

    except Exception as e:
        logger.exception(f"Error testing {request.node_type} node")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/node-types")
async def get_node_types():
    """사용 가능한 노드 타입들 반환"""
    return {
        "node_types": [
            "llm",
            "code",
            "http_request",
            "if_else",
            "variable_assigner",
            "parameter_extractor",
            "template_transform",
            "knowledge_retrieval",
            "tool",
            "start",
            "end",
            "answer"
        ]
    }

@app.get("/node-examples/{node_type}")
async def get_node_example(node_type: str):
    """특정 노드 타입의 예제 설정 반환"""
    examples = {
        "llm": {
            "node_config": {
                "model": "gpt-3.5-turbo",
                "prompt_template": "{{input}}에 대해 설명해주세요.",
                "temperature": 0.7
            },
            "inputs": {
                "input": "파이썬"
            }
        },
        "code": {
            "node_config": {
                "code": "result = inputs['number'] * 2\noutput = {'doubled': result}",
                "code_language": "python3"
            },
            "inputs": {
                "number": 5
            }
        },
        "http_request": {
            "node_config": {
                "method": "GET",
                "url": "https://api.github.com/users/octocat",
                "headers": {}
            },
            "inputs": {}
        },
        "template_transform": {
            "node_config": {
                "template": "안녕하세요, {{name}}님! 오늘은 {{date}}입니다."
            },
            "inputs": {
                "name": "홍길동",
                "date": "2024-01-15"
            }
        }
    }
    
    if node_type in examples:
        return examples[node_type]
    else:
        return {"error": f"No example available for node type: {node_type}"}

@app.get("/")
async def root():
    return {
        "message": "Workflow Node Tester API",
        "endpoints": {
            "/run-node": "전체 워크플로우 실행",
            "/test-node": "단일 노드 테스트",
            "/node-types": "사용 가능한 노드 타입들",
            "/node-examples/{node_type}": "특정 노드의 예제 설정"
        }
    }
