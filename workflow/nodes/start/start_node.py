from collections.abc import Mapping
from typing import Any, Optional

from core.workflow.constants import SYSTEM_VARIABLE_NODE_ID
from core.workflow.entities.node_entities import NodeRunResult
from core.workflow.entities.workflow_node_execution import WorkflowNodeExecutionStatus
from core.workflow.nodes.base import BaseNode
from core.workflow.nodes.base.entities import BaseNodeData, RetryConfig
from core.workflow.nodes.enums import ErrorStrategy, NodeType
from core.workflow.nodes.start.entities import StartNodeData


class StartNode(BaseNode):
    """
    시작 노드 - 워크플로우 실행의 진입점이 되는 노드
    
    주요 기능:
    - 워크플로우 실행 시작점 정의
    - 사용자 입력 변수 수집 및 전달
    - 시스템 변수 초기화 (conversation_id, user_id 등)
    - 워크플로우 실행 컨텍스트 설정
    - 입력 데이터 검증 및 형변환
    
    사용 예시:
    - 사용자로부터 질문 받기
    - 파일 업로드 처리
    - API 요청 데이터 수신
    - 폼 데이터 수집
    - 초기 설정값 정의
    """
    _node_type = NodeType.START

    _node_data: StartNodeData

    def init_node_data(self, data: Mapping[str, Any]) -> None:
        self._node_data = StartNodeData(**data)

    def _get_error_strategy(self) -> Optional[ErrorStrategy]:
        return self._node_data.error_strategy

    def _get_retry_config(self) -> RetryConfig:
        return self._node_data.retry_config

    def _get_title(self) -> str:
        return self._node_data.title

    def _get_description(self) -> Optional[str]:
        return self._node_data.desc

    def _get_default_value_dict(self) -> dict[str, Any]:
        return self._node_data.default_value_dict

    def get_base_node_data(self) -> BaseNodeData:
        return self._node_data

    @classmethod
    def version(cls) -> str:
        return "1"

    def _run(self) -> NodeRunResult:
        node_inputs = dict(self.graph_runtime_state.variable_pool.user_inputs)
        system_inputs = self.graph_runtime_state.variable_pool.system_variables.to_dict()

        # TODO: System variables should be directly accessible, no need for special handling
        # Set system variables as node outputs.
        for var in system_inputs:
            node_inputs[SYSTEM_VARIABLE_NODE_ID + "." + var] = system_inputs[var]
        outputs = dict(node_inputs)

        return NodeRunResult(status=WorkflowNodeExecutionStatus.SUCCEEDED, inputs=node_inputs, outputs=outputs)
