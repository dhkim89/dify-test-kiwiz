from collections.abc import Mapping
from typing import Any, Optional

from core.workflow.entities.node_entities import NodeRunResult
from core.workflow.entities.workflow_node_execution import WorkflowNodeExecutionStatus
from core.workflow.nodes.base import BaseNode
from core.workflow.nodes.base.entities import BaseNodeData, RetryConfig
from core.workflow.nodes.end.entities import EndNodeData
from core.workflow.nodes.enums import ErrorStrategy, NodeType


class EndNode(BaseNode):
    """
    종료 노드 - 워크플로우 실행을 완료하고 최종 결과를 출력하는 노드
    
    주요 기능:
    - 워크플로우 실행 종료점 정의
    - 최종 출력 변수 선택 및 포맷팅
    - 실행 결과 집계 및 요약
    - 출력 데이터 구조화
    - 워크플로우 성공/실패 상태 설정
    
    사용 예시:
    - 최종 답변 반환
    - 처리 결과 요약
    - 생성된 파일 출력
    - 계산 결과 반환
    - 상태 보고서 생성
    """
    _node_type = NodeType.END

    _node_data: EndNodeData

    def init_node_data(self, data: Mapping[str, Any]) -> None:
        self._node_data = EndNodeData(**data)

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
        """
        Run node
        :return:
        """
        output_variables = self._node_data.outputs

        outputs = {}
        for variable_selector in output_variables:
            variable = self.graph_runtime_state.variable_pool.get(variable_selector.value_selector)
            value = variable.to_object() if variable is not None else None
            outputs[variable_selector.variable] = value

        return NodeRunResult(
            status=WorkflowNodeExecutionStatus.SUCCEEDED,
            inputs=outputs,
            outputs=outputs,
        )
