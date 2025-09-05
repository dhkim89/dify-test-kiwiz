import logging
import time
import uuid
from collections.abc import Generator, Mapping, Sequence
from typing import Any, Optional, cast

from models import Workflow
# from configs import dify_config
# from core.app.apps.exc import GenerateTaskStoppedError
# from core.app.entities.app_invoke_entities import InvokeFrom
# from core.file.models import File
# from core.workflow.callbacks import WorkflowCallback
from workflow.constants import ENVIRONMENT_VARIABLE_NODE_ID
from workflow.entities.variable_pool import VariablePool
from workflow.errors import WorkflowNodeRunFailedError
from workflow.graph_engine.entities.event import GraphEngineEvent, GraphRunFailedEvent, InNodeEvent
from workflow.graph_engine.entities.graph import Graph
from workflow.graph_engine.entities.graph_init_params import GraphInitParams
from workflow.graph_engine.entities.graph_runtime_state import GraphRuntimeState
from workflow.graph_engine.graph_engine import GraphEngine
from workflow.nodes import NodeType
from workflow.nodes.base import BaseNode
from workflow.nodes.event import NodeEvent
from workflow.nodes.node_mapping import NODE_TYPE_CLASSES_MAPPING
from workflow.system_variable import SystemVariable
from workflow.variable_loader import DUMMY_VARIABLE_LOADER, VariableLoader, load_into_variable_pool
# from factories import file_factory
# from models.enums import UserFrom
# from models.workflow import (
#     Workflow,
#     WorkflowType,
# )

logger = logging.getLogger(__name__)

class SimpleInvokeFrom:
    DEBUGGER = 'debugger'

class SimpleUserFrom:
    ACCOUNT = 'account'

class WorkflowType:
    WORKFLOW = "workflow"
    CHAT = "chat"

    @classmethod
    def value_of(cls, value: str) -> "WorkflowType":
        for mode in cls:
            if mode.value == value:
                return mode
        raise ValueError(f"invalid workflow type value {value}")

class WorkflowEntry:
    def __init__(
        self,
        tenant_id: str,
        app_id: str,
        workflow_id: str,
        workflow_type: str,
        graph_config: Mapping[str, Any],
        graph: Graph,
        user_id: str,
        user_from: str,
        invoke_from: str,
        call_depth: int,
        variable_pool: VariablePool,
        thread_pool_id: Optional[str] = None,
    ) -> None:
        workflow_call_max_depth = 10 # dify_config.WORKFLOW_CALL_MAX_DEPTH
        if call_depth > workflow_call_max_depth:
            raise ValueError(f"Max workflow call depth {workflow_call_max_depth} reached.")

        graph_runtime_state = GraphRuntimeState(variable_pool=variable_pool, start_at=time.perf_counter())
        self.graph_engine = GraphEngine(
            tenant_id=tenant_id,
            app_id=app_id,
            workflow_type=workflow_type,
            workflow_id=workflow_id,
            user_id=user_id,
            user_from=user_from,
            invoke_from=invoke_from,
            call_depth=call_depth,
            graph=graph,
            graph_config=graph_config,
            graph_runtime_state=graph_runtime_state,
            max_execution_steps=100, # dify_config.WORKFLOW_MAX_EXECUTION_STEPS,
            max_execution_time=300, # dify_config.WORKFLOW_MAX_EXECUTION_TIME,
            thread_pool_id=thread_pool_id,
        )

    def run(
        self,
        *,
        callbacks: Sequence[Any], # WorkflowCallback
    ) -> Generator[GraphEngineEvent, None, None]:
        graph_engine = self.graph_engine

        try:
            generator = graph_engine.run()
            for event in generator:
                if callbacks:
                    for callback in callbacks:
                        callback.on_event(event=event)
                yield event
        except Exception as e:
            logger.exception("Unknown Error when workflow entry running")
            if callbacks:
                for callback in callbacks:
                    callback.on_event(event=GraphRunFailedEvent(error=str(e)))
            return

    @classmethod
    def single_step_run(
        cls,
        *,
        workflow: Workflow,
        node_id: str,
        user_id: str,
        user_inputs: Mapping[str, Any],
        variable_pool: VariablePool,
        variable_loader: VariableLoader = DUMMY_VARIABLE_LOADER,
    ) -> tuple[BaseNode, Generator[NodeEvent | InNodeEvent, None, None]]:
        node_config = workflow.get_node_config_by_id(node_id)
        node_config_data = node_config.get("data", {})

        node_type = NodeType(node_config_data.get("type"))
        node_version = node_config_data.get("version", "1")
        node_cls = NODE_TYPE_CLASSES_MAPPING[node_type][node_version]

        graph = Graph.init(graph_config=workflow.graph)

        node = node_cls(
            id=str(uuid.uuid4()),
            config=node_config,
            graph_init_params=GraphInitParams(
                tenant_id=workflow.tenant_id,
                app_id=workflow.app_id,
                workflow_type=workflow.type,
                workflow_id=workflow.id,
                graph_config=workflow.graph,
                user_id=user_id,
                user_from=SimpleUserFrom.ACCOUNT,
                invoke_from=SimpleInvokeFrom.DEBUGGER,
                call_depth=0,
            ),
            graph=graph,
            graph_runtime_state=GraphRuntimeState(variable_pool=variable_pool, start_at=time.perf_counter()),
        )
        node.init_node_data(node_config_data)

        try:
            variable_mapping = node_cls.extract_variable_selector_to_variable_mapping(
                graph_config=workflow.graph, config=node_config
            )
        except NotImplementedError:
            variable_mapping = {}

        load_into_variable_pool(
            variable_loader=variable_loader,
            variable_pool=variable_pool,
            variable_mapping=variable_mapping,
            user_inputs=user_inputs,
        )

        cls.mapping_user_inputs_to_variable_pool(
            variable_mapping=variable_mapping,
            user_inputs=user_inputs,
            variable_pool=variable_pool,
            tenant_id=workflow.tenant_id,
        )

        try:
            generator = node.run()
        except Exception as e:
            logger.exception(
                "error while running node, workflow_id=%s, node_id=%s, node_type=%s, node_version=%s",
                workflow.id,
                node.id,
                node.type_,
                node.version(),
            )
            raise WorkflowNodeRunFailedError(node=node, err_msg=str(e))
        return node, generator

    @staticmethod
    def _handle_special_values(value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, dict):
            res = {}
            for k, v in value.items():
                res[k] = WorkflowEntry._handle_special_values(v)
            return res
        if isinstance(value, list):
            res_list = []
            for item in value:
                res_list.append(WorkflowEntry._handle_special_values(item))
            return res_list
        # if isinstance(value, File):
        #     return value.to_dict()
        return value

    @classmethod
    def mapping_user_inputs_to_variable_pool(
        cls,
        *,
        variable_mapping: Mapping[str, Sequence[str]],
        user_inputs: Mapping[str, Any],
        variable_pool: VariablePool,
        tenant_id: str,
    ) -> None:
        for node_variable, variable_selector in variable_mapping.items():
            node_variable_list = node_variable.split(".")
            if len(node_variable_list) < 1:
                raise ValueError(f"Invalid node variable {node_variable}")

            node_variable_key = ".".join(node_variable_list[1:])

            if (node_variable_key not in user_inputs and node_variable not in user_inputs) and not variable_pool.get(
                variable_selector
            ):
                raise ValueError(f"Variable key {node_variable} not found in user inputs.")

            if variable_pool.get(variable_selector):
                continue

            variable_node_id = variable_selector[0]
            variable_key_list = cast(list, variable_selector[1:])

            input_value = user_inputs.get(node_variable)
            if not input_value:
                input_value = user_inputs.get(node_variable_key)

            # if isinstance(input_value, dict) and "type" in input_value and "transfer_method" in input_value:
            #     input_value = file_factory.build_from_mapping(mapping=input_value, tenant_id=tenant_id)
            # if (
            #     isinstance(input_value, list)
            #     and all(isinstance(item, dict) for item in input_value)
            #     and all("type" in item and "transfer_method" in item for item in input_value)
            # ):
            #     input_value = file_factory.build_from_mappings(mappings=input_value, tenant_id=tenant_id)

            if variable_node_id != ENVIRONMENT_VARIABLE_NODE_ID:
                variable_pool.add([variable_node_id] + variable_key_list, input_value)
