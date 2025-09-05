import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Union, cast

from pydantic import BaseModel, Field

from variables import Segment, SegmentGroup, FileSegment, ObjectSegment
from variables.variables import Variable, VariableUnion
from variables.variable_selector import VariableSelector
from workflow.constants import CONVERSATION_VARIABLE_NODE_ID, ENVIRONMENT_VARIABLE_NODE_ID, SYSTEM_VARIABLE_NODE_ID
from workflow.system_variable import SystemVariable
from mocks import file_manager, variable_factory, MockFile as File

VariableValue = Union[str, int, float, dict, list, File]

VARIABLE_PATTERN = re.compile(r"\{\{#([a-zA-Z0-9_]{1,50}(?:\.[a-zA-Z_][a-zA-Z0-9_]{0,29}){1,10})#\}\}")

class VariablePool(BaseModel):
    variable_dictionary: defaultdict[str, Annotated[dict[str, VariableUnion], Field(default_factory=dict)]] = Field(
        description="Variables mapping",
        default=defaultdict(dict),
    )
    user_inputs: Mapping[str, Any] = Field(
        description="User inputs",
        default_factory=dict,
    )
    system_variables: SystemVariable = Field(
        description="System variables",
        default_factory=SystemVariable.empty,
    )
    environment_variables: Sequence[VariableUnion] = Field(
        description="Environment variables.",
        default_factory=list,
    )
    conversation_variables: Sequence[VariableUnion] = Field(
        description="Conversation variables.",
        default_factory=list,
    )

    def model_post_init(self, context: Any, /) -> None:
        self._add_system_variables(self.system_variables)
        for var in self.environment_variables:
            self.add(VariableSelector(node_id=ENVIRONMENT_VARIABLE_NODE_ID, variable=var.name).to_selector(), var)
        for var in self.conversation_variables:
            self.add(VariableSelector(node_id=CONVERSATION_VARIABLE_NODE_ID, variable=var.name).to_selector(), var)

    def add(self, selector: Sequence[str], value: Any, /) -> None:
        if len(selector) < 2:
            raise ValueError("Invalid selector")

        if isinstance(value, Variable):
            variable = value
        elif isinstance(value, Segment):
            variable = variable_factory.segment_to_variable(segment=value, selector=selector)
        else:
            segment = variable_factory.build_segment(value)
            variable = variable_factory.segment_to_variable(segment=segment, selector=selector)

        node_id, name = selector[0], '.'.join(selector[1:])
        self.variable_dictionary[node_id][name] = cast(VariableUnion, variable)

    def _selector_to_keys(self, selector: Sequence[str]) -> tuple[str, str]:
        return selector[0], '.'.join(selector[1:])

    def _has(self, selector: Sequence[str]) -> bool:
        if len(selector) < 2:
            return False
        node_id, name = self._selector_to_keys(selector)
        return node_id in self.variable_dictionary and name in self.variable_dictionary[node_id]

    def get(self, selector: Sequence[str], /) -> Segment | None:
        if len(selector) < 2:
            return None

        node_id, name = self._selector_to_keys(selector)
        variable: Variable | None = self.variable_dictionary[node_id].get(name)

        if variable is None:
            return None

        segment = variable.value

        if len(selector) == 2:
            return segment

        if isinstance(segment, FileSegment):
            attr = selector[2]
            attr_value = file_manager.get_attr(file=segment.value, attr=attr)
            return variable_factory.build_segment(attr_value)

        result: Any = segment
        for attr in selector[2:]:
            result = self._extract_value(result)
            result = self._get_nested_attribute(result, attr)
            if result is None:
                return None

        return result if isinstance(result, Segment) else variable_factory.build_segment(result)

    def _extract_value(self, obj: Any) -> Any:
        return obj.value if isinstance(obj, ObjectSegment) else obj

    def _get_nested_attribute(self, obj: Mapping[str, Any], attr: str) -> Any:
        if not isinstance(obj, dict):
            return None
        return obj.get(attr)

    def remove(self, selector: Sequence[str], /):
        if not selector:
            return
        if len(selector) == 1:
            self.variable_dictionary[selector[0]] = {}
            return
        key, hash_key = self._selector_to_keys(selector)
        self.variable_dictionary[key].pop(hash_key, None)

    def convert_template(self, template: str, /):
        parts = VARIABLE_PATTERN.split(template)
        segments = []
        for part in filter(lambda x: x, parts):
            if "." in part and (variable := self.get(part.split("."))):
                segments.append(variable)
            else:
                segments.append(variable_factory.build_segment(part))
        return SegmentGroup(value=segments)

    def get_file(self, selector: Sequence[str], /) -> FileSegment | None:
        segment = self.get(selector)
        if isinstance(segment, FileSegment):
            return segment
        return None

    def _add_system_variables(self, system_variable: SystemVariable):
        sys_var_mapping = system_variable.to_dict()
        for key, value in sys_var_mapping.items():
            if value is None:
                continue
            selector = (SYSTEM_VARIABLE_NODE_ID, key)
            if self._has(selector):
                continue
            self.add(selector, value)

    @classmethod
    def empty(cls) -> "VariablePool":
        return cls(system_variables=SystemVariable.empty())
