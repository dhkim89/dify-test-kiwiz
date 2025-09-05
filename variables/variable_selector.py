from typing import List

class VariableSelector:
    def __init__(self, node_id: str, variable: str):
        self.node_id = node_id
        self.variable = variable

    def to_selector(self) -> List[str]:
        return [self.node_id, self.variable]
