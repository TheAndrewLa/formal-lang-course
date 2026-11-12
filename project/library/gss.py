from dataclasses import dataclass, field


@dataclass(eq=False)
class Node:
    state: int | None
    vertex: int
    preds: set["Node"] = field(default_factory=set)


@dataclass
class Stack:
    nodes: dict[tuple[str, int | None, int], Node] = field(default_factory=dict)

    def get_or_create(self, symbol: str, state: int | None, vertex: int) -> Node:
        key = (symbol, state, vertex)
        node = self.nodes.get(key)
        if node is not None:
            return node
        self.nodes[key] = Node(state, vertex)
        return self.nodes[key]
