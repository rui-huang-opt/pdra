import traceback

try:
    from .node import NodeException, Node, gen_communication_edges
    from .edge import Edge
except NodeException:
    traceback.print_exc()
