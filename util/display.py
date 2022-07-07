from clingo import Symbol, SymbolType

import networkx as nx

import matplotlib.pyplot as plt

def symbol_to_str(symbol: Symbol):
    string = ""
    stack = [symbol]
    while stack:
        current = stack.pop()
        if isinstance(current, Symbol):
            if current.type == SymbolType.Number:
                string += str(current.number)
            elif current.type == SymbolType.Function:
                string += current.name
                if current.arguments:
                    stack.append(')')
                    stack.extend(reversed(current.arguments))
                    stack.append('(')
        elif isinstance(current, str):
            string += current
    return string


_edge_type_map = {
    1: 'solid',
    -1: 'dashed',
    0: 'dotted',
}


def display_explanation_graph(graph: nx.DiGraph, root=None):
    # p = plt.figure()
    pos = nx.nx_pydot.pydot_layout(graph, 'dot', root)
    # nodes = nx.draw_networkx_nodes(graph, pos=pos)
    # es = graph.edges(data=True)
    # styles = [_edge_type_map.get(data['edge_type'], 'solid') for u, v, data in es]
    #edges = nx.draw_networkx_edges(graph, pos=pos, style=styles)
    #return p
    return nx.draw_networkx(graph, pos, with_labels=True)

