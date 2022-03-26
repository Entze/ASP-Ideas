from typing import Sequence, Union, Optional, Dict, FrozenSet

import clingo
import networkx as nx


def explain(program,
            answer_sets: Optional[Sequence[Sequence[clingo.Symbol]]] = None,
            cautious_consequence: Optional[Sequence[clingo.Symbol]] = None,
            roots: Union[None, clingo.Symbol, Sequence[clingo.Symbol]] = None) -> Dict[FrozenSet[clingo.Symbol],
                                                                                       Dict[clingo.Symbol, nx.DiGraph]]:
    explanations_graphs: Dict[FrozenSet[clingo.Symbol], Dict[clingo.Symbol, nx.DiGraph]] = dict()



    return explanations_graphs
