from copy import deepcopy

import networkx as nx

from util.explain import preprocess, negation_atoms, _derivation_path, _dependency_assumption, explanation_graph

"""
asp 1 0 0
5 1 2
1 0 1 2 0 1 -3
1 0 1 4 0 2 -2 1
1 0 1 3 0 2 -2 4
1 0 1 5 0 2 2 3
1 0 1 5 0 1 4
1 0 1 6 0 3 -5 -4 1
4 1 b 1 2
4 1 k 1 4
4 1 a 1 3
4 1 e 1 1
4 1 c 1 5
4 1 f 1 6
0
"""

example_literal_mapping = dict(b=2, k=4, a=3, e=1, c=5, f=6)
a, b, c, e, f, k = example_literal_mapping['a'], example_literal_mapping['b'], example_literal_mapping['c'], \
                   example_literal_mapping['e'], example_literal_mapping['f'], example_literal_mapping['k']

example_program = {
    e: [dict()],
    b: [dict(positive=[], negative=[a])],
    k: [dict(positive=[e], negative=[b])],
    a: [dict(positive=[k], negative=[b])],
    c: [dict(positive=[a, b], negative=[]), dict(positive=[k], negative=[])],
    f: [dict(positive=[e], negative=[k, c])]
}

example_facts = {e}

example_answer_set = {b, f, e}

example_derivable_dict = {
    f: [{-k, -c, e}],
    b: [{-a}],
    -c: [{-k, -a}],
    -k: [{b}],
    -a: [{-k}, {b}],
    e: [{"T"}]
}


def preprocess_test():
    ep = preprocess(deepcopy(example_program), deepcopy(example_facts), deepcopy(example_answer_set))
    assert f in ep
    assert len(ep[f]) == 1, f"{len(ep[f])} != 1"
    assert sorted(ep[f][0]) == sorted({-k, -c, e}), f"{sorted(ep[f][0])} != {sorted({-k, -c, e})}"

    assert b in ep
    assert len(ep[b]) == 1, f"{len(ep[b])} != 1"
    assert sorted(ep[b][0]) == sorted({-a}), f"{sorted(ep[b][0])} != {sorted({-a})}"

    assert -c in ep
    assert len(ep[-c]) == 1, f"{ep[-c]} != 1"
    assert sorted(ep[-c][0]) == sorted({-k, -a}), f"{sorted(ep[-c][0])} != {sorted({-k, -a})}"

    assert -k in ep
    assert len(ep[-k]) == 1, f"{ep[-k]} != 1"
    assert sorted(ep[-k][0]) == sorted({b}), f"{sorted(ep[-k][0])} != {sorted({b})}"

    assert -a in ep
    assert len(ep[-a]) == 2, f"{len(ep[-a])} != 2"
    assert sorted(ep[-a][0]) == sorted({-k}), f"{sorted(ep[-a][0])} != {sorted({-k})}"
    assert sorted(ep[-a][1]) == sorted({b}), f"{sorted(ep[-a][1])} != {sorted({b})}"

    assert e in ep
    assert len(ep[e]) == 1, f"{len(ep[e])} != 1"
    assert sorted(ep[e][0]) == sorted({"T"}), f"{sorted(ep[e][0])} != {sorted({'T'})}"

    assert len(ep) == 6, f"{len(ep)} != 6"


def negation_atoms_test():
    nant = negation_atoms(deepcopy(example_program))
    assert sorted(nant) == sorted({a, b, c, k}), f"{sorted(nant)} != f{sorted({a, b, c, k})}"


def derivation_path_test():
    TA = {c, a, k}
    T, DA = _derivation_path(TA, deepcopy(example_derivable_dict))
    assert not T
    expected_DA = {a: {k}, k: {a}, c: {k, a}}
    assert DA == expected_DA, f"{DA} != {expected_DA}"


def assumption_func_test():
    T, DA = set(), {a: {k}, k: {a}, c: {k, a}}
    D = _dependency_assumption(DA)
    assert D == {frozenset({a}), frozenset({k})}, f"{D} != {({frozenset({a}), frozenset({k})})}"


def explanation_graph_test():
    graph = explanation_graph(f, deepcopy(example_derivable_dict), {a}, {b, f, e})
    assert graph is not None
    nx.draw(graph)


preprocess_test()
negation_atoms_test()
derivation_path_test()
assumption_func_test()
explanation_graph_test()
