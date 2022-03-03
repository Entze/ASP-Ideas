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
from util.explain import _preprocess, negation_atoms

example_literal_mapping = dict(b=2, k=4, a=3, e=1, c=5, f=6)
a, b, c, e, f, k = example_literal_mapping['a'], example_literal_mapping['b'], example_literal_mapping['c'], \
                   example_literal_mapping['e'], example_literal_mapping['f'], example_literal_mapping['k']

example_program = {
    1: [dict()],
    2: [dict(positive=[], negative=[3])],
    4: [dict(positive=[1], negative=[2])],
    3: [dict(positive=[4], negative=[2])],
    5: [dict(positive=[2, 3], negative=[]), dict(positive=[1, 4], negative=[])],
    6: [dict(positive=[1], negative=[5, 4])]
}
example_facts = {e}

example_answer_set = {b, f, e}


def preprocess_test():
    ep = _preprocess(example_program, example_facts, example_answer_set)
    assert f in ep
    assert len(ep[f]) == 1, f"{len(ep[f])} != 1"
    assert sorted(ep[f][0]) == sorted({-k, -c, e}), f"{sorted(ep[f][0])} != " + "{" + ", ".join(
        map(str, sorted((-k, -c, e)))) + "}"

    assert b in ep
    assert len(ep[b]) == 1, f"{len(ep[b])} != 1"
    assert sorted(ep[b][0]) == sorted({-a}), f"{sorted(ep[b][0])} != " + "{" + ", ".join(map(str, sorted((-a,)))) + "}"

    assert -c in ep
    assert len(ep[-c]) == 1, f"{ep[-c]} != 1"
    assert sorted(ep[-c][0]) == sorted({-k, -a}), f"{sorted(ep[-c][0])} != " + "{" + ", ".join(
        map(str, sorted((-k, -a))))

    assert -k in ep
    assert len(ep[-k]) == 1, f"{ep[-k]} != 1"
    assert sorted(ep[-k][0]) == sorted({b}), f"{sorted(ep[-k][0])} != " + "{" + ", ".join(map(str, sorted((b,)))) + "}"

    assert -a in ep
    assert len(ep[-a]) == 2, f"{len(ep[-a])} != 2"
    assert sorted(ep[-a][0]) == sorted({-k}), f"{sorted(ep[-a][0])} != " + "{" + ", ".join(
        map(str, sorted((-k,)))) + "}"
    assert sorted(ep[-a][1]) == sorted({b}), f"{sorted(ep[-a][1])} != " + "{" + ", ".join(map(str, sorted((b,)))) + "}"

    assert e in ep
    assert len(ep[e]) == 1, f"{len(ep[e])} != 1"
    assert sorted(ep[e][0]) == sorted({"T"}), f"{sorted(ep[e][0])} != " + "{" + ", ".join(
        map(str, sorted(("T",)))) + "}"

    assert len(ep) == 6, f"{len(ep)} != 6"


def negation_atoms_test():
    nant = negation_atoms(example_program)
    assert sorted(nant) == sorted({a, b, c, k}), f"{sorted(nant)} != f{sorted({a, b, c, k})}"


preprocess_test()
negation_atoms_test()
