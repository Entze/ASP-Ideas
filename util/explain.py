import operator


def _preprocess(program_dict: dict, facts: set, answer_set: set):
    derivable_dict = dict()
    for head, bodies in program_dict.items():
        if head in answer_set:
            derivable_dict[head] = []
            for body in bodies:
                _true_atoms_process(head, body.get('positive', list()), body.get('negative', list()),
                                    derivable_dict, answer_set, facts)
        else:
            derivable_dict[-head] = []
            for body in bodies:
                _false_atoms_process(head, body.get('positive', list()), body.get('negative', list()),
                                     derivable_dict, answer_set)
    return derivable_dict


def _true_atoms_process(head: int, positive_symbols: list, negative_symbols: list, deriveable_dict: dict,
                        answer_set: set, facts: set):
    positive_rules_are_subset = True
    for positive_symbol in positive_symbols:
        positive_rules_are_subset = positive_rules_are_subset and positive_symbol in answer_set
        if not positive_rules_are_subset:
            break
    negative_rules_are_complement = True
    for negative_symbol in negative_symbols:
        negative_rules_are_complement = negative_rules_are_complement and negative_symbol not in answer_set
        if not negative_rules_are_complement:
            break
    if positive_rules_are_subset and negative_rules_are_complement:
        minimal_set = set(positive_symbols).union(map(operator.neg, negative_symbols))  # TODO: better name
        if head in facts and not positive_symbols and not negative_symbols:
            minimal_set = {"T"}
        deriveable_dict[head].append(minimal_set)


def _false_atoms_process(head: int, positive_rules: list, negative_rules: list, derivable_dict: dict,
                         answer_set: set):
    assert isinstance(derivable_dict, dict)
    answer_set_complement = []
    for p in positive_rules:
        if p not in answer_set:
            answer_set_complement.append({-p})
    for n in negative_rules:
        if n in answer_set and {n} not in answer_set_complement:
            answer_set_complement.append({n})
    if not derivable_dict[-head]:
        derivable_dict[-head] = answer_set_complement
    else:
        t_cap = []  # TODO: better name
        for singleton in answer_set_complement:
            for element in derivable_dict[-head]:
                t = singleton.union(element)
                t_cap.append(t)
        derivable_dict[-head] = t_cap


def negation_atoms(program_dict: dict) -> set:
    nant = set()
    for head, bodies in program_dict.items():
        for body in bodies:
            if 'negative' in body:
                nant.update(body['negative'])
    return nant


def assumption_func(cautious_consequences, nant, derivable_dict, answer_set):
    tu = set()
    tentative_assumptions = set()
    for atom in nant:
        if atom not in answer_set and atom not in cautious_consequences:
            tentative_assumptions.add(atom)
    t, da = _derivation_path(tentative_assumptions, derivable_dict)
    d = _dependency_assumption(da)
    for m in d:
        u = m.union(t)
        tu.add({u})
    return tu


def _derivation_path(tentative_assumptions, derivable_dict):
    da = dict()
    t = set()
    for atom in tentative_assumptions:
        O = tentative_assumptions.copy()
        O.discard(atom)
        L = derivable_dict[-atom]
        for S in L:
            E_a = dict()
            E_a[-atom] = list(S)
            _get_connection(S, derivable_dict, E_a)
            M = set()
            for k in E_a:
                for V in E_a[k]:
                    pass
            # TODO: Untangle the paper
    return None, None


def _dependency_assumption(dependency_assumption):
    dc = set()
    for u in dependency_assumption:
        pass  # TODO: Untangle the paper
    pass


def _get_connection(S, derivable_dict, E_a):
    for e in S:
        if e not in E_a and e in derivable_dict:
            E_a[e] = derivable_dict[e]
            for S_e in derivable_dict[e]:
                _get_connection(S_e, derivable_dict, E_a)


def _check_derivation_path(k, N, O, V, R, C, dependency_assumptions):
    I = N.get(k, set())
    V.add(k)
    R.add(k)
    for i in I:
        C[k] = i
        j = 0 # TODO: How to calculate j?
        if i < 0 and j in O:
            dependency_assumptions.add(j)
        else:
            if i not in V:
                if not _check_derivation_path(i, N, O, V, R, C, dependency_assumptions)[0]:
                    return False, dependency_assumptions
            else:
                if i in R:
                    
    return None, None


def _cycle_identification(C, s, e):
    v = C[s]
    if v != e and s < 0 and v < 0:
        return _cycle_identification(C, v, e)
    return v == e and s < 0 and v < 0


def explanation_graph(atom, derivable_dict, minimal_assumption, answer_set):
    for assumed_atom in minimal_assumption:
        derivable_dict[-assumed_atom] = ["assume"]
    atom = atom if atom in answer_set else -atom
    L = derivable_dict[atom]
    for S in L:
        E_a = dict()
        E_a[atom] = [S]
        _get_connection(S, derivable_dict, E_a)
        # TODO: Untangle paper


def get_graph(k, G, N, V:set, R:list, C):
    I = N.get(k, set())
    V.add(k)
    R.append(k)
    for i in I:
        G.add(i)
        e = (k, i, 0)  # TODO: 0 should be sign, but what is sign?
        G.add(e)       # TODO: G needs to be a graph datastructure networkx?
        if i not in V:
            if not get_graph(i, G, N, V, R, C):
                return False
        else:
            if i in R:
                if i < 0:
                    if not _cycle_identification(C, i, i):
                        return False
                else:
                    return False

    l = R.pop()
    if l in C:
        C.discard(l)
    return G

