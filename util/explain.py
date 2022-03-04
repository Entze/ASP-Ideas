import itertools


def _preprocess(program_dict: dict, facts: set, answer_set: set) -> dict:
    derivable_dict = dict()
    for head, bodies in program_dict.items():
        # print(f"[_preprocess]: {head} start.")
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
        # print(f"[_preprocess]: {head} done.")
    return derivable_dict


def _true_atoms_process(head: int, positive_symbols: list, negative_symbols: list, deriveable_dict: dict,
                        answer_set: set, facts: set):
    if all(p in answer_set for p in positive_symbols) and not any(n in answer_set for n in negative_symbols):
        derivable = set(positive_symbols) | set(-n for n in negative_symbols)
        if head in facts and not positive_symbols and not negative_symbols:
            derivable = {"T"}
        deriveable_dict[head].append(derivable)


def _false_atoms_process(head: int, positive_rules: list, negative_rules: list, derivable_dict: dict,
                         answer_set: set):
    assert isinstance(derivable_dict, dict)
    derivable = list(set(frozenset({-p}) for p in positive_rules if p not in answer_set) | set(
        frozenset({n}) for n in negative_rules if n in answer_set))
    if not derivable_dict[-head]:
        derivable_dict[-head] = derivable
    else:
        derivables = []
        for new_derivables in derivable:
            for old_derivables in derivable_dict[-head]:
                derivables.append(new_derivables | old_derivables)
        derivable_dict[-head] = derivables


def negation_atoms(program_dict: dict) -> set:
    nant = set()
    for head, bodies in program_dict.items():
        for body in bodies:
            if 'negative' in body:
                nant.update(body['negative'])
    return nant


def _assumption_func(cautious_consequences, nant, derivable_dict, answer_set):
    assumptions = set()
    tentative_assumptions = set(a for a in nant if a not in answer_set and a not in cautious_consequences)
    # print(f"[_assumption_func]: tentative_assumptions = {sorted(tentative_assumptions)}.")
    terminals, dependency_assumptions = _derivation_path(tentative_assumptions, derivable_dict)
    # print(f"[_assumption_func]: terminals = {terminals}")
    # print(f"[_assumption_func]: dependency_assumptions = {dependency_assumptions}")
    minimal_assumptions = _dependency_assumption(dependency_assumptions)
    for minimal_assumption in minimal_assumptions:
        assumptions.add({minimal_assumption | terminals})
    return assumptions


def _derivation_path(tentative_assumptions, derivable_dict):
    dependency_assumptions = dict()
    terminals = set()
    for atom in tentative_assumptions:
        tentative_destinations = tentative_assumptions.copy() - {atom}
        safe = False
        for derivables in derivable_dict[-atom]:
            edge_dict = dict()
            edge_dict[-atom] = [derivables]
            _get_connection(derivables, derivable_dict, edge_dict)
            paths = [{}]
            for k in edge_dict:
                n_paths = []
                for neighbourhood in edge_dict[k]:
                    for _path in paths:
                        path = _path.copy()
                        path[k] = neighbourhood.copy()
                        n_paths.append(path)
                paths = n_paths

            for path in paths:
                vertex_set = set()
                reached_stack = []
                cycle_dict = dict()
                derivation_set = set()
                safe, derivation = _check_derivation_path(-atom, path, tentative_destinations, vertex_set,
                                                          reached_stack, cycle_dict,
                                                          derivation_set)
                if safe:
                    terminals.add(atom)
                    dependency_assumptions[atom] = derivation
                    break  #
            if safe:  # for the GOTO statement
                break

    return tentative_assumptions - terminals, dependency_assumptions


def _dependency_assumption(dependency_assumption: dict):
    directed_cycles = set()
    paths = [[]]
    nexts = [list(dependency_assumption.keys())]
    while nexts:
        while nexts[-1]:
            next = nexts[-1].pop()
            path = paths[-1].copy()
            path.append(next)
            neighbours = []
            for neighbour in dependency_assumption[next]:
                if neighbour in path:
                    if path[0] == neighbour:
                        directed_cycles.add(frozenset(path))
                else:
                    neighbours.append(neighbour)

            nexts.append(list(neighbour for neighbour in neighbours if neighbour not in path))
            paths.append(path)

        nexts.pop()
        paths.pop()

    return frozenset(map(frozenset, itertools.product(*directed_cycles)))


def _get_connection(derivables, derivable_dict, edge_dict):
    for next in derivables:
        # print(f"[_get_connection]: {next} start.")
        if next not in edge_dict and next in derivable_dict:
            edge_dict[next] = derivable_dict[next]
            for next_derivables in derivable_dict[next]:
                _get_connection(next_derivables, derivable_dict, edge_dict)  # TODO: untangle recursion
        # print(f"[_get_connection]: {next} done.")


def _check_derivation_path(k, path, tentative_destinations, vertex_set, reached_stack, cycle_dict,
                           dependency_assumptions):
    I = path.get(k, set())
    vertex_set.add(k)
    reached_stack.append(k)
    for i in I:
        cycle_dict[k] = i
        j = abs(i)
        if i < 0 and j in tentative_destinations:
            dependency_assumptions.add(j)
        else:
            if i not in vertex_set:
                if not \
                        _check_derivation_path(i, path, tentative_destinations, vertex_set, reached_stack,
                                               cycle_dict,
                                               dependency_assumptions)[0]:
                    return False, dependency_assumptions
            elif i in reached_stack:
                if i < 0:
                    if not _cycle_identification(cycle_dict, i, i):
                        return False, dependency_assumptions
                else:
                    return False, dependency_assumptions

    reached = reached_stack.pop()
    if reached in cycle_dict:
        del cycle_dict[reached]
    return True, dependency_assumptions


def _cycle_identification(cycle_dict, s, e):
    v = cycle_dict[s]
    if v != e and s < 0 and v < 0:
        return _cycle_identification(cycle_dict, v, e)
    return v == e and s < 0 and v < 0


def explanation_graph(atom, derivable_dict, minimal_assumption, answer_set):
    for assumed_atom in minimal_assumption:
        derivable_dict[-assumed_atom] = ["assume"]
    atom = atom if atom in answer_set else -atom
    for S in derivable_dict[atom]:
        E_a = dict()
        E_a[atom] = [S]
        _get_connection(S, derivable_dict, E_a)
        # TODO: Untangle paper


def get_graph(k, G, N, V: set, R: list, C):
    I = N.get(k, set())
    V.add(k)
    R.append(k)
    for i in I:
        G.add(i)
        e = (k, i, 0)  # TODO: 0 should be sign, but what is sign?
        G.add(e)  # TODO: G needs to be a graph datastructure networkx?
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
