import datetime
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Sequence, Union, Optional

import clingo
import networkx as nx
from clingo import SymbolType, Symbol

from util.convert import program_str_to_aspif, process_aspif, prepare_program, preground
from util.display import symbol_to_str
from util.explain import preprocess, get_minimal_assumptions, negation_atoms, explanation_graph
from util.literal import Literal


def solve(programs, clingo_args=("--models", "0"), grounding_context=None, filter_symbols=None, report=True, sep=' '):
    ctl = clingo.Control(clingo_args)
    if isinstance(programs, str):
        if os.path.isfile(programs):
            # program is a path
            ctl.load(programs)
        else:
            ctl.add("base", [], programs)
    elif isinstance(programs, Path):
        ctl.load(str(programs))
    elif isinstance(programs, Iterable):
        for program in programs:
            if isinstance(program, str):
                if os.path.isfile(program):
                    ctl.load(program)
                else:
                    ctl.add("base", [], program)
            elif isinstance(program, Path):
                ctl.load(str(program))

    ctl.ground([("base", [])], context=grounding_context)

    with ctl.solve(yield_=True, async_=True) as solver:

        done = False
        i = 0
        calc_time = 0.
        models = []
        answer_sets = []

        while not done:
            stime = time.monotonic()
            solver.resume()
            _ = solver.wait()
            delta_time = time.monotonic() - stime
            calc_time += delta_time
            model = solver.model()
            done = model is None
            if not done:
                i += 1
                models.append(model)
                symbols = model.symbols(shown=True)
                answer_set = set()
                if filter_symbols is not None:
                    if isinstance(filter_symbols, list):
                        for filter_symbol in filter_symbols:
                            for symbol in symbols:
                                if compare_symbols(filter_symbol, symbol):
                                    answer_set.add(symbol)
                    else:
                        for symbol in symbols:
                            if compare_symbols(filter_symbols, symbol):
                                answer_set.add(symbol)
                else:
                    answer_set = set(symbols)

                answer_sets.append(answer_set)
                if report:
                    if delta_time > 0.01:
                        time_out = " in {}".format(datetime.timedelta(seconds=calc_time))
                    else:
                        time_out = ""
                    cost = model.cost
                    optimal = model.optimality_proven
                    if cost:
                        cost_out = "@{}{}".format(cost, " optimal" if optimal else "")
                    else:
                        cost_out = ""

                    print(
                        "Answer {:2d}: {}{}{}{}.{}".format(model.number, "{" + sep,
                                                           sep.join(map(symbol_to_str, sorted(answer_set))),
                                                           sep + "}",
                                                           cost_out, time_out))
        result = solver.get()
        if result.unknown:
            status_out = "UNKWN"
        elif result.satisfiable:
            status_out = "SAT"
        else:
            assert result.unsatisfiable
            status_out = "UNSAT"

        if result.interrupted:
            status_out += " (INTERRUPTED)"

        nr_of_solutions_out = str(i)
        if not result.exhausted:
            nr_of_solutions_out += "+"

        if calc_time > 0.1:
            time_out = datetime.timedelta(seconds=calc_time)
        else:
            time_out = ""
        if report:
            print(status_out, nr_of_solutions_out, time_out)
        return answer_sets, None


def compare_symbols(filter_symbol, symbol: Symbol):
    symbol_name = None
    symbol_type = None
    predicate_nr_of_args = None
    if isinstance(filter_symbol, Symbol):
        return filter_symbol == symbol
    elif isinstance(filter_symbol, str):
        if filter_symbol.endswith("."):
            symbol_name, predicate_nr_of_args = filter_symbol.split('/')
            predicate_nr_of_args = int(predicate_nr_of_args[0:-2])
        else:
            symbol_name = filter_symbol
    elif isinstance(filter_symbol, SymbolType):
        symbol_type = filter_symbol
    elif isinstance(filter_symbol, tuple):
        for e in filter_symbol:
            if isinstance(e, str):
                symbol_name = e
            elif isinstance(e, SymbolType):
                symbol_type = e
            elif isinstance(e, int):
                predicate_nr_of_args = e
    return (symbol_name is not None or symbol_type is not None or predicate_nr_of_args is not None) and (
            symbol_name is None or symbol.name == symbol_name) and (
                   symbol_type is None or symbol.type == symbol_type) and (
                   predicate_nr_of_args is None or predicate_nr_of_args == len(symbol.arguments))


def explain(programs: Union[Path, str, Sequence[Path], Sequence[str]],
            clingo_args=(),
            grounding_context=None,
            answer_set: Sequence[Union[clingo.Symbol, Literal]] = None,
            cautious_consequence: Sequence[Union[clingo.Symbol, Literal]] = None,
            root=None) -> Sequence[Optional[nx.DiGraph]]:
    if answer_set is None:
        answer_sets, _ = solve(programs, clingo_args=clingo_args, grounding_context=grounding_context, report=False)
        if answer_sets:
            answer_set = deepcopy(answer_sets[0])
        else:
            raise Exception("No answer set")
        if cautious_consequence is None and answer_sets:
            cautious_consequence = deepcopy(answer_sets[0])
            cautious_consequence.intersection_update(*answer_sets)

    answer_set_literals = set()
    for symbol in answer_set:
        literal = Literal.to_literal(symbol)
        if root is None:
            root = literal
        answer_set_literals.add(literal)

    if isinstance(programs, Sequence) and not isinstance(programs, str):
        pregrounded_program = '\n'.join(preground(program) for program in programs)
    else:
        pregrounded_program = preground(programs)

    prepared_program = prepare_program(pregrounded_program)
    aspif = program_str_to_aspif(prepared_program)

    program_dict, literal_dict = process_aspif(aspif)
    facts = {head for head in program_dict if
             len(program_dict[head]) == 0 or (len(program_dict[head]) == 1 and not any(
                 program_dict[head][0].values()))}

    derivable_dict = preprocess(program_dict=program_dict,
                                facts=facts,
                                answer_set=answer_set_literals)
    if cautious_consequence is None:
        cautious_consequence_literals = deepcopy(answer_set_literals)
    else:
        cautious_consequence_literals = {Literal.to_literal(symbol) for symbol in cautious_consequence}

    minimal_assumptions = tuple(
        get_minimal_assumptions(cautious_consequence_literals, negation_atoms(program_dict), deepcopy(derivable_dict),
                                answer_set_literals))

    return tuple(explanation_graph(root, deepcopy(derivable_dict), minimal_assumption, answer_set_literals) for
                 minimal_assumption in
                 minimal_assumptions)
