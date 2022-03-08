import functools
import os.path
import subprocess
import sys
from pathlib import Path
from typing import Union

import clingo.ast

from util.literal import Literal


def file_to_aspif(path):
    p = Path(path)
    output = subprocess.run(['gringo', '--warn', 'none', p], stdout=subprocess.PIPE)
    return output.stdout.decode('utf-8')


def program_str_to_aspif(program: str):
    output = None
    try:
        output = subprocess.run(['gringo', '--warn', 'none'], input=program.encode('utf-8'), capture_output=True)
        if output.returncode != 0:
            raise Exception()

    except (subprocess.CalledProcessError, Exception):
        print(program, file=sys.stderr)
        print()
        if output is not None:
            print(output.stdout, file=sys.stderr)
            print(output.stderr, file=sys.stderr)
        raise
    return output.stdout.decode('utf-8')


def prepare_program(program: Union[str, Path]):
    if (isinstance(program, str) and os.path.isfile(program)) or isinstance(program, Path):
        statements = _prepare_program_file(program)
    else:
        statements = _prepare_program_str(program)
    return '\n'.join(map(str, statements))


def _prepare_program_file(path):
    statements = []
    clingo.ast.parse_files((str(path),), functools.partial(_prepare_statement, statements=statements))
    return statements


def _prepare_program_str(program):
    statements = []
    clingo.ast.parse_string(program, functools.partial(_prepare_statement, statements=statements))
    return statements


def _prepare_statement(statement: clingo.ast.AST, statements):
    if statement.ast_type != clingo.ast.ASTType.Rule:
        if statement.ast_type != clingo.ast.ASTType.Program:
            statements.append(str(statement))
        else:
            loc, program, seq = statement.values()
            if program != 'base':
                statements.append(str(statement))
    else:
        loc, head, body = statement.values()
        if not body:
            clingo.ast.parse_string("#external {}. [true]".format(head),
                                    functools.partial(_prepare_statement, statements=statements))
        else:
            statements.append(statement)


def process_aspif(aspif: str, program_dict=None, literal_dict=None):
    if program_dict is None:
        program_dict = {}
    if literal_dict is None:
        literal_dict = {}
    for line in aspif.split('\n'):
        process_aspif_line(line, program_dict, literal_dict)
    return program_dict, literal_dict


def process_aspif_line(aspif: str, program_dict, literal_dict):
    asp = aspif.split(' ')
    if not asp:
        return  # Did not parse anything
    elif asp[0] == '0':
        return  # End of file
    elif asp[0] == '':
        return  # Blank line
    statement_type = asp[0]
    statement = asp[1:]
    if statement_type == 'asp':  #
        return  # ASP Version Info
    elif statement_type == '1':
        _process_aspif_rule(statement, program_dict, literal_dict)
    elif statement_type == '4':
        _process_aspif_show(statement, program_dict, literal_dict)
    elif statement_type == '5':
        _process_aspif_external(statement, program_dict, literal_dict)


def _process_aspif_rule(statement: list, program_dict: dict, literal_dict: dict):
    h = int(statement[0])
    if h == 1:
        raise Exception("Unsupported language construct: Choice rules are not supported")
    assert h == 0, "Unexpected language construct"
    m = int(statement[1])
    if m != 1:
        raise Exception("Unsupported language construct: Multiple head atoms are not supported")
    assert m == 1, "Unexpected language construct"
    head_atoms = [int(a) for a in statement[2:2 + m]]
    body_start = m + 2
    b = int(statement[body_start])
    n = int(statement[body_start + 1])
    if b == 1:
        raise Exception("Unsupported language construct: Weight bodies are not supported")
    assert b == 0, "Unexpected language construct"
    body_literals = [int(l) for l in statement[body_start + 2:]]

    head = head_atoms[0]
    if head in literal_dict:
        head = literal_dict[head]

    positive = []
    negative = []
    for literal in body_literals:
        a = abs(literal)
        if a in literal_dict:
            a = literal_dict[a]
        if literal < 0:
            negative.append(a)
        else:
            assert literal > 0
            positive.append(a)
    if head not in program_dict:
        program_dict[head] = []
    program_dict[head].append(dict(positive=positive, negative=negative))


def _process_aspif_show(statement: list, program_dict: dict, literal_dict: dict):
    m = statement[0]
    s = statement[1]
    n = int(statement[2])
    if n == 0:
        raise Exception("Unsupported language construct: Empty show statements are not supported")
    elif n > 1:
        raise Exception("Unsupported language construct: Multi-show statements are not supported")
    assert n == 1, "Unexpected language construct"
    literals = [int(l) for l in statement[3:]]
    literal_index = literals[0]
    literal = Literal(name=s)
    literal_dict[literal_index] = literal
    if literal_index in program_dict:
        program_dict[literal] = program_dict[literal_index]
        program_dict.pop(literal_index)
    for head, bodies in program_dict.items():
        for body in bodies:
            if literal_index in body['positive']:
                positive: list = body['positive']
                i = positive.index(literal_index)
                positive[i] = literal
            elif literal_index in body['negative']:
                negative: list = body['negative']
                i = negative.index(literal_index)
                negative[i] = literal


def _process_aspif_external(statement: list, program_dict: dict, literal_dict: dict):
    literal = int(statement[0])
    v = int(statement[1])

    if literal in literal_dict:
        literal = literal_dict[literal]
    program_dict[literal] = [dict(positive=[], negative=[])]
