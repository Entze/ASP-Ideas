from util.explain import Literal


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
    if n != 1:
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
    if v != 2:
        raise Exception("Unsupported language construct: Only allowed to use true external facts")
    assert v == 2, "Unexpected language construct"

    if literal in literal_dict:
       literal = literal_dict[literal]
    program_dict[literal] = [dict(positive=[], negative=[])]
