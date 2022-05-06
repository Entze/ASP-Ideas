from typing import Optional

import clingo
import clingo.ast

program = """

%domA.

domA(0).

domA(1) :- domA(A).

"""


def report(node):
    print(node)
    if not isinstance(node, clingo.ast.AST):
        print("Type:", type(node).__name__)
        return

    print("Type:", node.ast_type)
    print("Keys:", node.keys())

    if 'head' in node.keys():
        print("Head:", end=' ')
        report(node.head)
    if 'body' in node.keys():
        print("Body:", end=' ')
        report(node.body)
    if 'atom' in node.keys():
        print("Atom: ", end='')
        report(node.atom)
    if 'symbol' in node.keys():
        print("Symbol: ", end='')
        report(node.symbol)
    if 'sign' in node.keys():
        print("Sign:", node.sign, type(node.sign).__name__)
    if 'external' in node.keys():
        print("External:", node.external, type(node.external).__name__)
    if 'name' in node.keys():
        print("Name:", node.name, type(node.name).__name__)
    if 'arguments' in node.keys():
        print("Arguments:")
        for argument in node.arguments:
            report(argument)


class NonGroundReifyTransformer(clingo.ast.Transformer):
    pos = clingo.ast.Position('<string>', 1, 1)
    loc = clingo.ast.Location(pos, pos)

    def _ast_sym(self, sym: clingo.Symbol):
        return clingo.ast.SymbolicTerm(self.loc, sym)

    def _ast_str(self, string: str):
        return self._ast_sym(clingo.String(string))

    def _ast_num(self, num: int):
        return self._ast_sym(clingo.Number(num))

    def visit_SymbolicTerm(self, term: clingo.ast.AST):
        print("Term:", term)
        symbol = term.symbol
        symbol_node = clingo.ast.Function(self.loc, 'symbol',
                                          [self._ast_sym(symbol)],
                                          False)
        print("Meta-Term:", symbol_node)
        return symbol_node

    def visit_Variable(self, variable: clingo.ast.AST):
        print("Variable:", variable)
        name = variable.name
        variable_node = clingo.ast.Function(self.loc, 'variable', [
            clingo.ast.Function(self.loc, 'name', [
                self._ast_str(name)
            ], False)], False)
        print("Meta-Variable:", variable_node)
        return variable_node

    def visit_Function(self, function: clingo.ast.AST):
        print("Function:", function)
        name = function.name
        external = function.external
        meta_function = self.visit_children(function)
        arguments = meta_function.get('arguments', ())
        arguments_sub = None
        if arguments:
            arguments_sub = clingo.ast.Function(self.loc, 'arguments', arguments, False)
        else:
            arguments_sub = self._ast_sym(clingo.Function('arguments'))
        function_node = clingo.ast.Function(self.loc, 'function', [
            clingo.ast.Function(self.loc, 'name', [self._ast_str(name)], False),
            arguments_sub,
            clingo.ast.Function(self.loc, 'external', [self._ast_num(external)], False)
        ], False)
        print("Meta-Function:", function_node)
        return function_node

    def visit_Literal(self, literal: clingo.ast.AST):
        print("Literal:", literal)
        sign = literal.sign
        meta_literal = self.visit_children(literal)
        atom = meta_literal['atom'].symbol

        literal_node = clingo.ast.Function(self.loc, 'literal', [
            clingo.ast.Function(self.loc, 'sign', [self._ast_num(sign)], False),
            clingo.ast.Function(self.loc, 'atom', [
                clingo.ast.Function(self.loc, 'symbol', [
                    atom
                ], False),
            ], False)
        ], False)
        print("Meta-Literal:", literal_node)
        return literal_node


    def visit_Rule(self, rule: clingo.ast.AST):
        print("Rule:", rule)
        meta_rule = self.visit_children(rule)
        head = clingo.ast.Function(self.loc, 'head', [meta_rule['head']], False)
        body = clingo.ast.Function(self.loc, 'body', [
            clingo.ast.Function(self.loc, 'elements', meta_rule.get('body', ()), False)], False)
        rule_head = clingo.ast.Literal(self.loc, clingo.ast.Sign.NoSign, clingo.ast.SymbolicAtom(
            clingo.ast.Function(self.loc, 'rule', (head, body), False)))
        rule_node = clingo.ast.Rule(self.loc, rule_head, ())
        print("Meta-Rule:", rule_node)
        return rule_node


ctl = clingo.Control()
ctl.configuration.solve.models = 0

ngrt = NonGroundReifyTransformer()

nodes = []

clingo.ast.parse_string(
    'rule(head(literal(sign(0),atom(symbol(function(name("domA"),arguments(symbol(0)),external(0)))))),body(elements)).',
     report)

clingo.ast.parse_string(program, lambda stm: nodes.append(ngrt.visit(stm)))
print("-" * 80)
with clingo.ast.ProgramBuilder(ctl) as bld:
    for node in nodes:
        print("#" * 80)
        report(node)
        print("#" * 80)
        print(node.ast_type)
        bld.add(node)
        print(node)

ctl.ground([('base', [])])
with ctl.solve(yield_=True) as solve_handle:
    models = []
    for model in solve_handle:
        symbols = model.symbols(atoms=True)
        models.append(symbols)
print(' '.join(map(str, models[0])))

pos = clingo.ast.Position('<string>', 1, 1)
loc = clingo.ast.Location(pos, pos)
_false = clingo.ast.BooleanConstant(False)


def translate_variable(variable: clingo.Symbol):
    print(variable)
    assert variable.match('variable', 1)
    name = variable.arguments[0]
    name_node = translate_name(name)
    variable_node = clingo.ast.Variable(loc, name_node)
    return variable_node


def translate_term(term: clingo.Symbol):
    print(term)
    assert term.match('term', 1)
    symbol = term.arguments[0]
    term_node = translate_symbol(symbol)
    return term_node


def translate_arguments(arguments: clingo.Symbol):
    print(arguments)
    assert arguments.type is clingo.SymbolType.Function and arguments.name == 'arguments'
    arguments_node = tuple(translate(argument) for argument in arguments.arguments)
    return arguments_node


def translate_elements(elements: clingo.Symbol):
    print(elements)
    assert elements.type is clingo.SymbolType.Function and elements.name == 'elements'
    elements_node = tuple(translate(argument) for argument in elements.arguments)
    return elements_node


def translate_name(name: clingo.Symbol):
    print(name)
    assert name.match('name', 1)
    assert name.arguments[0].type is clingo.SymbolType.String
    name_node = name.arguments[0].string
    return name_node


def translate_external(external: clingo.Symbol):
    print(external)
    assert external.match('external', 1)
    assert external.arguments[0].type is clingo.SymbolType.Number
    external_node = external.arguments[0].number
    return external_node


def translate_function(function: clingo.Symbol):
    print(function)
    assert function.match('function', 3)
    name = function.arguments[0]
    arguments = function.arguments[1]
    external = function.arguments[2]
    name_node = translate_name(name)
    arguments_node = translate_arguments(arguments)
    external_node = translate_external(external)
    function_node = clingo.ast.Function(loc, name_node, arguments_node, external_node)
    return function_node


def translate_symbol(symbol: clingo.Symbol):
    print(symbol)
    assert symbol.match('symbol', 1)
    child = symbol.arguments[0]
    child_node = translate(child)
    return child_node


def translate_atom(atom: clingo.Symbol):
    print(atom)
    assert atom.match('atom', 1)
    symbol = atom.arguments[0]
    symbol_node = translate_symbol(symbol)
    atom_node = clingo.ast.SymbolicAtom(symbol_node)
    return atom_node


def translate_sign(sign: clingo.Symbol):
    print(sign)
    assert sign.match('sign', 1)
    sign_node = sign.arguments[0].number
    return sign_node


def translate_literal(literal: clingo.Symbol):
    print(literal)
    assert literal.match('literal', 2)
    sign = literal.arguments[0]
    atom = literal.arguments[1]
    sign_node = translate_sign(sign)
    atom_node = translate_atom(atom)
    literal_node = clingo.ast.Literal(loc, sign_node, atom_node)
    return literal_node


def translate_head(head: Optional[clingo.Symbol] = None):
    print(head)
    assert head.match('head', 1)
    child = head.arguments[0]
    head_node = translate(child)
    return head_node


def translate_body(body: Optional[clingo.Symbol] = None):
    print(body)
    assert body.match('body', 1)
    elements = body.arguments[0]
    elements_node = translate_elements(elements)
    return elements_node


def translate_rule(rule: clingo.Symbol):
    print(rule)
    assert rule.match('rule', 2)
    head = rule.arguments[0]
    body = rule.arguments[1]
    head_node = translate_head(head)
    body_node = translate_body(body)
    rule_node = clingo.ast.Rule(loc, head_node, body_node)
    return rule_node


def translate(symbol: clingo.Symbol):
    print(symbol)
    if symbol.type is clingo.SymbolType.Function:
        if symbol.name == 'literal':
            return translate_literal(symbol)
        elif symbol.name == 'function':
            return translate_function(symbol)
        elif symbol.name == 'term':
            return translate_term(symbol)
        elif symbol.name == 'variable':
            return translate_variable(symbol)
        elif symbol.name == 'symbol':
            return translate_symbol(symbol)
    else:
        return clingo.ast.SymbolicTerm(loc, symbol)

#%%

prg = []

for symbol in models[0]:
    prg.append(translate_rule(symbol))

print("-"*80)

print('\n'.join(map(str, prg)))

