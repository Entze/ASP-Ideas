from copy import copy

from clingo import Symbol

from util.display import symbol_to_str


class Literal:
    def __init__(self, name, sign=1, preferred_negation_symbol=None):
        self.name = name
        self.sign = sign
        self._preferred_negation_symbol = preferred_negation_symbol

    @staticmethod
    def literal_from_string(literal):
        take_from = 0
        sign = 1
        preferred_negation_symbol = None
        if literal[0] in ('¬', '~', '!', '-'):
            take_from = 1
            sign = -1
            preferred_negation_symbol = literal[0]
        return Literal(name=literal[take_from:], sign=sign, preferred_negation_symbol=preferred_negation_symbol)

    @staticmethod
    def literal_from_symbol(symbol: Symbol):
        name = symbol_to_str(symbol)
        sign = -1 if symbol.negative else 1
        return Literal(name=name, sign=sign)

    @staticmethod
    def to_literal(obj):
        if isinstance(obj, Literal):
            return copy(obj)
        elif isinstance(obj, str):
            return Literal.literal_from_string(obj)
        elif isinstance(obj, Symbol):
            return Literal.literal_from_symbol(obj)
        raise TypeError("Cannot convert {} to Literal".format(type(obj).__name__))

    def is_negated(self):
        return self.sign < 0

    def get_tuple(self):
        return self.sign, self.name

    def __neg__(self):
        return Literal(name=self.name,
                       sign=-self.sign,
                       preferred_negation_symbol=self._preferred_negation_symbol or '-')

    def __invert__(self):
        return Literal(name=self.name,
                       sign=-self.sign,
                       preferred_negation_symbol=self._preferred_negation_symbol or '~')

    def lnot(self):
        return Literal(name=self.name,
                       sign=-self.sign,
                       preferred_negation_symbol=self._preferred_negation_symbol or '¬')

    def __abs__(self):
        return Literal(name=self.name,
                       sign=1,
                       preferred_negation_symbol=self._preferred_negation_symbol)

    def __eq__(self, other):
        if isinstance(other, Literal):
            return self.get_tuple() == other.get_tuple()
        elif isinstance(other, int):
            return False
        elif isinstance(other, str) and other.lower() in ('t', 'assume'):
            return False
        raise TypeError("Cannot compare Literal with {}".format(type(other).__name__))

    def __gt__(self, other):
        if isinstance(other, Literal):
            return self.get_tuple() > other.get_tuple()
        elif isinstance(other, int) and other == 0:
            return self.sign > 0
        raise TypeError("Cannot compare Literal with {}".format(type(other).__name__))

    def __ge__(self, other):
        if isinstance(other, Literal):
            return self.get_tuple() >= other.get_tuple()
        raise TypeError("Cannot compare Literal with {}".format(type(other).__name__))

    def __lt__(self, other):
        if isinstance(other, Literal):
            return self.get_tuple() < other.get_tuple()
        elif isinstance(other, int) and other == 0:
            return self.sign < 0
        raise TypeError("Cannot compare Literal with {}".format(type(other).__name__))

    def __le__(self, other):
        if isinstance(other, Literal):
            return self.get_tuple() <= other.get_tuple()
        raise TypeError("Cannot compare Literal with {}".format(type(other).__name__))

    def __repr__(self):
        return "{}{}".format('~' if self.sign < 0 else "", self.name)

    def __str__(self):
        return "{}{}".format(self._preferred_negation_symbol or '¬' if self.sign < 0 else "", self.name)

    def __hash__(self):
        return hash(self.get_tuple())

    def __copy__(self):
        return Literal(name=self.name, sign=self.sign, preferred_negation_symbol=self._preferred_negation_symbol)
