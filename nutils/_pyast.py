from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from io import TextIOBase
import itertools
from types import MappingProxyType
import typing


class Expression:
    '''Expression which can be converted to Python code using `py_expr`.'''

    @property
    def py_expr(self) -> str:
        '''the expression in Python code'''

        raise NotImplementedError

    @property
    def variables(self) -> frozenset[Variable]:
        '''the set of `Variable`s used in this expression'''
        raise NotImplementedError

    @property
    def py_paren_expr(self) -> str:
        '''the expression in Python code enclosed in parenthesis if necessary'''

        return f'({self.py_expr})'

    def get_attr(self, attr: str):
        '''returns `self.{attr}` as an `Expression`'''

        return GetAttr(self, attr)

    def get_item(self, item: Expression) -> Expression:
        '''returns `self[item]` as an `Expression`'''

        return GetItem(self, item)

    def call(self, /, *args: Expression, **kwargs) -> Expression:
        '''return `self(*args, **kwargs)` as an `Expression`'''

        return Call(self, *args, **kwargs)


def _isinstance(obj, cls):
    origin = typing.get_origin(cls)
    if origin == typing.Union:
        return any(_isinstance(obj, arg) for arg in typing.get_args(cls))
    elif origin == tuple and typing.get_args(cls)[1:] == (...,):
        return isinstance(obj, tuple) and all(_isinstance(item, typing.get_args(cls)[0]) for item in obj)
    elif cls is None:
        return obj is None
    else:
        return isinstance(obj, cls)


def _dataclass_type_checker(cls):
    def __post_init__(self):
        for field in dataclasses.fields(cls):
            field_value = getattr(self, field.name)
            if not _isinstance(field_value, eval(field.type)):
                raise ValueError(f'expected {field.name!r} to be a {field.type} but got {field_value!r}')
    cls.__post_init__ = __post_init__
    return cls


@dataclass
@_dataclass_type_checker
class Raw(Expression):
    '''Raw Python code'''

    value: str

    @property
    def py_expr(self) -> str:
        return self.value

    @property
    def variables(self) -> frozenset[Variable]:
        return frozenset()


@dataclass(frozen=True, order=True)
@_dataclass_type_checker
class Variable(Expression):
    '''Python Variable'''

    name: str

    @property
    def py_expr(self) -> str:
        return self.name

    py_paren_expr = py_expr

    @property
    def variables(self):
        return frozenset({self})


@dataclass
@_dataclass_type_checker
class Tuple(Expression):
    '''Tuple of `Expression`s.'''

    items: typing.Tuple[Expression, ...]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index) -> Expression:
        return self.items[index]

    @property
    def py_expr(self) -> str:
        return '(' + ', '.join([item.py_expr for item in self.items]) + (',)' if len(self.items) == 1 else ')')

    py_paren_expr = py_expr

    @property
    def variables(self) -> frozenset[VariablVariable]:
        return frozenset().union(*(item.variables for item in self.items))


class Call(Expression):
    '''Function call.'''

    def __init__(self, func: Expression, /, *args: Expression, **kwargs: Expression) -> None:
        if not isinstance(func, Expression):
            raise ValueError('expected `func` to be an `Expression` but got {func!r}')
        for arg in itertools.chain(args, kwargs.values()):
            if not isinstance(func, Expression):
                raise ValueError('expected call argument to be an `Expression` but got {func!r}')
        self.func = func
        self.args = tuple(args)
        self.kwargs = MappingProxyType(kwargs)

    @property
    def py_expr(self) -> str:
        args = ', '.join([arg.py_expr for arg in self.args] + [f'{k}={v.py_expr}' for k, v in self.kwargs.items()])
        return f'{self.func.py_expr}({args})'

    py_paren_expr = py_expr

    @property
    def variables(self) -> frozenset[Variable]:
        return frozenset().union(self.func.variables, *(arg.variables for arg in self.args), *(arg.variables for arg in self.kwargs.values()))


@dataclass
@_dataclass_type_checker
class GetAttr(Expression):
    '''Attribute getter.'''

    value: Expression
    attr: str

    @property
    def py_expr(self) -> str:
        return f'{self.value.py_paren_expr}.{self.attr}'

    py_paren_expr = py_expr

    @property
    def variables(self) -> frozenset[Variable]:
        return self.value.variables


@dataclass
@_dataclass_type_checker
class GetItem(Expression):
    '''Item getter.'''

    value: Expression
    item: Expression

    @property
    def py_expr(self) -> str:
        if not isinstance(self.item, Tuple) or len(self.item) == 0:
            item = self.item.py_expr
        else:
            item = ''.join(f'{item.py_expr}, ' for item in self.item)
        return f'{self.value.py_paren_expr}[{item}]'

    py_paren_expr = py_expr

    @property
    def variables(self) -> frozenset[Variable]:
        return self.value.variables | self.item.variables


class _LiteralRepr(Expression):

    @property
    def py_expr(self) -> str:
        return repr(self.value)

    py_paren_expr = py_expr

    @property
    def variables(self) -> frozenset[Variable]:
        return frozenset()


@dataclass
@_dataclass_type_checker
class LiteralBool(_LiteralRepr):
    '''Literal `bool`.'''

    value: bool


@dataclass
@_dataclass_type_checker
class LiteralInt(_LiteralRepr):
    '''Literal `int`.'''

    value: int


@dataclass
@_dataclass_type_checker
class LiteralStr(_LiteralRepr):
    '''Literal `str`.'''

    value: str


@dataclass
@_dataclass_type_checker
class UnaryOp(Expression):
    '''Unary operation `{op}{rhs}`.'''
    op: str
    rhs: Expression

    @property
    def py_expr(self) -> str:
        return f'{self.op}{self.rhs.py_paren_expr}'

    @property
    def variables(self) -> frozenset[Variable]:
        return self.rhs.variables


@dataclass
@_dataclass_type_checker
class BinOp(Expression):
    '''Binary operation `{lhs} {op} {rhs}`.'''

    lhs: Expression
    op: str
    rhs: Expression

    @property
    def py_expr(self) -> str:
        return f'{self.lhs.py_paren_expr} {self.op} {self.rhs.py_paren_expr}'

    @property
    def variables(self) -> frozenset[Variable]:
        return self.lhs.variables | self.rhs.variables


class Statements:
    '''Statements which can be converted to Python code using `lines`.'''

    @property
    def lines(self) -> typing.Iterator[str]:
        '''Writes Python code to `script` with each line prefixed with `indent`.'''

        raise NotImplementedError

    def __bool__(self) -> bool:
        '''If False `write_py` writes nothing.'''

        raise NotImplementedError


class Block(Statements):
    '''List of statements.

    Generates

        {items[0]}
        {items[1]}
        ...
    '''

    def __init__(self, /, items = ()) -> None:
        self._items = list(items)

    @property
    def lines(self) -> typing.Iterator[str]:
        for item in self._items:
            yield from item.lines

    def __bool__(self) -> bool:
        return any(self._items)

    def append(self, item: Statements):
        '''Appends `item` the the list.'''

        self._items.append(item)


class If(Statements):
    '''If statement and body.

    Generates

        if {condition}:
            {body}

    or nothing if `body` is empty.
    '''

    def __init__(self, condition: Expression, body: Statements, else_body: Statements = None):
        self.condition = condition
        self.body = body
        self.else_body = else_body if else_body is not None else Block()
        super().__init__()

    @property
    def lines(self) -> typing.Iterator[str]:
        if self:
            yield f'if {self.condition.py_expr}:'
            if self.body:
                for line in self.body.lines:
                    yield '    ' + line
            else:
                yield '    pass'
            if self.else_body:
                yield 'else:'
                for line in self.else_body.lines:
                    yield '    ' + line

    def __bool__(self):
        return bool(self.body) or bool(self.else_body)


@dataclass
@_dataclass_type_checker
class ForLoop(Statements):
    '''For loop and body.

    Generates

        for {var} in {iterable}:
            {body}

    or nothing if `body` is empty.
    '''

    var: Expression
    iterable: Expression
    body: Statements

    @property
    def lines(self) -> typing.Iterator[str]:
        if self:
            yield f'for {self.var.py_expr} in {self.iterable.py_expr}:'
            for line in self.body.lines:
                yield '    ' + line

    def __bool__(self):
        return bool(self.body)


@dataclass
@_dataclass_type_checker
class With(Statements):
    '''With statement and body.

    Generates nothing if `body` is empty and `omit_if_body_is_empty is true.
    Otherwise if `body` is empty, `body` is replaced with `pass` and generates

        with {item} as {as_}:
            {body}

    or

        with {item}:
            {body}

    if `as_` is `None`.
    '''

    item: Expression
    body: Statements
    as_: typing.Optional[Expression] = None
    omit_if_body_is_empty: bool = False

    @property
    def lines(self) -> typing.Iterator[str]:
        if self:
            if self.as_:
                yield f'with {self.item.py_expr} as {self.as_.py_expr}:'
            else:
                yield f'with {self.item.py_expr}:'
            if self.body:
                for line in self.body.lines:
                    yield '    ' + line
            else:
                yield '    pass'

    def __bool__(self):
        return not self.omit_if_body_is_empty or bool(self.body)


@dataclass
@_dataclass_type_checker
class Assign(Statements):
    '''Assignment.

    Generates

        {lhs} = {rhs}
    '''

    lhs: Expression
    rhs: Expression

    @property
    def lines(self) -> typing.Iterator[str]:
        yield f'{self.lhs.py_expr} = {self.rhs.py_expr}'

    def __bool__(self):
        return True


@dataclass
@_dataclass_type_checker
class Exec(Statements):
    '''Execute expression.

    Generates

        {expression}
    '''

    expression: Expression

    @property
    def lines(self) -> typing.Iterator[str]:
        yield f'{self.expression.py_expr}'

    def __bool__(self):
        return True


@dataclass
@_dataclass_type_checker
class Assert(Statements):
    '''Assertion.

    Generates

        assert {condition}
    '''

    condition: Expression

    @property
    def lines(self) -> typing.Iterator[str]:
        yield f'assert {self.condition.py_expr}'

    def __bool__(self):
        return True


@dataclass
@_dataclass_type_checker
class Raise(Statements):
    '''Raise statement.

    Generates

        raise {exception}
    '''

    exception: Expression

    @property
    def lines(self) -> typing.Iterator[str]:
        yield f'raise {self.exception.py_expr}'

    def __bool__(self):
        return True


@dataclass
@_dataclass_type_checker
class Global(Statements):
    '''Global statement.

    Generates

        global {variables[0]}, {variables[1]}, ...
    '''

    variables: typing.Tuple[Variable, ...]

    @property
    def lines(self) -> typing.Iterator[str]:
        if self:
            variables = ', '.join(var.py_expr for var in self.variables)
            yield f'global {variables}'

    def __bool__(self):
        return bool(self.variables)


@dataclass
@_dataclass_type_checker
class CommentBlock(Statements):
    '''Comment statement

    Generates

        # {comment}
        {statements.lines[0]}
        {statements.lines[1]}
        ...

    or

        {statements.lines[0]} # {comment}

    if `comment` contains no newline characters and `statements` consists of a
    single line.
    '''

    comment: str
    statements: Statements

    def _get_single_line_statement(self):
        line = None
        for i, line in enumerate(self.statements.lines):
            if i == 1:
                return None
        return line

    @property
    def lines(self) -> typing.Iterator[str]:
        if not self:
            pass
        elif '\n' not in self.comment and (line := self._get_single_line_statement()) is not None:
            yield f'{line} # {self.comment}'
        else:
            for line in self.comment.splitlines():
                yield f'# {line}'
            yield from self.statements.lines

    def __bool__(self):
        return bool(self.statements)
