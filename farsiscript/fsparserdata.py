from __future__ import unicode_literals

import sys
import unicodedata
from collections import defaultdict

PY3 = sys.version_info >= (3, 0)

if PY3:
    unichr = chr
    xrange = range
    unicode = str

token = {
    "BooleanLiteral": 1,
    "EOF": 2,
    "Identifier": 3,
    "Keyword": 4,
    "NullLiteral": 5,
    "NumericLiteral": 6,
    "Punctuator": 7,
    "StringLiteral": 8,
    "RegularExpression": 9,
    "Template": 10,
}

TokenName = dict((v, k) for k, v in token.items())

FnExprTokens = [
    "(",
    "{",
    "[",
    "dar",
    "typeof",
    "instanceof",
    "jadid",
    "bazgardan",
    "mored",
    "delete",
    "partkon",
    "void",
    "=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "<<=",
    ">>=",
    ">>>=",
    "&=",
    "|=",
    "^=",
    ",",
    "+",
    "-",
    "*",
    "/",
    "%",
    "++",
    "--",
    "<<",
    ">>",
    ">>>",
    "&",
    "|",
    "^",
    "!",
    "~",
    "&&",
    "||",
    "?",
    ":",
    "===",
    "==",
    ">=",
    "<=",
    "<",
    ">",
    "!=",
    "!==",
]

syntax = {
    "AssignmentExpression",
    "AssignmentPattern",
    "ArrayExpression",
    "ArrayPattern",
    "ArrowFunctionExpression",
    "BlockStatement",
    "BinaryExpression",
    "BreakStatement",
    "CallExpression",
    "CatchClause",
    "ClassBody",
    "ClassDeclaration",
    "ClassExpression",
    "ConditionalExpression",
    "ContinueStatement",
    "DoWhileStatement",
    "DebuggerStatement",
    "EmptyStatement",
    "ExportAllDeclaration",
    "ExportDefaultDeclaration",
    "ExportNamedDeclaration",
    "ExportSpecifier",
    "ExpressionStatement",
    "ForStatement",
    "ForInStatement",
    "FunctionDeclaration",
    "FunctionExpression",
    "Identifier",
    "IfStatement",
    "ImportDeclaration",
    "ImportDefaultSpecifier",
    "ImportNamespaceSpecifier",
    "ImportSpecifier",
    "Literal",
    "LabeledStatement",
    "LogicalExpression",
    "MemberExpression",
    "MethodDefinition",
    "NewExpression",
    "ObjectExpression",
    "ObjectPattern",
    "Program",
    "Property",
    "RestElement",
    "ReturnStatement",
    "SequenceExpression",
    "SpreadElement",
    "Super",
    "SwitchCase",
    "SwitchStatement",
    "TaggedTemplateExpression",
    "TemplateElement",
    "TemplateLiteral",
    "ThisExpression",
    "ThrowStatement",
    "TryStatement",
    "UnaryExpression",
    "UpdateExpression",
    "VariableDeclaration",
    "VariableDeclarator",
    "WhileStatement",
    "WithStatement",
}

messages = {
    "UnexpectedToken": "Unexpected token %s",
    "UnexpectedNumber": "Unexpected number",
    "UnexpectedString": "Unexpected string",
    "UnexpectedIdentifier": "Unexpected identifier",
    "UnexpectedReserved": "Unexpected reserved word",
    "UnexpectedTemplate": "Unexpected quasi %s",
    "UnexpectedEOS": "Unexpected end of input",
    "NewlineAfterThrow": "Illegal newline after throw",
    "InvalidRegExp": "Invalid regular expression",
    "UnterminatedRegExp": "Invalid regular expression: missing /",
    "InvalidLHSInAssignment": "Invalid left-hand side in assignment",
    "InvalidLHSInForIn": "Invalid left-hand side in for-in",
    "MultipleDefaultsInSwitch": "More than one default clause in switch statement",
    "NoCatchOrFinally": "Missing catch or finally after try",
    "UnknownLabel": "Undefined label '%s'",
    "Redeclaration": "%s '%s' has already been declared",
    "IllegalContinue": "Illegal continue statement",
    "IllegalBreak": "Illegal break statement",
    "IllegalReturn": "Illegal return statement",
    "StrictModeWith": "Strict mode code may not include a with statement",
    "StrictCatchVariable": "Catch variable may not be eval or arguments in strict mode",
    "StrictVarName": "Variable name may not be eval or arguments in strict mode",
    "StrictParamName": "Parameter name eval or arguments is not allowed in strict mode",
    "StrictParamDupe": "Strict mode function may not have duplicate parameter names",
    "StrictFunctionName": "Function name may not be eval or arguments in strict mode",
    "StrictOctalLiteral": "Octal literals are not allowed in strict mode.",
    "StrictDelete": "Delete of an unqualified identifier in strict mode.",
    "StrictLHSAssignment": "Assignment to eval or arguments is not allowed in strict mode",
    "StrictLHSPostfix": "Postfix increment/decrement may not have eval or arguments operand in strict mode",
    "StrictLHSPrefix": "Prefix increment/decrement may not have eval or arguments operand in strict mode",
    "StrictReservedWord": "Use of future reserved word in strict mode",
    "TemplateOctalLiteral": "Octal literals are not allowed in template strings.",
    "ParameterAfterRestParameter": "Rest parameter must be last formal parameter",
    "DefaultRestParameter": "Unexpected token =",
    "ObjectPatternAsRestParameter": "Unexpected token {",
    "DuplicateProtoProperty": "Duplicate __proto__ fields are not allowed in object literals",
    "ConstructorSpecialMethod": "Class constructor may not be an accessor",
    "DuplicateConstructor": "A class may only have one constructor",
    "StaticPrototype": "Classes may not have static property named prototype",
    "MissingFromClause": "Unexpected token",
    "NoAsAfterImportNamespace": "Unexpected token",
    "InvalidModuleSpecifier": "Unexpected token",
    "IllegalImportDeclaration": "Unexpected token",
    "IllegalExportDeclaration": "Unexpected token",
}

PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "|": 3,
    "^": 4,
    "&": 5,
    "==": 6,
    "!=": 6,
    "===": 6,
    "!==": 6,
    "<": 7,
    ">": 7,
    "<=": 7,
    ">=": 7,
    "instanceof": 7,
    "dar": 7,
    "<<": 8,
    ">>": 8,
    ">>>": 8,
    "+": 9,
    "-": 9,
    "*": 11,
    "/": 11,
    "%": 11,
}


class Token:
    pass


class Syntax:
    pass


class Messages:
    pass


class PlaceHolders:
    ArrowParameterPlaceHolder = "ArrowParameterPlaceHolder"


for k, v in token.items():
    setattr(Token, k, v)

for e in syntax:
    setattr(Syntax, e, e)

for k, v in messages.items():
    setattr(Messages, k, v)

BOM = "\ufeff"
ZWJ = "\u200d"
ZWNJ = "\u200c"
TAB = "\u0009"
VT = "\u000b"
FF = "\u000c"
SP = "\u0020"
NBSP = "\u00a0"
LF = "\u000a"
CR = "\u000d"
LS = "\u2028"
PS = "\u2029"

U_CATEGORIES = defaultdict(list)
for c in map(unichr, range(sys.maxunicode + 1)):
    U_CATEGORIES[unicodedata.category(c)].append(c)
UNICODE_LETTER = set(
    U_CATEGORIES["Lu"]
    + U_CATEGORIES["Ll"]
    + U_CATEGORIES["Lt"]
    + U_CATEGORIES["Lm"]
    + U_CATEGORIES["Lo"]
    + U_CATEGORIES["Nl"]
)
UNICODE_COMBINING_MARK = set(U_CATEGORIES["Mn"] + U_CATEGORIES["Mc"])
UNICODE_DIGIT = set(U_CATEGORIES["Nd"])
UNICODE_CONNECTOR_PUNCTUATION = set(U_CATEGORIES["Pc"])
IDENTIFIER_START = UNICODE_LETTER.union({"$", "_", "\\"})
IDENTIFIER_PART = (
    IDENTIFIER_START
    .union(UNICODE_COMBINING_MARK)
    .union(UNICODE_DIGIT)
    .union(UNICODE_CONNECTOR_PUNCTUATION)
    .union({ZWJ, ZWNJ})
)

WHITE_SPACE = {
    0x20,
    0x09,
    0x0B,
    0x0C,
    0xA0,
    0x1680,
    0x180E,
    0x2000,
    0x2001,
    0x2002,
    0x2003,
    0x2004,
    0x2005,
    0x2006,
    0x2007,
    0x2008,
    0x2009,
    0x200A,
    0x202F,
    0x205F,
    0x3000,
    0xFEFF,
}

LINE_TERMINATORS = {0x0A, 0x0D, 0x2028, 0x2029}


def is_identifier_start(ch):
    return (ch if isinstance(ch, unicode) else unichr(ch)) in IDENTIFIER_START


def is_identifier_part(ch):
    return (ch if isinstance(ch, unicode) else unichr(ch)) in IDENTIFIER_PART


def is_white_space(ch):
    return (ord(ch) if isinstance(ch, unicode) else ch) in WHITE_SPACE


def is_line_terminator(ch):
    return (ord(ch) if isinstance(ch, unicode) else ch) in LINE_TERMINATORS


OCTAL = {"0", "1", "2", "3", "4", "5", "6", "7"}
DEC = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
HEX = set("0123456789abcdefABCDEF")
HEX_CONV = dict(("0123456789abcdef"[n], n) for n in xrange(16))
for i, e in enumerate("ABCDEF", 10):
    HEX_CONV[e] = i


def is_decimal_digit(ch):
    return (ch if isinstance(ch, unicode) else unichr(ch)) in DEC


def is_hex_digit(ch):
    return (ch if isinstance(ch, unicode) else unichr(ch)) in HEX


def is_octal_digit(ch):
    return (ch if isinstance(ch, unicode) else unichr(ch)) in OCTAL


def is_future_reserved_word(w):
    return w in ("enum", "export", "import", "super")


RESERVED_WORD = {"implements", "interface", "package", "private", "protected", "public", "static", "yield", "let"}


def is_strict_mode_reserved_word(w):
    return w in RESERVED_WORD


def is_restricted_word(w):
    return w in ("eval", "arguments")


KEYWORDS = {
    "agar",
    "dar",
    "anjambede",
    "motaghayer",
    "baraye",
    "jadid",
    "bekoosh",
    "let",
    "this",
    "varna",
    "mored",
    "void",
    "with",
    "enum",
    "tavaghti",
    "bebor",
    "begir",
    "partkon",
    "const",
    "yield",
    "class",
    "super",
    "bazgardan",
    "typeof",
    "delete",
    "bargozin",
    "export",
    "import",
    "pishfarz",
    "darnahayat",
    "extends",
    "tabe",
    "edame",
    "eshkalzodaee",
    "instanceof",
    "pyimport",
}


def is_keyword(w):
    return w in KEYWORDS


class JsSyntaxError(Exception):
    pass


if __name__ == "__main__":
    assert is_line_terminator("\n")
    assert is_line_terminator(0x0A)
    assert is_identifier_start("$")
    assert is_identifier_start(100)
    assert is_white_space(" ")
