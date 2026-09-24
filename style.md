# Python Style Guide

PEP 8, condensed. Project-specific rules take precedence. Consistency priority: within a function, then within a module, then across the project, then this guide. Break a rule only when following it hurts readability or clashes with the surrounding code.

## Typing & Tooling

- All code must be fully typed and pass `basedpyright` and `ruff` with no errors or warnings.
- Annotate every parameter and return type. Annotate variables when the type can't be inferred.
- Use modern syntax: `list[int]`, `X | None`.

## Layout

- Indent with 4 spaces. No tabs.
- Wrap lines using implicit continuation inside `()`, `[]`, `{}`. Don't use backslashes.
- Continuation lines either align with the opening delimiter or use a hanging indent.
    - A hanging indent leaves nothing after the opening bracket on the first line and adds one indent level.
    - In a `def`, indent the arguments one extra level so they stand apart from the body.
- A closing bracket goes under the first character of either the last line or the construct's first line.
- Break long lines _before_ binary operato - c)`.
- Blank lines: 2 around top-level functions/classes, 1 between methods, and only a few inside functions.
- One statement per line. No `;` and no `if x: y` on one line.
- Files are UTF-8 with no encoding declaration. Identifiers are ASCII only.

## Imports

- Imports go at the top of the file, in this order:
    1. The module docstring
    2. `from __future__` imports
    3. Dunders such as `__all__`
    4. All other imports
- Group imports as standard library, then third-party, then local, with a blank line between groups.
- Import one module per line. `from x import a, b` is fine.
- Prefer absolute imports. No wildcard imports.

## Whitespace

- No spaces directly inside brackets, befo `(`/`[` of a call or index.
- Don't add extra spaces to line up `=` across lines.
- Put one space around `=`, augmented assignment, comparisons, `in`, `is`, `and`, `or`, `not`.
- With mixed operator priorities, you may drop spaces around the higher-priority operators: `x = x*2 - 1`, `c =
(a+b) * (a-b)`. Always use the same spacinr.
- Treat the slice colon like the lowest-priority operator.
    - Write `ham[1:9]` and `ham[lo + off : hi + off]`.
    - Leave out the space when a slice part is omitted: `ham[lo:]`.
- Annotations: `x: int` and `def f(a: int) -> str:`.
- Default and keyword `=`:
    - No spaces when there is no annotation: `f(x=1)`, `def f(x=0)`.
    - Spaces when there is an annotation: `def f(x: int = 0)`.
- No trailing whitespace.

## Strings & Trailing Commas

- Use double quotes. Switch quote style only to avoid escaping.
- Triple-quoted strings always use `"""`.
- A one-element tuple needs a trailing comma. Write it in parentheses: `("a",)`.
- Multi-line collections put one item per line, each followed by a comma, with the closing bracket on its own line.
- Never write `[a, b,]` on a single line.

## Comments & Docstrings

- Comments must be a single line. Only wrihat isn't obvious from the code.
- Docstrings (`"""`) may span multiple lines.

## Naming

- Modules: short, lowercase, underscores allowed. Packages: lowercase, avoid underscores.
- Classes, exceptions and type variables use CapWords.
    - Keep acronyms uppercase: `HTTPError`.
    - Exception names end in `Error`.
    - Type variables are short (`T`), with aen covariant or contravariant.
- Functions, methods, variables and argumelevel constants use `UPPER_SNAKE`.
- The first argument is `self` for instance methods and `cls` for class methods.
- If a name clashes with a keyword, add a trailing underscore: `class_`.
- Mark non-public names with a single leading `_`. Use `__name` only to avoid clashes in subclasses.
- Never invent new `__dunder__` names.
- Never use `l`, `O` or `I` as single-character names.
- Declare a module's public API with `__all__`.

## Programming

- Compare to `None` with `is` or `is not`.
    - Write `x is not None` rather than `if
    - Write `is not` rather than `not ... is`.
- Don't compare booleans with `==` or `is`.
- Use the falsiness of empty sequences: `if not seq`.
- Use `isinstance()` rather than comparing `type()`.
- Use `str.startswith()`/`endswith()` rather than slicing.
- Use `def` rather than assigning a lambda to a name.
- Build strings with `"".join()`, not `+=`
- For ordering comparisons, implement all six methods or use `functools.total_ordering`.
- Exceptions:
    - Derive from `Exception` and catch specific exception types. No bare `except:`.
    - Keep the `try` body minimal and put follow-up code in `else`.
    - Chain with `raise X from Y`.
- Use `with` for resources. A context manager that does more than acquire/release should be reached through a named method: `with conn.begin_transaction():`.
- Returns must be consistent: if any `retuurn` does, including an explicit `returnNone` at the end.
- Never `return`, `break` or `continue` inside `finally`.
