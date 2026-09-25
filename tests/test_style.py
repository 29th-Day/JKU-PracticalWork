import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "project"
# The package root is a user-owned placeholder without a public API.
MODULES = sorted(set(SOURCE_ROOT.rglob("*.py")) - {SOURCE_ROOT / "__init__.py"})


def _is_all(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Assign):
        return False
    return any(
        isinstance(target, ast.Name) and target.id == "__all__"
        for target in node.targets
    )


def _is_import(node: ast.stmt) -> bool:
    if isinstance(node, ast.ImportFrom):
        return node.module != "__future__"
    return isinstance(node, ast.Import)


def test_every_module_declares_all_before_imports() -> None:
    misplaced: list[str] = []
    for path in MODULES:
        body = ast.parse(path.read_text()).body
        all_index = next((i for i, node in enumerate(body) if _is_all(node)), None)
        imports = [i for i, node in enumerate(body) if _is_import(node)]
        if all_index is None:
            misplaced.append(f"{path.name}: no __all__")
        elif imports and all_index > imports[0]:
            misplaced.append(f"{path.name}: __all__ after imports")

    assert MODULES
    assert not misplaced
