"""Repo-wide audit of the railway_extension sync/async boundary.

Every function in railway_extension.py is synchronous and does a blocking
requests.post() to Railway's GraphQL API. Awaiting one raises
"TypeError: object dict can't be used in 'await' expression" at runtime -- a
bug that has now shipped three times (railway_list, and two branches of
approval_routes._execute_approved_action). Importing a name that does not
exist there fails the same way, only louder.

These checks are static (ast only, no imports), so they hold no matter what
other test modules have left in sys.modules.
"""
import ast
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tree(relpath):
    with open(os.path.join(REPO_ROOT, relpath)) as f:
        return ast.parse(f.read(), filename=relpath)


def _tree_or_none(relpath):
    """Some checked-in helper scripts embed unbalanced source in string
    literals and do not parse. They are not importable modules; skip them."""
    try:
        return _tree(relpath)
    except SyntaxError:
        return None


def _python_files():
    skip = {".git", "__pycache__", "scratch", "node_modules"}
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for name in filenames:
            if name.endswith(".py"):
                full = os.path.join(dirpath, name)
                yield os.path.relpath(full, REPO_ROOT)


def railway_extension_names():
    """Top-level names railway_extension.py actually exports."""
    names = set()
    for node in _tree("railway_extension.py").body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def sync_function_names():
    return {
        n.name
        for n in _tree("railway_extension.py").body
        if isinstance(n, ast.FunctionDef)
    }


def test_railway_extension_functions_are_all_sync():
    """If one ever becomes async, these guards need revisiting."""
    tree = _tree("railway_extension.py")
    async_defs = [n.name for n in tree.body if isinstance(n, ast.AsyncFunctionDef)]
    assert not async_defs, f"railway_extension gained async functions: {async_defs}"


def test_no_railway_extension_function_is_awaited_directly():
    sync_names = sync_function_names()
    offenders = []
    for relpath in _python_files():
        if relpath == os.path.join("tests", "test_railway_sync_boundary.py"):
            continue
        tree = _tree_or_none(relpath)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Await):
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            fn = call.func
            name = fn.id if isinstance(fn, ast.Name) else None
            if name in sync_names:
                offenders.append(f"{relpath}:{node.lineno} await {name}(...)")
    assert not offenders, (
        "awaiting a sync railway_extension function raises "
        "\"object dict can't be used in 'await' expression\"; "
        "use `await asyncio.to_thread(fn, ...)` instead:\n  " + "\n  ".join(offenders)
    )


def test_every_railway_extension_import_resolves():
    exported = railway_extension_names()
    offenders = []
    for relpath in _python_files():
        if relpath.startswith("tests" + os.sep):
            continue  # test modules legitimately stub the module
        tree = _tree_or_none(relpath)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "railway_extension":
                for alias in node.names:
                    if alias.name != "*" and alias.name not in exported:
                        offenders.append(f"{relpath}:{node.lineno} imports missing name {alias.name!r}")
    assert not offenders, "railway_extension does not define:\n  " + "\n  ".join(offenders)


if __name__ == "__main__":
    test_railway_extension_functions_are_all_sync()
    test_no_railway_extension_function_is_awaited_directly()
    test_every_railway_extension_import_resolves()
    print("All railway sync-boundary audits pass.")
