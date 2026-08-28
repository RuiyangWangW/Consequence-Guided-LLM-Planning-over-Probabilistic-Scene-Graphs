"""Report names a function loads that nothing it can see defines.

Catches the NameError class of bug - a helper using `math` or `np` that its enclosing
function imported but it did not - without launching the simulator.

    python undef.py primitive_patches.py test_primitives.py

Scoping follows Python's: a nested function sees its own bindings, then those of each
enclosing function, then module level, then builtins. Names bound anywhere in a scope
count as visible throughout it, which is what `import` inside a function does.
"""
import ast
import builtins
import sys

BUILTINS = set(dir(builtins))
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def bindings(node):
    """Every name bound directly in `node`'s own scope."""
    names = set()
    if isinstance(node, SCOPES):
        a = node.args
        names.update(x.arg for x in a.args + a.posonlyargs + a.kwonlyargs)
        if a.vararg:
            names.add(a.vararg.arg)
        if a.kwarg:
            names.add(a.kwarg.arg)

    def visit(n, top=False):
        # Do not descend into a nested scope: its bindings are its own.
        if not top and isinstance(n, SCOPES):
            names.add(getattr(n, "name", ""))
            return
        if not top and isinstance(n, ast.ClassDef):
            names.add(n.name)
            return
        for child in ast.iter_child_nodes(n):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                names.add(child.id)
            elif isinstance(child, ast.Import):
                names.update(x.asname or x.name.split(".")[0] for x in child.names)
            elif isinstance(child, ast.ImportFrom):
                names.update(x.asname or x.name for x in child.names)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                names.add(child.name)
            visit(child)

    visit(node, top=True)
    names.discard("")
    return names


def loads(node):
    """Names read in `node`'s own scope, excluding nested scopes."""
    used = set()

    def visit(n, top=False):
        if not top and isinstance(n, SCOPES):
            return                          # its reads are checked in its own frame
        for child in ast.iter_child_nodes(n):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                used.add(child.id)
            visit(child)

    visit(node, top=True)
    return used


def check(path):
    tree = ast.parse(open(path).read())
    bad = []

    def walk(node, visible):
        here = visible | bindings(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            missing = sorted(loads(node) - here - BUILTINS)
            if missing:
                bad.append((node.lineno, node.name, missing))
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(child, here)
            else:
                for g in ast.walk(child):
                    if isinstance(g, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                        walk(g, here)

    walk(tree, BUILTINS | bindings(tree))
    return bad


def main(paths):
    failed = False
    for path in paths:
        for line, name, missing in sorted(check(path)):
            print(f"{path}:{line}  {name}()  UNDEFINED: {missing}")
            failed = True
    print("STATIC CHECK", "FAILED" if failed else "PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
