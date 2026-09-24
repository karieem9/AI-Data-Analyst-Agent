import ast
import builtins
import threading

import pandas as pd
import plotly.express as px

from forecasting import forecast

FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "os", "sys", "subprocess", "shutil", "socket", "pathlib",
}

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "len", "list", "max", "min", "range", "round", "set", "sorted",
        "str", "sum", "tuple", "zip",
    )
}


class UnsafeCodeError(Exception):
    """Raised when generated code fails the safety check, before it ever runs."""


def _references_df(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Name) and n.id == "df" for n in ast.walk(node))


def is_code_safe(code: str) -> tuple[bool, str]:
    """Reject anything but a read-only computation over `df` using `pd`/`px`."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "Imports are not allowed."

        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return False, f"Use of '{node.id}' is not allowed."

        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, "Dunder attribute access is not allowed."

        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_references_df(t) for t in targets):
                return False, "Modifying 'df' is not allowed; df is read-only."

        if isinstance(node, ast.Delete):
            if any(_references_df(t) for t in node.targets):
                return False, "Deleting 'df' is not allowed."

        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    return False, "inplace=True is not allowed; df is read-only."

    return True, ""


def run_safely(code: str, df: pd.DataFrame, timeout: int = 5):
    """
    Validate then execute generated code in a restricted scope.

    Raises UnsafeCodeError if the code fails validation, TimeoutError if it
    runs too long, or whatever exception the code itself raised.

    Caveat: a genuinely stuck/looping thread can't be killed from here (no
    signal.alarm on Windows) -- it keeps running in the background even after
    this raises TimeoutError. It's started as a daemon thread specifically so
    it can't block the app itself from exiting; the leak is still real,
    just contained. That's an accepted limit of this first layer, not a
    guarantee -- see the build guide's risk notes.
    """
    safe, reason = is_code_safe(code)
    if not safe:
        raise UnsafeCodeError(reason)

    scope = {"pd": pd, "df": df.copy(), "px": px, "forecast": forecast}
    outcome = {}

    def _target():
        try:
            exec(code, {"__builtins__": SAFE_BUILTINS}, scope)
            outcome["result"] = scope.get("result")
        except BaseException as e:
            outcome["error"] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(f"Execution took longer than {timeout}s.")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")
