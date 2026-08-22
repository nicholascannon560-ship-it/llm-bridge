"""Tests for get_service_domains and the empty-project-listing diagnostic.

Discovery by listing is not dependable: a project-scoped RAILWAY_API_TOKEN can
read and redeploy its own resources by UUID while `projects` returns zero
edges. get_service_domains works by service ID, so it keeps working in that
case and gives callers a base URL without going through any listing.
"""
import ast
import contextlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@contextlib.contextmanager
def _fresh_railway_extension():
    """Import the real railway_extension, restoring sys.modules afterwards.

    Other test modules install a fake `railway_extension`; this leaves their
    entry exactly as it was.
    """
    saved = sys.modules.get("railway_extension", ...)
    sys.modules.pop("railway_extension", None)
    try:
        import railway_extension
        yield railway_extension
    finally:
        sys.modules.pop("railway_extension", None)
        if saved is not ...:
            sys.modules["railway_extension"] = saved


@contextlib.contextmanager
def _stub_query(module, result):
    """Replace railway_query so nothing touches the network."""
    calls = []

    def fake(query, variables=None):
        calls.append((query, variables))
        return result

    original = module.railway_query
    module.railway_query = fake
    try:
        yield calls
    finally:
        module.railway_query = original


def _instances(*domain_lists):
    return {
        "service": {
            "serviceInstances": {
                "edges": [
                    {"node": {"id": f"si-{i}",
                              "domains": {"serviceDomains": [{"domain": d} for d in domains]}}}
                    for i, domains in enumerate(domain_lists)
                ]
            }
        }
    }


def test_domains_are_flattened_and_returned_as_urls():
    with _fresh_railway_extension() as rx:
        with _stub_query(rx, _instances(["kalshiml-production-b2e9.up.railway.app"])) as calls:
            out = rx.get_service_domains("svc-1")

    assert out == {
        "service_id": "svc-1",
        "domains": ["kalshiml-production-b2e9.up.railway.app"],
        "urls": ["https://kalshiml-production-b2e9.up.railway.app"],
    }, out
    assert calls[-1][1] == {"id": "svc-1"}, calls[-1][1]


def test_domains_across_instances_are_merged_and_deduped():
    with _fresh_railway_extension() as rx:
        with _stub_query(rx, _instances(["a.up.railway.app"], ["b.up.railway.app", "a.up.railway.app"])):
            out = rx.get_service_domains("svc-1")

    assert out["domains"] == ["a.up.railway.app", "b.up.railway.app"], out["domains"]


def test_service_with_no_domains_returns_empty_lists_not_an_error():
    for payload in (_instances([]), _instances(), {"service": None}, {}):
        with _fresh_railway_extension() as rx:
            with _stub_query(rx, payload):
                out = rx.get_service_domains("svc-1")
        assert out["domains"] == [], (payload, out)
        assert out["urls"] == [], (payload, out)


def test_domains_query_only_uses_fields_already_proven_against_the_live_schema():
    """One unverified field fails the whole document with GRAPHQL_VALIDATION_FAILED.

    list_env_vars' selection is known to work against the live API; the domains
    query must not reach past it (notably: no `customDomains`).
    """
    with open(os.path.join(REPO_ROOT, "railway_extension.py")) as f:
        source = f.read()

    tree = ast.parse(source)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "get_service_domains" in fns

    def selection(fn):
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str) and "serviceInstances" in node.value.value:
                    return set(node.value.value.replace("{", " ").replace("}", " ").split())
        raise AssertionError(f"no serviceInstances query found in {fn.name}")

    proven = selection(fns["list_env_vars"])
    used = selection(fns["get_service_domains"])
    unproven = used - proven
    assert not unproven, f"query uses fields not proven in list_env_vars: {sorted(unproven)}"


def test_empty_project_listing_carries_a_diagnostic_hint():
    with _fresh_railway_extension() as rx:
        with _stub_query(rx, {"projects": {"edges": []}}):
            out = rx.list_projects()

    assert out["projects"] == {"edges": []}, out
    assert "_hint" in out, "an empty listing must explain itself"
    assert "UUID" in out["_hint"]


def test_non_empty_project_listing_is_returned_untouched():
    payload = {"projects": {"edges": [{"node": {"id": "p1", "name": "llm-bridge"}}]}}
    with _fresh_railway_extension() as rx:
        with _stub_query(rx, payload):
            out = rx.list_projects()

    assert out == payload, out
    assert "_hint" not in out


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
    print("All railway domain tests pass.")
