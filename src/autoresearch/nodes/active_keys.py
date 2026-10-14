"""D25: node-declared active keys.

A node may declare which editable keys are active for its current
configuration, keyed by ``model_type`` with dataset-level exclusions::

    "active_keys": {
      "by_model_type": {
        "logistic_regression": ["model_type", "C", "class_weight", "max_iter", "scaler"],
        ...
      },
      "dataset_excluded": ["imputer"]
    }

Nodes without a declaration resolve to ``None`` and behave exactly as before.
The order of a ``by_model_type`` list is significant: callers that map it onto
an ordered action space (the scripted proposer) rely on it.
"""
from __future__ import annotations

import re

from autoresearch.nodes.spec import NodeSpec

_CHANGE_RE = re.compile(
    r"change\s+([A-Za-z_][A-Za-z0-9_]*)\s+from\s+(\S+)\s+to\s+(\S+)",
    re.IGNORECASE,
)


def resolve_active_keys(
    node_spec: NodeSpec,
    constants: dict[str, str] | None = None,
) -> tuple[str, ...] | None:
    """Active editable keys for the current configuration, or None if undeclared.

    The declaration is keyed by the current ``model_type`` (read from
    *constants* when given).  When the current ``model_type`` is not covered by
    the declaration the function returns ``None`` so that callers fall back to
    the pre-D25 behaviour rather than guessing an active set.
    """
    declaration = getattr(node_spec, "active_keys", None)
    if not declaration:
        return None
    by_model_type = declaration.get("by_model_type")
    if not isinstance(by_model_type, dict):
        return None
    model_type = str((constants or {}).get("model_type", "")).strip()
    keys = by_model_type.get(model_type)
    if keys is None:
        return None
    excluded = set(declaration.get("dataset_excluded") or ())
    return tuple(str(key) for key in keys if str(key) not in excluded)


def named_edit_key(proposal: object) -> str | None:
    """The editable key a proposal names, from its structured edit or objective.

    The manager may name a key in its structured ``param`` field (recorded as
    ``extra["named_param"]`` when the node declares active keys) or only in the
    objective text (``Change KEY from OLD to NEW``).  Returns ``None`` when no
    key is named.
    """
    extra = getattr(proposal, "extra", None) or {}
    edit = extra.get("structured_edit")
    if isinstance(edit, dict) and edit.get("symbol"):
        return str(edit["symbol"])
    named = extra.get("named_param")
    if named:
        return str(named)
    objective = str(getattr(proposal, "objective", "") or "")
    match = _CHANGE_RE.search(objective)
    if match:
        return match.group(1)
    return None


def inactive_key_in_proposal(
    proposal: object,
    node_spec: NodeSpec,
    constants: dict[str, str] | None = None,
) -> str | None:
    """Return the named edit key when it is declared editable but inactive.

    Returns ``None`` for undeclared nodes, for proposals that name no key, or
    for keys outside the node's editable symbols (those are handled by the
    scope and precondition machinery, not by D25).
    """
    active = resolve_active_keys(node_spec, constants)
    if active is None:
        return None
    key = named_edit_key(proposal)
    if key is None:
        return None
    editable = set(getattr(node_spec, "editable_symbols", ()) or ())
    if editable and key not in editable:
        return None
    return None if key in set(active) else key
