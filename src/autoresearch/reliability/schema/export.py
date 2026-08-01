"""Manifest-bound LaTeX export with provenance.

Writes paper-facing LaTeX tables and scalar macros that reference the frozen
run manifest, analysis code hash, and endpoint provenance. No number is typed
manually; every exported value carries its provenance chain.

Schema version 1.0 · 2026-08-01
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .manifest import RunManifest
from ..warning.endpoints import EndpointResult


class ExportError(RuntimeError):
    """Raised when a manifest-bound export cannot be written."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def export_scalar_macro(
    endpoint: EndpointResult,
    output_path: str | Path,
    manifest: RunManifest | None = None,
) -> dict[str, Any]:
    """Write a single LaTeX scalar macro bound to endpoint provenance.

    Returns a provenance record for the exported artifact.
    """
    macro_name = "\\endpointvalue"
    value_text = f"{endpoint.value:.6f}"
    manifest_hash = endpoint.manifest_hash
    if manifest is not None:
        manifest_hash = manifest.manifest_hash

    provenance_comment = (
        "% Generated file — do not edit by hand.\n"
        "% Provenance:\n"
        f"%   endpoint_id: {endpoint.endpoint_id}\n"
        f"%   manifest_hash: {manifest_hash}\n"
        f"%   analysis_code_hash: {endpoint.analysis_code_hash}\n"
        f"%   n_observations: {endpoint.n_observations}\n"
    )
    content = (
        f"{provenance_comment}"
        f"\\newcommand{{{macro_name}}}{{{value_text}}}\n"
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    provenance = {
        "artifact": str(path),
        "type": "scalar_macro",
        "macro_name": macro_name,
        "value": value_text,
        "manifest_hash": manifest_hash,
        "analysis_code_hash": endpoint.analysis_code_hash,
        "export_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    return provenance


def export_latex_table(
    rows: list[dict[str, Any]],
    columns: list[str],
    output_path: str | Path,
    *,
    caption: str,
    label: str,
    manifest: RunManifest | None = None,
    endpoint_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write a LaTeX table from structured rows with provenance headers.

    Each row maps column names to cell values. Every numeric cell may carry
    an ``_artifact`` sibling key identifying the generating analysis.
    """
    if not rows:
        raise ExportError("cannot export an empty table")

    endpoint_hashes = endpoint_hashes or {}
    manifest_hash = manifest.manifest_hash if manifest is not None else "unknown"
    all_hashes = dict(endpoint_hashes)
    all_hashes.setdefault("manifest_hash", manifest_hash)

    header = " & ".join(_tex_escape(c) for c in columns) + r" \\"
    body_lines: list[str] = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            artifact = row.get(f"{col}_artifact")
            cell = _tex_escape(str(value)) if value not in ("", None) else "\\textit{[pending]}"
            if artifact:
                cell += f" \\tiny{{\\texttt{{{artifact}}}}}"
            cells.append(cell)
        body_lines.append(" & ".join(cells) + r" \\")

    provenance_comment = (
        "% Generated file — do not edit by hand.\n"
        "% Provenance:\n"
        + "".join(f"%   {k}: {v}\n" for k, v in all_hashes.items())
    )

    body_text = "\n".join(body_lines) + "\n"

    parts = [
        f"{provenance_comment}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_tex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{" + ("c" * len(columns)) + "}",
        r"\toprule",
        header,
        r"\midrule",
        body_text.rstrip("\n"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    content = "\n".join(parts) + "\n"

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    provenance = {
        "artifact": str(path),
        "type": "latex_table",
        "caption": caption,
        "rows": len(rows),
        "columns": len(columns),
        "manifest_hash": manifest_hash,
        "export_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    return provenance


def _tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )
