# Schema directory — versioned JSON schema exports
# Version 1.0 · 2026-08-01
#
# The authoritative schema implementation lives in
# src/autoresearch/reliability/schema/. This directory holds JSON Schema
# exports for external validators and the paper artifact.
#
# Planned exports:
#   - event_envelope.v1.json       (ReliabilityEvent envelope)
#   - run_manifest.v1.json         (RunManifest)
#   - fault_manifest.v1.json       (FaultManifest)
#
# Export is generated from the dataclass definitions; see
# src/autoresearch/reliability/schema/manifest.py for the canonical fields.
