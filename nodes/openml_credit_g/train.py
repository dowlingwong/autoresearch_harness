#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark import ConfigError, emit_failure, emit_success, run_openml_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    try:
        metrics = run_openml_benchmark(Path(args.config))
    except ConfigError as exc:
        emit_failure("invalid_config", str(exc), Path(args.config))
        return 2
    except Exception as exc:  # noqa: BLE001
        emit_failure("runtime_error", f"{type(exc).__name__}: {exc}", Path(args.config))
        return 1
    emit_success(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
