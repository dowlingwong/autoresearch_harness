from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml
from numpy.lib.stride_tricks import sliding_window_view


ALLOWED_IMPLEMENTATIONS = {"loop", "im2col", "einsum"}
ALLOWED_RANGES = {
    "batch_size": (2, 32),
    "image_size": (16, 64),
    "num_filters": (4, 32),
    "kernel_size": (2, 5),
    "stride": (1, 4),
    "padding": (0, 4),
    "repeat_count": (1, 7),
}
SEED = 42


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("config root must be a mapping")
    return payload


def _validate_config(config: dict) -> None:
    impl = str(config.get("implementation", "")).strip()
    if impl not in ALLOWED_IMPLEMENTATIONS:
        raise ValueError(f"implementation must be one of {sorted(ALLOWED_IMPLEMENTATIONS)}")
    for key, (low, high) in ALLOWED_RANGES.items():
        if key not in config:
            raise ValueError(f"missing config key: {key}")
        value = int(config[key])
        if value < low or value > high:
            raise ValueError(f"{key}={value} outside allowed range [{low}, {high}]")


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _add_padding(x: np.ndarray, padding: int) -> np.ndarray:
    if padding <= 0:
        return x
    return np.pad(
        x,
        pad_width=((0, 0), (padding, padding), (padding, padding), (0, 0)),
        mode="constant",
    )


def _conv_loop(
    features: np.ndarray,
    kernels: np.ndarray,
    biases: np.ndarray,
    *,
    stride: int,
    padding: int,
) -> np.ndarray:
    padded = _add_padding(features, padding)
    batch_size, height, width, _ = padded.shape
    kernel_size = kernels.shape[0]
    num_filters = kernels.shape[-1]
    out_h = ((height - kernel_size) // stride) + 1
    out_w = ((width - kernel_size) // stride) + 1
    output = np.zeros((batch_size, out_h, out_w, num_filters), dtype=np.float64)
    for index in range(batch_size):
        image = padded[index]
        for y in range(out_h):
            y0 = y * stride
            y1 = y0 + kernel_size
            for x in range(out_w):
                x0 = x * stride
                x1 = x0 + kernel_size
                patch = image[y0:y1, x0:x1, :]
                for filter_index in range(num_filters):
                    output[index, y, x, filter_index] = (
                        np.sum(patch * kernels[:, :, :, filter_index]) + biases[filter_index]
                    )
    return _relu(output)


def _conv_im2col(
    features: np.ndarray,
    kernels: np.ndarray,
    biases: np.ndarray,
    *,
    stride: int,
    padding: int,
) -> np.ndarray:
    padded = _add_padding(features, padding)
    windows = sliding_window_view(
        padded,
        window_shape=(kernels.shape[0], kernels.shape[1], kernels.shape[2]),
        axis=(1, 2, 3),
    )
    windows = windows[:, ::stride, ::stride, 0, :, :, :]
    flat_windows = windows.reshape(-1, kernels.shape[0] * kernels.shape[1] * kernels.shape[2])
    flat_kernels = kernels.reshape(-1, kernels.shape[-1])
    out = flat_windows @ flat_kernels + biases.reshape(1, -1)
    out_h, out_w = windows.shape[1], windows.shape[2]
    return _relu(out.reshape(features.shape[0], out_h, out_w, kernels.shape[-1]))


def _conv_einsum(
    features: np.ndarray,
    kernels: np.ndarray,
    biases: np.ndarray,
    *,
    stride: int,
    padding: int,
) -> np.ndarray:
    padded = _add_padding(features, padding)
    windows = sliding_window_view(
        padded,
        window_shape=(kernels.shape[0], kernels.shape[1], kernels.shape[2]),
        axis=(1, 2, 3),
    )
    windows = windows[:, ::stride, ::stride, 0, :, :, :]
    out = np.einsum("bhwijk,ijkf->bhwf", windows, kernels) + biases.reshape(1, 1, 1, -1)
    return _relu(out)


def _make_problem(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    batch_size = int(config["batch_size"])
    image_size = int(config["image_size"])
    num_filters = int(config["num_filters"])
    kernel_size = int(config["kernel_size"])
    features = rng.normal(size=(batch_size, image_size, image_size, 3))
    kernels = rng.normal(size=(kernel_size, kernel_size, 3, num_filters))
    biases = rng.normal(size=(num_filters,))
    return features, kernels, biases


def _run_impl(config: dict, features: np.ndarray, kernels: np.ndarray, biases: np.ndarray) -> np.ndarray:
    kwargs = {"stride": int(config["stride"]), "padding": int(config["padding"])}
    implementation = str(config["implementation"])
    if implementation == "loop":
        return _conv_loop(features, kernels, biases, **kwargs)
    if implementation == "im2col":
        return _conv_im2col(features, kernels, biases, **kwargs)
    if implementation == "einsum":
        return _conv_einsum(features, kernels, biases, **kwargs)
    raise AssertionError(f"unreachable implementation: {implementation}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    try:
        config = _load_config(config_path)
        _validate_config(config)
        features, kernels, biases = _make_problem(config)

        # Correctness check uses a small prefix so validation remains cheap even
        # when the edited config increases the full benchmark size.
        check_features = features[: min(2, features.shape[0]), :16, :16, :]
        expected = _conv_loop(
            check_features,
            kernels,
            biases,
            stride=int(config["stride"]),
            padding=int(config["padding"]),
        )
        observed = _run_impl(config, check_features, kernels, biases)
        max_abs_error = float(np.max(np.abs(expected - observed)))
        if max_abs_error > 1e-8:
            raise ValueError(f"implementation correctness check failed: {max_abs_error}")

        timings: list[float] = []
        for _ in range(int(config["repeat_count"])):
            started = time.perf_counter()
            _run_impl(config, features, kernels, biases)
            timings.append(time.perf_counter() - started)
        runtime_seconds = float(np.median(timings))
        speed_score = 1.0 / max(runtime_seconds, 1e-12)

        metrics = {
            "node": "mlagentbench_vectorization",
            "source_benchmark": "MLAgentBench/vectorization",
            "seed": SEED,
            "implementation": str(config["implementation"]),
            "runtime_seconds": runtime_seconds,
            "speed_score": speed_score,
            "max_abs_error": max_abs_error,
            "status": "success",
        }
        (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (artifacts_dir / "run.log").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with Path("submission.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow([runtime_seconds])

        print("NODE=mlagentbench_vectorization")
        print(f"IMPLEMENTATION={config['implementation']}")
        print(f"RUNTIME_SECONDS={runtime_seconds:.9f}")
        print(f"SPEED_SCORE={speed_score:.9f}")
        print("STATUS=success")
        return 0
    except Exception as exc:  # noqa: BLE001
        failure = {
            "node": "mlagentbench_vectorization",
            "status": "failed",
            "failure_category": "invalid_config",
            "message": str(exc),
        }
        (artifacts_dir / "metrics.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        (artifacts_dir / "run.log").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print("STATUS=failed")
        print("FAILURE_CATEGORY=invalid_config")
        print(f"ERROR={exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
