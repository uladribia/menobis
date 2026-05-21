"""Compatibility entrypoint for the canonical full E2E benchmark."""

from __future__ import annotations

import sys

from benchmarks.bench_e2e import bench_e2e


def main() -> None:
    """Run the canonical E2E benchmark for CLI-provided node sizes."""
    nodes = [int(value) for value in sys.argv[1:]] if len(sys.argv) > 1 else [100, 500]
    bench_e2e(max_n=max(nodes), nodes=nodes)


if __name__ == "__main__":
    main()
