#!/usr/bin/env python3
"""Top-level HeCBench CUDA benchmark runner."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only on minimal installs
    yaml = None


BENCHMARKS = ["rsbench", "xsbench", "jacobi", "bfs", "lulesh"]
OUTPUT_DIR = "results/offloading"
NVIDIA_ARCH = "sm_90"

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
DEFAULT_METADATA = (
    ROOT / "benchmarks.yaml",
    ROOT / "benchmark.yaml",
    ROOT / "subset.yaml",
    ROOT / "src" / "scripts" / "benchmarks" / "subset.yaml",
    ROOT / "src" / "scripts" / "benchmarks" / "subset.json",
)
SUMMARY_FIELDS = (
    "benchmark",
    "status",
    "runs_completed",
    "warmups",
    "mean",
    "stddev",
    "run_values",
    "job_id",
    "metadata_source",
    "error",
    "log_dir",
)


@dataclass(frozen=True)
class BenchmarkMetadata:
    regex: str | None
    source: Path | None
    timeout: int = 300
    args: tuple[str, ...] = ()
    binary: str = "main"


@dataclass(frozen=True)
class RunnerConfig:
    arch: str
    warmups: int
    runs: int
    output_dir: Path
    multi_match: str
    make_args: tuple[str, ...]
    build_timeout: int
    run_timeout: int | None
    metadata_paths: tuple[Path, ...]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def normalize_benchmark_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("empty benchmark name")
    if name.endswith("-cuda"):
        return name.removesuffix("-cuda")
    for suffix in ("-hip", "-sycl", "-omp"):
        if name.endswith(suffix):
            raise ValueError(f"{name}: only CUDA benchmarks are supported")
    return name


def parse_benchmark_list(path: Path) -> list[str]:
    benchmarks: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.split("#", 1)[0].strip()
            if not stripped:
                continue
            try:
                benchmark = normalize_benchmark_name(stripped)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if benchmark not in seen:
                benchmarks.append(benchmark)
                seen.add(benchmark)
    if not benchmarks:
        raise ValueError(f"{path}: no benchmarks found")
    return benchmarks


def normalize_benchmark_list(benchmarks: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for benchmark in benchmarks:
        name = normalize_benchmark_name(benchmark)
        if name not in seen:
            normalized.append(name)
            seen.add(name)
    if not normalized:
        raise ValueError("BENCHMARKS list is empty")
    return normalized


def benchmark_dir(name: str, root: Path = ROOT) -> Path:
    return root / "src" / f"{name}-cuda"


def default_metadata_paths(root: Path = ROOT) -> tuple[Path, ...]:
    if root == ROOT:
        return DEFAULT_METADATA
    return (
        root / "benchmarks.yaml",
        root / "benchmark.yaml",
        root / "subset.yaml",
        root / "src" / "scripts" / "benchmarks" / "subset.yaml",
        root / "src" / "scripts" / "benchmarks" / "subset.json",
    )


def metadata_candidates(root: Path, explicit_paths: Sequence[Path]) -> tuple[Path, ...]:
    if explicit_paths:
        return tuple(path if path.is_absolute() else root / path for path in explicit_paths)
    return default_metadata_paths(root)


def load_yaml_metadata(path: Path, benchmark: str) -> tuple[BenchmarkMetadata | None, bool]:
    if yaml is None:
        raise RuntimeError(f"{path}: PyYAML is required to read YAML metadata")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    entry = data.get(benchmark)
    if not isinstance(entry, dict):
        return None, False
    test = entry.get("test")
    if not isinstance(test, dict):
        return BenchmarkMetadata(None, path, 300), True
    run_args = tuple(str(arg) for arg in test.get("args", []) or [])
    binary = str(test.get("binary", "main"))
    regex = test.get("regex")
    if not regex:
        timeout = int(test.get("timeout", 300))
        return BenchmarkMetadata(None, path, timeout, run_args, binary), True
    timeout = int(test.get("timeout", 300))
    return BenchmarkMetadata(str(regex), path, timeout, run_args, binary), True


def load_json_subset_metadata(path: Path, benchmark: str) -> tuple[BenchmarkMetadata | None, bool]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    entry = data.get(benchmark)
    if entry is None:
        return None, False
    if not isinstance(entry, list) or not entry or not entry[0]:
        return BenchmarkMetadata(None, path, 300), True
    run_args = tuple(str(arg) for arg in (entry[1] if len(entry) > 1 else []) or [])
    binary = str(entry[2]) if len(entry) > 2 else "main"
    return BenchmarkMetadata(str(entry[0]), path, 300, run_args, binary), True


def find_metadata(
    benchmark: str,
    root: Path = ROOT,
    explicit_paths: Sequence[Path] = (),
) -> BenchmarkMetadata | None:
    searched: list[Path] = []
    for path in metadata_candidates(root, explicit_paths):
        if not path.exists():
            continue
        searched.append(path)
        if path.suffix in (".yaml", ".yml"):
            metadata, _found = load_yaml_metadata(path, benchmark)
        elif path.suffix == ".json":
            metadata, _found = load_json_subset_metadata(path, benchmark)
        else:
            continue
        if metadata is not None:
            return metadata

    if searched:
        return None
    return None


def metadata_source(metadata: BenchmarkMetadata | None) -> str:
    return str(metadata.source) if metadata and metadata.source else ""


def should_write_summary(metadata: BenchmarkMetadata | None) -> bool:
    return metadata is not None and metadata.regex is not None


def warn_unsummarized_metadata(benchmark: str, metadata: BenchmarkMetadata | None) -> None:
    if metadata is None:
        reason = "no metadata found"
    elif metadata.regex is None:
        reason = f"metadata in {metadata.source} has no regex"
    else:
        return
    print(
        f"warning: {benchmark}: {reason}; running without a summary.csv row",
        file=sys.stderr,
    )


def validate_benchmarks(
    benchmarks: Iterable[str],
    root: Path,
    explicit_metadata: Sequence[Path],
) -> dict[str, BenchmarkMetadata]:
    metadata_by_benchmark: dict[str, BenchmarkMetadata] = {}
    errors: list[str] = []
    for benchmark in benchmarks:
        path = benchmark_dir(benchmark, root)
        if not path.is_dir():
            errors.append(f"{benchmark}: missing CUDA directory {path}")
            continue
        metadata = find_metadata(benchmark, root, explicit_metadata)
        warn_unsummarized_metadata(benchmark, metadata)
        if metadata is not None:
            metadata_by_benchmark[benchmark] = metadata
    if errors:
        raise RuntimeError("\n".join(errors))
    return metadata_by_benchmark


def flatten_match(match: object) -> str:
    if isinstance(match, tuple):
        for item in match:
            if item != "":
                return str(item)
        return ""
    return str(match)


def extract_runtime(output: str, regex: str, multi_match: str) -> float:
    matches = re.findall(regex, output)
    if not matches:
        raise RuntimeError(f"no regex match for {regex!r}")
    values = [float(flatten_match(match)) for match in matches]
    if multi_match == "first":
        return values[0]
    if multi_match == "last":
        return values[-1]
    if multi_match == "sum":
        return sum(values)
    raise ValueError(f"unsupported multi-match mode: {multi_match}")


def ensure_output_dirs(output_dir: Path) -> None:
    for child in ("jobs", "logs"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def create_run_output_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    for suffix in ("", *(f"-{index}" for index in range(1, 1000))):
        run_dir = base_dir / f"run-{stamp}{suffix}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a unique run directory under {base_dir}")


def write_locked_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_locked_summary(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0, os.SEEK_END)
            needs_header = handle.tell() == 0
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_command(
    command: Sequence[str],
    cwd: Path,
    timeout_seconds: int,
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        timeout=timeout_seconds,
        check=False,
    )
    log_path.write_text(
        "$ " + " ".join(shlex.quote(part) for part in command) + "\n\n" + proc.stdout,
        encoding="utf-8",
    )
    return proc


def make_command(target: str | None, config: RunnerConfig) -> list[str]:
    command = ["make"]
    if target:
        command.append(target)
    command.append(f"ARCH={config.arch}")
    command.extend(config.make_args)
    return command


def benchmark_run_command(metadata: BenchmarkMetadata | None, config: RunnerConfig) -> list[str]:
    if metadata is None:
        return make_command("run", config)
    return [f"./{metadata.binary}", *metadata.args]


def sample_record(
    benchmark: str,
    phase: str,
    index: int,
    status: str,
    value: float | None,
    metadata: BenchmarkMetadata | None,
    log_path: Path,
    error: str = "",
) -> dict[str, object]:
    return {
        "timestamp": timestamp(),
        "benchmark": benchmark,
        "phase": phase,
        "sample_index": index,
        "status": status,
        "value": value,
        "metadata_source": metadata_source(metadata),
        "log_path": str(log_path),
        "error": error,
    }


def run_benchmark(
    benchmark: str,
    config: RunnerConfig,
    root: Path = ROOT,
) -> int:
    ensure_output_dirs(config.output_dir)
    metadata = find_metadata(benchmark, root, config.metadata_paths)
    warn_unsummarized_metadata(benchmark, metadata)
    bench_path = benchmark_dir(benchmark, root)
    log_dir = config.output_dir / "logs" / benchmark
    raw_path = config.output_dir / "raw.jsonl"
    summary_path = config.output_dir / "summary.csv"
    values: list[float] = []
    completed_runs = 0
    status = "success"
    error = ""

    try:
        if not bench_path.is_dir():
            raise RuntimeError(f"missing CUDA directory {bench_path}")

        build_log = log_dir / "build.log"
        build = run_command(make_command(None, config), bench_path, config.build_timeout, build_log)
        if build.returncode != 0:
            raise RuntimeError(f"build failed with exit code {build.returncode}; see {build_log}")

        run_timeout = config.run_timeout or (metadata.timeout if metadata else 300)
        for phase, count in (("warmup", config.warmups), ("run", config.runs)):
            for index in range(1, count + 1):
                log_path = log_dir / f"{phase}-{index}.log"
                try:
                    command = benchmark_run_command(metadata, config)
                    proc = run_command(command, bench_path, run_timeout, log_path)
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"benchmark run failed with exit code {proc.returncode}; see {log_path}"
                        )
                    value = (
                        extract_runtime(proc.stdout, metadata.regex, config.multi_match)
                        if metadata and metadata.regex
                        else None
                    )
                    write_locked_jsonl(
                        raw_path,
                        sample_record(benchmark, phase, index, "success", value, metadata, log_path),
                    )
                    if phase == "run":
                        completed_runs += 1
                        if value is not None:
                            values.append(value)
                except Exception as exc:
                    write_locked_jsonl(
                        raw_path,
                        sample_record(benchmark, phase, index, "failed", None, metadata, log_path, str(exc)),
                    )
                    raise
    except Exception as exc:
        status = "failed"
        error = str(exc)

    if should_write_summary(metadata):
        mean = statistics.fmean(values) if values else ""
        stddev = statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")
        write_locked_summary(
            summary_path,
            {
                "benchmark": benchmark,
                "status": status,
                "runs_completed": completed_runs,
                "warmups": config.warmups,
                "mean": mean,
                "stddev": stddev,
                "run_values": json.dumps(values),
                "job_id": os.environ.get("SLURM_JOB_ID", ""),
                "metadata_source": metadata_source(metadata),
                "error": error,
                "log_dir": str(log_dir),
            },
        )
    return 0 if status == "success" else 1


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def print_line_command(label: str, value: str) -> str:
    return f"printf '%s\\n' {shlex.quote(f'{label}: {value}')}"


def sanitize_job_name(benchmark: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{benchmark}")[:120]


def worker_command(script: Path, benchmark: str, config: RunnerConfig) -> list[str]:
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--benchmark",
        benchmark,
        "--warmups",
        str(config.warmups),
        "--runs",
        str(config.runs),
        "--arch",
        config.arch,
        "--output-dir",
        str(config.output_dir),
        "--multi-match",
        config.multi_match,
        "--build-timeout",
        str(config.build_timeout),
    ]
    if config.run_timeout is not None:
        command.extend(["--timeout", str(config.run_timeout)])
    for make_arg in config.make_args:
        command.extend(["--make-arg", make_arg])
    for metadata_path in config.metadata_paths:
        command.extend(["--metadata", str(metadata_path)])
    return command


def write_sbatch_script(
    benchmark: str,
    config: RunnerConfig,
    sbatch_args: Sequence[str],
    root: Path,
) -> Path:
    jobs_dir = config.output_dir / "jobs"
    logs_dir = config.output_dir / "logs" / benchmark
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    script_path = jobs_dir / f"{benchmark}.sbatch"
    job_name = sanitize_job_name(benchmark)
    worker = worker_command(root / "run_benchmarks.py", benchmark, config)
    benchmark_command = benchmark_run_command(find_metadata(benchmark, root, config.metadata_paths), config)
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gres=gpu:1",
        f"#SBATCH --output={logs_dir / 'slurm-%x-%j.out'}",
        f"#SBATCH --error={logs_dir / 'slurm-%x-%j.err'}",
    ]
    lines.extend(f"#SBATCH {arg}" for arg in sbatch_args)
    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(root))}",
            print_line_command("benchmark", benchmark),
            print_line_command("output_dir", str(config.output_dir)),
            print_line_command("benchmark_command", shell_join(benchmark_command)),
            shell_join(worker),
            "",
        ]
    )
    script_path.write_text("\n".join(lines), encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def submit_slurm_jobs(
    benchmarks: Sequence[str],
    config: RunnerConfig,
    sbatch_args: Sequence[str],
    root: Path,
    dry_run: bool,
) -> int:
    ensure_output_dirs(config.output_dir)
    manifest = config.output_dir / "jobs.jsonl"
    for benchmark in benchmarks:
        script = write_sbatch_script(benchmark, config, sbatch_args, root)
        if dry_run:
            print(f"wrote {script}")
            write_locked_jsonl(
                manifest,
                {
                    "timestamp": timestamp(),
                    "benchmark": benchmark,
                    "status": "dry-run",
                    "script": str(script),
                },
            )
            continue

        proc = subprocess.run(
            ["sbatch", str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=False,
        )
        match = re.search(r"Submitted batch job\s+(\d+)", proc.stdout)
        job_id = match.group(1) if match else ""
        status = "submitted" if proc.returncode == 0 else "failed"
        write_locked_jsonl(
            manifest,
            {
                "timestamp": timestamp(),
                "benchmark": benchmark,
                "status": status,
                "script": str(script),
                "job_id": job_id,
                "sbatch_output": proc.stdout.strip(),
            },
        )
        if proc.returncode != 0:
            print(f"{benchmark}: sbatch failed:\n{proc.stdout}", file=sys.stderr)
            return proc.returncode
        print(f"{benchmark}: submitted job {job_id or proc.stdout.strip()}")
    return 0


def make_config(args: argparse.Namespace) -> RunnerConfig:
    return RunnerConfig(
        arch=args.arch,
        warmups=args.warmups,
        runs=args.runs,
        output_dir=Path(args.output_dir).resolve(),
        multi_match=args.multi_match,
        make_args=tuple(args.make_arg or ()),
        build_timeout=args.build_timeout,
        run_timeout=args.timeout,
        metadata_paths=tuple(Path(path) for path in (args.metadata or ())),
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run CUDA HeCBench benchmarks")
    parser.add_argument(
        "--benchmarks-file",
        help="Optional file containing one benchmark per line; defaults to BENCHMARKS in this script",
    )
    parser.add_argument("--warmups", type=nonnegative_int, default=3, help="Warmup make-run executions")
    parser.add_argument("--runs", type=positive_int, default=10, help="Measured make-run executions")
    parser.add_argument("--arch", default=NVIDIA_ARCH, help="CUDA ARCH make variable, e.g. sm_90")
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Base directory; each parent invocation creates a run-* child for logs and results",
    )
    parser.add_argument("--slurm", action="store_true",default=True, help="Submit one sbatch job per benchmark")
    parser.add_argument("--sbatch-arg", action="append", default=[], help="Additional #SBATCH directive")
    parser.add_argument("--dry-run", action="store_true", help="Write Slurm scripts without submitting")
    parser.add_argument("--metadata", action="append", default=[], help="Additional metadata file path")
    parser.add_argument("--make-arg", action="append", default=[], help="Extra make argument, e.g. CC=nvcc")
    parser.add_argument("--build-timeout", type=positive_int, default=1800, help="Build timeout in seconds")
    parser.add_argument("--timeout", type=positive_int, help="Override per-run timeout in seconds")
    parser.add_argument(
        "--multi-match",
        choices=("last", "sum", "first"),
        default="last",
        help="How to handle multiple regex matches from one make run",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--benchmark", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    config = make_config(args)

    if args.worker:
        if not args.benchmark:
            print("--worker requires --benchmark", file=sys.stderr)
            return 2
        benchmark = normalize_benchmark_name(args.benchmark)
        return run_benchmark(benchmark, config)

    config = replace(config, output_dir=create_run_output_dir(config.output_dir))
    print(f"output directory: {config.output_dir}", flush=True)

    try:
        benchmarks = (
            parse_benchmark_list(Path(args.benchmarks_file))
            if args.benchmarks_file
            else normalize_benchmark_list(BENCHMARKS)
        )
        validate_benchmarks(benchmarks, ROOT, config.metadata_paths)
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.slurm:
        return submit_slurm_jobs(benchmarks, config, args.sbatch_arg, ROOT, args.dry_run)

    failures = 0
    for benchmark in benchmarks:
        print(f"running {benchmark}", flush=True)
        failures += 1 if run_benchmark(benchmark, config) else 0
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
