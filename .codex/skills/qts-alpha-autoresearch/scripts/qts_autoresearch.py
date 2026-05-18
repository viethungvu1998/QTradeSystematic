#!/usr/bin/env python
"""QTS Alpha Autoresearch — autonomous experiment loop for alpha research.

Subcommands
-----------
preflight <manifest>             Validate manifest; print JSON diagnostic.
loop <manifest> [--max-steps N]  Run baseline + experiment trials.
status <manifest>                Print current ledger status.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PYTHON = Path(__file__).resolve().parents[4] / ".venv" / "bin" / "python"
if not PYTHON.exists():
    # fallback to sys.executable (e.g. when called from inside the venv)
    PYTHON = Path(sys.executable)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def project_root(manifest_path: Path) -> Path:
    """Walk up from manifest to find pyproject.toml."""
    for candidate in manifest_path.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    # fallback: 4 levels up from autoresearch.yaml
    return manifest_path.parents[3]


def ledger_dir(manifest: dict[str, Any], manifest_path: Path) -> Path:
    root = project_root(manifest_path)
    raw = manifest.get("paths", {}).get("ledger", "")
    p = Path(raw) if raw else manifest_path.parent / "ledger"
    return p if p.is_absolute() else (root / p).resolve()


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def cmd_preflight(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        result = {"ok": False, "blocked_by": "manifest_missing", "details": str(manifest_path)}
        print(json.dumps(result, indent=2))
        return result

    manifest = load_manifest(manifest_path)
    warnings: list[str] = []

    algo_dir = manifest_path.parent.parent
    if not (algo_dir / "ALGORITHM.md").exists():
        warnings.append(f"ALGORITHM.md missing at {algo_dir / 'ALGORITHM.md'}")

    root = project_root(manifest_path)
    module_rel = manifest.get("command", {}).get("module", "")
    module_path = root / module_rel
    if not module_path.exists():
        result = {
            "ok": False,
            "blocked_by": "command_not_found",
            "details": str(module_path),
        }
        print(json.dumps(result, indent=2))
        return result

    allowed = manifest.get("research_limits", {}).get("allowed_actions", [])
    if not allowed:
        result = {"ok": False, "blocked_by": "no_allowed_actions"}
        print(json.dumps(result, indent=2))
        return result

    result = {
        "ok": True,
        "blocked_by": None,
        "contract_warnings": warnings,
        "algorithm": manifest.get("algorithm", "unknown"),
        "quota": manifest.get("quota", {}),
        "criteria": manifest.get("criteria", {}),
        "first_experiments": len(manifest.get("first_experiments", [])),
    }
    print(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Metrics I/O
# ---------------------------------------------------------------------------

def read_metrics(trial_dir: Path) -> dict[str, float] | None:
    metrics_json = trial_dir / "metrics.json"
    if metrics_json.exists():
        with open(metrics_json) as f:
            return json.load(f)

    results_csv = trial_dir / "results_summary.csv"
    if results_csv.exists():
        with open(results_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            row = rows[0]
            metrics: dict[str, float] = {}
            for key in ("sharpe", "sortino", "cagr", "max_drawdown", "win_rate"):
                try:
                    metrics[key] = float(row[key])
                except (KeyError, ValueError, TypeError):
                    metrics[key] = 0.0
            with open(metrics_json, "w") as f:
                json.dump(metrics, f, indent=2)
            return metrics

    return None


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

RESULTS_HEADER = [
    "trial_id", "trial_name", "action", "args",
    "sharpe", "sortino", "cagr", "max_drawdown", "win_rate",
    "status", "kept", "timestamp",
]


def results_tsv(ledger: Path) -> Path:
    return ledger / "results.tsv"


def events_jsonl(ledger: Path) -> Path:
    return ledger / "events.jsonl"


def read_results(ledger: Path) -> list[dict[str, Any]]:
    path = results_tsv(ledger)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def append_result(ledger: Path, row: dict[str, Any]) -> None:
    path = results_tsv(ledger)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULTS_HEADER})


def append_event(ledger: Path, event: dict[str, Any]) -> None:
    with open(events_jsonl(ledger), "a") as f:
        f.write(json.dumps(event) + "\n")


def best_sharpe(results: list[dict[str, Any]]) -> float:
    kept = [r for r in results if r.get("kept") == "true" and r.get("status") == "ok"]
    if not kept:
        return float("-inf")
    return max(float(r.get("sharpe", 0)) for r in kept)


def best_row(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    kept = [r for r in results if r.get("kept") == "true" and r.get("status") == "ok"]
    if not kept:
        return None
    return max(kept, key=lambda r: float(r.get("sharpe", 0)))


def criteria_met(manifest: dict[str, Any], results: list[dict[str, Any]]) -> bool:
    crit = manifest.get("criteria", {})
    row = best_row(results)
    if row is None:
        return False
    checks = []
    if "sharpe_ge" in crit:
        checks.append(float(row.get("sharpe", 0)) >= float(crit["sharpe_ge"]))
    if "max_drawdown_le" in crit:
        checks.append(float(row.get("max_drawdown", 1)) <= float(crit["max_drawdown_le"]))
    if "cagr_ge" in crit:
        checks.append(float(row.get("cagr", 0)) >= float(crit["cagr_ge"]))
    return bool(checks) and all(checks)


def trials_run(ledger: Path) -> int:
    return len(read_results(ledger))


def experiments_tried(ledger: Path) -> set[str]:
    return {r["trial_name"] for r in read_results(ledger)}


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

def run_trial(
    manifest: dict[str, Any],
    root: Path,
    trial_id: int,
    trial_name: str,
    action: str,
    extra_args: list[str],
    ledger: Path,
    current_best_args: list[str],
) -> dict[str, Any]:
    cmd_cfg = manifest.get("command", {})
    module = root / cmd_cfg["module"]
    base_args: list[str] = cmd_cfg.get("base_args", [])

    trial_dir = ledger / f"trial_{trial_id:03d}_{trial_name}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Cumulative: current_best_args carry forward improvements
    full_args = base_args + current_best_args + extra_args + [
        f"--output-dir={trial_dir}",
        f"--run-name={trial_name}",
    ]
    cmd = [str(PYTHON), str(module)] + full_args

    print(f"\n  Trial {trial_id:03d} [{action}]: {trial_name}")
    print(f"  Args: {' '.join(extra_args) or '(baseline)'}")

    log_path = trial_dir / "run.log"
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(root),
    )
    log_path.write_text(result.stdout)

    # Print tail of output for visibility
    lines = result.stdout.strip().splitlines()
    tail = lines[-20:] if len(lines) > 20 else lines
    for line in tail:
        print(f"    {line}")

    if result.returncode != 0:
        return {
            "ok": False,
            "error": f"Exit code {result.returncode}",
            "trial_dir": trial_dir,
        }

    metrics = read_metrics(trial_dir)
    if metrics is None:
        return {
            "ok": False,
            "error": "No metrics output found",
            "trial_dir": trial_dir,
        }

    return {"ok": True, "metrics": metrics, "trial_dir": trial_dir}


# ---------------------------------------------------------------------------
# Summary documents
# ---------------------------------------------------------------------------

def write_summary(manifest_path: Path, ledger: Path, results: list[dict[str, Any]]) -> None:
    manifest = load_manifest(manifest_path)
    row = best_row(results)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Autoresearch Summary — {manifest.get('algorithm', 'unknown')}",
        f"_Updated {now}_",
        "",
        f"**Trials run**: {len(results)}",
        f"**Max trials**: {manifest.get('quota', {}).get('max_trials', '?')}",
        "",
    ]
    if row:
        lines += [
            "## Best trial",
            f"- **Name**: `{row['trial_name']}`",
            f"- **Sharpe**: {float(row.get('sharpe', 0)):.4f}",
            f"- **CAGR**: {float(row.get('cagr', 0)):.2%}",
            f"- **Max drawdown**: {float(row.get('max_drawdown', 0)):.2%}",
            f"- **Sortino**: {float(row.get('sortino', 0)):.4f}",
            f"- **Win rate**: {float(row.get('win_rate', 0)):.2%}",
            "",
        ]
    else:
        lines += ["## Best trial", "_No successful trials yet._", ""]

    crit = manifest.get("criteria", {})
    lines += ["## Success criteria"]
    for k, v in crit.items():
        met = "✓" if row and _check_criterion(k, v, row) else "✗"
        lines.append(f"- {met} `{k}`: {v}")
    lines.append("")

    lines += ["## All trials", "| ID | Name | Sharpe | CAGR | MaxDD | Kept |", "|---|---|---|---|---|---|"]
    for r in results:
        sharpe = float(r.get("sharpe", 0))
        cagr = float(r.get("cagr", 0))
        maxdd = float(r.get("max_drawdown", 0))
        kept = "✓" if r.get("kept") == "true" else "✗"
        lines.append(f"| {r['trial_id']} | {r['trial_name']} | {sharpe:.3f} | {cagr:.1%} | {maxdd:.1%} | {kept} |")

    (ledger / "summary.md").write_text("\n".join(lines) + "\n")


def _check_criterion(key: str, value: Any, row: dict[str, Any]) -> bool:
    if key == "sharpe_ge":
        return float(row.get("sharpe", 0)) >= float(value)
    if key == "max_drawdown_le":
        return float(row.get("max_drawdown", 1)) <= float(value)
    if key == "cagr_ge":
        return float(row.get("cagr", 0)) >= float(value)
    return False


def write_next_steps(manifest_path: Path, ledger: Path, remaining: list[dict[str, Any]]) -> None:
    lines = [
        "# Next Steps",
        "",
        f"**Queued experiments** ({len(remaining)} remaining):",
        "",
    ]
    for exp in remaining[:10]:
        lines.append(f"- `{exp['name']}` — {exp['action']} {' '.join(exp.get('args', []))}")
    if len(remaining) > 10:
        lines.append(f"- … and {len(remaining) - 10} more")
    (ledger / "next_steps.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def cmd_loop(manifest_path: Path, max_steps: int | None) -> int:
    pf = cmd_preflight(manifest_path)
    if not pf["ok"]:
        print(f"\nPreflight failed: blocked_by={pf['blocked_by']}", file=sys.stderr)
        return 1

    manifest = load_manifest(manifest_path)
    root = project_root(manifest_path)
    ledger = ledger_dir(manifest, manifest_path)
    ledger.mkdir(parents=True, exist_ok=True)

    quota_max = int(manifest.get("quota", {}).get("max_trials", 20))
    first_exps: list[dict[str, Any]] = manifest.get("first_experiments", [])

    already_tried = experiments_tried(ledger)
    results = read_results(ledger)
    trial_count = len(results)

    print(f"\n=== QTS Autoresearch: {manifest.get('algorithm')} ===")
    print(f"Ledger: {ledger}")
    print(f"Trials so far: {trial_count} / {quota_max}")
    print(f"Already tried: {already_tried or '(none)'}")

    # Track current best args cumulatively
    current_best_args: list[str] = []
    # Reconstruct from existing results
    for r in results:
        if r.get("kept") == "true" and r.get("args"):
            current_best_args = r["args"].split("|||") if "|||" in r["args"] else []

    step = 0

    # --- Baseline (trial 000) ---
    if "baseline" not in already_tried:
        print("\n--- Running baseline (trial 000) ---")
        out = run_trial(
            manifest, root,
            trial_id=0,
            trial_name="baseline",
            action="baseline",
            extra_args=[],
            ledger=ledger,
            current_best_args=[],
        )
        ts = datetime.now(timezone.utc).isoformat()
        if out["ok"]:
            m = out["metrics"]
            row: dict[str, Any] = {
                "trial_id": 0,
                "trial_name": "baseline",
                "action": "baseline",
                "args": "",
                **m,
                "status": "ok",
                "kept": "true",
                "timestamp": ts,
            }
            append_result(ledger, row)
            append_event(ledger, {"type": "trial", "name": "baseline", **row})
            results = read_results(ledger)
            print(f"  Baseline Sharpe: {m['sharpe']:.4f}, CAGR: {m['cagr']:.2%}, MaxDD: {m['max_drawdown']:.2%}")
        else:
            row = {
                "trial_id": 0,
                "trial_name": "baseline",
                "action": "baseline",
                "args": "",
                "sharpe": 0, "sortino": 0, "cagr": 0, "max_drawdown": 0, "win_rate": 0,
                "status": "failed",
                "kept": "false",
                "timestamp": ts,
            }
            append_result(ledger, row)
            append_event(ledger, {"type": "trial_failed", "name": "baseline", "error": out.get("error")})
            results = read_results(ledger)
            print(f"  Baseline FAILED: {out.get('error')}", file=sys.stderr)
        step += 1
        write_summary(manifest_path, ledger, results)
        trial_count = len(results)
        already_tried.add("baseline")

        if max_steps is not None and step >= max_steps:
            print(f"\nReached --max-steps {max_steps}. Stopping.")
            _print_stop_summary(results, ledger, "max_steps")
            return 0

    # --- Experiment loop ---
    remaining = [e for e in first_exps if e["name"] not in already_tried]

    while True:
        if trial_count >= quota_max:
            print(f"\nQuota exhausted ({trial_count}/{quota_max}).")
            write_next_steps(manifest_path, ledger, remaining)
            write_summary(manifest_path, ledger, results)
            _print_stop_summary(results, ledger, "quota_exhausted")
            return 0

        if criteria_met(manifest, results):
            print("\nCriteria met! Stopping.")
            write_summary(manifest_path, ledger, results)
            _print_stop_summary(results, ledger, "criteria_met")
            return 0

        if max_steps is not None and step >= max_steps:
            print(f"\nReached --max-steps {max_steps}. Stopping.")
            write_next_steps(manifest_path, ledger, remaining)
            write_summary(manifest_path, ledger, results)
            _print_stop_summary(results, ledger, "max_steps")
            return 0

        if not remaining:
            print("\nNo more queued experiments. Stopping (quota not exhausted, criteria not met).")
            write_summary(manifest_path, ledger, results)
            _print_stop_summary(results, ledger, "experiments_exhausted")
            return 0

        exp = remaining.pop(0)
        trial_id = trial_count
        trial_name = exp["name"]
        action = exp.get("action", "tune_hyperparams")
        extra_args: list[str] = exp.get("args", [])

        out = run_trial(
            manifest, root,
            trial_id=trial_id,
            trial_name=trial_name,
            action=action,
            extra_args=extra_args,
            ledger=ledger,
            current_best_args=current_best_args,
        )

        ts = datetime.now(timezone.utc).isoformat()
        prev_best = best_sharpe(results)

        if out["ok"]:
            m = out["metrics"]
            improved = m["sharpe"] > prev_best
            kept = improved or trial_count == 0  # always keep if first non-baseline
            if kept:
                current_best_args = current_best_args + extra_args
            row = {
                "trial_id": trial_id,
                "trial_name": trial_name,
                "action": action,
                "args": "|||".join(extra_args),
                **m,
                "status": "ok",
                "kept": str(kept).lower(),
                "timestamp": ts,
            }
            verdict = "KEPT ↑" if kept else "rolled back (no improvement)"
            print(f"  Sharpe: {m['sharpe']:.4f} vs prev best {prev_best:.4f} → {verdict}")
        else:
            row = {
                "trial_id": trial_id,
                "trial_name": trial_name,
                "action": action,
                "args": "|||".join(extra_args),
                "sharpe": 0, "sortino": 0, "cagr": 0, "max_drawdown": 0, "win_rate": 0,
                "status": "failed",
                "kept": "false",
                "timestamp": ts,
            }
            print(f"  FAILED: {out.get('error')}", file=sys.stderr)

        append_result(ledger, row)
        append_event(ledger, {"type": "trial", "name": trial_name, **row})
        results = read_results(ledger)
        write_summary(manifest_path, ledger, results)
        already_tried.add(trial_name)
        trial_count = len(results)
        step += 1

    return 0  # unreachable


def _print_stop_summary(results: list[dict[str, Any]], ledger: Path, reason: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Stop reason : {reason}")
    print(f"Trials run  : {len(results)}")
    row = best_row(results)
    if row:
        print(f"Best trial  : {row['trial_name']}")
        print(f"  Sharpe    : {float(row.get('sharpe', 0)):.4f}")
        print(f"  CAGR      : {float(row.get('cagr', 0)):.2%}")
        print(f"  Max DD    : {float(row.get('max_drawdown', 0)):.2%}")
    else:
        print("Best trial  : none (all failed)")
    print(f"summary.md  : {ledger / 'summary.md'}")
    print(f"next_steps  : {ledger / 'next_steps.md'}")
    print(f"results.tsv : {ledger / 'results.tsv'}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Status subcommand
# ---------------------------------------------------------------------------

def cmd_status(manifest_path: Path) -> int:
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    manifest = load_manifest(manifest_path)
    ledger = ledger_dir(manifest, manifest_path)
    results = read_results(ledger)
    _print_stop_summary(results, ledger, "status_query")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preflight", help="Validate manifest and print JSON diagnostic")
    p_pre.add_argument("manifest", type=Path)

    p_loop = sub.add_parser("loop", help="Run the full autoresearch loop")
    p_loop.add_argument("manifest", type=Path)
    p_loop.add_argument("--max-steps", type=int, default=None)

    p_status = sub.add_parser("status", help="Print current ledger status")
    p_status.add_argument("manifest", type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "preflight":
        result = cmd_preflight(args.manifest)
        return 0 if result["ok"] else 1
    if args.cmd == "loop":
        return cmd_loop(args.manifest, args.max_steps)
    if args.cmd == "status":
        return cmd_status(args.manifest)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
