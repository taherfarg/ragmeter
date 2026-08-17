"""Gate verdict as plain text, for a CI log."""

from ragmeter.gate.compare import GateResult

__all__ = ["render_gate"]


def _num(value: float | None) -> str:
    if value is None:
        return "-"
    if value != 0 and abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def render_gate(result: GateResult, run: str, baseline: str, k: int) -> str:
    verdict = "PASS" if result.passed else "FAIL"
    lines = [
        f"gate: {verdict}",
        f"  run={run}  baseline={baseline}  k={k}  paired questions={result.n_paired}",
        "",
        f"{'metric':<22}{'baseline':>12}{'candidate':>12}{'delta':>12}"
        f"{'limit':>10}{'+/-':>10}  verdict",
        "-" * 90,
    ]

    for out in result.outcomes:
        spread = f"+{out.n_improved}/-{out.n_regressed}" if out.kind == "paired" else ""
        lines.append(
            f"{out.name:<22}{_num(out.baseline):>12}{_num(out.candidate):>12}"
            f"{_num(out.delta):>12}{out.limit:>10.4f}{spread:>10}  "
            f"{'ok' if out.passed else 'FAIL'}"
            + (f"  ({out.reason})" if out.reason else "")
        )

    if result.blocking_reasons:
        lines.append("")
        lines.append("blocking:")
        lines.extend(f"  - {reason}" for reason in result.blocking_reasons)

    lines.append("")
    lines.append("+/- counts improved/regressed questions. A mean delta near zero "
                 "with a large spread is a redistribution, not a no-op.")
    return "\n".join(lines)
