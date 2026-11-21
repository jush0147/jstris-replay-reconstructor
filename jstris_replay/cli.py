"""Command-line interface for Jstris Replay decoding and full match reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .decoder import decode_replay_code
from .exporter import build_summary, export_all_json
from .simulator import JstrisSimulator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jstris-replay",
        description="Decode Jstris Replay Codes and reconstruct full step-by-step match states.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Path to a text file containing the Jstris Replay Code (or raw JSON).",
    )
    group.add_argument(
        "-c",
        "--code",
        type=str,
        help="Direct Jstris Replay Code string.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Directory to write metadata.json, events.json, placements.json, states.json, and summary.json.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console summary output.",
    )

    args = parser.parse_args(argv)
    if args.input is not None:
        if not args.input.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            return 1
        replay_str = args.input.read_text(encoding="utf-8").strip()
    else:
        replay_str = args.code.strip()

    decoded = decode_replay_code(replay_str)
    sim = JstrisSimulator(decoded)
    sim.run()

    files = export_all_json(decoded, sim, args.output_dir)
    summary = build_summary(decoded, sim)

    if not args.quiet:
        ov = summary["game_overview"]
        perf = summary["performance_metrics"]
        ver = summary["verification"]
        print("=== Jstris Replay Reconstruction Summary ===")
        print(f"Status             : {ver['status']} (All locks legal: {ver['all_locks_legal']})")
        print(f"Mode               : {ov['mode_description']} (raw m={ov['mode_raw']} -> pmode={ov['pmode']}, subMode={ov['sub_mode']})")
        print(f"Version / Seed     : v{ov['version']} / seed='{ov['seed']}'")
        print(f"Game Duration      : {ov['duration_sec']:.3f} s ({ver['final_event_clock_ms']} ms)")
        print(f"Total Events       : {ver['events_processed']} / {ver['total_events']}")
        print(f"Placed Blocks      : {ov['placed_blocks']}")
        print(f"Lines Remaining    : {ov['lines_remaining']} (Target: {ov['target_lines']})")
        print(f"Garbage Cleared    : {ov['garbage_lines_cleared']} / {ov['garbage_lines_received']} received")
        print(f"Total Lines Cleared: {ov['total_lines_cleared']} (Singles={summary['line_clear_breakdown']['singles']}, Doubles={summary['line_clear_breakdown']['doubles']}, Triples={summary['line_clear_breakdown']['triples']}, Tetrises={summary['line_clear_breakdown']['tetrises']})")
        print(f"Holds Used         : {ov['holds_used']}")
        print(f"PPS / KPP / APM    : {perf['pps']} PPS | {perf['kpp']} KPP | {perf['apm']} APM | {perf['vs_score']} VS")
        print(f"Attack / Score     : {perf['attack_sent']} attack | {ov['score']} pts | Max Combo={perf['max_combo']} | TSDs={perf['tsd_count']}")
        print(f"Finesse Faults     : {perf['total_finesse_faults']} (Total Keypresses: {perf['total_key_presses']})")
        print("Exported Files     :")
        for name, fpath in files.items():
            print(f"  - {name:10s}: {fpath.resolve()}")

    return 0 if ver["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
