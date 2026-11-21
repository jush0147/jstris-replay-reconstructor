"""Structured data export for Jstris Replay analysis and Tetris AI research."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from .decoder import DecodedReplay
from .rng import PIECE_NAMES
from .simulator import JstrisSimulator, matrix_to_ascii_rows


def build_summary(decoded: DecodedReplay, sim: JstrisSimulator) -> dict[str, Any]:
    """Build comprehensive summary and verification report dictionary."""
    action_counts = Counter(ev.action_name for ev in decoded.events)
    clear_type_counts = Counter(p.clear_type for p in sim.placements)
    piece_usage_counts = Counter(p.piece for p in sim.placements)

    # Full generated piece sequence from seed (post-hoc replay info)
    full_generated_sequence = [PIECE_NAMES[pid] for pid in sim.generated_pieces_history]
    # Sequence of all 100 garbage hole columns added over the entire match (post-hoc replay info)
    full_garbage_hole_sequence = [g["hole_column"] for g in sim.all_garbage_events]

    return {
        "verification": {
            "status": "VERIFIED" if not sim.anomalies else "ANOMALIES_DETECTED",
            "anomalies": sim.anomalies,
            "all_locks_legal": len(sim.anomalies) == 0,
            "terminator_0xfff_found": decoded.terminator_found,
            "bits_read": decoded.total_bits_read,
            "bits_available": decoded.total_bits_available,
            "events_processed": sim.ptr,
            "total_events": len(decoded.events),
            "target_lines_completed": sim.lines_remaining == 0,
            "recorded_duration_ms": decoded.metadata.duration_ms,
            "final_event_clock_ms": sim.clock,
            "clock_matches_metadata_duration": sim.clock == decoded.metadata.duration_ms,
            "verified_162_blocks_claim": sim.placed_blocks == 162,
        },
        "game_overview": {
            "mode_raw": decoded.metadata.raw_mode,
            "pmode": decoded.metadata.pmode,
            "sub_mode": decoded.metadata.sub_mode,
            "mode_description": decoded.metadata.mode_description,
            "target_lines": decoded.metadata.target_lines,
            "seed": decoded.metadata.seed,
            "version": decoded.metadata.version,
            "game_start_iso": decoded.metadata.game_start_iso,
            "game_end_iso": decoded.metadata.game_end_iso,
            "duration_sec": decoded.metadata.duration_sec,
            "placed_blocks": sim.placed_blocks,
            "lines_remaining": sim.lines_remaining,
            "total_lines_cleared": sim.gamedata["lines"],
            "garbage_lines_cleared": sim.gamedata["garbageCleared"],
            "garbage_lines_received": sim.gamedata["linesReceived"],
            "holds_used": sim.gamedata["holds"],
            "score": sim.gamedata["score"],
        },
        "performance_metrics": {
            "pps": sim.get_pps(),
            "kpp": sim.get_kpp(),
            "apm": sim.get_apm(),
            "vs_score": sim.get_vs(),
            "total_key_presses": sim.total_key_presses,
            "total_finesse_faults": sim.total_finesse,
            "finesse_fault_placement_ids": sim.fin_faults,
            "attack_sent": sim.gamedata["attack"],
            "lines_sent": sim.gamedata["linesSent"],
            "max_combo": sim.gamedata["maxCombo"],
            "b2b_count": sim.gamedata["B2B"],
            "t_pieces_placed": sim.gamedata["tpieces"],
            "t_spins": sim.gamedata["tspins"],
            "tsd_count": sim.gamedata["TSD"],
            "wasted_t_pieces": sim.gamedata["wasted"],
            "wasted_t_ratio": sim.get_wasted_t_ratio(),
            "perfect_clears": sim.gamedata["PCs"],
        },
        "line_clear_breakdown": {
            "singles": sim.gamedata["singles"],
            "doubles": sim.gamedata["doubles"],
            "triples": sim.gamedata["triples"],
            "tetrises": sim.gamedata["tetrises"],
            "by_clear_type": dict(clear_type_counts),
        },
        "event_breakdown": {
            "total_events": len(decoded.events),
            "by_action_name": dict(action_counts),
        },
        "piece_breakdown": {
            "placed_counts": dict(piece_usage_counts),
            "total_generated_from_rng": len(full_generated_sequence),
        },
        "final_board": {
            "matrix": sim.matrix,
            "ascii": matrix_to_ascii_rows(sim.matrix),
            "deadline": sim.deadline,
            "active_piece_at_finise": PIECE_NAMES[sim.active_piece_id],
            "hold_piece_at_finise": PIECE_NAMES[sim.hold_piece_id] if sim.hold_piece_id is not None else None,
            "visible_previews_at_finise": [PIECE_NAMES[pid] for pid in sim.queue[: sim.show_previews]],
        },
        "information_boundary": {
            "legally_observable_at_decision_time": {
                "description": (
                    "At each placement decision, a legal player/bot can only observe the current 20x10 board "
                    "(including currently visible garbage rows), the active falling piece, the current hold piece, "
                    f"and the {sim.show_previews} visible preview pieces in the queue."
                ),
                "visible_preview_count": sim.show_previews,
                "hold_enabled": sim.hold_enabled,
                "initial_cheese_garbage_rows": 9,
                "min_cheese_garbage_rows": 3,
                "max_cheese_garbage_rows": 9,
            },
            "post_hoc_replay_reconstructed_only": {
                "description": (
                    "The following full sequences are reconstructed from the replay seed and GARBAGE_ADD stream. "
                    "They MUST NOT be leaked to a bot as lookahead beyond the 5 visible previews or current board."
                ),
                "full_generated_piece_sequence_str": "".join(full_generated_sequence),
                "full_generated_piece_sequence": full_generated_sequence,
                "full_garbage_hole_columns_100": full_garbage_hole_sequence,
                "garbage_events": sim.all_garbage_events,
            },
            "unrecoverable_from_replay_alone": [
                "Live garbage PRNG seed (Jstris uses alea(Date.now()) for garbage holes in practice/cheese and records the resulting hole columns via GARBAGE_ADD events rather than deriving them from c.seed).",
                "Bot internal search statistics (nodes expanded, search depth, evaluation weights, or thinking time prior to first keypress on each piece).",
                "Key release timestamps for non-soft-drop keys (Jstris only records discrete movement/rotation trigger timestamps).",
            ],
        },
    }


def export_all_json(decoded: DecodedReplay, sim: JstrisSimulator, output_dir: str | Path) -> dict[str, Path]:
    """Write metadata.json, events.json, placements.json, states.json, and summary.json to output_dir."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metadata_file = out_path / "metadata.json"
    events_file = out_path / "events.json"
    placements_file = out_path / "placements.json"
    states_file = out_path / "states.json"
    summary_file = out_path / "summary.json"

    metadata_dict = decoded.metadata.to_dict()
    events_list = [ev.to_dict() for ev in decoded.events]
    placements_list = [p.to_dict() for p in sim.placements]
    states_list = [s.to_dict() for s in sim.decision_states]
    summary_dict = build_summary(decoded, sim)

    metadata_file.write_text(json.dumps(metadata_dict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    events_file.write_text(json.dumps(events_list, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    placements_file.write_text(json.dumps(placements_list, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    states_file.write_text(json.dumps(states_list, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_file.write_text(json.dumps(summary_dict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "metadata": metadata_file,
        "events": events_file,
        "placements": placements_file,
        "states": states_file,
        "summary": summary_file,
    }
