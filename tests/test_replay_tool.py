"""Comprehensive automated test suite for Jstris Replay decoding, full match reconstruction, and integrity verification."""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from jstris_replay import (
    ActionType,
    AleaPRNG,
    AuxType,
    BagRandomizer,
    JstrisSimulator,
    PIECE_NAMES,
    ReplayEvent,
    build_summary,
    compress_to_encoded_uri_component,
    decode_replay_code,
    decompress_from_encoded_uri_component,
    export_all_json,
)
from jstris_replay.simulator import get_occupied_cells

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_REPLAY_1_PATH = DATA_DIR / "sample_100l_cheese_replay.txt"
SAMPLE_REPLAY_2_PATH = DATA_DIR / "sample_100l_cheese_replay_2.txt"

ALLOWED_OBSERVABLE_STATE_KEYS = {
    "active_piece",
    "hold_piece",
    "can_hold",
    "visible_previews",
    "board",
    "board_ascii",
    "deadline",
    "garbage_height",
    "lines_remaining",
    "cumulative_lines_cleared",
    "cumulative_garbage_cleared",
    "combo",
    "back_to_back",
}


class TestJstrisReplaySample1Zglrnd(unittest.TestCase):
    """Regression test suite for original Sample 1 ('zglrnd', 162 blocks, 47.695s)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.replay_code = SAMPLE_REPLAY_1_PATH.read_text(encoding="utf-8").strip()
        cls.decoded = decode_replay_code(cls.replay_code)
        cls.sim = JstrisSimulator(cls.decoded)
        cls.sim.run()
        cls.summary = build_summary(cls.decoded, cls.sim)

    def test_lzstring_roundtrip(self) -> None:
        raw_json_str = decompress_from_encoded_uri_component(self.replay_code)
        self.assertIsNotNone(raw_json_str)
        assert raw_json_str is not None
        recompressed = compress_to_encoded_uri_component(raw_json_str)
        redecompressed = decompress_from_encoded_uri_component(recompressed)
        self.assertEqual(raw_json_str, redecompressed)

    def test_alea_prng_reference_vectors(self) -> None:
        rng = AleaPRNG("asdf")
        self.assertAlmostEqual(rng(), 0.8024188503623009, places=15)
        self.assertAlmostEqual(rng(), 0.4725297694094479, places=15)
        self.assertAlmostEqual(rng(), 0.949664750834927, places=15)
        self.assertAlmostEqual(rng(), 0.5619115477893502, places=15)
        self.assertAlmostEqual(rng(), 0.6947485841810703, places=15)

    def test_seed_zglrnd_bag_sequence(self) -> None:
        rng = AleaPRNG("zglrnd")
        bag = BagRandomizer(rng, 7, 1)
        first_14 = [PIECE_NAMES[bag.get_block_id()] for _ in range(14)]
        self.assertEqual(sorted(first_14[:7]), sorted(PIECE_NAMES))
        self.assertEqual(sorted(first_14[7:14]), sorted(PIECE_NAMES))
        self.assertEqual("".join(first_14), "ZITJLSOJSOZTLI")

    def test_metadata_and_mode_196611_decoding(self) -> None:
        meta = self.decoded.metadata
        self.assertEqual(meta.version, 3.3)
        self.assertEqual(meta.soft_drop_id, 4)
        self.assertEqual(meta.soft_drop_name, "Instant")
        self.assertEqual(meta.soft_drop_steps, 20)
        self.assertEqual(meta.game_start_ms, 1794817929644)
        self.assertEqual(meta.game_end_ms, 1794817977339)
        self.assertEqual(meta.duration_ms, 47695)
        self.assertEqual(meta.seed, "zglrnd")
        self.assertEqual(meta.raw_mode, 196611)
        self.assertEqual(meta.pmode, 3)
        self.assertEqual(meta.sub_mode, 3)
        self.assertEqual(meta.mode_description, "Cheese Race 100L")
        self.assertEqual(meta.target_lines, 100)
        self.assertEqual(meta.das_ms, 75)
        self.assertEqual(meta.ruleset_id, 0)
        self.assertEqual(meta.binary_data_bytes, 1844)

    def test_v3_bitstream_decoding_and_4094_epoch_wrap(self) -> None:
        events = self.decoded.events
        self.assertEqual(len(events), 864)
        self.assertTrue(self.decoded.terminator_found)

        for i in range(9):
            self.assertEqual(events[i].timestamp_ms, 0)
            self.assertEqual(events[i].action_code, ActionType.GARBAGE_ADD)
            self.assertEqual(events[i].data[0], 1)

        timestamps = [ev.timestamp_ms for ev in events]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(timestamps[-1], 47695)
        self.assertEqual(events[-1].epoch_wrap_count, 11)
        self.assertEqual(events[-1].raw_12bit_ts, 2661)

    def test_full_match_simulation_and_162_blocks_verification(self) -> None:
        sim = self.sim
        self.assertEqual(sim.anomalies, [])
        self.assertEqual(self.summary["verification"]["status"], "VERIFIED")
        self.assertEqual(sim.placed_blocks, 162)
        self.assertEqual(len(sim.placements), 162)
        self.assertEqual(len(sim.decision_states), 162)
        self.assertEqual(sim.lines_remaining, 0)

        gd = sim.gamedata
        self.assertEqual(gd["lines"], 154)
        self.assertEqual(gd["singles"], 71)
        self.assertEqual(gd["doubles"], 40)
        self.assertEqual(gd["triples"], 1)
        self.assertEqual(gd["tetrises"], 0)
        self.assertEqual(gd["garbageCleared"], 100)
        self.assertEqual(gd["linesReceived"], 100)
        self.assertEqual(gd["holds"], 62)
        self.assertEqual(gd["attack"], 106)
        self.assertEqual(gd["linesSent"], 106)
        self.assertEqual(gd["maxCombo"], 7)
        self.assertEqual(gd["TSD"], 2)
        self.assertEqual(gd["tspins"], 2)
        self.assertEqual(gd["tpieces"], 23)
        self.assertEqual(gd["wasted"], 21)
        self.assertEqual(gd["score"], 34004)
        self.assertEqual(sim.total_finesse, 203)
        self.assertEqual(sim.total_key_presses, 572)
        self.assertEqual(sim.get_pps(), 3.4)
        self.assertEqual(sim.get_kpp(), 4.53)

    def test_observable_vs_posthoc_separation(self) -> None:
        for state in self.sim.decision_states:
            obs = state.to_dict()["observable_state"]
            self.assertEqual(set(obs.keys()), ALLOWED_OBSERVABLE_STATE_KEYS)
            self.assertEqual(len(obs["visible_previews"]), 5)
            self.assertEqual(len(obs["board"]), 20)
            self.assertTrue(all(len(row) == 10 for row in obs["board"]))

        info_bound = self.summary["information_boundary"]
        self.assertEqual(info_bound["legally_observable_at_decision_time"]["visible_preview_count"], 5)
        self.assertEqual(len(info_bound["post_hoc_replay_reconstructed_only"]["full_garbage_hole_columns_100"]), 100)

    def test_exporter_writes_all_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_all_json(self.decoded, self.sim, tmpdir)
            for key in ("metadata", "events", "placements", "states", "summary"):
                self.assertIn(key, paths)
                self.assertTrue(paths[key].exists())
                parsed = json.loads(paths[key].read_text(encoding="utf-8"))
                self.assertTrue(len(parsed) > 0)


class TestJstrisReplaySample2Cktcp(unittest.TestCase):
    """Independent validation test suite for Sample 2 ('cktcp', 167 blocks, 48.838s)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.replay_code = SAMPLE_REPLAY_2_PATH.read_text(encoding="utf-8").strip()
        cls.decoded = decode_replay_code(cls.replay_code)
        cls.sim = JstrisSimulator(cls.decoded)
        cls.sim.run()
        cls.summary = build_summary(cls.decoded, cls.sim)

    def test_lzstring_and_metadata_decoding(self) -> None:
        meta = self.decoded.metadata
        self.assertEqual(meta.version, 3.3)
        self.assertEqual(meta.seed, "cktcp")
        self.assertEqual(meta.raw_mode, 196611)
        self.assertEqual(meta.pmode, 3)
        self.assertEqual(meta.sub_mode, 3)
        self.assertEqual(meta.mode_description, "Cheese Race 100L")
        self.assertEqual(meta.target_lines, 100)
        self.assertEqual(meta.soft_drop_id, 4)
        self.assertEqual(meta.soft_drop_name, "Instant")
        self.assertEqual(meta.soft_drop_steps, 20)
        self.assertEqual(meta.das_ms, 75)
        self.assertEqual(meta.ruleset_id, 0)
        self.assertEqual(meta.game_start_ms, 1793337992481)
        self.assertEqual(meta.game_end_ms, 1793338041319)
        self.assertEqual(meta.duration_ms, 48838)
        self.assertEqual(meta.duration_sec, 48.838)
        self.assertEqual(meta.binary_data_bytes, 1864)

    def test_v3_bitstream_events_and_timestamps(self) -> None:
        events = self.decoded.events
        self.assertEqual(len(events), 875)
        self.assertTrue(self.decoded.terminator_found)
        self.assertEqual(self.decoded.total_bits_read, 14912)
        self.assertEqual(self.decoded.total_bits_available, 14912)

        for i in range(9):
            self.assertEqual(events[i].timestamp_ms, 0)
            self.assertEqual(events[i].action_code, ActionType.GARBAGE_ADD)
            self.assertEqual(events[i].data[0], 1)

        timestamps = [ev.timestamp_ms for ev in events]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(timestamps[-1], 48838)
        self.assertEqual(events[-1].epoch_wrap_count, 11)
        self.assertEqual(events[-1].raw_12bit_ts, 3804)

    def test_full_match_simulation_and_verification(self) -> None:
        sim = self.sim
        ver = self.summary["verification"]
        self.assertEqual(sim.anomalies, [])
        self.assertEqual(ver["status"], "VERIFIED")
        self.assertTrue(ver["all_locks_legal"])
        self.assertTrue(ver["all_events_processed"])
        self.assertTrue(ver["timestamps_monotonic"])
        self.assertTrue(ver["target_lines_completed"])
        self.assertTrue(ver["mode_completion_verified"])
        self.assertTrue(ver["state_self_consistent"])
        self.assertTrue(ver["clock_matches_metadata_duration"])

        self.assertEqual(sim.placed_blocks, 167)
        self.assertEqual(len(sim.placements), 167)
        self.assertEqual(len(sim.decision_states), 167)
        self.assertEqual(sim.lines_remaining, 0)
        self.assertEqual(sim.count_garbage_height(), 0)

        gd = sim.gamedata
        self.assertEqual(gd["lines"], 156)
        self.assertEqual(gd["singles"], 82)
        self.assertEqual(gd["doubles"], 37)
        self.assertEqual(gd["triples"], 0)
        self.assertEqual(gd["tetrises"], 0)
        self.assertEqual(gd["garbageCleared"], 100)
        self.assertEqual(gd["linesReceived"], 100)
        self.assertEqual(gd["holds"], 63)
        self.assertEqual(gd["attack"], 104)
        self.assertEqual(gd["linesSent"], 104)
        self.assertEqual(gd["maxCombo"], 7)
        self.assertEqual(gd["TSD"], 0)
        self.assertEqual(gd["tspins"], 1)
        self.assertEqual(gd["tpieces"], 24)
        self.assertEqual(gd["wasted"], 23)
        self.assertEqual(gd["score"], 33834)
        self.assertEqual(sim.total_finesse, 204)
        self.assertEqual(sim.total_key_presses, 575)
        self.assertEqual(sim.get_pps(), 3.42)
        self.assertEqual(sim.get_kpp(), 4.44)

    def test_observable_state_audit_and_exporter(self) -> None:
        for state in self.sim.decision_states:
            obs = state.to_dict()["observable_state"]
            self.assertEqual(set(obs.keys()), ALLOWED_OBSERVABLE_STATE_KEYS)
            self.assertEqual(len(obs["visible_previews"]), 5)
            self.assertEqual(len(obs["board"]), 20)
            self.assertTrue(all(len(row) == 10 for row in obs["board"]))

        for p in self.sim.placements:
            p_dict = p.to_dict()
            self.assertIn("observable_at_lock", p_dict)
            self.assertNotIn("observable_at_oock", p_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_all_json(self.decoded, self.sim, tmpdir)
            for key in ("metadata", "events", "placements", "states", "summary"):
                self.assertIn(key, paths)
                self.assertTrue(paths[key].exists())
                parsed = json.loads(paths[key].read_text(encoding="utf-8"))
                self.assertTrue(len(parsed) > 0)


class TestIntegrityAndErrorDetection(unittest.TestCase):
    """Tests for format corruption, bitstream truncation, verification false-positive prevention, and bug fixes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_code = SAMPLE_REPLAY_2_PATH.read_text(encoding="utf-8").strip()
        cls.decoded = decode_replay_code(cls.sample_code)

    def test_get_occupied_cells_helper(self) -> None:
        # T piece (id=2), rot=0 at (3, 10)
        cells = get_occupied_cells(2, 0, 3, 10)
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells, [(11, 4), (12, 3), (12, 4), (12, 5)])

    def test_add_solid_garbage_shifts_board_and_increments_solid_height(self) -> None:
        sim = JstrisSimulator(copy.deepcopy(self.decoded))
        sim.matrix = [[0] * 10 for _ in range(20)]
        sim.matrix[0][0] = 5
        sim.matrix[19][3] = 2
        sim.add_solid_garbage()
        self.assertEqual(sim.solid_height, 1)
        self.assertEqual(sim.deadline[0], 5)
        self.assertEqual(sim.matrix[18][3], 2)
        self.assertEqual(sim.matrix[19], [9] * 10)

    def test_corrupted_format_inputs_rejected(self) -> None:
        with self.assertRaises(ValueError):
            decode_replay_code("   ")
        with self.assertRaises(ValueError):
            decode_replay_code("{" + '"foo": 123}')
        with self.assertRaises(ValueError):
            decode_replay_code(json.dumps({"c": {"v": 3.3}}))
        with self.assertRaises(NotImplementedError):
            decode_replay_code(json.dumps({"c": {"v": 2.0}, "d": ""}))
        # Base64 length not multiple of 4 bytes
        bad_b64 = base64.b64encode(b"\x01\x02\x03").decode("ascii")
        with self.assertRaises(ValueError):
            decode_replay_code(json.dumps({"c": {"v": 3.3}, "d": bad_b64}))

    def test_truncated_event_payload_raises_value_error(self) -> None:
        # Event 1: 16 bits (ts=10, MOVE_LEFT=0). Event 2 at bit 16: 12-bit ts (20) + 4-bit GARBAGE_ADD (11) = 32 bits,
        # leaving 0 bits for the required 5-bit lines and 4-bit hole_column payload!
        ev1 = (10 << 4) | int(ActionType.MOVE_LEFT)
        ev2_header = (20 << 4) | int(ActionType.GARBAGE_ADD)
        word = (ev1 << 16) | ev2_header
        raw_bytes = struct.pack(">I", word)
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        with self.assertRaises(ValueError):
            decode_replay_code(json.dumps({"c": {"v": 3.3}, "d": b64}))

    def test_missing_0xfff_terminator_fails_verification(self) -> None:
        # Single word with two 16-bit events (MOVE_LEFT at t=10, HARD_DROP at t=20) and NO 0xFFF terminator
        ev1 = (10 << 4) | int(ActionType.MOVE_LEFT)
        ev2 = (20 << 4) | int(ActionType.HARD_DROP)
        word = (ev1 << 16) | ev2
        b64 = base64.b64encode(struct.pack(">I", word)).decode("ascii")
        payload = json.dumps(
            {
                "c": {
                    "v": 3.3,
                    "gameStart": 1000,
                    "gameEnd": 1020,
                    "seed": "test",
                    "m": 2 << 16,  # Practice mode
                },
                "d": b64,
            }
        )
        decoded = decode_replay_code(payload)
        self.assertFalse(decoded.terminator_found)
        sim = JstrisSimulator(decoded)
        sim.run()
        summary = build_summary(decoded, sim)
        self.assertEqual(summary["verification"]["status"], "ANOMALIES_DETECTED")
        self.assertTrue(any("terminator" in a for a in sim.anomalies))

    def test_truncated_events_incomplete_mode_fails_verification(self) -> None:
        truncated_decoded = copy.deepcopy(self.decoded)
        # Keep only first 100 events so 100L Cheese Race cannot complete and duration mismatches
        truncated_decoded.events = truncated_decoded.events[:100]
        sim = JstrisSimulator(truncated_decoded)
        sim.run()
        summary = build_summary(truncated_decoded, sim)
        self.assertEqual(summary["verification"]["status"], "ANOMALIES_DETECTED")
        self.assertFalse(summary["verification"]["target_lines_completed"])
        self.assertFalse(summary["verification"]["mode_completion_verified"])
        self.assertFalse(summary["verification"]["clock_matches_metadata_duration"])
        self.assertGreater(len(sim.anomalies), 0)

    def test_clock_mismatch_fails_verification(self) -> None:
        mismatched_decoded = copy.deepcopy(self.decoded)
        mismatched_decoded.metadata.duration_ms += 500
        sim = JstrisSimulator(mismatched_decoded)
        sim.run()
        summary = build_summary(mismatched_decoded, sim)
        self.assertEqual(summary["verification"]["status"], "ANOMALIES_DETECTED")
        self.assertFalse(summary["verification"]["clock_matches_metadata_duration"])
        self.assertTrue(any("does not match metadata duration" in a for a in sim.anomalies))

    def test_unhandled_aux_event_fails_verification(self) -> None:
        aux_decoded = copy.deepcopy(self.decoded)
        last_ts = aux_decoded.events[-1].timestamp_ms
        aux_decoded.events.append(
            ReplayEvent(
                index=len(aux_decoded.events),
                timestamp_ms=last_ts,
                raw_12bit_ts=last_ts % 4094,
                epoch_wrap_count=last_ts // 4094,
                action_code=int(ActionType.AUX),
                action_name="AUX",
                data=[0, 1],
                aux_code=int(AuxType.BLOCK_SET),
                aux_name="BLOCK_SET",
            )
        )
        sim = JstrisSimulator(aux_decoded)
        sim.run()
        summary = build_summary(aux_decoded, sim)
        self.assertEqual(summary["verification"]["status"], "ANOMALIES_DETECTED")
        self.assertTrue(any("Unhandled AUX sub-action" in a for a in sim.anomalies))


if __name__ == "__main__":
    unittest.main()
