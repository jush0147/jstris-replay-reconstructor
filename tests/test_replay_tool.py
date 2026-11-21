"""Comprehensive automated test suite for Jstris Replay decoding and full match reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jstris_replay import (
    ActionType,
    AleaPRNG,
    BagRandomizer,
    JstrisSimulator,
    PIECE_NAMES,
    build_summary,
    compress_to_encoded_uri_component,
    decode_replay_code,
    decompress_from_encoded_uri_component,
    export_all_json,
)

SAMPLE_REPLAY_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_100l_cheese_replay.txt"


class TestJstrisReplayTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.replay_code = SAMPLE_REPLAY_PATH.read_text(encoding="utf-8").strip()
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
        # Verify 7-bag property on first two bags
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
        # 196611 = (3 << 16) | 3 => pmode=3 (Cheese Race), subMode=3 (100L)
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

        # First 9 events are initial Cheese Race garbage rows at t=0
        for i in range(9):
            self.assertEqual(events[i].timestamp_ms, 0)
            self.assertEqual(events[i].action_code, ActionType.GARBAGE_ADD)
            self.assertEqual(events[i].data[0], 1)

        # Monotonically non-decreasing timestamps ending at 47695 ms (exact match with gameEnd - gameStart)
        timestamps = [ev.timestamp_ms for ev in events]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(timestamps[-1], 47695)
        # Verify epoch wrap count at end: 47695 // 4094 = 11 wraps (47695 = 11 * 4094 + 2661)
        self.assertEqual(events[-1].epoch_wrap_count, 11)
        self.assertEqual(events[-1].raw_12bit_ts, 2661)

    def test_full_match_simulationpand_162_blocks_verification(self) -> None:
        sim = self.sim
        self.assertEqual(sim.anomalies, [])
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
            # Must only expose 5 visible previews, never full future piece sequence
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


if __name__ == "__main__":
    unittest.main()
