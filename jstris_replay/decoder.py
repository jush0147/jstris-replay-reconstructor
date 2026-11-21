"""Jstris Replay outer (LZ-string + JSON) and inner (Base64 V3 bitstream) decoder."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
import json
import struct
from typing import Any

from .lzstring import compress_to_encoded_uri_component, decompress_from_encoded_uri_component


class ActionType(IntEnum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    DAS_LEFT = 2
    DAS_RIGHT = 3
    ROTATE_LEFT = 4
    ROTATE_RIGHT = 5
    ROTATE_180 = 6
    HARD_DROP = 7
    SOFT_DROP_BEGIN_END = 8
    GRAVITY_STEP = 9
    HOLD_BLOCK = 10
    GARBAGE_ADD = 11
    SGARBAGE_ADD = 12
    REDBAR_SET = 13
    ARR_MOVE = 14
    AUX = 15


class AuxType(IntEnum):
    AFK = 0
    BLOCK_SET = 1
    MOVE_TO = 2
    RANDOMIZER = 3
    MATRIX_MOD = 4
    WIDE_GARBAGE_ADD = 5


PMODE_NAMES: dict[int, str] = {
    1: "Sprint",
    2: "Practice / Free Play",
    3: "Cheese Race",
    4: "Survival",
    5: "Ultra",
    6: "Custom Map",
    7: "20TSD",
    8: "PC Mode",
    9: "User Mode",
    100: "Multiplayer / Live",
}

SPRINT_SUBMODES: dict[int, int] = {
    1: 40,
    2: 20,
    3: 100,
    4: 1000,
}

CHEESE_SUBMODES: dict[int, int] = {
    1: 10,
    2: 18,
    3: 100,
    100: 1000000,  # Infinite cheese
}

RULESET_NAMES: dict[int, str] = {
    0: "Default",
    1: "Big mode",
    2: "Pentomino",
    3: "MPH (Memoryless, previewless, holdless)",
}

SOFT_DROP_SPEEDS: dict[int, dict[str, Any]] = {
    0: {"name": "Slow", "time": 0.05, "steps": 0},
    1: {"name": "Medium", "time": 0.008, "steps": 0},
    2: {"name": "Fast", "time": 0.0, "steps": 1},
    3: {"name": "Ultra", "time": 0.0, "steps": 2},
    4: {"name": "Instant", "time": 0.0, "steps": 20},
}


@dataclass
class ReplayMetadata:
    raw_config: dict[str, Any]
    version: float
    soft_drop_id: int
    soft_drop_name: str
    soft_drop_steps: int
    game_start_ms: int
    game_end_ms: int
    duration_ms: int
    duration_sec: float
    game_start_iso: str
    game_end_iso: str
    seed: str
    raw_mode: int
    pmode: int
    pmode_name: str
    sub_mode: int
    mode_description: str
    target_lines: int
    block_skin: int
    sound_effects: int
    das_ms: int
    arr_ms: int
    ruleset_id: int
    ruleset_name: str
    base_block_set: int
    randomizer_type: int
    show_previews: int
    hold_enabled: bool
    replay_code_length: int
    binary_data_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_config": self.raw_config,
            "version": self.version,
            "soft_drop_id": self.soft_drop_id,
            "soft_drop_name": self.soft_drop_name,
            "soft_drop_steps": self.soft_drop_steps,
            "game_start_ms": self.game_start_ms,
            "game_end_ms": self.game_end_ms,
            "duration_ms": self.duration_ms,
            "duration_sec": self.duration_sec,
            "game_start_iso": self.game_start_iso,
            "game_end_iso": self.game_end_iso,
            "seed": self.seed,
            "raw_mode": self.raw_mode,
            "pmode": self.pmode,
            "pmode_name": self.pmode_name,
            "sub_mode": self.sub_mode,
            "mode_description": self.mode_description,
            "target_lines": self.target_lines,
            "block_skin": self.block_skin,
            "sound_effects": self.sound_effects,
            "das_ms": self.das_ms,
            "arr_ms": self.arr_ms,
            "ruleset_id": self.ruleset_id,
            "ruleset_name": self.ruleset_name,
            "base_block_set": self.base_block_set,
            "randomizer_type": self.randomizer_type,
            "show_previews": self.show_previews,
            "hold_enabled": self.hold_enabled,
            "replay_code_length": self.replay_code_length,
            "binary_data_bytes": self.binary_data_bytes,
        }


@dataclass
class ReplayEvent:
    index: int
    timestamp_ms: int
    raw_12bit_ts: int
    epoch_wrap_count: int
    action_code: int
    action_name: str
    data: list[int] = field(default_factory=list)
    aux_code: int | None = None
    aux_name: str | None = None
    bit_offset: int = 0
    bit_length: int = 16
    hard_drop_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "index": self.index,
            "timestamp_ms": self.timestamp_ms,
            "raw_12bit_ts": self.raw_12bit_ts,
            "epoch_wrap_count": self.epoch_wrap_count,
            "action_code": self.action_code,
            "action_name": self.action_name,
            "data": self.data,
            "bit_offset": self.bit_offset,
            "bit_length": self.bit_length,
        }
        if self.aux_code is not None:
            out["aux_code"] = self.aux_code
            out["aux_name"] = self.aux_name
        if self.hard_drop_id is not None:
            out["hard_drop_id"] = self.hard_drop_id
        return out


@dataclass
class DecodedReplay:
    metadata: ReplayMetadata
    events: list[ReplayEvent]
    raw_json: dict[str, Any]
    terminator_found: bool
    total_bits_read: int
    total_bits_available: int


class _V3BitReader:
    """Exact port of Jstris Replayer._base64ToArrayBuffer + Replayer.prototype.pullBits."""

    def __init__(self, b64_data: str) -> None:
        raw_bytes = base64.b64decode(b64_data)
        if len(raw_bytes) % 4 != 0:
            raise ValueError(f"Base64 binary payload length ({len(raw_bytes)} bytes) is not a multiple of 4.")
        self.raw_bytes = raw_bytes
        self.words: list[int] = []
        # Jstris _base64ToArrayBuffer reverses every 4-byte chunk before viewing as Uint32Array (LE),
        # which is equivalent to unpacking each 4-byte chunk as a Big-Endian uint32!
        for i in range(0, len(raw_bytes), 4):
            (word,) = struct.unpack(">I", raw_bytes[i : i + 4])
            self.words.append(word)
        self.word_idx = 0
        self.bit_pos = 0
        self.total_bits_read = 0

    @property
    def current_bit_offset(self) -> int:
        return self.word_idx * 32 + self.bit_pos

    @property
    def total_bits(self) -> int:
        return len(self.words) * 32

    def pull_bits(self, n_bits: int) -> int | None:
        if self.word_idx == len(self.words):
            return None
        rem = 32 - (self.bit_pos + n_bits)
        if rem >= 0:
            mask = ((1 << n_bits) - 1) << rem
            val = (self.words[self.word_idx] & mask) >> rem
        else:
            high_bits = n_bits + rem
            mask_high = (1 << high_bits) - 1
            val_high = (self.words[self.word_idx] & mask_high) << (-rem)
            low_bits = -rem
            mask_low = ((1 << low_bits) - 1) << (32 + rem)
            if self.word_idx + 1 == len(self.words) and rem != 0:
                return None
            val = val_high | ((self.words[self.word_idx + 1] & mask_low) >> (32 + rem))

        self.bit_pos += n_bits
        if self.bit_pos >= 32:
            self.bit_pos -= 32
            self.word_idx += 1
        self.total_bits_read += n_bits
        return val


def parse_metadata(raw_c: dict[str, Any], coy_code: int, binary_bytes: int) -> ReplayMetadata:
    """Parse Jstris replay metadata ('c' object) including composite mode 'm'."""
    version = float(raw_c.get("v", 3.3))
    soft_drop_id = int(raw_c.get("softDropId", 2))
    sd_info = SOFT_DROP_SPEEDS.get(soft_drop_id, {"name": f"Unknown({soft_drop_id})", "steps": 0})

    game_start = int(raw_c.get("gameStart", 0))
    game_end = int(raw_c.get("gameEnd", 0))
    duration_ms = game_end - game_start

    start_iso = datetime.fromtimestamp(game_start / 1000.0, tz=timezone.utc).isoformat() if game_start else ""
    end_iso = datetime.fromtimestamp(game_end / 1000.0, tz=timezone.utc).isoformat() if game_end else ""

    seed = str(raw_c.get("seed", ""))
    raw_m = int(raw_c.get("m", 0))
    pmode = (raw_m & 0xFFFF0000) >> 16
    sub_mode = raw_m & 0xFFFF
    if pmode == 0:
        pmode = 1

    pmode_name = PMODE_NAMES.get(pmode, f"Mode({pmode})")
    target_lines = 1000000
    if pmode == 1:
        target_lines = SPRINT_SUBMODES.get(sub_mode, 40)
        mode_desc = f"Sprint {target_lines}L"
    elif pmode == 3:
        target_lines = CHEESE_SUBMODES.get(sub_mode, 100)
        sub_label = f"{target_lines}L" if sub_mode != 100 else "Infinite"
        mode_desc = f"Cheese Race {sub_label}"
    elif pmode == 4:
        mode_desc = "Survival"
    elif pmode == 5:
        mode_desc = "Ultra (2min)"
    elif pmode == 6:
        mode_desc = f"Custom Map ({raw_c.get('map', sub_mode)})"
    elif pmode == 7:
        mode_desc = "20TSD"
    elif pmode == 8:
        mode_desc = "PC Mode"
    else:
        mode_desc = f"{pmode_name} (subMode={sub_mode})"

    ruleset_id = int(raw_c.get("r", 0))
    ruleset_name = RULESET_NAMES.get(ruleset_id, f"Ruleset({ruleset_id})")

    base_block_set = int(raw_c.get("bbs", 0))
    if ruleset_id == 1 and "bbs" not in raw_c:
        base_block_set = 1
    elif ruleset_id == 2 and "bbs" not in raw_c:
        base_block_set = 4

    rnd_type = int(raw_c.get("rnd", 0))
    show_previews = 5
    hold_enabled = True
    if ruleset_id == 3:
        rnd_type = 2
        show_previews = 0
        hold_enabled = False

    return ReplayMetadata(
        raw_config=dict(raw_c),
        version=version,
        soft_drop_id=soft_drop_id,
        soft_drop_name=sd_info["name"],
        soft_drop_steps=sd_info["steps"],
        game_start_ms=game_start,
        game_end_ms=game_end,
        duration_ms=duration_ms,
        duration_sec=round(duration_ms / 1000.0, 3),
        game_start_iso=start_iso,
        game_end_iso=end_iso,
        seed=seed,
        raw_mode=raw_m,
        pmode=pmode,
        pmode_name=pmode_name,
        sub_mode=sub_mode,
        mode_description=mode_desc,
        target_lines=target_lines,
        block_skin=int(raw_c.get("bs", 0)),
        sound_effects=int(raw_c.get("se", 0)),
        das_ms=int(raw_c.get("das", 0)),
        arr_ms=int(raw_c.get("arr", 0)),
        ruleset_id=ruleset_id,
        ruleset_name=ruleset_name,
        base_block_set=base_block_set,
        randomizer_type=rnd_type,
        show_previews=show_previews,
        hold_enabled=hold_enabled,
        replay_code_length=coy_code,
        binary_data_bytes=binary_bytes,
    )


def decode_binary_actions_v3(b64_data: str) -> tuple[list[ReplayEvent], bool, int, int]:
    """Exact port of Jstris Replayer.prototype.loadBinaryActionsV3."""
    reader = _V3BitReader(b64_data)
    events: list[ReplayEvent] = []
    prev_ts = 0
    wrap_count = 0
    terminator_found = False
    hd_counter = 0

    while True:
        start_bit = reader.current_bit_offset
        raw_ts = reader.pull_bits(12)
        if raw_ts is None:
            break
        if raw_ts == 4095:
            terminator_found = True
            break

        if raw_ts < prev_ts:
            wrap_count += 1
        prev_ts = raw_ts
        timestamp_ms = 4094 * wrap_count + raw_ts

        act_code = reader.pull_bits(4)
        if act_code is None:
            raise ValueError(f"Unexpected end of bitstream after timestamp at bit {start_bit}")

        act_name = ActionType(act_code).name
        data: list[int] = []
        aux_code: int | None = None
        aux_name: str | None = None

        if act_code == ActionType.GARBAGE_ADD:
            lines = reader.pull_bits(5)
            col = reader.pull_bits(4)
            if lines is None or col is None:
                raise ValueError("Truncated GARBAGE_ADD payload")
            data = [lines, col]
        elif act_code == ActionType.REDBAR_SET:
            val = reader.pull_bits(5)
            if val is None:
                raise ValueError("Truncated REDBAR_SET payload")
            data = [val]
        elif act_code == ActionType.ARR_MOVE:
            direction = reader.pull_bits(1)
            if direction is None:
                raise ValueError("Truncated ARR_MOVE payload")
            data = [direction]
        elif act_code == ActionType.AUX:
            aux_code = reader.pull_bits(4)
            if aux_code is None:
                raise ValueError("Truncated AUX sub-action code")
            aux_name = AuxType(aux_code).name if aux_code in AuxType._value2member_map_ else f"AUX_{aux_code}"
            if aux_code == AuxType.AFK:
                afk_dur = reader.pull_bits(16)
                if afk_dur is None:
                    raise ValueError("Truncated AUX.AFK payload")
                data = [afk_dur]
                prev_ts += afk_dur % 4094
                if prev_ts >= 4094:
                    prev_ts -= 4094
                    wrap_count += 1
                wrap_count += afk_dur // 4094
            elif aux_code == AuxType.BLOCK_SET:
                b0 = reader.pull_bits(1)
                b1 = reader.pull_bits(4)
                if b0 is None or b1 is None:
                    raise ValueError("Truncated AUX.BLOCK_SET payload")
                data = [b0, b1]
            elif aux_code == AuxType.MOVE_TO:
                x_raw = reader.pull_bits(4)
                y_raw = reader.pull_bits(5)
                if x_raw is None or y_raw is None:
                    raise ValueError("Truncated AUX.MOVE_TO payload")
                data = [x_raw - 3, y_raw - 12]
            elif aux_code == AuxType.RANDOMIZER:
                r0 = reader.pull_bits(1)
                r1 = reader.pull_bits(5)
                if r0 is None or r1 is None:
                    raise ValueError("Truncated AUX.RANDOMIZER payload")
                data = [r0, r1]
            elif aux_code == AuxType.MATRIX_MOD:
                m0 = reader.pull_bits(4)
                m1 = reader.pull_bits(5)
                if m0 is None or m1 is None:
                    raise ValueError("Truncated AUX.MATRIX_MOD payload")
                data = [m0, m1]
            elif aux_code == AuxType.WIDE_GARBAGE_ADD:
                w0 = reader.pull_bits(5)
                w1 = reader.pull_bits(4)
                w2 = reader.pull_bits(3)
                w3 = reader.pull_bits(1)
                if None in (w0, w1, w2, w3):
                    raise ValueError("Truncated AUX.WIDE_GARBAGE_ADD payload")
                data = [w0, w1, w2, w3]  # type: ignore[list-item]

        end_bit = reader.current_bit_offset
        hd_id: int | None = None
        if act_code == ActionType.HARD_DROP:
            hd_counter += 1
            hd_id = hd_counter

        events.append(
            ReplayEvent(
                index=len(events),
                timestamp_ms=timestamp_ms,
                raw_12bit_ts=raw_ts,
                epoch_wrap_count=wrap_count,
                action_code=act_code,
                action_name=act_name,
                data=data,
                aux_code=aux_code,
                aux_name=aux_name,
                bit_offset=start_bit,
                bit_length=end_bit - start_bit,
                hard_drop_id=hd_id,
            )
        )

    return events, terminator_found, reader.total_bits_read, reader.total_bits


def decode_replay_code(replay_input: str) -> DecodedReplay:
    """Decode a Jstris Replay Code (either URI-safe LZ-String or raw JSON string)."""
    cleaned = replay_input.strip()
    if not cleaned:
        raise ValueError("Empty replay input.")

    if cleaned.startswith("{"):
        raw_json_str = cleaned
    else:
        decompressed = decompress_from_encoded_uri_component(cleaned)
        if not decompressed:
            raise ValueError("Failed to decompress LZ-String replay code.")
        raw_json_str = decompressed

    parsed = json.loads(raw_json_str)
    if isinstance(parsed, str):
        decompressed_inner = decompress_from_encoded_uri_component(parsed)
        if not decompressed_inner:
            raise ValueError("Failed to decompress double-encoded LZ-String replay code.")
        parsed = json.loads(decompressed_inner)

    if not isinstance(parsed, dict) or "c" not in parsed:
        raise ValueError("Decoded JSON does not contain Jstris metadata key 'c'.")

    raw_c = parsed["c"]
    version = float(raw_c.get("v", 3.3))

    if version >= 3.0:
        if "d" not in parsed or not isinstance(parsed["d"], str):
            raise ValueError("V3+ replay missing Base64 string field 'd'.")
        b64_data = parsed["d"]
        raw_bytes = base64.b64decode(b64_data)
        metadata = parse_metadata(raw_c, len(cleaned), len(raw_bytes))
        events, terminator_found, bits_read, total_bits = decode_binary_actions_v3(b64_data)
    else:
        raise NotImplementedError(f"Replay version {version} (< 3.0) is not supported.")

    return DecodedReplay(
        metadata=metadata,
        events=events,
        raw_json=parsed,
        terminator_found=terminator_found,
        total_bits_read=bits_read,
        total_bits_available=total_bits,
    )
