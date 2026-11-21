"""Exact port of Jstris GameCore + Replayer game rules, physics, scoring, and state reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .decoder import ActionType, AuxType, DecodedReplay, ReplayEvent, ReplayMetadata
from .rng import PIECE_COLORS, PIECE_NAMES, AleaPRNG, create_randomizer

# Cell color IDs in Jstris 20x10 matrix:
# 0: Empty ('.')
# 1: Z (Red)
# 2: L (Orange)
# 3: O (Yellow)
# 4: S (Green)
# 5: I (Cyan)
# 6: J (Blue)
# 7: T (Purple)
# 8: Garbage ('G')
# 9: Solid Garbage ('#')
CELL_ID_TO_CHAR: dict[int, str] = {
    0: ".",
    1: "Z",
    2: "L",
    3: "O",
    4: "S",
    5: "I",
    6: "J",
    7: "T",
    8: "G",
    9: "#",
}

# Standard SRS + Jstris 180 kicks for J, L, S, T, Z (kicks table _0xb75bx3 in Jstris)
JLSTZ_KICKS: dict[int, dict[int, list[tuple[int, int]]]] = {
    0: {
        -1: [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
        1: [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        2: [(0, 0), (0, 1)],
    },
    1: {
        -1: [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        1: [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        2: [(0, 0), (1, 0)],
    },
    2: {
        -1: [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        1: [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
        2: [(0, 0), (0, -1)],
    },
    3: {
        -1: [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        1: [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        2: [(0, 0), (-1, 0)],
    },
}

# Standard SRS + Jstris 180 kicks for I piece (kicks table _0xb75bx4 in Jstris)
I_KICKS: dict[int, dict[int, list[tuple[int, int]]]] = {
    0: {
        -1: [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
        1: [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
        2: [(0, 0), (0, 1)],
    },
    1: {
        -1: [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
        1: [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
        2: [(0, 0), (1, 0)],
    },
    2: {
        -1: [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
        1: [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
        2: [(0, 0), (0, -1)],
    },
    3: {
        -1: [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
        1: [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
        2: [(0, 0), (-1, 0)],
    },
}

# O piece kicks (_0xb75bx5 in Jstris)
O_KICKS: dict[int, dict[int, list[tuple[int, int]]]] = {
    rot: {-1: [(0, 0)], 1: [(0, 0)], 2: [(0, 0)]} for rot in range(4)
}

# Jstris finesse lookup table: FINESSE_TABLE[piece_id][rot][col_index]
FINESSE_TABLE: list[list[list[int]]] = [
    [
        [1, 2, 1, 0, 1, 2, 1],
        [2, 2, 2, 2, 1, 1, 2, 2, 2, 2],
        [1, 2, 1, 0, 1, 2, 1],
        [2, 2, 2, 2, 1, 1, 2, 2, 2, 2],
    ],
    [
        [1, 2, 2, 1, 0, 1, 2, 2, 1],
        [1, 2, 2, 1, 0, 1, 2, 2, 1],
        [1, 2, 2, 1, 0, 1, 2, 2, 1],
        [1, 2, 2, 1, 0, 1, 2, 2, 1],
    ],
    [
        [1, 2, 1, 0, 1, 2, 2, 1],
        [2, 2, 3, 2, 1, 2, 3, 3, 2],
        [3, 4, 3, 2, 3, 4, 4, 3],
        [2, 3, 2, 1, 2, 3, 3, 2, 2],
    ],
    [
        [1, 2, 1, 0, 1, 2, 2, 1],
        [2, 2, 3, 2, 1, 2, 3, 3, 2],
        [3, 4, 3, 2, 3, 4, 4, 3],
        [2, 3, 2, 1, 2, 3, 3, 2, 2],
    ],
    [
        [1, 2, 1, 0, 1, 2, 2, 1],
        [2, 2, 3, 2, 1, 2, 3, 3, 2],
        [3, 4, 3, 2, 3, 4, 4, 3],
        [2, 3, 2, 1, 2, 3, 3, 2, 2],
    ],
    [
        [1, 2, 1, 0, 1, 2, 2, 1],
        [2, 2, 2, 1, 1, 2, 3, 2, 2],
        [1, 2, 1, 0, 1, 2, 2, 1],
        [2, 2, 2, 1, 1, 2, 3, 2, 2],
    ],
    [
        [1, 2, 1, 0, 1, 2, 2, 1],
        [2, 2, 2, 1, 1, 2, 3, 2, 2],
        [1, 2, 1, 0, 1, 2, 2, 1],
        [2, 2, 2, 1, 1, 2, 3, 2, 2],
    ],
]

# Piece definitions matching Jstris BlockSet 0
PIECE_DEFS: dict[int, dict[str, Any]] = {
    0: {
        "id": 0,
        "name": "I",
        "color": 5,
        "spawn": (3, -1),
        "cc": [0, 2, 0, 1],
        "kicks": I_KICKS,
        "blocks": [
            [[0, 0, 0, 0], [1, 2, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [4, 3, 2, 1], [0, 0, 0, 0]],
            [[0, 4, 0, 0], [0, 3, 0, 0], [0, 2, 0, 0], [0, 1, 0, 0]],
        ],
    },
    1: {
        "id": 1,
        "name": "O",
        "color": 3,
        "spawn": (3, -2),
        "cc": [1, 1, 1, 1],
        "kicks": O_KICKS,
        "blocks": [
            [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 3, 1, 0], [0, 4, 2, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 4, 3, 0], [0, 2, 1, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 2, 4, 0], [0, 1, 3, 0], [0, 0, 0, 0]],
        ],
    },
    2: {
        "id": 2,
        "name": "T",
        "color": 7,
        "spawn": (3, -2),
        "cc": [0, 1, 0, 0],
        "kicks": JLSTZ_KICKS,
        "blocks": [
            [[0, 0, 0, 0], [0, 1, 0, 0], [2, 3, 4, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 2, 0, 0], [0, 3, 1, 0], [0, 4, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [4, 3, 2, 0], [0, 1, 0, 0]],
            [[0, 0, 0, 0], [0, 4, 0, 0], [1, 3, 0, 0], [0, 2, 0, 0]],
        ],
    },
    3: {
        "id": 3,
        "name": "L",
        "color": 2,
        "spawn": (3, -2),
        "cc": [0, 1, 0, 0],
        "kicks": JLSTZ_KICKS,
        "blocks": [
            [[0, 0, 0, 0], [0, 0, 1, 0], [2, 3, 4, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0], [0, 4, 1, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [4, 3, 2, 0], [1, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 4, 0, 0], [0, 3, 0, 0], [0, 2, 0, 0]],
        ],
    },
    4: {
        "id": 4,
        "name": "J",
        "color": 6,
        "spawn": (3, -2),
        "cc": [0, 1, 0, 0],
        "kicks": JLSTZ_KICKS,
        "blocks": [
            [[0, 0, 0, 0], [1, 0, 0, 0], [2, 3, 4, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 2, 1, 0], [0, 3, 0, 0], [0, 4, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [4, 3, 2, 0], [0, 0, 1, 0]],
            [[0, 0, 0, 0], [0, 4, 0, 0], [0, 3, 0, 0], [1, 2, 0, 0]],
        ],
    },
    5: {
        "id": 5,
        "name": "S",
        "color": 4,
        "spawn": (3, -2),
        "cc": [0, 1, 0, 0],
        "kicks": JLSTZ_KICKS,
        "blocks": [
            [[0, 0, 0, 0], [0, 1, 2, 0], [3, 4, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 3, 0, 0], [0, 4, 1, 0], [0, 0, 2, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 4, 3, 0], [2, 1, 0, 0]],
            [[0, 0, 0, 0], [2, 0, 0, 0], [1, 4, 0, 0], [0, 3, 0, 0]],
        ],
    },
    6: {
        "id": 6,
        "name": "Z",
        "color": 1,
        "spawn": (3, -2),
        "cc": [0, 1, 0, 0],
        "kicks": JLSTZ_KICKS,
        "blocks": [
            [[0, 0, 0, 0], [1, 2, 0, 0], [0, 3, 4, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 3, 2, 0], [0, 4, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [4, 3, 0, 0], [0, 2, 1, 0]],
            [[0, 0, 0, 0], [0, 4, 0, 0], [2, 3, 0, 0], [1, 0, 0, 0]],
        ],
    },
}

LINES_ATTACK_DEFAULT = [0, 0, 1, 2, 4, 4, 6, 2, 0, 10, 1]
COMBO_ATTACK_DEFAULT = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5]

SCORE_VALUES: dict[str, int] = {
    "SOFT_DROP": 1,
    "HARD_DROP": 2,
    "CLEAR1": 100,
    "CLEAR2": 300,
    "CLEAR3": 500,
    "CLEAR4": 800,
    "TSPIN_MINI": 100,
    "TSPIN": 400,
    "TSPIN_MINI_SINGLE": 200,
    "TSPIN_SINGLE": 800,
    "TSPIN_DOUBLE": 1200,
    "TSPIN_TRIPLE": 1600,
    "PERFECT_CLEAR": 3000,
    "COMBO": 50,
    "CLEAR5": 1600,
}

SCORE_B2B_BONUS: dict[str, int] = {
    "CLEAR4": 400,
    "CLEAR5": 800,
    "TSPIN_MINI_SINGLE": 100,
    "TSPIN_SINGLE": 400,
    "TSPIN_DOUBLE": 600,
    "TSPIN_TRIPLE": 800,
}


def matrix_to_ascii_rows(matrix: list[list[int]]) -> list[str]:
    return ["".join(CELL_ID_TO_CHAR.get(cell & 0x0F, "?") for cell in row) for row in matrix]


def clone_matrix(matrix: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in matrix]


def get_occupied_cells(piece_id: int, rot: int, x: int, y: int) -> list[tuple[int, int]]:
    """Return list of (row, col) coordinates occupied by piece at (x, y, rot)."""
    grid = PIECE_DEFS[piece_id]["blocks"][rot]
    cells: list[tuple[int, int]] = []
    for r in range(4):
        for c in range(4):
            if grid[r][c] > 0:
                cells.append((y + r, x + c))
    return cells


@dataclass
class PlacementRecord:
    """Complete record of a single piece lock (placement), separating legal decision state from post-hoc replay state."""

    placement_index: int  # 1-based index (1..162)
    # Timing & event indices
    turn_start_event_index: int
    lock_event_index: int
    turn_end_event_index: int  # includes any immediate post-lock GARBAGE_ADD events at the same or before next move
    turn_start_time_ms: int
    lock_time_ms: int
    turn_duration_ms: int
    # Bot-observable decision state at the start of the turn (before any move/hold on this turn)
    observable_at_turn_start: dict[str, Any]
    # Bot-observable state right before HARD_DROP (after optional hold and movement)
    observable_at_lock: dict[str, Any]
    # Placement details
    piece: str
    piece_id: int
    piece_color: int
    hold_used: bool
    hold_before_turn: str | None
    hold_after_lock: str | None
    visible_previews_at_turn_start: list[str]
    visible_previews_at_lock: list[str]
    spawn_x: int
    spawn_y: int
    spawn_rot: int
    lock_x: int
    lock_y: int
    y_before_hard_drop: int
    hard_drop_distance: int
    lock_rot: int
    occupied_cells: list[tuple[int, int]]
    # Board states
    board_before_lock: list[list[int]]
    board_before_lock_ascii: list[str]
    deadline_before_lock: list[int]
    board_after_lock_before_clear: list[list[int]]
    board_after_clear_before_garbage: list[list[int]]
    board_after_clear_before_garbage_ascii: list[str]
    board_after_turn_garbage: list[list[int]]
    board_after_turn_garbage_ascii: list[str]
    # Line clear & Cheese Race progress
    lines_cleared: int
    cleared_row_indices: list[int]
    clear_type: str
    is_tspin: bool
    is_tspin_mini: bool
    is_perfect_clear: bool
    b2b_before: bool
    b2b_after: bool
    combo_before: int
    combo_after: int
    attack_sent: int
    combo_attack_sent: int
    total_attack_sent_on_turn: int
    cumulative_lines_cleared: int
    cumulative_attack_sent: int
    # Cheese / Garbage tracking
    garbage_lines_cleared_on_turn: int
    cumulative_garbage_lines_cleared: int
    garbage_height_before_lock: int
    garbage_height_after_clear: int
    garbage_added_after_placement: list[dict[str, int]]
    garbage_height_after_turn: int
    lines_remaining_before: int
    lines_remaining_after: int
    # Finesse & input actions during this turn
    finesse_used: int
    optimal_finesse: int
    finesse_faults_on_turn: int
    cumulative_finesse_faults: int
    score_before: int
    score_after: int
    turn_actions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "placement_index": self.placement_index,
            "turn_start_event_index": self.turn_start_event_index,
            "lock_event_index": self.lock_event_index,
            "turn_end_event_index": self.turn_end_event_index,
            "turn_start_time_ms": self.turn_start_time_ms,
            "lock_time_ms": self.lock_time_ms,
            "turn_duration_ms": self.turn_duration_ms,
            "observable_at_turn_start": self.observable_at_turn_start,
            "observable_at_lock": self.observable_at_lock,
            "piece": self.piece,
            "piece_id": self.piece_id,
            "piece_color": self.piece_color,
            "hold_used": self.hold_used,
            "hold_before_turn": self.hold_before_turn,
            "hold_after_lock": self.hold_after_lock,
            "visible_previews_at_turn_start": self.visible_previews_at_turn_start,
            "visible_previews_at_lock": self.visible_previews_at_lock,
            "spawn_position": {"x": self.spawn_x, "y": self.spawn_y, "rot": self.spawn_rot},
            "lock_position": {
                "x": self.lock_x,
                "y": self.lock_y,
                "rot": self.lock_rot,
                "y_before_hard_drop": self.y_before_hard_drop,
                "hard_drop_distance": self.hard_drop_distance,
                "occupied_cells_row_col": [[r, c] for r, c in self.occupied_cells],
            },
            "board_before_lock": self.board_before_lock,
            "board_before_lock_ascii": self.board_before_lock_ascii,
            "deadline_before_lock": self.deadline_before_lock,
            "board_after_lock_before_clear": self.board_after_lock_before_clear,
            "board_after_clear_before_garbage": self.board_after_clear_before_garbage,
            "board_after_clear_before_garbage_ascii": self.board_after_clear_before_garbage_ascii,
            "board_after_turn_garbage": self.board_after_turn_garbage,
            "board_after_turn_garbage_ascii": self.board_after_turn_garbage_ascii,
            "lines_cleared": self.lines_cleared,
            "cleared_row_indices": self.cleared_row_indices,
            "clear_type": self.clear_type,
            "is_tspin": self.is_tspin,
            "is_tspin_mini": self.is_tspin_mini,
            "is_perfect_clear": self.is_perfect_clear,
            "b2b_before": self.b2b_before,
            "b2b_after": self.b2b_after,
            "combo_before": self.combo_before,
            "combo_after": self.combo_after,
            "attack_sent": self.attack_sent,
            "combo_attack_sent": self.combo_attack_sent,
            "total_attack_sent_on_turn": self.total_attack_sent_on_turn,
            "cumulative_lines_cleared": self.cumulative_lines_cleared,
            "cumulative_attack_sent": self.cumulative_attack_sent,
            "garbage_lines_cleared_on_turn": self.garbage_lines_cleared_on_turn,
            "cumulative_garbage_lines_cleared": self.cumulative_garbage_lines_cleared,
            "garbage_height_before_lock": self.garbage_height_before_lock,
            "garbage_height_after_clear": self.garbage_height_after_clear,
            "garbage_added_after_placement": self.garbage_added_after_placement,
            "garbage_height_after_turn": self.garbage_height_after_turn,
            "lines_remaining_before": self.lines_remaining_before,
            "lines_remaining_after": self.lines_remaining_after,
            "finesse_used": self.finesse_used,
            "optimal_finesse": self.optimal_finesse,
            "finesse_faults_on_turn": self.finesse_faults_on_turn,
            "cumulative_finesse_faults": self.cumulative_finesse_faults,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "turn_actions": self.turn_actions,
        }


@dataclass
class DecisionStateSnapshot:
    """Clean snapshot of the legally observable game state at each decision point (for AI research)."""

    step_index: int  # 1..162
    timestamp_ms: int
    event_index: int
    active_piece: str
    hold_piece: str | None
    can_hold: bool
    visible_previews: list[str]
    board: list[list[int]]
    board_ascii: list[str]
    deadline: list[int]
    garbage_height: int
    lines_remaining: int
    cumulative_lines_cleared: int
    cumulative_garbage_cleared: int
    combo: int
    back_to_back: bool
    chosen_action: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "timestamp_ms": self.timestamp_ms,
            "event_index": self.event_index,
            "observable_state": {
                "active_piece": self.active_piece,
                "hold_piece": self.hold_piece,
                "can_hold": self.can_hold,
                "visible_previews": self.visible_previews,
                "board": self.board,
                "board_ascii": self.board_ascii,
                "garbage_height": self.garbage_height,
                "lines_remaining": self.lines_remaining,
                "cumulative_lines_cleared": self.cumulative_lines_cleared,
                "cumulative_garbage_cleared": self.cumulative_garbage_cleared,
                "combo": self.combo,
                "back_to_back": self.back_to_back,
            },
            "internal_state": {
                "deadline": self.deadline,
            },
            "chosen_action": self.chosen_action,
        }


class JstrisSimulator:
    """Exact simulation of Jstris Replayer / GameCore."""

    def __init__(self, decoded: DecodedReplay) -> None:
        self.decoded = decoded
        self.meta: ReplayMetadata = decoded.metadata
        self.events: list[ReplayEvent] = decoded.events

        # GameCore state
        self.matrix: list[list[int]] = [[0] * 10 for _ in range(20)]
        self.deadline: list[int] = [0] * 10
        self.queue_length: int = 5
        self.show_previews: int = self.meta.show_previews
        self.hold_enabled: bool = self.meta.hold_enabled
        self.queue: list[int] = []
        self.generated_pieces_history: list[int] = []

        self.active_piece_id: int = 0
        self.active_x: int = 3
        self.active_y: int = 0
        self.active_rot: int = 0
        self.ghost_x: int = 3
        self.ghost_y: int = 0

        self.hold_piece_id: int | None = None
        self.hold_used_already: bool = False

        self.spin_possible: bool = False
        self.spin_mini_possible: bool = False
        self.is_back2back: bool = False
        self.was_back2back: bool = False
        self.combo_counter: int = -1

        self.soft_drop: bool = False
        self.soft_drop_id: int = self.meta.soft_drop_id
        self.soft_drop_steps: int = self.meta.soft_drop_steps
        self.game_step: float = 0.9
        self.timer: float = 0.0
        self.clock: int = 0
        self.red_bar: int = 0
        self.solid_height: int = 0
        self.reached_end: bool = False

        self.placed_blocks: int = 0
        self.finesse: int = 0
        self.used_180: int = 0
        self.total_finesse: int = 0
        self.total_key_presses: int = 0
        self.fin_faults: list[int] = []
        self.score_stamps: list[list[int]] = []

        self.lines_remaining: int = self.meta.target_lines
        self.time_remaining: int = 120

        self.gamedata: dict[str, int] = {
            "lines": 0,
            "singles": 0,
            "doubles": 0,
            "triples": 0,
            "tetrises": 0,
            "maxCombo": 0,
            "linesSent": 0,
            "linesReceived": 0,
            "PCs": 0,
            "lastPC": 0,
            "TSD": 0,
            "TSD20": 0,
            "B2B": 0,
            "attack": 0,
            "score": 0,
            "holds": 0,
            "garbageCleared": 0,
            "wasted": 0,
            "tpieces": 0,
            "tspins": 0,
        }

        self.ptr: int = 0
        self.placements: list[PlacementRecord] = []
        self.decision_states: list[DecisionStateSnapshot] = []
        self.initial_garbage_events: list[dict[str, Any]] = []
        self.all_garbage_events: list[dict[str, Any]] = []
        self.anomalies: list[str] = []

        # Tracking per-turn information
        self._turn_start_ptr: int = 0
        self._turn_start_time: int = 0
        self._turn_start_board: list[list[int]] = []
        self._turn_start_deadline: list[int] = []
        self._turn_start_active_id: int = 0
        self._turn_start_hold_id: int | None = None
        self._turn_start_previews: list[str] = []
        self._turn_start_garbage_height: int = 0
        self._turn_start_lines_remaining: int = 0
        self._turn_hold_used: bool = False
        self._turn_spawn_pos: tuple[int, int, int] = (3, -2, 0)
        self._turn_actions: list[dict[str, Any]] = []

    def _draw_from_randomizer(self) -> int:
        piece_id = self.randomizer.get_block_id()
        self.generated_pieces_history.append(piece_id)
        return piece_id

    def _generate_initial_queue(self) -> None:
        self.queue.clear()
        for _ in range(self.queue_length):
            self.queue.append(self._draw_from_randomizer())

        # Jstris GameCore.prototype.generateQueue start-piece adjustment in Replayer (!this.ISGAME)
        if self.meta.pmode == 1:
            if self.queue[0] >= 5 and self.meta.base_block_set == 0:
                tmp = self.queue[0]
                if self.queue[1] < 5:
                    self.queue[0] = self.queue[1]
                    self.queue[1] = tmp
                else:
                    self.queue[0] = self.queue[2]
                    self.queue[2] = tmp
        else:
            if (
                self.meta.version < 3.3
                and self.meta.pmode != 2
                and len(self.queue) >= 3
                and self.queue[0] >= 5
                and self.queue[1] >= 5
            ):
                self.queue[0], self.queue[2] = self.queue[2], self.queue[0]

    def _get_block_from_queue(self) -> int:
        if not self.queue:
            self.reached_end = True
            return self.active_piece_id
        piece_id = self.queue.pop(0)
        self.queue.append(self._draw_from_randomizer())
        return piece_id

    def check_intersection(self, x: int, y: int, rot: int | None = None) -> bool:
        if rot is None:
            rot = self.active_rot
        grid = PIECE_DEFS[self.active_piece_id]["blocks"][rot]
        for r in range(4):
            for c in range(4):
                if grid[r][c] > 0:
                    if y + r >= 20:
                        return True
                    if x + c < 0 or x + c >= 10:
                        return True
                    if y + r >= 0 and self.matrix[y + r][x + c] > 0:
                        return True
        return False

    def set_current_piece_to_default_pos(self) -> None:
        pdef = PIECE_DEFS[self.active_piece_id]
        self.active_rot = 0
        self.active_x, self.active_y = pdef["spawn"]
        if self.meta.base_block_set == 0:
            row_in_grid = pdef["blocks"][0][-self.active_y]
            if (
                (self.matrix[0][3] and row_in_grid[0])
                or (self.matrix[0][4] and row_in_grid[1])
                or (self.matrix[0][5] and row_in_grid[2])
                or (self.matrix[0][6] and row_in_grid[3])
            ):
                self.active_y -= 1
        else:
            while self.check_intersection(self.active_x, self.active_y, None):
                self.active_y -= 1

        deadline_row = -(1 + self.active_y)
        if 0 <= deadline_row < 4:
            for c in range(4):
                if pdef["blocks"][self.active_rot][deadline_row][c] and self.deadline[self.active_x + c]:
                    self.reached_end = True
                    break

    def update_ghost_piece(self) -> None:
        for test_y in range(self.active_y, 21):
            if self.check_intersection(self.active_x, test_y, None):
                self.ghost_x = self.active_x
                self.ghost_y = test_y - 1
                break

    def get_next_block(self) -> None:
        self.active_piece_id = self._get_block_from_queue()
        self.set_current_piece_to_default_pos()
        self.update_ghost_piece()
        self.timer = 0.0

    def move_current_block(self, dx: int) -> bool:
        if not self.check_intersection(self.active_x + dx, self.active_y, self.active_rot):
            self.spin_possible = False
            self.active_x += dx
            self.update_ghost_piece()
            return True
        return False

    def move_block_to_the_wall(self, dx: int) -> int:
        steps = 0
        while not self.check_intersection(self.active_x + dx, self.active_y, None):
            self.active_x += dx
            steps += 1
        if steps > 0:
            self.update_ghost_piece()
        return steps

    def rotate_current_block(self, dr: int) -> bool:
        if dr == 2:
            self.finesse += 2
            self.used_180 += 1
        else:
            self.finesse += 1

        new_rot = 3 if (self.active_rot + dr) == -1 else (self.active_rot + dr) % 4
        kicks = PIECE_DEFS[self.active_piece_id]["kicks"][self.active_rot][dr]
        rotated = False
        for kx, ky in kicks:
            if not self.check_intersection(self.active_x + kx, self.active_y - ky, new_rot):
                self.spin_possible = True
                self.active_rot = new_rot
                self.active_x += kx
                self.active_y -= ky
                if ky > 0:
                    self.timer = 0.0
                rotated = True
                break
        self.update_ghost_piece()
        return rotated

    def gravity_step(self, extra_steps: int) -> int:
        if self.check_intersection(self.active_x, self.active_y + 1, None):
            return 0
        self.active_y += 1
        self.spin_possible = False
        if extra_steps > 0:
            return 1 + self.gravity_step(extra_steps - 1)
        return 1

    def hold_block(self) -> None:
        if not self.hold_used_already and self.hold_enabled:
            if self.hold_piece_id is None:
                self.hold_piece_id = self.active_piece_id
                self.get_next_block()
            else:
                self.hold_piece_id, self.active_piece_id = self.active_piece_id, self.hold_piece_id
                self.set_current_piece_to_default_pos()
            self.update_ghost_piece()
            self.hold_used_already = True
            self.total_key_presses += 1
            self.gamedata["holds"] += 1
            self._turn_hold_used = True
            self._turn_spawn_pos = (self.active_x, self.active_y, self.active_rot)

    def add_garbage(self, lines: int, hole_col: int, gap_w: int = 1, g_inv: int = 0) -> None:
        if gap_w <= 0:
            gap_w = 1
        if lines > 0:
            self.gamedata["linesReceived"] += lines
            if lines > 20:
                lines = 20
            row = [8] * 10
            row[hole_col] = 0
            if gap_w > 1:
                if hole_col + gap_w > 10:
                    hole_col = 10 - gap_w
                for i in range(1, gap_w):
                    row[hole_col + i] = 0
            if g_inv:
                row = [0 if cell == 8 else 8 for cell in row]
            self.deadline = list(self.matrix[lines - 1])
            limit = len(self.matrix) - self.solid_height
            for r in range(limit):
                if limit - r > lines:
                    self.matrix[r] = list(self.matrix[r + lines])
                else:
                    self.matrix[r] = list(row)

    def garbage_add_adv(self, lines: int, hole_col: int, gap_w: int = 1, g_inv: int = 0) -> None:
        if self.red_bar > 0:
            self.red_bar -= min(lines, self.red_bar)
        self.ghost_y -= lines
        self.add_garbage(lines, hole_col, gap_w, g_inv)
        if self.ptr + 1 == len(self.events) or (
            self.ptr + 1 < len(self.events) and self.events[self.ptr + 1].action_code != ActionType.GARBAGE_ADD
        ):
            while self.check_intersection(self.active_x, self.active_y, self.active_rot):
                self.active_y -= 1
                if self.active_y < -4:
                    break
            self.update_ghost_piece()

    def add_solid_garbage(self) -> None:
        if self.solid_height == 20:
            return
        solid_row = [9] * 10
        self.deadline = list(self.matrix[0])
        for r in range(len(self.matrix)):
            if len(self.matrix) - r > 1:
                self.matrix[r] = list(self.matrix[r + 1])
            else:
                self.matrix[r] = list(solid_row)
        self.solid_height += 1

    def count_garbage_height(self, max_h: int = 20) -> int:
        height = 0
        for i in range(max_h):
            if self.matrix[19 - i][0] != 8 and self.matrix[19 - i][1] != 8:
                height = i
                break
        return height

    def check_tspin(self) -> None:
        front = 0
        back = 0
        rot = self.active_rot
        x = self.active_x
        y = self.active_y
        if y < -2:
            return
        if rot == 0:
            if y >= -1:
                front = int(self.matrix[y + 1][x] > 0) + int(self.matrix[y + 1][x + 2] > 0)
            elif y == -2:
                front = int(self.deadline[x] > 0) + int(self.deadline[x + 2] > 0)
            back = 2 if y == 17 else (int(self.matrix[y + 3][x] > 0) + int(self.matrix[y + 3][x + 2] > 0))
        elif rot == 1:
            if x == -1:
                back = 2
            if y >= -1:
                front = int(self.matrix[y + 1][x + 2] > 0) + int(self.matrix[y + 3][x + 2] > 0)
                if not back:
                    back = int(self.matrix[y + 1][x] > 0) + int(self.matrix[y + 3][x] > 0)
            elif y == -2:
                front = int(self.deadline[x + 2] > 0) + int(self.matrix[y + 3][x + 2] > 0)
                if not back:
                    back = int(self.deadline[x] > 0) + int(self.matrix[y + 3][x] > 0)
        elif rot == 2:
            if y >= -1:
                back = int(self.matrix[y + 1][x] > 0) + int(self.matrix[y + 1][x + 2] > 0)
            elif y == -2:
                back = int(self.deadline[x] > 0) + int(self.deadline[x + 2] > 0)
            front = 2 if y == 17 else (int(self.matrix[y + 3][x] > 0) + int(self.matrix[y + 3][x + 2] > 0))
        elif rot == 3:
            if x == 8:
                back = 2
            if y >= -1:
                front = int(self.matrix[y + 1][x] > 0) + int(self.matrix[y + 3][x] > 0)
                if not back:
                    back = int(self.matrix[y + 1][x + 2] > 0) + int(self.matrix[y + 3][x + 2] > 0)
            elif y == -2:
                front = int(self.deadline[x] > 0) + int(self.matrix[y + 3][x] > 0)
                if not back:
                    back = int(self.deadline[x + 2] > 0) + int(self.matrix[y + 3][x + 2] > 0)

        self.spin_possible = front == 2 and back >= 1
        self.spin_mini_possible = front == 1 and back == 2

    def _add_score(self, score_type: str, multiplier: int = 1) -> None:
        base = SCORE_VALUES[score_type]
        if self.was_back2back and score_type in SCORE_B2B_BONUS:
            base += SCORE_B2B_BONUS[score_type]
        pts = multiplier * base
        self.gamedata["score"] += pts
        if score_type not in ("SOFT_DROP", "HARD_DROP"):
            t_stamp = self.events[self.ptr].timestamp_ms if self.ptr < len(self.events) else self.clock
            # Map score_type to Scoring.A integer lay_
            lay__map = {
                "SOFT_DROP": 0,
                "HARD_DROP": 1,
                "CLEAR1": 2,
                "CLEAR2": 3,
                "CLEAR3": 4,
                "CLEAR4": 5,
                "TSPIN_MINI": 6,
                "TSPIN": 7,
                "TSPIN_MINI_SINGLE": 8,
                "TSPIN_SINGLE": 9,
                "TSPIN_DOUBLE": 10,
                "TSPIN_TRIPLE": 11,
                "PERFECT_CLEAR": 12,
                "COMBO": 13,
                "CLEAR5": 14,
            }
            self.score_stamps.append([t_stamp, lay__map[score_type], pts, multiplier, 0])

    def _get_combo_attack(self, combo: int) -> int:
        if combo < 0:
            return 0
        return COMBO_ATTACK_DEFAULT[combo] if combo <= 12 else COMBO_ATTACK_DEFAULT[-1]

    def _capture_turn_start(self, start_ptr: int, start_time: int) -> None:
        self._turn_start_ptr = start_ptr
        self._turn_start_time = start_time
        self._turn_start_board = clone_matrix(self.matrix)
        self._turn_start_deadline = list(self.deadline)
        self._turn_start_active_id = self.active_piece_id
        self._turn_start_hold_id = self.hold_piece_id
        self._turn_start_previews = [PIECE_NAMES[pid] for pid in self.queue[: self.show_previews]]
        self._turn_start_garbage_height = self.count_garbage_height()
        self._turn_start_lines_remaining = self.lines_remaining
        self._turn_hold_used = False
        self._turn_spawn_pos = (self.active_x, self.active_y, self.active_rot)
        self._turn_actions = []

    def _execute_hard_drop(self, ev: ReplayEvent) -> None:
        pdef = PIECE_DEFS[self.active_piece_id]
        col_cc = self.active_x + pdef["cc"][self.active_rot]
        opt_finesse = FINESSE_TABLE[self.active_piece_id][self.active_rot][col_cc]
        fin_diff = self.finesse - opt_finesse
        if fin_diff > 0:
            self.total_finesse += fin_diff
            if ev.hard_drop_id is not None:
                self.fin_faults.append(ev.hard_drop_id)

        finesse_used_on_turn = self.finesse
        self.total_key_presses += self.finesse - self.used_180
        self.finesse = 0
        self.used_180 = 0

        self.update_ghost_piece()
        lock_x = self.ghost_x
        lock_y = self.ghost_y
        y_before_drop = self.active_y
        hd_dist = lock_y - y_before_drop

        score_before = self.gamedata["score"]
        b2b_before = self.is_back2back
        combo_before = self.combo_counter
        lines_rem_before = self.lines_remaining
        garbage_h_before = self.count_garbage_height()
        board_before_lock = clone_matrix(self.matrix)
        deadline_before_lock = list(self.deadline)
        previews_at_lock = [PIECE_NAMES[pid] for pid in self.queue[: self.show_previews]]
        hold_before_turn_str = PIECE_NAMES[self._turn_start_hold_id] if self._turn_start_hold_id is not None else None

        # Validate lock legality
        if self.check_intersection(lock_x, lock_y, self.active_rot):
            self.anomalies.append(f"Placement {self.placed_blocks + 1}: Illegal intersection at lock ({lock_x}, {lock_y})")
        if not self.check_intersection(lock_x, lock_y + 1, self.active_rot):
            self.anomalies.append(f"Placement {self.placed_blocks + 1}: Piece not supported from below at ({lock_x}, {lock_y})")

        self._add_score("HARD_DROP", hd_dist)

        # Place block on matrix
        self.placed_blocks += 1
        if self.active_piece_id == 2:
            self.gamedata["wasted"] += 1
            self.gamedata["tpieces"] += 1

        grid = pdef["blocks"][self.active_rot]
        above_top_cells = 0
        total_cells = 0
        occupied_coords: list[tuple[int, int]] = []
        for r in range(4):
            for c in range(4):
                if grid[r][c] > 0:
                    total_cells += 1
                    target_r = lock_y + r
                    target_c = lock_x + c
                    occupied_coords.append((target_r, target_c))
                    if target_r >= 0 and target_c >= 0:
                        self.matrix[target_r][target_c] = pdef["color"]
                    else:
                        above_top_cells += 1
                        if target_r == -1 and self.deadline[target_c] == 0:
                            self.deadline[target_c] = pdef["color"]

        if above_top_cells == total_cells:
            self.reached_end = True

        board_after_lock_before_clear = clone_matrix(self.matrix)

        if self.spin_possible and lock_y != y_before_drop:
            # Note: in Jstris Replayer.prototype.hardDrop, this check happens AFTER placeBlock,
            # whereas during placeBlock -> checkLineClears, spinPossible is still True if lock_y == y_before_drop.
            # Wait: let's match Jstris exact order! In Jstris Replayer.prototype.hardDrop:
            # this.placeBlock(this.ghostPiece.pos.x, _0xb75bx5), this.spinPossible && _0xb75bx5 !== this.activeBlock.pos.y && (this.spinPossible = !1)
            pass

        # checkLineClears
        self.hold_used_already = False
        self.was_back2back = self.is_back2back
        self.spin_mini_possible = False
        garbage_height_pre_clear = self.count_garbage_height() if self.meta.pmode == 3 else 0

        if self.spin_possible:
            if self.active_piece_id == 2:
                self.check_tspin()
            else:
                self.spin_possible = False

        is_tspin = self.spin_possible
        is_tspin_mini = self.spin_mini_possible

        lines_cleared = 0
        remaining_blocks_count = 0
        cleared_rows: list[int] = []

        # Check deadline row clear
        dl_filled = 0
        for c in range(10):
            if self.deadline[c] != 0:
                dl_filled += 1
            elif dl_filled > 0:
                break
        if dl_filled == 10:
            self.deadline = [0] * 10
            lines_cleared += 1
            cleared_rows.append(-1)
        else:
            remaining_blocks_count += dl_filled

        garbage_cleared_on_turn = 0
        for r in range(20):
            row_filled = 0
            for c in range(10):
                val = self.matrix[r][c]
                if val == 9:
                    break
                if val != 0:
                    row_filled += 1
                elif remaining_blocks_count + row_filled > 0:
                    break
            if row_filled == 10:
                cleared_rows.append(r)
                if 8 in self.matrix[r]:
                    self.gamedata["garbageCleared"] += 1
                    garbage_cleared_on_turn += 1
                for shift_r in range(r, 0, -1):
                    self.matrix[shift_r] = self.matrix[shift_r - 1]
                self.matrix[0] = list(self.deadline)
                self.deadline = [0] * 10
                row_filled = 0
                lines_cleared += 1
            remaining_blocks_count += row_filled

        clear_type = "NONE"
        line_atk = 0
        combo_atk = 0
        is_pc = False

        if lines_cleared > 0:
            self.gamedata["lines"] += lines_cleared
            if lines_cleared == 1:
                self.gamedata["singles"] += 1
                line_atk = LINES_ATTACK_DEFAULT[1]
                if self.spin_possible:
                    clear_type = "TSPIN_SINGLE"
                    line_atk = LINES_ATTACK_DEFAULT[7]
                    if self.is_back2back:
                        self.gamedata["B2B"] += 1
                        line_atk += LINES_ATTACK_DEFAULT[10]
                    else:
                        self.is_back2back = True
                elif self.spin_mini_possible:
                    clear_type = "TSPIN_MINI_SINGLE"
                    if self.is_back2back:
                        self.gamedata["B2B"] += 1
                    else:
                        self.is_back2back = True
                    line_atk = LINES_ATTACK_DEFAULT[8]
                else:
                    self.is_back2back = False
                    clear_type = "CLEAR1"
            elif lines_cleared == 2:
                self.gamedata["doubles"] += 1
                line_atk = LINES_ATTACK_DEFAULT[2]
                if self.spin_possible or self.spin_mini_possible:
                    self.gamedata["TSD"] += 1
                    line_atk = LINES_ATTACK_DEFAULT[5]
                    clear_type = "TSPIN_DOUBLE"
                    if self.is_back2back:
                        self.gamedata["B2B"] += 1
                        line_atk += LINES_ATTACK_DEFAULT[10]
                    else:
                        self.is_back2back = True
                else:
                    self.is_back2back = False
                    clear_type = "CLEAR2"
            elif lines_cleared == 3:
                self.gamedata["triples"] += 1
                line_atk = LINES_ATTACK_DEFAULT[3]
                if (self.spin_possible or self.spin_mini_possible) and self.active_piece_id == 2:
                    line_atk = LINES_ATTACK_DEFAULT[6]
                    clear_type = "TSPIN_TRIPLE"
                    if self.is_back2back:
                        self.gamedata["B2B"] += 1
                        line_atk += LINES_ATTACK_DEFAULT[10]
                    else:
                        self.is_back2back = True
                else:
                    self.is_back2back = False
                    clear_type = "CLEAR3"
            elif lines_cleared == 4:
                self.gamedata["tetrises"] += 1
                clear_type = "CLEAR4"
                line_atk = LINES_ATTACK_DEFAULT[4]
                if self.is_back2back:
                    self.gamedata["B2B"] += 1
                    line_atk += LINES_ATTACK_DEFAULT[10]
                else:
                    self.is_back2back = True
            else:
                self.gamedata["tetrises"] += 1
                clear_type = "CLEAR5"
                line_atk = LINES_ATTACK_DEFAULT[6]
                if self.is_back2back:
                    self.gamedata["B2B"] += 1
                    line_atk += LINES_ATTACK_DEFAULT[10]
                else:
                    self.is_back2back = True

            if clear_type in ("TSPIN_MINI_SINGLE", "TSPIN_SINGLE", "TSPIN_DOUBLE", "TSPIN_TRIPLE"):
                self.gamedata["wasted"] -= 1
                self.gamedata["tspins"] += 1

            self._add_score(clear_type)

            if remaining_blocks_count == 0:
                is_pc = True
                self.gamedata["PCs"] += 1
                line_atk = LINES_ATTACK_DEFAULT[9]
                self._add_score("PERFECT_CLEAR")

            self.combo_counter += 1
            if self.combo_counter > 0:
                self._add_score("COMBO", self.combo_counter)
            if self.combo_counter > self.gamedata["maxCombo"]:
                self.gamedata["maxCombo"] = self.combo_counter

            combo_atk = self._get_combo_attack(self.combo_counter)
            self.gamedata["linesSent"] += line_atk + combo_atk

            if self.meta.pmode != 0:
                self.gamedata["attack"] = self.gamedata["linesSent"]
                if self.meta.pmode == 1:
                    self.lines_remaining = max(0, self.lines_remaining - lines_cleared)
                elif self.meta.pmode == 3:
                    new_garbage_h = self.count_garbage_height()
                    self.lines_remaining -= garbage_height_pre_clear - new_garbage_h
        else:
            self.combo_counter = -1
            if self.spin_possible:
                clear_type = "TSPIN"
                self._add_score("TSPIN")
            elif self.spin_mini_possible:
                clear_type = "TSPIN_MINI"
                self._add_score("TSPIN_MINI")

        board_after_clear_before_garbage = clone_matrix(self.matrix)
        garbage_h_after_clear = self.count_garbage_height()

        locked_piece_id = self.active_piece_id
        locked_piece_name = PIECE_NAMES[locked_piece_id]
        locked_rot = self.active_rot
        hold_after_lock_str = PIECE_NAMES[self.hold_piece_id] if self.hold_piece_id is not None else None

        # Spawn next piece
        self.get_next_block()
        if self.spin_possible and lock_y != y_before_drop:
            self.spin_possible = False

        rec = PlacementRecord(
            placement_index=self.placed_blocks,
            turn_start_event_index=self._turn_start_ptr,
            lock_event_index=self.ptr,
            turn_end_event_index=self.ptr,
            turn_start_time_ms=self._turn_start_time,
            lock_time_ms=ev.timestamp_ms,
            turn_duration_ms=ev.timestamp_ms - self._turn_start_time,
            observable_at_turn_start={
                "active_piece": PIECE_NAMES[self._turn_start_active_id],
                "hold_piece": hold_before_turn_str,
                "can_hold": self.hold_enabled,
                "visible_previews": list(self._turn_start_previews),
                "board_ascii": matrix_to_ascii_rows(self._turn_start_board),
                "garbage_height": self._turn_start_garbage_height,
                "lines_remaining": self._turn_start_lines_remaining,
            },
            observable_at_lock={
                "active_piece": locked_piece_name,
                "hold_piece": hold_after_lock_str,
                "can_hold": self.hold_enabled and not self._turn_hold_used,
                "visible_previews": list(previews_at_lock),
                "board_ascii": matrix_to_ascii_rows(board_before_lock),
                "garbage_height": garbage_h_before,
                "lines_remaining": lines_rem_before,
            },
            piece=locked_piece_name,
            piece_id=locked_piece_id,
            piece_color=PIECE_COLORS[locked_piece_id],
            hold_used=self._turn_hold_used,
            hold_before_turn=hold_before_turn_str,
            hold_after_lock=hold_after_lock_str,
            visible_previews_at_turn_start=list(self._turn_start_previews),
            visible_previews_at_lock=list(previews_at_lock),
            spawn_x=self._turn_spawn_pos[0],
            spawn_y=self._turn_spawn_pos[1],
            spawn_rot=self._turn_spawn_pos[2],
            lock_x=lock_x,
            lock_y=lock_y,
            y_before_hard_drop=y_before_drop,
            hard_drop_distance=hd_dist,
            lock_rot=locked_rot,
            occupied_cells=occupied_coords,
            board_before_lock=board_before_lock,
            board_before_lock_ascii=matrix_to_ascii_rows(board_before_lock),
            deadline_before_lock=deadline_before_lock,
            board_after_lock_before_clear=board_after_lock_before_clear,
            board_after_clear_before_garbage=board_after_clear_before_garbage,
            board_after_clear_before_garbage_ascii=matrix_to_ascii_rows(board_after_clear_before_garbage),
            board_after_turn_garbage=clone_matrix(board_after_clear_before_garbage),
            board_after_turn_garbage_ascii=matrix_to_ascii_rows(board_after_clear_before_garbage),
            lines_cleared=lines_cleared,
            cleared_row_indices=cleared_rows,
            clear_type=clear_type,
            is_tspin=is_tspin,
            is_tspin_mini=is_tspin_mini,
            is_perfect_clear=is_pc,
            b2b_before=b2b_before,
            b2b_after=self.is_back2back,
            combo_before=combo_before,
            combo_after=self.combo_counter,
            attack_sent=line_atk,
            combo_attack_sent=combo_atk,
            total_attack_sent_on_turn=line_atk + combo_atk,
            cumulative_lines_cleared=self.gamedata["lines"],
            cumulative_attack_sent=self.gamedata["linesSent"],
            garbage_lines_cleared_on_turn=garbage_cleared_on_turn,
            cumulative_garbage_lines_cleared=self.gamedata["garbageCleared"],
            garbage_height_before_lock=garbage_h_before,
            garbage_height_after_clear=garbage_h_after_clear,
            garbage_added_after_placement=[],
            garbage_height_after_turn=garbage_h_after_clear,
            lines_remaining_before=lines_rem_before,
            lines_remaining_after=self.lines_remaining,
            finesse_used=finesse_used_on_turn,
            optimal_finesse=opt_finesse,
            finesse_faults_on_turn=max(0, fin_diff),
            cumulative_finesse_faults=self.total_finesse,
            score_before=score_before,
            score_after=self.gamedata["score"],
            turn_actions=list(self._turn_actions),
        )
        self.placements.append(rec)

        self.decision_states.append(
            DecisionStateSnapshot(
                step_index=rec.placement_index,
                timestamp_ms=rec.turn_start_time_ms,
                event_index=rec.turn_start_event_index,
                active_piece=PIECE_NAMES[self._turn_start_active_id],
                hold_piece=hold_before_turn_str,
                can_hold=self.hold_enabled,
                visible_previews=list(self._turn_start_previews),
                board=clone_matrix(self._turn_start_board),
                board_ascii=matrix_to_ascii_rows(self._turn_start_board),
                deadline=list(self._turn_start_deadline),
                garbage_height=self._turn_start_garbage_height,
                lines_remaining=self._turn_start_lines_remaining,
                cumulative_lines_cleared=self.gamedata["lines"] - lines_cleared,
                cumulative_garbage_cleared=self.gamedata["garbageCleared"] - garbage_cleared_on_turn,
                combo=combo_before,
                back_to_back=b2b_before,
                chosen_action={
                    "hold_used": rec.hold_used,
                    "placed_piece": rec.piece,
                    "lock_x": rec.lock_x,
                    "lock_y": rec.lock_y,
                    "lock_rot": rec.lock_rot,
                    "lines_cleared": rec.lines_cleared,
                    "garbage_lines_cleared": rec.garbage_lines_cleared_on_turn,
                    "clear_type": rec.clear_type,
                },
            )
        )

    def process_action(self, ev: ReplayEvent) -> None:
        code = ev.action_code
        self._turn_actions.append(
            {
                "event_index": ev.index,
                "timestamp_ms": ev.timestamp_ms,
                "action": ev.action_name,
                "data": ev.data,
            }
        )

        if code == ActionType.MOVE_LEFT:
            self.finesse += 1
            self.move_current_block(-1)
        elif code == ActionType.MOVE_RIGHT:
            self.finesse += 1
            self.move_current_block(1)
        elif code == ActionType.DAS_LEFT:
            self.move_block_to_the_wall(-1)
        elif code == ActionType.DAS_RIGHT:
            self.move_block_to_the_wall(1)
        elif code == ActionType.ROTATE_LEFT:
            self.rotate_current_block(-1)
        elif code == ActionType.ROTATE_RIGHT:
            self.rotate_current_block(1)
        elif code == ActionType.ROTATE_180:
            self.rotate_current_block(2)
        elif code == ActionType.HARD_DROP:
            self._execute_hard_drop(ev)
        elif code == ActionType.SOFT_DROP_BEGIN_END:
            if self.meta.version >= 3.0:
                self.soft_drop = not self.soft_drop
            else:
                self.soft_drop = True
        elif code == ActionType.GRAVITY_STEP:
            if self.meta.version >= 3.0:
                steps = self.gravity_step(self.soft_drop_steps if self.soft_drop else 0)
                if steps > 0 and self.soft_drop:
                    self._add_score("SOFT_DROP", steps)
            else:
                self.soft_drop = False
                self.game_step = 0.9
                self.timer = 0.0
        elif code == ActionType.HOLD_BLOCK:
            self.hold_block()
        elif code == ActionType.GARBAGE_ADD:
            lines, col = ev.data[0], ev.data[1]
            self.garbage_add_adv(lines, col)
            g_rec = {
                "event_index": ev.index,
                "timestamp_ms": ev.timestamp_ms,
                "lines": lines,
                "hole_column": col,
                "after_placement_index": self.placed_blocks,
            }
            self.all_garbage_events.append(g_rec)
            if self.placed_blocks == 0:
                self.initial_garbage_events.append(g_rec)
            elif self.placements:
                last_p = self.placements[-1]
                last_p.garbage_added_after_placement.append({"lines": lines, "hole_column": col, "event_index": ev.index})
                last_p.turn_end_event_index = ev.index
                last_p.board_after_turn_garbage = clone_matrix(self.matrix)
                last_p.board_after_turn_garbage_ascii = matrix_to_ascii_rows(self.matrix)
                last_p.garbage_height_after_turn = self.count_garbage_height()
        elif code == ActionType.SGARBAGE_ADD:
            self.ghost_y -= 1
            self.add_solid_garbage()
            while self.check_intersection(self.active_x, self.active_y, self.active_rot):
                self.active_y -= 1
                if self.active_y < -4:
                    break
            self.update_ghost_piece()
        elif code == ActionType.REDBAR_SET:
            self.red_bar = ev.data[0]
        elif code == ActionType.ARR_MOVE:
            self.move_current_block(-1 if ev.data[0] == 0 else 1)
        elif code == ActionType.AUX:
            if ev.aux_code == AuxType.MOVE_TO:
                self.ghost_x = self.active_x = ev.data[0]
                self.ghost_y = self.active_y = ev.data[1]
                self.update_ghost_piece()
            elif ev.aux_code == AuxType.WIDE_GARBAGE_ADD:
                lines, col, gap_w, g_inv = ev.data[0], ev.data[1], ev.data[2], ev.data[3]
                self.garbage_add_adv(lines, col, gap_w, g_inv)
            elif ev.aux_code == AuxType.AFK:
                pass
            else:
                self.anomalies.append(
                    f"Event {ev.index} (t={ev.timestamp_ms}ms): Unhandled AUX sub-action '{ev.aux_name}' (code={ev.aux_code})"
                )

    def _run_integrity_verification(self) -> None:
        """Perform comprehensive post-simulation integrity, bitstream, timing, and mode-completion verification."""
        if not self.decoded.terminator_found:
            self.anomalies.append(
                f"Bitstream missing 0xFFF (4095) terminator (read {self.decoded.total_bits_read}/{self.decoded.total_bits_available} bits)."
            )

        if self.ptr != len(self.events):
            self.anomalies.append(
                f"Incomplete event processing: processed {self.ptr} of {len(self.events)} events."
            )

        for i in range(1, len(self.events)):
            if self.events[i].timestamp_ms < self.events[i - 1].timestamp_ms:
                self.anomalies.append(
                    f"Non-monotonic event timestamp at index {i}: {self.events[i].timestamp_ms}ms < {self.events[i - 1].timestamp_ms}ms"
                )

        if self.meta.duration_ms > 0 and self.clock != self.meta.duration_ms:
            self.anomalies.append(
                f"Final event clock ({self.clock}ms) does not match metadata duration ({self.meta.duration_ms}ms)."
            )

        if len(self.placements) != self.placed_blocks:
            self.anomalies.append(
                f"Placement record count ({len(self.placements)}) does not match placed_blocks ({self.placed_blocks})."
            )

        if len(self.decision_states) != self.placed_blocks:
            self.anomalies.append(
                f"Decision state count ({len(self.decision_states)}) does not match placed_blocks ({self.placed_blocks})."
            )

        if self._turn_actions:
            self.anomalies.append(
                f"Found {len(self._turn_actions)} trailing uncommitted action(s) after the final placement."
            )

        # Mode-specific completion & invariant verification
        if self.meta.pmode in (1, 3) and self.meta.sub_mode != 100:
            if self.lines_remaining != 0:
                self.anomalies.append(
                    f"Mode '{self.meta.mode_description}' target not completed: lines_remaining={self.lines_remaining} (expected 0)."
                )
        if self.meta.pmode == 3 and self.meta.sub_mode != 100:
            if self.gamedata["garbageCleared"] != self.meta.target_lines:
                self.anomalies.append(
                    f"Cheese Race garbageCleared ({self.gamedata['garbageCleared']}) != target_lines ({self.meta.target_lines})."
                )
            final_garbage_h = self.count_garbage_height()
            if final_garbage_h != 0:
                self.anomalies.append(
                    f"Cheese Race finished with non-zero remaining garbage height ({final_garbage_h})."
                )

    def run(self) -> None:
        """Execute full Jstris Replayer lifecycle: initReplay() -> restart() -> playUntilTime()."""
        # 1. initReplay
        self.block_rng = AleaPRNG(self.meta.seed)
        self.randomizer = create_randomizer(self.meta.randomizer_type, self.block_rng, 7)
        self._generate_initial_queue()

        # 2. restart
        self.matrix = [[0] * 10 for _ in range(20)]
        self.deadline = [0] * 10
        self.hold_piece_id = None
        self.combo_counter = -1
        self.hold_used_already = False
        self.is_back2back = False
        self.soft_drop = False
        self.timer = 0.0
        self.clock = 0
        self.red_bar = 0
        self.solid_height = 0
        self.placed_blocks = 0
        self.total_finesse = 0
        self.total_key_presses = 0
        self.finesse = 0
        self.used_180 = 0
        self.ptr = 0

        self.get_next_block()

        # Process all t=0 actions (e.g. initial 9 garbage rows in Cheese Race)
        while self.ptr < len(self.events) and self.events[self.ptr].timestamp_ms == 0:
            self.process_action(self.events[self.ptr])
            self.ptr += 1

        # Capture turn 1 start state (after t=0 setup events)
        self._capture_turn_start(self.ptr, 0)

        # 3. Main event loop
        while not self.reached_end and self.ptr < len(self.events):
            ev = self.events[self.ptr]
            if self.ptr > 0:
                self.timer += (ev.timestamp_ms - self.events[self.ptr - 1].timestamp_ms) / 1000.0
            self.clock = ev.timestamp_ms
            self.process_action(ev)
            self.ptr += 1

            # When a HARD_DROP occurs, any immediately following GARBAGE_ADD events at the same timestamp
            # belong to the post-placement Cheese Race garbage refill before the next turn's decision state.
            if ev.action_code == ActionType.HARD_DROP:
                while self.ptr < len(self.events) and self.events[self.ptr].action_code == ActionType.GARBAGE_ADD:
                    g_ev = self.events[self.ptr]
                    self.timer += (g_ev.timestamp_ms - self.events[self.ptr - 1].timestamp_ms) / 1000.0
                    self.clock = g_ev.timestamp_ms
                    self.process_action(g_ev)
                    self.ptr += 1
                self._turn_actions = []
                if self.ptr < len(self.events):
                    self._capture_turn_start(self.ptr, self.clock)

        if self.ptr == len(self.events):
            self.reached_end = True
            if self.events:
                self.clock = self.events[-1].timestamp_ms

        self._run_integrity_verification()

    def get_pps(self) -> float:
        if self.clock <= 0:
            return 0.0
        return round(100.0 * self.placed_blocks / (self.clock / 1000.0)) / 100.0

    def get_kpp(self) -> float:
        if self.placed_blocks <= 0:
            return 0.0
        val = (self.total_key_presses + self.placed_blocks) / self.placed_blocks
        return round(100.0 * val) / 100.0

    def get_apm(self) -> float:
        if self.clock <= 0:
            return 0.0
        return round(100.0 * self.gamedata["attack"] / ((self.clock / 1000.0) / 60.0)) / 100.0

    def get_vs(self) -> float:
        if self.clock <= 0:
            return 0.0
        return round(10000.0 * (self.gamedata["garbageCleared"] + self.gamedata["attack"]) / (self.clock / 1000.0)) / 100.0

    def get_wasted_t_ratio(self) -> float:
        if self.gamedata["tpieces"] <= 0:
            return 0.0
        return round((self.gamedata["wasted"] / self.gamedata["tpieces"]) * 100.0) / 100.0
