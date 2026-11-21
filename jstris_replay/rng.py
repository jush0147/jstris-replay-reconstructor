"""Alea PRNG and Jstris piece randomizers (7-Bag, 14-Bag, Classic, C2Sim, etc.)."""

from __future__ import annotations

import math
from typing import Callable

PIECE_NAMES = ["I", "O", "T", "L", "J", "S", "Z"]
PIECE_COLORS = {
    0: 5,  # I -> Cyan (5)
    1: 3,  # O -> Yellow (3)
    2: 7,  # T -> Purple (7)
    3: 2,  # L -> Orange (2)
    4: 6,  # J -> Blue (6)
    5: 4,  # S -> Green (4)
    6: 1,  # Z -> Red (1)
}


def _utf16play_cunits(s: str) -> list[int]:
    data = s.encode("utf-16-le")
    return [data[i] | (data[i + 1] << 8) for i in range(0, len(data), 2)]


class _Mash:
    """Exact port of Johannes Baagøe's Mash hash function used by Alea PRNG."""

    def __init__(self) -> None:
        self.n: int = 4022871197  # 0xefc8249d

    def __call__(self, data: object) -> float:
        s = str(data)
        n = float(self.n)
        for code in _utf16play_cunits(s):
            n += code
            h = 0.02519603282416938 * n
            n = float(int(h) & 0xFFFFFFFF)
            h -= n
            h *= n
            n = float(int(h) & 0xFFFFFFFF)
            h -= n
            n += h * 4294967296.0
        self.n = int(n) & 0xFFFFFFFF
        return self.n * 2.3283064365386963e-10


class AleaPRNG:
    """Exact port of Jstris Alea PRNG."""

    def __init__(self, *seeds: object) -> None:
        if not seeds:
            seeds = ("",)
        mash = _Mash()
        self.c: int = 1
        self.s0: float = mash(" ")
        self.s1: float = mash(" ")
        self.s2: float = mash(" ")

        for seed in seeds:
            self.s0 -= mash(seed)
            if self.s0 < 0.0:
                self.s0 += 1.0
            self.s1 -= mash(seed)
            if self.s1 < 0.0:
                self.s1 += 1.0
            self.s2 -= mash(seed)
            if self.s2 < 0.0:
                self.s2 += 1.0

    def next(self) -> float:
        t = 2091639.0 * self.s0 + self.c * 2.3283064365386963e-10
        self.s0 = self.s1
        self.s1 = self.s2
        self.c = int(t)
        self.s2 = t - self.c
        return self.s2

    def __call__(self) -> float:
        return self.next()

    def clone(self) -> AleaPRNG:
        copy = AleaPRNG.__new__(AleaPRNG)
        copy.c = self.c
        copy.s0 = self.s0
        copy.s1 = self.s1
        copy.s2 = self.s2
        return copy


class BagRandomizer:
    """Jstris N-Bag randomizer (e.g. 7-bag when n=7, mult=1; 14-bag when n=7, mult=2)."""

    def __init__(self, rng: Callable[[], float], n: int = 7, mult: int = 1) -> None:
        self.rng = rng
        self.usebag: list[int] = []
        for piece_id in range(n):
            for _ in range(mult):
                self.usebag.append(piece_id)
        self.bag: list[int] = list(self.usebag)

    def get_block_id(self) -> int:
        idx = int(math.floor(self.rng() * len(self.bag)))
        piece_id = self.bag.pop(idx)
        if len(self.bag) == 0:
            self.bag = list(self.usebag)
        return piece_id


class ClassicRandomizer:
    """Jstris Classic (memoryless) randomizer (rnd=2, or MPH ruleset r=3)."""

    def __init__(self, rng: Callable[[], float], n: int = 7) -> None:
        self.rng = rng
        self.n = n

    def get_block_id(self) -> int:
        return int(math.floor(self.rng() * self.n))


class C2SimRandomizer:
    """Jstris Cultris II simulation randomizer (rnd=7)."""

    def __init__(self, rng: Callable[[], float], n: int = 7) -> None:
        self.rng = rng
        self.n = n
        self.hist = [-1, -2]

    def _get_random_except(self, exc: int) -> int:
        val = int(math.floor(self.rng() * (self.n - 1)))
        if exc >= 0 and val >= exc:
            val += 1
        if val == self.n:
            val = 0
        return val

    def get_block_id(self) -> int:
        if self.hist[0] < 0:
            piece_id = int(math.floor(self.rng() * self.n))
        elif self.hist[0] == self.hist[1] or self.hist[1] < 0:
            piece_id = self.hist[0] if self.rng() <= 0.028105 else self._get_random_except(self.hist[0])
        else:
            r = self.rng()
            if r <= 0.027055:
                piece_id = self.hist[0]
            elif r >= 0.882639:
                piece_id = self.hist[1]
            else:
                while True:
                    piece_id = self._get_random_except(self.hist[0])
                    if piece_id != self.hist[1]:
                        break
        self.hist[1] = self.hist[0]
        self.hist[0] = piece_id
        return piece_id


def create_randomizer(rnd_type: int, rng: Callable[[], float], n_blocks: int = 7):
    """Factory matching Jstris GameCore.prototype.randomizerFactory."""
    if rnd_type == 1:
        return BagRandomizer(rng, n_blocks, 2)
    if rnd_type == 2:
        return ClassicRandomizer(rng, n_blocks)
    if rnd_type == 7:
        return C2SimRandomizer(rng, n_blocks)
    return BagRandomizer(rng, n_blocks, 1)
