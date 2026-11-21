"""Allow running package via `python3 -m jstris_replay`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
