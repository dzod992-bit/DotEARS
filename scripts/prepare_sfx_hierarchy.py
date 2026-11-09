"""Compose separately calibrated whoosh and broad-signal policies without refitting."""

import argparse
import json
from pathlib import Path

from dotears.audiosep import file_digest
from dotears.sfx_hierarchy import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whoosh-policy", type=Path, required=True)
    parser.add_argument("--signal-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = build(args.whoosh_policy.resolve(), args.signal_policy.resolve(), args.output.resolve())
    print(
        json.dumps(
            {"policy_path": str(path), "policy_sha256": file_digest(path), "refitted": False}
        )
    )


if __name__ == "__main__":
    main()
