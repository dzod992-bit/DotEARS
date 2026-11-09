"""Version a reviewed reference policy with broad UI scope, preserving every rejection.

This updates label membership, not model training or recognition performance.
Existing notification/other labels, samples, thresholds and disabled flags stay intact.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

from dotears.audiosep import file_digest
from dotears.sfx import local_artifact
from dotears.sfx_match import BROAD_SCHEMA, ReferenceMatcher
from dotears.sfx_taxonomy import UI_DEFINITIONS, UI_TAXONOMY, broad_ui_label


def migrate(source: Path, destination: Path) -> Path:
    matcher = ReferenceMatcher(source)
    if destination.exists():
        raise ValueError("Use a new directory for the broad UI policy.")
    source_hash = file_digest(source)
    policy = copy.deepcopy(matcher.policy)
    bank = local_artifact(source.resolve().parent, policy["bank_path"])
    data = bank.read_bytes()
    policy.update(
        schema_version=BROAD_SCHEMA,
        label_taxonomy=UI_TAXONOMY,
        label_definitions=UI_DEFINITIONS,
        bank_path="bank.npz",
        created_at=datetime.now(UTC).isoformat(),
        parent_policy_sha256=source_hash,
        status="active_known_reference_patterns_broad_ui_scope_no_new_training",
        scope_update={
            "authorization": "User includes clicks/taps, beeps and UI confirmation tones.",
            "method": "Label mapping only; scores, thresholds and enable flags are unchanged.",
            "performance_revalidated_for_new_tones": False,
        },
    )
    if "test_gate" in policy:
        policy["legacy_narrow_test_gate"] = policy.pop("test_gate")
    for template in policy["templates"]:
        template.setdefault("original_review_label", template["label"])
        template["label"] = broad_ui_label(template["label"])
    if file_digest(source) != source_hash:
        raise ValueError("Source policy changed during scope migration.")
    destination.mkdir(parents=True)
    (destination / "bank.npz").write_bytes(data)
    target = destination / "active-policy.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump(policy, handle, ensure_ascii=False, indent=2, allow_nan=False)
    ReferenceMatcher(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = migrate(args.policy.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "policy_path": str(target),
                "policy_sha256": file_digest(target),
                "label_taxonomy": UI_TAXONOMY,
                "retrained": False,
            }
        )
    )


if __name__ == "__main__":
    main()
