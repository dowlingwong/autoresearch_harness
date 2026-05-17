from __future__ import annotations

from dataclasses import asdict, dataclass

from autoresearch.memory.similarity import fuzzy_sequence_similarity, normalize_text


@dataclass(frozen=True)
class DoomLoopFinding:
    repeated: bool
    count: int
    last_action: str
    matched_actions: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["matched_actions"] = list(self.matched_actions)
        return payload


def detect_doom_loop(
    actions: list[str] | tuple[str, ...],
    *,
    max_repeats: int = 3,
    fuzzy_threshold: float = 0.92,
) -> DoomLoopFinding:
    """Detect repeated intra-proposal actions before a manager burns worker budget."""
    cleaned = tuple(normalize_text(action) for action in actions if normalize_text(action))
    if not cleaned:
        return DoomLoopFinding(False, 0, "", (), "no_actions")
    last = cleaned[-1]
    matches = tuple(action for action in cleaned if _same_action(last, action, fuzzy_threshold=fuzzy_threshold))
    repeated = len(matches) >= max_repeats
    return DoomLoopFinding(
        repeated=repeated,
        count=len(matches),
        last_action=last,
        matched_actions=matches,
        reason="max_repeats_reached" if repeated else "below_threshold",
    )


def reject_doom_loop(
    actions: list[str] | tuple[str, ...],
    *,
    max_repeats: int = 3,
    fuzzy_threshold: float = 0.92,
) -> None:
    finding = detect_doom_loop(
        actions,
        max_repeats=max_repeats,
        fuzzy_threshold=fuzzy_threshold,
    )
    if finding.repeated:
        raise ValueError(
            f"repeated proposal action detected {finding.count} times: {finding.last_action}"
        )


def _same_action(left: str, right: str, *, fuzzy_threshold: float) -> bool:
    return left == right or fuzzy_sequence_similarity(left, right) >= fuzzy_threshold
