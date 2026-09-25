"""The two store-wide sweeps: same rows written, and the walk is not the cost.

``backfill_session_origins`` and ``backfill_session_titles`` each walked every
directory in the store — three ``stat`` calls apiece, re-paid by every runtime's
maintenance thread every host minute and again on every ``lop --resume``. On the
reporting machine's store shape (11,546 directories, 93% of them delegated runs)
that is **17,636 + 22,815 = 40,451 syscalls and ~603 ms CPU per cycle** for two
passes that write nothing, because every answer was written long ago.

Three things are pinned here, and the FIRST one outranks the other two:

* IDENTITY OF THE WRITES. Both sweeps write to the store, so "the same rows" is
  the contract: the same markers, the same sentinels, the same sidecars — with
  the same bytes and the same preserved directory mtimes — for the same
  directories, in the same order, and with the same return values. The removed
  implementations are kept VERBATIM as oracles and run against an identical
  fixture, and the two stores are compared file by file.
* THE PROJECTION. The origin sweep's candidate set now comes from
  ``_scan_sessions``, so the ~93% of a store that is delegated runs costs it
  nothing at all — asserted as flatness while that population grows, not as a
  bound.
* THE FRONTIER. The title sweep records the instant a COMPLETED pass swept the
  store, and answers a directory older than that with one stat against the
  directory itself. Every way a directory becomes un-answered (an entry added or
  removed inside it) moves that stat's value, so the write set is unchanged —
  and an INCOMPLETE pass (a ``limit`` that cut it short, a write that failed)
  may not advance the frontier at all.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator import resume as resume_mod
from local_operator.resume import (
    ORIGIN_NAME,
    ORIGIN_SCAN_SENTINEL_NAME,
    TITLE_SCAN_SENTINEL_NAME,
    TITLE_SIDECAR_NAME,
    TRANSCRIPT_NAME,
    backfill_session_origins,
    backfill_session_titles,
    title_sweep_stamp_path,
)

USER_OPENING = {"role": "user", "content": [{"text": "fix the resume picker"}]}
ROLE_OPENING = {
    "role": "user",
    "content": [{"text": "[role: reviewer]\nYou are an INDEPENDENT reviewer."}],
}
SCOUT_OPENING = {
    "role": "user",
    "content": [{"text": "[scout mode: you are a READ-ONLY agent.]\nfind it"}],
}
TITLE_ROW = {
    "custom_type": "conversation_name",
    "details": {"text": "A Measured Title", "user_set": False},
}
PINNED_MTIME = 1_700_000_000.0


# ---------------------------------------------------------------------------
# Fixtures: built deterministically, with every mtime and directory stamp pinned
# ---------------------------------------------------------------------------


def _message(opening: dict[str, Any], *, title: bool = False) -> str:
    rows = [{"id": "e1", "ts": 0, "type": "message", "payload": {"kind": "message", **opening}}]
    if title:
        rows.append({"id": "e2", "ts": 1, "type": "custom", "payload": TITLE_ROW})
    return "".join(json.dumps(row) + "\n" for row in rows)


def _session(
    root: Path,
    name: str,
    *,
    opening: dict[str, Any] | None = None,
    transcript: bool = True,
    title_row: bool = False,
    origin: str | None = None,
    corrupt_marker: bool = False,
    sidecar: str | None = None,
    sentinel: bool = False,
    inbox: bool = False,
) -> Path:
    directory = root / "sessions" / name
    directory.mkdir(parents=True, exist_ok=True)
    if transcript:
        (directory / TRANSCRIPT_NAME).write_text(
            _message(opening or USER_OPENING, title=title_row), encoding="utf-8"
        )
    if inbox:
        (directory / "inbox.jsonl").write_text("{}\n", encoding="utf-8")
    if origin is not None:
        (directory / ORIGIN_NAME).write_text(json.dumps({"origin": origin}), encoding="utf-8")
    if corrupt_marker:
        (directory / ORIGIN_NAME).write_bytes(b'{"origin": "suba')
    if sidecar is not None:
        (directory / TITLE_SIDECAR_NAME).write_text(sidecar, encoding="utf-8")
    if sentinel:
        (directory / TITLE_SCAN_SENTINEL_NAME).write_text('{"scanned": true}\n', encoding="utf-8")
    return directory


def _pin(root: Path) -> None:
    """Pin every file's and directory's mtime so two builds are comparable.

    The title sweep's frontier is a wall-clock instant compared against a
    directory's ``st_mtime_ns``, so a fixture that let ``mkdir`` stamp "now"
    would make the steady state unreachable and the comparison meaningless.
    """
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        os.utime(path, (PINNED_MTIME, PINNED_MTIME))
    os.utime(root / "sessions", (PINNED_MTIME, PINNED_MTIME))


SHAPES: dict[str, Callable[[Path], None]] = {
    # A delegated run and a scout run: the sweep's two stampable openings.
    "stampable": lambda root: (
        _session(root, "0child000001", opening=ROLE_OPENING),
        _session(root, "0scout000001", opening=SCOUT_OPENING),
        _session(root, "user00000001"),
    ),
    # Every shape that must be SKIPPED, each for a different reason.
    "skips": lambda root: (
        _session(root, "1user0000001"),
        _session(root, "2fork0000001", origin="fork"),
        _session(root, "3shell000001", origin="agent-shell"),
        _session(root, "4subagent01", origin="subagent"),
        _session(root, "5corrupt001", corrupt_marker=True),
        _session(root, "6answered01", sentinel=True),
        _session(root, "7empty00001", transcript=False),
        _session(root, "8inbox00001", transcript=False, inbox=True),
        _session(
            root,
            "9quoting01",
            opening={"role": "user", "content": [{"text": "why [role: reviewer]?"}]},
        ),
    ),
    # A directory the sweep must answer on a LATER run: the quoted preamble is
    # not a delegated opening, so it gets the sentinel and the run goes on.
    "quoting-then-stampable": lambda root: (
        _session(
            root,
            "0quoting001",
            opening={"role": "user", "content": [{"text": "why [role: reviewer]?"}]},
        ),
        _session(root, "1child000001", opening=ROLE_OPENING),
        _session(root, "2child000002", opening=SCOUT_OPENING),
    ),
    "titles-everywhere": lambda root: (
        _session(root, "0titled00001", title_row=True),
        _session(root, "1untitled001"),
        _session(root, "2subagent001", origin="subagent", title_row=True),
        _session(root, "3answered01", sidecar='{"text": "kept", "names": []}'),
        _session(root, "4sentinel01", sentinel=True),
        _session(root, "5corrupt001", sidecar='{"text": "trunc'),
        _session(root, "6empty00001", transcript=False),
        _session(root, "7inbox00001", transcript=False, inbox=True),
    ),
}


def _fingerprint(root: Path) -> list[tuple[str, str, int]]:
    """Every entry under ``sessions/``, with its bytes and its DIRECTORY's mtime.

    The COMPARISON SURFACE is the session store, because that is where both
    sweeps write and what "the same rows" means. Derived data under ``cache/``
    (the verdict cache, the sweep frontier) is deliberately outside it: it is a
    cache, and a cache that differed while the store did not would be a finding
    about the cache, not about the writes.

    A FILE's own mtime is deliberately not part of the comparison: the answer
    files are CREATED by the run under test, so their timestamps are that run's
    own clock and can never match across two runs. What must not move is the
    DIRECTORY's mtime — the retention and listing clock every writer here is
    careful to preserve — and that IS compared. A transcript's mtime is checked
    separately by :func:`test_neither_sweep_moves_a_transcripts_clock`.
    """
    sessions = root / "sessions"
    rows: list[tuple[str, str, int]] = []
    for path in sorted(sessions.rglob("*")):
        relative = str(path.relative_to(sessions))
        if path.is_dir():
            rows.append(("dir " + relative, "", path.stat().st_mtime_ns))
        else:
            rows.append(("file " + relative, path.read_text(encoding="utf-8", errors="replace"), 0))
    return rows


def _both_arms(
    tmp_path: Path, shape: str, old: Callable[[Path], int], new: Callable[[Path], int]
) -> tuple[int, int]:
    """The oracle and the implementation over two identical builds of ``shape``."""
    old_root, new_root = tmp_path / "oracle-store", tmp_path / "lane-store"
    for root in (old_root, new_root):
        SHAPES[shape](root)
        _pin(root)
    before = old(old_root)
    after = new(new_root)
    assert _fingerprint(new_root) == _fingerprint(old_root), "the store differs, file by file"
    return before, after


# ---------------------------------------------------------------------------
# The oracles: the removed implementations, verbatim
# ---------------------------------------------------------------------------


def _oracle_origins(config_dir: Path, limit: int = 500) -> int:
    """``backfill_session_origins`` before this lane, copied rather than described."""
    from local_operator.resume import _ROLE_PREAMBLE, _SCOUT_PREAMBLE, NAME_MAX_CHARS
    from local_operator.resume import ORIGIN_SUBAGENT as SUBAGENT
    from local_operator.resume import (
        _write_origin_scan_sentinel,
        mark_session_origin,
        session_name,
    )

    stamped = 0
    sessions = config_dir / "sessions"
    try:
        directories = sorted(sessions.iterdir())
    except OSError:
        return 0
    for directory in directories:
        if stamped >= limit:
            break
        try:
            if not (directory / TRANSCRIPT_NAME).is_file():
                continue
            if (directory / ORIGIN_NAME).exists():
                continue
            if (directory / ORIGIN_SCAN_SENTINEL_NAME).exists():
                continue
        except OSError:
            continue
        opening = session_name(directory, max_chars=NAME_MAX_CHARS, condense=False)
        if not opening:
            continue
        if _ROLE_PREAMBLE.match(opening) or opening.startswith(_SCOUT_PREAMBLE):
            mark_session_origin(directory, SUBAGENT, backfilled=True)
            stamped += 1
        else:
            _write_origin_scan_sentinel(directory)
    return stamped


def _oracle_titles(config_dir: Path, limit: int = 500) -> int:
    """``backfill_session_titles`` before this lane, copied rather than described."""
    from local_operator.resume import (
        _scan_all_titles,
        _write_title_scan_sentinel,
        write_session_title,
    )

    written = 0
    sessions = config_dir / "sessions"
    try:
        directories = sorted(sessions.iterdir())
    except OSError:
        return 0
    for directory in directories:
        if written >= limit:
            break
        try:
            transcript = directory / TRANSCRIPT_NAME
            if not transcript.is_file():
                continue
            if (directory / TITLE_SIDECAR_NAME).exists():
                continue
            if (directory / TITLE_SCAN_SENTINEL_NAME).exists():
                continue
        except OSError:
            continue
        titles = _scan_all_titles(transcript)
        if not titles:
            _write_title_scan_sentinel(directory)
            continue
        past_names = [text for text, _ in titles]
        newest_text, newest_user_set = titles[-1]
        write_session_title(directory, newest_text, user_set=newest_user_set, past_names=past_names)
        written += 1
    return written


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_origin_sweep_writes_exactly_what_it_used_to(tmp_path: Path, shape: str) -> None:
    """Same rows: markers, sentinels, bytes and preserved mtimes, per shape.

    The oracle runs against an identical build rather than against a described
    rule, and the comparison is the whole ``sessions/`` tree — every file's
    bytes and mtime, and every directory's mtime — so a difference in WHICH
    directory was stamped, what was written into it, or whether the stamping
    moved the retention clock the writers are careful to preserve all fail here.
    """
    before, after = _both_arms(tmp_path, shape, _oracle_origins, backfill_session_origins)
    assert after == before, "the sweep stamped a different number of directories"


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_title_sweep_writes_exactly_what_it_used_to(tmp_path: Path, shape: str) -> None:
    """Same rows, including the sidecar's BYTES: an older scan must not clobber
    a newer name, and the two arms of the sweep (sidecar vs sentinel) must land
    on the same directories."""
    before, after = _both_arms(tmp_path, shape, _oracle_titles, backfill_session_titles)
    assert after == before, "the sweep wrote a different number of sidecars"


@pytest.mark.parametrize(
    "shape, sweep",
    [
        ("stampable", "origins"),
        ("quoting-then-stampable", "origins"),
        ("titles-everywhere", "titles"),
    ],
)
def test_a_run_bounded_by_limit_answers_the_same_directories_in_the_same_order(
    tmp_path: Path, shape: str, sweep: str
) -> None:
    """``limit`` decides WHICH directories a run answers, and it decides by NAME.

    Both arms run three times at ``limit=1``: the same directory must be
    answered on each run, and the runs together must cover the store exactly as
    the old implementation did — the cap is on work done, never on how far the
    walk reaches.
    """
    if sweep == "origins":
        old, new = _oracle_origins, backfill_session_origins
    else:
        old, new = _oracle_titles, backfill_session_titles

    old_root, new_root = tmp_path / "old", tmp_path / "new"
    for root in (old_root, new_root):
        SHAPES[shape](root)
        _pin(root)

    for run in range(3):
        assert new(new_root, limit=1) == old(old_root, limit=1), f"run {run}"
        assert _fingerprint(new_root) == _fingerprint(old_root), f"run {run}"


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_second_run_writes_nothing_at_all(tmp_path: Path, shape: str) -> None:
    """Idempotence, in both sweeps: every answer is event-sourced from its first
    write, so the second pass must add no file and touch no mtime."""
    root = tmp_path / shape
    SHAPES[shape](root)
    _pin(root)
    backfill_session_origins(root)
    backfill_session_titles(root)
    settled = _fingerprint(root)

    assert backfill_session_origins(root) == 0
    assert backfill_session_titles(root) == 0
    assert _fingerprint(root) == settled


def test_a_corrupt_marker_is_never_re_stamped_by_the_projection(tmp_path: Path) -> None:
    """THE EXACTNESS CASE the candidate set has to preserve.

    A row whose ``origin`` the scan reports as ``""`` is either a directory with
    no marker or one whose marker exists but will not parse — and the second is
    a session the fail-safe deliberately keeps VISIBLE. Skipping the marker
    re-check (which the projection cannot answer) would read the opener of that
    directory and could stamp a fresh marker over the corrupt one, hiding a row
    the old sweep left alone.
    """
    root = tmp_path / "store"
    _session(root, "0corrupt001", opening=ROLE_OPENING, corrupt_marker=True)
    _pin(root)
    marker = root / "sessions" / "0corrupt001" / ORIGIN_NAME
    before = marker.read_bytes()

    assert backfill_session_origins(root) == 0
    assert marker.read_bytes() == before
    assert not (root / "sessions" / "0corrupt001" / ORIGIN_SCAN_SENTINEL_NAME).exists()


def test_a_missing_transcript_is_never_given_a_title_sentinel(tmp_path: Path) -> None:
    """A directory with no transcript is not a session yet: the old sweep wrote
    nothing for it and the new one must not either, however cheap the check
    becomes."""
    root = tmp_path / "store"
    empty = _session(root, "0empty00001", transcript=False)
    _pin(root)
    assert backfill_session_titles(root) == 0
    assert sorted(entry.name for entry in empty.iterdir()) == []


def test_neither_sweep_moves_a_transcripts_clock(tmp_path: Path) -> None:
    """The activity clock is the transcript's mtime, and both sweeps are
    bookkeeping ABOUT a session rather than work IN it — so neither may move it,
    on a directory it answers or on one it walks past."""
    root = tmp_path / "store"
    SHAPES["titles-everywhere"](root)
    _pin(root)
    clocks = {
        path: path.stat().st_mtime_ns for path in sorted((root / "sessions").rglob(TRANSCRIPT_NAME))
    }
    assert clocks, "fixture: no transcript to watch"

    backfill_session_origins(root)
    backfill_session_titles(root)

    for path, stamp in clocks.items():
        assert path.stat().st_mtime_ns == stamp, f"{path} moved its session's clock"


# ---------------------------------------------------------------------------
# Work done: the projection for origins, the frontier for titles
# ---------------------------------------------------------------------------


def _counting(names: tuple[str, ...] = ("stat", "lstat", "scandir")) -> Any:
    """Count ``os`` filesystem calls made inside the context (audit hook)."""

    class Counter:
        def __init__(self) -> None:
            self.counts: dict[str, int] = {name: 0 for name in names}
            self._originals: dict[str, Callable[..., Any]] = {}

        def __enter__(self) -> "Counter":
            for name in names:
                real = getattr(os, name)
                self._originals[name] = real

                def wrap(real: Callable[..., Any] = real, name: str = name) -> Any:
                    def counting(*args: Any, **kwargs: Any) -> Any:
                        self.counts[name] += 1
                        return real(*args, **kwargs)

                    return counting

                setattr(os, name, wrap())
            return self

        def __exit__(self, *exc: Any) -> None:
            for name, real in self._originals.items():
                setattr(os, name, real)

    return Counter()


def _store(root: Path, *, users: int, hidden: int) -> None:
    for index in range(users):
        _session(root, f"user{index:08x}")
    for index in range(hidden):
        _session(root, f"sub{index:09x}", origin="subagent")
    _pin(root)


def _warm_origins(root: Path) -> None:
    """Two passes: the first fills the verdict cache, the second is the steady
    state the maintenance thread actually repeats."""
    backfill_session_origins(root)
    backfill_session_origins(root)


class TestTheOriginSweepRidesTheProjection:
    def test_a_warm_pass_over_a_store_of_delegated_runs_issues_no_stat_at_all(
        self, tmp_path: Path
    ) -> None:
        """The extreme the projection implies, and the shape of a real store's
        majority: 300 delegated runs, every one already marked, so the scan
        skips them whole from the readdir batch and the sweep has no candidate
        to probe. One ``scandir``. Zero stats."""
        _store(tmp_path, users=0, hidden=300)
        _warm_origins(tmp_path)
        with _counting() as counter:
            assert backfill_session_origins(tmp_path) == 0
        assert counter.counts == {"stat": 0, "lstat": 0, "scandir": 1}, counter.counts

    def test_the_hidden_population_costs_the_sweep_nothing_however_large_it_gets(
        self, tmp_path: Path
    ) -> None:
        """FLATNESS, and the property that separates this from the old walk: 300
        more delegated runs must not move a warm pass's syscall counts at all.

        Equalities, not bounds: a bound on the total would let the per-directory
        slope creep back in underneath a constant that grows with it — which is
        precisely how the walk this replaces stayed linear while looking cheap.
        """
        _store(tmp_path, users=4, hidden=20)
        _warm_origins(tmp_path)
        with _counting() as before:
            backfill_session_origins(tmp_path)

        for index in range(300):
            _session(tmp_path, f"grow{index:08x}", origin="subagent")
        _pin(tmp_path)
        _warm_origins(tmp_path)
        with _counting() as after:
            assert backfill_session_origins(tmp_path) == 0

        assert after.counts == before.counts, (before.counts, after.counts)

    def test_the_never_active_population_costs_the_sweep_nothing_of_its_own(
        self, tmp_path: Path
    ) -> None:
        """WHERE THE FLATNESS ABOVE STOPS, recorded rather than left to be found.

        A directory with neither an origin marker nor any activity is in neither
        the listing nor the hidden set, so it can never arm the scan's
        zero-syscall skip and the SCAN pays its documented constant for it (a
        marker stat plus the two activity files). That cost is the poll's own,
        already pinned by ``test_catalog_scan_cost``, and this sweep inherits it
        because the scan is what produces its candidates.

        What must be true here is that the sweep adds NOTHING on top: the whole
        marginal cost of an idle directory is that scan constant, and not the
        three existence checks the old walk paid per directory per pass. On the
        reporting machine's store there is no such population at all (every
        directory is either marked or visible), so this is the fixture's limit
        rather than the real store's."""
        _store(tmp_path, users=3, hidden=0)
        _warm_origins(tmp_path)
        with _counting() as before:
            assert backfill_session_origins(tmp_path) == 0

        for index in range(200):
            (tmp_path / "sessions" / f"idle{index:08x}").mkdir(parents=True)
        _pin(tmp_path)
        _warm_origins(tmp_path)
        with _counting() as after:
            assert backfill_session_origins(tmp_path) == 0

        assert after.counts["scandir"] == before.counts["scandir"] == 1
        # The scan's own constant: one marker stat and two activity stats. The
        # sweep contributes zero — the old walk would have added 200 more.
        assert after.counts["stat"] - before.counts["stat"] == 200 * 3, after.counts


class TestTheTitleSweepFrontier:
    def test_a_completed_pass_records_a_frontier_and_the_next_touches_one_stat_per_directory(
        self, tmp_path: Path
    ) -> None:
        """THE STEADY STATE, as exact counts.

        After one completed pass, the next one must issue exactly one stat per
        directory — the frontier check against the directory itself — and not
        one ``is_file``/``exists`` inside any of them. The old implementation
        issued two to three per directory, forever.
        """
        _store(tmp_path, users=3, hidden=20)
        assert backfill_session_titles(tmp_path) >= 0
        assert title_sweep_stamp_path(tmp_path).exists(), "a completed pass recorded no frontier"
        directories = len(list((tmp_path / "sessions").iterdir()))

        settled = _fingerprint(tmp_path)
        with _counting() as counter:
            assert backfill_session_titles(tmp_path) == 0

        # EXACTLY one stat per directory, and the second term is the rewrite of
        # the frontier itself: ``mkdir(parents=True, exist_ok=True)`` resolves a
        # collision with one ``is_dir`` probe on ``cache/``. Nothing else — not
        # one ``is_file`` or ``exists`` inside any of the 23 directories.
        assert counter.counts == {"stat": directories + 1, "lstat": 0, "scandir": 1}, counter.counts
        assert _fingerprint(tmp_path) == settled, "a covered pass wrote something"

    def test_a_limit_that_cut_the_pass_short_does_not_advance_the_frontier(
        self, tmp_path: Path
    ) -> None:
        """An incomplete pass must leave the frontier where it was, or the
        directories it never reached would be skipped for good — the same
        failure the "cap on work done, never on reach" rule exists to prevent.
        """
        root = tmp_path / "store"
        for index in range(3):
            _session(root, f"titled{index:08x}", title_row=True)
        _pin(root)

        assert backfill_session_titles(root, limit=1) == 1
        assert not title_sweep_stamp_path(root).exists(), "a capped pass claimed the store swept"
        assert backfill_session_titles(root, limit=1) == 1
        assert backfill_session_titles(root, limit=1) == 1
        assert backfill_session_titles(root, limit=1) == 0
        assert title_sweep_stamp_path(root).exists(), "the completing pass recorded nothing"

    def test_a_session_created_after_the_frontier_is_answered(self, tmp_path: Path) -> None:
        """The frontier must not hide work that appears after it — the failure
        that would make this a permanent hole rather than a cache."""
        _store(tmp_path, users=2, hidden=2)
        backfill_session_titles(tmp_path)
        fresh = _session(tmp_path, "0newtitled01", title_row=True)

        assert backfill_session_titles(tmp_path) == 1
        assert (fresh / TITLE_SIDECAR_NAME).exists()

    def test_a_directory_that_loses_its_answer_is_answered_again(self, tmp_path: Path) -> None:
        """BEHAVIOUR PRESERVED, and the reason the check is the DIRECTORY's own
        mtime rather than a name set: deleting a sidecar or sentinel is an entry
        removal, which moves that mtime, so the directory is re-probed and
        re-answered exactly as the old sweep would have done."""
        root = tmp_path / "store"
        directory = _session(root, "0untitled001")
        _pin(root)
        assert backfill_session_titles(root) == 0
        assert (directory / TITLE_SCAN_SENTINEL_NAME).exists()

        (directory / TITLE_SCAN_SENTINEL_NAME).unlink()
        assert backfill_session_titles(root) == 0
        assert (directory / TITLE_SCAN_SENTINEL_NAME).exists(), "the answer was not restored"

    def test_a_frontier_from_the_future_is_ignored_and_replaced(self, tmp_path: Path) -> None:
        """A clock that stepped BACKWARDS wrote a frontier every new directory
        would fall below; trusting it would skip the whole store forever. The
        pass must notice, do the full walk, and rewrite the frontier."""
        root = tmp_path / "store"
        directory = _session(root, "0untitled001")
        _pin(root)
        stamp = title_sweep_stamp_path(root)
        stamp.parent.mkdir(parents=True, exist_ok=True)
        future = 4_000_000_000_000_000_000  # year ~2096, in ns
        stamp.write_text(
            json.dumps(
                {"version": resume_mod.TITLE_SWEEP_STAMP_VERSION, "completed_at_ns": future}
            ),
            encoding="utf-8",
        )

        assert backfill_session_titles(root) == 0
        assert (directory / TITLE_SCAN_SENTINEL_NAME).exists(), "the future frontier was trusted"
        assert resume_mod._read_title_sweep_stamp(root) not in (None, future)

    @pytest.mark.parametrize("corrupt", ["", "{torn", "null", "[]", '{"version": 99}'])
    def test_an_unusable_frontier_costs_a_full_pass_and_never_an_answer(
        self, tmp_path: Path, corrupt: str
    ) -> None:
        """Every failure mode of a cache whose worst cost must be the work it
        replaces: absent, torn, a wrong shape, an unknown version."""
        root = tmp_path / "store"
        _session(root, "0untitled001")
        _pin(root)
        stamp = title_sweep_stamp_path(root)
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(corrupt, encoding="utf-8")

        assert resume_mod._read_title_sweep_stamp(root) is None
        assert backfill_session_titles(root) == 0
        assert (root / "sessions" / "0untitled001" / TITLE_SCAN_SENTINEL_NAME).exists()

    def test_a_deleted_store_answers_nothing_and_writes_nothing(self, tmp_path: Path) -> None:
        """A sweep on a config dir with no store at all: no stamp, no raise."""
        assert backfill_session_titles(tmp_path) == 0
        assert backfill_session_origins(tmp_path) == 0
        assert not title_sweep_stamp_path(tmp_path).exists()


def test_the_frontier_lives_under_cache_and_is_removable(tmp_path: Path) -> None:
    """Derived data, not a source of truth: deleting it must force the full
    pass back, which is the documented remedy for the one shape it cannot see
    (a directory whose mtime was moved backwards below the frontier)."""
    root = tmp_path / "store"
    directory = _session(root, "0untitled001")
    _pin(root)
    backfill_session_titles(root)
    assert title_sweep_stamp_path(root).parent.name == "cache"

    (directory / TITLE_SCAN_SENTINEL_NAME).unlink()
    shutil.rmtree(title_sweep_stamp_path(root).parent)
    assert backfill_session_titles(root) == 0
    assert (directory / TITLE_SCAN_SENTINEL_NAME).exists()
