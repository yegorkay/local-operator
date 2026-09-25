"""The batched process-table read, and the per-pid readers it replaces.

`GET /v1/desktop/runtimes` used to fork ELEVEN processes per poll: one ``lsof``,
one census ``ps``, one ``ps -Eww``, FIVE ``ps -o ppid= -p <pid>`` (the ancestry
walk is one fork per hop) and THREE ``/bin/ps -o pid=,state=,lstart= -p <pids>``
(one per record population — a foreign root scanned is a batch, and there are up
to ``roster.FOREIGN_ROOT_LIMIT`` of them). Measured on the reference host that was
286.1 ms of CPU per poll, 240.8 ms of it CHILD CPU that ``time.process_time()``
cannot see.

Eight of those eleven are the per-pid readers, and this module pins the claim that
removing them changes NO VALUE a caller reads. Two levels, because they fail
differently:

* the PARSE level — a ``pid=,state=,lstart=`` line and a
  ``pid=,ppid=,etime=,time=,state=,lstart=,command=`` line for the same process
  must produce the same :class:`~local_operator.procstate.ProcessSample`, and a
  merged line must produce the same census row as the census invocation's line. A
  difference here is a parser bug, and it is asserted on synthetic text so it
  cannot be a sampling artifact;
* the VALUE level — with ONE machine described as a fixture, the composition
  inside a batch scope and the composition over the per-pid readers must produce
  equal ``RosterRow`` objects, field for field, including the fields only a zombie
  probe or the census can fill.

The structural half is what survives a loaded host: fork counts and fork SHAPES,
asserted by wrapping ``subprocess.run``, never a wall-clock bound (see ``AGENTS.md``,
"Timing, flakes, and how to assert that something is fast").
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from local_operator import procstate
from local_operator.procstate import ProcessSample, ProcessTable, parse_process_table
from local_operator.server.routes import desktop_runtimes
from local_operator.session.runtime import reclaim, roster
from local_operator.session.runtime.reclaim import Fleet, RuntimeProcess, SocketEvidence
from local_operator.session.runtime.roster import RosterRow, build_roster
from local_operator.session.runtime.types import SessionRecord

#: A REAL live pid: the roster asks signal-0 about every row, and a pid above the
#: platform's pid space answers "gone" whatever the fixture's census says.
MINE = os.getpid()
#: A second real, live pid that is NOT a session runtime — the census's spawn
#: contract has to keep excluding it on both paths.
OTHER_LIVE = os.getppid()
GONE_PID = 999_999

#: THE MACHINE BOTH PATHS ARE SHOWN. Rendered into BOTH ``ps`` layouts, so neither
#: path is ever compared against a value the other could not have read.
#:
#: ``MINE`` carries the spawn contract in its argv (so the census matches it) and a
#: ``Z`` state (so the zombie question has to be asked about a pid signal-0 still
#: finds — the R1-3 case, where a corpse must not read live). ``OTHER_LIVE`` has an
#: ordinary argv (so the census must skip it) and a single-digit day in its
#: ``lstart``, which is the two-space rendering ``ps`` pads and the case a naive
#: split gets wrong. ``GONE_PID`` is a runtime whose process is not there at all.
MACHINE: dict[int, dict[str, str]] = {
    MINE: {
        "ppid": "1",
        "etime": "02:00",
        "cpu": "0:01.00",
        "state": "Z",
        "lstart": "Mon Sep  8 11:27:42 2026",
        "command": f"/usr/bin/python3 -P -m {reclaim.RUNTIME_MODULE}",
    },
    OTHER_LIVE: {
        "ppid": "1",
        "etime": "01-04:05:06",
        "cpu": "0:00.50",
        "state": "Ss",
        "lstart": "Wed Oct 21 09:00:01 2026",
        "command": "/usr/bin/some-other-program --with an argument",
    },
    GONE_PID: {
        "ppid": str(OTHER_LIVE),
        "etime": "10:00",
        "cpu": "0:02.00",
        "state": "Z",
        "lstart": "Fri Sep 25 12:00:00 2026",
        "command": f"/usr/bin/python3 -P -m {reclaim.RUNTIME_MODULE}",
    },
}


def census_text(machine: dict[int, dict[str, str]] | None = None) -> str:
    """The machine in ``ps -eo pid=,ppid=,etime=,time=,command=`` layout."""
    return "\n".join(
        f"{pid:>6} {row['ppid']:>6} {row['etime']:>10} {row['cpu']:>9} {row['command']}"
        for pid, row in (machine or MACHINE).items()
    )


def table_text(machine: dict[int, dict[str, str]] | None = None) -> str:
    """The same machine in ``PROCESS_TABLE_KEYWORDS`` layout, as ``ps`` pads it.

    The padding is the point of the fixture, not decoration: ``lstart`` is a
    five-token column whose day is SPACE-padded for a single-digit date, and it
    sits BEFORE ``command`` (which must be last, because it is the only column that
    may itself contain spaces).
    """
    return "\n".join(
        f"{pid:>6} {row['ppid']:>6} {row['etime']:>10} {row['cpu']:>9} "
        f"{row['state']:<3} {row['lstart']:<28} {row['command']}"
        for pid, row in (machine or MACHINE).items()
    )


def samples_text(machine: dict[int, dict[str, str]], pids) -> str:
    """The machine in the per-list ``pid=,state=,lstart=`` layout."""
    return "\n".join(
        f"{pid:>6} {machine[pid]['state']:<3} {machine[pid]['lstart']}"
        for pid in pids
        if pid in machine
    )


def fixture_table(machine: dict[int, dict[str, str]] | None = None) -> ProcessTable:
    return ProcessTable(rows=parse_process_table(table_text(machine)))


def stub_platform_probe(monkeypatch: pytest.MonkeyPatch, machine=None) -> None:
    """Answer the platform's per-pid batch from the fixture, on any host.

    Both spellings are replaced because which one runs is a property of the HOST
    (``/proc`` on Linux, ``/bin/ps -p`` on macOS), and this module is about the
    values the two PATHS produce, not about which host is running it.
    """
    table = fixture_table(machine)

    def probe(pids):
        return {int(pid): table.sample(int(pid)) for pid in pids if int(pid) in table.rows}

    monkeypatch.setattr(procstate, "_ps_samples", probe)
    monkeypatch.setattr(procstate, "_proc_samples", probe)


def census_runner(machine=None):
    """A ``run`` that answers the census invocation from the fixture."""

    def run(command, timeout):
        return census_text(machine)

    return run


def record(pid: int, *, session_id: str = "s1", port: int = 5000, **extra) -> SessionRecord:
    payload = {
        "pid": pid,
        "kind": "daemon",
        "session_id": session_id,
        "conversation_name": "synthetic",
        "cwd": "/tmp/synthetic",
        "model_label": "synthetic-model",
        "control_port": port,
        "control_key": "k",
        "heartbeat_at": time.time(),
        "version": "0.56.11",
        "install_root": "/opt/local-operator",
    }
    payload.update(extra)
    return SessionRecord.from_json(payload)


def fleet_for(root: Path) -> Fleet:
    """A record for one row, a boot record for the census-only row, and sockets.

    The boot record is what gives the SECOND source a row at all (a runtime with no
    discovery record is the population this roster exists to name), so the two arms
    are compared over rows that came from all three sources rather than over one
    shape three times.
    """
    return Fleet(
        root=root,
        records={MINE: record(MINE, session_id="s-mine", port=5001)},
        boots={
            OTHER_LIVE: {
                "pid": OTHER_LIVE,
                "session_id": "s-other",
                "build_version": "0.56.11",
                "install_root": "/opt/local-operator",
                "started_at": time.time() - 30.0,
            }
        },
        viewers=[],
        sockets=SocketEvidence(ports={GONE_PID: 5002}, available=True),
        own_pids=frozenset({MINE}),
    )


def env_of(root: Path):
    """One environment reader for both arms: the last source for a record-less row."""
    text = f" HOME={root} LOP_MOBILE_CHILD_RESUME=s-env "

    def read(pid: int) -> str:
        return text

    return read


# ---------------------------------------------------------------------------
# The parse level: one machine, two `ps` layouts, equal values
# ---------------------------------------------------------------------------


def test_the_merged_read_parses_the_same_state_and_start_time_as_the_per_list_probe() -> None:
    """Same process, two layouts: the sample must be equal FIELD FOR FIELD.

    This is the half of the equivalence that cannot be a sampling artifact — both
    strings describe one instant, because both are derived from one fixture. It
    covers the two renderings a positional parse gets wrong: a state padded to its
    column (``Z``, ``Ss``), and an ``lstart`` whose day is space-padded because it
    is a single digit (``Mon Sep  8``).
    """
    per_list = procstate._samples_from_lines(samples_text(MACHINE, sorted(MACHINE)))
    table = fixture_table()
    assert len(per_list) == len(MACHINE), "the fixture must produce samples to compare"
    for pid, sample in per_list.items():
        assert sample == table.sample(pid)
        assert sample.birth == " ".join(MACHINE[pid]["lstart"].split())
        assert sample.zombie is MACHINE[pid]["state"].startswith("Z")


def test_a_row_the_merged_read_cannot_parse_is_absent_rather_than_misread() -> None:
    """A torn or unexpected rendering degrades to "not in the table".

    Absence is the direction every caller of this table already fails in — a failed
    probe answers nothing and each reader has its own fail-closed rule for that —
    so a ``ps`` that printed something this parser does not understand must not be
    able to feed a half-parsed row to a verdict.
    """
    rows = parse_process_table(
        "\n".join(
            [
                "     1     0 03-22:11:52  70:55.00 Ss Mon Sep 21 11:27:42 2026 /sbin/launchd",
                "not a ps line at all",
                "   123     1  00:01     0:00.01 S",  # truncated: no lstart, no command
                "",
            ]
        )
    )
    assert set(rows) == {1}
    assert rows[1].ppid == 0
    assert rows[1].command == "/sbin/launchd"
    assert rows[1].lstart == "Mon Sep 21 11:27:42 2026"


def test_the_merged_read_keeps_the_command_column_verbatim() -> None:
    """The census matches the spawn contract against this text, so spacing counts.

    ``ps`` pads a non-final column, which is why ``command`` is last in
    :data:`procstate.PROCESS_TABLE_KEYWORDS`; and the parser must not collapse the
    runs of whitespace INSIDE an argv, because the census reader it replaces does
    not (it takes the rest of the line with ``split(None, 4)``). Both are asserted
    against that reader directly, so the two cannot drift.
    """
    command = "/bin/sh -c 'a  b   c'"
    row = parse_process_table(
        f"    42     1  00:01     0:00.01 S  Fri Sep 25 12:00:00 2026 {command}"
    )[42]
    assert row.command == command
    assert reclaim.parse_process_row(f"    42     1  00:01  0:00.01 {command}") == (
        reclaim._runtime_from_fields("42", "1", "00:01", "0:00.01", command)
    )


def test_the_two_census_sources_produce_equal_candidates() -> None:
    """``runtime_processes`` has two sources; they must agree about a RUNTIME.

    The fields it fills are the ones the sweep's ladder reads (``parent_pid``,
    ``age_s``, ``cpu_s``) plus the argv the spawn contract matches, and equality is
    asserted on the dataclass — so a conversion that drifted on one path (an
    ``etime_seconds`` applied to a different column, say) fails here rather than in
    a verdict about a live process.
    """
    from_invocation = sorted(reclaim.runtime_processes(run=census_runner()), key=lambda r: r.pid)
    with procstate.process_table_scope(fixture_table()):
        from_table = sorted(reclaim.runtime_processes(), key=lambda r: r.pid)
    assert from_table == from_invocation
    assert [row.pid for row in from_table] == [MINE, GONE_PID]
    assert from_table[0].parent_pid == 1
    assert from_table[0].age_s == 120.0
    assert from_table[0].command == MACHINE[MINE]["command"]


# ---------------------------------------------------------------------------
# The value level: one machine, two compositions, equal rows
# ---------------------------------------------------------------------------


def test_the_batched_composition_produces_the_same_rows_as_the_per_pid_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE EQUIVALENCE. One fixture, two compositions, every field equal.

    Both arms are handed the same fleet, the same environment reader and the same
    connect stub; they differ ONLY in where the process table comes from. Arm B is
    the code that shipped before the batching — a census ``run`` and the platform's
    own per-pid probe for the zombie batch. Arm A is one table read answering all of
    them, which the composition reaches because it was NOT handed a census.
    """
    common = {
        "fleet": fleet_for(tmp_path),
        "env_of": env_of(tmp_path),
        "connect": lambda port, timeout: True,
        "now": time.time(),
    }

    # ARM B — the per-pid readers, each stubbed from the SAME fixture.
    stub_platform_probe(monkeypatch)
    census: list[RuntimeProcess] = reclaim.runtime_processes(run=census_runner())
    assert census, "the census fixture must match at least one runtime"
    per_pid = build_roster(tmp_path, processes=census, **common)

    # ARM A — the table answers the census and every zombie batch instead.
    monkeypatch.setattr(procstate, "_ps_samples", lambda pids: pytest.fail("forked ps"))
    monkeypatch.setattr(procstate, "_proc_samples", lambda pids: pytest.fail("read /proc"))
    with procstate.process_table_scope(fixture_table()):
        batched = build_roster(tmp_path, **common)

    # "EVERY FIELD ANY CALLER READS" is the wire row: pin the key set first, so a
    # field added to RosterRow without being compared here fails this module.
    assert set(batched[0].to_json()) == set(RosterRow.__dataclass_fields__)
    assert [row.to_json() for row in batched] == [row.to_json() for row in per_pid]
    assert batched == per_pid

    # ...and the comparison is not vacuous. Both arms really did produce rows from
    # all three sources, so a path that silently dropped one could not pass this.
    # Rows are sorted by pid, and MINE's parent may sort either side of it.
    assert [row.pid for row in batched] == sorted({MINE, OTHER_LIVE, GONE_PID})
    by_pid = {row.pid: row for row in batched}
    assert [by_pid[pid].has_record for pid in (MINE, OTHER_LIVE, GONE_PID)] == [True, False, False]
    assert [by_pid[pid].has_boot_record for pid in (MINE, OTHER_LIVE, GONE_PID)] == [
        False,
        True,
        False,
    ]
    # MINE is a corpse in the fixture and ALIVE to signal-0 (the R1-3 case), and
    # GONE_PID is not there at all — the two must not read the same.
    assert by_pid[MINE].state == "stale"
    assert by_pid[GONE_PID].reachability == roster.REACHABLE_GONE


def test_the_table_answers_the_zombie_question_exactly_as_the_per_pid_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``zombie_states`` in and out of a scope: equal verdicts, one fork vs none."""
    monkeypatch.setattr(procstate, "_ps_samples", lambda pids: pytest.fail("forked ps"))
    monkeypatch.setattr(procstate, "_proc_samples", lambda pids: pytest.fail("read /proc"))
    population = sorted(MACHINE)
    with procstate.process_table_scope(fixture_table()):
        batched = procstate.zombie_states(population)
        # A pid the table does not name is ABSENT, exactly as a pid the per-list
        # probe does not report is: the caller keeps its own rule for absence.
        assert procstate.zombie_states([GONE_PID + 1]) == {}
    monkeypatch.undo()
    stub_platform_probe(monkeypatch)
    assert batched == procstate.zombie_states(population)
    assert batched[MINE] is True and batched[OTHER_LIVE] is False


def test_process_samples_takes_the_table_only_where_lstart_is_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The birth token is served from the table ONLY under the table's own scheme.

    On a ``/proc`` host the token is a tick count and its scheme tag says so;
    serving a rendered start time under that tag would make two samples of the SAME
    process compare UNEQUAL, which is the direction that lets a second runtime take
    a live transcript. So the platform's own probe wins there, and only the zombie
    question is batched.
    """
    if procstate.is_windows():
        pytest.skip("no process probe on Windows")
    table = fixture_table()
    platform = {MINE: ProcessSample(zombie=False, birth="a-platform-token")}
    monkeypatch.setattr(procstate, "_ps_samples", lambda pids: platform)
    monkeypatch.setattr(procstate, "_proc_samples", lambda pids: platform)
    with procstate.process_table_scope(table):
        got = procstate.process_samples([MINE])
    if procstate.birth_scheme() == procstate.BIRTH_SCHEME_PS_LSTART:
        assert got == table.samples([MINE])
    else:
        assert got == platform


# ---------------------------------------------------------------------------
# The ancestry walk: same chain, without the fork per hop
# ---------------------------------------------------------------------------


def _chain_machine(pairs) -> dict[str, dict[str, str]]:
    return {
        str(pid): {
            "ppid": str(ppid),
            "etime": "1:00",
            "cpu": "0:00",
            "state": "S",
            "lstart": "Mon Sep  8 11:27:42 2026",
            "command": "x",
        }
        for pid, ppid in pairs
    }


def test_the_ancestry_chain_is_equal_from_the_table_and_from_ps_per_hop() -> None:
    """The exclusion set a sweep must never signal, both ways, including the stop.

    A SHORT chain is the dangerous direction — ``own_pids`` is the set a sweep may
    not touch — so the walk must stop in the same place on both paths: at the pid
    whose parent is 0.
    """
    machine = _chain_machine(((500, 400), (400, 300), (300, 1), (1, 0)))
    parents = {int(pid): int(row["ppid"]) for pid, row in machine.items()}
    asked: list[str] = []

    def per_hop(command, timeout):
        asked.append(" ".join(command))
        return f"{parents[int(command[-1])]}\n"

    assert reclaim.ancestor_pids(500, run=per_hop) == frozenset({500, 400, 300, 1})
    # One fork PER HOP, which is the cost this lane removes: five of them on the
    # reference host's chain, on every roster poll.
    assert asked == [
        "ps -o ppid= -p 500",
        "ps -o ppid= -p 400",
        "ps -o ppid= -p 300",
        "ps -o ppid= -p 1",
    ]

    with procstate.process_table_scope(fixture_table(machine)):
        assert reclaim.ancestor_pids(
            500, run=lambda *a: pytest.fail("forked ps inside a scope")
        ) == frozenset({500, 400, 300, 1})


def test_a_hop_the_table_cannot_answer_falls_back_to_ps_for_that_hop() -> None:
    """A TRUNCATED table must never shorten the chain — it degrades per hop.

    "I could not read that parent" is not "there is no parent": a table read that
    timed out half way, or a parent that is genuinely gone, must not become a
    shorter list of pids the sweep is allowed to signal.
    """
    truncated = _chain_machine(((500, 400),))
    parents = {500: 400, 400: 1, 1: 0}
    asked: list[str] = []

    def per_hop(command, timeout):
        asked.append(" ".join(command))
        return f"{parents[int(command[-1])]}\n"

    with procstate.process_table_scope(fixture_table(truncated)):
        assert reclaim.ancestor_pids(500, run=per_hop) == frozenset({500, 400, 1})
    assert asked == ["ps -o ppid= -p 400", "ps -o ppid= -p 1"]


# ---------------------------------------------------------------------------
# The structure: how many forks, which ones, and where the scope ends
# ---------------------------------------------------------------------------


class _Completed:
    """The one attribute ``_run_command`` and ``process_table`` read off a result."""

    def __init__(self, stdout: str = "") -> None:
        self.stdout = stdout


class ForkLog:
    """Every ``subprocess.run`` this composition makes, answered from the fixture.

    The instrument is the FORK, not the clock: what this lane claims is a count and
    a shape, and both are load-independent. It answers the reads the composition is
    allowed to make and records every argv, so an unexpected per-pid probe shows up
    as an assertion about the argv rather than as a millisecond nobody can reproduce
    on a busy host.
    """

    def __init__(self, machine=None) -> None:
        self.calls: list[list[str]] = []
        self._machine = machine or MACHINE

    def __call__(self, argv, *args, **kwargs):
        self.calls.append(list(argv))
        joined = " ".join(str(part) for part in argv)
        if argv[0].endswith("lsof"):
            return _Completed("")
        # THE MERGED TABLE FIRST, on the exact column list: the per-list probe's
        # column list is a SUBSTRING of it, so a looser test would answer the
        # per-list shape from the merged renderer and hide a stray fork.
        if procstate.PROCESS_TABLE_KEYWORDS in joined:
            return _Completed(table_text(self._machine))
        if "pid=,state=,lstart=" in joined:
            pids = [int(part) for part in joined.split("-p", 1)[1].replace(",", " ").split()]
            return _Completed(samples_text(self._machine, pids))
        if joined.startswith("ps -eo pid=,ppid=,etime=,time=,command="):
            return _Completed(census_text(self._machine))
        return _Completed("")

    def shapes(self) -> list[str]:
        return [" ".join(argv) for argv in self.calls]


def test_the_composition_forks_three_times_and_never_for_a_single_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE STRUCTURAL CLAIM: eleven forks become three, and none is per-pid.

    What is asserted is that the two per-pid families are GONE — ``ps -o ppid=``
    per ancestor hop, and ``ps -o pid=,state=,lstart=`` per record population — and
    that what is left is the three reads that are not per-pid: the socket table, the
    process table (which IS the census, with the state and start-time columns
    appended) and the environment dump.

    ``MINE`` is in the stub table with a parent chain on purpose: an ancestry walk
    that could not answer a hop would fall back to ``ps`` and be counted here.
    """
    machine = {int(pid): dict(row) for pid, row in MACHINE.items()}
    machine[MINE]["state"] = "Ss"  # alive, so MINE is a runtime the census keeps
    # pid 1 with ppid 0: the walk has to terminate INSIDE the table, or its last hop
    # falls back to `ps` and this test would be measuring the fallback.
    machine[1] = {
        "ppid": "0",
        "etime": "03-22:11:52",
        "cpu": "70:55.00",
        "state": "Ss",
        "lstart": "Mon Sep 21 11:27:42 2026",
        "command": "/sbin/launchd",
    }
    log = ForkLog(machine)
    monkeypatch.setattr(subprocess, "run", log)

    desktop_runtimes._collect(tmp_path, probe=False, budget_s=2.5)

    forked = "\n".join(log.shapes())
    assert "ps -o ppid=" not in forked, f"a per-hop ancestry fork survived:\n{forked}"
    assert "pid=,state=,lstart=" not in forked, f"a per-population zombie fork survived:\n{forked}"
    assert len(log.calls) == 3, f"expected three forks, got:\n{forked}"
    assert log.calls[0][0].endswith("lsof")
    assert log.calls[1][:2] == ["/bin/ps", "-e"]
    assert log.calls[1][-1] == procstate.PROCESS_TABLE_KEYWORDS
    assert log.calls[2][:2] == ["ps", "-Eww"]


def test_a_nested_scope_reads_the_table_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two scopes, one read.

    The route wraps both of its calls and ``build_roster`` wraps itself, so the
    nesting is the normal case rather than an exotic one: it must not cost a second
    fork, and the inner scope must see the OUTER table rather than sampling a second
    instant.
    """
    reads: list[int] = []

    def counted(**kwargs):
        reads.append(1)
        return fixture_table()

    monkeypatch.setattr(procstate, "process_table", counted)
    with procstate.process_table_scope():
        outer = procstate.current_process_table()
        assert outer is not None
        with procstate.process_table_scope():
            assert procstate.current_process_table() is outer
        assert procstate.current_process_table() is outer
    assert reads == [1]
    assert procstate.current_process_table() is None


def test_a_composition_given_its_own_answers_never_reads_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read is on DEMAND, so injecting a census costs nothing.

    This is what keeps the per-pid path reachable at all — and it is a failure-mode
    rule rather than a cost one: the questions that used to be answered by a short
    ``ps -p <list>`` must not be able to force a whole-machine read, which would
    replace a 1 s bound with a 5 s one on a machine whose table read is wedged.
    """
    monkeypatch.setattr(procstate, "process_table", lambda **kwargs: pytest.fail("read the table"))
    stub_platform_probe(monkeypatch)
    census = reclaim.runtime_processes(run=census_runner())
    rows = build_roster(
        tmp_path,
        fleet=fleet_for(tmp_path),
        processes=census,
        env_of=env_of(tmp_path),
        connect=lambda port, timeout: True,
    )
    assert [row.pid for row in rows] == sorted({MINE, OTHER_LIVE, GONE_PID})


def test_an_empty_table_is_not_evidence_that_nothing_is_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed table read falls back to the per-pid readers, fork for fork.

    The empty table is what a Windows host and a timed-out ``ps`` both produce, and
    reading it as "this machine has no processes" is the failure mode pinned here:
    the census must take its own read, the ancestry walk must ask per hop, and the
    zombie question must go back to its own probe rather than answering "nobody is
    a corpse" from a read that returned nothing.
    """
    asked: list[str] = []

    def census_run(command, timeout):
        asked.append("census")
        return census_text()

    def per_hop(command, timeout):
        asked.append(" ".join(command))
        return "0\n"

    if procstate.is_windows():
        pytest.skip("no process probe on Windows")
    monkeypatch.setattr(
        procstate, "_ps_samples", lambda pids: asked.append("per-list probe") or {}
    )
    monkeypatch.setattr(
        procstate, "_proc_samples", lambda pids: asked.append("per-list probe") or {}
    )

    with procstate.process_table_scope(ProcessTable(rows={})) as active:
        assert active is None  # an empty read is not a table
        assert reclaim.runtime_processes(run=census_run)
        assert reclaim.ancestor_pids(MINE, run=per_hop) == frozenset({MINE})
        assert procstate.zombie_states([GONE_PID]) == {}
    assert asked == ["census", f"ps -o ppid= -p {MINE}", "per-list probe"]


def test_a_scope_is_per_thread_and_restores_its_outer_one() -> None:
    """The scope must not leak out of the worker thread that opened it.

    The route hands the composition to ``asyncio.to_thread`` while the daemon keeps
    serving other requests on the event loop, so a PROCESS-global table would let
    one composition's snapshot answer another request's liveness question. Both
    halves of that are asserted here — the other thread sees no scope, and a raise
    inside a nested scope restores the outer one rather than stranding a table.
    """
    import threading

    seen: list[object] = []

    def reader() -> None:
        seen.append(procstate.current_process_table())

    with procstate.process_table_scope(fixture_table()) as outer:
        assert outer is not None
        thread = threading.Thread(target=reader)
        thread.start()
        thread.join()
        with pytest.raises(RuntimeError):
            with procstate.process_table_scope():
                raise RuntimeError("boom")
        assert procstate.current_process_table() is outer
    assert seen == [None]
    assert procstate.current_process_table() is None


def test_the_table_timeout_is_the_census_timeout() -> None:
    """One number, two spellings, pinned — because they mean the same bound.

    ``procstate`` is a leaf module with no local imports, so it cannot read
    ``reclaim.CENSUS_TIMEOUT_S``; the route's stated ceiling and the supervisor's
    slice both assume a wedged ``ps`` costs this much and no more. A drift between
    the two is a docstring that quietly stops being true.
    """
    assert procstate.PROCESS_TABLE_TIMEOUT_S == reclaim.CENSUS_TIMEOUT_S


# ---------------------------------------------------------------------------
# The host itself, and the route
# ---------------------------------------------------------------------------


def test_the_batched_read_sees_a_real_process_the_way_the_census_does() -> None:
    """One live check, because no fixture can prove the ARGV is right.

    Everything above stubs ``ps``. A column list that ``ps`` rejects, or a selection
    flag left off — ``ps`` with only ``-o`` lists the CURRENT TERMINAL, which is one
    or two rows for a machine of a thousand processes and reads exactly like a
    machine running nothing — would pass every synthetic test in this module.

    The comparison is over a CHILD THIS TEST OWNS, and that is deliberate rather
    than convenient: the table read and the census read are two samples of a moving
    table, so a runtime that starts or exits between them is in one and not the
    other. ``ppid`` and ``command`` are the two fields the census's interpretation
    rests on and the two that cannot move while the child is alive, so they are
    compared exactly; ``etime`` and CPU tick between any two reads and are covered
    by the synthetic fixtures above instead.
    """
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        table = procstate.process_table()
        # `-e`: without it this is the two processes on the current terminal.
        assert len(table.rows) > 1, "the table selected almost nothing; check `-e`"
        assert table.ppid_of(os.getpid()) == os.getppid()
        assert child.pid in table.rows, "the table missed a process this test owns"

        census = {
            row.pid: row
            for row in reclaim.runtime_processes(
                run=lambda command, timeout: subprocess.run(
                    command, capture_output=True, text=True, check=False
                ).stdout
            )
        }
        # The census keeps only the spawn contract, so the child is compared
        # through the raw rendering both readers parse.
        output = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,etime=,time=,command="],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        raw = {
            int(fields[0]): fields
            for fields in (line.split(None, 4) for line in output.splitlines())
            if len(fields) == 5 and fields[0].isdigit()
        }
        assert child.pid in raw
        assert table.rows[child.pid].ppid == int(raw[child.pid][1])
        assert table.rows[child.pid].command == raw[child.pid][4]
        # ...and the two readers agree about every live runtime BOTH of them saw.
        # The intersection is the honest comparison: a runtime that started between
        # the two reads is legitimately in the census alone.
        assert census, "no session runtime is live on this host; the comparison would be empty"
        for pid, row in census.items():
            if pid in table.rows:
                assert table.rows[pid].ppid == row.parent_pid
                assert table.rows[pid].command == row.command
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_the_route_cannot_tell_the_batching_happened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HTTP half: same rows, same sources, with the scope and without it.

    The comparison is against the SAME composition with the scope disabled — the
    code that shipped before this lane. A difference here would be a value the
    batching changed, which is the one thing the lane claims it does not do.
    """
    store = tmp_path / "run" / "mobile"
    store.mkdir(parents=True, exist_ok=True)
    (store / f"{MINE}.json").write_text(json.dumps(record(MINE, port=5001).to_json()))
    log = ForkLog()
    monkeypatch.setattr(subprocess, "run", log)
    monkeypatch.setattr(
        roster, "socket_evidence", lambda: SocketEvidence(ports={MINE: 5001}, available=True)
    )
    monkeypatch.setattr(roster, "_connect_ok", lambda port, timeout: True)

    batched = desktop_runtimes._collect(tmp_path, probe=True, budget_s=2.5)
    monkeypatch.setattr(roster, "process_table_scope", lambda: contextlib.nullcontext())
    unbatched = desktop_runtimes._collect(tmp_path, probe=True, budget_s=2.5)

    assert batched.model_dump() == unbatched.model_dump()
    assert batched.count == 2  # MINE and the boot-record pid; GONE_PID is not in the census
    assert batched.sources  # ...and it says which of the four sources it used
