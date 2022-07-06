import copy
from unittest.mock import patch
from datetime import datetime as dt
from ert.ensemble_evaluator import identifiers as ids
import pytest
from ert_gui.model.snapshot import SnapshotModel
from ert_gui.simulation.run_dialog import RunDialog
from ert.ensemble_evaluator.snapshot import (
    PartialSnapshot,
    Realization,
    SnapshotBuilder,
)
from ert.ensemble_evaluator import state
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.ensemble_evaluator.identifiers import (
    MAX_MEMORY_USAGE,
    CURRENT_MEMORY_USAGE,
)
from ert.ensemble_evaluator.snapshot import (
    Job,
    Realization,
    Snapshot,
    SnapshotBuilder,
    SnapshotDict,
    Step,
)
from ert.ensemble_evaluator.state import (
    ENSEMBLE_STATE_STARTED,
    JOB_STATE_FAILURE,
    JOB_STATE_FINISHED,
    JOB_STATE_RUNNING,
    JOB_STATE_START,
    REALIZATION_STATE_FAILED,
    REALIZATION_STATE_FINISHED,
    REALIZATION_STATE_PENDING,
    REALIZATION_STATE_RUNNING,
    REALIZATION_STATE_UNKNOWN,
    REALIZATION_STATE_WAITING,
    STEP_STATE_UNKNOWN,
)

from qtpy.QtCore import Qt


def test_success(runmodel, qtbot, mock_tracker):
    widget = RunDialog("poly.ert", runmodel)
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert_gui.simulation.run_dialog.EvaluatorTracker") as tracker:
        tracker.return_value = mock_tracker([EndEvent(failed=False, failed_msg="")])
        widget.startSimulation()

    with qtbot.waitExposed(widget, timeout=30000):
        qtbot.waitUntil(lambda: widget._total_progress_bar.value() == 100)
        assert widget.done_button.isVisible()
        assert widget.done_button.text() == "Done"


def test_large_snapshot(runmodel, large_snapshot, qtbot, mock_tracker):
    widget = RunDialog("poly.ert", runmodel)
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert_gui.simulation.run_dialog.EvaluatorTracker") as tracker:
        iter_0 = FullSnapshotEvent(
            snapshot=large_snapshot,
            phase_name="Foo",
            current_phase=0,
            total_phases=1,
            progress=0.5,
            iteration=0,
            indeterminate=False,
        )
        iter_1 = FullSnapshotEvent(
            snapshot=large_snapshot,
            phase_name="Foo",
            current_phase=0,
            total_phases=1,
            progress=0.5,
            iteration=1,
            indeterminate=False,
        )
        tracker.return_value = mock_tracker(
            [iter_0, iter_1, EndEvent(failed=False, failed_msg="")]
        )
        widget.startSimulation()

    with qtbot.waitExposed(widget, timeout=30000):
        qtbot.waitUntil(lambda: widget._total_progress_bar.value() == 100, timeout=5000)
        qtbot.mouseClick(widget.show_details_button, Qt.LeftButton)
        qtbot.waitUntil(lambda: widget._tab_widget.count() == 2, timeout=5000)


@pytest.mark.parametrize(
    "events,tab_widget_count",
    [
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            name="job_0",
                            data={},
                            status=state.JOB_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    iteration=0,
                    indeterminate=False,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder().build(
                            [], status=state.REALIZATION_STATE_FINISHED
                        )
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    iteration=0,
                    indeterminate=False,
                ),
                EndEvent(failed=False, failed_msg=""),
            ],
            1,
            id="real_less_partial",
        ),
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            name="job_0",
                            data={
                                ids.MAX_MEMORY_USAGE: 1000,
                                ids.CURRENT_MEMORY_USAGE: 500,
                            },
                            status=state.JOB_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    iteration=0,
                    indeterminate=False,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_SUCCESS)
                        .build(["0"], status=state.REALIZATION_STATE_FINISHED)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    iteration=0,
                    indeterminate=False,
                ),
                EndEvent(failed=False, failed_msg=""),
            ],
            1,
            id="jobless_partial",
        ),
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            name="job_0",
                            data={},
                            status=state.JOB_STATE_START,
                        )
                        .add_job(
                            step_id="0",
                            job_id="1",
                            index="1",
                            name="job_1",
                            data={},
                            status=state.JOB_STATE_START,
                        )
                        .build(["0", "1"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    iteration=0,
                    indeterminate=False,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_SUCCESS)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            status=state.JOB_STATE_FINISHED,
                            name="job_0",
                            data={},
                        )
                        .build(["1"], status=state.REALIZATION_STATE_RUNNING)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    iteration=0,
                    indeterminate=False,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_FAILURE)
                        .add_job(
                            step_id="0",
                            job_id="1",
                            index="1",
                            status=state.JOB_STATE_FAILURE,
                            name="job_1",
                            data={},
                        )
                        .build(["0"], status=state.REALIZATION_STATE_FAILED)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    iteration=0,
                    indeterminate=False,
                ),
                EndEvent(failed=False, failed_msg=""),
            ],
            1,
            id="two_job_updates_over_two_partials",
        ),
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            name="job_0",
                            data={},
                            status=state.JOB_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    iteration=0,
                    indeterminate=False,
                ),
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            name="job_0",
                            data={},
                            status=state.JOB_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    iteration=1,
                    indeterminate=False,
                ),
                EndEvent(failed=False, failed_msg=""),
            ],
            2,
            id="two_iterations",
        ),
    ],
)
def test_run_dialog(events, tab_widget_count, runmodel, qtbot, mock_tracker):
    widget = RunDialog("poly.ert", runmodel)
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert_gui.simulation.run_dialog.EvaluatorTracker") as tracker:
        tracker.return_value = mock_tracker(events)
        widget.startSimulation()

    with qtbot.waitExposed(widget, timeout=30000):
        qtbot.mouseClick(widget.show_details_button, Qt.LeftButton)
        qtbot.waitUntil(
            lambda: widget._tab_widget.count() == tab_widget_count, timeout=5000
        )
        qtbot.waitUntil(widget.done_button.isVisible, timeout=5000)


@pytest.fixture
def js_case():
    things = []
    step = Step(status="")
    for j in range(70):
        step.jobs[str(j)] = Job(
            start_time=dt.now(),
            end_time=dt.now(),
            name="poly_eval",
            index=str(j),
            status=JOB_STATE_START,
            error="error",
            stdout="std_out_file",
            stderr="std_err_file",
            data={
                CURRENT_MEMORY_USAGE: j,
                MAX_MEMORY_USAGE: j,
            },
        )

    real = Realization(status=REALIZATION_STATE_UNKNOWN, active=True, steps={"0": step})
    snapshot = SnapshotDict(
        status=ENSEMBLE_STATE_STARTED,
        reals={},
    )
    for i in range(0, 200):
        snapshot.reals[str(i)] = copy.deepcopy(real)
    snapshot = Snapshot(snapshot.dict())
    things.append(snapshot)
    real_states = [
        REALIZATION_STATE_WAITING,
        REALIZATION_STATE_PENDING,
        REALIZATION_STATE_FAILED,
        REALIZATION_STATE_FINISHED,
        REALIZATION_STATE_RUNNING,
    ]
    job_states = [
        JOB_STATE_START,
        JOB_STATE_RUNNING,
        JOB_STATE_FAILURE,
        JOB_STATE_RUNNING,
        JOB_STATE_FINISHED,
    ]
    for i in range(4):
        partial = PartialSnapshot(snapshot)
        for r in range(200):
            partial.update_real(str(r), Realization(status=real_states[i]))
            for j in range(70):
                partial.update_job(str(r), "0", str(j), Job(status=job_states[i]))
        things.append(partial)
    return things


def test_js_bench(runmodel, js_case, qtbot, benchmark):
    widget = RunDialog("poly.ert", runmodel)
    widget.show()
    qtbot.addWidget(widget)
    qtbot.mouseClick(widget.show_details_button, Qt.LeftButton)

    model = widget._snapshot_model

    def target():
        for thing in js_case:
            if isinstance(thing, Snapshot):
                model._add_snapshot(SnapshotModel.prerender(thing), 0)
            else:
                model._add_partial_snapshot(SnapshotModel.prerender(thing), 0)

    benchmark(target)
