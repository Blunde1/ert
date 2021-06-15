import copy
from datetime import datetime as dt
from unittest.mock import Mock

import pytest

import ert_shared.ensemble_evaluator.entity.identifiers as ids
from ert_shared.ensemble_evaluator.entity.identifiers import (
    CURRENT_MEMORY_USAGE,
    MAX_MEMORY_USAGE,
)
from ert_shared.ensemble_evaluator.entity.snapshot import (
    Job,
    Realization,
    Snapshot,
    SnapshotBuilder,
    SnapshotDict,
    Step,
)
from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_STARTED,
    JOB_STATE_START,
    REALIZATION_STATE_UNKNOWN,
    STEP_STATE_UNKNOWN,
)


@pytest.fixture()
def full_snapshot() -> Snapshot:
    real = Realization(
        status=REALIZATION_STATE_UNKNOWN,
        active=True,
        steps={
            "0": Step(
                status="",
                jobs={
                    "0": Job(
                        start_time=dt.now(),
                        end_time=dt.now(),
                        name="poly_eval",
                        status=JOB_STATE_START,
                        error="error",
                        stdout="std_out_file",
                        stderr="std_err_file",
                        data={
                            CURRENT_MEMORY_USAGE: "123",
                            MAX_MEMORY_USAGE: "312",
                        },
                    ),
                    "1": Job(
                        start_time=dt.now(),
                        end_time=dt.now(),
                        name="poly_postval",
                        status=JOB_STATE_START,
                        error="error",
                        stdout="std_out_file",
                        stderr="std_err_file",
                        data={
                            CURRENT_MEMORY_USAGE: "123",
                            MAX_MEMORY_USAGE: "312",
                        },
                    ),
                },
            )
        },
    )
    snapshot = SnapshotDict(
        status=ENSEMBLE_STATE_STARTED,
        reals={},
    )
    for i in range(0, 100):
        snapshot.reals[str(i)] = copy.deepcopy(real)

    return Snapshot(snapshot.dict())


@pytest.fixture()
def large_snapshot() -> Snapshot:
    builder = SnapshotBuilder().add_step(step_id="0", status=STEP_STATE_UNKNOWN)
    for i in range(0, 150):
        builder.add_job(
            step_id="0",
            job_id=str(i),
            name=f"job_{i}",
            data={ids.MAX_MEMORY_USAGE: 1000, ids.CURRENT_MEMORY_USAGE: 500},
            status=JOB_STATE_START,
            stdout=f"job_{i}.stdout",
            stderr=f"job_{i}.stderr",
            start_time=dt(1999, 1, 1).isoformat(),
            end_time=dt(2019, 1, 1).isoformat(),
        )
    real_ids = [str(i) for i in range(0, 150)]
    return builder.build(real_ids, REALIZATION_STATE_UNKNOWN)


@pytest.fixture()
def small_snapshot() -> Snapshot:
    builder = SnapshotBuilder().add_step(step_id="0", status=STEP_STATE_UNKNOWN)
    for i in range(0, 2):
        builder.add_job(
            step_id="0",
            job_id=str(i),
            name=f"job_{i}",
            data={ids.MAX_MEMORY_USAGE: 1000, ids.CURRENT_MEMORY_USAGE: 500},
            status=JOB_STATE_START,
            stdout=f"job_{i}.stdout",
            stderr=f"job_{i}.stderr",
            start_time=dt(1999, 1, 1).isoformat(),
            end_time=dt(2019, 1, 1).isoformat(),
        )
    real_ids = [str(i) for i in range(0, 5)]
    return builder.build(real_ids, REALIZATION_STATE_UNKNOWN)


@pytest.fixture
def runmodel() -> Mock:
    brm = Mock()
    brm.get_runtime = Mock(return_value=100)
    brm.hasRunFailed = Mock(return_value=False)
    brm.getFailMessage = Mock(return_value="")
    brm.support_restart = True
    return brm


@pytest.fixture
def active_realizations() -> Mock:
    active_reals = Mock()
    active_reals.count = Mock(return_value=10)
    return active_reals
