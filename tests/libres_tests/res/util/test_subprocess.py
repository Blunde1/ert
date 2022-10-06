import os
from subprocess import PIPE, Popen, TimeoutExpired

import pytest

from ert._c_wrappers.util.subprocess import await_process_tee


def _find_system_pipe_max_size():
    """This method finds the limit of the system pipe-buffer which
    might be taken into account when using subprocesses with pipes."""
    # pylint: disable=consider-using-with
    p = Popen(["dd", "if=/dev/zero", "bs=1"], stdin=PIPE, stdout=PIPE)
    try:
        p.wait(timeout=1)
    except TimeoutExpired:
        p.kill()
        return len(p.stdout.read())

    return None


_maxBytes = _find_system_pipe_max_size() - 1


@pytest.mark.usefixtures("use_tmpdir")
def test_await_process_tee():
    with open("original", "wb") as fh:
        fh.write(bytearray(os.urandom(_maxBytes)))

    with open("a", "wb") as a_fh, open("b", "wb") as b_fh:
        # pylint: disable=consider-using-with
        # await_process_tee() ensures the process is terminated
        process = Popen(["/bin/cat", "original"], stdout=PIPE)
        await_process_tee(process, a_fh, b_fh)

    with open("a", "rb") as f:
        a_content = f.read()
    with open("b", "rb") as f:
        b_content = f.read()
    with open("original", "rb") as f:
        original_content = f.read()

    assert process.stdout.closed
    assert a_content == original_content
    assert b_content == original_content


@pytest.mark.usefixtures("use_tmpdir")
def test_await_process_finished_tee():
    with open("original", "wb") as fh:
        fh.write(bytearray(os.urandom(_maxBytes)))

    with open("a", "wb") as a_fh, open("b", "wb") as b_fh:
        # pylint: disable=consider-using-with
        # await_process_tee() ensures the process is terminated
        process = Popen(["/bin/cat", "original"], stdout=PIPE)
        process.wait()
        await_process_tee(process, a_fh, b_fh)

    with open("a", "rb") as f:
        a_content = f.read()
    with open("b", "rb") as f:
        b_content = f.read()
    with open("original", "rb") as f:
        original_content = f.read()

    assert process.stdout.closed
    assert a_content == original_content
    assert b_content == original_content
