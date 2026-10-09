"""Owned compiler-process cleanup, extracted unchanged from the reviewed guard.

No machine-specific paths or run authority are imported by the figure writer.
"""

import os
import signal
import subprocess

import psutil

TERM_GRACE_SECONDS = 2.0


def _identity(proc):
    process = psutil.Process(proc.pid)
    return proc, process, (proc.pid, process.create_time())


def identity_matches(pid, identity):
    try:
        return (pid, psutil.Process(pid).create_time()) == identity
    except psutil.Error:
        return False


def terminate_owned_child(proc, identity):
    # Popen owns this unreaped leader even if psutil identity acquisition failed.
    if proc.poll() is None:
        if identity is not None and not identity_matches(proc.pid, identity):
            raise RuntimeError("Owned analysis child identity changed.")
        if os.getpgid(proc.pid) != proc.pid:
            proc.terminate()
            proc.wait(timeout=TERM_GRACE_SECONDS)
            raise RuntimeError("Analysis child is not its own process-group leader.")
    elif psutil.pid_exists(proc.pid):
        # A reaped leader's numeric PID may have been recycled. Never signal a
        # new session identified only by that old integer.
        if identity is None or not identity_matches(proc.pid, identity):
            raise RuntimeError("Reaped analysis PID was recycled; group cleanup refused.")
    # Also clean unexpected descendants after a fast leader exit. The new session
    # belongs to this launch; no user-selected PID or process group is accepted.
    members = []
    for p in psutil.process_iter():
        try:
            if os.getsid(p.pid) == proc.pid:
                members.append(p)
        except ProcessLookupError:
            continue
    if members:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=TERM_GRACE_SECONDS)
    _, alive = psutil.wait_procs(members, timeout=TERM_GRACE_SECONDS)
    for p in alive:
        if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
            p.kill()  # psutil verifies PID/create-time before sending a signal.
    _, alive = psutil.wait_procs(alive, timeout=TERM_GRACE_SECONDS)
    if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in alive):
        raise RuntimeError("Analysis child cleanup failed.")
    return any(p.pid != proc.pid for p in members)
