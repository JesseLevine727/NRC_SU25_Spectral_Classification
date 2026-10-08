"""T296 process transport adapter: trusted caller factory, no numerical code."""

import contextlib
import math
import multiprocessing
import threading
import time

_CTX = multiprocessing.get_context("spawn")
_PERIOD, _STALE = 0.5, 5.0


class ProcessWorkerError(RuntimeError):
    """Transport failure; the owning controller decides recovery."""


def _send_error(conn, exc, job):
    reply = {
        "type": "error",
        "error_type": type(exc).__name__,
        "message": str(exc),
        "job_id": job.get("job_id") if isinstance(job, dict) else None,
    }
    with contextlib.suppress(BaseException):
        conn.send(reply)


def _publish(engine, telemetry):
    with telemetry.get_lock():
        if telemetry[3] < 0:
            return  # A failed observation is latched, never replaced by a later healthy sample.
    try:
        sample = engine.telemetry()
        alloc = float(sample["allocated_gpu_bytes"])
        reserved = float(sample["reserved_gpu_bytes"])
        if not (
            math.isfinite(alloc)
            and math.isfinite(reserved)
            and alloc >= 0
            and reserved >= 0
            and alloc.is_integer()
            and reserved.is_integer()
        ):
            raise ValueError("invalid telemetry")
        stamp, health = time.monotonic(), 1.0
    except BaseException:
        alloc, reserved, stamp, health = 0.0, 0.0, time.monotonic(), -1.0
    with telemetry.get_lock():
        telemetry[0], telemetry[1], telemetry[2], telemetry[3] = stamp, alloc, reserved, health


def _sample_loop(engine, telemetry, stop, first):
    _publish(engine, telemetry)
    first.set()
    while not stop.wait(_PERIOD):
        _publish(engine, telemetry)


def _child(conn, factory, config, worker_id, kind, telemetry):
    engine, thread, stop = None, None, threading.Event()
    try:
        engine = factory(config, worker_id, kind)
        first = threading.Event()
        thread = threading.Thread(
            target=_sample_loop, daemon=True, args=(engine, telemetry, stop, first)
        )
        thread.start()
        first.wait()
        with telemetry.get_lock():
            if telemetry[3] < 0:
                raise ProcessWorkerError("initial telemetry unhealthy")
        conn.send({"type": "ready"})
        while True:
            msg = conn.recv()
            if msg is None:
                return
            job = msg.get("job") if isinstance(msg, dict) else None
            try:
                if (
                    not isinstance(msg, dict)
                    or msg.get("type") != "job"
                    or not isinstance(job, dict)
                ):
                    raise ProcessWorkerError("bad child protocol")
                with telemetry.get_lock():
                    healthy = telemetry[3]
                if healthy < 0:
                    raise ProcessWorkerError("telemetry unhealthy")
                conn.send({"type": "result", "result": engine.execute(job)})
            except BaseException as exc:
                _send_error(conn, exc, job)
                return
    except BaseException as exc:
        _send_error(conn, exc, None)
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=1.0)
        closer = getattr(engine, "close", None)
        if callable(closer):
            with contextlib.suppress(BaseException):
                closer()
        with contextlib.suppress(BaseException):
            conn.close()


class ProcessWorker:
    worker_id = property(lambda self: self._worker_id)
    kind = property(lambda self: self._kind)
    pid = property(lambda self: self._process.pid)

    def __init__(self, worker_id, kind, *, factory, config):
        if kind not in ("CPU", "GPU") or not isinstance(worker_id, str) or not worker_id:
            raise ProcessWorkerError("invalid worker_id or kind")
        if not callable(factory):
            raise ProcessWorkerError("factory must be callable")
        self._worker_id, self._kind = worker_id, kind
        self._busy, self._ready, self._closed, self._current_id = False, False, False, None
        self._telemetry = _CTX.Array("d", 4, lock=True)
        self._conn, child_conn = _CTX.Pipe(duplex=True)
        self._process = _CTX.Process(
            target=_child,
            daemon=False,
            args=(child_conn, factory, config, worker_id, kind, self._telemetry),
        )
        self._process.start()
        child_conn.close()

    def ready(self):
        if self._closed:
            raise ProcessWorkerError("worker terminated")
        if self._ready:
            return True
        if self._conn.poll(0):
            msg = self._conn.recv()
            mtype = msg.get("type") if isinstance(msg, dict) else None
            if mtype == "ready":
                self._ready = True
                return True
            if mtype == "error":
                raise ProcessWorkerError(f"{msg.get('error_type')}: {msg.get('message')}")
            raise ProcessWorkerError("unexpected handshake message")
        if not self._process.is_alive():
            raise ProcessWorkerError("worker died during initialisation")
        return False

    def submit(self, job):
        if self._closed or not self._ready or self._busy:
            raise ProcessWorkerError("worker unavailable or busy")
        if not isinstance(job, dict) or job.get("job_id") is None:
            raise ProcessWorkerError("job must be a dict with an id")
        self._conn.send({"type": "job", "job": job})
        self._busy, self._current_id = True, job["job_id"]

    def poll(self):
        if self._closed:
            raise ProcessWorkerError("worker terminated")
        if not self._busy:
            if self._conn.poll(0):
                raise ProcessWorkerError("unexpected message while idle")
            return None
        if not self._conn.poll(0):
            if not self._process.is_alive():
                raise ProcessWorkerError("worker died before replying")
            return None
        msg = self._conn.recv()
        mtype = msg.get("type") if isinstance(msg, dict) else None
        if mtype == "error":
            raise ProcessWorkerError(f"{msg.get('error_type')}: {msg.get('message')}")
        result = msg.get("result") if isinstance(msg, dict) else None
        if (
            mtype != "result"
            or not isinstance(result, dict)
            or result.get("job_id") != self._current_id
        ):
            raise ProcessWorkerError("unexpected or mismatched result")
        self._busy, self._current_id = False, None
        return result

    def telemetry(self):
        if not self._ready or self._closed:
            raise ProcessWorkerError("telemetry unavailable before ready")
        lock = self._telemetry.get_lock()
        if not lock.acquire(timeout=0.25):
            raise ProcessWorkerError("telemetry lock unavailable")
        try:
            health, stamp = self._telemetry[3], float(self._telemetry[0])
            alloc, reserved = float(self._telemetry[1]), float(self._telemetry[2])
        finally:
            lock.release()
        if health != 1.0:
            raise ProcessWorkerError("telemetry unhealthy")
        age = time.monotonic() - stamp
        if not math.isfinite(stamp) or age < 0 or age > _STALE:
            raise ProcessWorkerError("telemetry stale or invalid")
        if not all(math.isfinite(v) and v.is_integer() and v >= 0 for v in (alloc, reserved)):
            raise ProcessWorkerError("telemetry value invalid")
        return {
            "timestamp": stamp,
            "allocated_gpu_bytes": int(alloc),
            "reserved_gpu_bytes": int(reserved),
        }

    def alive(self):
        return not self._closed and bool(self._process.is_alive())

    def terminate(self):
        if self._closed:
            return
        self._closed, self._busy = True, False
        process = self._process
        if process.is_alive():
            process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        if process.is_alive():
            raise ProcessWorkerError("owned worker could not be stopped")
        with contextlib.suppress(BaseException):
            self._conn.close()
