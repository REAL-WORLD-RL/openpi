from __future__ import annotations

"""
Libero websocket env server (RemoteEnv-compatible), but with episode/time-limit logic aligned
to `examples/libero/main.py`.

Key alignment to main.py:
- Uses per-suite `max_steps` (libero_spatial/object/goal/10/90) instead of relying on robosuite horizon.
- Performs `num_steps_wait` dummy steps during reset (so client policy steps start at t=0).
- Treats LIBERO `done=True` as success/termination; time-limit is reported as truncated=True.

Protocol compatibility:
- Speaks the `pi_link.remote_env.RemoteEnv` websocket protocol (handshake + reset/step/close).
- Supports optional session_id routing (fixed worker pool) so multiple RemoteEnv clients can connect.

This file intentionally reuses the existing server's structure with minimal changes, only swapping
the episode length logic to match `examples/libero/main.py`.
"""

import argparse
import asyncio
import logging
import math
import pathlib
import sys
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import numpy as np
import websockets
import websockets.asyncio.server as _server

# Allow running as a script: `python examples/libero/libero_env_server_mainlike.py`
# Add repo root to sys.path so local imports like `pi_link.*` work.
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from pi_link import msgpack_numpy  # noqa: E402
from pi_link.spaces import libero_default_space_specs  # noqa: E402

logger = logging.getLogger(__name__)


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    # Copied from robosuite, same as examples/libero/main.py
    quat = np.asarray(quat).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _get_libero_env(task: Any, resolution: int, seed: int):
    from libero.libero import get_libero_path  # imported lazily for clearer error messages
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: affects object positions even with fixed initial state
    return env, task_description


def _load_task_suite(task_suite_name: str):
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown task suite: {task_suite_name}. Options: {sorted(benchmark_dict.keys())}")
    return benchmark_dict[task_suite_name]()


def _max_steps_for_suite(task_suite_name: str) -> int:
    # Mirror `examples/libero/main.py` exactly.
    if task_suite_name == "libero_spatial":
        return 220  # longest training demo has 193 steps
    if task_suite_name == "libero_object":
        return 280  # longest training demo has 254 steps
    if task_suite_name == "libero_goal":
        return 300  # longest training demo has 270 steps
    if task_suite_name == "libero_10":
        return 520  # longest training demo has 505 steps
    if task_suite_name == "libero_90":
        return 400  # longest training demo has 373 steps
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _process_obs(raw_obs: dict, *, task_description: str, resize_size: int) -> dict:
    # Reuse openpi-client preprocessing utilities (same as main.py).
    try:
        from openpi_client import image_tools  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "缺少 openpi_client.image_tools。runtime 镜像里通常会有 openpi-client；"
            "如果你自己裁剪了依赖，请补上 openpi-client 或在这里实现等价的 resize/uint8。"
        ) from e

    # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))

    state = np.concatenate(
        (
            raw_obs["robot0_eef_pos"],
            _quat2axisangle(raw_obs["robot0_eef_quat"]),
            raw_obs["robot0_gripper_qpos"],
        )
    )

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }


########################################################################################
# Multiprocess worker (one env per worker)
########################################################################################


class _WorkerCrashed(RuntimeError):
    pass


def _now_s() -> float:
    return time.time()


def _best_effort_set_env_horizon(env: Any, horizon: int) -> Optional[int]:
    """Try to set horizon on common wrapped envs. Returns readback if possible."""
    horizon = int(horizon)
    # Minimal, predictable approach (avoid deep introspection / side effects):
    candidates = [env, getattr(env, "env", None), getattr(getattr(env, "env", None), "env", None)]
    for obj in candidates:
        if obj is None:
            continue
        try:
            if hasattr(obj, "horizon"):
                setattr(obj, "horizon", horizon)
        except Exception:  # noqa: BLE001
            pass
    for obj in candidates:
        if obj is None:
            continue
        try:
            if hasattr(obj, "horizon"):
                return int(getattr(obj, "horizon"))
        except Exception:  # noqa: BLE001
            continue
    return None


def _worker_loop(
    *,
    task_suite_name: str,
    task_id: int,
    seed: int,
    resize_size: int,
    num_steps_wait: int,
    resolution: int,
    max_steps_override: Optional[int],
    conn: Any,  # multiprocessing.connection.Connection
) -> None:
    """Worker process: owns one Libero env and handles reset/step."""
    task_suite = _load_task_suite(task_suite_name)
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = _get_libero_env(task, resolution, seed)

    # Episode length in *policy steps* (warmup already happens inside reset).
    max_steps = int(max_steps_override) if max_steps_override is not None else _max_steps_for_suite(task_suite_name)

    # Best-effort: keep underlying robosuite time limit safely above our own,
    # so the server controls truncation instead of robosuite raising.
    # We already do `num_steps_wait` steps during reset, which count internally.

    episode_idx = 0
    current_obs_raw: Optional[dict] = None
    episode_over = False
    steps_since_reset = 0  # policy steps (does not include warmup)

    def _select_init_state(options: Optional[dict]) -> Any:
        nonlocal episode_idx
        if options and "init_state_idx" in options:
            idx = int(options["init_state_idx"])
            idx = idx % len(initial_states)
            episode_idx = idx + 1
            return initial_states[idx]
        if episode_idx < len(initial_states):
            init_state = initial_states[episode_idx]
        else:
            init_state = initial_states[episode_idx % len(initial_states)]
        episode_idx += 1
        return init_state

    def _do_reset(reset_seed: Optional[int], options: Optional[dict]) -> dict:
        nonlocal current_obs_raw, episode_over, steps_since_reset
        if reset_seed is not None:
            env.seed(int(reset_seed))
        env.reset()
        init_state = _select_init_state(options)
        current_obs_raw = env.set_init_state(init_state)
        for _ in range(num_steps_wait):
            current_obs_raw, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        episode_over = False
        steps_since_reset = 0
        return _process_obs(current_obs_raw, task_description=task_description, resize_size=resize_size)

    def _do_step(action: Any) -> Tuple[dict, float, bool, bool, dict]:
        nonlocal current_obs_raw, episode_over, steps_since_reset
        if current_obs_raw is None:
            raise RuntimeError("Environment not reset; call reset first.")
        if episode_over:
            raise RuntimeError("Episode already finished; call reset before step.")

        act = np.asarray(action, dtype=np.float32)
        raw_obs, reward, done, info = env.step(act.tolist())
        current_obs_raw = raw_obs
        steps_since_reset += 1

        terminated = bool(done)  # LIBERO often uses done as success-only
        truncated = False
        if not terminated and steps_since_reset >= max_steps:
            truncated = True

        episode_over = bool(terminated or truncated)
        obs = _process_obs(raw_obs, task_description=task_description, resize_size=resize_size)
        info = info or {}
        if truncated:
            info = dict(info)
            info["TimeLimit.truncated"] = True
            info["TimeLimit.max_steps"] = int(max_steps)
        return obs, float(reward), terminated, truncated, info

    def _infer_specs() -> dict:
        # Size `observation/state` without advancing episode_idx (use initial_states[0]).
        nonlocal current_obs_raw, episode_over, steps_since_reset
        env.reset()
        current_obs_raw = env.set_init_state(initial_states[0])
        for _ in range(num_steps_wait):
            current_obs_raw, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        episode_over = False
        steps_since_reset = 0
        processed = _process_obs(current_obs_raw, task_description=task_description, resize_size=resize_size)
        state_dim = int(np.asarray(processed["observation/state"]).reshape(-1).shape[0])
        return {
            "state_dim": state_dim,
            "task_description": str(task_description),
            "max_steps": int(max_steps),
            "num_steps_wait": int(num_steps_wait),
        }

    while True:
        req = conn.recv()
        if not isinstance(req, dict):
            conn.send({"error": {"code": "bad_request", "message": f"Expected dict, got {type(req)}"}})
            continue
        cmd = req.get("cmd")
        try:
            if cmd == "infer_specs":
                conn.send(_infer_specs())
                continue
            if cmd == "reset":
                conn.send(
                    {
                        "obs": _do_reset(req.get("seed"), req.get("options")),
                        "info": {},
                    }
                )
                continue
            if cmd == "step":
                obs, reward, terminated, truncated, info = _do_step(req.get("action"))
                conn.send(
                    {
                        "obs": obs,
                        "reward": reward,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "done": bool(terminated or truncated),
                        "info": info,
                    }
                )
                continue
            if cmd == "close":
                conn.send({"ok": True})
                return
            conn.send({"error": {"code": "unknown_cmd", "message": f"Unknown cmd: {cmd}"}})
        except Exception as e:  # noqa: BLE001
            conn.send({"error": {"code": "worker_error", "message": str(e), "type": type(e).__name__}})


########################################################################################
# Router / session manager (fixed worker pool)
########################################################################################


class LiberoEnvServer:
    """Websocket env server with a fixed worker pool and session_id routing."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        task_suite_name: str,
        task_id: int,
        seed: int,
        resize_size: int,
        num_steps_wait: int,
        max_steps: Optional[int],
        horizon: Optional[int],
        max_sessions: int,
        session_idle_timeout_s: float,
    ) -> None:
        import multiprocessing as mp

        self._host = host
        self._port = port
        self._task_suite_name = task_suite_name
        self._task_id = task_id
        self._seed = seed
        self._resize_size = resize_size
        self._num_steps_wait = num_steps_wait
        self._max_steps_override = int(max_steps) if max_steps is not None else None
        self._max_sessions = int(max_sessions)
        self._session_idle_timeout_s = float(session_idle_timeout_s)

        if self._max_sessions <= 0:
            raise ValueError("--max_sessions must be > 0")

        self._mp = mp.get_context("spawn")

        self._workers: list[dict] = []
        self._free_workers: list[int] = []
        self._session_to_worker: Dict[str, int] = {}
        self._worker_to_session: Dict[int, Optional[str]] = {}
        self._last_used_s: Dict[str, float] = {}

        self._space_specs: Optional[Tuple[Dict, Dict]] = None
        self._task_description: Optional[str] = None
        self._max_steps_effective: Optional[int] = None

        self._start_workers()

    def _start_workers(self) -> None:
        for wid in range(self._max_sessions):
            parent_conn, child_conn = self._mp.Pipe(duplex=True)
            proc = self._mp.Process(
                target=_worker_loop,
                kwargs=dict(
                    task_suite_name=self._task_suite_name,
                    task_id=self._task_id,
                    seed=self._seed,
                    resize_size=self._resize_size,
                    num_steps_wait=self._num_steps_wait,
                    resolution=LIBERO_ENV_RESOLUTION,
                    max_steps_override=self._max_steps_override,
                    conn=child_conn,
                ),
                daemon=True,
                name=f"libero-worker-{wid}",
            )
            proc.start()
            self._workers.append({"proc": proc, "conn": parent_conn, "lock": asyncio.Lock()})
            self._free_workers.append(wid)
            self._worker_to_session[wid] = None

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        await self._ensure_space_specs()
        async with _server.serve(self._handler, self._host, self._port, compression=None, max_size=None) as server:
            logger.info("Libero env websocket server listening on ws://%s:%d", self._host, self._port)
            logger.info("Task suite=%s task_id=%d", self._task_suite_name, self._task_id)
            if self._task_description:
                logger.info("Task description=%s", self._task_description)
            if self._max_steps_effective is not None:
                logger.info("Episode max_steps=%d (main.py-aligned)", self._max_steps_effective)
            reaper_task = asyncio.create_task(self._reap_idle_sessions())
            try:
                await server.serve_forever()
            finally:
                reaper_task.cancel()

    async def _ensure_space_specs(self) -> None:
        if self._space_specs is not None:
            return
        resp = await self._call_worker(0, {"cmd": "infer_specs"})
        if "error" in resp:
            raise RuntimeError(f"Failed to infer specs: {resp['error']}")
        self._task_description = str(resp.get("task_description", ""))
        if resp.get("max_steps") is not None:
            self._max_steps_effective = int(resp["max_steps"])
        state_dim = int(resp["state_dim"])
        self._space_specs = libero_default_space_specs(resize_size=self._resize_size, state_dim=state_dim)

    async def _reap_idle_sessions(self) -> None:
        if self._session_idle_timeout_s <= 0:
            return
        while True:
            await asyncio.sleep(1.0)
            now = _now_s()
            for sid, last_used in list(self._last_used_s.items()):
                if now - last_used < self._session_idle_timeout_s:
                    continue
                wid = self._session_to_worker.get(sid)
                if wid is None:
                    self._last_used_s.pop(sid, None)
                    continue
                if self._workers[wid]["lock"].locked():
                    continue
                logger.info("Auto-releasing idle session_id=%s (worker stays alive)", sid)
                self._release_session(sid)

    def _release_session(self, session_id: str) -> None:
        wid = self._session_to_worker.pop(session_id, None)
        self._last_used_s.pop(session_id, None)
        if wid is None:
            return
        self._worker_to_session[wid] = None
        self._free_workers.append(wid)

    def _alloc_session(self, requested_session_id: Optional[str]) -> Tuple[str, int]:
        if not self._free_workers:
            raise RuntimeError("capacity_full")
        # If client requested a custom session_id and it's not already taken, honor it.
        if requested_session_id and requested_session_id not in self._session_to_worker:
            sid = requested_session_id
        else:
            sid = uuid.uuid4().hex
        wid = self._free_workers.pop()
        self._session_to_worker[sid] = wid
        self._worker_to_session[wid] = sid
        self._last_used_s[sid] = _now_s()
        return sid, wid

    async def _call_worker(self, worker_id: int, msg: dict) -> dict:
        w = self._workers[worker_id]
        proc = w["proc"]
        if not proc.is_alive():
            raise _WorkerCrashed(f"worker {worker_id} is not alive (exitcode={proc.exitcode})")

        async with w["lock"]:
            conn = w["conn"]

            def _sync_roundtrip() -> dict:
                conn.send(msg)
                return conn.recv()

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _sync_roundtrip)

    def _err(self, *, code: str, message: str, details: Optional[dict] = None, request_id: Any = None) -> dict:
        out: dict = {"error": {"code": code, "message": message}}
        if details:
            out["error"]["details"] = details
        if request_id is not None:
            out["request_id"] = request_id
        return out

    async def _handler(self, ws: _server.ServerConnection) -> None:
        packer = msgpack_numpy.Packer()
        await self._ensure_space_specs()

        await ws.send(
            packer.pack(
                {
                    "kind": "libero_env_server",
                    "task_suite_name": self._task_suite_name,
                    "task_id": self._task_id,
                    "task_description": self._task_description,
                    "resize_size": self._resize_size,
                    "max_steps": self._max_steps_effective,
                    "observation_space_spec": self._space_specs[0] if self._space_specs else None,
                    "action_space_spec": self._space_specs[1] if self._space_specs else None,
                    "session_protocol": {
                        "enabled": True,
                        "max_sessions": self._max_sessions,
                        "idle_timeout_s": self._session_idle_timeout_s,
                    },
                }
            )
        )

        while True:
            try:
                req_raw = await ws.recv()
                req = msgpack_numpy.unpackb(req_raw)
                if not isinstance(req, dict):
                    await ws.send(packer.pack(self._err(code="bad_request", message=f"Expected dict, got {type(req)}")))
                    continue

                cmd = req.get("cmd")
                request_id = req.get("request_id")
                session_id = req.get("session_id")

                if cmd == "close":
                    await ws.send(packer.pack({"ok": True, "request_id": request_id}))
                    await ws.close(code=websockets.frames.CloseCode.NORMAL_CLOSURE, reason="Closed by client.")
                    return

                if cmd == "close_session":
                    if not session_id or str(session_id) not in self._session_to_worker:
                        await ws.send(
                            packer.pack(
                                self._err(
                                    code="invalid_session",
                                    message="close_session requires a valid session_id",
                                    details={"session_id": session_id},
                                    request_id=request_id,
                                )
                            )
                        )
                        continue
                    self._release_session(str(session_id))
                    await ws.send(packer.pack({"ok": True, "session_id": str(session_id), "request_id": request_id}))
                    continue

                if cmd == "ping":
                    # Heartbeat to keep session alive
                    if session_id and str(session_id) in self._session_to_worker:
                        self._last_used_s[str(session_id)] = _now_s()
                        await ws.send(packer.pack({"ok": True, "cmd": "pong", "request_id": request_id}))
                    else:
                        # If session is invalid/expired, let client know so it can stop pinging or re-request
                        await ws.send(
                            packer.pack(
                                self._err(
                                    code="invalid_session",
                                    message="ping requires a valid active session_id",
                                    details={"session_id": session_id},
                                    request_id=request_id,
                                )
                            )
                        )
                    continue

                if cmd == "reset":
                    # If new_session=True, force allocate new worker lease.
                    new_session = bool(req.get("new_session", False))
                    if new_session:
                        session_id = None

                    if session_id and str(session_id) in self._session_to_worker:
                        sid = str(session_id)
                        wid = self._session_to_worker[sid]
                    else:
                        try:
                            sid, wid = self._alloc_session(str(session_id) if session_id else None)
                        except RuntimeError as e:
                            if str(e) == "capacity_full":
                                await ws.send(
                                    packer.pack(
                                        self._err(
                                            code="capacity_full",
                                            message="No free env workers; server at max_sessions",
                                            details={"max_sessions": self._max_sessions},
                                            request_id=request_id,
                                        )
                                    )
                                )
                                continue
                            raise

                    self._last_used_s[sid] = _now_s()
                    # Forward seed/options to worker (RemoteEnv sends these fields).
                    resp = await self._call_worker(
                        wid,
                        {
                            "cmd": "reset",
                            "seed": req.get("seed"),
                            "options": req.get("options"),
                        },
                    )
                    if "error" in resp:
                        await ws.send(
                            packer.pack(self._err(code="reset_failed", message=str(resp["error"]), request_id=request_id))
                        )
                        continue
                    await ws.send(
                        packer.pack(
                            {
                                "session_id": sid,
                                "obs": resp.get("obs"),
                                "info": resp.get("info") or {},
                                "request_id": request_id,
                            }
                        )
                    )
                    continue

                if cmd == "step":
                    if not session_id or str(session_id) not in self._session_to_worker:
                        await ws.send(
                            packer.pack(
                                self._err(
                                    code="invalid_session",
                                    message="step requires a valid session_id (call reset first)",
                                    details={"session_id": session_id},
                                    request_id=request_id,
                                )
                            )
                        )
                        continue
                    sid = str(session_id)
                    wid = self._session_to_worker[sid]
                    self._last_used_s[sid] = _now_s()
                    resp = await self._call_worker(wid, {"cmd": "step", "action": req.get("action")})
                    if "error" in resp:
                        err = resp.get("error") or {}
                        await ws.send(
                            packer.pack(
                                {
                                    "error": {
                                        "code": "step_failed",
                                        "message": err.get("message", str(err)),
                                        "details": {"session_id": sid, "worker_error": err},
                                    },
                                    "request_id": request_id,
                                }
                            )
                        )
                        continue
                    await ws.send(
                        packer.pack(
                            {
                                "session_id": sid,
                                "obs": resp.get("obs"),
                                "reward": float(resp.get("reward", 0.0)),
                                "terminated": bool(resp.get("terminated", False)),
                                "truncated": bool(resp.get("truncated", False)),
                                "done": bool(resp.get("done", False)),
                                "info": resp.get("info") or {},
                                "request_id": request_id,
                            }
                        )
                    )
                    continue

                await ws.send(
                    packer.pack(self._err(code="unknown_cmd", message=f"Unknown cmd: {cmd}", request_id=request_id))
                )

            except websockets.ConnectionClosed:
                return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Libero websocket env server (RemoteEnv-compatible, main.py-aligned max_steps)."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override per-suite max_steps (otherwise matches examples/libero/main.py).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional: best-effort override underlying env horizon (robosuite time limit).",
    )
    parser.add_argument("--max_sessions", type=int, default=1, help="Max concurrent env sessions (fixed worker pool).")
    parser.add_argument(
        "--session_idle_timeout_s",
        type=float,
        default=30.0,
        help="Auto-release idle session_id after this many seconds (worker is kept alive). Set <=0 to disable.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    LiberoEnvServer(
        host=args.host,
        port=args.port,
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        seed=args.seed,
        resize_size=args.resize_size,
        num_steps_wait=args.num_steps_wait,
        max_steps=args.max_steps,
        horizon=args.horizon,
        max_sessions=args.max_sessions,
        session_idle_timeout_s=args.session_idle_timeout_s,
    ).serve_forever()


if __name__ == "__main__":
    main()

