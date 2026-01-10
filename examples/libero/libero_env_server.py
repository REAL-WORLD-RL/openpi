from __future__ import annotations

"""
Libero websocket env server (RemoteEnv-compatible), built on GymEnvServer base class.

Key features specific to LIBERO:
- Uses per-suite `max_steps` (libero_spatial/object/goal/10/90) aligned to examples/libero/main.py.
- Performs `num_steps_wait` dummy steps during reset (so client policy steps start at t=0).
- Treats LIBERO `done=True` as success/termination; time-limit is reported as truncated=True.
- Manages task suites, task IDs, and initial states.

Protocol compatibility:
- Inherits RemoteEnv-compatible websocket protocol from GymEnvServer.
- Supports session_id routing with fixed worker pool.
"""

import argparse
import asyncio
import logging
import math
import pathlib
import sys
import time
from typing import Any, Optional

import numpy as np

# Allow running as a script: `python examples/libero/libero_env_server.py`
# Add repo root to sys.path so local imports like `pi_link.*` work.
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from pi_link.gym_env_server import GymEnvServer  # noqa: E402
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


def _libero_worker_loop(
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
    """Worker process for LIBERO env with special initialization and warmup logic."""
    task_suite = _load_task_suite(task_suite_name)
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = _get_libero_env(task, resolution, seed)

    # Episode length in *policy steps* (warmup already happens inside reset).
    max_steps = int(max_steps_override) if max_steps_override is not None else _max_steps_for_suite(task_suite_name)

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
        # LIBERO-specific warmup steps
        for _ in range(num_steps_wait):
            current_obs_raw, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        episode_over = False
        steps_since_reset = 0
        processed_obs = _process_obs(current_obs_raw, task_description=task_description, resize_size=resize_size)
        return {"obs": processed_obs, "info": {}}

    def _do_step(action: Any) -> dict:
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
        
        return {
            "obs": obs,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done": bool(terminated or truncated),
            "info": info,
        }

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

    # Worker message loop
    while True:
        req = conn.recv()
        if not isinstance(req, dict):
            conn.send({"error": {"code": "bad_request", "message": f"Expected dict, got {type(req)}"}})
            continue
        cmd = req.get("cmd")
        try:
            if cmd == "infer_specs":
                conn.send(_infer_specs())
            elif cmd == "reset":
                conn.send(_do_reset(req.get("seed"), req.get("options")))
            elif cmd == "step":
                conn.send(_do_step(req.get("action")))
            elif cmd == "close":
                if hasattr(env, 'close'):
                    env.close()
                conn.send({"ok": True})
                return
            else:
                conn.send({"error": {"code": "unknown_cmd", "message": f"Unknown cmd: {cmd}"}})
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Worker error on cmd={cmd}")
            conn.send({"error": {"code": "worker_error", "message": str(e), "type": type(e).__name__}})



class LiberoEnvServer(GymEnvServer):
    """Websocket env server for LIBERO with task suite management."""

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
        max_sessions: int,
        session_idle_timeout_s: float,
    ) -> None:
        import multiprocessing as mp

        self._task_suite_name = task_suite_name
        self._task_id = task_id
        self._seed = seed
        self._resize_size = resize_size
        self._num_steps_wait = num_steps_wait
        self._max_steps_override = int(max_steps) if max_steps is not None else None
        
        # Cache for metadata
        self._task_description: Optional[str] = None
        self._max_steps_effective: Optional[int] = None
        self._state_dim: Optional[int] = None
        
        # Use custom multiprocessing context
        self._mp = mp.get_context("spawn")
        
        # Initialize base class WITHOUT starting workers (we'll do custom worker start)
        self._host = host
        self._port = port
        self._max_sessions = int(max_sessions)
        self._session_idle_timeout_s = float(session_idle_timeout_s)
        
        if self._max_sessions <= 0:
            raise ValueError("max_sessions must be > 0")
        
        # Worker pool
        self._workers: list[dict] = []
        self._free_workers: list[int] = []
        
        # Session management
        self._session_to_worker: dict[str, int] = {}
        self._worker_to_session: dict[int, Optional[str]] = {}
        self._last_used_s: dict[str, float] = {}
        
        # Cached specs
        self._space_specs: Optional[dict] = None
        
        # Start LIBERO-specific workers
        self._start_libero_workers()

    def _start_libero_workers(self) -> None:
        """Start worker pool with LIBERO-specific configuration."""
        import asyncio
        
        for wid in range(self._max_sessions):
            parent_conn, child_conn = self._mp.Pipe(duplex=True)
            proc = self._mp.Process(
                target=_libero_worker_loop,
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

    async def _ensure_space_specs(self) -> None:
        """Infer and cache LIBERO-specific space specs."""
        if self._space_specs is not None:
            return
        
        # Import needed here to avoid issues
        from pi_link.gym_env_server import _WorkerCrashed
        
        resp = await self._call_worker(0, {"cmd": "infer_specs"})
        if "error" in resp:
            raise RuntimeError(f"Failed to infer specs: {resp['error']}")
        
        self._task_description = str(resp.get("task_description", ""))
        self._max_steps_effective = int(resp.get("max_steps", 0))
        self._state_dim = int(resp.get("state_dim", 0))
        
        # Build space specs
        obs_spec, action_spec = libero_default_space_specs(
            resize_size=self._resize_size, 
            state_dim=self._state_dim
        )
        
        self._space_specs = {
            "observation_space_spec": obs_spec,
            "action_space_spec": action_spec,
            "sample_obs": None,  # Not needed for LIBERO
        }
        
        logger.info("LIBERO space specs inferred successfully")

    def get_metadata(self) -> dict:
        """Return LIBERO-specific server metadata."""
        metadata = {
            "kind": "libero_env_server",
            "task_suite_name": self._task_suite_name,
            "task_id": self._task_id,
            "task_description": self._task_description,
            "resize_size": self._resize_size,
            "max_steps": self._max_steps_effective,
            "max_sessions": self._max_sessions,
            "idle_timeout_s": self._session_idle_timeout_s,
        }
        return metadata

    async def run(self) -> None:
        """Override to add LIBERO-specific logging."""
        import websockets.asyncio.server as _server
        
        await self._ensure_space_specs()
        
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
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

    # These methods are not used since we override worker management
    def create_env(self, **kwargs) -> Any:
        """Not used - LIBERO uses custom worker loop."""
        raise NotImplementedError("LIBERO uses custom worker management")
    
    def process_observation(self, obs: Any) -> dict:
        """Not used - LIBERO uses custom worker loop."""
        raise NotImplementedError("LIBERO uses custom worker management")




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Libero websocket env server (RemoteEnv-compatible, built on GymEnvServer base class)."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
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
        max_sessions=args.max_sessions,
        session_idle_timeout_s=args.session_idle_timeout_s,
    ).serve_forever()


if __name__ == "__main__":
    main()
