import asyncio
import http
import logging
import time
import traceback

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        request_count = 0
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                recv_time = time.monotonic()
                request_count += 1
                
                # ========== 日志：打印接收到的输入 ==========
                logger.info(f"[Request #{request_count}] Received obs at t={recv_time:.3f}s")
                for key, value in obs.items():
                    if isinstance(value, (np.ndarray, list)):
                        arr = np.asarray(value)
                        logger.info(f"  obs['{key}'].shape = {arr.shape}, dtype = {arr.dtype}")
                    elif isinstance(value, dict):
                        logger.info(f"  obs['{key}'] = dict with keys: {list(value.keys())}")
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (np.ndarray, list)):
                                sub_arr = np.asarray(sub_value)
                                logger.info(f"    obs['{key}']['{sub_key}'].shape = {sub_arr.shape}, dtype = {sub_arr.dtype}")
                            else:
                                logger.info(f"    obs['{key}']['{sub_key}'] = {type(sub_value).__name__}")
                    else:
                        logger.info(f"  obs['{key}'] = {type(value).__name__}: {str(value)[:100]}")
                # ==========================================

                infer_start = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_start
                output_time = time.monotonic()

                # ========== 日志：打印输出的 action ==========
                logger.info(f"[Request #{request_count}] Output action at t={output_time:.3f}s")
                for key, value in action.items():
                    if isinstance(value, (np.ndarray, list)):
                        arr = np.asarray(value)
                        logger.info(f"  action['{key}'].shape = {arr.shape}, dtype = {arr.dtype}")
                    elif isinstance(value, dict):
                        logger.info(f"  action['{key}'] = {value}")
                    else:
                        logger.info(f"  action['{key}'] = {type(value).__name__}")
                logger.info(f"[Request #{request_count}] Inference time: {infer_time * 1000:.2f} ms")
                logger.info(f"[Request #{request_count}] Total throughput time: {(output_time - recv_time) * 1000:.2f} ms")
                # ==========================================

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
