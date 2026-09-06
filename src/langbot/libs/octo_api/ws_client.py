"""Asyncio WebSocket client for the Octo (WuKongIM) real-time endpoint.

Connection lifecycle mirrors the reference implementation:
- 15s deadline over the whole build-up (TCP + upgrade + CONNECT + CONNACK);
- binary PING every 60s, 3 unanswered pings force a reconnect;
- exponential backoff min(3000 * 2^n, 60000) ms with 0.75-1.25 jitter,
  attempts reset after 30s of stable connection;
- 3 consecutive connections shorter than 5s escalate to the auth-error
  callback (token refresh path);
- RECVACK is sent immediately on RECV, before decryption is attempted.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import typing

import aiohttp

from . import wire
from .types import MessagePayload, OctoMessage

CONNECT_DEADLINE_SECONDS = 15.0
PING_INTERVAL_SECONDS = 60.0
PING_MAX_MISSES = 3
STABLE_AFTER_SECONDS = 30.0
RAPID_DISCONNECT_SECONDS = 5.0
RAPID_DISCONNECT_LIMIT = 3


class OctoAuthError(Exception):
    """Kicked / connect-failed / rapidly disconnecting: the im_token must be refreshed."""


class OctoWSClient:
    def __init__(
        self,
        ws_url: str,
        uid: str,
        token: str,
        on_message: typing.Callable[[OctoMessage], typing.Awaitable[None]],
        on_auth_error: typing.Callable[[], typing.Awaitable[typing.Optional[tuple[str, str, str]]]],
        logger: typing.Any = None,
    ) -> None:
        """on_auth_error re-registers and returns (uid, token, ws_url) or None to give up."""
        self._ws_url = ws_url
        self._uid = uid
        self._token = token
        self._on_message = on_message
        self._on_auth_error = on_auth_error
        self._logger = logger
        self._stopped = False
        self._session: typing.Optional[aiohttp.ClientSession] = None
        self._pong_received = True
        self.connected = asyncio.Event()

    async def _log(self, level: str, msg: str) -> None:
        if self._logger is not None:
            await getattr(self._logger, level)(msg)

    def update_credentials(self, uid: str, token: str, ws_url: str = '') -> None:
        self._uid = uid
        self._token = token
        if ws_url:
            self._ws_url = ws_url

    async def stop(self) -> None:
        self._stopped = True
        self.connected.clear()
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def run(self) -> None:
        """Run until stop(); reconnects internally."""
        self._session = aiohttp.ClientSession()
        reconnect_attempts = 0
        rapid_disconnects = 0
        try:
            while not self._stopped:
                connected_at = 0.0
                try:
                    connected_at = await self._run_one_connection()
                    reconnect_attempts = 0
                except OctoAuthError as e:
                    await self._log('warning', f'Octo WS auth error: {e}, refreshing token...')
                    if not await self._refresh_credentials():
                        raise
                    rapid_disconnects = 0
                    # Random 0-5s stagger avoids refresh storms across bots.
                    await asyncio.sleep(random.uniform(0, 5))
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    await self._log('warning', f'Octo WS connection error: {e}')
                finally:
                    self.connected.clear()

                if self._stopped:
                    break

                if connected_at and time.monotonic() - connected_at < RAPID_DISCONNECT_SECONDS:
                    rapid_disconnects += 1
                elif connected_at:
                    rapid_disconnects = 0
                if rapid_disconnects >= RAPID_DISCONNECT_LIMIT:
                    rapid_disconnects = 0
                    await self._log('warning', 'Octo WS rapid disconnects, refreshing token...')
                    if not await self._refresh_credentials():
                        raise OctoAuthError('rapid disconnect and token refresh failed')
                    await asyncio.sleep(random.uniform(0, 5))
                    continue

                delay = min(3.0 * (2**reconnect_attempts), 60.0) * random.uniform(0.75, 1.25)
                reconnect_attempts += 1
                await asyncio.sleep(delay)
        finally:
            if self._session is not None:
                await self._session.close()
                self._session = None

    async def _refresh_credentials(self) -> bool:
        try:
            result = await self._on_auth_error()
        except Exception as e:
            await self._log('error', f'Octo token refresh failed: {e}')
            return False
        if result is None:
            return False
        self.update_credentials(*result)
        return True

    async def _run_one_connection(self) -> float:
        """One full connection: build-up, read loop. Returns the connect monotonic time."""
        assert self._session is not None
        async with self._session.ws_connect(
            self._ws_url,
            timeout=aiohttp.ClientWSTimeout(ws_close=10.0),
            heartbeat=None,
            max_msg_size=16 * 1024 * 1024,
        ) as ws:
            keypair = wire.generate_keypair()
            await ws.send_bytes(
                wire.encode_connect(
                    uid=self._uid,
                    token=self._token,
                    device_id=wire.generate_device_id(),
                    client_key_b64=keypair.public_b64,
                )
            )

            buffer = bytearray()
            aes_key = b''
            aes_iv = b''
            server_version = 0

            # Build-up: wait for CONNACK under the deadline.
            deadline = time.monotonic() + CONNECT_DEADLINE_SECONDS
            connack: typing.Optional[wire.ConnackPacket] = None
            while connack is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ConnectionError('connect deadline expired before CONNACK')
                msg = await ws.receive(timeout=remaining)
                frames = self._frames_from(msg, buffer)
                for frame in frames:
                    if frame.packet_type == wire.PacketType.CONNACK:
                        connack = wire.parse_connack(frame)
                        break

            if connack.reason_code != 1:
                raise OctoAuthError(f'CONNACK reason_code={connack.reason_code}')
            server_version = connack.server_version
            aes_key, aes_iv = wire.derive_aes(keypair.private, connack.server_key, connack.salt)
            connected_at = time.monotonic()
            self.connected.set()
            await self._log('info', 'Octo WS connected')

            ping_task = asyncio.create_task(self._ping_loop(ws))
            try:
                while True:
                    msg = await ws.receive()
                    for frame in self._frames_from(msg, buffer):
                        await self._handle_frame(ws, frame, server_version, aes_key, aes_iv)
            except ConnectionError as e:
                # Normal close (including a ping-timeout close from _ping_loop):
                # fall through to the outer reconnect with backoff.
                await self._log('warning', f'Octo WS connection closed ({e}), reconnecting')
            except _ServerDisconnect as e:
                raise OctoAuthError(str(e)) from None
            finally:
                self.connected.clear()
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass
            return connected_at

    def _frames_from(self, msg: aiohttp.WSMessage, buffer: bytearray) -> list[wire.Frame]:
        if msg.type == aiohttp.WSMsgType.BINARY:
            buffer += msg.data
            return wire.unpack_frames(buffer)
        if msg.type in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.ERROR,
        ):
            detail = ''
            if msg.type == aiohttp.WSMsgType.CLOSE:
                detail = f' code={msg.data} reason={msg.extra!r}'
            elif msg.type == aiohttp.WSMsgType.ERROR:
                detail = f' error={msg.data!r}'
            raise ConnectionError(f'websocket closed: {msg.type.name}{detail}')
        return []

    async def _handle_frame(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        frame: wire.Frame,
        server_version: int,
        aes_key: bytes,
        aes_iv: bytes,
    ) -> None:
        if frame.packet_type == wire.PacketType.PONG:
            self._pong_received = True
            return
        if frame.packet_type == wire.PacketType.DISCONNECT:
            packet = wire.parse_disconnect(frame)
            raise _ServerDisconnect(f'server disconnect: {packet.reason_code} {packet.reason}')
        if frame.packet_type != wire.PacketType.RECV:
            return

        recv = wire.parse_recv(frame, server_version)
        # RECVACK must go out before decryption: an unparseable payload must
        # not stall delivery.
        await ws.send_bytes(wire.encode_recvack(recv.message_id, recv.message_seq))

        try:
            plaintext = wire.aes_decrypt_payload(recv.encrypted_payload, aes_key, aes_iv)
            payload_obj = json.loads(plaintext.decode('utf-8'))
        except Exception as e:
            await self._log('warning', f'Octo payload decrypt/parse error: {e}')
            return

        message = OctoMessage(
            message_id=recv.message_id,
            message_seq=recv.message_seq,
            from_uid=recv.from_uid,
            channel_id=recv.channel_id,
            channel_type=recv.channel_type,
            timestamp=recv.timestamp,
            payload=MessagePayload.model_validate(payload_obj),
        )
        try:
            await self._on_message(message)
        except Exception as e:
            await self._log('error', f'Octo message handler error: {e}')

    async def _ping_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        misses = 0
        while True:
            await asyncio.sleep(PING_INTERVAL_SECONDS)
            if self._pong_received:
                misses = 0
            else:
                misses += 1
                if misses >= PING_MAX_MISSES:
                    # Closing the socket wakes the read loop with CLOSED.
                    await ws.close()
                    return
            self._pong_received = False
            try:
                await ws.send_bytes(wire.encode_ping())
            except Exception:
                return


class _ServerDisconnect(Exception):
    pass
