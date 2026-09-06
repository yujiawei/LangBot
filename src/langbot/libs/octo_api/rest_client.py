"""REST client for the Octo bot API.

All calls authenticate with ``Authorization: Bearer <bot_token>`` (the bf_/app_
token, never the WS im_token). 429 handling follows the reference client: at
most 2 retries, honor Retry-After (seconds) with upward-only jitter, give up
when a single wait exceeds 10s or the cumulative sleep budget (15s) is spent.
Discardable calls (heartbeat) opt out of 429 retries entirely.
"""

from __future__ import annotations

import asyncio
import random
import typing
import uuid

import aiohttp

from .types import RegisterResult, SendMessageResult

DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_429_RETRIES = 2
MAX_SINGLE_WAIT_SECONDS = 10.0
SLEEP_BUDGET_SECONDS = 15.0


class OctoApiError(Exception):
    def __init__(self, status: int, message: str, retry_after_ms: typing.Optional[int] = None):
        super().__init__(f'Octo API error {status}: {message}')
        self.status = status
        self.retry_after_ms = retry_after_ms


class OctoRestClient:
    def __init__(self, api_url: str, bot_token: str) -> None:
        self.api_url = api_url.rstrip('/')
        self.bot_token = bot_token
        self._session: typing.Optional[aiohttp.ClientSession] = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT_SECONDS)
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _post_json(
        self,
        path: str,
        body: dict,
        retry_on_429: bool = True,
    ) -> dict:
        session = self._get_session()
        headers = {
            'Authorization': f'Bearer {self.bot_token}',
            'Content-Type': 'application/json',
        }
        slept = 0.0
        attempt = 0
        while True:
            async with session.post(f'{self.api_url}{path}', json=body, headers=headers) as resp:
                if resp.status == 429 and retry_on_429 and attempt < MAX_429_RETRIES:
                    wait = self._retry_after_seconds(resp)
                    if wait > MAX_SINGLE_WAIT_SECONDS or slept + wait > SLEEP_BUDGET_SECONDS:
                        raise OctoApiError(429, 'rate limited', int(wait * 1000))
                    # Upward-only jitter: Retry-After is an *earliest* time.
                    wait *= 1 + random.random() * 0.25
                    attempt += 1
                    slept += wait
                    await asyncio.sleep(wait)
                    continue
                if resp.status >= 400:
                    text = await resp.text()
                    raise OctoApiError(resp.status, text[:500])
                if resp.content_type and 'json' in resp.content_type:
                    return await resp.json()
                return {}

    @staticmethod
    def _retry_after_seconds(resp: aiohttp.ClientResponse) -> float:
        header = resp.headers.get('Retry-After', '')
        try:
            return max(float(header), 0.1)
        except ValueError:
            return 1.0

    async def register(
        self,
        force_refresh: bool = False,
        agent_platform: str = 'LangBot',
        agent_version: str = '',
    ) -> RegisterResult:
        path = '/v1/bot/register'
        if force_refresh:
            path += '?force_refresh=true'
        body: dict = {'agent_platform': agent_platform}
        if agent_version:
            body['agent_version'] = agent_version
        data = await self._post_json(path, body)
        return RegisterResult.model_validate(data)

    async def send_text(
        self,
        channel_id: str,
        channel_type: int,
        content: str,
        mention: typing.Optional[dict] = None,
        reply: typing.Optional[dict] = None,
        client_msg_no: typing.Optional[str] = None,
    ) -> SendMessageResult:
        """Send a text message.

        ``reply`` must be the full nested quote structure
        ``{message_id, from_uid, from_name, payload}``: servers do not resolve
        a bare message_id and render an empty quote block for it.
        """
        if not channel_id or not channel_id.strip():
            raise ValueError('octo: channel_id is required to send a message')
        payload: dict = {'type': 1, 'content': content}
        if mention:
            payload['mention'] = mention
        if reply:
            payload['reply'] = reply
        data = await self._post_json(
            '/v1/bot/sendMessage',
            {
                'channel_id': channel_id,
                'channel_type': channel_type,
                'payload': payload,
                # Server-side idempotency key: reuse the same UUID on retry.
                'client_msg_no': client_msg_no or str(uuid.uuid4()),
            },
        )
        return SendMessageResult.model_validate(data)

    async def heartbeat(self) -> None:
        await self._post_json('/v1/bot/heartbeat', {}, retry_on_429=False)

    async def typing(self, channel_id: str, channel_type: int) -> None:
        """Show the typing indicator in a channel. Discardable: no 429 retry."""
        await self._post_json(
            '/v1/bot/typing',
            {'channel_id': channel_id, 'channel_type': channel_type},
            retry_on_429=False,
        )

    async def upload_media(self, filename: str, data: bytes, content_type: str) -> str:
        """Three-phase upload: presign -> PUT -> return the downloadUrl.

        fileSize must be the exact byte count and the presign response's
        contentType/contentDisposition must be replayed verbatim on the PUT:
        both are folded into the signed canonical headers, and any mismatch
        returns 403 SignatureDoesNotMatch.
        """
        if not data:
            raise ValueError('octo: cannot upload empty media')
        session = self._get_session()
        headers = {'Authorization': f'Bearer {self.bot_token}'}
        params = {'filename': filename, 'fileSize': str(len(data)), 'contentType': content_type}
        async with session.get(
            f'{self.api_url}/v1/bot/upload/presigned', params=params, headers=headers
        ) as resp:
            if resp.status >= 400:
                raise OctoApiError(resp.status, (await resp.text())[:500])
            presign = await resp.json()
        upload_url = presign.get('uploadUrl')
        download_url = presign.get('downloadUrl')
        if not upload_url or not download_url:
            raise OctoApiError(500, 'presign response missing uploadUrl/downloadUrl')

        put_headers = {
            'Content-Type': presign.get('contentType') or 'application/octet-stream',
            'Content-Length': str(len(data)),
        }
        if presign.get('contentDisposition'):
            put_headers['Content-Disposition'] = presign['contentDisposition']
        async with session.put(
            upload_url,
            data=data,
            headers=put_headers,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status >= 400:
                raise OctoApiError(resp.status, f'presigned PUT failed: {(await resp.text())[:300]}')
        return str(download_url)

    async def send_media(
        self,
        channel_id: str,
        channel_type: int,
        media_type: int,
        url: str,
        name: typing.Optional[str] = None,
        size: typing.Optional[int] = None,
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        client_msg_no: typing.Optional[str] = None,
    ) -> SendMessageResult:
        """Send an already-uploaded media message (type 2 image / 4 voice / 8 file)."""
        if not channel_id or not channel_id.strip():
            raise ValueError('octo: channel_id is required to send a message')
        payload: dict = {'type': media_type, 'url': url}
        if name:
            payload['name'] = name
        if size is not None:
            payload['size'] = size
        if width:
            payload['width'] = width
        if height:
            payload['height'] = height
        data = await self._post_json(
            '/v1/bot/sendMessage',
            {
                'channel_id': channel_id,
                'channel_type': channel_type,
                'payload': payload,
                'client_msg_no': client_msg_no or str(uuid.uuid4()),
            },
        )
        return SendMessageResult.model_validate(data)

    async def download_media(self, url: str, with_auth: bool, max_bytes: int) -> typing.Optional[bytes]:
        """Download media content.

        Inline media (images) is public-read object storage: no Authorization
        header. File-message content requires the Bearer token. Sending the
        token to public storage would leak it to a CDN; omitting it on file
        content gets a 401 - do not mix the two up.
        """
        session = self._get_session()
        headers = {'Authorization': f'Bearer {self.bot_token}'} if with_auth else {}
        async with session.get(
            url, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            if resp.status != 200:
                return None
            length = resp.headers.get('Content-Length')
            if length and int(length) > max_bytes:
                return None
            data = await resp.content.read(max_bytes + 1)
            if len(data) > max_bytes:
                return None
            return data

    async def user_info(self, uid: str) -> typing.Optional[dict]:
        """GET /v1/bot/user/info; returns None when the endpoint is not deployed."""
        session = self._get_session()
        headers = {'Authorization': f'Bearer {self.bot_token}'}
        try:
            async with session.get(
                f'{self.api_url}/v1/bot/user/info', params={'uid': uid}, headers=headers
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data if isinstance(data, dict) else None
        except Exception:
            return None

    async def read_receipt(
        self, channel_id: str, channel_type: int, message_ids: typing.Optional[list[str]] = None
    ) -> None:
        """Mark messages as read. Discardable: no 429 retry."""
        body: dict = {'channel_id': channel_id, 'channel_type': channel_type}
        if message_ids:
            body['message_ids'] = message_ids
        await self._post_json('/v1/bot/readReceipt', body, retry_on_429=False)
