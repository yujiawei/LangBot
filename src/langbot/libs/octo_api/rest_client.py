"""REST client for the Octo bot API.

All calls authenticate with ``Authorization: Bearer <bot_token>`` (the bf_/app_
token, never the WS im_token). 429 handling follows the reference client: at
most 2 retries, honor Retry-After (seconds) with upward-only jitter, give up
when a single wait exceeds 10s or the cumulative sleep budget (15s) is spent.
Discardable calls (heartbeat) opt out of 429 retries entirely.
"""

from __future__ import annotations

import asyncio
import json
import random
import typing
import uuid

import aiohttp

from .types import RegisterResult, SendMessageResult

CARD_VERSION = '1.5'
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

    async def card_profile(self) -> dict:
        """GET /v1/bot/card/profile - capability discovery, fail closed.

        Returns {'available': False} when the endpoint is not deployed (404),
        so callers never guess a server-side Bot policy locally.
        """
        session = self._get_session()
        headers = {'Authorization': f'Bearer {self.bot_token}'}
        async with session.get(f'{self.api_url}/v1/bot/card/profile', headers=headers) as resp:
            if resp.status == 404:
                return {'available': False, 'enabled': False}
            if resp.status >= 400:
                raise OctoApiError(resp.status, (await resp.text())[:300])
            raw = await resp.json()
        if not isinstance(raw, dict):
            return {'available': True, 'enabled': False}
        raw['available'] = True
        # enabled is serialized as either boolean true or 1 depending on version.
        raw['enabled'] = raw.get('enabled') is True or raw.get('enabled') == 1
        return raw

    async def send_card(
        self,
        channel_id: str,
        channel_type: int,
        card: dict,
        plain: str,
        profile: str = 'octo/v1',
        client_msg_no: typing.Optional[str] = None,
    ) -> SendMessageResult:
        """Send an interactive card (payload type 17)."""
        if not channel_id or not channel_id.strip():
            raise ValueError('octo: channel_id is required to send a message')
        payload = {
            'type': 17,
            'card': card,
            'plain': plain,
            'profile': profile,
            'card_version': CARD_VERSION,
        }
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

    async def edit_card(
        self,
        message_id: str,
        channel_id: str,
        channel_type: int,
        card: dict,
        plain: str,
        profile: str = 'octo/v1',
        card_seq: typing.Optional[int] = None,
        transient: bool = False,
    ) -> None:
        """Replace a sent card's content.

        content_edit carries the complete type-17 envelope serialized to a JSON
        string. card_seq must be reserved inside the same serialized section as
        this call, so reservation order equals wire order - the server rejects
        stale frames, and a higher seq committing first wedges the card
        permanently. transient keeps intermediate frames out of the revision
        history (capped at 20); terminal frames must not be transient.
        """
        if not message_id:
            raise ValueError('octo: message_id is required to edit a card')
        if not channel_id or not channel_id.strip():
            raise ValueError('octo: channel_id is required to edit a card')
        envelope: dict = {
            'type': 17,
            'card': card,
            'plain': plain,
            'profile': profile,
            'card_version': CARD_VERSION,
        }
        if card_seq is not None:
            if card_seq <= 0:
                raise ValueError('octo: card_seq must be a positive integer')
            envelope['card_seq'] = card_seq
        if transient:
            envelope['transient'] = True
        await self._post_json(
            '/v1/bot/message/edit',
            {
                'message_id': message_id,
                'channel_id': channel_id,
                'channel_type': channel_type,
                'content_edit': json.dumps(envelope, ensure_ascii=False),
            },
            # A transient progress frame is discardable: holding the flush
            # while backing off would block the frames queued behind it.
            retry_on_429=not transient,
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
            if length and length.isdigit() and int(length) > max_bytes:
                return None
            # Accumulate to EOF: StreamReader.read(n) returns only what is
            # currently buffered, which silently truncates the body (a
            # truncated image is then rejected by vision models).
            body = bytearray()
            async for chunk in resp.content.iter_chunked(64 * 1024):
                body.extend(chunk)
                if len(body) > max_bytes:
                    return None
            return bytes(body)

    async def group_members(self, group_no: str) -> list[dict]:
        """GET /v1/bot/groups/{group_no}/members.

        Threads inherit their parent group's roster, so callers must pass the
        parent group_no, never a composite thread channel id.
        """
        data = await self._get_json(f'/v1/bot/groups/{group_no}/members')
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict)]
        if isinstance(data, dict):
            members = data.get('members') or data.get('data')
            if isinstance(members, list):
                return [m for m in members if isinstance(m, dict)]
        return []

    async def group_info(self, group_no: str) -> typing.Optional[dict]:
        """GET /v1/bot/groups/{group_no}."""
        data = await self._get_json(f'/v1/bot/groups/{group_no}')
        if isinstance(data, dict):
            inner = data.get('data')
            return inner if isinstance(inner, dict) else data
        return None

    async def _get_json(self, path: str) -> typing.Any:
        """GET returning parsed JSON, or None on any non-200 / transport error."""
        session = self._get_session()
        headers = {'Authorization': f'Bearer {self.bot_token}'}
        try:
            async with session.get(f'{self.api_url}{path}', headers=headers) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
        except Exception:
            return None

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
