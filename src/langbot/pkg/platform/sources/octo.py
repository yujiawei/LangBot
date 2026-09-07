"""Octo IM adapter for LangBot.

Connects to a self-hosted Octo server: REST for outbound (Bearer bot token),
WuKongIM binary WebSocket for inbound (short-lived im_token from register).

P1 scope: text conversation in DMs, groups and sub-groups (threads), mention
detection, quote replies. Media, cards and streaming come in later phases.

Reference protocol implementation:
https://github.com/Mininglamp-OSS/openclaw-channel-octo
"""

from __future__ import annotations

import asyncio
import traceback
import typing

import pydantic

from langbot.libs.octo_api import (
    ChannelType,
    OctoMessage,
    OctoRestClient,
    OctoWSClient,
    PayloadType,
)
from langbot.libs.octo_api import cards as octo_cards
from langbot.libs.octo_api import media as octo_media
from langbot.libs.octo_api.types import MentionInfo, is_thread_channel, strip_space_prefix

import base64
import dataclasses
import os
import time

import langbot_plugin.api.definition.abstract.platform.adapter as abstract_platform_adapter
import langbot_plugin.api.definition.abstract.platform.event_logger as abstract_platform_logger
import langbot_plugin.api.entities.builtin.platform.entities as platform_entities
import langbot_plugin.api.entities.builtin.platform.events as platform_events
import langbot_plugin.api.entities.builtin.platform.message as platform_message

HEARTBEAT_INTERVAL_SECONDS = 30.0
TYPING_INTERVAL_SECONDS = 5.0
TYPING_MAX_SECONDS = 120.0
# Minimum spacing between streaming card edits. Each chunk carries the full
# accumulated text, so a skipped frame is superseded by the next one.
CARD_EDIT_INTERVAL_SECONDS = 0.8
CARD_STATE_MAX = 256
# Card policy is per-bot server state that an operator can change at any time,
# so the probe result is cached with a TTL rather than for the process lifetime.
CARD_CAPABILITY_TTL_SECONDS = 600.0


@dataclasses.dataclass
class _CardState:
    """One in-flight streaming card, keyed by the runner's resp_message_id."""

    card_message_id: str
    channel_id: str
    channel_type: int
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    seq: int = 1
    last_text: str = ''
    last_edit_at: float = 0.0
    finalized: bool = False

    def next_seq(self) -> int:
        """Reserve the next frame number. Callers must hold ``lock``."""
        self.seq += 1
        return self.seq


def _utf16_len(s: str) -> int:
    """Length of s in UTF-16 code units (Octo mention offsets use these)."""
    return sum(2 if ord(ch) > 0xFFFF else 1 for ch in s)


def _utf16_index_map(text: str) -> dict[int, int]:
    """Map UTF-16 code-unit offsets to Python string indices."""
    mapping: dict[int, int] = {}
    units = 0
    for i, ch in enumerate(text):
        mapping[units] = i
        units += 2 if ord(ch) > 0xFFFF else 1
    mapping[units] = len(text)
    return mapping


class OctoMessageConverter(abstract_platform_adapter.AbstractMessageConverter):
    """Converts between LangBot MessageChain and Octo message payloads."""

    @staticmethod
    async def yiri2target(
        message_chain: platform_message.MessageChain,
    ) -> tuple[str, typing.Optional[dict], list[platform_message.MessageComponent]]:
        """Convert a MessageChain to (text content, mention dict or None,
        media components to send separately).

        Mention entity offsets are computed in UTF-16 code units over the
        final content, and length includes the '@' sign.
        """
        content = ''
        uids: list[str] = []
        entities: list[dict] = []
        mention_all = False
        media: list[platform_message.MessageComponent] = []

        for component in message_chain:
            if isinstance(component, platform_message.Plain):
                content += component.text
            elif isinstance(component, platform_message.At):
                display = component.display or str(component.target)
                segment = f'@{display}'
                entities.append(
                    {
                        'uid': str(component.target),
                        'offset': _utf16_len(content),
                        'length': _utf16_len(segment),
                    }
                )
                uids.append(str(component.target))
                content += segment + ' '
            elif isinstance(component, platform_message.AtAll):
                mention_all = True
                content += '@所有人 '
            elif isinstance(component, platform_message.Forward):
                for node in component.node_list:
                    if node.message_chain:
                        sub_content, _, sub_media = await OctoMessageConverter.yiri2target(node.message_chain)
                        content += sub_content + '\n'
                        media.extend(sub_media)
            elif isinstance(component, (platform_message.Image, platform_message.File, platform_message.Voice)):
                media.append(component)

        mention: typing.Optional[dict] = None
        if uids or entities or mention_all:
            mention = {}
            if uids:
                mention['uids'] = uids
            if entities:
                mention['entities'] = entities
            if mention_all:
                mention['all'] = 1
        return content, mention, media

    @staticmethod
    async def target2yiri(
        msg: OctoMessage,
        bot_uid: str,
        downloads: typing.Optional[dict[str, tuple[bytes, str]]] = None,
    ) -> platform_message.MessageChain:
        """Convert an inbound Octo message to a MessageChain.

        downloads maps a payload url to its downloaded (bytes, mime), filled
        by the adapter before conversion.
        """
        downloads = downloads or {}
        components: list[platform_message.MessageComponent] = []
        payload = msg.payload

        if payload.reply is not None and payload.reply.payload is not None:
            quoted_text = OctoMessageConverter._payload_plain_text(payload.reply.payload)
            if quoted_text:
                components.append(
                    platform_message.Quote(
                        sender_id=payload.reply.from_uid,
                        origin=platform_message.MessageChain(
                            [platform_message.Plain(text=quoted_text)]
                        ),
                    )
                )

        if payload.type == PayloadType.TEXT:
            text = payload.content if isinstance(payload.content, str) else ''
            components.extend(OctoMessageConverter._split_text_with_mentions(text, payload.mention, bot_uid))
        elif payload.type == PayloadType.RICH_TEXT:
            components.extend(OctoMessageConverter._richtext_components(payload, downloads))
        elif payload.type == PayloadType.INTERACTIVE_CARD:
            # Never parse the card tree inbound; the server-generated plain
            # text is authoritative.
            plain = getattr(payload, 'plain', None) or '[卡片]'
            components.append(platform_message.Plain(text=plain))
        elif payload.type in (PayloadType.IMAGE, PayloadType.GIF):
            hit = downloads.get(payload.url or '')
            if hit:
                data, mime = hit
                b64 = (await asyncio.to_thread(base64.b64encode, data)).decode('ascii')
                components.append(platform_message.Image(base64=f'data:{mime};base64,{b64}'))
            else:
                components.append(platform_message.Unknown(text='[Image]'))
        elif payload.type == PayloadType.VOICE:
            hit = downloads.get(payload.url or '')
            if hit:
                data, _ = hit
                b64 = (await asyncio.to_thread(base64.b64encode, data)).decode('ascii')
                components.append(platform_message.Voice(base64=b64))
            else:
                components.append(platform_message.Unknown(text='[Voice]'))
        elif payload.type == PayloadType.VIDEO:
            components.append(platform_message.Unknown(text='[Video]'))
        elif payload.type == PayloadType.FILE:
            hit = downloads.get(payload.url or '')
            if hit:
                data, _ = hit
                b64 = (await asyncio.to_thread(base64.b64encode, data)).decode('ascii')
                components.append(
                    platform_message.File(
                        name=payload.name or 'file',
                        size=payload.size or len(data),
                        base64=b64,
                    )
                )
            else:
                components.append(platform_message.Unknown(text=f'[File: {payload.name or ""}]'))
        else:
            components.append(platform_message.Unknown(text='[Unsupported message type]'))

        return platform_message.MessageChain(components)

    @staticmethod
    def build_reply_quote(msg: OctoMessage, from_name: str) -> dict:
        """Build the outbound quote block for replying to msg.

        Servers do not resolve a bare message_id (it renders as an empty quote
        box), so the full nested structure is required. mention/reply/event are
        stripped from the quoted payload to keep the preview flat.
        """
        quoted_payload = msg.payload.model_dump(exclude_none=True)
        for key in ('mention', 'reply', 'event'):
            quoted_payload.pop(key, None)
        return {
            'message_id': msg.message_id,
            'from_uid': msg.from_uid,
            'from_name': from_name or msg.from_uid,
            'payload': quoted_payload,
        }

    @staticmethod
    def _payload_plain_text(payload_dict: dict) -> str:
        """Best-effort plain text of a nested (quoted) payload."""
        ptype = payload_dict.get('type', 0)
        if ptype == PayloadType.TEXT and isinstance(payload_dict.get('content'), str):
            return payload_dict['content']
        plain = payload_dict.get('plain')
        if isinstance(plain, str) and plain:
            return plain
        return ''

    @staticmethod
    def _richtext_components(
        payload,
        downloads: typing.Optional[dict[str, tuple[bytes, str]]] = None,
    ) -> list[platform_message.MessageComponent]:
        """RichText(14) content is an ordered block array; string content is
        the backward-compatible single text block."""
        downloads = downloads or {}
        components: list[platform_message.MessageComponent] = []
        content = payload.content
        if isinstance(content, str):
            return [platform_message.Plain(text=content)]
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get('type') == 'text' and block.get('text'):
                    components.append(platform_message.Plain(text=block['text']))
                elif block.get('type') == 'image':
                    hit = downloads.get(block.get('url') or '')
                    if hit:
                        data, mime = hit
                        b64 = base64.b64encode(data).decode('ascii')
                        components.append(platform_message.Image(base64=f'data:{mime};base64,{b64}'))
                    else:
                        components.append(platform_message.Unknown(text='[图片]'))
        if not components:
            plain = getattr(payload, 'plain', None)
            if isinstance(plain, str) and plain:
                components.append(platform_message.Plain(text=plain))
        return components

    @staticmethod
    def _split_text_with_mentions(
        text: str,
        mention: typing.Optional[MentionInfo],
        bot_uid: str,
    ) -> list[platform_message.MessageComponent]:
        """Split text into Plain/At/AtAll components using UTF-16 entity spans.

        Broadcast semantics: when all/humans is set the server also sets ais,
        so ais counts as "the bot is addressed" only without a broadcast flag.
        """
        components: list[platform_message.MessageComponent] = []
        if mention is None:
            if text:
                components.append(platform_message.Plain(text=text))
            return components

        is_broadcast = mention.all or mention.humans
        if is_broadcast:
            components.append(platform_message.AtAll())

        def canonical(uid: str) -> str:
            """Emit the bot's own id exactly as bot_account_id spells it.

            The at-bot response rule compares At.target to bot_account_id as
            an exact string. Octo uids may carry an "s{space}_" prefix that
            register()'s robot_id does not, so a space-scoped deployment would
            never match and the bot would silently ignore every group mention.
            """
            if strip_space_prefix(uid)[1] == strip_space_prefix(bot_uid)[1]:
                return bot_uid
            return uid

        entities = [e for e in mention.entities if e.uid]
        entities.sort(key=lambda e: e.offset)
        index_map = _utf16_index_map(text)
        cursor = 0
        mentioned_uids: set[str] = set()
        for entity in entities:
            start = index_map.get(entity.offset)
            end = index_map.get(entity.offset + entity.length)
            if start is None or end is None or start < cursor or end > len(text):
                continue
            if start > cursor:
                components.append(platform_message.Plain(text=text[cursor:start]))
            display = text[start:end].lstrip('@')
            target = canonical(entity.uid)
            components.append(platform_message.At(target=target, display=display))
            mentioned_uids.add(target)
            cursor = end
        if not entities and mention.uids:
            # Fallback when the server sent uids without entity spans.
            for uid in mention.uids:
                target = canonical(uid)
                mentioned_uids.add(target)
                components.append(platform_message.At(target=target))
        if cursor < len(text):
            components.append(platform_message.Plain(text=text[cursor:]))

        if mention.ais and not is_broadcast and bot_uid not in mentioned_uids:
            components.append(platform_message.At(target=bot_uid))
        return components


class OctoEventConverter(abstract_platform_adapter.AbstractEventConverter):
    """Converts Octo messages to LangBot events."""

    @staticmethod
    async def yiri2target(event: platform_events.MessageEvent) -> typing.Any:
        return event.source_platform_object

    @staticmethod
    async def target2yiri(
        msg: OctoMessage,
        bot_uid: str,
        downloads: typing.Optional[dict[str, tuple[bytes, str]]] = None,
    ) -> typing.Optional[platform_events.MessageEvent]:
        # System events (group_md_updated etc.) ride on ordinary messages.
        if msg.payload.event is not None:
            return None
        # Never react to the bot's own messages.
        if msg.from_uid == bot_uid or msg.from_uid.endswith(f'_{bot_uid}'):
            return None
        if not msg.from_uid:
            return None

        message_chain = await OctoMessageConverter.target2yiri(msg, bot_uid, downloads)
        if not message_chain:
            return None

        if msg.channel_type == ChannelType.DM:
            return platform_events.FriendMessage(
                sender=platform_entities.Friend(
                    id=msg.from_uid,
                    nickname=msg.from_uid,
                    remark='',
                ),
                message_chain=message_chain,
                time=float(msg.timestamp),
                source_platform_object=msg,
            )
        if msg.channel_type in (ChannelType.GROUP, ChannelType.COMMUNITY_TOPIC):
            # A thread's composite channel_id ("<group_no>____<short_id>") is
            # used directly as the group id: it isolates the session AND is a
            # valid send target, so no launcher-id override is needed.
            return platform_events.GroupMessage(
                sender=platform_entities.GroupMember(
                    id=msg.from_uid,
                    member_name=msg.from_uid,
                    permission=platform_entities.Permission.Member,
                    group=platform_entities.Group(
                        id=msg.channel_id,
                        name='',
                        permission=platform_entities.Permission.Member,
                    ),
                    special_title='',
                ),
                message_chain=message_chain,
                time=float(msg.timestamp),
                source_platform_object=msg,
            )
        return None


class OctoAdapter(abstract_platform_adapter.AbstractMessagePlatformAdapter):
    """LangBot adapter for Octo IM."""

    name: str = 'octo'

    config: dict

    message_converter: OctoMessageConverter = OctoMessageConverter()
    event_converter: OctoEventConverter = OctoEventConverter()

    _rest: typing.Optional[OctoRestClient] = pydantic.PrivateAttr(default=None)
    _ws: typing.Optional[OctoWSClient] = pydantic.PrivateAttr(default=None)
    _heartbeat_task: typing.Optional[asyncio.Task] = pydantic.PrivateAttr(default=None)
    # One typing-indicator loop per reply channel, stopped when the reply goes out.
    _typing_tasks: dict[str, asyncio.Task] = pydantic.PrivateAttr(default_factory=dict)
    # uid -> display name cache for reply quotes.
    _user_names: dict[str, str] = pydantic.PrivateAttr(default_factory=dict)
    # resp_message_id -> streaming card state.
    _cards: dict[str, _CardState] = pydantic.PrivateAttr(default_factory=dict)
    _card_capability: typing.Optional[octo_cards.CardCapability] = pydantic.PrivateAttr(default=None)
    _card_capability_at: float = pydantic.PrivateAttr(default=0.0)

    listeners: typing.Dict[
        typing.Type[platform_events.Event],
        typing.Callable[[platform_events.Event, abstract_platform_adapter.AbstractMessagePlatformAdapter], None],
    ] = {}

    def __init__(self, config: dict, logger: abstract_platform_logger.AbstractEventLogger):
        super().__init__(
            config=config,
            logger=logger,
            bot_account_id='',
            listeners={},
            name='octo',
        )

    async def send_message(
        self,
        target_type: str,
        target_id: str,
        message: platform_message.MessageChain,
    ):
        """Send a message to a user, a group, or a sub-group (composite id)."""
        if self._rest is None:
            raise RuntimeError('Octo adapter is not running')
        if target_type == 'person':
            channel_type = ChannelType.DM
        else:
            channel_type = ChannelType.COMMUNITY_TOPIC if is_thread_channel(target_id) else ChannelType.GROUP
        self._stop_typing(target_id)
        await self._send_chain(target_id, channel_type, message)

    async def reply_message(
        self,
        message_source: platform_events.MessageEvent,
        message: platform_message.MessageChain,
        quote_origin: bool = False,
    ):
        source_msg = message_source.source_platform_object
        if not isinstance(source_msg, OctoMessage):
            await self.logger.warning('Octo reply without a source platform object, dropping')
            return
        if self._rest is None:
            raise RuntimeError('Octo adapter is not running')

        channel_id, channel_type = self._reply_channel(source_msg)
        self._stop_typing(channel_id)

        reply = None
        if quote_origin:
            from_name = await self._resolve_user_name(source_msg.from_uid)
            reply = OctoMessageConverter.build_reply_quote(source_msg, from_name)
        await self._send_chain(channel_id, channel_type, message, reply=reply)

    async def _send_chain(
        self,
        channel_id: str,
        channel_type: int,
        message: platform_message.MessageChain,
        reply: typing.Optional[dict] = None,
    ) -> None:
        """Send text first (with mention/quote), then each media component."""
        content, mention, media = await OctoMessageConverter.yiri2target(message)
        if content.strip():
            await self._rest.send_text(
                channel_id=channel_id,
                channel_type=channel_type,
                content=content,
                mention=mention,
                reply=reply,
            )
        for component in media:
            try:
                await self._send_media_component(channel_id, channel_type, component)
            except Exception:
                await self.logger.error(
                    f'Octo failed to send {type(component).__name__}: {traceback.format_exc()}'
                )

    async def _send_media_component(
        self,
        channel_id: str,
        channel_type: int,
        component: platform_message.MessageComponent,
    ) -> None:
        data = await self._get_component_bytes(component)
        if not data:
            await self.logger.warning(f'Octo media component {type(component).__name__} has no content, skipped')
            return
        if isinstance(component, platform_message.Image):
            mime = octo_media.sniff_image_mime(data)
            name = f'image{octo_media.extension_for_mime(mime)}'
            url = await self._rest.upload_media(name, data, mime)
            dims = octo_media.sniff_image_dimensions(data)
            await self._rest.send_media(
                channel_id,
                channel_type,
                PayloadType.IMAGE,
                url,
                name=name,
                size=len(data),
                width=dims[0] if dims else None,
                height=dims[1] if dims else None,
            )
        elif isinstance(component, platform_message.Voice):
            name = 'voice.mp3'
            url = await self._rest.upload_media(name, data, 'audio/mpeg')
            await self._rest.send_media(channel_id, channel_type, PayloadType.VOICE, url, name=name, size=len(data))
        else:  # File
            name = getattr(component, 'name', '') or 'file.bin'
            content_type = octo_media.infer_content_type(name)
            url = await self._rest.upload_media(name, data, content_type)
            await self._rest.send_media(channel_id, channel_type, PayloadType.FILE, url, name=name, size=len(data))

    @staticmethod
    async def _get_component_bytes(component: platform_message.MessageComponent) -> typing.Optional[bytes]:
        """Extract raw bytes from an Image/File/Voice component (base64/url/path)."""
        if isinstance(component, platform_message.Image):
            try:
                data, _ = await component.get_bytes()
                if data and len(data) <= octo_media.MAX_UPLOAD_BYTES:
                    return data
                return None
            except Exception:
                pass
        b64_val = getattr(component, 'base64', None)
        url_val = getattr(component, 'url', None)
        path_val = getattr(component, 'path', None)
        if b64_val:
            if ',' in b64_val[:80] and b64_val.startswith('data:'):
                b64_val = b64_val.split(',', 1)[1]
            data = await asyncio.to_thread(base64.b64decode, b64_val)
            return data if len(data) <= octo_media.MAX_UPLOAD_BYTES else None
        if url_val and url_val.startswith(('http://', 'https://')):
            from langbot.pkg.utils import httpclient

            session = httpclient.get_session()
            async with session.get(url_val) as resp:
                if resp.status == 200:
                    return await httpclient.read_limited(resp, max_bytes=octo_media.MAX_UPLOAD_BYTES)
            return None
        if path_val:
            if await asyncio.to_thread(os.path.getsize, path_val) > octo_media.MAX_UPLOAD_BYTES:
                return None

            def read_file() -> bytes:
                with open(path_val, 'rb') as f:
                    return f.read(octo_media.MAX_UPLOAD_BYTES + 1)

            data = await asyncio.to_thread(read_file)
            return data if len(data) <= octo_media.MAX_UPLOAD_BYTES else None
        return None

    async def _resolve_user_name(self, uid: str) -> str:
        """Display name for a uid, via /v1/bot/user/info with a bounded cache."""
        cached = self._user_names.get(uid)
        if cached is not None:
            return cached
        name = ''
        if self._rest is not None:
            info = await self._rest.user_info(uid)
            if info:
                data = info.get('data') if isinstance(info.get('data'), dict) else info
                name = str(data.get('name', '') or '')
        self._user_names[uid] = name
        while len(self._user_names) > 4096:
            self._user_names.pop(next(iter(self._user_names)), None)
        return name

    async def is_muted(self, group_id: int) -> bool:
        return False

    # ─── Streaming card output ──────────────────────────────────────────────

    async def is_stream_output_supported(self) -> bool:
        """Stream only when the operator enabled it and the server allows cards.

        Capability comes from the server, never from a local guess: a send
        probe cannot tell "disabled" from "invalid", and a 404 means the
        endpoint is not deployed at all.
        """
        if not self.config.get('enable-stream-reply'):
            return False
        capability = await self._get_card_capability()
        return capability.can_send_display_card

    async def _get_card_capability(self) -> octo_cards.CardCapability:
        now = time.monotonic()
        if self._card_capability is not None and now - self._card_capability_at < CARD_CAPABILITY_TTL_SECONDS:
            return self._card_capability
        closed = octo_cards.CardCapability(False, False, frozenset(), octo_cards.DEFAULT_MAX_PAYLOAD_BYTES)
        if self._rest is None:
            return closed
        try:
            profile = await self._rest.card_profile()
        except Exception as e:
            await self.logger.warning(f'Octo card capability probe failed, cards disabled: {e}')
            self._card_capability = closed
            self._card_capability_at = now
            return closed
        capability = octo_cards.parse_capability(profile)
        self._card_capability = capability
        self._card_capability_at = now
        await self.logger.info(
            f'Octo card capability: enabled={capability.enabled} profiles={sorted(capability.profiles)}'
        )
        return capability

    async def create_message_card(self, message_id: str, event: platform_events.MessageEvent) -> bool:
        """Send the placeholder card that later chunks edit in place."""
        source_msg = event.source_platform_object
        if not isinstance(source_msg, OctoMessage) or self._rest is None:
            return False
        capability = await self._get_card_capability()
        if not capability.can_send_display_card:
            return False

        channel_id, channel_type = self._reply_channel(source_msg)
        try:
            result = await self._rest.send_card(
                channel_id=channel_id,
                channel_type=channel_type,
                card=octo_cards.build_text_card('…'),
                plain=octo_cards.plain_preview(''),
            )
        except Exception:
            await self.logger.error(f'Octo failed to create stream card: {traceback.format_exc()}')
            return False
        if not result.message_id:
            return False

        self._stop_typing(channel_id)
        self._cards[str(message_id)] = _CardState(
            card_message_id=result.message_id,
            channel_id=channel_id,
            channel_type=channel_type,
        )
        while len(self._cards) > CARD_STATE_MAX:
            self._cards.pop(next(iter(self._cards)), None)
        return True

    async def reply_message_chunk(
        self,
        message_source: platform_events.MessageEvent,
        bot_message: typing.Any,
        message: platform_message.MessageChain,
        quote_origin: bool = False,
        is_final: bool = False,
    ):
        """Update the streaming card with the accumulated reply.

        Each chunk carries the full text, so the card is replaced rather than
        appended to, and a throttled-away frame is superseded by the next one.
        """
        state = self._cards.get(str(getattr(bot_message, 'resp_message_id', '')))
        if state is None:
            # No card (creation failed, or cards disabled mid-stream): only the
            # terminal chunk is worth sending, as a plain message.
            if is_final:
                await self.reply_message(message_source, message, quote_origin)
            return

        content, _, media = await OctoMessageConverter.yiri2target(message)
        capability = await self._get_card_capability()
        text = octo_cards.fit_text_to_payload(content, capability.max_payload_bytes)

        async with state.lock:
            if state.finalized:
                return
            if text == state.last_text and not is_final:
                return
            now = time.monotonic()
            if not is_final and now - state.last_edit_at < CARD_EDIT_INTERVAL_SECONDS:
                return
            # Reserved inside the lock so reservation order equals wire order:
            # a higher seq committing first wedges the card permanently.
            seq = state.next_seq()
            try:
                await self._rest.edit_card(
                    message_id=state.card_message_id,
                    channel_id=state.channel_id,
                    channel_type=state.channel_type,
                    card=octo_cards.build_text_card(text or '…'),
                    plain=octo_cards.plain_preview(text),
                    card_seq=seq,
                    # Terminal frames must enter the revision history.
                    transient=not is_final,
                )
            except Exception:
                await self.logger.error(f'Octo card edit failed: {traceback.format_exc()}')
                if not is_final:
                    return
            state.last_text = text
            state.last_edit_at = now
            if is_final:
                state.finalized = True

        if is_final:
            self._cards.pop(str(getattr(bot_message, 'resp_message_id', '')), None)
            for component in media:
                try:
                    await self._send_media_component(state.channel_id, state.channel_type, component)
                except Exception:
                    await self.logger.error(f'Octo failed to send streamed media: {traceback.format_exc()}')

    def register_listener(
        self,
        event_type: typing.Type[platform_events.Event],
        callback: typing.Callable[
            [platform_events.Event, abstract_platform_adapter.AbstractMessagePlatformAdapter],
            None,
        ],
    ):
        self.listeners[event_type] = callback

    def unregister_listener(
        self,
        event_type: typing.Type[platform_events.Event],
        callback: typing.Callable[
            [platform_events.Event, abstract_platform_adapter.AbstractMessagePlatformAdapter],
            None,
        ],
    ):
        self.listeners.pop(event_type, None)

    async def run_async(self):
        api_url = self.config.get('api_url', '')
        bot_token = self.config.get('bot_token', '')
        if not api_url or not bot_token:
            raise ValueError('Octo adapter requires api_url and bot_token')

        self._rest = OctoRestClient(api_url=api_url, bot_token=bot_token)
        await self.logger.info('Octo adapter registering bot...')
        register_result = await self._rest.register()
        self.bot_account_id = register_result.robot_id

        ws_url = self.config.get('ws_url', '') or register_result.ws_url
        if not ws_url:
            raise ValueError('Octo register returned no ws_url and none is configured')

        self._ws = OctoWSClient(
            ws_url=ws_url,
            uid=register_result.robot_id,
            token=register_result.im_token,
            on_message=self._handle_inbound_message,
            on_auth_error=self._refresh_credentials,
            logger=self.logger,
        )
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        await self.logger.info(f'Octo adapter running, robot_id={register_result.robot_id}')
        await self._ws.run()

    async def _refresh_credentials(self) -> typing.Optional[tuple[str, str, str]]:
        """Re-register with force_refresh to obtain a fresh im_token."""
        assert self._rest is not None
        result = await self._rest.register(force_refresh=True)
        self.bot_account_id = result.robot_id
        return result.robot_id, result.im_token, result.ws_url

    async def _heartbeat_loop(self):
        """REST heartbeat; failures must never touch the WS connection."""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            if self._rest is None:
                return
            if self._ws is None or not self._ws.connected.is_set():
                continue
            try:
                await self._rest.heartbeat()
            except Exception as e:
                await self.logger.warning(f'Octo heartbeat failed (ignored): {e}')

    async def _handle_inbound_message(self, msg: OctoMessage):
        try:
            downloads = await self._download_inbound_media(msg)
            event = await OctoEventConverter.target2yiri(msg, self.bot_account_id, downloads)
        except Exception:
            await self.logger.error(f'Octo event conversion error: {traceback.format_exc()}')
            return
        if event is None:
            return
        listener = self.listeners.get(type(event))
        if listener is not None:
            if self._will_likely_reply(event):
                self._start_typing(msg)
            await listener(event, self)

    def _will_likely_reply(self, event: platform_events.MessageEvent) -> bool:
        """Whether to show a typing indicator for this message.

        Group messages are dropped by the default at-bot response rule unless
        the bot is mentioned, and typing on a message that is never answered
        reads as a hung bot. Mirror that rule rather than indicating on every
        inbound message.
        """
        if isinstance(event, platform_events.FriendMessage):
            return True
        return any(
            isinstance(component, platform_message.At) and str(component.target) == str(self.bot_account_id)
            for component in event.message_chain
        )

    @staticmethod
    def _reply_channel(msg: OctoMessage) -> tuple[str, int]:
        """Channel a reply to msg goes to: DM peers are addressed by uid."""
        if msg.channel_type == ChannelType.DM:
            return msg.from_uid, ChannelType.DM
        return msg.channel_id, msg.channel_type

    def _start_typing(self, msg: OctoMessage) -> None:
        """Read receipt + typing loop while the pipeline composes a reply."""
        if self._rest is None:
            return
        channel_id, channel_type = self._reply_channel(msg)
        self._stop_typing(channel_id)
        self._typing_tasks[channel_id] = asyncio.create_task(
            self._typing_loop(channel_id, channel_type, msg.message_id)
        )

    def _stop_typing(self, channel_id: str) -> None:
        task = self._typing_tasks.pop(channel_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def _typing_loop(self, channel_id: str, channel_type: int, message_id: str) -> None:
        rest = self._rest
        if rest is None:
            return
        try:
            await rest.read_receipt(channel_id, channel_type, [message_id])
        except Exception:
            pass
        for _ in range(int(TYPING_MAX_SECONDS / TYPING_INTERVAL_SECONDS)):
            try:
                await rest.typing(channel_id, channel_type)
            except Exception:
                pass
            await asyncio.sleep(TYPING_INTERVAL_SECONDS)

    async def _download_inbound_media(self, msg: OctoMessage) -> dict[str, tuple[bytes, str]]:
        """Download media referenced by an inbound message, keyed by payload url.

        Inline media (image/gif/voice, and richtext images) is public-read: no
        auth. File content requires the Bearer token. Oversized or failed
        downloads are skipped; the converter falls back to placeholders.
        """
        downloads: dict[str, tuple[bytes, str]] = {}
        if self._rest is None:
            return downloads
        api_url = self.config.get('api_url', '')
        cdn_url = self.config.get('cdn_url', '')
        payload = msg.payload

        async def fetch(rel_url: typing.Optional[str], with_auth: bool, mime_hint: str = '') -> None:
            if not rel_url or rel_url in downloads:
                return
            full_url = octo_media.build_media_url(rel_url, api_url, cdn_url)
            if not full_url:
                return
            try:
                data = await self._rest.download_media(
                    full_url, with_auth=with_auth, max_bytes=octo_media.MAX_INBOUND_INLINE_BYTES
                )
            except Exception:
                await self.logger.warning(f'Octo media download failed: {traceback.format_exc()}')
                return
            if data:
                mime = octo_media.sniff_image_mime(data)
                if mime == 'application/octet-stream' and mime_hint:
                    mime = mime_hint
                if mime.startswith('image/') and not octo_media.is_complete_image(data, mime):
                    # Vision models reject a truncated image with an opaque
                    # parse error; drop it and let the placeholder stand in.
                    await self.logger.warning(
                        f'Octo image download incomplete ({len(data)} bytes, {mime}), skipping inline'
                    )
                    return
                downloads[rel_url] = (data, mime)

        if payload.type in (PayloadType.IMAGE, PayloadType.GIF):
            await fetch(payload.url, with_auth=False)
        elif payload.type == PayloadType.VOICE:
            await fetch(payload.url, with_auth=False, mime_hint='audio/mpeg')
        elif payload.type == PayloadType.FILE:
            await fetch(payload.url, with_auth=True, mime_hint='application/octet-stream')
        elif payload.type == PayloadType.RICH_TEXT and isinstance(payload.content, list):
            image_urls = [
                block.get('url')
                for block in payload.content
                if isinstance(block, dict) and block.get('type') == 'image' and block.get('url')
            ]
            for url in image_urls[:4]:
                await fetch(url, with_auth=False)
        return downloads

    async def kill(self) -> bool:
        for channel_id in list(self._typing_tasks):
            self._stop_typing(channel_id)
        self._cards.clear()
        self._card_capability = None
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self._ws is not None:
            await self._ws.stop()
            self._ws = None
        if self._rest is not None:
            await self._rest.close()
            self._rest = None
        await self.logger.info('Octo adapter stopped')
        return True
