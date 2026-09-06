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
from langbot.libs.octo_api.types import MentionInfo, is_thread_channel

import langbot_plugin.api.definition.abstract.platform.adapter as abstract_platform_adapter
import langbot_plugin.api.definition.abstract.platform.event_logger as abstract_platform_logger
import langbot_plugin.api.entities.builtin.platform.entities as platform_entities
import langbot_plugin.api.entities.builtin.platform.events as platform_events
import langbot_plugin.api.entities.builtin.platform.message as platform_message

HEARTBEAT_INTERVAL_SECONDS = 30.0


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
    ) -> tuple[str, typing.Optional[dict]]:
        """Convert a MessageChain to (text content, mention dict or None).

        Mention entity offsets are computed in UTF-16 code units over the
        final content, and length includes the '@' sign.
        """
        content = ''
        uids: list[str] = []
        entities: list[dict] = []
        mention_all = False

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
                        sub_content, _ = await OctoMessageConverter.yiri2target(node.message_chain)
                        content += sub_content + '\n'
            elif isinstance(component, platform_message.Image):
                # Media outbound lands in P2; keep the conversation coherent.
                content += '[image]'
            elif isinstance(component, platform_message.File):
                content += f'[file: {component.name or ""}]'

        mention: typing.Optional[dict] = None
        if uids or entities or mention_all:
            mention = {}
            if uids:
                mention['uids'] = uids
            if entities:
                mention['entities'] = entities
            if mention_all:
                mention['all'] = 1
        return content, mention

    @staticmethod
    async def target2yiri(msg: OctoMessage, bot_uid: str) -> platform_message.MessageChain:
        """Convert an inbound Octo message to a MessageChain."""
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
            components.extend(OctoMessageConverter._richtext_components(payload))
        elif payload.type == PayloadType.INTERACTIVE_CARD:
            # Never parse the card tree inbound; the server-generated plain
            # text is authoritative.
            plain = getattr(payload, 'plain', None) or '[卡片]'
            components.append(platform_message.Plain(text=plain))
        elif payload.type in (PayloadType.IMAGE, PayloadType.GIF):
            components.append(platform_message.Unknown(text='[Image]'))
        elif payload.type == PayloadType.VOICE:
            components.append(platform_message.Unknown(text='[Voice]'))
        elif payload.type == PayloadType.VIDEO:
            components.append(platform_message.Unknown(text='[Video]'))
        elif payload.type == PayloadType.FILE:
            components.append(platform_message.Unknown(text=f'[File: {payload.name or ""}]'))
        else:
            components.append(platform_message.Unknown(text='[Unsupported message type]'))

        return platform_message.MessageChain(components)

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
    def _richtext_components(payload) -> list[platform_message.MessageComponent]:
        """RichText(14) content is an ordered block array; string content is
        the backward-compatible single text block."""
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
            components.append(platform_message.At(target=entity.uid, display=display))
            mentioned_uids.add(entity.uid)
            cursor = end
        if not entities and mention.uids:
            # Fallback when the server sent uids without entity spans.
            mentioned_uids.update(mention.uids)
            for uid in mention.uids:
                components.append(platform_message.At(target=uid))
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
        msg: OctoMessage, bot_uid: str
    ) -> typing.Optional[platform_events.MessageEvent]:
        # System events (group_md_updated etc.) ride on ordinary messages.
        if msg.payload.event is not None:
            return None
        # Never react to the bot's own messages.
        if msg.from_uid == bot_uid or msg.from_uid.endswith(f'_{bot_uid}'):
            return None
        if not msg.from_uid:
            return None

        message_chain = await OctoMessageConverter.target2yiri(msg, bot_uid)
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
        content, mention = await OctoMessageConverter.yiri2target(message)
        if not content.strip():
            return
        await self._rest.send_text(
            channel_id=target_id,
            channel_type=channel_type,
            content=content,
            mention=mention,
        )

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

        if source_msg.channel_type == ChannelType.DM:
            # An inbound DM channel_id is a compound "sX_a@sX_b"; outbound DMs
            # address the peer uid directly.
            channel_id = source_msg.from_uid
            channel_type = ChannelType.DM
        else:
            channel_id = source_msg.channel_id
            channel_type = source_msg.channel_type

        content, mention = await OctoMessageConverter.yiri2target(message)
        if not content.strip():
            return
        await self._rest.send_text(
            channel_id=channel_id,
            channel_type=channel_type,
            content=content,
            mention=mention,
            reply_message_id=source_msg.message_id if quote_origin else None,
        )

    async def is_muted(self, group_id: int) -> bool:
        return False

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
            event = await OctoEventConverter.target2yiri(msg, self.bot_account_id)
        except Exception:
            await self.logger.error(f'Octo event conversion error: {traceback.format_exc()}')
            return
        if event is None:
            return
        listener = self.listeners.get(type(event))
        if listener is not None:
            await listener(event, self)

    async def kill(self) -> bool:
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
