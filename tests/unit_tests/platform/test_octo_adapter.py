"""Unit tests for the Octo adapter converters."""

import pytest

from langbot.libs.octo_api.types import MessagePayload, OctoMessage
from langbot.pkg.platform.sources.octo import (
    OctoEventConverter,
    OctoMessageConverter,
    _utf16_index_map,
    _utf16_len,
)

import langbot_plugin.api.entities.builtin.platform.events as platform_events
import langbot_plugin.api.entities.builtin.platform.message as platform_message

BOT_UID = 'bot32hexbot32hexbot32hexbot32hex'


def _msg(payload: dict, channel_type: int = 2, channel_id: str = 'grp_1', from_uid: str = 'user1') -> OctoMessage:
    return OctoMessage(
        message_id='7385920184320000001',
        message_seq=1,
        from_uid=from_uid,
        channel_id=channel_id,
        channel_type=channel_type,
        timestamp=1753939200,
        payload=MessagePayload.model_validate(payload),
    )


class TestUtf16:
    def test_len_counts_utf16_units(self):
        assert _utf16_len('abc') == 3
        assert _utf16_len('你好') == 2
        assert _utf16_len('👍') == 2  # astral plane = 2 code units

    def test_index_map_with_astral_chars(self):
        text = '👍@Bot hi'
        mapping = _utf16_index_map(text)
        assert mapping[0] == 0
        assert mapping[2] == 1  # after the emoji (2 units), python index 1
        assert mapping[_utf16_len(text)] == len(text)


class TestInboundMentions:
    @pytest.mark.asyncio
    async def test_entity_split_with_utf16_offsets(self):
        # '👍' shifts UTF-16 offsets by 2 while occupying one Python char.
        text = '👍@Bot 帮我看看'
        payload = {
            'type': 1,
            'content': text,
            'mention': {
                'uids': [BOT_UID],
                'entities': [{'uid': BOT_UID, 'offset': 2, 'length': 4}],
            },
        }
        chain = await OctoMessageConverter.target2yiri(_msg(payload), BOT_UID)
        components = list(chain)
        assert isinstance(components[0], platform_message.Plain)
        assert components[0].text == '👍'
        assert isinstance(components[1], platform_message.At)
        assert components[1].target == BOT_UID
        assert components[1].display == 'Bot'
        assert isinstance(components[2], platform_message.Plain)
        assert components[2].text == ' 帮我看看'

    @pytest.mark.asyncio
    async def test_broadcast_suppresses_ais(self):
        # The server double-writes @所有人 as humans+ais; ais alone must not
        # count as an At on the bot when a broadcast flag is present.
        payload = {
            'type': 1,
            'content': '@所有人 大家好',
            'mention': {'humans': 1, 'ais': 1},
        }
        chain = await OctoMessageConverter.target2yiri(_msg(payload), BOT_UID)
        components = list(chain)
        assert any(isinstance(c, platform_message.AtAll) for c in components)
        assert not any(isinstance(c, platform_message.At) for c in components)

    @pytest.mark.asyncio
    async def test_ais_without_broadcast_mentions_bot(self):
        payload = {'type': 1, 'content': '@所有AI 在吗', 'mention': {'ais': True}}
        chain = await OctoMessageConverter.target2yiri(_msg(payload), BOT_UID)
        ats = [c for c in chain if isinstance(c, platform_message.At)]
        assert len(ats) == 1
        assert ats[0].target == BOT_UID

    @pytest.mark.asyncio
    async def test_boolean_and_numeric_flags_both_accepted(self):
        for flag in (True, 1):
            payload = {'type': 1, 'content': 'x', 'mention': {'humans': flag}}
            chain = await OctoMessageConverter.target2yiri(_msg(payload), BOT_UID)
            assert any(isinstance(c, platform_message.AtAll) for c in chain)


class TestOutbound:
    @pytest.mark.asyncio
    async def test_plain_text(self):
        chain = platform_message.MessageChain([platform_message.Plain(text='hello')])
        content, mention = await OctoMessageConverter.yiri2target(chain)
        assert content == 'hello'
        assert mention is None

    @pytest.mark.asyncio
    async def test_at_builds_utf16_entities(self):
        chain = platform_message.MessageChain(
            [
                platform_message.Plain(text='👍'),
                platform_message.At(target='uid1', display='张三'),
                platform_message.Plain(text='收到'),
            ]
        )
        content, mention = await OctoMessageConverter.yiri2target(chain)
        assert content == '👍@张三 收到'
        entity = mention['entities'][0]
        assert entity['uid'] == 'uid1'
        assert entity['offset'] == 2  # after the astral emoji
        assert entity['length'] == 3  # '@张三' including the @ sign
        assert mention['uids'] == ['uid1']

    @pytest.mark.asyncio
    async def test_at_all(self):
        chain = platform_message.MessageChain([platform_message.AtAll()])
        content, mention = await OctoMessageConverter.yiri2target(chain)
        assert '@所有人' in content
        assert mention['all'] == 1


class TestRichAndCardInbound:
    @pytest.mark.asyncio
    async def test_richtext_blocks(self):
        payload = {
            'type': 14,
            'content': [
                {'type': 'text', 'text': 'hello '},
                {'type': 'image', 'url': 'file/abc.png'},
                {'type': 'text', 'text': 'world'},
            ],
        }
        chain = await OctoMessageConverter.target2yiri(_msg(payload), BOT_UID)
        texts = [c.text for c in chain if isinstance(c, platform_message.Plain)]
        assert texts == ['hello ', 'world']

    @pytest.mark.asyncio
    async def test_interactive_card_uses_server_plain(self):
        payload = {'type': 17, 'card': {'type': 'AdaptiveCard'}, 'plain': '审批单：请查看'}
        chain = await OctoMessageConverter.target2yiri(_msg(payload), BOT_UID)
        components = list(chain)
        assert isinstance(components[0], platform_message.Plain)
        assert components[0].text == '审批单：请查看'

    @pytest.mark.asyncio
    async def test_quote_from_reply(self):
        payload = {
            'type': 1,
            'content': '同意',
            'reply': {
                'payload': {'type': 1, 'content': '这个方案可以吗'},
                'from_uid': 'user2',
                'from_name': '李四',
            },
        }
        chain = await OctoMessageConverter.target2yiri(_msg(payload), BOT_UID)
        quotes = [c for c in chain if isinstance(c, platform_message.Quote)]
        assert len(quotes) == 1
        assert str(quotes[0].origin) == '这个方案可以吗'


class TestEventConverter:
    @pytest.mark.asyncio
    async def test_dm_becomes_friend_message(self):
        msg = _msg({'type': 1, 'content': 'hi'}, channel_type=1, channel_id='s14_a@s14_b', from_uid='s14_user1')
        event = await OctoEventConverter.target2yiri(msg, BOT_UID)
        assert isinstance(event, platform_events.FriendMessage)
        assert event.sender.id == 's14_user1'

    @pytest.mark.asyncio
    async def test_group_message(self):
        msg = _msg({'type': 1, 'content': 'hi'}, channel_type=2, channel_id='grp_1')
        event = await OctoEventConverter.target2yiri(msg, BOT_UID)
        assert isinstance(event, platform_events.GroupMessage)
        assert event.sender.group.id == 'grp_1'

    @pytest.mark.asyncio
    async def test_thread_message_keeps_composite_channel_id(self):
        msg = _msg({'type': 1, 'content': 'hi'}, channel_type=5, channel_id='grp_1____t7x9')
        event = await OctoEventConverter.target2yiri(msg, BOT_UID)
        assert isinstance(event, platform_events.GroupMessage)
        # The composite id is both the session key and a valid send target.
        assert event.sender.group.id == 'grp_1____t7x9'

    @pytest.mark.asyncio
    async def test_system_event_payload_dropped(self):
        msg = _msg({'type': 1, 'content': '', 'event': {'type': 'group_md_updated'}})
        assert await OctoEventConverter.target2yiri(msg, BOT_UID) is None

    @pytest.mark.asyncio
    async def test_own_message_dropped(self):
        msg = _msg({'type': 1, 'content': 'echo'}, from_uid=BOT_UID)
        assert await OctoEventConverter.target2yiri(msg, BOT_UID) is None
        # Space-prefixed self uid is also filtered.
        msg = _msg({'type': 1, 'content': 'echo'}, from_uid=f's14_{BOT_UID}')
        assert await OctoEventConverter.target2yiri(msg, BOT_UID) is None

    @pytest.mark.asyncio
    async def test_unknown_channel_type_dropped(self):
        msg = _msg({'type': 1, 'content': 'hi'}, channel_type=9)
        assert await OctoEventConverter.target2yiri(msg, BOT_UID) is None
