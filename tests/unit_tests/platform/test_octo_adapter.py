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
        content, mention, media = await OctoMessageConverter.yiri2target(chain)
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
        content, mention, media = await OctoMessageConverter.yiri2target(chain)
        assert content == '👍@张三 收到'
        entity = mention['entities'][0]
        assert entity['uid'] == 'uid1'
        assert entity['offset'] == 2  # after the astral emoji
        assert entity['length'] == 3  # '@张三' including the @ sign
        assert mention['uids'] == ['uid1']

    @pytest.mark.asyncio
    async def test_at_all(self):
        chain = platform_message.MessageChain([platform_message.AtAll()])
        content, mention, media = await OctoMessageConverter.yiri2target(chain)
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


class TestReplyQuote:
    def test_build_reply_quote_full_nested_structure(self):
        # A bare message_id renders as an empty quote box on the server;
        # the full nested structure is required.
        msg = _msg({'type': 1, 'content': '原始消息', 'mention': {'ais': 1}})
        quote = OctoMessageConverter.build_reply_quote(msg, '张三')
        assert quote['message_id'] == msg.message_id
        assert quote['from_uid'] == 'user1'
        assert quote['from_name'] == '张三'
        assert quote['payload']['content'] == '原始消息'
        # mention/reply/event must be stripped from the quoted payload.
        assert 'mention' not in quote['payload']

    def test_build_reply_quote_falls_back_to_uid(self):
        msg = _msg({'type': 1, 'content': 'hi'})
        quote = OctoMessageConverter.build_reply_quote(msg, '')
        assert quote['from_name'] == 'user1'


class TestMedia:
    @pytest.mark.asyncio
    async def test_outbound_collects_media_components(self):
        chain = platform_message.MessageChain(
            [
                platform_message.Plain(text='看这张图'),
                platform_message.Image(base64='data:image/png;base64,aGk='),
                platform_message.File(name='report.pdf'),
            ]
        )
        content, mention, media = await OctoMessageConverter.yiri2target(chain)
        assert content == '看这张图'
        assert len(media) == 2
        assert isinstance(media[0], platform_message.Image)
        assert isinstance(media[1], platform_message.File)

    @pytest.mark.asyncio
    async def test_inbound_image_with_download(self):
        png = b'\x89PNG\r\n\x1a\n' + b'\x00' * 30
        payload = {'type': 2, 'url': 'file/abc.png', 'name': 'abc.png', 'size': 38}
        chain = await OctoMessageConverter.target2yiri(
            _msg(payload), BOT_UID, downloads={'file/abc.png': (png, 'image/png')}
        )
        images = [c for c in chain if isinstance(c, platform_message.Image)]
        assert len(images) == 1
        assert images[0].base64.startswith('data:image/png;base64,')

    @pytest.mark.asyncio
    async def test_inbound_image_without_download_falls_back(self):
        payload = {'type': 2, 'url': 'file/abc.png'}
        chain = await OctoMessageConverter.target2yiri(_msg(payload), BOT_UID)
        assert any(isinstance(c, platform_message.Unknown) for c in chain)

    @pytest.mark.asyncio
    async def test_inbound_file_with_download(self):
        payload = {'type': 8, 'url': 'file/doc.txt', 'name': 'doc.txt', 'size': 5}
        chain = await OctoMessageConverter.target2yiri(
            _msg(payload), BOT_UID, downloads={'file/doc.txt': (b'hello', 'text/plain')}
        )
        files = [c for c in chain if isinstance(c, platform_message.File)]
        assert len(files) == 1
        assert files[0].name == 'doc.txt'


class TestMediaHelpers:
    def test_build_media_url(self):
        from langbot.libs.octo_api import media as m

        api = 'https://im.example.com/api'
        assert m.build_media_url('http://x/y.png', api) == 'http://x/y.png'
        assert m.build_media_url('file/a/b.png', api) == 'https://im.example.com/api/file/a/b.png'
        assert m.build_media_url('file/preview/a.png', api) == 'https://im.example.com/api/file/a.png'
        assert m.build_media_url('file/a.png', api, 'https://cdn.example.com') == 'https://cdn.example.com/a.png'
        assert m.build_media_url('', api) is None

    def test_sniff_png_dimensions(self):
        import struct
        from langbot.libs.octo_api import media as m

        header = b'\x89PNG\r\n\x1a\n' + b'\x00\x00\x00\rIHDR' + struct.pack('>II', 640, 480) + b'\x00' * 10
        assert m.sniff_image_dimensions(header) == (640, 480)
        assert m.sniff_image_mime(header) == 'image/png'

    def test_sniff_gif_and_jpeg(self):
        import struct
        from langbot.libs.octo_api import media as m

        gif = b'GIF89a' + struct.pack('<HH', 100, 50) + b'\x00' * 20
        assert m.sniff_image_dimensions(gif) == (100, 50)
        jpeg = b'\xff\xd8\xff\xc0\x00\x11\x08' + struct.pack('>HH', 480, 640) + b'\x00' * 20
        assert m.sniff_image_dimensions(jpeg) == (640, 480)
        assert m.sniff_image_mime(jpeg) == 'image/jpeg'


class TestImageIntegrity:
    def test_complete_and_truncated_png(self):
        from langbot.libs.octo_api import media as m

        png = b'\x89PNG\r\n\x1a\n' + b'\x00' * 40 + b'IEND\xaeB`\x82'
        assert m.is_complete_image(png, 'image/png')
        # A truncated PNG keeps a valid header but loses its IEND marker.
        assert not m.is_complete_image(png[:-4], 'image/png')

    def test_complete_and_truncated_jpeg(self):
        from langbot.libs.octo_api import media as m

        jpeg = b'\xff\xd8' + b'\x00' * 40 + b'\xff\xd9'
        assert m.is_complete_image(jpeg, 'image/jpeg')
        assert not m.is_complete_image(jpeg[:-2], 'image/jpeg')

    def test_non_image_mime_is_not_checked(self):
        from langbot.libs.octo_api import media as m

        assert m.is_complete_image(b'anything', 'application/octet-stream')


class TestDownloadMedia:
    @pytest.mark.asyncio
    async def test_download_reads_whole_body_across_chunks(self, aiohttp_server_factory):
        """Regression: StreamReader.read(n) returns only buffered bytes, which
        silently truncated multi-chunk downloads."""
        from langbot.libs.octo_api import OctoRestClient

        payload = bytes(range(256)) * 40  # ~10KB, delivered as delayed 1KB chunks
        url = await aiohttp_server_factory(payload)
        client = OctoRestClient(api_url='http://unused', bot_token='t')
        try:
            got = await client.download_media(url, with_auth=False, max_bytes=10 * 1024 * 1024)
        finally:
            await client.close()
        assert got == payload

    @pytest.mark.asyncio
    async def test_download_rejects_oversize(self, aiohttp_server_factory):
        from langbot.libs.octo_api import OctoRestClient

        url = await aiohttp_server_factory(b'x' * 5000)
        client = OctoRestClient(api_url='http://unused', bot_token='t')
        try:
            got = await client.download_media(url, with_auth=False, max_bytes=1000)
        finally:
            await client.close()
        assert got is None


class TestCardCapability:
    def test_fails_closed_when_endpoint_missing(self):
        from langbot.libs.octo_api import cards as c

        cap = c.parse_capability({'available': False, 'enabled': False})
        assert not cap.can_send_display_card

    def test_fails_closed_when_bot_policy_disables_cards(self):
        from langbot.libs.octo_api import cards as c

        cap = c.parse_capability(
            {'available': True, 'enabled': True, 'profiles': ['octo/v1'], 'config': {'card_enabled': False}}
        )
        assert not cap.can_send_display_card

    def test_enabled_with_display_profile(self):
        from langbot.libs.octo_api import cards as c

        cap = c.parse_capability(
            {
                'available': True,
                'enabled': 1,  # numeric serialization
                'profiles': ['octo/v1', 'octo/v2'],
                'config': {'card_enabled': 1},
                'limits': {'max_payload_bytes': 524288},
            }
        )
        assert cap.can_send_display_card
        assert cap.max_payload_bytes == 524288

    def test_missing_profiles_fails_closed(self):
        from langbot.libs.octo_api import cards as c

        cap = c.parse_capability({'available': True, 'enabled': True})
        assert not cap.can_send_display_card


class TestCardBuilding:
    def test_text_card_shape(self):
        from langbot.libs.octo_api import cards as c

        card = c.build_text_card('你好')
        assert card['type'] == 'AdaptiveCard'
        assert card['version'] == '1.5'
        assert card['body'][0]['text'] == '你好'
        assert card['body'][0]['wrap'] is True

    def test_plain_preview_never_empty(self):
        from langbot.libs.octo_api import cards as c

        assert c.plain_preview('') == '[卡片]'
        assert c.plain_preview('   \n ') == '[卡片]'
        assert c.plain_preview('hello\n\nworld') == 'hello world'
        long = c.plain_preview('字' * 500)
        assert len(long) <= c.PLAIN_PREVIEW_CHARS

    def test_fit_text_respects_envelope_limit(self):
        import json
        from langbot.libs.octo_api import cards as c

        limit = 4096
        fitted = c.fit_text_to_payload('长' * 20000, limit)
        envelope = {
            'type': 17,
            'card': c.build_text_card(fitted),
            'plain': c.plain_preview(fitted),
            'profile': c.PROFILE_DISPLAY,
            'card_version': c.CARD_VERSION,
            'card_seq': 2**31,
            'transient': True,
        }
        assert len(json.dumps(envelope, ensure_ascii=False).encode()) <= limit
        assert fitted.endswith('（内容过长，已截断）')

    def test_short_text_is_untouched(self):
        from langbot.libs.octo_api import cards as c

        assert c.fit_text_to_payload('短文本', 524288) == '短文本'


class TestStreamingCard:
    @staticmethod
    def _adapter(cap_enabled=True, stream=True):
        import langbot.pkg.platform.sources.octo as octo_mod

        import langbot_plugin.api.definition.abstract.platform.event_logger as abstract_logger

        class _Logger(abstract_logger.AbstractEventLogger):
            async def info(self, *a, **k): pass
            async def debug(self, *a, **k): pass
            async def warning(self, *a, **k): pass
            async def error(self, *a, **k): pass

        adapter = octo_mod.OctoAdapter(
            config={'api_url': 'http://x/api', 'bot_token': 'bf_x', 'enable-stream-reply': stream},
            logger=_Logger(),
        )
        from langbot.libs.octo_api import cards as c

        import time as _t
        adapter._card_capability_at = _t.monotonic()
        adapter._card_capability = c.CardCapability(
            available=True, enabled=cap_enabled,
            profiles=frozenset({'octo/v1'}) if cap_enabled else frozenset(),
            max_payload_bytes=524288,
        )
        return adapter

    class _FakeRest:
        def __init__(self):
            self.edits = []
            self.texts = []
            self.sent_cards = 0

        async def send_text(self, **kw):
            from langbot.libs.octo_api.types import SendMessageResult
            self.texts.append(kw)
            return SendMessageResult(message_id='text-msg-1')

        async def user_info(self, uid):
            return None

        async def send_card(self, **kw):
            from langbot.libs.octo_api.types import SendMessageResult
            self.sent_cards += 1
            return SendMessageResult(message_id='card-msg-1')

        async def edit_card(self, **kw):
            self.edits.append(kw)

    class _BotMessage:
        def __init__(self, rid='r1'):
            self.resp_message_id = rid

    @pytest.mark.asyncio
    async def test_streaming_disabled_by_config(self):
        adapter = self._adapter(stream=False)
        assert await adapter.is_stream_output_supported() is False

    @pytest.mark.asyncio
    async def test_streaming_disabled_when_server_forbids_cards(self):
        adapter = self._adapter(cap_enabled=False)
        assert await adapter.is_stream_output_supported() is False

    @pytest.mark.asyncio
    async def test_streaming_enabled(self):
        adapter = self._adapter()
        assert await adapter.is_stream_output_supported() is True

    @pytest.mark.asyncio
    async def test_card_seq_is_monotonic_and_final_is_not_transient(self):
        adapter = self._adapter()
        rest = self._FakeRest()
        adapter._rest = rest
        msg = _msg({'type': 1, 'content': 'hi'}, channel_type=1, from_uid='u1')
        event = await OctoEventConverter.target2yiri(msg, BOT_UID)

        assert await adapter.create_message_card('r1', event) is True
        assert rest.sent_cards == 1

        bot_msg = self._BotMessage('r1')
        # Force each chunk past the debounce window.
        for text in ('一', '一二', '一二三'):
            adapter._cards['r1'].last_edit_at = 0.0
            chain = platform_message.MessageChain([platform_message.Plain(text=text)])
            await adapter.reply_message_chunk(msg_event_stub(event), bot_msg, chain, is_final=(text == '一二三'))

        seqs = [e['card_seq'] for e in rest.edits]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), f'card_seq must be monotonic: {seqs}'
        assert all(e['transient'] for e in rest.edits[:-1]), 'progress frames must be transient'
        assert rest.edits[-1]['transient'] is False, 'terminal frame must enter revision history'
        assert rest.edits[-1]['card']['body'][0]['text'] == '一二三'
        assert 'r1' not in adapter._cards, 'card state must be released after the final frame'

    @pytest.mark.asyncio
    async def test_debounce_skips_rapid_frames_but_final_always_sends(self):
        adapter = self._adapter()
        rest = self._FakeRest()
        adapter._rest = rest
        msg = _msg({'type': 1, 'content': 'hi'}, channel_type=1, from_uid='u1')
        event = await OctoEventConverter.target2yiri(msg, BOT_UID)
        await adapter.create_message_card('r1', event)
        bot_msg = self._BotMessage('r1')

        # Back-to-back chunks: the debounce window suppresses the middle ones.
        for text in ('a', 'ab', 'abc'):
            chain = platform_message.MessageChain([platform_message.Plain(text=text)])
            await adapter.reply_message_chunk(msg_event_stub(event), bot_msg, chain, is_final=False)
        assert len(rest.edits) <= 1

        chain = platform_message.MessageChain([platform_message.Plain(text='abcd')])
        await adapter.reply_message_chunk(msg_event_stub(event), bot_msg, chain, is_final=True)
        assert rest.edits[-1]['card']['body'][0]['text'] == 'abcd'
        assert rest.edits[-1]['transient'] is False

    @pytest.mark.asyncio
    async def test_unknown_card_falls_back_to_plain_reply_on_final(self):
        adapter = self._adapter()
        rest = self._FakeRest()
        adapter._rest = rest
        msg = _msg({'type': 1, 'content': 'hi'}, channel_type=1, from_uid='u1')
        event = await OctoEventConverter.target2yiri(msg, BOT_UID)
        chain = platform_message.MessageChain([platform_message.Plain(text='done')])

        # No card exists for this id: intermediate chunks are dropped and only
        # the terminal one is delivered, as a plain message.
        await adapter.reply_message_chunk(event, self._BotMessage('missing'), chain, is_final=False)
        assert rest.texts == []
        await adapter.reply_message_chunk(event, self._BotMessage('missing'), chain, is_final=True)
        assert [t['content'] for t in rest.texts] == ['done']
        assert rest.edits == []


def msg_event_stub(event):
    return event


class TestCapabilityCacheTTL:
    @pytest.mark.asyncio
    async def test_capability_is_reprobed_after_ttl(self):
        import langbot.pkg.platform.sources.octo as octo_mod

        adapter = TestStreamingCard._adapter()
        probes = []

        class _Rest:
            async def card_profile(self):
                probes.append(1)
                return {'available': True, 'enabled': True, 'profiles': ['octo/v1'],
                        'config': {'card_enabled': True}, 'limits': {'max_payload_bytes': 524288}}

        adapter._rest = _Rest()
        # Fresh cache: no probe.
        await adapter._get_card_capability()
        assert probes == []
        # Expire it: the server policy is re-read rather than trusted forever.
        adapter._card_capability_at -= octo_mod.CARD_CAPABILITY_TTL_SECONDS + 1
        await adapter._get_card_capability()
        assert len(probes) == 1


class TestGroupMentionMatching:
    """The at-bot rule compares At.target to bot_account_id as an exact string,
    so a space-prefixed mention uid must still resolve to the bot's own id."""

    @pytest.mark.asyncio
    async def test_space_prefixed_mention_resolves_to_bot_account_id(self):
        text = '@Bot 帮我看看'
        payload = {
            'type': 1,
            'content': text,
            # The server names the bot with a space prefix; register()'s
            # robot_id (BOT_UID) has none.
            'mention': {'entities': [{'uid': f's14_{BOT_UID}', 'offset': 0, 'length': 4}]},
        }
        chain = await OctoMessageConverter.target2yiri(_msg(payload), BOT_UID)
        ats = [c for c in chain if isinstance(c, platform_message.At)]
        assert len(ats) == 1
        assert str(ats[0].target) == BOT_UID, 'mention must match bot_account_id exactly'

    @pytest.mark.asyncio
    async def test_uids_fallback_also_normalized(self):
        payload = {'type': 1, 'content': 'hi', 'mention': {'uids': [f's14_{BOT_UID}']}}
        chain = await OctoMessageConverter.target2yiri(_msg(payload), BOT_UID)
        ats = [c for c in chain if isinstance(c, platform_message.At)]
        assert [str(a.target) for a in ats] == [BOT_UID]

    @pytest.mark.asyncio
    async def test_other_users_keep_their_uid(self):
        text = '@张三 你看下'
        payload = {
            'type': 1,
            'content': text,
            'mention': {'entities': [{'uid': 's14_someoneelse', 'offset': 0, 'length': 3}]},
        }
        chain = await OctoMessageConverter.target2yiri(_msg(payload), BOT_UID)
        ats = [c for c in chain if isinstance(c, platform_message.At)]
        assert str(ats[0].target) == 's14_someoneelse'

    @pytest.mark.asyncio
    async def test_ais_flag_mention_matches_bot_account_id(self):
        payload = {'type': 1, 'content': '@所有AI 在吗', 'mention': {'ais': 1}}
        chain = await OctoMessageConverter.target2yiri(_msg(payload), BOT_UID)
        ats = [c for c in chain if isinstance(c, platform_message.At)]
        assert [str(a.target) for a in ats] == [BOT_UID]


class TestTypingIndicatorGating:
    """Typing on a message the pipeline will drop reads as a hung bot."""

    @staticmethod
    def _adapter():
        adapter = TestStreamingCard._adapter()
        adapter.bot_account_id = BOT_UID
        return adapter

    @pytest.mark.asyncio
    async def test_dm_always_indicates(self):
        adapter = self._adapter()
        msg = _msg({'type': 1, 'content': 'hi'}, channel_type=1, from_uid='u1')
        event = await OctoEventConverter.target2yiri(msg, BOT_UID)
        assert adapter._will_likely_reply(event) is True

    @pytest.mark.asyncio
    async def test_group_without_mention_does_not_indicate(self):
        adapter = self._adapter()
        msg = _msg({'type': 1, 'content': '你好'}, channel_type=2)
        event = await OctoEventConverter.target2yiri(msg, BOT_UID)
        assert adapter._will_likely_reply(event) is False

    @pytest.mark.asyncio
    async def test_group_with_bot_mention_indicates(self):
        adapter = self._adapter()
        msg = _msg(
            {'type': 1, 'content': '@Bot 在吗',
             'mention': {'entities': [{'uid': BOT_UID, 'offset': 0, 'length': 4}]}},
            channel_type=2,
        )
        event = await OctoEventConverter.target2yiri(msg, BOT_UID)
        assert adapter._will_likely_reply(event) is True

    @pytest.mark.asyncio
    async def test_group_mention_of_someone_else_does_not_indicate(self):
        adapter = self._adapter()
        msg = _msg(
            {'type': 1, 'content': '@张三 看下',
             'mention': {'entities': [{'uid': 'other-uid', 'offset': 0, 'length': 3}]}},
            channel_type=2,
        )
        event = await OctoEventConverter.target2yiri(msg, BOT_UID)
        assert adapter._will_likely_reply(event) is False
