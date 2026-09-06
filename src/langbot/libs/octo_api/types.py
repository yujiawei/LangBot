"""Type definitions for the Octo IM bot protocol.

Wire shapes are derived from the reference implementation in
https://github.com/Mininglamp-OSS/openclaw-channel-octo (src/types.ts).
"""

from __future__ import annotations

import typing

import pydantic


class ChannelType:
    """Octo channel types."""

    DM = 1
    GROUP = 2
    COMMUNITY_TOPIC = 5  # sub-group / thread; channel_id = "<group_no>____<short_id>"


THREAD_SEPARATOR = '____'


class PayloadType:
    """Octo message payload types."""

    TEXT = 1
    IMAGE = 2
    GIF = 3
    VOICE = 4
    VIDEO = 5
    LOCATION = 6
    CONTACT_CARD = 7
    FILE = 8
    MULTIPLE_FORWARD = 11
    RICH_TEXT = 14
    INTERACTIVE_CARD = 17


class MentionEntity(pydantic.BaseModel):
    """One @mention span inside a text payload.

    offset/length are UTF-16 code units and length includes the '@' sign.
    """

    uid: str = ''
    offset: int = 0
    length: int = 0


class MentionInfo(pydantic.BaseModel):
    """The mention block of a text payload.

    The broadcast flags are serialized as either boolean true or the number 1
    depending on server version, so they are normalized through validators.
    """

    uids: list[str] = pydantic.Field(default_factory=list)
    entities: list[MentionEntity] = pydantic.Field(default_factory=list)
    all: bool = False
    humans: bool = False
    ais: bool = False

    @pydantic.field_validator('all', 'humans', 'ais', mode='before')
    @classmethod
    def _coerce_flag(cls, v: typing.Any) -> bool:
        return v is True or v == 1


class ReplyInfo(pydantic.BaseModel):
    """Inbound quote block: carries the full nested payload of the quoted message."""

    payload: typing.Optional[dict] = None
    from_uid: str = ''
    from_name: str = ''


class MessagePayload(pydantic.BaseModel):
    """Decrypted message payload."""

    model_config = pydantic.ConfigDict(extra='allow')

    type: int = 0
    content: typing.Any = None
    mention: typing.Optional[MentionInfo] = None
    reply: typing.Optional[ReplyInfo] = None
    # System events (group_md_updated etc.) ride on ordinary messages and
    # must not be forwarded into the pipeline.
    event: typing.Optional[dict] = None
    # Media fields (used from P2 on)
    url: typing.Optional[str] = None
    name: typing.Optional[str] = None
    size: typing.Optional[int] = None


class OctoMessage(pydantic.BaseModel):
    """One inbound message decoded from a WS RECV frame.

    message_id is an int64 snowflake kept as a string to avoid precision loss.
    timestamp is in SECONDS.
    """

    message_id: str
    message_seq: int
    from_uid: str
    channel_id: str
    channel_type: int
    timestamp: int
    payload: MessagePayload


class RegisterResult(pydantic.BaseModel):
    """Response of POST /v1/bot/register."""

    robot_id: str
    im_token: str
    ws_url: str = ''
    api_url: str = ''
    owner_uid: str = ''
    owner_channel_id: str = ''


class SendMessageResult(pydantic.BaseModel):
    """Response of POST /v1/bot/sendMessage.

    message_seq is always 0 from the REST API and must not be relied on.
    """

    model_config = pydantic.ConfigDict(extra='allow')

    message_id: typing.Optional[str] = None
    client_msg_no: typing.Optional[str] = None
    message_seq: int = 0

    @pydantic.field_validator('message_id', mode='before')
    @classmethod
    def _stringify_id(cls, v: typing.Any) -> typing.Optional[str]:
        return None if v is None else str(v)


def strip_space_prefix(uid: str) -> tuple[typing.Optional[str], str]:
    """Split a possibly space-prefixed uid ("s14_abc...") into (space_id, base_uid)."""
    if uid.startswith('s') and '_' in uid:
        head, _, rest = uid.partition('_')
        if head[1:].isdigit() and rest:
            return head[1:], rest
    return None, uid


def is_thread_channel(channel_id: str) -> bool:
    """True when channel_id addresses a sub-group (thread)."""
    return THREAD_SEPARATOR in channel_id


def parent_group_no(channel_id: str) -> str:
    """Parent group_no of a (possibly composite) channel id."""
    return channel_id.split(THREAD_SEPARATOR, 1)[0]
