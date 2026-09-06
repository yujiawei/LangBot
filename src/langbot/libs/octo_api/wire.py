"""WuKongIM binary wire protocol (v4) used by the Octo WebSocket endpoint.

Pure encode/decode functions plus the per-connection crypto derivation, ported
byte-for-byte from openclaw-channel-octo/src/socket.ts. Framing is MQTT-style:
header byte ``type << 4 | flags``, base-128 variable remaining length, body.
Strings are big-endian uint16 length + UTF-8 bytes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import secrets
import time
import typing

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

import base64


PROTO_VERSION = 4


class PacketType:
    CONNECT = 1
    CONNACK = 2
    SEND = 3
    SENDACK = 4
    RECV = 5
    RECVACK = 6
    PING = 7
    PONG = 8
    DISCONNECT = 9


class _Writer:
    def __init__(self) -> None:
        self._buf = bytearray()

    def u8(self, v: int) -> None:
        self._buf.append(v & 0xFF)

    def u32(self, v: int) -> None:
        self._buf += (v & 0xFFFFFFFF).to_bytes(4, 'big')

    def u64(self, v: int) -> None:
        self._buf += (v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, 'big')

    def string(self, s: str) -> None:
        raw = s.encode('utf-8')
        self._buf += len(raw).to_bytes(2, 'big')
        self._buf += raw

    def bytes(self) -> bytes:
        return bytes(self._buf)


class Reader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def u8(self) -> int:
        v = self._data[self._pos]
        self._pos += 1
        return v

    def u16(self) -> int:
        v = int.from_bytes(self._data[self._pos : self._pos + 2], 'big')
        self._pos += 2
        return v

    def u32(self) -> int:
        v = int.from_bytes(self._data[self._pos : self._pos + 4], 'big')
        self._pos += 4
        return v

    def u64(self) -> int:
        v = int.from_bytes(self._data[self._pos : self._pos + 8], 'big')
        self._pos += 8
        return v

    def string(self) -> str:
        length = self.u16()
        if length <= 0:
            return ''
        raw = self._data[self._pos : self._pos + length]
        self._pos += length
        return raw.decode('utf-8')

    def remaining(self) -> bytes:
        d = self._data[self._pos :]
        self._pos = len(self._data)
        return d


def encode_varlen(length: int) -> bytes:
    out = bytearray()
    while True:
        digit = length % 0x80
        length //= 0x80
        if length > 0:
            digit |= 0x80
        out.append(digit)
        if length <= 0:
            break
    return bytes(out)


def _frame(packet_type: int, flags: int, body: bytes) -> bytes:
    return bytes([(packet_type << 4) | flags]) + encode_varlen(len(body)) + body


def encode_connect(uid: str, token: str, device_id: str, client_key_b64: str) -> bytes:
    w = _Writer()
    w.u8(PROTO_VERSION)
    w.u8(0)  # deviceFlag 0 = app/bot
    w.string(device_id)
    w.string(uid)
    w.string(token)
    w.u64(int(time.time() * 1000))
    w.string(client_key_b64)
    return _frame(PacketType.CONNECT, 0, w.bytes())


def encode_ping() -> bytes:
    return bytes([PacketType.PING << 4])


def encode_recvack(message_id: str, message_seq: int) -> bytes:
    w = _Writer()
    w.u64(int(message_id))
    w.u32(message_seq)
    return _frame(PacketType.RECVACK, 0, w.bytes())


@dataclasses.dataclass
class Frame:
    packet_type: int
    flags: int
    body: bytes


def unpack_frames(buffer: bytearray) -> list[Frame]:
    """Extract complete frames from ``buffer`` in place, leaving partial data.

    PING/PONG are single-byte packets with no remaining-length field.
    Raises ValueError on a malformed variable length so the caller can reset
    the connection.
    """
    frames: list[Frame] = []
    while buffer:
        header = buffer[0]
        packet_type = header >> 4
        if packet_type in (PacketType.PING, PacketType.PONG):
            frames.append(Frame(packet_type, header & 0x0F, b''))
            del buffer[:1]
            continue

        # Decode base-128 variable remaining length.
        rem_length = 0
        multiplier = 1
        pos = 1
        complete = False
        while True:
            if pos > 5:
                raise ValueError('malformed variable length')
            if pos >= len(buffer):
                break
            digit = buffer[pos]
            pos += 1
            rem_length += (digit & 0x7F) * multiplier
            multiplier *= 128
            if (digit & 0x80) == 0:
                complete = True
                break
        if not complete:
            break

        total = pos + rem_length
        if total > len(buffer):
            break
        frames.append(Frame(packet_type, header & 0x0F, bytes(buffer[pos:total])))
        del buffer[:total]
    return frames


@dataclasses.dataclass
class ConnackPacket:
    server_version: int
    reason_code: int  # 1 = success, 0 = kicked, other = connect failed
    server_key: str
    salt: str


def parse_connack(frame: Frame) -> ConnackPacket:
    r = Reader(frame.body)
    server_version = 0
    if frame.flags & 0x01:
        server_version = r.u8()
    r.u64()  # timeDiff
    reason_code = r.u8()
    server_key = r.string()
    salt = r.string()
    return ConnackPacket(server_version, reason_code, server_key, salt)


@dataclasses.dataclass
class RecvPacket:
    setting_receipt: bool
    setting_topic: bool
    setting_stream: bool
    from_uid: str
    channel_id: str
    channel_type: int
    message_id: str  # int64 as decimal string
    message_seq: int
    timestamp: int  # seconds
    topic: str
    encrypted_payload: bytes


def parse_recv(frame: Frame, server_version: int) -> RecvPacket:
    r = Reader(frame.body)
    setting = r.u8()
    r.string()  # msgKey
    from_uid = r.string()
    channel_id = r.string()
    channel_type = r.u8()
    if server_version >= 3:
        r.u32()  # expire
    r.string()  # clientMsgNo
    message_id = str(r.u64())
    message_seq = r.u32()
    timestamp = r.u32()
    topic = ''
    if (setting >> 3) & 0x01:
        topic = r.string()
    return RecvPacket(
        setting_receipt=bool((setting >> 7) & 0x01),
        setting_topic=bool((setting >> 3) & 0x01),
        setting_stream=bool((setting >> 1) & 0x01),
        from_uid=from_uid,
        channel_id=channel_id,
        channel_type=channel_type,
        message_id=message_id,
        message_seq=message_seq,
        timestamp=timestamp,
        topic=topic,
        encrypted_payload=r.remaining(),
    )


@dataclasses.dataclass
class DisconnectPacket:
    reason_code: int
    reason: str


def parse_disconnect(frame: Frame) -> DisconnectPacket:
    r = Reader(frame.body)
    return DisconnectPacket(r.u8(), r.string())


# ─── Crypto ──────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class KeyPair:
    private: X25519PrivateKey
    public_b64: str


def generate_keypair() -> KeyPair:
    private = X25519PrivateKey.generate()
    public_raw = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return KeyPair(private=private, public_b64=base64.b64encode(public_raw).decode('ascii'))


def derive_aes(private: X25519PrivateKey, server_key_b64: str, salt: str) -> tuple[bytes, bytes]:
    """Derive the AES-128-CBC key/IV from the DH exchange, mirroring the JS client:
    key = hex(MD5(base64(shared_secret)))[:16] as ASCII; IV = salt[:16] as ASCII.
    """
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey

    server_pub = X25519PublicKey.from_public_bytes(base64.b64decode(server_key_b64))
    secret = private.exchange(server_pub)
    secret_b64 = base64.b64encode(secret).decode('ascii')
    aes_key = hashlib.md5(secret_b64.encode('ascii')).hexdigest()[:16].encode('ascii')
    iv = (salt[:16] if len(salt) > 16 else salt).encode('utf-8')
    # AES-CBC requires exactly 16 IV bytes; a short salt is zero-padded.
    iv = iv[:16].ljust(16, b'\x00')
    return aes_key, iv


def aes_decrypt_payload(encrypted: bytes, key: bytes, iv: bytes) -> bytes:
    """RECV payloads are base64 text wrapping AES-128-CBC/PKCS7 ciphertext."""
    ciphertext = base64.b64decode(encrypted)
    decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    if not padded:
        return b''
    pad = padded[-1]
    if pad < 1 or pad > 16 or pad > len(padded):
        raise ValueError('invalid PKCS7 padding')
    return padded[:-pad]


def generate_device_id() -> str:
    """32-hex UUIDv4-shaped device id with the web-client 'W' suffix."""
    raw = secrets.token_hex(16)
    return raw[:12] + '4' + raw[13:16] + typing.cast(str, '89ab'[secrets.randbelow(4)]) + raw[17:] + 'W'
