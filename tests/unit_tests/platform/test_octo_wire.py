"""Unit tests for the Octo WuKongIM wire protocol codec."""

import base64
import hashlib

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from langbot.libs.octo_api import wire


def _build_recv_frame(
    *,
    setting: int = 0,
    from_uid: str = 'u1',
    channel_id: str = 'grp_1',
    channel_type: int = 2,
    message_id: int = 7385920184320000001,
    message_seq: int = 412,
    timestamp: int = 1753939200,
    payload: bytes = b'PAYLOAD',
    server_version: int = 4,
    topic: str = '',
) -> bytes:
    w = wire._Writer()
    w.u8(setting)
    w.string('msgkey')
    w.string(from_uid)
    w.string(channel_id)
    w.u8(channel_type)
    if server_version >= 3:
        w.u32(0)  # expire
    w.string('clientmsgno')
    w.u64(message_id)
    w.u32(message_seq)
    w.u32(timestamp)
    if (setting >> 3) & 0x01:
        w.string(topic)
    body = w.bytes() + payload
    return bytes([wire.PacketType.RECV << 4]) + wire.encode_varlen(len(body)) + body


class TestVarlen:
    def test_roundtrip_boundaries(self):
        for n in (0, 1, 127, 128, 16383, 16384, 2097151, 2097152):
            encoded = wire.encode_varlen(n)
            buf = bytearray(bytes([wire.PacketType.RECVACK << 4]) + encoded + b'\x00' * n)
            frames = wire.unpack_frames(buf)
            assert len(frames) == 1
            assert len(frames[0].body) == n
            assert not buf


class TestFraming:
    def test_single_byte_ping_pong(self):
        buf = bytearray(wire.encode_ping() + bytes([wire.PacketType.PONG << 4]))
        frames = wire.unpack_frames(buf)
        assert [f.packet_type for f in frames] == [wire.PacketType.PING, wire.PacketType.PONG]
        assert not buf

    def test_sticky_packets(self):
        frame = _build_recv_frame()
        buf = bytearray(frame + frame + bytes([wire.PacketType.PONG << 4]))
        frames = wire.unpack_frames(buf)
        assert len(frames) == 3
        assert not buf

    def test_fragmented_packet(self):
        frame = _build_recv_frame()
        buf = bytearray(frame[:10])
        assert wire.unpack_frames(buf) == []
        assert len(buf) == 10  # partial data retained
        buf += frame[10:]
        frames = wire.unpack_frames(buf)
        assert len(frames) == 1
        assert not buf


class TestConnect:
    def test_connect_packet_shape(self):
        packet = wire.encode_connect('uid1', 'tok1', 'device1', 'ckey==')
        assert packet[0] >> 4 == wire.PacketType.CONNECT
        buf = bytearray(packet)
        frames = wire.unpack_frames(buf)
        r = wire.Reader(frames[0].body)
        assert r.u8() == wire.PROTO_VERSION
        assert r.u8() == 0  # deviceFlag
        assert r.string() == 'device1'
        assert r.string() == 'uid1'
        assert r.string() == 'tok1'
        assert r.u64() > 0  # timestamp ms
        assert r.string() == 'ckey=='

    def test_connack_parse_with_server_version(self):
        w = wire._Writer()
        w.u8(4)  # serverVersion (flag bit0 set)
        w.u64(5)  # timeDiff
        w.u8(1)  # reasonCode success
        w.string('c2VydmVya2V5')
        w.string('salt-16-bytes-xx-extra')
        w.u64(99)  # nodeId (serverVersion >= 4)
        frame = wire.Frame(wire.PacketType.CONNACK, 0x01, w.bytes())
        connack = wire.parse_connack(frame)
        assert connack.server_version == 4
        assert connack.reason_code == 1
        assert connack.server_key == 'c2VydmVya2V5'
        assert connack.salt == 'salt-16-bytes-xx-extra'


class TestRecv:
    def test_recv_parse(self):
        frame_bytes = _build_recv_frame(setting=0b10001010, topic='t1')
        buf = bytearray(frame_bytes)
        frames = wire.unpack_frames(buf)
        recv = wire.parse_recv(frames[0], server_version=4)
        assert recv.setting_receipt is True
        assert recv.setting_topic is True
        assert recv.setting_stream is True
        assert recv.from_uid == 'u1'
        assert recv.channel_id == 'grp_1'
        assert recv.channel_type == 2
        assert recv.message_id == '7385920184320000001'
        assert recv.message_seq == 412
        assert recv.timestamp == 1753939200
        assert recv.topic == 't1'
        assert recv.encrypted_payload == b'PAYLOAD'

    def test_recvack_roundtrip(self):
        packet = wire.encode_recvack('7385920184320000001', 412)
        buf = bytearray(packet)
        frames = wire.unpack_frames(buf)
        r = wire.Reader(frames[0].body)
        assert r.u64() == 7385920184320000001
        assert r.u32() == 412


class TestCrypto:
    def test_dh_and_aes_roundtrip(self):
        client = wire.generate_keypair()
        server = wire.generate_keypair()
        salt = 'This-Is-A-Long-Salt-String'

        server_pub_b64 = server.public_b64
        key, iv = wire.derive_aes(client.private, server_pub_b64, salt)
        assert len(key) == 16 and len(iv) == 16
        assert iv == salt[:16].encode()

        # The server derives the same key from the client's public key.
        client_pub_b64 = client.public_b64
        server_key, server_iv = wire.derive_aes(server.private, client_pub_b64, salt)
        assert key == server_key and iv == server_iv

        # Simulate the server encrypting a payload (AES-128-CBC/PKCS7, base64-wrapped).
        plaintext = '{"type":1,"content":"你好 Octo"}'.encode('utf-8')
        pad = 16 - len(plaintext) % 16
        padded = plaintext + bytes([pad] * pad)
        encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
        ciphertext = base64.b64encode(encryptor.update(padded) + encryptor.finalize())

        assert wire.aes_decrypt_payload(ciphertext, key, iv) == plaintext

    def test_key_derivation_matches_reference_formula(self):
        # aesKey must be the first 16 hex chars of MD5(base64(secret)) as ASCII.
        client = wire.generate_keypair()
        server = wire.generate_keypair()
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey

        server_pub = X25519PublicKey.from_public_bytes(base64.b64decode(server.public_b64))
        secret = client.private.exchange(server_pub)
        expected = hashlib.md5(base64.b64encode(secret)).hexdigest()[:16].encode('ascii')
        key, _ = wire.derive_aes(client.private, server.public_b64, 'saltsaltsaltsalt')
        assert key == expected

    def test_short_salt_zero_padded(self):
        client = wire.generate_keypair()
        server = wire.generate_keypair()
        _, iv = wire.derive_aes(client.private, server.public_b64, 'short')
        assert iv == b'short' + b'\x00' * 11

    def test_device_id_shape(self):
        device_id = wire.generate_device_id()
        assert len(device_id) == 33
        assert device_id.endswith('W')
        assert device_id[12] == '4'
