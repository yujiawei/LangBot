"""Media helpers for the Octo protocol: URL building, image dimension
sniffing and content-type inference.

Upload is three-phase (presign -> PUT -> sendMessage) and lives in
OctoRestClient; these are the pure helpers around it.
"""

from __future__ import annotations

import mimetypes
import struct
import typing

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # aligned with server file.MaxFileSize
MAX_INBOUND_INLINE_BYTES = 10 * 1024 * 1024  # cap for base64-inlining into the pipeline


def build_media_url(
    rel_url: typing.Optional[str],
    api_url: str,
    cdn_url: str = '',
) -> typing.Optional[str]:
    """Build a downloadable URL from a payload's (possibly relative) url field."""
    if not rel_url:
        return None
    if rel_url.startswith('http'):
        return rel_url
    storage_path = rel_url
    if storage_path.startswith('file/preview/'):
        storage_path = storage_path[len('file/preview/') :]
    elif storage_path.startswith('file/'):
        storage_path = storage_path[len('file/') :]
    if cdn_url:
        return f'{cdn_url.rstrip("/")}/{storage_path}'
    return f'{api_url.rstrip("/")}/file/{storage_path}'


def infer_content_type(filename: str) -> str:
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or 'application/octet-stream'


def extension_for_mime(mime: str) -> str:
    return {
        'image/jpeg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/webp': '.webp',
        'audio/mpeg': '.mp3',
        'audio/amr': '.amr',
        'audio/ogg': '.ogg',
        'audio/wav': '.wav',
    }.get(mime.split(';')[0].strip(), mimetypes.guess_extension(mime.split(';')[0].strip()) or '.bin')


def sniff_image_dimensions(data: bytes) -> typing.Optional[tuple[int, int]]:
    """Parse (width, height) from PNG/JPEG/GIF/WebP header bytes."""
    if len(data) < 26:
        return None
    # PNG: 8-byte signature, IHDR at offset 16
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        w, h = struct.unpack('>II', data[16:24])
        return (w, h)
    # GIF87a / GIF89a
    if data[:6] in (b'GIF87a', b'GIF89a'):
        w, h = struct.unpack('<HH', data[6:10])
        return (w, h)
    # WebP: RIFF....WEBP
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        fmt = data[12:16]
        if fmt == b'VP8X' and len(data) >= 30:
            w = int.from_bytes(data[24:27], 'little') + 1
            h = int.from_bytes(data[27:30], 'little') + 1
            return (w, h)
        if fmt == b'VP8 ' and len(data) >= 30:
            w = int.from_bytes(data[26:28], 'little') & 0x3FFF
            h = int.from_bytes(data[28:30], 'little') & 0x3FFF
            return (w, h)
        if fmt == b'VP8L' and len(data) >= 25:
            bits = int.from_bytes(data[21:25], 'little')
            w = (bits & 0x3FFF) + 1
            h = ((bits >> 14) & 0x3FFF) + 1
            return (w, h)
        return None
    # JPEG: scan segments for SOFn
    if data[:2] == b'\xff\xd8':
        pos = 2
        while pos + 9 < len(data):
            if data[pos] != 0xFF:
                pos += 1
                continue
            marker = data[pos + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                pos += 2
                continue
            if pos + 4 > len(data):
                return None
            seg_len = struct.unpack('>H', data[pos + 2 : pos + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                if pos + 9 <= len(data):
                    h, w = struct.unpack('>HH', data[pos + 5 : pos + 9])
                    return (w, h)
                return None
            pos += 2 + seg_len
    return None


def is_complete_image(data: bytes, mime: str) -> bool:
    """Check that an image's terminating marker is present.

    A truncated image still carries a parseable header, so it passes mime and
    dimension sniffing and is only rejected later by the vision model as an
    opaque "image parse error". Catch it at the download boundary instead.
    """
    if mime == 'image/png':
        return data.endswith(b'IEND\xaeB`\x82')
    if mime == 'image/jpeg':
        return data.endswith(b'\xff\xd9')
    if mime == 'image/gif':
        return data.endswith(b'\x3b')
    if mime == 'image/webp':
        # RIFF stores its payload length in bytes 4..8, excluding the 8-byte header.
        if len(data) < 12:
            return False
        return len(data) >= int.from_bytes(data[4:8], 'little') + 8
    return True


def sniff_image_mime(data: bytes) -> str:
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return 'image/gif'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    if data[:2] == b'\xff\xd8':
        return 'image/jpeg'
    return 'application/octet-stream'
