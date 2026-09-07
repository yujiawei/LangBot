"""Adaptive Card construction for Octo streaming replies.

The server (pkg/cardmsg) is the schema authority; this module only assembles
the minimal Adaptive Card 1.5 documents the adapter needs and enforces the
limits advertised by GET /v1/bot/card/profile.
"""

from __future__ import annotations

import json
import typing

CARD_VERSION = '1.5'
PROFILE_DISPLAY = 'octo/v1'
PROFILE_INTERACTIVE = 'octo/v2'

# Conservative defaults, overridden by the server's advertised limits.
DEFAULT_MAX_PAYLOAD_BYTES = 512 * 1024
PLAIN_PREVIEW_CHARS = 200


class CardCapability(typing.NamedTuple):
    """What the server allows this bot to do with cards."""

    available: bool
    enabled: bool
    profiles: frozenset[str]
    max_payload_bytes: int

    @property
    def can_send_display_card(self) -> bool:
        return self.available and self.enabled and PROFILE_DISPLAY in self.profiles


def parse_capability(profile: dict) -> CardCapability:
    """Read a card/profile response, failing closed on anything unexpected."""
    available = bool(profile.get('available'))
    enabled = bool(profile.get('enabled'))
    config = profile.get('config')
    if isinstance(config, dict):
        # Per-bot server policy; card_enabled gates everything below it.
        card_enabled = config.get('card_enabled')
        if card_enabled is not None:
            enabled = enabled and (card_enabled is True or card_enabled == 1)
    raw_profiles = profile.get('profiles')
    profiles = frozenset(p for p in raw_profiles if isinstance(p, str)) if isinstance(raw_profiles, list) else frozenset()
    limits = profile.get('limits')
    max_payload = DEFAULT_MAX_PAYLOAD_BYTES
    if isinstance(limits, dict):
        candidate = limits.get('max_payload_bytes')
        if isinstance(candidate, int) and candidate > 0:
            max_payload = candidate
    return CardCapability(
        available=available,
        enabled=enabled,
        profiles=profiles,
        max_payload_bytes=max_payload,
    )


def build_text_card(text: str) -> dict:
    """A single wrapping TextBlock - the streaming answer surface."""
    return {
        'type': 'AdaptiveCard',
        '$schema': 'http://adaptivecards.io/schemas/adaptive-card.json',
        'version': CARD_VERSION,
        'body': [
            {
                'type': 'TextBlock',
                'text': text,
                'wrap': True,
            }
        ],
    }


def plain_preview(text: str) -> str:
    """Fallback text for clients that cannot render the card.

    The server recomputes ``plain`` authoritatively, but it must never be
    empty on the wire.
    """
    collapsed = ' '.join(text.split())
    if not collapsed:
        return '[卡片]'
    if len(collapsed) <= PLAIN_PREVIEW_CHARS:
        return collapsed
    return collapsed[: PLAIN_PREVIEW_CHARS - 1] + '…'


def fit_text_to_payload(text: str, max_payload_bytes: int) -> str:
    """Trim text so the serialized type-17 envelope stays under the cap.

    Measured against the whole envelope, not just the card, because that is
    what the server limits.
    """

    def envelope_bytes(candidate: str) -> int:
        envelope = {
            'type': 17,
            'card': build_text_card(candidate),
            'plain': plain_preview(candidate),
            'profile': PROFILE_DISPLAY,
            'card_version': CARD_VERSION,
            # Room for the optional card_seq/transient keys.
            'card_seq': 2**31,
            'transient': True,
        }
        return len(json.dumps(envelope, ensure_ascii=False).encode('utf-8'))

    if envelope_bytes(text) <= max_payload_bytes:
        return text

    ellipsis = '\n…（内容过长，已截断）'
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if envelope_bytes(text[:mid] + ellipsis) <= max_payload_bytes:
            low = mid
        else:
            high = mid - 1
    return text[:low] + ellipsis
