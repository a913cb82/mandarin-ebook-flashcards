import re
from re import Match

PINYIN_TONE_MARKS = {
    "a": "āáǎà",
    "e": "ēéěè",
    "i": "īíǐì",
    "o": "ōóǒò",
    "u": "ūúǔù",
    "ü": "ǖǘǚǜ",
    "A": "ĀÁǍÀ",
    "E": "ĒÉĚÈ",
    "I": "ĪÍǏÌ",
    "O": "ŌÓǑÒ",
    "U": "ŪÚǓÙ",
    "Ü": "ǕǗǙǛ",
}


def _convert_pinyin_callback(m: Match[str]) -> str:
    tone = int(m.group(3)) % 5
    r = m.group(1).replace("v", "ü").replace("V", "Ü")
    pos = 1 if len(r) > 1 and r[0] not in "aeoAEO" else 0
    if tone != 0:
        r = r[0:pos] + PINYIN_TONE_MARKS[r[pos]][tone - 1] + r[pos + 1 :]
    return r + m.group(2)


def convert_pinyin(s: str) -> str:
    """Converts numbered pinyin (e.g., ni2 hao3) to tone-marked pinyin (e.g., ní hǎo)."""
    return re.sub(
        r"([aeiouüvÜ]{1,3})(n?g?r?)([012345])", _convert_pinyin_callback, s, flags=re.IGNORECASE
    )
