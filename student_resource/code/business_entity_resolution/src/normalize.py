"""
normalize.py — High-performance text normalization for business entity resolution.

Features:
- Lowercasing
- Unicode NFKD + combining-mark strip for Latin-script text (café == cafe)
- Devanagari left untouched (no cross-script transliteration)
- Legal suffix expansion/contraction with single-pass regex
- Address abbreviation standardization with single-pass regex
- Punctuation normalization
- NaN / None → empty string safe handling
"""

import re
import unicodedata
from typing import Dict

# ---------------------------------------------------------------------------
# Legal suffix normalization map
# ---------------------------------------------------------------------------
LEGAL_SUFFIX_MAP: Dict[str, str] = {
    # English (US / UK)
    r'incorporated': 'inc',
    r'corporation': 'corp',
    r'corporations': 'corp',
    r'limited\s+liability\s+company': 'llc',
    r'limited\s+liability\s+partnership': 'llp',
    r'limited\s+partnership': 'lp',
    r'limited': 'ltd',
    r'ltd': 'ltd',
    r'llc': 'llc',
    r'llp': 'llp',
    r'corp': 'corp',
    r'inc': 'inc',
    r'pllc': 'pllc',
    r'pc': 'pc',
    r'pa': 'pa',
    r'dba': '',
    r't/a': '',
    r'and': '&',
    # India
    r'private\s+limited': 'pvt ltd',
    r'pvt\.\s*ltd': 'pvt ltd',
    r'pvt\s+ltd': 'pvt ltd',
    r'private': 'pvt',
    r'proprietary': 'prop',
    r'enterprises': 'ent',
    r'enterprise': 'ent',
    r'traders': 'traders',
    r'trader': 'trader',
    r'industries': 'ind',
    r'industry': 'ind',
    # France
    r'soci[eé]t[eé]\s+(?:anonyme|par\s+actions\s+simplifi[eé]e)': 'sa',
    r'soci[eé]t[eé]\s+[àa]\s+responsabilit[eé]\s+limit[eé]e': 'sarl',
    r'soci[eé]t[eé]\s+en\s+nom\s+collectif': 'snc',
    r'sarl': 'sarl',
    r'sas': 'sas',
    r'sa': 'sa',
    r'snc': 'snc',
    r'ei?url': 'eurl',
    r'sci': 'sci',
}

# ---------------------------------------------------------------------------
# Address abbreviation map
# ---------------------------------------------------------------------------
ADDR_ABBREV_MAP: Dict[str, str] = {
    r'street': 'st',
    r'avenue': 'ave',
    r'boulevard': 'blvd',
    r'drive': 'dr',
    r'road': 'rd',
    r'lane': 'ln',
    r'court': 'ct',
    r'place': 'pl',
    r'terrace': 'ter',
    r'circle': 'cir',
    r'way': 'wy',
    r'highway': 'hwy',
    r'parkway': 'pkwy',
    r'expressway': 'expy',
    r'freeway': 'fwy',
    r'apartment': 'apt',
    r'suite': 'ste',
    r'building': 'bldg',
    r'floor': 'fl',
    r'unit': 'unit',
    r'north': 'n',
    r'south': 's',
    r'east': 'e',
    r'west': 'w',
    r'northeast': 'ne',
    r'northwest': 'nw',
    r'southeast': 'se',
    r'southwest': 'sw',
}


def _build_combined_regex(mapping: Dict[str, str]):
    sorted_pats = sorted(mapping.keys(), key=lambda p: len(p), reverse=True)
    pattern_str = r'\b(' + '|'.join(sorted_pats) + r')\b'
    compiled_re = re.compile(pattern_str, re.IGNORECASE | re.UNICODE)

    def replacer(match: re.Match) -> str:
        matched_text = match.group(0).lower()
        clean = re.sub(r'\s+', ' ', matched_text)
        res = mapping.get(clean) or mapping.get(matched_text)
        if res is not None:
            return ' ' + res + ' ' if res else ' '
        for pat, val in mapping.items():
            if re.fullmatch(pat, matched_text, re.IGNORECASE):
                return ' ' + val + ' ' if val else ' '
        return ' '

    return compiled_re, replacer


_LEGAL_RE, _LEGAL_REPL = _build_combined_regex(LEGAL_SUFFIX_MAP)
_ADDR_RE, _ADDR_REPL = _build_combined_regex(ADDR_ABBREV_MAP)

_PUNCT_RE = re.compile(r'[^\w\s&@#/\-]', re.UNICODE)
_SPACE_RE = re.compile(r'\s+')

_DEVANAGARI_START = '\u0900'
_DEVANAGARI_END = '\u097F'


def fold_latin_accents(text: str) -> str:
    """
    After lowercasing: NFKD-normalize Latin text and strip combining marks
    so 'café' and 'cafe' match. Devanagari codepoints are copied unchanged;
    no transliteration is attempted.
    """
    out = []
    for ch in text:
        if _DEVANAGARI_START <= ch <= _DEVANAGARI_END:
            out.append(ch)
            continue
        for c in unicodedata.normalize('NFKD', ch):
            if unicodedata.category(c) != 'Mn':
                out.append(c)
    return ''.join(out)


def normalize_name(text: str) -> str:
    """Normalize business name for matching."""
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    text = fold_latin_accents(text)
    text = _LEGAL_RE.sub(_LEGAL_REPL, text)
    text = _PUNCT_RE.sub(' ', text)
    return _SPACE_RE.sub(' ', text).strip()


def normalize_address(text: str) -> str:
    """Normalize business address for matching."""
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    text = fold_latin_accents(text)
    text = _ADDR_RE.sub(_ADDR_REPL, text)
    text = _PUNCT_RE.sub(' ', text)
    return _SPACE_RE.sub(' ', text).strip()


def safe_str(value) -> str:
    """Return string or empty string for None/NaN."""
    if value is None:
        return ''
    try:
        import math
        if math.isnan(float(value)):
            return ''
    except (TypeError, ValueError):
        pass
    return str(value).strip()
