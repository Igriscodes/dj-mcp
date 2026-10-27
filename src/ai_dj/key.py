from typing import Any, Dict, List, Optional, Tuple

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Mapping standard musical key to Camelot notation
KEY_TO_CAMELOT: Dict[str, str] = {
    # Minor keys (A)
    "Ab minor": "1A", "G# minor": "1A",
    "Eb minor": "2A", "D# minor": "2A",
    "Bb minor": "3A", "A# minor": "3A",
    "F minor": "4A",
    "C minor": "5A",
    "G minor": "6A",
    "D minor": "7A",
    "A minor": "8A",
    "E minor": "9A",
    "B minor": "10A",
    "F# minor": "11A", "Gb minor": "11A",
    "C# minor": "12A", "Db minor": "12A",
    # Major keys (B)
    "B major": "1B", "Cb major": "1B",
    "F# major": "2B", "Gb major": "2B",
    "Db major": "3B", "C# major": "3B",
    "Ab major": "4B", "G# major": "4B",
    "Eb major": "5B", "D# major": "5B",
    "Bb major": "6B", "A# major": "6B",
    "F major": "7B",
    "C major": "8B",
    "G major": "9B",
    "D major": "10B",
    "A major": "11B",
    "E major": "12B",
}

CAMELOT_TO_KEY: Dict[str, str] = {
    "1A": "Ab minor", "1B": "B major",
    "2A": "Eb minor", "2B": "F# major",
    "3A": "Bb minor", "3B": "Db major",
    "4A": "F minor", "4B": "Ab major",
    "5A": "C minor", "5B": "Eb major",
    "6A": "G minor", "6B": "Bb major",
    "7A": "D minor", "7B": "F major",
    "8A": "A minor", "8B": "C major",
    "9A": "E minor", "9B": "G major",
    "10A": "B minor", "10B": "D major",
    "11A": "F# minor", "11B": "A major",
    "12A": "C# minor", "12B": "E major",
}


def normalize_key_name(key_str: str) -> str:
    cleaned = key_str.strip().capitalize()
    # Normalize aliases
    alias_map = {
        "A min": "A minor", "Amin": "A minor", "Am": "A minor",
        "C maj": "C major", "Cmaj": "C major", "C": "C major",
    }
    return alias_map.get(cleaned, key_str)


def key_to_camelot(key_name: str) -> Optional[str]:
    norm = normalize_key_name(key_name)
    if norm in KEY_TO_CAMELOT:
        return KEY_TO_CAMELOT[norm]
    upper = key_name.upper().strip()
    if upper in CAMELOT_TO_KEY:
        return upper
    return None


def get_compatible_camelot_keys(camelot_code: str) -> List[str]:
    code = camelot_code.upper().strip()
    if code not in CAMELOT_TO_KEY:
        return []

    num = int(code[:-1])
    letter = code[-1]
    other_letter = "B" if letter == "A" else "A"

    # Camelot wheel wraps around 1 to 12
    prev_num = 12 if num == 1 else num - 1
    next_num = 1 if num == 12 else num + 1
    boost_num = ((num + 1) % 12) + 1  # +2 energy boost

    compatible = [
        f"{num}{letter}",          # Exact same key
        f"{num}{other_letter}",    # Relative major / minor
        f"{prev_num}{letter}",     # 1 step down harmonic
        f"{next_num}{letter}",     # 1 step up harmonic
        f"{boost_num}{letter}",    # Energy boost (+2 semitone equivalent)
    ]
    return list(dict.fromkeys(compatible))


def check_key_compatibility(key_a: str, key_b: str) -> Dict[str, Any]:
    c_a = key_to_camelot(key_a)
    c_b = key_to_camelot(key_b)
    if not c_a or not c_b:
        return {
            "compatible": False,
            "reason": f"Unknown key format: '{key_a}' or '{key_b}'",
            "relationship": "unknown",
        }

    compatibles = get_compatible_camelot_keys(c_a)
    is_compat = c_b in compatibles

    relationship = "clash"
    if c_a == c_b:
        relationship = "exact_match"
    elif c_a[:-1] == c_b[:-1]:
        relationship = "relative_major_minor"
    elif abs(int(c_a[:-1]) - int(c_b[:-1])) in (1, 11) and c_a[-1] == c_b[-1]:
        relationship = "adjacent_harmonic"
    elif is_compat:
        relationship = "energy_boost"

    return {
        "compatible": is_compat,
        "key_a": key_a,
        "camelot_a": c_a,
        "key_b": key_b,
        "camelot_b": c_b,
        "relationship": relationship,
        "suggested_keys": compatibles,
    }
