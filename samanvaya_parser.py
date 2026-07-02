#!/usr/bin/env python3
"""
samanvaya_parser.py — Finite State Machine for Samanvaya LWG rules.

States
------
FREE        : No active obligation. Most decode steps live here.
MUST_AUX    : Previous word ended in ा or ी → next word must be an auxiliary verb.
MUST_CASE   : Previous word was standalone "के" or "की" → compound postposition must follow.
MUST_CONT   : Mid fixed-phrase (RULE1 or RULE3 non-ke/ki) → next word constrained.

Design notes
------------
- MUST_CONT stores the SET of valid next words (handles "हो" → {गयी, गया}).
- For 3-word RULE1 phrases (e.g. हाल→ही→में), MUST_CONT also stores a
  continuation map so it can chain into the next step after the first word.
- "के" / "की" are always routed to MUST_CASE, not MUST_CONT, because they
  start multiple possible continuations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from wordgrouping_rules import (
    A_ENDING,
    ATTACH_TO_LEFT,
    AUX_AFTER_A,
    AUX_AFTER_EE,
    EE_ENDING,
    RULE1_PHRASES,
    RULE3_MULTIWORDS,
)

# ---------------------------------------------------------------------------
# Derived constants (built once at import time)
# ---------------------------------------------------------------------------

ALL_AUX: frozenset[str] = frozenset(AUX_AFTER_EE | AUX_AFTER_A)

# Words that end in ा/ी but are case markers / postpositions, NOT verb forms.
# Excluding these from the verb-aux trigger prevents false MUST_AUX obligations.
# Example: "की" (genitive, ends in ी), "वाला/वाली" (agentive), "द्वारा" (instrumental)
NON_VERB_AUX_TRIGGERS: frozenset[str] = frozenset(ATTACH_TO_LEFT)

# Suffix patterns (char-level) that identify non-verb words ending in ा/ी.
# Words ending in these suffixes are excluded from MUST_AUX trigger.
#   "ता"  → abstract nouns: स्वतंत्रता, क्षमता, सुंदरता, समता
#   "ना"  → infinitives used as nouns: खाना (food), जाना, सोना, गाना
#   "त्मा" → soul/self suffix: महात्मा, आत्मा, परमात्मा
#   "कारी" → agentive/adjectival suffix: लाभकारी, हानिकारी, नुकसानकारी
#   "दारी" → obligation suffix: जिम्मेदारी, जवाबदेही-दारी, अधिकारी
#   "वारी" → festival/periodic suffix: त्यौहारी, दीपावली-... etc.
#   "शिला" → historical place suffix: तक्षशिला, विक्रमशिला
#   "नदी" → river suffix used in compound proper names like महानदी
_NON_VERB_SUFFIXES: tuple[str, ...] = ("ता", "ना", "त्मा", "कारी", "दारी", "वारी", "शिला")

# Curated set of common Hindi words ending in ा/ी that are NOT verb past participles.
# Includes: nouns, adjectives, pronouns, particles, proper nouns.
_COMMON_NOUN_EXCLUSIONS: frozenset[str] = frozenset({
    # === Particles / function words ===
    "क्या", "या", "जा",
    # === Pronouns / demonstratives ending in ा (possessive masculine) ===
    "अपना", "उनका", "इनका", "इसका", "उसका", "जिसका", "किसका",
    "हमारा", "तुम्हारा", "आपका",
    # === Pronouns / demonstratives ending in ी (possessive feminine) ===
    "अपनी", "उनकी", "इनकी", "इसकी", "उसकी", "जिसकी", "किसकी",
    "हमारी", "तुम्हारी", "आपकी",
    # === Common adjectives ending in ी ===
    "अच्छी", "बुरी", "बड़ी", "छोटी", "लंबी", "नई", "पुरानी",
    "ऐसी", "वैसी", "जैसी", "कैसी", "कितनी",
    "गहरी", "जरूरी", "भारी", "हल्की", "सही", "सच्ची",
    "पूरी", "आधी", "साफ़ी", "असली", "नकली",
    "विदेशी", "भारतीय", "दक्षिणी", "उत्तरी", "पूर्वी", "पश्चिमी",
    "सरकारी", "निजी", "मुफ़्त", "किफायती",
    # === Common adjectives ending in ा ===
    "अच्छा", "बुरा", "बड़ा", "छोटा", "लंबा", "नया", "पुराना",
    "पूरा", "आधा", "असली", "सच्चा", "साफ़", "मोटा", "पतला",
    # === Common nouns ending in ी ===
    "राजधानी", "शताब्दी", "पृथ्वी", "ज़िंदगी", "ज़िन्दगी", "जिंदगी",
    "सेनानी", "पत्नी", "नदी", "बेटी", "नानी", "दादी", "माँ",
    "रोटी", "मिट्टी", "साड़ी", "गाड़ी", "चाँदी", "चांदी", "धरती",
    "रोशनी", "चाँदनी", "चांदनी", "पहाड़ी", "हरियाली",
    "पानी", "नौकरी", "जानकारी", "ज़िम्मेदारी", "जिम्मेदारी",
    "तैयारी", "बीमारी", "लड़की", "खुशी", "आसानी",
    "बिजली", "डिग्री", "कनेक्टिविटी", "प्रौद्योगिकी",
    "देवनागरी", "राजभाषा", "न्यायपालिका", "जनसंख्या",
    "जवाबदेही", "खरीदारी", "यात्री", "विद्यार्थी",
    "बिरयानी", "घी", "दही",
    # === Festivals, culture, religion ===
    "होली", "दीपावली", "बैसाखी", "जयंती", "चतुर्थी",
    "दुर्गा", "कृष्णा",
    # === Common nouns ending in ा ===
    "पिता", "दादा", "नाना", "राजा", "नेता", "सेना", "ऊर्जा",
    "विद्या", "संध्या", "शाला", "माला", "कला", "भाषा", "आशा",
    "आज्ञा", "सभा", "सेवा", "दिशा", "रचना", "योजना", "संरचना",
    "व्यवस्था", "अर्थव्यवस्था", "समीक्षा", "चिंता", "सीमा",
    "कृपा", "प्रतिभा", "शिक्षा", "चिकित्सा", "कक्षा",
    "सुरक्षा", "सुविधा", "मात्रा", "भूमिका", "परंपरा", "वर्षा",
    "मीडिया", "दुनिया", "समस्या", "भूमिका", "जनसंख्या",
    "महामारी", "महिला", "कथा", "अहिंसा",
    # === River and place names ===
    "गंगा", "गोदावरी", "कावेरी", "नर्मदा", "ताप्ती", "महानदी",
    "नालंदा", "लंका",
    # === Languages / scripts ===
    "हिंदी", "उर्दू", "मराठी", "बंगाली", "गुजराती", "पंजाबी",
    # === Proper nouns / names ===
    "गांधी", "गाँधी", "दिल्ली", "महादेवी", "द्विवेदी", "रामधारी",
    "मुंशी", "वर्मा", "सिंधी",
    # === Technology / English loanwords ===
    "जावा", "डेटा", "जीडीपी", "मीडिया", "डोसा", "समोसा",
    # === Miscellaneous common nouns ===
    "शर्करा", "तारा", "अनूठा", "कमी", "दर्जा", "हिस्सा",
    "खतरा", "शाखा", "किला", "पूजा", "जमा", "मखनी",
    "देश-दुनिया", "इंडिया", "तकनीकी",
})

# MUST_CASE triggers — only "के" (11 clear compound postpositions).
# "की" is excluded: its only RULE3 continuation is "ओर" which is rare,
# while possessive "की" (e.g. भारत की राजधानी) is ubiquitous and would
# create massive false positives.
MUST_CASE_TRIGGERS: frozenset[str] = frozenset({"के"})
MUST_CASE_VALID: dict[str, frozenset[str]] = {}
for _p in RULE3_MULTIWORDS:
    if _p[0] in MUST_CASE_TRIGGERS:
        MUST_CASE_VALID.setdefault(_p[0], set()).add(_p[1])
MUST_CASE_VALID = {k: frozenset(v) for k, v in MUST_CASE_VALID.items()}

# RELIABLE phrase starters for MUST_CONT.
# We exclude:
#   - "के"/"की" → handled by MUST_CASE or excluded above
#   - "में", "ने", "ही", "ओर" → too commonly standalone (case markers/particles)
#   - "हो" → "हो गया/गयी" is valid, kept below
# Included: compound verb patterns (रहा/रहे/रही/सकता/…) and clear phrases.
_EXCLUDED_CONT_STARTERS: frozenset[str] = frozenset({"के", "की", "में", "ने", "ही", "ओर"})

_PHRASE_VALID_NEXT: dict[str, set[str]] = {}
_PHRASE_EXTENSION: dict[str, dict[str, str]] = {}  # for 3-word phrases (हाल→ही→में)

for _p in RULE1_PHRASES + RULE3_MULTIWORDS:
    if _p[0] in MUST_CASE_TRIGGERS or _p[0] in _EXCLUDED_CONT_STARTERS:
        continue
    _PHRASE_VALID_NEXT.setdefault(_p[0], set()).add(_p[1])
    if len(_p) >= 3:
        _PHRASE_EXTENSION.setdefault(_p[0], {})[_p[1]] = _p[2]

PHRASE_VALID_NEXT: dict[str, frozenset[str]] = {
    k: frozenset(v) for k, v in _PHRASE_VALID_NEXT.items()
}
PHRASE_EXTENSION: dict[str, dict[str, str]] = _PHRASE_EXTENSION

# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

FREE = "FREE"
MUST_AUX = "MUST_AUX"
MUST_CASE = "MUST_CASE"
MUST_CONT = "MUST_CONT"


class SamanvayaParser:
    """
    Word-level finite state machine implementing Samanvaya LWG rules.

    Usage
    -----
    parser = SamanvayaParser()
    for word in sentence.split():
        state = parser.update(word)
    print(parser.compliance_rate())
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._state: str = FREE

        # MUST_CASE bookkeeping
        self._case_trigger: Optional[str] = None

        # MUST_CONT bookkeeping
        self._cont_valid_next: frozenset[str] = frozenset()
        # maps current-step word → next-next expected word (for 3-word phrases)
        self._cont_extension: dict[str, str] = {}
        # phrase-start that triggered this MUST_CONT (for extension lookup)
        self._cont_start: Optional[str] = None

        # Statistics
        self._obligations: int = 0    # times FSM entered a constrained state
        self._completions: int = 0    # obligations completed correctly
        self._violations: int = 0     # obligations violated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    def is_constrained(self) -> bool:
        """True when the parser has an active obligation."""
        return self._state in (MUST_AUX, MUST_CASE, MUST_CONT)

    def valid_next_words(self) -> frozenset[str]:
        """Return the set of words that satisfy the current obligation."""
        if self._state == MUST_AUX:
            return ALL_AUX
        if self._state == MUST_CASE:
            return MUST_CASE_VALID.get(self._case_trigger, frozenset())
        if self._state == MUST_CONT:
            return self._cont_valid_next
        return frozenset()

    def compliance_rate(self) -> float:
        """Fraction of obligations that were completed correctly."""
        total = self._completions + self._violations
        return self._completions / total if total > 0 else 1.0

    def stats(self) -> dict:
        return {
            "obligations": self._obligations,
            "completions": self._completions,
            "violations": self._violations,
            "compliance_rate": self.compliance_rate(),
        }

    def update(self, word: str) -> str:
        """
        Process one complete Hindi word; return the new FSM state.

        Transitions:
            FREE + ke/ki                 → MUST_CASE
            FREE + phrase_start          → MUST_CONT
            FREE + ends_ा/ी (non-aux)    → MUST_AUX
            FREE + other                 → FREE

            MUST_AUX + aux_word          → FREE (completion)
            MUST_AUX + other             → FREE (violation), re-evaluate from FREE

            MUST_CASE + valid_cont       → FREE (completion)
            MUST_CASE + other            → FREE (violation), re-evaluate from FREE

            MUST_CONT + valid_next       → MUST_CONT (if extension) or FREE (completion)
            MUST_CONT + invalid          → FREE (violation), re-evaluate from FREE
        """
        word = word.strip().rstrip("।!?.,;:\"')")
        if not word:
            return self._state

        if self._state == FREE:
            self._state = self._from_free(word)

        elif self._state == MUST_AUX:
            if word in ALL_AUX:
                self._completions += 1
                self._state = FREE
            else:
                self._violations += 1
                self._state = FREE
                self._state = self._from_free(word)

        elif self._state == MUST_CASE:
            valid = MUST_CASE_VALID.get(self._case_trigger, frozenset())
            self._case_trigger = None
            if word in valid:
                self._completions += 1
                self._state = FREE
            else:
                self._violations += 1
                self._state = FREE
                self._state = self._from_free(word)

        elif self._state == MUST_CONT:
            if word in self._cont_valid_next:
                # Check whether this step has a further continuation
                ext = self._cont_extension.get(word)
                if ext is not None:
                    # Stay in MUST_CONT for the next step
                    self._cont_valid_next = frozenset({ext})
                    self._cont_extension = {}
                    self._cont_start = None
                    # still in MUST_CONT — obligation not yet complete
                else:
                    self._completions += 1
                    self._cont_valid_next = frozenset()
                    self._cont_extension = {}
                    self._cont_start = None
                    self._state = FREE
            else:
                self._violations += 1
                self._cont_valid_next = frozenset()
                self._cont_extension = {}
                self._cont_start = None
                self._state = FREE
                self._state = self._from_free(word)

        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _from_free(self, word: str) -> str:
        """Determine next state when currently in FREE state."""

        # Priority 1: ke/ki → MUST_CASE (compound postposition must follow)
        if word in MUST_CASE_TRIGGERS:
            self._case_trigger = word
            self._obligations += 1
            return MUST_CASE

        # Priority 2: other phrase starters → MUST_CONT
        if word in PHRASE_VALID_NEXT:
            self._cont_valid_next = PHRASE_VALID_NEXT[word]
            self._cont_extension = PHRASE_EXTENSION.get(word, {})
            self._cont_start = word
            self._obligations += 1
            return MUST_CONT

        # Priority 3: verb-auxiliary chain (ends in ा/ी, is a verb form).
        # Exclude words already classified as auxiliaries, case markers/postpositions,
        # abstract nouns (ता/ना/त्मा suffix), known common nouns, or very short words
        # (≤2 Unicode chars are typically subword prefixes like 'गा', 'जा', not verb forms).
        if (
            len(word) >= 3
            and (word.endswith(A_ENDING) or word.endswith(EE_ENDING))
            and word not in ALL_AUX
            and word not in NON_VERB_AUX_TRIGGERS
            and word not in _COMMON_NOUN_EXCLUSIONS
            and not any(word.endswith(sfx) for sfx in _NON_VERB_SUFFIXES)
        ):
            self._obligations += 1
            return MUST_AUX

        return FREE


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        ("राम खाना खा गया है",  "gaya→MUST_AUX, hai=completion"),
        ("राम के लिए खाना लाया", "के→MUST_CASE, लिए=completion"),
        ("राम के बाद गया",       "के→MUST_CASE, बाद=completion"),
        ("वह जाती रहे हैं बाजार","jaati→MUST_AUX, rahe=MUST_CONT, hain=completion"),
        ("सीता ने खाना खाया था", "sita-name FP, ne→MUST_CONT, khana viol, khaya→MUST_AUX, tha=completion"),
        ("यह हाल ही में हुआ",    "haal→MUST_CONT 3-word phrase"),
        ("राम गया बाजार",        "gaya is AUX itself→no obligation"),
        ("वह हो गया",            "ho→MUST_CONT, gaya=completion"),
        ("वह हो गयी",            "ho→MUST_CONT, gayi=completion"),
    ]

    for sentence, desc in test_cases:
        parser = SamanvayaParser()
        states = [(w, parser.update(w)) for w in sentence.split()]
        print(f"Sentence : {sentence}")
        print(f"Expected : {desc}")
        print(f"States   : {states}")
        print(f"Stats    : {parser.stats()}")
        print()
