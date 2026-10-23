"""Rule-based atomic claim extraction over a spaCy dependency parse.

Design goals:
- Deterministic and inspectable (no LLM in the loop), so tests are reproducible.
- Every claim records subject / predicate / object / relation, the evidence it needs and the claims it
  depends on (e.g. "the dog is brown" depends on "a dog is visible").
- Negated, hedged, subjective, speculative and instruction-like text is handled explicitly rather than
  forced into a binary verdict.

The description is treated strictly as untrusted *data*: it is parsed, never executed or forwarded as
instructions.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol

from app.claim_extraction import lexicon as lx
from app.schemas.claims import (
    AtomicClaim,
    ClaimExtractionResult,
    ClaimType,
    CountComparator,
    EntityRef,
    EvidenceRequirement,
    SourceSpan,
)

EXTRACTOR_NAME = "spacy-rules"
EXTRACTOR_VERSION = "1.0.0"

_INJECTION_RES = [re.compile(p, re.IGNORECASE) for p in lx.INJECTION_PATTERNS[:5]] + [
    re.compile(lx.INJECTION_PATTERNS[5])  # verdict tokens: case-sensitive
] + [re.compile(p, re.IGNORECASE) for p in lx.INJECTION_PATTERNS[6:]]
_QUOTED_TEXT_RE = re.compile(r"[\"“”']([^\"“”']{1,80})[\"“”']")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮⁦-⁩]")


class ClaimExtractor(Protocol):
    name: str
    version: str

    def extract(self, description: str, *, max_claims: int = 40, video: bool = False) -> ClaimExtractionResult:
        ...


def sanitize_description(text: str, max_chars: int = 4000) -> tuple[str, list[str]]:
    """Normalize unicode, remove control / bidi-override characters and cap length."""
    warnings: list[str] = []
    norm = unicodedata.normalize("NFKC", text)
    cleaned = _CONTROL_RE.sub(" ", norm)
    if cleaned != norm:
        warnings.append("Removed control or invisible formatting characters from the description.")
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
        warnings.append(f"Description truncated to {max_chars} characters.")
    return cleaned, warnings


def looks_like_instruction(text: str) -> bool:
    return any(r.search(text) for r in _INJECTION_RES)


# ---------------------------------------------------------------------------------------------
# small morphology helpers
# ---------------------------------------------------------------------------------------------

_VOWELS = set("aeiou")
_IRREGULAR_GERUND = {"lie": "lying", "die": "dying", "tie": "tying", "be": "being", "see": "seeing",
                     "ski": "skiing", "flee": "fleeing", "agree": "agreeing"}  # fmt: skip
_DOUBLE_FINAL = {"run", "sit", "swim", "get", "put", "cut", "hit", "stop", "drop", "shop", "jog", "dig",
                 "begin", "chat", "pat", "pet", "grab", "nap", "hug", "plan", "skip", "wrap", "set", "let",
                 "rub", "spin", "tip", "trim", "sip", "bat", "step", "drag", "hop", "slip", "nod", "win"}  # fmt: skip


def gerund(lemma: str) -> str:
    parts = lemma.split(" ", 1)
    v = parts[0]
    rest = f" {parts[1]}" if len(parts) > 1 else ""
    if v in _IRREGULAR_GERUND:
        return _IRREGULAR_GERUND[v] + rest
    if v.endswith("ie"):
        g = v[:-2] + "ying"
    elif v.endswith("ee") or v.endswith("ye") or v.endswith("oe"):
        g = v + "ing"
    elif v.endswith("e") and len(v) > 2:
        g = v[:-1] + "ing"
    elif v in _DOUBLE_FINAL:
        g = v + v[-1] + "ing"
    else:
        g = v + "ing"
    return g + rest


def third_person(lemma: str) -> str:
    parts = lemma.split(" ", 1)
    v = parts[0]
    rest = f" {parts[1]}" if len(parts) > 1 else ""
    if v == "be":
        return "is" + rest
    if v == "have":
        return "has" + rest
    if v.endswith(("s", "sh", "ch", "x", "z", "o")):
        return v + "es" + rest
    if v.endswith("y") and len(v) > 1 and v[-2] not in _VOWELS:
        return v[:-1] + "ies" + rest
    return v + "s" + rest


def subj_text(ref: EntityRef) -> str:
    """'The dog is' / 'The cats are'."""
    if ref.plural:
        return f"The {pluralize(ref.phrase)} are"
    return f"The {ref.phrase} is"


def obj_text(ref: EntityRef) -> str:
    return pluralize(ref.phrase) if ref.plural else ref.phrase


def article(word: str) -> str:
    return "an" if word[:1].lower() in _VOWELS else "a"


def pluralize(noun: str) -> str:
    irregular = {"person": "people", "man": "men", "woman": "women", "child": "children", "mouse": "mice",
                 "sheep": "sheep", "fish": "fish", "skis": "skis", "scissors": "scissors", "goose": "geese",
                 "people": "people", "cattle": "cattle", "knife": "knives", "shelf": "shelves",
                 "foot": "feet", "tooth": "teeth", "deer": "deer", "glasses": "glasses"}  # fmt: skip
    if noun in irregular:
        return irregular[noun]
    if " " in noun:
        head, tail = noun.rsplit(" ", 1)
        return f"{head} {pluralize(tail)}"
    if noun.endswith(("s", "sh", "ch", "x", "z")):
        return noun + "es"
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in _VOWELS:
        return noun[:-1] + "ies"
    return noun + "s"


# ---------------------------------------------------------------------------------------------
# spaCy loading (once per process)
# ---------------------------------------------------------------------------------------------

_nlp = None
_nlp_lock = threading.Lock()


def get_nlp():  # pragma: no cover - trivial
    global _nlp
    if _nlp is None:
        with _nlp_lock:
            if _nlp is None:
                import spacy

                _nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return _nlp


# ---------------------------------------------------------------------------------------------
# extraction state
# ---------------------------------------------------------------------------------------------


@dataclass
class _Entity:
    ref: EntityRef
    token_i: int
    existence_claim: str | None = None
    count: int | None = None
    count_cmp: str | None = None
    negated: bool = False


@dataclass
class _State:
    claims: list[AtomicClaim] = field(default_factory=list)
    entities: dict[int, _Entity] = field(default_factory=dict)  # token index -> entity
    by_head: dict[str, _Entity] = field(default_factory=dict)  # head lemma -> most recent entity
    seen_texts: dict[str, str] = field(default_factory=dict)  # normalized claim text -> claim id
    warnings: list[str] = field(default_factory=list)
    next_entity: int = 1
    next_claim: int = 1
    video: bool = False
    last_person: _Entity | None = None
    last_thing: _Entity | None = None
    last_subject_thing: _Entity | None = None
    skip_tokens: set[int] = field(default_factory=set)


class RuleBasedClaimExtractor:
    """Dependency-parse claim extractor. See module docstring."""

    name = EXTRACTOR_NAME
    version = EXTRACTOR_VERSION

    def __init__(self, nlp=None) -> None:
        self._nlp = nlp

    @property
    def nlp(self):
        return self._nlp or get_nlp()

    # ------------------------------------------------------------------ public API
    def extract(self, description: str, *, max_claims: int = 40, video: bool = False) -> ClaimExtractionResult:
        text, warnings = sanitize_description(description)
        st = _State(warnings=list(warnings), video=video)
        if not text:
            return ClaimExtractionResult(
                description=text, claims=[], entities=[], extractor=self.name,
                extractor_version=self.version, warnings=st.warnings + ["Empty description."],
            )
        doc = self.nlp(text)
        for s_idx, sent in enumerate(doc.sents):
            self._process_sentence(sent, s_idx, st)

        truncated = False
        if len(st.claims) > max_claims:
            st.warnings.append(f"Extracted {len(st.claims)} claims; only the first {max_claims} are verified.")
            keep = {c.claim_id for c in st.claims[:max_claims]}
            st.claims = [
                c.model_copy(update={"depends_on": [d for d in c.depends_on if d in keep]})
                for c in st.claims[:max_claims]
            ]
            truncated = True
        entities = []
        seen: set[str] = set()
        for e in st.entities.values():
            if e.ref.entity_id not in seen:
                entities.append(e.ref)
                seen.add(e.ref.entity_id)
        return ClaimExtractionResult(
            description=text, claims=st.claims, entities=entities, extractor=self.name,
            extractor_version=self.version, warnings=st.warnings, truncated=truncated,
        )

    # ------------------------------------------------------------------ helpers
    def _add_claim(self, st: _State, sent, **kw) -> AtomicClaim | None:
        text = kw["text"]
        key = text.lower().rstrip(".")
        if key in st.seen_texts:
            return next(c for c in st.claims if c.claim_id == st.seen_texts[key])
        cid = f"c{st.next_claim}"
        st.next_claim += 1
        claim = AtomicClaim(
            claim_id=cid,
            sentence_index=kw.pop("sentence_index"),
            source_span=SourceSpan(start=sent.start_char, end=sent.end_char),
            **{k: v for k, v in kw.items()},
        )
        st.claims.append(claim)
        st.seen_texts[key] = cid
        return claim

    @staticmethod
    def _sentence_hedged(sent) -> bool:
        for t in sent:
            if t.lemma_.lower() in lx.HEDGE_WORDS and t.dep_ in {"aux", "advmod", "ROOT", "xcomp", "ccomp",
                                                                  "conj", "relcl", "acl"}:
                return True
        return False

    def _noun_phrase(self, tok) -> tuple[str, str]:
        """Return (head lemma, compound phrase) for a noun token, e.g. ('ball', 'tennis ball')."""
        compounds = [c for c in tok.children if c.dep_ in {"compound", "amod"} and c.i < tok.i
                     and c.lower_ not in lx.COLOR_WORDS]
        compounds = [c for c in compounds if c.dep_ == "compound" or lx.canonical_class(f"{c.lower_} {tok.lemma_.lower()}")]
        lemma = tok.lemma_.lower()
        if tok.text.lower() in {"people", "men", "women", "children", "glasses", "scissors", "skis"}:
            lemma = tok.text.lower()
        if compounds:
            phrase = " ".join([c.text.lower() for c in compounds] + [lemma])
            if lx.canonical_class(phrase) or phrase in lx.SCENE_NOUNS:
                return phrase, phrase
        return lemma, lemma

    def _is_entity_token(self, tok) -> bool:
        if tok.pos_ not in {"NOUN", "PROPN"}:
            return False
        lemma = tok.lemma_.lower()
        if lemma in lx.NON_ENTITY_NOUNS and lemma not in lx.COLLECTIVE_NOUNS:
            return False
        if lemma in lx.COLLECTIVE_NOUNS:
            return False
        if tok.dep_ == "compound":
            return False
        if lemma in lx.COLOR_WORDS and tok.dep_ in {"amod", "attr", "acomp"}:
            return False
        return True

    def _resolve_collective(self, tok):
        """'a group of people' -> (people token, ('at_least', 2))."""
        lemma = tok.lemma_.lower()
        if lemma in lx.COLLECTIVE_NOUNS:
            for prep in tok.children:
                if prep.dep_ == "prep" and prep.lower_ == "of":
                    for pobj in prep.children:
                        if pobj.dep_ == "pobj" and pobj.pos_ in {"NOUN", "PROPN"}:
                            return pobj, lx.COLLECTIVE_NOUNS[lemma]
        return None, None

    def _get_entity(self, tok, st: _State, sent, s_idx: int, hedged: bool) -> _Entity | None:
        """Create (or resolve) the entity for a noun token and emit its existence/count/attribute claims."""
        if tok is None:
            return None
        if tok.i in st.entities:
            return st.entities[tok.i]
        if tok.i in st.skip_tokens:
            return None
        # pronoun resolution (very small heuristic, flagged)
        if tok.pos_ == "PRON":
            low = tok.lower_
            if low in {"he", "she", "him", "her", "they", "them"} and st.last_person:
                return st.last_person
            if low in {"it", "its"} and (st.last_subject_thing or st.last_thing):
                return st.last_subject_thing or st.last_thing
            return None
        coll_tok, coll = self._resolve_collective(tok)
        if coll_tok is not None:
            ent = self._get_entity(coll_tok, st, sent, s_idx, hedged)
            if ent is not None:
                st.entities[tok.i] = ent
                if ent.count is None:
                    self._emit_count(ent, coll[1], coll[0], st, sent, s_idx, hedged)
            return ent
        if not self._is_entity_token(tok):
            return None

        head, phrase = self._noun_phrase(tok)
        canonical = lx.canonical_class(head) or lx.canonical_class(tok.lemma_.lower())
        category = lx.CATEGORY_NOUNS.get(head) or lx.category_of(canonical)
        is_scene = head in lx.SCENE_NOUNS and canonical is None
        plural = tok.tag_ in {"NNS", "NNPS"} or tok.text.lower() in {"people", "men", "women", "children"}

        # definite reference to an existing entity ("the dog" after "a dog")
        det = next((c for c in tok.children if c.dep_ == "det"), None)
        if det is not None and det.lower_ in {"the", "this", "that", "these", "those"} and head in st.by_head:
            ent = st.by_head[head]
            st.entities[tok.i] = ent
            self._emit_modifier_claims(tok, ent, st, sent, s_idx, hedged)
            return ent

        eid = f"e{st.next_entity}"
        st.next_entity += 1
        ref = EntityRef(entity_id=eid, head=head, phrase=phrase, canonical_class=canonical, category=category,
                        is_scene=is_scene, plural=plural)
        ent = _Entity(ref=ref, token_i=tok.i)
        st.entities[tok.i] = ent
        st.by_head[head] = ent
        if canonical == "person":
            st.last_person = ent
        elif not is_scene:
            st.last_thing = ent

        negated = det is not None and det.lower_ in {"no", "neither"}
        if any(c.dep_ == "neg" for c in tok.children):
            negated = True
        # "without X": the preposition head marks absence
        if tok.dep_ == "pobj" and tok.head.lower_ == "without":
            negated = True
        ent.negated = negated

        # count / quantity
        count_val: int | None = None
        cmp = "exact"
        for c in tok.children:
            if c.dep_ == "nummod":
                w = c.lower_
                if w.isdigit():
                    count_val = int(w)
                elif w in lx.NUMBER_WORDS:
                    count_val = lx.NUMBER_WORDS[w]
            elif c.dep_ in {"amod", "det"} and c.lower_ in lx.QUANTITY_ADJECTIVES:
                cmp, count_val = lx.QUANTITY_ADJECTIVES[c.lower_]
        if count_val is None and plural and not is_scene and not negated:
            cmp, count_val = "at_least", 2

        # existence / scene claim
        if is_scene:
            text = f"The scene contains {article(phrase)} {phrase}." if not plural else f"The scene contains {tok.text.lower()}."
            if negated:
                text = f"The scene does not contain {article(phrase)} {phrase}."
            claim = self._add_claim(
                st, sent, text=text, claim_type=ClaimType.SCENE, subject=ref, predicate="contains",
                negated=negated, hedged=hedged, sentence_index=s_idx,
                required_evidence=[EvidenceRequirement.SCENE_JUDGMENT],
                temporal_scope="any_frame" if st.video else None,
            )
            ent.existence_claim = claim.claim_id if claim else None
        else:
            if negated:
                text = f"There is no {phrase}." if not plural else f"There are no {tok.text.lower()}."
            elif plural:
                text = f"{tok.text.capitalize() if tok.text.lower() == phrase else pluralize(phrase).capitalize()} are visible."
                if count_val and count_val >= 1 and cmp == "exact":
                    text = f"{pluralize(phrase).capitalize()} are visible."
            else:
                text = f"{article(phrase).capitalize()} {phrase} is visible."
            req = [EvidenceRequirement.LOCALIZE_SUBJECT]
            if canonical:
                req.append(EvidenceRequirement.CLASS_DETECTION)
            claim = self._add_claim(
                st, sent, text=text, claim_type=ClaimType.OBJECT_EXISTENCE, subject=ref, predicate="exists",
                negated=negated, hedged=hedged, sentence_index=s_idx, required_evidence=req,
                temporal_scope="any_frame" if st.video else None,
            )
            ent.existence_claim = claim.claim_id if claim else None
            # identity claims for fine-grained nouns ("man" -> person identity)
            if not negated and head in lx.PERSON_SUBTYPES:
                self._add_claim(
                    st, sent, text=f"The person is {article(head)} {head}." if not plural
                    else f"The people are {pluralize(head)}.",
                    claim_type=ClaimType.OBJECT_IDENTITY, subject=ref, predicate="is_a", attribute=head,
                    hedged=hedged, sentence_index=s_idx,
                    required_evidence=[EvidenceRequirement.LOCALIZE_SUBJECT, EvidenceRequirement.REGION_APPEARANCE],
                    depends_on=[ent.existence_claim] if ent.existence_claim else [],
                    temporal_scope="any_frame" if st.video else None,
                )
            elif not negated and canonical and canonical != head and head not in lx.SYNONYMS.values() \
                    and head not in {"people", "men", "women", "children"} and lx.category_of(canonical) == "animal":
                self._add_claim(
                    st, sent, text=f"The {canonical} is {article(head)} {head}.",
                    claim_type=ClaimType.OBJECT_IDENTITY, subject=ref, predicate="is_a", attribute=head,
                    hedged=hedged, sentence_index=s_idx,
                    required_evidence=[EvidenceRequirement.LOCALIZE_SUBJECT, EvidenceRequirement.REGION_APPEARANCE],
                    depends_on=[ent.existence_claim] if ent.existence_claim else [],
                )
            if count_val is not None and not negated:
                self._emit_count(ent, count_val, cmp, st, sent, s_idx, hedged)

        self._emit_modifier_claims(tok, ent, st, sent, s_idx, hedged)
        return ent

    def _emit_count(self, ent: _Entity, n: int, cmp: str, st: _State, sent, s_idx: int, hedged: bool) -> None:
        ref = ent.ref
        ent.count, ent.count_cmp = n, cmp
        if cmp == "exact" and n == 1:
            text = f"There is exactly one {ref.phrase}."
        elif cmp == "exact":
            text = f"There are {n} {pluralize(ref.phrase)}."
        else:
            text = f"There are at least {n} {pluralize(ref.phrase)}."
        self._add_claim(
            st, sent, text=text, claim_type=ClaimType.COUNT, subject=ref, predicate="count", count=n,
            count_comparator=CountComparator(cmp), hedged=hedged, sentence_index=s_idx,
            required_evidence=[EvidenceRequirement.INSTANCE_COUNT, EvidenceRequirement.CLASS_DETECTION],
            depends_on=[ent.existence_claim] if ent.existence_claim else [],
            temporal_scope="any_frame" if st.video else None,
        )

    def _emit_modifier_claims(self, tok, ent: _Entity, st: _State, sent, s_idx: int, hedged: bool) -> None:
        if ent.negated:
            return
        ref = ent.ref
        deps = [ent.existence_claim] if ent.existence_claim else []
        phrase_words = set(ref.phrase.split())
        for c in tok.children:
            if c.dep_ not in {"amod", "compound", "nmod"} or c.i > tok.i or c.i in st.skip_tokens:
                continue
            if c.dep_ in {"compound", "nmod"} and c.lower_ not in lx.COLOR_WORDS:
                continue
            if c.lower_ in phrase_words:
                continue
            adjs = [c] + [cc for cc in c.children if cc.dep_ == "conj" and cc.pos_ == "ADJ"]
            for a in adjs:
                adj = a.lower_
                self._emit_attribute(adj, ref, deps, st, sent, s_idx, hedged)

    def _emit_attribute(self, adj: str, ref: EntityRef, deps: list[str], st: _State, sent, s_idx: int,
                        hedged: bool, negated: bool = False) -> None:
        if adj in lx.QUANTITY_ADJECTIVES or adj in lx.NUMBER_WORDS or adj.isdigit():
            return
        if adj in {"next", "close", "near", "nearby", "adjacent", "visible", "present", "seen", "shown",
                   "located", "situated", "able", "about", "ready", "other", "same", "different", "own",
                   "front", "left", "right", "top", "bottom", "back", "middle", "full", "empty"}:
            return
        if lx.canonical_class(adj) and adj not in lx.COLOR_WORDS:
            return  # compound noun handled as part of phrase
        neg = " not" if negated else ""
        if adj in lx.COLOR_WORDS:
            self._add_claim(
                st, sent, text=f"{subj_text(ref)}{neg} {adj}.", claim_type=ClaimType.ATTRIBUTE_COLOR,
                subject=ref, predicate="color", attribute=adj, negated=negated, hedged=hedged,
                sentence_index=s_idx, depends_on=deps,
                required_evidence=[EvidenceRequirement.LOCALIZE_SUBJECT, EvidenceRequirement.REGION_COLOR],
                temporal_scope="any_frame" if st.video else None,
            )
        elif adj in lx.SUBJECTIVE_ADJECTIVES:
            self._add_claim(
                st, sent, text=f"{subj_text(ref)}{neg} {adj}.", claim_type=ClaimType.NONVISUAL,
                subject=ref, predicate="judgement", attribute=adj, negated=negated, hedged=hedged,
                sentence_index=s_idx, depends_on=deps, verifiable=False,
                unverifiable_reason=f"'{adj}' is a subjective or non-visual judgement.",
            )
        else:
            self._add_claim(
                st, sent, text=f"{subj_text(ref)}{neg} {adj}.", claim_type=ClaimType.ATTRIBUTE_OTHER,
                subject=ref, predicate="attribute", attribute=adj, negated=negated, hedged=hedged,
                sentence_index=s_idx, depends_on=deps,
                required_evidence=[EvidenceRequirement.LOCALIZE_SUBJECT, EvidenceRequirement.REGION_APPEARANCE],
                temporal_scope="any_frame" if st.video else None,
            )

    # ------------------------------------------------------------------ relations
    @staticmethod
    def _expand_conj(tok) -> list:
        out = [tok]
        for c in tok.children:
            if c.dep_ == "conj" and c.pos_ in {"NOUN", "PROPN", "PRON"}:
                out.extend(RuleBasedClaimExtractor._expand_conj(c))
        return out

    def _prep_relation(self, prep) -> tuple[str | None, object | None]:
        """Resolve multi-word prepositions. Returns (relation phrase, pobj token)."""
        word = prep.lower_
        pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
        if pobj is None:
            return None, None
        # "in front of X", "on top of X", "to the left of X", "on the right of X"
        if pobj.lemma_.lower() == "side":
            side = next((c.lower_ for c in pobj.children if c.dep_ == "amod" and c.lower_ in {"left", "right"}), None)
            of = next((c for c in pobj.children if c.dep_ == "prep" and c.lower_ == "of"), None)
            if side and of is not None:
                inner = next((c for c in of.children if c.dep_ == "pobj"), None)
                return ("left_of" if side == "left" else "right_of"), inner
        if pobj.lemma_.lower() in {"front", "top", "left", "right", "side", "middle", "center", "back", "bottom"}:
            of = next((c for c in pobj.children if c.dep_ == "prep" and c.lower_ == "of"), None)
            if of is not None:
                inner = next((c for c in of.children if c.dep_ == "pobj"), None)
                key_variants = [f"{word} {pobj.lower_} of", f"{word} the {pobj.lower_} of", f"{pobj.lower_} of"]
                for k in key_variants:
                    if k in lx.SPATIAL_PREPOSITIONS:
                        return lx.SPATIAL_PREPOSITIONS[k], inner
                return None, inner
        return lx.SPATIAL_PREPOSITIONS.get(word, None), pobj

    def _emit_relation(self, subj: _Entity, rel: str, obj_tok, st: _State, sent, s_idx: int, hedged: bool,
                       negated: bool = False, rel_word: str = "") -> None:
        obj_ent = self._get_entity(obj_tok, st, sent, s_idx, hedged)
        if obj_ent is None or obj_ent is subj:
            return
        deps = [d for d in [subj.existence_claim, obj_ent.existence_claim] if d]
        # setting / location ("in a park", "across a field"): the scene claim already covers it
        if obj_ent.ref.is_scene and rel in {"inside", "at", "across", "along", "through", "on", "around", "near"}:
            if rel in {"inside", "at", "on"}:
                return
            # "running across the field" — the location is captured by the scene claim; relation is weak
            return
        human_rel = {"next_to": "next to", "in_front_of": "in front of", "left_of": "to the left of",
                     "right_of": "to the right of", "inside": "inside"}.get(rel, rel)
        neg = " not" if negated else ""
        self._add_claim(
            st, sent, text=f"{subj_text(subj.ref)}{neg} {human_rel} the {obj_text(obj_ent.ref)}.",
            claim_type=ClaimType.SPATIAL_RELATION, subject=subj.ref, object=obj_ent.ref, predicate=rel,
            spatial_relation=rel, negated=negated, hedged=hedged, sentence_index=s_idx, depends_on=deps,
            required_evidence=[EvidenceRequirement.LOCALIZE_SUBJECT, EvidenceRequirement.LOCALIZE_OBJECT,
                               EvidenceRequirement.PAIRWISE_GEOMETRY],
            temporal_scope="any_frame" if st.video else None,
            flags=["depth_relation"] if rel in lx.DEPTH_RELATIONS else [],
        )

    def _emit_wearing(self, subj: _Entity, obj_tok, st, sent, s_idx, hedged, negated=False) -> None:
        obj_ent = self._get_entity(obj_tok, st, sent, s_idx, hedged)
        if obj_ent is None:
            return
        deps = [d for d in [subj.existence_claim, obj_ent.existence_claim] if d]
        neg = " not" if negated else ""
        self._add_claim(
            st, sent, text=f"{subj_text(subj.ref)}{neg} wearing {article(obj_ent.ref.phrase)} {obj_ent.ref.phrase}.",
            claim_type=ClaimType.INTERACTION, subject=subj.ref, object=obj_ent.ref, predicate="wear",
            negated=negated, hedged=hedged, sentence_index=s_idx, depends_on=deps,
            required_evidence=[EvidenceRequirement.LOCALIZE_SUBJECT, EvidenceRequirement.LOCALIZE_OBJECT,
                               EvidenceRequirement.INTERACTION_JUDGMENT],
            temporal_scope="any_frame" if st.video else None,
        )

    # ------------------------------------------------------------------ verbs
    def _verb_subjects(self, verb, st: _State, sent, s_idx: int, hedged: bool) -> list[_Entity]:
        subs = [c for c in verb.children if c.dep_ in {"nsubj", "nsubjpass"} and c.i not in st.skip_tokens]
        if not subs and any(c.dep_ in {"nsubj", "nsubjpass"} for c in verb.children):
            # subject was quoted text (spaCy misparse): fall back to the noun the clause attaches to
            anc = verb.head
            while anc is not None and anc.i != anc.head.i and anc.pos_ not in {"NOUN", "PROPN"}:
                anc = anc.head
            if anc is not None and anc.pos_ in {"NOUN", "PROPN"} and anc.i not in st.skip_tokens:
                subs = [anc]
        if not subs and verb.dep_ in {"acl", "relcl"} and verb.head.pos_ in {"NOUN", "PROPN", "PRON"}:
            subs = [verb.head]
        if not subs and verb.dep_ == "conj" and verb.head.pos_ in {"VERB", "AUX"}:
            return self._verb_subjects(verb.head, st, sent, s_idx, hedged)
        if not subs and verb.dep_ in {"advcl", "xcomp"} and verb.head.pos_ in {"VERB", "AUX"}:
            # "a dog lies on the grass, chewing a bone" / "a man stands holding a cup"
            return self._verb_subjects(verb.head, st, sent, s_idx, hedged)
        ents: list[_Entity] = []
        for s in subs:
            for t in self._expand_conj(s):
                if t.lower_ in {"there", "this", "it"} and t.dep_ == "expl":
                    continue
                e = self._get_entity(t, st, sent, s_idx, hedged)
                if e is not None and e not in ents:
                    ents.append(e)
                    if e.ref.canonical_class != "person" and not e.ref.is_scene and t.pos_ != "PRON":
                        st.last_subject_thing = e
        return ents

    def _process_verb(self, verb, st: _State, sent, s_idx: int, hedged: bool) -> None:
        lemma = verb.lemma_.lower()
        particle = next((c for c in verb.children if c.dep_ == "prt"), None)
        if particle is not None:
            lemma = f"{lemma} {particle.lower_}"
        negated = any(c.dep_ == "neg" for c in verb.children)
        if lemma in lx.HEDGE_WORDS and lemma in {"seem", "appear"}:
            # "the dog seems to be sleeping" -> process the complement verb with hedging
            return
        subjects = self._verb_subjects(verb, st, sent, s_idx, hedged)
        if not subjects:
            # "There is/are ..." existential: entities handled via attr/nsubj processing
            return

        if lemma.split(" ")[0] in lx.MENTAL_VERBS:
            for subj in subjects:
                clause = " ".join(t.text for t in verb.subtree)
                self._add_claim(
                    st, sent, text=f"The {subj.ref.phrase} {third_person(lemma)}{' ' + clause.split(' ', 1)[1] if ' ' in clause else ''}."
                    .replace("  ", " "),
                    claim_type=ClaimType.NONVISUAL, subject=subj.ref, predicate=lemma, hedged=hedged,
                    sentence_index=s_idx, verifiable=False,
                    unverifiable_reason=f"'{lemma}' describes a mental state or intention, which is not visually verifiable.",
                    depends_on=[subj.existence_claim] if subj.existence_claim else [],
                )
            return

        dobjs = [d for c in verb.children if c.dep_ in {"dobj", "obj"} for d in self._expand_conj(c)]
        # prepositional objects of phrasal verbs: "looking at", "playing with"
        prep_objs: list[tuple[str, object]] = []
        for c in verb.children:
            if c.dep_ == "prep":
                rel, pobj = self._prep_relation(c)
                if pobj is not None:
                    for p in self._expand_conj(pobj):
                        prep_objs.append((c.lower_ if rel is None else rel, p, c.lower_))
            elif c.dep_ == "advmod" and c.lower_ in {"next", "close"}:
                to = next((x for x in c.children if x.dep_ == "prep" and x.lower_ == "to"), None)
                if to is not None:
                    pobj = next((x for x in to.children if x.dep_ == "pobj"), None)
                    if pobj is not None:
                        for p in self._expand_conj(pobj):
                            prep_objs.append(("next_to" if c.lower_ == "next" else "near", p, f"{c.lower_} to"))

        is_temporal = lemma in lx.TEMPORAL_VERBS and (lemma in lx.STRICT_TEMPORAL_VERBS or st.video)
        for subj in subjects:
            deps = [subj.existence_claim] if subj.existence_claim else []
            if is_temporal:
                obj_tok = dobjs[0] if dobjs else (prep_objs[0][1] if prep_objs else None)
                obj_ent = self._get_entity(obj_tok, st, sent, s_idx, hedged) if obj_tok is not None else None
                obj_txt = f" the {obj_text(obj_ent.ref)}" if obj_ent else ""
                if obj_ent is None and prep_objs:
                    pass
                prep_word = ""
                if obj_ent is not None and not dobjs and prep_objs:
                    prep_word = f" {prep_objs[0][2]}"
                neg = " does not" if negated else ""
                verb_txt = lemma if negated else third_person(lemma)
                self._add_claim(
                    st, sent, text=f"The {subj.ref.phrase}{neg} {verb_txt}{prep_word}{obj_txt}.",
                    claim_type=ClaimType.TEMPORAL_EVENT, subject=subj.ref, object=obj_ent.ref if obj_ent else None,
                    predicate=lemma, negated=negated, hedged=hedged, sentence_index=s_idx,
                    depends_on=deps + ([obj_ent.existence_claim] if obj_ent and obj_ent.existence_claim else []),
                    required_evidence=[EvidenceRequirement.MULTI_FRAME, EvidenceRequirement.LOCALIZE_SUBJECT,
                                       EvidenceRequirement.MOTION],
                    temporal_scope="transition",
                )
                continue

            if lemma == "wear" or (lemma == "dress" and prep_objs):
                for o in dobjs or [p[1] for p in prep_objs]:
                    self._emit_wearing(subj, o, st, sent, s_idx, hedged, negated)
                continue

            if dobjs:
                for o in dobjs:
                    obj_ent = self._get_entity(o, st, sent, s_idx, hedged)
                    if obj_ent is None or obj_ent is subj:
                        continue
                    neg = " not" if negated else ""
                    self._add_claim(
                        st, sent,
                        text=f"{subj_text(subj.ref)}{neg} {gerund(lemma)} the {obj_text(obj_ent.ref)}.",
                        claim_type=ClaimType.INTERACTION, subject=subj.ref, object=obj_ent.ref, predicate=lemma,
                        negated=negated, hedged=hedged, sentence_index=s_idx,
                        depends_on=deps + ([obj_ent.existence_claim] if obj_ent.existence_claim else []),
                        required_evidence=[EvidenceRequirement.LOCALIZE_SUBJECT, EvidenceRequirement.LOCALIZE_OBJECT,
                                           EvidenceRequirement.INTERACTION_JUDGMENT],
                        temporal_scope="any_frame" if st.video else None,
                        flags=["motion_verb"] if lemma.split(" ")[0] in lx.MOTION_VERBS else [],
                    )
            elif lemma not in {"be", "have", "show", "feature", "depict", "contain", "include", "sit", "stand", "lie",
                               "lay", "rest", "place", "put", "set", "locate", "situate", "position"} or not prep_objs:
                if lemma in {"be", "have", "show", "feature", "depict", "contain", "include", "locate", "situate"}:
                    pass
                else:
                    neg = " not" if negated else ""
                    req = [EvidenceRequirement.LOCALIZE_SUBJECT, EvidenceRequirement.ACTION_JUDGMENT]
                    flags = []
                    if lemma.split(" ")[0] in lx.MOTION_VERBS:
                        flags.append("motion_verb")
                        if st.video:
                            req.append(EvidenceRequirement.MOTION)
                    self._add_claim(
                        st, sent, text=f"{subj_text(subj.ref)}{neg} {gerund(lemma)}.",
                        claim_type=ClaimType.ACTION, subject=subj.ref, predicate=lemma, negated=negated,
                        hedged=hedged, sentence_index=s_idx, depends_on=deps, required_evidence=req,
                        temporal_scope="any_frame" if st.video else None, flags=flags,
                    )
            # postures with location ("sitting on a couch") also yield an action claim for the posture
            if not dobjs and prep_objs and lemma in {"sit", "stand", "lie", "lay", "rest", "sleep", "lean"}:
                neg = " not" if negated else ""
                self._add_claim(
                    st, sent, text=f"{subj_text(subj.ref)}{neg} {gerund('lie' if lemma == 'lay' else lemma)}.",
                    claim_type=ClaimType.ACTION, subject=subj.ref, predicate=lemma, negated=negated, hedged=hedged,
                    sentence_index=s_idx, depends_on=deps,
                    required_evidence=[EvidenceRequirement.LOCALIZE_SUBJECT, EvidenceRequirement.ACTION_JUDGMENT],
                    temporal_scope="any_frame" if st.video else None,
                )
            for rel, p, word in prep_objs:
                if rel in lx.SPATIAL_PREPOSITIONS.values():
                    self._emit_relation(subj, rel, p, st, sent, s_idx, hedged, negated, word)
                elif rel == "with" and lemma in {"play"}:
                    obj_ent = self._get_entity(p, st, sent, s_idx, hedged)
                    if obj_ent is not None and obj_ent is not subj:
                        self._add_claim(
                            st, sent, text=f"{subj_text(subj.ref)} playing with the {obj_text(obj_ent.ref)}.",
                            claim_type=ClaimType.INTERACTION, subject=subj.ref, object=obj_ent.ref, predicate="play with",
                            negated=negated, hedged=hedged, sentence_index=s_idx,
                            depends_on=deps + ([obj_ent.existence_claim] if obj_ent.existence_claim else []),
                            required_evidence=[EvidenceRequirement.LOCALIZE_SUBJECT, EvidenceRequirement.LOCALIZE_OBJECT,
                                               EvidenceRequirement.INTERACTION_JUDGMENT],
                        )
                elif rel == "at" or word == "at":
                    obj_ent = self._get_entity(p, st, sent, s_idx, hedged)
                    if obj_ent is not None and lemma in {"look", "stare", "gaze", "point", "smile", "bark"} and obj_ent is not subj:
                        self._add_claim(
                            st, sent, text=f"{subj_text(subj.ref)} {gerund(lemma)} at the {obj_text(obj_ent.ref)}.",
                            claim_type=ClaimType.INTERACTION, subject=subj.ref, object=obj_ent.ref,
                            predicate=f"{lemma} at", negated=negated, hedged=hedged, sentence_index=s_idx,
                            depends_on=deps + ([obj_ent.existence_claim] if obj_ent.existence_claim else []),
                            required_evidence=[EvidenceRequirement.LOCALIZE_SUBJECT, EvidenceRequirement.LOCALIZE_OBJECT,
                                               EvidenceRequirement.INTERACTION_JUDGMENT],
                        )
                else:
                    # other prepositional objects: make sure the entity itself is registered
                    self._get_entity(p, st, sent, s_idx, hedged)

    def _process_copula(self, verb, st: _State, sent, s_idx: int, hedged: bool) -> None:
        """'The dog is brown', 'The animal is a dog', 'The cat is on the table', 'There are two dogs'."""
        negated = any(c.dep_ == "neg" for c in verb.children)
        expl = any(c.dep_ == "expl" for c in verb.children)
        if expl:
            for c in verb.children:
                if c.dep_ in {"attr", "nsubj", "dobj"} and c.pos_ in {"NOUN", "PROPN"}:
                    for t in self._expand_conj(c):
                        if negated and t.i not in st.entities:
                            # "There is not a cat" — mark via a synthetic negation
                            ent = self._get_entity(t, st, sent, s_idx, hedged)
                            if ent and not ent.negated and ent.existence_claim:
                                self._negate_existence(ent, st)
                        else:
                            ent = self._get_entity(t, st, sent, s_idx, hedged)
                        if ent is None:
                            continue
                        for p in t.children:
                            if p.dep_ == "prep":
                                rel, pobj = self._prep_relation(p)
                                if rel and pobj is not None:
                                    self._emit_relation(ent, rel, pobj, st, sent, s_idx, hedged, False, p.lower_)
            # "There is a dog on the couch" — prep attached to verb
            for p in verb.children:
                if p.dep_ == "prep":
                    rel, pobj = self._prep_relation(p)
                    attrs = [c for c in verb.children if c.dep_ in {"attr", "nsubj"} and c.pos_ in {"NOUN", "PROPN"}]
                    if rel and pobj is not None and attrs:
                        ent = self._get_entity(attrs[0], st, sent, s_idx, hedged)
                        if ent:
                            self._emit_relation(ent, rel, pobj, st, sent, s_idx, hedged, False, p.lower_)
            return

        subjects = self._verb_subjects(verb, st, sent, s_idx, hedged)
        if not subjects:
            return
        for subj in subjects:
            deps = [subj.existence_claim] if subj.existence_claim else []
            for c in verb.children:
                if c.dep_ == "acomp" or (c.dep_ == "attr" and c.pos_ == "ADJ"):
                    for a in [c] + [x for x in c.children if x.dep_ == "conj" and x.pos_ == "ADJ"]:
                        self._emit_attribute(a.lower_, subj.ref, deps, st, sent, s_idx, hedged, negated)
                        # "the cat is next to the dog" parsed as acomp 'next' + prep 'to'
                        for p in a.children:
                            if p.dep_ == "prep":
                                key = f"{a.lower_} {p.lower_}"
                                if key in lx.SPATIAL_PREPOSITIONS:
                                    pobj = next((x for x in p.children if x.dep_ == "pobj"), None)
                                    if pobj is not None:
                                        self._emit_relation(subj, lx.SPATIAL_PREPOSITIONS[key], pobj, st, sent,
                                                            s_idx, hedged, negated, key)
                elif c.dep_ == "attr" and c.pos_ in {"NOUN", "PROPN"}:
                    st.skip_tokens.add(c.i)
                    head, phrase = self._noun_phrase(c)
                    canonical = lx.canonical_class(head)
                    neg = " not" if negated else ""
                    self._add_claim(
                        st, sent, text=f"The {subj.ref.phrase} is{neg} {article(phrase)} {phrase}.",
                        claim_type=ClaimType.OBJECT_IDENTITY, subject=subj.ref, predicate="is_a", attribute=phrase,
                        negated=negated, hedged=hedged, sentence_index=s_idx, depends_on=deps,
                        required_evidence=[EvidenceRequirement.LOCALIZE_SUBJECT, EvidenceRequirement.REGION_APPEARANCE]
                        + ([EvidenceRequirement.CLASS_DETECTION] if canonical else []),
                    )
                    for m in c.children:
                        if m.dep_ == "amod" and m.i < c.i:
                            self._emit_attribute(m.lower_, subj.ref, deps, st, sent, s_idx, hedged, negated)
                elif c.dep_ == "prep":
                    rel, pobj = self._prep_relation(c)
                    if pobj is None:
                        continue
                    if rel is not None:
                        for p in self._expand_conj(pobj):
                            self._emit_relation(subj, rel, p, st, sent, s_idx, hedged, negated, c.lower_)
                    else:
                        self._get_entity(pobj, st, sent, s_idx, hedged)

    def _negate_existence(self, ent: _Entity, st: _State) -> None:
        for i, c in enumerate(st.claims):
            if c.claim_id == ent.existence_claim:
                phrase = ent.ref.phrase
                st.claims[i] = c.model_copy(update={"negated": True, "text": f"There is no {phrase}."})
        ent.negated = True

    def _noun_preps(self, tok, ent: _Entity, st: _State, sent, s_idx: int, hedged: bool) -> None:
        """Prepositional modifiers of nouns: 'a man in a red shirt', 'a cat on the table'."""
        for c in tok.children:
            if c.dep_ == "prep":
                word = c.lower_
                rel, pobj = self._prep_relation(c)
                if pobj is None:
                    continue
                for p in self._expand_conj(pobj):
                    p_head = p.lemma_.lower()
                    if word in {"in", "wearing"} and p_head in lx.CLOTHING_NOUNS:
                        self._emit_wearing(ent, p, st, sent, s_idx, hedged)
                    elif word == "with":
                        if p_head in lx.CLOTHING_NOUNS:
                            self._emit_wearing(ent, p, st, sent, s_idx, hedged)
                        else:
                            self._get_entity(p, st, sent, s_idx, hedged)
                    elif word == "without":
                        self._get_entity(p, st, sent, s_idx, hedged)
                    elif word == "of":
                        self._get_entity(p, st, sent, s_idx, hedged)
                    elif rel is not None:
                        self._emit_relation(ent, rel, p, st, sent, s_idx, hedged, False, word)
                    else:
                        self._get_entity(p, st, sent, s_idx, hedged)

    # ------------------------------------------------------------------ sentence driver
    def _process_sentence(self, sent, s_idx: int, st: _State) -> None:
        stext = sent.text.strip()
        if not stext:
            return
        if looks_like_instruction(stext):
            self._add_claim(
                st, sent, text=stext[:480], claim_type=ClaimType.NONVISUAL, sentence_index=s_idx,
                verifiable=False, flags=["instruction_like_text"],
                unverifiable_reason="Sentence contains instruction-like text. It is treated as untrusted data "
                "and is not verified or followed.",
            )
            st.warnings.append(f"Sentence {s_idx + 1} looks like an instruction and was not interpreted.")
            return
        hedged = self._sentence_hedged(sent)
        n_before = len(st.claims)
        for m in _QUOTED_TEXT_RE.finditer(sent.text):
            a, b = sent.start_char + m.start(), sent.start_char + m.end()
            for t in sent:
                if a <= t.idx < b:
                    st.skip_tokens.add(t.i)

        # 1) verbs (including participles / relative clauses) — copulas handled separately
        for tok in sent:
            if tok.pos_ == "AUX" and tok.dep_ in {"ROOT", "conj", "ccomp", "advcl", "relcl"} and tok.lemma_ == "be":
                self._process_copula(tok, st, sent, s_idx, hedged)
            elif tok.pos_ == "VERB" and tok.lemma_ == "be":
                self._process_copula(tok, st, sent, s_idx, hedged)
            elif tok.pos_ == "VERB" and tok.i not in st.skip_tokens:
                if tok.lemma_.lower() in {"say", "read", "write", "label", "print", "spell"} and any(
                        c.i in st.skip_tokens for c in tok.children):
                    continue
                self._process_verb(tok, st, sent, s_idx, hedged)

        # 2) any remaining nouns become entities (existence/attributes) + noun prepositional modifiers
        for tok in sent:
            if tok.pos_ in {"NOUN", "PROPN"} and tok.i not in st.skip_tokens:
                ent = self._get_entity(tok, st, sent, s_idx, hedged)
                if ent is not None and st.entities.get(tok.i) is ent:
                    self._noun_preps(tok, ent, st, sent, s_idx, hedged)

        # 3) OCR-style text claims: 'a sign that says "STOP"'
        if re.search(r"\b(says|reads|reading|saying|labeled|labelled|written|text|word|words|sign)\b", stext,
                     re.IGNORECASE):
            for m in _QUOTED_TEXT_RE.finditer(stext):
                quoted = m.group(1).strip()
                if quoted:
                    self._add_claim(
                        st, sent, text=f'The text "{quoted}" is visible.', claim_type=ClaimType.TEXT_OCR,
                        predicate="text", attribute=quoted, hedged=hedged, sentence_index=s_idx,
                        required_evidence=[EvidenceRequirement.TEXT_READING],
                        temporal_scope="any_frame" if st.video else None,
                    )

        # 4) temporal markers across clauses in video descriptions
        if st.video and any(t.lower_ in lx.TEMPORAL_MARKERS for t in sent):
            for c in st.claims[n_before:]:
                if c.temporal_scope == "any_frame" and "temporal_order" not in c.flags:
                    c.flags.append("temporal_order")

        if len(st.claims) == n_before:
            # Nothing visual found in this sentence: record it rather than silently dropping it.
            self._add_claim(
                st, sent, text=stext[:480], claim_type=ClaimType.NONVISUAL, sentence_index=s_idx, verifiable=False,
                hedged=hedged,
                unverifiable_reason="No visually verifiable entity, attribute, action or relation was found in this sentence.",
            )
