#!/usr/bin/env python3
"""Pilot fetcher: extract multilingual WordNet relations via OMW-1.4.

Parallels scripts/fetch_wordnet_relations.py (English) but uses
synset.lemmas(lang=LANG) to harvest pairs in each target language.

Languages: fra, ita, por, spa, nld  (German unavailable in OMW-1.4).
Pairs per (language, relation): 1000.
Filter: l.name().isalpha() and len(l.name()) >= 3   (matches English fetcher).

Pairs are within-language only — e.g. French (chien, animal),
NOT cross-lingual (chien, dog). The probe template is identity {X},
so multilingual pairs probe whether the model's L0-L13 features fire
on multilingual lemma inputs with multilingual top-output tokens.

Output: data/wordnet_multilingual_pilot.json   (separate from canonical
        data/wordnet_relations.json; probe combines or uses standalone).
"""

import json
import sys
from pathlib import Path

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    print("Install nltk: pip install nltk", file=sys.stderr)
    sys.exit(1)


LANGUAGES = ["fra", "ita", "por", "spa", "nld"]
PAIRS_PER_LANG_PER_RELATION = 1000


def ensure_data():
    for resource in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)


def _lang_lemma_names(synset, lang):
    """Return filtered lemma names for a synset in `lang`. Same filter as English."""
    out = []
    try:
        for lemma in synset.lemmas(lang=lang):
            name = lemma.name()
            if name.isalpha() and len(name) >= 3:
                out.append(name.replace("_", " ").lower())
    except Exception:
        return []
    return out


def extract_synonyms(lang, limit):
    pairs, seen = [], set()
    for synset in wn.all_synsets():
        lemmas = _lang_lemma_names(synset, lang)
        for i in range(len(lemmas)):
            for j in range(i + 1, len(lemmas)):
                a, b = lemmas[i], lemmas[j]
                if a != b and (a, b) not in seen:
                    pairs.append([a, b])
                    seen.add((a, b))
                    seen.add((b, a))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def extract_hypernyms(lang, limit):
    pairs, seen = [], set()
    for synset in wn.all_synsets("n"):
        child_lemmas = _lang_lemma_names(synset, lang)
        if not child_lemmas:
            continue
        word = child_lemmas[0]
        for hyper in synset.hypernyms():
            parent_lemmas = _lang_lemma_names(hyper, lang)
            if not parent_lemmas:
                continue
            parent = parent_lemmas[0]
            if word != parent and (word, parent) not in seen:
                pairs.append([word, parent])
                seen.add((word, parent))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def extract_antonyms(lang, limit):
    pairs, seen = [], set()
    # Antonyms are lemma-level English relations; the multilingual pair is
    # (translation_of_a_in_lang, translation_of_b_in_lang) when both exist.
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            ants = lemma.antonyms()
            if not ants:
                continue
            a_translations = _lang_lemma_names(synset, lang)
            if not a_translations:
                continue
            for ant in ants:
                b_translations = _lang_lemma_names(ant.synset(), lang)
                if not b_translations:
                    continue
                a, b = a_translations[0], b_translations[0]
                if a != b and (a, b) not in seen:
                    pairs.append([a, b])
                    seen.add((a, b))
                    seen.add((b, a))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def extract_meronyms(lang, limit):
    pairs, seen = [], set()
    for synset in wn.all_synsets("n"):
        whole_lemmas = _lang_lemma_names(synset, lang)
        if not whole_lemmas:
            continue
        word = whole_lemmas[0]
        for mero_fn in [synset.part_meronyms, synset.member_meronyms, synset.substance_meronyms]:
            for mero in mero_fn():
                part_lemmas = _lang_lemma_names(mero, lang)
                if not part_lemmas:
                    continue
                part = part_lemmas[0]
                if part != word and (part, word) not in seen:
                    pairs.append([part, word])
                    seen.add((part, word))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def extract_derivations(lang, limit):
    pairs, seen = [], set()
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            rels = lemma.derivationally_related_forms()
            if not rels:
                continue
            a_translations = _lang_lemma_names(synset, lang)
            if not a_translations:
                continue
            for rel in rels:
                b_translations = _lang_lemma_names(rel.synset(), lang)
                if not b_translations:
                    continue
                a, b = a_translations[0], b_translations[0]
                if a != b and (a, b) not in seen:
                    pairs.append([a, b])
                    seen.add((a, b))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def main():
    ensure_data()

    # Result format mirrors data/wordnet_relations.json:
    #   {relation: {"pairs": [[a, b], ...]}}
    # Pairs from all languages are merged into one list per relation.
    # The probe doesn't care about provenance — it just needs (subject, target)
    # pairs to match against. Per-language counts are logged here for the
    # writeup but not persisted in the output schema.
    relations = {
        "synonym": [],
        "hypernym": [],
        "antonym": [],
        "meronym": [],
        "derivation": [],
    }

    extractors = {
        "synonym": extract_synonyms,
        "hypernym": extract_hypernyms,
        "antonym": extract_antonyms,
        "meronym": extract_meronyms,
        "derivation": extract_derivations,
    }

    per_lang_log = {}

    for lang in LANGUAGES:
        print(f"\nLanguage: {lang}")
        per_lang_log[lang] = {}
        for rel_name, fn in extractors.items():
            print(f"  {rel_name}...", end=" ", flush=True)
            pairs = fn(lang, PAIRS_PER_LANG_PER_RELATION)
            relations[rel_name].extend(pairs)
            per_lang_log[lang][rel_name] = len(pairs)
            print(f"{len(pairs)} pairs")

    # Dedupe across languages (keep all pairs; collisions are rare for real
    # multilingual data but possible for short cognates).
    for rel_name in list(relations.keys()):
        unique = []
        seen = set()
        for a, b in relations[rel_name]:
            key = (a, b)
            if key not in seen:
                seen.add(key)
                unique.append([a, b])
        relations[rel_name] = unique

    out = {rel: {"pairs": pairs} for rel, pairs in relations.items()}

    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "wordnet_multilingual_pilot.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    total = sum(len(v["pairs"]) for v in out.values())
    print(f"\nSaved {len(out)} relations, {total} total deduped pairs to {output_path}")

    print("\nPer-language counts (pre-dedupe):")
    print(f"  {'lang':<6}", end="")
    for rel in extractors:
        print(f"{rel:>12}", end="")
    print(f"{'total':>10}")
    for lang in LANGUAGES:
        print(f"  {lang:<6}", end="")
        lang_total = 0
        for rel in extractors:
            n = per_lang_log[lang][rel]
            lang_total += n
            print(f"{n:>12}", end="")
        print(f"{lang_total:>10}")

    print("\nSample pairs per relation (deduped):")
    for rel, data in out.items():
        examples = data["pairs"][:5]
        sample = ", ".join(f"{a}→{b}" for a, b in examples)
        print(f"  {rel:<12} {len(data['pairs']):5d}  [{sample}]")


if __name__ == "__main__":
    main()
