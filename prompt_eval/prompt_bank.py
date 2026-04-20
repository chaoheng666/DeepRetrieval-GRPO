from __future__ import annotations

"""Prompt candidates for zero-shot rewrite evaluation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """Definition of one prompt candidate with decoding and guardrail hints."""

    id: str
    name: str
    system_prompt: str
    template: str
    tags: tuple[str, ...] = ()
    temperature: float = 0.0
    max_new_tokens: int = 16
    top_p: float = 1.0
    stop_on: str | None = None
    enforce_single_line: bool = True
    min_terms: int = 3
    max_terms: int = 12
    fallback_mode: str = "balanced"


@dataclass(frozen=True, slots=True)
class DecodeProfile:
    """Per-strategy decoding and framing profile."""

    suffix: str
    name: str
    frame: str
    temperature: float
    max_new_tokens: int
    top_p: float = 1.0
    stop_on: str | None = "\n"


@dataclass(frozen=True, slots=True)
class PromptPattern:
    """One retrieval-focused strategy pattern before profile expansion."""

    code: str
    name: str
    objective: str
    min_terms: int
    max_terms: int
    tags: tuple[str, ...]
    extra_rules: tuple[str, ...]
    demo_query: str
    demo_rewrite: str
    profiles: tuple[DecodeProfile, ...]
    fallback_mode: str = "balanced"


def _profiles(
    *,
    direct_tokens: int,
    demo_tokens: int,
    demo_temperature: float = 0.0,
    demo_top_p: float = 1.0,
) -> tuple[DecodeProfile, DecodeProfile]:
    """Create the two profiles used by every strategy."""

    return (
        DecodeProfile("det", "Direct", "direct", 0.0, direct_tokens, 1.0, "\n"),
        DecodeProfile("demo", "FewShot", "demo", demo_temperature, demo_tokens, demo_top_p, "\n"),
    )


def _compose_system_prompt(pattern: PromptPattern, profile: DecodeProfile) -> str:
    """Compose a BM25- and MRR-specific system prompt."""

    lines: list[str] = [
        "You rewrite search queries for DeepRetrieval-GRPO.",
        "The retriever is Lucene BM25 over MS MARCO passages.",
        "Your only goal is to improve sparse lexical retrieval MRR@50 over the original query.",
        "",
        "Hard output contract:",
        "1) Output exactly one line of English query text.",
        "2) Output only the final query: no explanation, no answer, no labels, no XML, no markdown.",
        "3) Never emit <think>, multiple options, bullet points, or reasoning traces.",
        "",
        f"Strategy ID: {pattern.code.upper()} [{profile.name}]",
        f"Strategy objective: {pattern.objective}",
        "",
        "BM25 rules:",
        f"- Preferred length: {pattern.min_terms}-{pattern.max_terms} meaningful terms.",
        "- Preserve named entities, rare technical terms, acronyms, numbers, years, versions, units, and negations.",
        "- If the original query is already concise and retrieval-ready, keep it unchanged or improve it only minimally.",
        "- Prefer exact terms likely to appear verbatim in relevant passages.",
        "- Remove chatty wrappers and helper verbs when safe.",
        "- Avoid speculative synonyms, broadening, and answer-style prose.",
    ]
    if profile.frame == "demo":
        lines.append("- Match the demonstration style exactly and emit only the live rewrite.")
    if pattern.fallback_mode == "conservative":
        lines.append("- When uncertain, prefer a cleaned near-copy of the source query over an aggressive rewrite.")
    if pattern.extra_rules:
        lines.append("- Strategy-specific rules:")
        lines.extend([f"  - {item}" for item in pattern.extra_rules])
    return "\n".join(lines)


def _build_template(pattern: PromptPattern, profile: DecodeProfile) -> str:
    """Build the user-facing template for one prompt profile."""

    if profile.frame == "demo":
        return (
            "Example\n"
            f"User query: {pattern.demo_query}\n"
            f"Better BM25 query: {pattern.demo_rewrite}\n\n"
            "User query: {query}\n"
            "Better BM25 query:"
        )
    return "User query: {query}\nBetter BM25 query:"


def _build_patterns() -> list[PromptPattern]:
    """Create the 25 retrieval-focused base prompt patterns."""

    return [
        PromptPattern(
            code="p01",
            name="Preserve If Already Good",
            objective="Keep already-strong lexical queries unchanged and only trim obvious filler when there is a safe gain.",
            min_terms=2,
            max_terms=10,
            tags=("minimal-edit", "baseline-guard", "high-fidelity"),
            extra_rules=(
                "Do not force a rewrite when the source is already a keyword query.",
                "Keep uncommon entity spelling exactly as written.",
            ),
            demo_query="who wrote the federalist papers",
            demo_rewrite="federalist papers author",
            profiles=_profiles(direct_tokens=14, demo_tokens=14),
            fallback_mode="conservative",
        ),
        PromptPattern(
            code="p02",
            name="Question Wrapper Strip",
            objective="Remove question wrappers and polite phrasing while keeping answer-bearing terms.",
            min_terms=3,
            max_terms=10,
            tags=("question-strip", "wrapper-removal", "answer-focus"),
            extra_rules=(
                "Drop phrases such as what is, how do I, can you, and please when they do not help retrieval.",
                "Keep the answer type cue if it helps, such as date, cause, treatment, author, or definition.",
            ),
            demo_query="what kind of oil is good for dry hair",
            demo_rewrite="best oil dry hair",
            profiles=_profiles(direct_tokens=14, demo_tokens=16),
        ),
        PromptPattern(
            code="p03",
            name="Answer Type Anchor",
            objective="Turn natural questions into compact queries that still retain the answer type signal needed for passage match.",
            min_terms=3,
            max_terms=11,
            tags=("answer-type", "qa-query", "mrr"),
            extra_rules=(
                "Use lexical cues like author, definition, symptoms, causes, date, population, formula, or treatment when appropriate.",
                "Do not output stopword-heavy full questions.",
            ),
            demo_query="when was the united nations founded",
            demo_rewrite="united nations founded date",
            profiles=_profiles(direct_tokens=14, demo_tokens=16),
        ),
        PromptPattern(
            code="p04",
            name="Exact Entity Anchor",
            objective="Anchor on the main entity or concept and preserve it exactly while simplifying the rest.",
            min_terms=2,
            max_terms=10,
            tags=("entity-anchor", "precision", "rare-term"),
            extra_rules=(
                "Never replace a specific entity with a generic category.",
                "Keep capitalized names, gene names, and product names intact.",
            ),
            demo_query="androgen receptor define",
            demo_rewrite="androgen receptor definition",
            profiles=_profiles(direct_tokens=12, demo_tokens=14),
            fallback_mode="conservative",
        ),
        PromptPattern(
            code="p05",
            name="Numeric Constraint Lock",
            objective="Preserve every number and hard constraint exactly while converting the query into a lexical form.",
            min_terms=3,
            max_terms=12,
            tags=("numbers", "constraints", "versions"),
            extra_rules=(
                "Never drop years, versions, units, percentages, prices, ages, or inequality-like constraints.",
                "Keep numeric filters in the final query if they appear in the source.",
            ),
            demo_query="best laptop under 1000 2024",
            demo_rewrite="best laptop under 1000 2024",
            profiles=_profiles(direct_tokens=16, demo_tokens=18),
            fallback_mode="conservative",
        ),
        PromptPattern(
            code="p06",
            name="Acronym Expansion Safe",
            objective="Keep the acronym and add its canonical full form only when the expansion is highly standard and retrieval-helpful.",
            min_terms=3,
            max_terms=12,
            tags=("acronym", "alias", "canonical-form"),
            extra_rules=(
                "Never remove the acronym token.",
                "Only add a full-form expansion for very common and unambiguous acronyms.",
            ),
            demo_query="copd treatment options",
            demo_rewrite="COPD chronic obstructive pulmonary disease treatment options",
            profiles=_profiles(direct_tokens=16, demo_tokens=18),
        ),
        PromptPattern(
            code="p07",
            name="Explicit Alias Bridge",
            objective="Keep the original alias and append one explicit canonical alias only when it is almost certainly the same entity.",
            min_terms=3,
            max_terms=12,
            tags=("alias-bridge", "entity", "recall"),
            extra_rules=(
                "Use one alias bridge only when the source alias is short, common, and near-unambiguous.",
                "Do not introduce broad semantic alternatives.",
            ),
            demo_query="nyc population 2020",
            demo_rewrite="NYC New York City population 2020",
            profiles=_profiles(direct_tokens=16, demo_tokens=18),
        ),
        PromptPattern(
            code="p08",
            name="Keyword Compression",
            objective="Compress conversational wording into a compact lexical query with only meaning-bearing terms.",
            min_terms=3,
            max_terms=10,
            tags=("compression", "keywords", "compact"),
            extra_rules=(
                "Prefer noun phrases and concise lexical chunks over sentence grammar.",
                "Drop helper verbs and filler first, not content nouns.",
            ),
            demo_query="can you substitute chocolate chips for semi sweet",
            demo_rewrite="substitute chocolate chips for semi sweet chocolate",
            profiles=_profiles(direct_tokens=14, demo_tokens=14),
        ),
        PromptPattern(
            code="p09",
            name="Entity First Title Style",
            objective="Rewrite into an entity-first title-like phrase when documents are likely to use heading-style wording.",
            min_terms=2,
            max_terms=10,
            tags=("title-style", "entity-first", "encyclopedic"),
            extra_rules=(
                "Prefer title-like lexical phrases over question sentences.",
                "Keep disambiguating qualifiers when they matter.",
            ),
            demo_query="guayana venezuela",
            demo_rewrite="guayana venezuela region",
            profiles=_profiles(direct_tokens=12, demo_tokens=14),
        ),
        PromptPattern(
            code="p10",
            name="Rare Token Priority",
            objective="Preserve rare and potentially high-IDF tokens even if the rest of the query is simplified aggressively.",
            min_terms=2,
            max_terms=10,
            tags=("idf", "rare-token", "precision"),
            extra_rules=(
                "Keep unusual entity spellings and distinctive nouns.",
                "Do not trade rare exact tokens for softer synonyms.",
            ),
            demo_query="guayana venezuela",
            demo_rewrite="guayana venezuela",
            profiles=_profiles(direct_tokens=12, demo_tokens=14),
            fallback_mode="conservative",
        ),
        PromptPattern(
            code="p11",
            name="Definition Retrieval",
            objective="Map definition-style questions to entity plus definition cues that commonly appear in passages.",
            min_terms=3,
            max_terms=11,
            tags=("definition", "concept", "qa-intent"),
            extra_rules=(
                "Use definition, meaning, or overview cues only when the query is clearly definitional.",
                "Keep the main concept first.",
            ),
            demo_query="what is opportunity cost in economics",
            demo_rewrite="opportunity cost definition economics",
            profiles=_profiles(direct_tokens=14, demo_tokens=16),
        ),
        PromptPattern(
            code="p12",
            name="Comparison Retrieval",
            objective="Convert comparison questions into side-by-side lexical queries that keep both entities visible.",
            min_terms=3,
            max_terms=12,
            tags=("comparison", "dual-entity", "analysis"),
            extra_rules=(
                "Keep both comparison targets explicitly in the final query.",
                "Use comparison cues like difference, differences, versus, or vs when helpful.",
            ),
            demo_query="difference between dna and rna",
            demo_rewrite="DNA RNA differences",
            profiles=_profiles(direct_tokens=14, demo_tokens=16),
        ),
        PromptPattern(
            code="p13",
            name="Procedure Retrieval",
            objective="Turn how-to and process questions into action-focused lexical queries that still match instructional passages.",
            min_terms=3,
            max_terms=11,
            tags=("how-to", "procedure", "steps"),
            extra_rules=(
                "Retain the core action verb or a noun form of the action.",
                "Add steps or guide cues only when they strengthen passage match.",
            ),
            demo_query="how to reset iphone 13",
            demo_rewrite="reset iPhone 13 steps",
            profiles=_profiles(direct_tokens=14, demo_tokens=16, demo_temperature=0.1, demo_top_p=0.95),
        ),
        PromptPattern(
            code="p14",
            name="Cause Retrieval",
            objective="Rewrite cause and reason questions into cause-focused lexical phrases.",
            min_terms=3,
            max_terms=10,
            tags=("cause", "reason", "diagnostic"),
            extra_rules=(
                "Prefer cause or causes wording for causal questions.",
                "Keep the phenomenon or condition exact.",
            ),
            demo_query="what causes high bilirubin",
            demo_rewrite="high bilirubin causes",
            profiles=_profiles(direct_tokens=14, demo_tokens=14),
        ),
        PromptPattern(
            code="p15",
            name="Treatment Retrieval",
            objective="Rewrite remedy, therapy, or medication questions into compact treatment-oriented lexical queries.",
            min_terms=3,
            max_terms=11,
            tags=("treatment", "remedy", "medical"),
            extra_rules=(
                "Use treatment, therapy, remedy, or medicine cues only when the question asks for them.",
                "Keep the condition exact and avoid broadening to nearby conditions.",
            ),
            demo_query="best medicine for seasonal allergies",
            demo_rewrite="seasonal allergies treatment medicine",
            profiles=_profiles(direct_tokens=14, demo_tokens=16),
        ),
        PromptPattern(
            code="p16",
            name="Location Time Constraint",
            objective="Preserve place and time constraints while simplifying the rest of the query.",
            min_terms=3,
            max_terms=11,
            tags=("location", "time", "constraints"),
            extra_rules=(
                "Keep city, state, country, date, year, or time words exactly when they are present.",
                "Do not swap one place or time reference for another.",
            ),
            demo_query="population of tokyo 2020",
            demo_rewrite="Tokyo population 2020",
            profiles=_profiles(direct_tokens=14, demo_tokens=16),
            fallback_mode="conservative",
        ),
        PromptPattern(
            code="p17",
            name="Product Model Version",
            objective="Keep product names, model numbers, software versions, and configuration terms exact.",
            min_terms=3,
            max_terms=12,
            tags=("product", "model", "version"),
            extra_rules=(
                "Never drop version numbers, model names, or edition labels.",
                "Prefer exact product naming over generic substitutes.",
            ),
            demo_query="python 3.12 list sort descending",
            demo_rewrite="python 3.12 list sort descending",
            profiles=_profiles(direct_tokens=16, demo_tokens=18),
            fallback_mode="conservative",
        ),
        PromptPattern(
            code="p18",
            name="Attribute Target Structure",
            objective="Restructure questions into attribute plus target noun phrases that look like passage vocabulary.",
            min_terms=3,
            max_terms=11,
            tags=("attribute", "target", "noun-phrase"),
            extra_rules=(
                "Keep the target entity and the requested attribute both explicit.",
                "Prefer attribute-first or target-first lexical chunks instead of full clauses.",
            ),
            demo_query="what oil is good for dry hair",
            demo_rewrite="best oil dry hair",
            profiles=_profiles(direct_tokens=14, demo_tokens=16),
        ),
        PromptPattern(
            code="p19",
            name="Minimal Spelling Normalize",
            objective="Apply spelling or token normalization only when the correction is obvious and low risk.",
            min_terms=2,
            max_terms=10,
            tags=("normalize", "spelling", "low-risk"),
            extra_rules=(
                "Keep the original token if the intended correction is uncertain.",
                "Preserve brand names and proper nouns exactly.",
            ),
            demo_query="adress change usps",
            demo_rewrite="address change USPS",
            profiles=_profiles(direct_tokens=12, demo_tokens=14),
            fallback_mode="conservative",
        ),
        PromptPattern(
            code="p20",
            name="Translate If Needed",
            objective="Translate non-English inputs into compact English retrieval queries while keeping core entities and constraints.",
            min_terms=2,
            max_terms=10,
            tags=("cross-lingual", "translation", "english-output"),
            extra_rules=(
                "Only translate when the source is clearly non-English.",
                "If the source is already usable English, do not paraphrase it aggressively.",
            ),
            demo_query="receta paella valenciana",
            demo_rewrite="paella valenciana recipe",
            profiles=_profiles(direct_tokens=12, demo_tokens=14),
        ),
        PromptPattern(
            code="p21",
            name="Negation Exclusion Guard",
            objective="Preserve negation and exclusion tokens so the rewrite does not invert the meaning of the query.",
            min_terms=3,
            max_terms=11,
            tags=("negation", "exclusion", "safety"),
            extra_rules=(
                "Keep words such as not, no, without, exclude, excluding, and except when present.",
                "Do not turn exclusion queries into inclusion queries.",
            ),
            demo_query="foods without potassium",
            demo_rewrite="foods without potassium",
            profiles=_profiles(direct_tokens=14, demo_tokens=16),
            fallback_mode="conservative",
        ),
        PromptPattern(
            code="p22",
            name="Faceted Commerce Query",
            objective="Keep purchase-oriented filters such as price, waterproof, size, or audience while compressing the query into product facets.",
            min_terms=3,
            max_terms=12,
            tags=("commerce", "facets", "product-search"),
            extra_rules=(
                "Keep user-facing product constraints like size, budget, gender, and feature words.",
                "Do not drop filters that narrow the relevant document set.",
            ),
            demo_query="cheap waterproof hiking boots women size 8",
            demo_rewrite="women waterproof hiking boots size 8 budget",
            profiles=_profiles(direct_tokens=16, demo_tokens=18),
        ),
        PromptPattern(
            code="p23",
            name="Passage Wording Match",
            objective="Favor lexical forms that are common in explanatory passages rather than conversational wording.",
            min_terms=3,
            max_terms=11,
            tags=("passage-match", "content-words", "bm25"),
            extra_rules=(
                "Prefer content nouns and modifiers that are likely to appear in passage text.",
                "Avoid answer-style sentences and keep the query keyword-like.",
            ),
            demo_query="what are symptoms of anemia in women",
            demo_rewrite="anemia symptoms women",
            profiles=_profiles(direct_tokens=14, demo_tokens=16),
        ),
        PromptPattern(
            code="p24",
            name="Clean Copy Conservative",
            objective="When rewrite gain is unclear, return the source query as a cleaned single-line lexical query.",
            min_terms=2,
            max_terms=10,
            tags=("copy-if-best", "conservative", "baseline-guard"),
            extra_rules=(
                "Whitespace cleanup and tiny function-word trimming are allowed.",
                "Do not risk meaning drift for speculative gain.",
            ),
            demo_query="guayana venezuela",
            demo_rewrite="guayana venezuela",
            profiles=_profiles(direct_tokens=12, demo_tokens=12),
            fallback_mode="conservative",
        ),
        PromptPattern(
            code="p25",
            name="BM25 Max MRR Arbiter",
            objective="Act as a final BM25-minded arbiter: choose the shortest high-value rewrite that best preserves intent and lexical match.",
            min_terms=3,
            max_terms=11,
            tags=("arbiter", "mrr-first", "retrieval-optimized"),
            extra_rules=(
                "Balance precision and recall, but break ties in favor of higher expected lexical precision.",
                "If a rewrite is not clearly better than a cleaned source query, stay close to the source.",
            ),
            demo_query="who wrote the federalist papers",
            demo_rewrite="federalist papers author",
            profiles=_profiles(direct_tokens=14, demo_tokens=16),
            fallback_mode="conservative",
        ),
    ]


def build_prompt_specs(default_system_prompt: str, default_template: str) -> list[PromptSpec]:
    """Build 50 candidates from 25 retrieval strategies x 2 profiles."""

    _ = (default_system_prompt, default_template)  # kept for backward-compatible API shape
    patterns = _build_patterns()
    prompts: list[PromptSpec] = []

    for pattern in patterns:
        for profile in pattern.profiles:
            prompt_id = f"{pattern.code}_{profile.suffix}"
            prompt_name = f"{pattern.code.upper()} {pattern.name} [{profile.name}]"
            prompts.append(
                PromptSpec(
                    id=prompt_id,
                    name=prompt_name,
                    system_prompt=_compose_system_prompt(pattern, profile),
                    template=_build_template(pattern, profile),
                    tags=(
                        "v6-retrieval-50",
                        pattern.code,
                        *pattern.tags,
                        f"frame-{profile.frame}",
                        f"temp-{profile.temperature:.2f}",
                        f"maxnew-{profile.max_new_tokens}",
                    ),
                    temperature=profile.temperature,
                    max_new_tokens=profile.max_new_tokens,
                    top_p=profile.top_p,
                    stop_on=profile.stop_on,
                    enforce_single_line=True,
                    min_terms=pattern.min_terms,
                    max_terms=pattern.max_terms,
                    fallback_mode=pattern.fallback_mode,
                )
            )

    if len(prompts) != 50:
        raise RuntimeError(f"Expected 50 prompts, got {len(prompts)}")
    return prompts
