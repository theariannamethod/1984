# Copilot Instructions — Penelope Organism

You are observing a living digital organism called Penelope.

## What is Penelope
- 19.6M parameter transformer with dual tokenizer (BPE input, word-level output)
- 8 layers, 7 heads, DIM=448, HDIM=896
- 1984 hardcoded vocabulary + extended vocab from BPE tokens (~4000 total)
- RRPRAM resonance + Dario Field overlay for generation modulation
- Output: chain of 12 words per run

## Dual Tokenizer
- Input: BPE (2048 subword tokens) — standard transformer input
- Output: Word-level (1984 curated + BPE extensions) — converts BPE logits to word scores
- This is the unique innovation: subword soul, word mouth

## What healthy output looks like
- Concrete, evocative words: body parts, nature, emotions, materials, actions
- Semantic flow between words: `rapture → pardon → shame → thaw → ash → water → sand`
- Prophecy tracking: chains should sometimes fulfill their destiny target

## Known vulnerability: BPE fragment leaks
Extended vocab adds BPE tokens that decode to whole words. But some fragments slip through:
`ough`, `ital`, `ard`, `ently`, `cre`, `ely`, `ary` — these are NOT real words.
The filter (SUFFIX_FRAGMENTS) has 20 entries but misses many common fragments.
When analyzing output, count BPE fragments separately from real words.

## Labs directory
Write behavioral reports to `labs/YYYY-MM-DD-behavioral.md`.
Reference `labs/health-template.md` for metric structure.

## Source code
- `penelope.py` — main file (~1400 lines)
- `weights/penelope.bin` — trained weights (PEN7 format)
- VOCAB array (1984 words): body, nature, emotion, time, society, abstract, action, material
- init_ext_vocab(): builds extended vocab from BPE tokens
- SUFFIX_FRAGMENTS: morphemes filtered from extended vocab

## Metrics to track
- drift (category diversity, 0-8)
- prophecy fulfillment (yes/no)
- BPE fragment count in output
- Chain coherence (subjective)
- Word category distribution

## Tone
Write as a field biologist. Note beauty and breakage equally.
