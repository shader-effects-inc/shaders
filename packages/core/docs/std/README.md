# Curating the std reference

The primitive reference on shaders.com (and `llms.txt`, and the JSON manifest agents read) is
generated from one **docs manifest** built by `packages/core/scripts/docsManifest.ts`. Three
sources feed it, in priority order:

1. **Signatures and types**, read from the TypeScript source. Never hand-written.
2. **Doc comments** on each exported word. The first sentence is the summary, the rest is the
   description. A small tag set adds structure:
   - `@example` — a fenced code block showing the word in a definition.
   - `@tip` — one editorial line: the trap, the idiom, the thing an eye-pass learned.
   - `@see` — related word names (bare names or `{@link name}`), resolved to links.
   - `@category` — move the word to another category (default: its module's).
   - `@param` / `@returns` / `@deprecated` / `@internal` (the last excludes the word).
3. **Curation** — one markdown file per category in this folder, named by category id
   (`paint.md`, `light.md`, `effects.color.md`). Until a category has one, its module's header
   comment is the intro and its words list alphabetically.

## The curation file

```md
# Title                      ← replaces the default category title

Intro paragraphs. What the words in this category ARE, how they compose, what they
produce. Written for someone who has never opened the repo.

## Reach for it when         ← the "I need X" table; every backticked name is validated
| When | Use |
|---|---|
| a gradient in a chosen color space | `rampOver` with `pair` or `stops` |

## Order                     ← display order; unlisted words follow alphabetically
- rampOver
- pair

## Example                   ← one complete definition that composes this category's words
```ts
…
```
```

The builder throws when an order entry or a reach-table name is not a word of the category,
so a typo cannot silently hide a word.

## Coverage is enforced

`src/__tests__/docsManifest.test.ts` ratchets the number of undocumented words: it may go
down, never up. When you document words, lower `UNDOCUMENTED_BASELINE`. Every exported word
must also carry a category (its module's, or `@category`).

## Writing comments for the reference

The comments were first written for maintainers. For the reference they must read for an
outside author: lead with what the word produces and when to reach for it, name inputs in
the author's units (uv, degrees, 0–1), give one line of example, link related words by name,
and move maintainer notes ("the D-1 framing", ticket references, port history) into `//`
comments or `PRIMITIVES.md`. Do it category by category alongside curation.
