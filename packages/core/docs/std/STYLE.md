# Writing the std reference

The reference is read by frontend developers and by AI agents writing shaders for them.
Neither is a graphics engineer. Every word's doc comment and every category page is written
for that reader. This is the style; the extractor (`scripts/docsManifest.ts`) turns it into
the site, `llms.txt` and the manifest, so the comment IS the documentation.

## The shape of a word's comment

```ts
/**
 * One sentence: what it produces, in the reader's terms.
 *
 * One or two more sentences only if they change how you'd use it: the unit an input is in,
 * what the output range is, what it composes with. Nothing about how it is implemented.
 *
 * @example
 * ```ts
 * paint: rampOver(dist.radial({center: p('center'), radius: p('radius'), aspect: 1, skew: 0}), pair(p('colorA'), p('colorB'), p('colorSpace')))
 * ```
 * @tip Sits inside `warped` when you want the rings to flow instead of staying still.
 * @see rings, tiles
 */
```

- **Summary first, one sentence, plain.** It answers "what do I get?" — `A soft round glow at
  a position.` not `The positional cousin of glowSpot taking the squared radius.` The extractor
  uses the first sentence as the summary shown in lists; make it stand alone.
- **Then only what changes usage.** Units (`uv`, degrees, 0–1, pixels), ranges, defaults, what
  it expects as a child or a coordinate, what it returns for the next word. Skip why it was
  built, what it replaced, which shader ported it, which kit file holds the math.
- **One example, one line if possible.** Show the word inside the field it lives in
  (`paint:`, `effect:`, `map:`) so the reader sees where it goes. A second example only for a
  genuinely different use.
- **`@tip` is for the thing you would tell a colleague.** The trap, the idiom, the default that
  surprises. One line. Not every word needs one.
- **`@see` names the neighbours** the reader would reach for next or confuse with this one.
  Bare names, comma-separated.
- **Inputs are described in the sentence, not a table.** The extractor builds the parameter
  table from the types. Add `@param name …` only when the type alone misleads (a number that
  is degrees, a 0–100 slider).

## Voice

- Second person, present tense, imperative when giving instructions. "Pass a prop ref." not
  "A prop ref may be passed."
- Short sentences. One idea each. About fifteen words. No semicolons joining clauses.
- Concrete nouns. "the child's color" not "the composed downstream fragment output".
- Say the reader's word for a thing, then the engine's if needed once: "the layer inside it
  (the child)". After that, "the child".
- No hedging, no marketing, no exclamation marks.
- Lean toward less. If a sentence can go, it goes.

## Words to use and words to avoid

| Say | Not |
|---|---|
| word, a std word | noun, part, primitive, recipe, verb, slot (unless it is the `slots` object) |
| the child, the layer inside | downstream node, the composed fragment, the RTT |
| the color at this pixel | the fragment output, `vec4f` |
| a field (a number per pixel) | a scalar field over the aspect plane |
| the layer's clock, `time` | `animatedTime`, `_animTime`, the node clock |
| in uv (0–1 across the canvas, y down) | in UV space, screen UV, the composed UV |
| props `p('name')` | prop bindings, PropRef slots, uniform accessors |
| compiles to WGSL | lowers, emits, transpiles, resolves |
| runs once per frame | per-frame CPU pass |

Never in a comment meant for readers: kit file names, `PRIMITIVES.md` rule numbers (`C5`,
`D-1`), "L1 tier", port history ("ported from", "the Beam precedent", "née"), "blessed",
"noun", "the language", "the fleet", Simon, ticket or phase references, `'use gpu'`,
`tgpu`, bind groups, uniform structs, hints, snapshots, resolve gates.

**Maintainer notes are not deleted.** Move them into a `//` comment inside the function body
or under the `/** */` block, where the extractor does not read them. A future maintainer still
needs the port history and the reasons; the reader does not.

## What a category page says

Each category has a curation file (`docs/std/<id>.md`). The intro is three to six sentences:

1. What these words are, in one sentence a frontend developer nods at.
2. The one or two concepts you need to hold (a *field* is a number per pixel; a *frame* is
   the coordinate system a light is drawn in). Define by use, not by theory.
3. How they compose: what wraps what, what comes first, what closes the recipe.
4. Where the output goes next (a palette, a `paint:`, an `effect:`).

Then the "Reach for it when" table, phrased as the reader's need ("a glow that fades at a
radius"), the display order (most-used first, plumbing last), and one complete `defineShader`
example that composes this category's words and would render as written.

## Internal words

A word exported for the engine's own use, not for authors (a resolver, a raw accessor, a
compile-time helper), gets `@internal` and no reader-facing comment. It disappears from the
reference. When in doubt, ask: would a frontend developer ever type this? If not, it is
internal.

## Checklist before you finish a module

- [ ] Every exported word has a one-sentence summary that stands alone.
- [ ] No sentence explains implementation. Maintainer notes moved to `//`.
- [ ] Units and ranges named wherever a number goes in.
- [ ] At least one `@example` per word a reader would call directly.
- [ ] `@see` on words with an obvious neighbour.
- [ ] Words nobody should call are `@internal`.
- [ ] The category file's example renders as written (a real definition, real prop names).
- [ ] `npx tsc --noEmit` and `npx vitest run src/__tests__/docsManifest.test.ts` pass.
