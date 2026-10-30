# Shader primitives catalog

An index of the reusable building blocks available when authoring a shader in
`packages/core/src/shaders/`. It is organized by **the question you're asking** ("I need a mask from
a distance field", "my shader is a wipe"), not by kit file.

**The rule: check here before writing inline math.** If a primitive covers what you're about to
write, call it. Hand-rolling something in this catalog is a review-blocking defect — the point of
the layer is that a correctness or performance fix made once reaches every consumer, and an inline
copy silently opts out of that.

Entries are terse on purpose. This is an index that points at real call sites, not a second copy of
the implementation. **Follow the "Used by" pointers and read one** — that is faster and more
reliable than reconstructing usage from a signature.

Every section is populated, but the catalog is an index and not an inventory — a missing entry does not
mean nothing exists, so grep the kit (`packages/core/src/gpu/kit/`) before concluding you must write
something yourself.

Conventions for *building or changing* a primitive live in `PRIMITIVES.md` (next to this file).
Don't read that one to author a shader.

## Sections

This file is long enough that a single read truncates partway through it. **Jump to the section you
need by its heading** rather than reading top to bottom — grep the heading text, or use an offset read.

| Section | You're in it when |
|---|---|
| Constants & scalar helpers | You're about to write a named numeric literal or a one-line scalar remap |
| Coordinate frames & aspect math | Turning `ctx.uv` into a centered, aspect-correct, possibly rotated space |
| Tiling & cells | The canvas is divided into a repeating grid, hex lattice or brick pattern |
| Distance fields & SDF | A shape's coverage comes from a signed distance |
| Noise & procedural fields | The shader's core value comes from a noise, fbm or cellular field |
| Anti-aliasing & masks | An edge needs to be soft by one pixel rather than one UV unit |
| Color & tone | Mapping a scalar to color, or blending colors in a chosen color space |
| Sampling & texture ops | Reading a texture at a computed coordinate, or displaying external media |
| Blur & bloom | The shader gathers many samples of its child — motion blur, Gaussian, glow, halation |
| Lighting & materials | Shading a surface (bevel, studio reflection, fresnel) or building a light generator |
| Wipes & transitions | A `progress` prop drives content disappearing |
| **Scaffolds (definition factories)** | **Your shader is a standard member of an established category — check here EARLY, the factory should write most of the file** |
| Simulation harnesses | The shader carries state between frames: particles, fluids, feedback trails |
| Host/CPU lifecycle helpers | Loading external resources, tracking pointer input, rasterizing to a canvas |
| Cross-cutting: per-node animated time | The shader has a continuously advancing clock |
| Cross-cutting: prop-config factories | Declaring a prop block the fleet already standardizes |

## How entries are written

Every primitive gets exactly this shape:

```md
### <primitiveName>
- **Import:** `import {x} from "@coreroot/gpu/kit"` (or porters/scaffolds path)
- **Signature:** `<args> → <return>` (note struct params, compile-time vs runtime args)
- **Use when:** one sentence. **Do NOT hand-roll:** the inline pattern this replaces.
- **Used by:** 2–3 exemplar shaders (so the author can read a real call site).
- **Example:** minimal composing snippet (3–8 lines).
- **Traps:** anything sharp (fn-arg texture rule, $name for factories, Gate C history).
```

Two words that recur and matter:

- **body fn** — a `tgpu.fn` you call from inside a `'use gpu'` body, on concrete vec values.
- **Expr builder** — a JS function you call from `fragment` / `uvRemap` at composition time, on
  `Expr` trees. The two are not interchangeable; a builder cannot be called from a body, and a body
  fn cannot be applied to an `Expr` except through `call(...)`.

---

## Constants & scalar helpers

*In this category when you're about to write a numeric literal with a name, or a one-line scalar
utility.*

### constants
- **Import:** `import {constants} from "@coreroot/gpu/kit"` (or named: `import {DEG_TO_RAD} from "@coreroot/gpu/kit/constants"`)
- **Signature:** module-scope `const number`s — `DEG_TO_RAD`, `RAD_TO_DEG`, `PI`, `TAU`, `TWO_PI`
  (alias of `TAU`), `HALF_PI`, `SQRT3`, `GOLDEN` (the golden angle, π·(3−√5)).
- **Use when:** a `'use gpu'` body needs π, a turn, a degree→radian factor, √3, or the golden angle.
  **Do NOT hand-roll:** a per-file `const DEG_TO_RAD = Math.PI / 180`, or a transcribed
  `0.0174532925…` literal. `Math.*` inside a body is still a hard trap — but the fix is importing
  these, not re-deriving them.
- **Used by:** `Stripes`, `HexGrid`, `Crystal` (≈60 shaders after the Phase 1 sweep).
- **Example:**
  ```ts
  import {tgpu, d, std, constants} from "@coreroot/gpu/kit"

  // Re-bind to a module const: the transpiler folds a module-scope const's JS value into a WGSL
  // literal, and that binding is what the body may reference.
  const DEG_TO_RAD = constants.DEG_TO_RAD

  export const rotate = tgpu.fn([d.vec2f, d.f32], d.vec2f)((p, deg) => {
      'use gpu'
      const a = deg * DEG_TO_RAD
      return d.vec2f(p.x * std.cos(a) - p.y * std.sin(a), p.x * std.sin(a) + p.y * std.cos(a))
  })
  ```
- **Traps:**
  - **Re-bind, don't dereference.** A member expression (`constants.DEG_TO_RAD`) inside a `'use gpu'`
    body is NOT the captured-const pattern and does not fold. Either use a named import from
    `@coreroot/gpu/kit/constants`, or write `const DEG_TO_RAD = constants.DEG_TO_RAD` at module scope
    and reference that identifier from the body.
  - An imported const inlines to **exactly** the same WGSL literal as an identical local one — Gate
    A, verified across 58 shader files (only the 5 whose local value was truncated changed at all).
  - Several shaders historically carried truncated values, which is why this file exists. If you find
    another truncated literal, replacing it is Gate C (pixel-changing) — log it, don't fold it into
    an otherwise Gate A change. The fleet has been swept; exactly one is knowingly outstanding, `Spiral`
    (`6.283185307`), left alone to honour a one-change-only scope on that file.

### tone.luma709 / luma601 / luma709Dot / luma601Dot
- **Import:** `import {tone} from "@coreroot/gpu/kit"`
- **Signature:** body fns `(rgb: vec3f) → f32`
- **Use when:** any luminance read **from a shader**. This is the one to call in new shader code;
  `blend.luminance` below is the internal blend-mode twin, not the shader-facing entry point.
  **Prefer the `Dot` spelling in new code** — it is what the fleet's
  inline sites already write. `luma709`/`luma709Dot` are BT.709 (0.2126/0.7152/0.0722), the default for
  new code; the 601 pair (0.299/0.587/0.114) exists so a shader already using those weights can name
  them without a pixel change. **Do NOT hand-roll:** either weight triple inline.
- **Used by:** `ContourLines` (709 dot), `LightLeak`, `LensFlare` (601 dot).
- **Example:**
  ```ts
  const lum = tone.luma709Dot(color.xyz)
  ```
- **Traps:** The two spellings are not interchangeable at Gate A. `luma709`/`luma601` are component
  multiply-add (`rgb.x * w + …`), character-identical to `blend.luminance`; the `Dot` twins are
  `std.dot(rgb, vec3f(w))`. They resolve to DIFFERENT WGSL, so a migration picks whichever spelling
  the file already used — that is the only way the swap stays byte-identical, and it is why both
  exist. (~33 sites repo-wide still inline the weights; the mad-vs-dot unification sweep is still
  available as one deliberate snapshot regeneration, but it no longer blocks anything.) Picking 601 vs
  709 for an EXISTING shader is not a free choice either: match the weights the file uses today (D-2).

### tone.pivotContrast / signedToUnit / ridged
- **Import:** `import {tone} from "@coreroot/gpu/kit"`
- **Signature:** body fns. `pivotContrast(x: f32, contrast: f32, pivot: f32) → f32`
  (`clamp((x − pivot) · contrast + pivot, 0, 1)`); `signedToUnit(v: f32) → f32` (`v · 0.5 + 0.5`);
  `ridged(v: f32) → f32` (`1 − abs(v)`).
- **Use when:** remapping a scalar field — contrast about mid-gray, a signed noise into [0,1], or a
  signed field's zero crossings into creases (ridged multifractal, marble veining).
  **Do NOT hand-roll:** any of those three one-liners.
- **Used by:** **nothing yet.** The generators that inline these took `tone.toneRemap`'s whole-tail
  variants instead (see Noise & procedural fields), so these three stay available for a new shader that
  needs just one step.
- **Traps:** `pivotContrast` clamps — if your shader relies on out-of-range values surviving the
  remap, it is not this primitive. `contrast = 1` is identity.

**Periodic wave shaping is deliberately inline — there is no `triangleWave` / `sawtooth` primitive.**
Write the one-liner (`abs(fract(p) * 2 - 1)`, `fract(p)`, …) in your own body. Two reasons it stays that
way: a waveform is usually a `compileTime` prop, so the shader JS-branches once and emits only the
active body (`WaveDistortion` keeps five per-waveform `tgpu.fn`s and picks one — read it before adding a
sixth waveform anywhere), and a periodic pattern that needs ANTI-ALIASING wants the closed-form filtered
form instead, not the raw wave — see `aa.quilezStepFilter` / `quilezCheckerFilter` under
Anti-aliasing & masks. If you find yourself writing a raw `fract`-based wave for a visible edge, you
probably want the filter.

---

## Coordinate frames & aspect math

*In this category when you need to turn `ctx.uv` into a centered, aspect-correct, possibly rotated
space.*

### sdf.shapeLocalCoords
- **Import:** `import {sdf} from "@coreroot/gpu/kit"`
- **Signature:** body fn `(center: vec2f, rotation: f32 /*degrees*/, uv: vec2f, aspect: f32) → vec2f`
- **Use when:** a shape or shape-like effect needs UV in shape-local space. **Do NOT hand-roll:** the
  aspect-multiply + center double-flip (`1.0 - center.y`) + degree→radian rotation preamble.
- **Used by:** `Heart`, `Star`, `RoundedRect` (13 shapes total).
- **Example:**
  ```ts
  const local = sdf.shapeLocalCoords(center, rotation, uv, aspect)
  const dist = sdf.heartSdf(local.x, local.y, radius)
  ```
- **Traps:** This is the canonical **center-scaled** convention (`center.x * aspect`, not
  UV-divided) — see PRIMITIVES.md D-1. `rotation` is degrees; it folds `DEG_TO_RAD` internally, so
  don't pre-convert. For a NEW non-rotatable shape, passing `rotation = 0` is the right call — it
  keeps the preamble to one line and the dead `cos 0`/`sin 0` pair costs nothing worth naming. Note
  this is not how the existing fleet is written: `Circle` and `Ring` inline the aspect-corrected
  delta instead, deliberately, and must stay that way — routing them through `shapeLocalCoords` would
  emit that trig pair and move their WGSL.

### geom.aspectOf
- **Import:** `import {geom} from "@coreroot/gpu/kit"`
- **Signature:** body fn `(viewport: vec2f) → f32` — `viewport.x / max(viewport.y, 1e-6)`
- **Use when:** a body derives aspect from a viewport size. **Do NOT hand-roll:** the divide, and
  in particular do not write it unguarded.
- **Used by:** `Truchet`, `HexGrid`, `Marble` (~30 generators).
- **Traps:** The guard is the whole point: a canvas can report zero height for one frame during
  init/resize, and unguarded that frame renders NaN which then propagates through everything
  downstream. `coords.aspectOf` is the older unguarded version this supersedes — don't reach for it.
  Reach for this bare form when you need `aspect` itself (a center of rotation, a later un-correct);
  when you only need the corrected UV, the framing fns below take the viewport and do the divide once.

### geom.aspectCorrectedUV / aspectCorrectedUVFlipY
- **Import:** `import {geom} from "@coreroot/gpu/kit"` then `const {aspectCorrectedUV} = geom`
- **Signature:** body fns `(uv: vec2f, viewport: vec2f) → vec2f`. All runtime.
- **Use when:** a pattern's cells must be square. Stretches X into Y's units, so the domain is
  `[0, aspect] × [0, 1]` and the canvas center sits at `(aspect * 0.5, 0.5)`. **Do NOT hand-roll:**
  `vec2f(uv.x * (viewport.x / viewport.y), uv.y)` — that is the unguarded divide that renders NaN for
  the frame a canvas initializes at zero height.
- **Used by:** `Stripes`, `ColorWheel` (unflipped); `Checkerboard`, `DotGrid` (flipped).
- **Example:**
  ```ts
  const {aspectCorrectedUV, directionalProjection} = geom
  // inside a 'use gpu' body:
  const p = directionalProjection(aspectCorrectedUV(uv, viewport), std.cos(a), std.sin(a))
  ```
- **Traps:** **The flip choice is in the NAME, not an argument.** D-1 makes `flipY` explicit with no
  default, and TGSL cannot branch on a JS option inside a body, so two fns is the only honest
  encoding — there is no `{flipY}` option to pass. If you also need `aspect` itself, call
  `geom.aspectOf` and spell the `vec2f` inline rather than paying for the divide twice; six migrated
  shaders do that, each with a one-line comment naming the framing it reproduces.

### geom.unflipPosition / aspectCentrePosition
- **Import:** `const {unflipPosition, aspectCentrePosition} = geom`
- **Signature:** body fns. `unflipPosition(p: vec2f) → vec2f` returns `(p.x, 1 - p.y)` and is its own
  inverse; `aspectCentrePosition(pos: vec2f, aspect: f32) → vec2f` returns `(pos.x * aspect, 1 - pos.y)`.
- **Use when:** reading any `transformPosition` prop. `unflipPosition` when the position stays in raw
  `[0,1]` UV space; `aspectCentrePosition` when several centres move into aspect-corrected space and
  you want the divide taken once. **Do NOT hand-roll:** `1.0 - center.y` — the single most-copied line
  in the gradient fleet — or `vec2f(pos.x * aspect, 1.0 - pos.y)` per point.
- **Used by:** `LinearGradient` (both endpoints, bare); `MultiPointGradient` (five control points).
- **Traps:** Most centered generators want `aspectCenteredDelta` below instead — it folds the unflip in.
  `aspectCentrePosition` takes `aspect`, not `viewport`, deliberately, so a five-point shader derives
  the ratio once; pair it with `geom.aspectOf`.

### geom.aspectCenteredDelta / canvasCentredDelta
- **Import:** `const {aspectCenteredDelta, canvasCentredDelta} = geom`
- **Signature:** body fns. `aspectCenteredDelta(uv: vec2f, center: vec2f, viewport: vec2f) → vec2f`
  — `center` is the RAW `transformPosition` prop value, the unflip happens inside;
  `canvasCentredDelta(uv: vec2f, aspect: f32) → vec2f`.
- **Use when:** opening any centered field — radial, conic, diamond, spiral. Everything downstream is a
  function of this delta. Use `canvasCentredDelta` when the pattern pivots on the canvas center and has
  no center prop (it is exactly `aspectCenteredDelta` at `(0.5, 0.5)`, asserted in the kit test).
  **Do NOT hand-roll:** the six-line `safeViewportY` / `aspectUV` / `centerPos` / `dx` / `dy` block
  that was verbatim in three gradients.
- **Used by:** `RadialGradient`, `ConicGradient`, `DiamondGradient`; `Chevron`, `BrickPattern` (canvas).
- **Example:**
  ```ts
  const delta = aspectCenteredDelta(uv, center, viewport)
  const dist = std.length(delta)
  ```
- **Traps:** This is the D-1 CANONICAL form — the **center** is aspect-scaled, not just the UV. Scaling
  the UV and leaving the center alone gives a different, subtly wrong offset at non-square aspects, and
  `kit-geom-primitives.test.ts` asserts the two differ so nobody "simplifies" it back. `LinearGradient`
  and `BrickPattern` keep delta-scaled variants that divide X back out afterwards — both carry a
  DEVIATION comment; do not convert them. `canvasCentredDelta` takes `aspect` because both consumers
  need the ratio again to un-correct the rotated result.

### geom.rotateAboutCanvasCentre / directionalProjection
- **Import:** `const {rotateAboutCanvasCentre, directionalProjection} = geom`
- **Signature:** body fns. `rotateAboutCanvasCentre(p: vec2f, aspect: f32, cosA: f32, sinA: f32) → vec2f`;
  `directionalProjection(p: vec2f, cosA: f32, sinA: f32) → f32` = `p.x * cosA + p.y * sinA`.
- **Use when:** a `rotation` prop should spin the pattern in place, or a pattern is a function of one
  scalar coordinate along a direction (stripe phase, gradient parameter, chevron axis). **Do NOT
  hand-roll:** the `centerX`/`centerY`/`cx`/`cy`/two-component-rotate block that was verbatim in six
  files.
- **Used by:** `Truchet`, `HexGrid`, `IsometricCubes`, `TriangularGrid`, `Weave`, `Grid` (rotate);
  `Stripes`, `ColorWheel` (projection).
- **Example:**
  ```ts
  const aspect = aspectOf(viewport)
  const corrected = d.vec2f(uv.x * aspect, uv.y)
  const ang = -(rotation * DEG_TO_RAD)     // negated → positive reads clockwise
  const rotated = rotateAboutCanvasCentre(corrected, aspect, std.cos(ang), std.sin(ang))
  ```
- **Traps:** `rotateAboutCanvasCentre` must be handed an ALREADY aspect-corrected UV — it derives the
  center from `aspect`, so a raw `[0,1]²` UV pivots around the wrong point. **The caller owns the sign
  of the angle:** three consumers negate it and three do not, and that difference is per-shader on
  purpose. If you need the perpendicular component as well as `directionalProjection`'s, use
  `geom.rotate2` rather than calling it twice (the kit test asserts this equals the X component of
  `rotate2(p, cos, -sin)`).

### geom.toPolar / fromPolar
- **Import:** `const {toPolar, fromPolar} = geom`
- **Signature:** body fns. `toPolar(p: vec2f) → vec2f` as `(radius, angle)`; `fromPolar(r: f32, theta: f32) → vec2f`.
  Angle in radians on `[-π, π]`, zero at 3 o'clock, increasing CCW in a Y-up frame.
- **Use when:** a field is naturally radius-and-angle — spirals, rings, angular sweeps. Divide the
  angle by `constants.TAU` for a turn fraction.
- **Used by:** **nothing yet.** `Spiral` is the natural adopter and is scope-limited to one change;
  the first radial shader that opens that file should take these there.
- **Traps:** `toPolar` computes the radius whether you read it or not, so for angle-only work
  (`ConicGradient`) call `std.atan2(delta.y, delta.x)` directly instead.

### geom.rotate2 / safeDiv / flooredMod1 / flooredMod2
- **Import:** `import {geom} from "@coreroot/gpu/kit"`
- **Signature:** body fns. `rotate2(p: vec2f, cosA: f32, sinA: f32) → vec2f`;
  `safeDiv(n: f32, den: f32) → f32`; `flooredMod1(x: f32, m: f32) → f32`;
  `flooredMod2(p: vec2f, m: vec2f) → vec2f`.
- **Use when:** rotating a point about the origin, dividing by a value that can approach zero, or
  wrapping a coordinate into a tile. **Do NOT hand-roll:** the four-multiply rotation matrix, a
  `max(den, eps)` guard, or `x - m * floor(x / m)`.
- **Used by:** `BrickPattern`, `Chevron`, `LinearGradient`, `RadialGradient`, `DiamondGradient`
  (`rotate2`); `BrickPattern`, `DotGrid`, `Weave` (`flooredMod1`); `cells.hexTiling` (`flooredMod2`).
- **Traps:** `rotate2` takes a **precomputed cos/sin pair**, not an angle, so a caller rotating many
  points computes the trig once and a constant angle folds to two literals. `safeDiv` pushes the
  denominator out to ±1e-5 **preserving its sign** — a plain `max(den, eps)` would flip the
  quotient's sign for small negative denominators, which reads as a discontinuity rather than a large
  value. `flooredMod*` is floored, not WGSL's truncated `%`, so it stays in [0, m) for negative
  coordinates — that difference is why tiling wants it. One identity worth knowing:
  **`rotate2(p, cosA, -sinA)` is the transpose, i.e. the INVERSE rotation** — five shaders had spelled
  that out longhand as `(x·cos + y·sin, −x·sin + y·cos)`. The substitution is pixel-neutral because
  `−(x·sin)` and `x·(−sin)` are bit-identical (negation is exact in IEEE) and addition commutes;
  associativity is not free, so the identity only holds while no additions are re-grouped.

---

## Tiling & cells

*In this category when the canvas is divided into a repeating grid, hex lattice, or brick pattern.*

*Two neighbours live elsewhere: `geom.flooredMod1`/`flooredMod2` for the wrap itself (above), and
`reveal.cellGrid` for a square-ish cell dice of the frame returning cell index / in-cell position /
grid resolution together (under Wipes & transitions — it is not wipe-specific despite living there).*

**Alias the import as `cellKit` in any shader with a `cells` prop**, or the prop shadows the namespace
inside the body: `import {cells as cellKit} from "@coreroot/gpu/kit"`, then destructure at module scope.

### cells.squareTiling → CellTile {cell, local}
- **Import:** `import {cells as cellKit} from "@coreroot/gpu/kit"` then `const {squareTiling} = cellKit`
- **Signature:** body fn `(p: vec2f) → CellTile`, a `d.struct` of `{cell: vec2f, local: vec2f}` =
  `floor`/`fract`.
- **Use when:** splitting a scaled UV into cells. **Do NOT hand-roll:** `floor`/`fract` separately —
  the pairing is what stays consistent for negative coordinates, which a rotation or an upstream
  distortion's `uvContext` will produce.
- **Used by:** `Truchet`, `Weave`, `DotGrid`.
- **Traps:** **Never bind a struct member to a local.** `const local = tile.local` makes typegpu
  promote the struct to a `var` and bind a WGSL pointer; read members inline (`tile.local.x`).
  Destructuring the return fails outright (`Unsupported JS functionality: ObjectPattern`). That is
  also the rule for when this struct is worth it: few reads → struct; many reads threaded through one
  expression → leave `floor`/`fract` inline, which is why `TriangularGrid` and `Grid` deliberately do.

### cells.hexTiling → HexCell {gv, id}
- **Import:** `const {hexTiling} = cellKit`
- **Signature:** body fn `(p: vec2f, s: vec2f) → HexCell`, a `d.struct` of `{gv, id}`. `gv` is the
  vector from the nearest hex center; `id` is that center (`p - gv`).
- **Use when:** any hexagonal lattice. The two-offset-grids fold finds the nearest center with no
  branching, using `geom.flooredMod2` so it stays periodic for negative coordinates. **Do NOT
  hand-roll:** the `sh`/`a`/`b`/`select` block — it was verbatim in two files, each with its own
  private copy of `flooredMod2` on top.
- **Used by:** `HexGrid` (`vec2f(SQRT3, 1)`, pointy-top), `IsometricCubes` (`vec2f(1, SQRT3)`, flat-top).
- **Example:**
  ```ts
  const hex = hexTiling(scaledUV, d.vec2f(SQRT3, 1.0))
  const pAbs = d.vec2f(std.abs(hex.gv.x), std.abs(hex.gv.y))     // read members INLINE
  const rand = cellHash(d.vec2f(std.round(hex.id.x * INV_SQRT3_2), std.round(hex.id.y * 2.0)))
  ```
- **Traps:** `s` is the lattice period AND the orientation — the two shipped orientations are the same
  code with `s` transposed, which is why it is a parameter and not two fns. Hex centres sit on
  half-integer multiples, so `round` `id` onto integers before hashing or the fills band with
  sub-pixel jitter.

### cells.triLattice
- **Import:** `const {triLattice} = cellKit`
- **Signature:** body fn `(p: vec2f) → vec2f`, applying `M = [[1, -1/√3], [0, 2/√3]]`.
- **Use when:** indexing an equilateral-triangle lattice. After the skew each unit square holds two
  triangles split by the anti-diagonal, so `floor`/`fract` indexing works and `fract.x + fract.y > 1`
  picks which of the pair you are in.
- **Used by:** `TriangularGrid`.
- **Traps:** the result is a SHEARED space — distances in it are not distances in the original lattice.
  Take anti-aliasing footprints from this coordinate (it is what the derivative sees) and read edge
  distances in it too, consistently. The X term divides by `SQRT3` rather than multiplying by a folded
  `1/√3`; the two differ in the last bit and the division is what shipped.

### cells.cellCentreUV / cellCentreUVRotated
- **Import:** `const {cellCentreUV, cellCentreUVRotated} = cellKit`
- **Signature:** body fns `(uv, viewport, cells) → vec2f` and `(uv, viewport, cells, rotationRad) → vec2f`.
  Guarded aspect, Y-flipped, returns a SCREEN UV. Rotation in RADIANS.
- **Use when:** returning `mapSampleUVs` from a cell-based generator, so mapped props are sampled once
  per cell instead of once per fragment. Without it a mapped `thickness` or `dotSize` varies *within* a
  cell, clipping the cell's content against the map source's boundaries — the difference between dots
  that grow and dots that get shaved.
- **Used by:** `DotGrid` (plain), `Grid` (rotated, via a one-line wrapper that converts its degrees prop).
- **Example:**
  ```ts
  mapSampleUVs: ({uniforms, ctx}) => {
      const cellCenter = call(cellKit.cellCentreUV, 'cellCentreUV', [ctx.uv, ctx.viewportSize, uniforms.density])
      return {dotSize: cellCenter, twinkle: cellCenter}
  }
  ```
- **Traps:** Two fns rather than one with a rotation argument, because a rotation of zero still costs a
  `cos`, a `sin` and a second un-rotate. The angle is RADIANS so the degrees constant stays at the
  shader's own module scope with the rest of its rotation math — `Grid` keeps a one-line wrapper for
  exactly that, which is also why its emitted WGSL still shows `gridCellCenterUV`. Call it with the RAW
  canvas `viewportSize`, not `effectiveViewportSize`.

### cells.cellHash — legacy sin-fract, FROZEN (D-6)
- **Import:** `const {cellHash} = cellKit`
- **Signature:** body fn `(id: vec2f) → f32` in `[0, 1)`.
- **Use when:** per-cell randomness in an EXISTING shader. **Do NOT hand-roll:** the
  `fract(id * (123.34, 345.45))` → dot-couple → `fract(q.x * q.y)` chain, byte-identical in five files.
- **Used by:** `HexGrid`, `IsometricCubes`, `TriangularGrid`, `Grid`, `BrickPattern`.
- **Traps:** **NEW shaders use `noise.hash*` instead.** This hash has visible structure at large indices
  and is float-precision sensitive, but it is the hash every shipped preset's cell randomness was
  authored against, so D-6 freezes it. `kit-cells.test.ts` asserts HARD-CODED output values rather than
  re-deriving the formula, on purpose: a test that recomputes the formula would happily follow it if
  someone "improved" it. Hash INTEGER indices only.

### cells.rowSpeedHash / rowSpeedMultiplier / variationFactor
- **Import:** `const {rowSpeedHash, rowSpeedMultiplier, variationFactor} = cellKit`
- **Signature:** body fns. `rowSpeedHash(row: f32) → f32` in `[0,1)`;
  `rowSpeedMultiplier(rowHash, variance) → f32`; `variationFactor(rand, variation) → f32` =
  `max(1 + (rand - 0.5) * 2 * variation, 0)`.
- **Use when:** giving each row of a pattern its own drift speed, or turning a cell hash into a per-cell
  brightness multiplier. The row multiplier is centered on 1 (variance 0 leaves every row at the base
  speed) and scaled by 4, spanning `[-1, 3]` at full variance — wide enough that some rows reverse,
  which is what makes the motion read as irregular rather than as one sheared drift. `variationFactor`
  is exactly 1 at variation 0 and spans `[0, 2]` at full variation.
- **Used by:** `DotGrid`, `TriangularGrid` (rows); `HexGrid`, `TriangularGrid`, `Grid`, `BrickPattern`
  (variation).
- **Traps:** `rowSpeedHash` is frozen under D-6 and is `fract(sin(row * 127.1) * 43758.5453)`
  **specifically** — not `scaffolds/shared.legacySinHash11` (12.9898, re-centered to `[-1,1]`), not
  `BrickPattern`'s seed-offset variant, not `DotGrid`'s own twinkle hash. Four distinct members of the
  family are live; check the literals before assuming a match. `variationFactor`'s `max(_, 0)` floor is
  load-bearing, not defensive — the factor multiplies a color, so a negative value would flip the RGB
  sign and, once blended, produce colors not in the authored palette at all.

---

## Distance fields & SDF

*In this category when a shape's coverage comes from a signed distance.*

### sdf.strokeMaskFromSdf
- **Import:** `import {sdf} from "@coreroot/gpu/kit"`
- **Signature:** body fn `(dist: f32, softness: f32, strokeThickness: f32, strokePosition: f32) → sdf.ShapeMask` — a struct `{overallMask: f32, strokeBlend: f32}`
- **Use when:** turning any SDF into fill coverage plus a stroke blend factor. **Do NOT hand-roll:**
  the pair of `smoothstep`s over stroke inner/outer boundaries, or the `strokePosition`
  outside/center/inside select ladder.
- **Used by:** `Heart`, `Circle`, `Ring` (15 shapes total).
- **Example:**
  ```ts
  const dist = sdf.heartSdf(local.x, local.y, radius)
  return sdf.strokeMaskFromSdf(dist, softness, strokeThickness, strokePosition)
  // fragment then mixes fill → stroke color by mask.strokeBlend, scaled by mask.overallMask
  ```
- **Traps:** Boundaries are in SDF space (surface at 0), so it works for any primitive — Circle
  feeds `distance - circleEdge`. `strokePosition` is a float enum compared with `>= 0.5` / `>= 1.5`
  (0 = outside, 1 = center, 2 = inside); pass the transformed numeric value, not a string.
  `strokeBlend` is exactly 0 when `strokeThickness <= 0`.

### sdf.fillStrokeColor
- **Import:** `import {sdf} from "@coreroot/gpu/kit"`
- **Signature:** Expr builder `(mask: Expr /* ShapeMask */, fill: Expr, stroke: Expr, colorSpaceMode: number /* compile-time */) → Expr /* vec4f */`
- **Use when:** you have a `ShapeMask` from `sdf.strokeMaskFromSdf` and need the finished
  premultiplied color. **Do NOT hand-roll:** the `mixColorsVariants[mode] ?? mixColorsLinear` lookup
  plus the `vec4(blended.rgb, blended.a * mask.overallMask)` tail — it was copy-pasted in all 15
  shapes before Phase 4.
- **Used by:** every 2D shape, via `scaffolds/sdfShape.ts`. Call it directly only from a custom
  `fragment` — a shape that needs an extra term before returning.
- **Example:**
  ```ts
  const mask = call(fooShape, 'fooShape', [...])
  return sdf.fillStrokeColor(mask, uniforms.color, uniforms.strokeColor, colorSpaceMode)
  ```
- **Traps:** `colorSpaceMode` is a **compile-time JS number** read off `propValues` (the `colorSpace`
  prop is `compileTime: true`), not a uniform — the variant is chosen in JS so only that color
  space's math is emitted. This is a builder: never call it from inside a `'use gpu'` body.

---

## Noise & procedural fields

*In this category when the shader's core value comes from a noise field.*

*`kit/noise.ts` holds the field functions themselves — `perlin12d`, `stone12`, `wool12`, the `hash*`
family — grep it before writing a field by hand. New shaders use the bit-exact `noise.hash*` family,
not sin-fract hashes. `kit/fields.ts` (below) holds the domain framings and the fbm/warp builders that
wrap those fields, and `kit/cellular.ts` the Voronoi/Worley fold.*

**The generator recipe.** A new noise generator is a prop schema plus one field function:
`fields.aspectScaledDomain` (or `pixelGridDomain`) for the sampling position → optionally
`fields.timeAxisDomain` for a morph clock → your `noise.*` field → `tone.toneRemap({…})` for the
contrast/balance/invert tail → `noiseColor.mixStopsOrColorsExpr(params, t)` for the color. Take the
prop block from `noiseColorProps()` so those names line up. `PerlinNoise` and `SimplexNoise` are the
exemplars; `WorleyNoise` is the same pipeline with a cellular field in the middle.

### fields.aspectScaledDomain / pixelGridDomain / timeAxisDomain
- **Import:** `import {fields} from "@coreroot/gpu/kit"`
- **Signature:** body fns. `aspectScaledDomain(uv: vec2f, viewport: vec2f, scale: f32, seed: f32) → vec2f`;
  `pixelGridDomain(uv: vec2f, viewport: vec2f, grain: f32, seed: f32) → vec2f`;
  `timeAxisDomain(pos: vec2f, t: f32, rate: f32) → vec3f`. All runtime.
- **Use when:** a noise/texture generator needs its sampling position (`aspectScaledDomain`), the pattern
  lives on the PIXEL grid rather than in UV space (`pixelGridDomain` — blue noise, ordered dither), or
  the noise should MORPH in place rather than slide (`timeAxisDomain`). **Do NOT hand-roll:** the
  three-line `aspect = vp.x/vp.y` → `vec2f(uv.x*aspect, uv.y)` → `.mul(exp(scale)).add(seed)` preamble
  (verbatim in eight shaders), `floor(uv * viewport / grain) + seed`, or `vec3f(pos.x, pos.y, t * rate)`.
- **Used by:** `PerlinNoise`, `BlockNoise`, `SimplexNoise`, `CurlNoise`, `GaborNoise`, `WaveletNoise`,
  `ErosionNoise` (scaled); `BlueNoise` (pixel grid); `PerlinNoise`/`BlockNoise` (rate 0.3),
  `SimplexNoise` (0.5) for the time axis.
- **Example:** `const pos = fields.aspectScaledDomain(uv, viewport, scale, seed)`
- **Traps:** `scale` is EXPONENTIAL (`exp(scale)`), so the prop belongs on a small signed range (−2…5),
  not a cell count. The aspect divide is guarded here, and the shaders that adopted it each gained that
  fix. Pass `pixelGridDomain` the EFFECTIVE viewport (`effectiveViewportSize ?? ctx.viewportSize`) or a
  resize-fit box grains at the canvas's pixel scale instead of its own; its `floor` is what makes the
  result piecewise-constant per cell. Offsetting the 2D position by time instead of using
  `timeAxisDomain` translates the pattern across the screen — a different, usually worse-looking effect
  that is easy to confuse when reading a diff.

### fields.fbmGated (builder)
- **Import:** `import {fields} from "@coreroot/gpu/kit"`
- **Signature:** `fbmGated(fieldFn, {maxOctaves, drift, stride?, name?}) → tgpu.fn(…) → vec2f(acc, totalWeight)`.
  `maxOctaves` and `drift` are STRUCTURAL (baked into the WGSL); the octave COUNT is a runtime argument.
- **Use when:** summing a field over octaves with a user-facing octave count. **Do NOT hand-roll:** the
  fixed-N loop with a `select(0, 1, i < n)` gate and the amp/freq accumulators.
- **Used by:** `FractalNoise` (`drift: 'goldenAngle'`, 8), `WorleyNoise` (`drift: 'timeSeed'`, 4).
- **Example:**
  ```ts
  const sum = fields.fbmGated(noise.mxNoiseFloat2, {maxOctaves: 8, drift: 'goldenAngle', name: 'mySum'})
  // inside a body:
  const s = sum(d.vec2f(u, v), 2.0, detail, contrast, animTime, d.vec2f(seed, seed * 0.7), octaveCount)
  const value = s.x / s.y
  ```
- **Traps:** **It returns `(accumulated, totalWeight)`, NOT a normalized value** — normalization stays
  with the caller on purpose, because the two consumers disagree about it (one divides raw and remaps to
  [0,1], the other guards the divide and applies a per-mode scale). The captured `fieldFn` is emitted
  under the BUILDER's local identifier (`octaveField`), so a migration diff shows the noise fn
  apparently renamed. Accumulators must be `d.f32(…)`-initialized. A per-octave stride table is
  deliberately absent, which is why `Marble`'s 3-octave sum (different seed multiplier AND time rate per
  octave) stays hand-unrolled.

### fields.domainWarp2 (builder)
- **Import:** `import {fields} from "@coreroot/gpu/kit"`
- **Signature:** `domainWarp2(field3, {offsets?, name?}) → tgpu.fn(pos: vec3f, t, warpAmount) → vec3f`
- **Use when:** you want flowing/swirling noise rather than smooth noise. **Do NOT hand-roll:** the four
  decorrelated noise taps and two displacement levels.
- **Used by:** `Plasma`.
- **Traps:** returns the POSITION to sample, not a value. Both levels displace the ORIGINAL position;
  chaining them instead compounds into mush at high warp.

### cellular.cellHash2 / nearest2Cells (builder)
- **Import:** `import {cellular} from "@coreroot/gpu/kit"`
- **Signature:** `cellHash2(p: vec2f) → vec2f` in [0,1]² (body fn);
  `nearest2Cells({distance, jitter, name?}) → tgpu.fn(scaledUV, animT, seed[, jitter, metric]) → vec2f(d1, d2)`.
- **Use when:** any cellular / Worley / Voronoi field. **Do NOT hand-roll:** the 3×3 neighbour loop, the
  drifting feature points, the two-nearest fold, or the `fract(sin(dot…) · 43758.5453)` pair.
- **Used by:** `Voronoi` (`{distance: 'length', jitter: 'none'}`), `WorleyNoise`
  (`{distance: 'selectableSquared', jitter: 'mix'}`).
- **Traps:** `selectableSquared` returns SQUARED Euclidean distances — the sqrt is deferred to
  `cellReduceSelectable`. The fold is branchless on purpose (`min`/`max`, with d2 reading the pre-update
  d1); it is exactly equivalent to the nested-select spelling and both only SELECT existing values, so
  the unification was bit-neutral — do not "optimize" it into a conditional. All three metrics are
  always computed, which is cheaper than recompiling on a compile-time metric prop. `cellHash2` is
  GPU-only (driver `sin` ≠ JS `Math.sin` at these magnitudes) so it cannot be CPU-goldened, and D-6
  forbids "improving" it; new shaders prefer `noise.hash*`.

### cellular.cellReduceSelectable / cellRatio / cellEdgeMask
- **Import:** `import {cellular} from "@coreroot/gpu/kit"`
- **Signature:** body fns. `cellReduceSelectable(d1, d2, metric, mode) → f32` · `cellRatio(d1, d2, power) → f32`
  · `cellEdgeMask(d1, d2, softness) → f32`. Mode/metric enums are `cellular.CELL_MODES` / `CELL_METRICS`.
- **Use when:** turning a `(d1, d2)` pair into a scalar — `cellReduceSelectable` for the F1/F2 family,
  `cellRatio` for a size-invariant fill gradient, `cellEdgeMask` for boundary lines.
- **Used by:** `WorleyNoise` (reduce), `Voronoi` (ratio + edge).
- **Traps:** `cellReduceSelectable` expects SQUARED distances when `metric` is 0. `cellRatio` and
  `cellEdgeMask` each recompute `d1 + d2`, so a caller using both pays it twice (negligible, and it kept
  the two independent).

### tone.toneRemap (variant table)
- **Import:** `import {tone} from "@coreroot/gpu/kit"`
- **Signature:** `toneRemap({domain, contrastMode, invert, glowGamma?, balance?}) → tgpu.fn(v, [glow,] contrast, balance) → f32`.
  Every option is COMPILE-TIME — a CPU-side dispatcher, like `mixColors`.
- **Use when:** mapping a raw field value to the [0,1] parameter a gradient is sampled at. **Do NOT
  hand-roll:** the contrast / balance / clamp / invert tail.
- **Used by:** every noise texture via `noiseColor.noiseToneKColor` (= `tone.toneUnitPivotInverted`),
  `SimplexNoise` (`toneSignedInverted`), `WorleyNoise` (`toneUnitMultiplicative`), `Plasma`
  (`toneGlowInverted`).
- **Example:** `const myTone = tone.toneRemap({domain: 'signed', contrastMode: 'additive', invert: true})`
- **Traps:** **An unsupported option combination THROWS rather than falling back**, deliberately — a
  silent fallback would be a wrong-looking texture with no error. Only the four combinations the library
  actually uses resolve; adding a fifth tone shape means writing its body in `tone.ts`, not composing
  more flags. It is a variant TABLE and not one parameterized body because the shapes pivot differently
  (`(n − 0.5)·(c + 1) + 0.5 + b` vs `(n·2 − 1)·c + b` then `·0.5 + 0.5`), and a single body would have
  moved every consumer's pixels. `additive` makes contrast 0 the identity, `multiplicative` makes 1 the
  identity — a shader's `contrast` prop range must match.

### legacySinHash11
- **Import:** `import {legacySinHash11} from "@coreroot/gpu/porters"`
- **Signature:** body fn `(x: f32) → f32` — a float in, [−1, 1] out
- **Use when:** **never, in new code.** It exists only so `BarShift` and `ConcentricSpin` can share
  the one sin-fract hash they had each declared locally. **Do NOT hand-roll:** another copy of
  `(fract(sin(x · 12.9898) · 43758.5453) − 0.5) · 2`.
- **Used by:** `BarShift` (as `barHash`), `ConcentricSpin` (as `csHash`) — both re-export it under
  their old local names so existing tests and importers keep working.
- **Traps:** **Not** interchangeable with `noise.hash11`: the sequences differ, so substituting one
  changes every consumer's look. Per hash policy (PRIMITIVES.md D-6) existing shaders keep this and
  new shaders use `noise.hash*`. It is also **not** `cells.cellHash` (vec2 index, different literals) or
  `cells.rowSpeedHash` (127.1, uncentred) — several distinct members of the sin-fract family are live,
  plus `BrickPattern`'s seed-offset variant, `DotGrid`'s twinkle hash and `Truchet`'s. Check the literals
  before assuming two are the same sequence.

### noiseStylize.applyNoiseReliefExpr
- **Import:** `import {noiseStylize, noise} from "@coreroot/gpu/kit"`
- **Signature:** Expr builder `(params: GpuFragmentParams, heightFn: tgpu.fn([vec2f], f32), heightHint: string) → Expr`
- **Use when:** the whole shader is "sample my child through a noise surface distortion and modulate
  its brightness by a height field" — a carved/relief stylization. **Do NOT hand-roll:** the
  RTT convert → Perlin-gradient-displaced sample → unpremultiply → contrast-remapped brightness
  chain.
- **Used by:** `Stone`, `Wool` (both are ~26 lines total because of this).
- **Example:**
  ```ts
  props: reliefStylizeProps({intensity: 0.5, scale: 4, contrast: -0.5, distortion: 0.15}),
  fragment: (params: GpuFragmentParams): Expr =>
      noiseStylize.applyNoiseReliefExpr(params, noise.wool12, 'wool12')
  ```
- **Traps:** This is a **higher-order builder** (C3) — `heightFn` is passed at composition time
  because TGSL cannot take fn arguments. `heightHint` seeds the emitted WGSL name, so give each
  shader a distinct one. Pair it with `reliefStylizeProps` from `@coreroot/utilities/noiseStylize`
  for the matching prop block; the builder reads those prop names.

---

## Anti-aliasing & masks

*In this category when an edge needs to be soft by one pixel rather than one UV unit.*

*`kit/mask.ts` also exists (tested by `kit-mask.test.ts`) for layer masking — a different job from the
per-pixel edge filters here.*

### aa.footprint1 / footprint2
- **Import:** `import {aa} from "@coreroot/gpu/kit"` then `const {footprint1, footprint2} = aa`
- **Signature:** body fns. `footprint1(x: f32, softness: f32) → f32`; `footprint2(p: vec2f, softness: f32) → vec2f`.
  Both compute `max(max(|dpdx|, |dpdy|) + softness, 1e-5)`, `footprint2` per axis.
- **Use when:** you need the screen-space footprint of a pattern coordinate to feed a filter or a
  smoothstep. `max` of the two derivatives (not a length) covers the wider axis of an
  anisotropically-stretched pixel, so a pattern at a grazing angle blurs rather than aliases.
- **Used by:** `Stripes` (`footprint1`), `Checkerboard` (`footprint2`).
- **Traps:** Fragment stage only. **Take them on the PRE-`fract`, PRE-`abs` coordinate** — a derivative
  of a folded value spikes at every seam and draws a bright line down each cell boundary. Taking a
  derivative of an fn PARAMETER works and is what these do (each quad lane carries its own value,
  asserted in `kit-aa.test.ts`). `softness` is added BEFORE the `1e-5` floor; `Grid` floors first and
  then adds, which differs by at most the floor and is left inline with a DEVIATION comment.
  `Chevron`, `Spiral` and `DotGrid` use other forms (`fwidth` of a scalar, `length(fwidth(vec2))`) —
  those measure genuinely different things and were not converted.

### aa.quilezStepFilter / quilezLineFilterAxis / quilezCheckerFilter
- **Import:** `const {quilezStepFilter, quilezLineFilterAxis, quilezCheckerFilter} = aa`
- **Signature:** body fns. `quilezStepFilter(p: f32, w: f32, threshold: f32) → f32`, clamped to [0,1];
  `quilezLineFilterAxis(p: f32, w: f32, N: f32) → f32`, UNclamped, `N = 1 / lineFraction`;
  `quilezCheckerFilter(p: vec2f, w: vec2f) → f32`, clamped to [0,1].
- **Use when:** filtering a two-tone stripe (`quilezStepFilter` — exact average of
  `step(threshold, fract(p))` over the footprint, ON fraction `1 - threshold`), one axis of a line grid
  (`quilezLineFilterAxis` — returns that axis's LINE coverage: 1 inside a line, 0 in the gap), or a 2D
  checkerboard (`quilezCheckerFilter`).
- **Used by:** `Stripes` (whose `balance` prop IS the threshold), `Grid`, `Checkerboard`.
- **Example:** combine two `quilezLineFilterAxis` axes by inclusion–exclusion, `1 - (1 - iX)(1 - iY)` —
  "on a line if either axis is", with the product form avoiding double-counting where the two line
  families cross.
- **Traps:** **`quilezStepFilter` and `quilezLineFilterAxis` are the same integral in two
  parameterisations and neither should be rewritten in terms of the other** — the argument that reads
  naturally ("balance" vs "how many line widths fit in a cell") is what keeps each prop mapping legible.
  The kit test asserts they are exact complements at `threshold = 1/N`, which catches a sign error
  without merging them. The line filter is unclamped by design; the caller's combine bounds it, and it
  returns LINE coverage, not gap coverage. `quilezCheckerFilter` converges to a uniform 0.5 as cells
  shrink below a pixel — that mid-gray is the CORRECT answer (it is what a checkerboard averages to, and
  it is why a receding checkerboard fades to flat gray instead of into moiré); do not "fix" it with a
  sharpening term.

### aa.lineMaskFromField / bandMask / discCoverage
- **Import:** `const {lineMaskFromField, bandMask, discCoverage} = aa`
- **Signature:** body fns. `lineMaskFromField(field: f32, lineWidth: f32, lo: f32, hi: f32) → f32`;
  `bandMask(x: f32, lo: f32, hi: f32) → f32` = `smoothstep(lo, hi, x) * smoothstep(lo, hi, 1 - x)`;
  `discCoverage(dist: f32, radius: f32, footprint: f32) → f32`.
- **Use when:** ending any SDF-based grid (`lineMaskFromField`), drawing an inset band inside a unit cell
  (`bandMask` — 1 mid-cell, falling to 0 within `[lo, hi]` of both edges; multiply an X band by a Y band
  for an inset rectangle: a brick face, a woven thread's crossing), or an anti-aliased disc
  (`discCoverage`). **Do NOT hand-roll:**
  `(1 - smoothstep(lo, hi, field)) * step(0.0001, lineWidth)`, verbatim in four files, or the disc's
  ramp placement, which is the part people get wrong.
- **Used by:** `Truchet`, `HexGrid`, `IsometricCubes`, `TriangularGrid` (line mask); `Weave`,
  `BrickPattern` (2 sites each, band); `DotGrid` (disc).
- **Traps:** `lineMaskFromField` carries two details. (1) The smoothstep is INVERTED, not reversed —
  `smoothstep(hi, lo, x)` with `hi > lo` is undefined in WGSL, which requires `edge0 < edge1`. (2) The
  `step(1e-4, lineWidth)` factor is a **kill switch**, not a rounding guard: at `lineWidth = 0` the band
  collapses onto the field's zero set but does not vanish, so without it a "thickness 0" pattern still
  draws hairlines wherever the field is exactly zero. `bandMask`'s `x` must be in [0,1] (the `fract` of a
  scaled coordinate) and `lo`/`hi` are absolute positions in that unit space: a gap `g` with footprint
  `f` is `(g, g + f)`. `discCoverage`'s ramp sits INSIDE the radius (`[radius - footprint/2, radius]`,
  not straddling it), so the silhouette stays put and only the edge feathers — a softened disc is the
  same size as a crisp one, and the half footprint keeps small dots sharp where a full one would turn a
  3px dot into a blur.

---

## Color & tone

*In this category when mapping a scalar to color, or blending colors in a chosen color space.*

### blend.luminance
- **Import:** `import {blend} from "@coreroot/gpu/kit"`
- **Signature:** body fn `(rgb: vec3f) → f32`
- **Use when:** you are inside the BLEND-MODE machinery and need BT.709 brightness. **New shader code
  should call `tone.luma709Dot` (or `tone.luma709`) instead** — this is the internal twin that
  `blend.color` / `luminosity` / `hue` / `saturation` share, and it is only listed here so you recognise
  it and don't add a third copy of the weights. **Do NOT hand-roll:**
  `rgb.x * 0.2126 + rgb.y * 0.7152 + rgb.z * 0.0722`.
- **Used by:** internal blend modes only (`blend.color`, `blend.luminosity`, `blend.hue`,
  `blend.saturation`) — no shader calls it directly, and that is now the intended split rather than a
  gap to close.
- **Example:**
  ```ts
  const lum = blend.luminance(color.xyz)
  ```
- **Traps:** This is 709. Some shaders use BT.601 (`0.299, 0.587, 0.114`) instead — those are
  *deliberately* different (PRIMITIVES.md D-2), so if you are matching an existing shader's look,
  check which weights it uses rather than assuming. For a brand-new shader, 709 is the default.

### blend.unpremultiplyAlpha
- **Import:** `import {blend} from "@coreroot/gpu/kit"`
- **Signature:** body fn `(color: vec4f) → vec4f`
- **Use when:** `fragment` sampled an RTT (child render target) and feeds the result into color
  math or returns it. RTT output is premultiplied; the blend pipeline expects straight alpha.
  **Do NOT hand-roll:** `rgb / alpha` with a divide-by-zero guard.
- **Used by:** `Twirl`, `Bulge`, `Sharpness` (~60 shaders — this is the most-used primitive in the
  kit).
- **Example:**
  ```ts
  const texture = convertToTexture(childNode)
  const sampled = texture.sample(distortedUV)
  return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [sampled])
  ```
- **Traps:** No-op at alpha 1, so a bug here is invisible on opaque content and obvious on
  transparent — test with a partially transparent child. Note the call shape: from a `fragment`
  builder you wrap it in `call(...)` with the hint `'unpremultiplyAlpha'` (matching the existing
  snapshots), not a direct application.

### noiseColor.toneAndColorExpr
- **Import:** `import {noiseColor} from "@coreroot/gpu/kit"`
- **Signature:** Expr builder `(params: {uniforms, propValues}, noise01: Expr, endpoints?: EndpointAccessors) → Expr`
  (vec4 P3-linear)
- **Use when:** a generator produces a grayscale `[0,1]` scalar and needs the standard
  contrast/balance tone controls plus the two-color-or-multi-stop gradient. **Do NOT hand-roll:**
  the `clamp((n-0.5)*(contrast+1)+0.5+balance, 0, 1)` remap, the `1 - k` inversion, or the
  stops-vs-colorA/B branch.
- **Used by:** `PerlinNoise`, `BlockNoise`, `GaborNoise` (8 noise generators).
- **Example:**
  ```ts
  const noiseVal = call(perlinNoiseField, 'perlinNoiseField', [uv, viewport, uniforms.scale, uniforms.seed, t])
  return noiseColor.toneAndColorExpr(params, noiseVal)
  ```
- **Traps:** Reads prop names `contrast`, `balance`, `colorA`, `colorB`, `stops`-derived uniforms,
  and the compile-time `colorSpace` / `stopCount` off `propValues` — so your prop block must use
  those exact names (pair with `noiseColorProps`). `contrast`/`balance` are optional and treated as
  0 if absent (Scratches does this). The tone map **inverts**: the gradient runs colorA→colorB as
  the noise *rises*.

### noiseColor.mixStopsOrColorsExpr
- **Import:** `import {noiseColor} from "@coreroot/gpu/kit"`
- **Signature:** Expr builder `(params: {uniforms, propValues}, t: Expr, endpoints?: EndpointAccessors) → Expr`
  (vec4 P3-linear)
- **Use when:** you have your own tone mapping (or none) and just need "gradient color at `t`",
  handling both the multi-stop and legacy two-color paths. **Do NOT hand-roll:** the
  `stopCount > 1 ? mixColorStopsRuntime(...) : mixColorsVariants[mode](...)` branch.
- **Used by:** `FractalNoise`, `SimplexNoise`, `Voronoi`, `Plasma`, `WorleyNoise`, `RadialGradient`,
  `ConicGradient`, `DiamondGradient`.
- **Example:**
  ```ts
  const t = call(myToneMap, 'myTone', [fieldValue, uniforms.contrast])
  return noiseColor.mixStopsOrColorsExpr(params, t)
  ```
- **Traps:** `colorSpace` and `stopCount` are read from `propValues` (compile-time, part of the
  pipeline hash) — declare the `stops` prop with `colorStopsPropConfig()` so the recompile boundary
  is right. `t` is clamped internally. It reads `colorA`/`colorB` by default; a shader naming its
  endpoints differently (`insideColor`/`outsideColor`) passes an `EndpointAccessors` record instead of
  renaming its props. **The module name is a wart:** `noiseColor` reads as noise-specific but this is
  the fleet's general gradient color path (`kit/colorMixing.ts` or `kit/colorStops.ts` would be a
  better home) — a move is a rename with snapshot churn across the noise fleet, so it is queued for
  whichever phase next opens `colorMixing.ts`. `LinearGradient`'s variant stays local: it takes a UV
  rather than a `t` and folds in transparent-edge coverage.

---

## Sampling & texture ops

*In this category when reading a texture at a computed coordinate, or handling coordinates that fall
outside `[0,1]`.*

### edges.sampleRemappedExpr
- **Import:** `import {edges} from "@coreroot/gpu/kit"`
- **Signature:** Expr builder `(tex: KitTexture, distortedUV: Expr, edgeMode: number /*compile-time*/) → Expr`
- **Use when:** a geometric distortion's RTT `fragment` path samples the child at a remapped UV.
  This is the one-liner that picks the right reconstruction filter *and* applies edge handling.
  **Do NOT hand-roll:** `(uv) => tex.sample(uv)` plus a hand-written edge-mode switch.
- **Used by:** `Twirl`, `Bulge`, `Mirror` (17 distortions).
- **Example:**
  ```ts
  const texture = convertToTexture(childNode)
  const distortedUV = call(myUvFn, 'myUv', [ctx.uv, ctx.aspect, uniforms.intensity])
  return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [
      edges.sampleRemappedExpr(texture, distortedUV, edgeMode),
  ])
  ```
- **Traps:** `edgeMode` is a **compile-time** JS number (0 stretch, 1 transparent, 2 mirror,
  3 wrap) from `transformEdges` — read it off `propValues`, not `uniforms`. Uses 9-tap Catmull-Rom
  for stretch/transparent/wrap and plain bilinear for mirror (no mirror-repeat sampler exists).
  Deliberately blocky (Pixelate), already low-pass (DiffuseBlur), or many-tap (chromatic split)
  distortions should keep plain bilinear instead.

### edges.composeEdgeRemapExpr
- **Import:** `import {edges} from "@coreroot/gpu/kit"`
- **Signature:** Expr builder `(distortedUV: Expr, mask: Expr, edgeMode: number /*compile-time*/) → {uv: Expr, mask: Expr}`
- **Use when:** writing the `uvRemap` half of a distortion — the analytic fast path that composes
  coordinates without an RTT pass. **Do NOT hand-roll:** the transparent-mode coverage multiply or
  the mirror/wrap UV transform.
- **Used by:** `Twirl`, `Bulge`, `Mirror` (16 distortions).
- **Example:**
  ```ts
  uvRemap: ({uv, mask, propValues}) => {
      const distorted = call(myUvFn, 'myUv', [uv, aspect, uniforms.intensity])
      return edges.composeEdgeRemapExpr(distorted, mask, propValues.edges as number)
  }
  ```
- **Traps:** Must agree with the `fragment` path or the shader looks different depending on whether
  the analytic fast path was taken — this divergence is the single most common distortion bug.
  Pair it with `sampleRemappedExpr` (same `edgeMode`) and keep the UV function shared between both
  hooks. Body-level twins (`applyEdgeToUV`, `composeEdgeRemap`) exist for use *inside* a
  `'use gpu'` body; the `*Expr` forms are for builders.

### edges.applyEdgeHandlingExpr
- **Import:** `import {edges} from "@coreroot/gpu/kit"`
- **Signature:** Expr builder `(distortedUV: Expr, sample: (uv: Expr) => Expr, edgeMode: number) → Expr`
- **Use when:** you need edge handling on a sampled color but must supply your own sampler (a
  compute-fed texture, a multi-tap gather, a non-standard sampler). **Do NOT hand-roll:** the alpha
  cut for transparent mode.
- **Used by:** `SliceWipe`, `Shatter`, `DiffuseBlur`.
- **Example:**
  ```ts
  return edges.applyEdgeHandlingExpr(distortedUV, (uv) => texture.sample(uv), edgeMode)
  ```
- **Traps:** Prefer `sampleRemappedExpr` unless you specifically need a custom sampler — it gives
  you the reconstruction filter for free.

### sampling.sampleCatmullRomExpr
- **Import:** `import {sampling} from "@coreroot/gpu/kit"`
- **Signature:** Expr builder `(tex: KitTexture, uv: Expr, sampler?: 'linearClamp' | 'linearRepeat' | …, range?: 'positive' | 'unit') → Expr`
- **Use when:** a distortion or transform magnifies content and single-tap bilinear shows facets on
  hard edges (text, logos, clip boundaries). **Do NOT hand-roll:** a 9-tap bicubic weight table.
- **Used by:** `GlassTiles`, `Spherize`.
- **Example:**
  ```ts
  return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [
      sampling.sampleCatmullRomExpr(texture, distortedUV),
  ])
  ```
- **Traps:** 9 taps, not 1 — do not put it inside a loop or a per-tap chain. It hoists its UV /
  texture-size / weight locals once via `ctx.memo`, so calling it twice with different UVs is fine
  but calling it with the same UV twice still emits one block. For wrap-mode edges pass
  `'linearRepeat'` so taps tile instead of clamping across the seam.

### sampling.sampleCoverageExpr
- **Import:** `import {sampling} from "@coreroot/gpu/kit"`
- **Signature:** Expr builder `(tex: KitTexture, uv: Expr, inside: Expr, sampler?) → Expr` (f32 coverage 0–1)
- **Use when:** sampling a single-channel coverage atlas (glyphs) rastered denser than the screen.
  It measures texels-per-pixel with derivatives, area-filters (4 bilinear taps = a box over the
  pixel's footprint) when minified and switches to the 9-tap Catmull-Rom when magnified, reading mip
  level 0 explicitly and only where `inside` > 0.5. **Do NOT:** sample a supersampled raster with
  implicit-LOD `textureSample` on a mipped texture: the GPU drops ~log2(supersample) levels and the
  supersampling is thrown away.
- **Used by:** `Text`.
- **Traps:** the atlas must have no mips and a clear gutter of at least 3 texels around each raster
  (taps reach 2.5 texels past `uv`). Reads `.r`.

### sampling.bSplineTapSetup
- **Import:** `import {sampling} from "@coreroot/gpu/kit"`
- **Signature:** factory `(gridSize: number, names?: {fnName?: string, structName?: string}) → {Taps, setup}`
  where `setup(uv) → Taps{uvA, uvB, uvC, uvD, w}`. Memoised per `gridSize|fnName|structName`.
  The caller samples the four UVs itself and weights by `w.x/.y/.z/.w`.
- **Use when:** reconstructing a LOW-RESOLUTION field whose derivative is visible — hardware bilinear
  is only C0, so a displacement grid kinks at every cell border and a sharp edge dragged across it
  scallops. **Do NOT hand-roll:** the cubic weight algebra or the two-taps-per-axis collapse.
- **Used by:** `PixelThrow` (its flow grid).
- **Example:**
  ```ts
  const {setup: flowTaps} = sampling.bSplineTapSetup(GRID, {fnName: 'flowTaps', structName: 'FlowTaps'})
  const t = call(flowTaps, 'flowTaps', [uv])
  // sample the grid texture at t.uvA…t.uvD, weight by t.w.x…t.w.w
  ```
- **Traps:** **Both names land verbatim in the emitted WGSL** (C3), so a migrating shader must pass
  the names it already emitted or its snapshot moves — that is how PixelThrow stayed Gate A. Pick the
  right filter: B-spline **approximates and softens**, which is what a displacement field wants;
  for reconstructing IMAGE detail under magnification use `sampleCatmullRomExpr` below instead, which
  interpolates and keeps edges crisp.

### media.mediaSurface + the object-fit tables
- **Import:** `import {media} from "@coreroot/gpu/kit"`
- **Signature:** Expr builder
  `media.mediaSurface({texture, uv, viewport, fit, allowNone?, uvPostProcess?, decode}) → Expr`.
  Plus the tables `media.OBJECT_FIT_MODES` (`cover 0, contain 1, fill 2, scale-down 3, none → 2`) and
  `media.OBJECT_FIT_MODES_ALLOW_NONE` (same, but `none` is its own mode 4).
- **Use when:** sampling a media / external texture through an object-fit prop — this is the whole
  surface, not just the scale. **Do NOT hand-roll:** the ~13 verbatim lines of fit branch + `sampleUV` +
  sample + alpha cut that Image, Video and Webcam each transcribed.
- **Used by:** `ImageTexture`, `VideoTexture`, `WebcamTexture`.
- **Example:**
  ```ts
  return media.mediaSurface({
      texture: tex.kit, uv: params.uvContext ?? ctx.uv,
      viewport: params.effectiveViewportSize ?? ctx.viewportSize,
      fit: propValues.objectFit, decode: 'srgb-linear-alpha-cut',
  })
  ```
- **Traps:**
  - `fit` takes `propValues.objectFit` RAW — a number post-transform, a string otherwise; the builder
    maps both. **Keep the prop `compileTime` or the fit branch freezes at its first value.**
  - `uvPostProcess` runs BEFORE the sample and before the alpha cut, because the cut has to see the UV
    that was actually sampled. WebcamTexture's selfie mirror relies on that ordering.
  - `allowNone` selects a different TABLE, not just an extra mode: without it `none` coalesces to `fill`
    (legacy presets keep filling their box); with it `none` is mode 4 (natural pixel size). The mode
    numbers are baked into shipped presets, so the two tables must not be unified.
  - Pair it with `objectFitProp(default, description, {allowNone})` from `utilities/propConfigs` — the
    two `allowNone` flags must agree or the prop's transform and the builder's table disagree.
  - `decode` is required rather than defaulted, the same reasoning as `flipY` under D-1.

### media.scaleCover / scaleContain / scaleFill / scaleScaleDown / scaleNone
- **Import:** `import {media} from "@coreroot/gpu/kit"`
- **Signature:** body fns `(dims: vec2f, vp: vec2f) → vec2f` (the per-mode `uvScale`); `scaleFill` takes no args. Plus `media.sampleUV(uv, uvScale) → vec2f` and `media.alphaCutLinear(color, uv) → vec4f`.
- **Use when:** you need one piece of the fit pipeline on its own. For a whole media surface reach for
  `media.mediaSurface` above instead — it assembles exactly these. **Do NOT hand-roll:** the
  aspect/cover/contain arithmetic or the letterbox alpha cut.
- **Used by:** `ImageTexture`, `VideoTexture`, `WebcamTexture`.
- **Example:**
  ```ts
  let uvScale: Expr
  if (mode === 1) uvScale = call(media.scaleContain, 'scaleContain', [dims, vp])
  else if (mode === 2) uvScale = call(media.scaleFill, 'scaleFill', [])
  else uvScale = call(media.scaleCover, 'scaleCover', [dims, vp])
  const finalUV = call(media.sampleUV, 'mediaSampleUV', [baseUV, uvScale])
  return call(media.alphaCutLinear, 'mediaAlphaCut', [texture.sample(finalUV), finalUV])
  ```
- **Traps:** **No Y-flip.** Both `importExternalTexture` and `copyExternalImageToTexture` are
  top-left origin, so with the composer's screen-matching `ctx.uv` the natural orientation needs no
  `1 - y` — adding one is a bug. `dims` comes from `KitTexture.dimensions()` on the GPU, so you do
  not need a CPU aspect uniform. `alphaCutLinear` also does the sRGB→linear decode, so don't decode
  twice. The fit mode is compile-time (a JS branch), not a uniform.

---

## Blur & bloom

*In this category when the shader gathers many samples of its child — a motion blur, a Gaussian, a
glow, a halation. Two recipes: a FRAGMENT-path blur (a fixed unrolled tap chain, no compute) and a
COMPUTE-path blur (`with*BlurCompute` owns the whole pass, and `composeBlurredOverSharp` is the
fragment).*

### blur.gaussianTapWeights / unrolledTapGather
- **Import:** `import {blur} from "@coreroot/gpu/kit"`
- **Signature:** `gaussianTapWeights(count: number, sigma: number) → number[]` — CPU, build time,
  normalized, taps spread over `t ∈ [-1, 1]`. `unrolledTapGather({weights, tapCoord, sample, component?}) → Expr`
  — a composition-time builder (it takes functions, so it cannot be a `tgpu.fn`).
- **Use when:** a fragment-path multi-tap blur — gathering N weighted samples of an RTT along a path (a
  motion blur, an alpha-silhouette blur). **Do NOT hand-roll:** the
  `Array.from({length: N}, …) / exp() / normalize` block that three shaders each carried, or the
  `let total; for (…) total = total ? total.add(…)` fold.
- **Used by:** `AngularBlur`, `LinearBlur`, `ZoomBlur` (all `gaussianTapWeights(32, 0.8)` + a vec4
  gather); `DropShadow` (two passes, `component: 'a'` then `'r'`).
- **Example:**
  ```ts
  build: ({uniforms, texture, ctx}) => blur.unrolledTapGather({
      weights: BLUR_WEIGHTS,
      tapCoord: (i) => call(zoomBlurTapCoord, 'zoomBlurTapCoord', [uniforms.center, uniforms.intensity, ctx.uv, ctx.aspect, floatE(i)]),
      sample: (coord) => texture.sample(coord),
  })
  ```
- **Traps:** **UNROLLED is the point** — `textureSample` is illegal under non-uniform control flow, so a
  runtime tap loop is not available. The fold is left-associative in tap order because float addition is
  not associative; any regrouping is a different result and a moved snapshot. The result is
  premultiplied when the source is an RTT — let `defineRttFilter` add the unpremultiply tail. In
  `gaussianTapWeights`, σ² is rounded to 15 significant digits internally *on purpose* (`0.8 * 0.8` is
  `0.6400000000000001`, and the weights are emitted at full double precision), with a bit-exactness test
  guarding it; do not "simplify" that away.

### blur.withFixedBlurCompute / withVariableBlurCompute / withBloomCompute
- **Import:** `blur.withFixedBlurCompute`, `blur.withVariableBlurCompute`, `blur.withBloomCompute`
- **Signature:** `(params: GpuFragmentParams, config) → BlurComputeNode | null`. Each returns `null` for
  "no child" and "no device", so a shader's `compute` hook is a one-liner plus its own bypasses.
  Configs: fixed `{radius(), halfKernel?, outputKey?}`; variable
  `{buildFill, fillValues, source?, halfKernel?, outputKey?}`; bloom
  `{buildExtract, buildExtractMap?, mapInfo?, threshold(), radius(), outputKey?}`.
- **Use when:** the shader's compute pass IS a blur. **Do NOT hand-roll:** RTT-ing the child,
  `registerComputeTexture`, the `onResize` → `setInputDimensions` wiring, the late `bindInputs` bind of
  the child RTT (and of a map source), or the per-frame params write ordering — ~110 lines per consumer.
- **Used by:** `Blur` (fixed + variable), `ChannelBlur` (fixed), `ProgressiveBlur` / `TiltShift` /
  `ReflectivePlane` (variable), `Glow` / `FilmStock` (bloom).
- **Example:**
  ```ts
  compute: (params) => blur.withVariableBlurCompute(params, {
      halfKernel: 14,
      buildFill: buildTiltShiftFillGraph,
      fillValues: (dims) => ({...readGeometry(dims), maxRadius: intensityToRadius(getCpuValue('intensity') ?? 0)}),
  })
  ```
- **Traps:** `radius`/`threshold`/`fillValues` are read LIVE each frame (so mouse/auto drivers pull
  through) and `fillValues` receives the live canvas size — read `dims.width/height`, never close over
  the first frame's. On the map-driven path `window` is defined and the fill graph's layout must carry a
  `source` texture entry (the wrapper binds it late); on the static path the fill bind group is built
  eagerly, which is why fixed and variable are separate calls rather than one config with both.
  `outputKey` defaults to `'blurredTexture'` — `FilmStock` overrides it to `'halationTexture'` because
  its fragment reads that name.

### blur.composeBlurredOverSharp
- **Import:** `blur.composeBlurredOverSharp`
- **Signature:** Expr builder `(params, compose: (blurred, sharp) => Expr, options?: {blurredKey?, sharpKey?}) → Expr`
- **Use when:** the fragment of a compute-backed blur. **Do NOT hand-roll:** the
  `if (!blurred || !sharp) → sharp passthrough` fallback or the `unpremultiplyAlpha` tail.
- **Used by:** `Blur` / `ProgressiveBlur` (`vec4(blurred.rgb, sharp.a)`), `TiltShift` (mix by the
  focus-line amount), `ChannelBlur` (per-channel mix), `Glow` (bloom add).
- **Traps:** **The fallback is not optional politeness** — it is what makes the GPU-free resolve gates,
  and every `compute → null` bypass (like Glow's `size === 0`), produce valid WGSL. `compose` receives
  `(blurred, sharp)` and the sharp sample is created first, matching the pre-existing emission order.

### blur.bloomExtractSoftKnee / buildBloomExtractGraph / buildBloomExtractMapGraph
- **Import:** `blur.bloomExtractSoftKnee`, `blur.buildBloomExtractGraph`, `blur.buildBloomExtractMapGraph`
- **Signature:** the body fn is
  `(childTexture: texture2d<f32>, cx: u32, cy: u32, inputSize: vec2f, computeSize: vec2f, threshold: f32) → vec4f`.
  The builders are `(computeWidth, computeHeight, kernelName[, channel]) → {layout, kernel, Params}`,
  GPU-free.
- **Use when:** a shader blurs the BRIGHT pixels of its child — a glow, a halation, any bloom. **Do NOT
  hand-roll:** the 4×4 area average or the quadratic soft-knee threshold.
- **Used by:** `Glow` (both variants, via one-line delegates that keep its kernel names), `FilmStock`
  (halation).
- **Example:**
  ```ts
  export const buildGlowPrepassGraph = (cw: number, ch: number) =>
      blur.buildBloomExtractGraph(cw, ch, 'glowExtractAndFill')
  ```
- **Traps:** Pass the consumer's EXISTING kernel name so its WGSL snapshot doesn't move and its resolve
  test keeps something to import. The bright-buffer format (`rgba16float`) is fixed by the kit and is not
  a choice: it is the one format that is filterable, legal as write-only storage, AND holds HDR above
  1.0. The extract writes the bright buffer AND the radius map, so one kernel feeds both halves of the
  variable blur. **`screenWithAlphaExtension` deliberately does not exist:** Glow's compose is ADDITIVE
  and FilmStock's halation is a true SCREEN, and unifying them is a visible change to a shipping look,
  not a refactor.

### blur.mapSourceScalar / applyRemapWindow / MAP_SOURCE_DIM_FIELDS / REMAP_WINDOW_FIELDS / remapWindowValues
- **Import:** all from `blur.*`
- **Signature:** `mapSourceScalar(channel: BlurMapChannel) → TgpuFn<(sample: vec4f) => f32>` (memoized and
  `$name`d per channel, C3); `applyRemapWindow` is a
  `tgpu.fn(raw, inputMin, inputMax, outputMin, outputMax, curve) → f32`; the two `*_FIELDS` records spread
  into a params `d.struct`; `remapWindowValues(window)` spreads a live `GpuMapWindow` into the per-frame
  write.
- **Use when:** a COMPUTE kernel has to reproduce the fragment path's map resolve for a map-driven prop.
  **Do NOT hand-roll:** the four-way comptime channel ternary or the
  `inputRange / normalized / exponent / eased / mix` chain — five verbatim copies before this.
- **Used by:** `Blur`, `ProgressiveBlur`, `TiltShift`, `Glow`, `BokehBlur`.
- **Example:**
  ```ts
  const Params = d.struct({angle: d.f32, …, ...blur.MAP_SOURCE_DIM_FIELDS, ...blur.REMAP_WINDOW_FIELDS})
  const channelScalar = blur.mapSourceScalar(channel)          // graph-build time
  // …inside the kernel body:
  const remapped = blur.applyRemapWindow(channelScalar(sample), p.inputMin, p.inputMax, p.outputMin, p.outputMax, p.curve)
  // …per frame:
  fillValues: (dims, window) => ({…, inputWidth: dims.width, inputHeight: dims.height, ...blur.remapWindowValues(window!)})
  ```
- **Traps:** **The `*_FIELDS` SPREAD ORDER is the uniform struct's member order** — keep the dims before
  the window, which is what all five kernels already did. `applyRemapWindow` takes six flat scalars
  rather than a struct (against C1's "about five" guidance) specifically so no consumer's uniform layout
  has to change; see PRIMITIVES.md. The luminance channel is `dot(rgb, vec3(0.2126, …))`, deliberately
  NOT `tone.luma709`, which spells the same standard as a multiply-add chain.

### blur.aspectAwareComputeRes
- **Import:** `blur.aspectAwareComputeRes`
- **Signature:** `(canvasWidth: number, canvasHeight: number) → {width, height}` — CPU.
- **Use when:** sizing a compute grid whose KERNEL works in compute pixels. **Do NOT hand-roll:** the
  `LONG_EDGE` / aspect branch (Glow and FilmStock had it twice).
- **Used by:** `Glow`, `FilmStock` (through `withBloomCompute`).
- **Traps:** floors at 8 px and guards `height === 0` — a canvas can initialize at zero height.

**Not extracted, deliberately.** `buildBlurMapFillGraph` (a generic fill-kernel envelope) is closed as
won't-do: TGSL cannot take a JS closure as a kernel body argument, and a `tgpu.fn` radius argument would
receive the params struct BY VALUE, turning `(*p).angle` reads into a struct copy — a real codegen change
for four lines of boilerplate. `separableGaussianExpr` likewise: `DropShadow` uses `unrolledTapGather`
and keeps its own hand table (whose weights sum to 6.210 against a divisor of 6.214, so regenerating
them would lighten the shadow by ~0.06%). `BokehBlur`'s kernel is non-separable by design and only its
map-window copy moved.

---

## Lighting & materials

*In this category when shading a surface — a bevelled rim, a studio reflection, a fresnel edge — or
when building a light/glow GENERATOR (see `kit/lightfields.ts` first).*

### lightfields — the ten light primitives
- **Import:** `import {lightfields} from "@coreroot/gpu/kit"`
- **Signature:** body fns, all pure float. Falloffs: `radialGaussianFalloff(dist, sharpness) → f32` ·
  `anisotropicGaussianSpot(du, dv, sharpness) → f32` ·
  `radialGaussianGlow(uv: vec2f, aspect, lx, ly, sizeSq) → f32`. Angular:
  `angularSineLobes(angle, count, phase, sharpness) → f32` ·
  `angularCosineSpikes(angle, halfCount, power) → f32` ·
  `seamlessAngularField(delta: vec2f) → vec3f` as `(angle, angleWrapped, blend)` ·
  `chromaticRingBand(dist, center, spread, width) → vec3f`. Segment/beam:
  `pointToSegment(p: vec2f, a: vec2f, b: vec2f) → vec2f` as `(t, dist)` ·
  `taperedSegmentGlow(dist, t, thickA, thickB, softA, softB) → vec2f` as `(colorT, alpha)` ·
  `beamLocalFrame(uv: vec2f, aspect, anchorRaw: vec2f) → vec2f` as `(u, v)`.
- **Use when:** building a light/glow generator. **Do NOT hand-roll:** gaussian falloffs, ray lobes,
  chromatic ring triples, beam projections.
- **Used by:** `SunBurst` (`angularSineLobes`), `Godrays` (`seamlessAngularField`), `LensFlare`
  (`chromaticRingBand` ×2, `angularCosineSpikes` ×2, `radialGaussianFalloff`), `LightLeak`
  (`beamLocalFrame`, `anisotropicGaussianSpot` ×2), `StudioBackground` (`radialGaussianGlow`), `Beam`
  (`pointToSegment` + `taperedSegmentGlow`).
- **Example:**
  ```ts
  const seg = lightfields.pointToSegment(p, a, b)          // (t, dist)
  const glow = lightfields.taperedSegmentGlow(seg.y, seg.x, thickA, thickB, softA, softB)
  ```
- **Traps:** `sharpness` is an INVERSE SQUARED radius (larger = tighter). `seamlessAngularField`'s two
  branches must BOTH be evaluated unconditionally — an `if` puts the seam back. `beamLocalFrame` takes a
  TRANSFORMED position prop and recovers the authored y itself, so do not pre-flip it. **Watch the
  multiply GROUPING when adopting a falloff:** `-500·d·d` and `-(d·d·500)` are equal in exact arithmetic
  but round differently, so an adoption at a site spelled the other way is a sub-ULP pixel change rather
  than a provable no-op — `LensFlare`'s `coreSoft`/`vertFall` and `Aurora`'s core gaussian stay inline
  for exactly that reason.

### effects.bevel.bevelSin
- **Import:** `import {effects} from "@coreroot/gpu/kit"` → `const {bevelSin} = effects.bevel`
- **Signature:** body fn `(t: f32, shape: f32) → f32` — returns **sinθ** of the surface tilt
- **Use when:** giving a flat 2D shape a dimensional edge from its SDF. `t` is bevel progress
  (`clamp(-sdf / bevelWidth, 0, 1)`; 0 at the silhouette, 1 where the bevel meets the face).
  **Do NOT hand-roll:** a bevel tilt profile.
- **Used by:** `Chrome`, `LiquidMetal`, `BrushedMetal`, `CarbonFiber` (all four share this one
  profile — that's the point).
- **Example:**
  ```ts
  const t = std.clamp(-dist / bevelWidth, 0.0, 1.0)
  const sBev = bevelSin(t, bevelShape)
  const nz = -std.sqrt(std.max(1.0 - sBev * sBev, 0.0))
  ```
- **Traps:** Returns **sinθ**, not a normal and not a height — the in-plane magnitude of the normal.
  Reconstruct `nz` yourself as `-sqrt(1 - sinθ²)` (note the negative). `shape` blends round
  (`0`, exact quarter-round fillet) → machined (`1`, chamfer plateau with two tangent breaks); it is
  a continuous blend, so a runtime uniform is fine.

### sdf3d.resolveShapeFieldSampler
- **Import:** `import {sdf3d} from "@coreroot/gpu/kit"`
- **Signature:** `(params: GpuFragmentParams, options?: {patternMode?: 'none' | 'raw' | 'triplanar', gradSampler?: 'fast' | 'same', bakedGradients?: 'none' | 'volumetric' | 'all'}) → {sampler(uv) → Expr, gradSampler(uv) → Expr, volumetric: boolean, bakedGradients: boolean}`.
  An Expr-builder — call it from `fragment`. Everything is **compile-time**: which of the three paths
  runs is a JS decision, and switching shape type recompiles.
- **Use when:** the shader has the `shape` / `shapeSdfUrl` / `shapeType` prop trio and a
  `createVolumetricFieldComputeNode` compute hook. **Do NOT hand-roll:** the ~26-line routing block
  that walks compute-marched volumetric field texture → flat SVG SDF data texture → analytic baked
  2D SDF, which existed verbatim in 18 files.
- **Used by:** `Glass`, `LiquidMetal`, `Chrome`, `Frost`, `Heatmap` (17 shaders — the whole
  shape-effect family except `SmokeFill`, which still hand-routes).
- **Example:**
  ```ts
  const {sampler, gradSampler, volumetric, bakedGradients} = sdf3d.resolveShapeFieldSampler(params, {
      patternMode: 'raw',
  })
  const surf0 = sampler(uv)
  const surfX = gradSampler(uv.add(...))
  ```
- **Traps:** `patternMode` must MATCH the mode passed to `createVolumetricFieldComputeNode` — a
  disagreement is silent, and the surface pattern slides off the shape when the user switches shape
  type. `volumetric` is a build-time boolean the consumer passes into its bodies as a 0/1 f32 so one
  body serves both paths. `bakedGradients` means the center tap already carries gradients in `.g/.b`
  and the consumer must skip its own taps (`'volumetric'` = compute path only, ThinFilm; `'all'` =
  compute + flat-SVG, Glass). A shader with no `shapeSdfUrl`/`shapeType` props still works — both
  read `''` and fall through to analytic. The companion
  `sdf3d.createVolumetricFieldComputeNode(params, getShape, patternMode)` now satisfies
  `GpuComputeNode` directly, so the `compute` hook is one line with no adapter.

### lighting.volumetricNormal / fieldGradient / bevelledFlatNormal
- **Import:** `import {lighting} from "@coreroot/gpu/kit"`
- **Signature:** body fns. `volumetricNormal(surf0: vec4f, surfX: vec4f, surfY: vec4f, eps: f32, gradClamp: f32) → vec3f`;
  `fieldGradient(surf0, surfX, surfY, eps) → vec2f`; `bevelledFlatNormal(grad: vec2f, sinTilt: f32) → vec3f`.
  All runtime, all pure.
- **Use when:** building a surface normal for a shape effect. `volumetricNormal` on the 3D/volumetric
  path (forward-differences the marched depth in `.w`); `fieldGradient` for the in-plane SDF gradient
  from `.x`; `bevelledFlatNormal` on the flat path, tilting away from the face along `grad` by a
  `sinTilt` from `effects.bevel.bevelSin`. **Do NOT hand-roll:** the finite-difference + clamp +
  normalize block, or an unguarded direction built from a raw gradient.
- **Used by:** `LiquidMetal`, `BrushedMetal`, `CarbonFiber` (all three); `Chrome`, `Plastic`,
  `Crystal`, `Water` (normal + gradient); `Frost`, `Holographic` (gradient only).
- **Example:**
  ```ts
  const n = lighting.volumetricNormal(surf0, surfX, surfY, eps, 5.0)   // volumetric path
  const grad = lighting.fieldGradient(surf0, surfX, surfY, eps)        // flat path
  const nFlat = lighting.bevelledFlatNormal(grad, effects.bevel.bevelSin(t, bevelShape))
  ```
- **Traps:** **`gradClamp` is load-bearing.** A chord field is discontinuous where one lobe occludes
  another, and an unclamped slope there yields a near-sideways normal that reads as a bright seam.
  Fleet values are ±5 (LiquidMetal, Chrome, Water, BrushedMetal, CarbonFiber) and ±4 (Plastic,
  Crystal) — pass the one your shader already used. `fieldGradient`'s magnitude is ≈1 for a true 2D
  distance field but ≫1 across a chord discontinuity, so build directions from it with a guarded
  length. `bevelledFlatNormal` caps `sinTilt` at 0.9995 — at exactly 1 the normal lies in the plane
  and the reflected ray grazes to infinity. If a consumer needs the clamped deltas a SECOND time
  (Frost's `steepness`, Holographic's `steepnessVol`) `volumetricNormal` can't serve it, since it
  returns only the normalized vector — those two use `fieldGradient` instead.

### lighting.outsideShape / insideMask / patternCoords
- **Import:** `import {lighting} from "@coreroot/gpu/kit"`
- **Signature:** body fns. `outsideShape(sdf: f32, pxH: f32) → bool` (outside by more than two device
  pixels); `insideMask(sdf, sharpEdge, pxH, minPixels) → f32` = `clamp(−sdf / max(sharpEdge/32, pxH·minPixels), 0, 1)`;
  `patternCoords(sdfUV: vec2f, surf0: vec4f, volumetric: f32) → vec2f`.
- **Use when:** `outsideShape` is the early-exit at the top of a composite body; `insideMask` is the
  silhouette coverage alpha; `patternCoords` is what makes a surface pattern (molten relief, brush
  grain, weave, foil crinkle) stick to a 3D shape instead of sliding across it in screen space.
  **Do NOT hand-roll:** any of the three — `patternCoords` was verbatim in 8 files, and an unclamped
  coverage mask is the D-7 aliasing bug.
- **Used by:** `LiquidMetal`, `Chrome`, `Plastic`, `Holographic` (all three); `Crystal` (mask +
  pattern, its own early exit); `BrushedMetal`, `CarbonFiber`, `Water`, `Frost` (the first and third —
  they still use the unclamped mask, see the §9 follow-up item).
- **Example:**
  ```ts
  if (lighting.outsideShape(sdf, pxH)) return d.vec4f(0)
  const pat = lighting.patternCoords(sdfUV, surf0, volumetric)
  const alpha = lighting.insideMask(sdf, std.max(edgeSoftness * 0.5, 0.001), pxH, 1.5)
  ```
- **Traps:** **`insideMask`'s pixel floor is the whole point of the fn** — the old fleet form
  `clamp(−sdf/sharpEdge · 32, 0, 1)` has no floor, so `edgeSoftness → 0` collapses the transition
  below one pixel and the silhouette resolves as a raw aliased step. `·32` and `/0.03125` are exact
  (32 is a power of two), so the pixel floor is the ONLY behavioral difference from the old form:
  adopting it is Gate C, and it is the D-7 change applied to five shaders (see the plan's §9). Pass
  `minPixels = 1.5`, the house value from `glassComposite`. `patternCoords`'s `volumetric` is the
  0/1 f32 flag from `resolveShapeFieldSampler`, not a bool.

### lighting.perspectiveViewRay
- **Import:** `import {lighting} from "@coreroot/gpu/kit"`
- **Signature:** body fn `(uv: vec2f, aspect: f32, fov: f32) → vec3f` — normalized, pointing INTO
  the scene (`+z`).
- **Use when:** a reflective material needs a per-pixel view direction. **Do NOT hand-roll:** the
  aspect-corrected off-axis spread. A perspective ray is what lets a flat face sweep the environment
  across the canvas instead of reflecting one constant direction.
- **Used by:** `LiquidMetal`, `BrushedMetal`, `CarbonFiber` (`fov` 0.6), `Chrome` (0.55).
- **Example:**
  ```ts
  const view = lighting.perspectiveViewRay(uv, aspect, 0.6)
  const refl = std.reflect(view, normal)
  ```
- **Traps:** The `+z`-into-scene convention is shared with `wardAnisotropicSpecular`'s `view`
  argument — don't negate it between the two. `fov` is a look constant, not a user prop; pass the
  value your shader already used rather than picking one.

### lighting.orthonormalTangentFrame / wardAnisotropicSpecular
- **Import:** `import {lighting} from "@coreroot/gpu/kit"`
- **Signature:** body fns. `orthonormalTangentFrame(normal: vec3f, dir: vec3f) → TangentFrame{tangent, bitangent}`;
  `wardAnisotropicSpecular(WardSpecularInput{normal, tangent, light, view, alphaAlong, alphaAcross}) → f32`
  — a struct argument per C1, at **unit gain**.
- **Use when:** a material has a directional grain — the streak that runs ALONG a brushed axis or a
  carbon tow, which an isotropic Blinn–Phong lobe cannot produce. **Do NOT hand-roll:** the
  Gram–Schmidt projection of an anisotropy axis into the surface plane, or the Ward lobe.
- **Used by:** `BrushedMetal`, `CarbonFiber`.
- **Example:**
  ```ts
  const frame = lighting.orthonormalTangentFrame(normal, grainDir)
  const spec = lighting.wardAnisotropicSpecular(lighting.WardSpecularInput({
      normal, tangent: frame.tangent, light, view, alphaAlong, alphaAcross,
  })) * specGain
  ```
- **Traps:** Unit gain — multiply by the material's gain at the call site, don't expect the fn to
  carry it. `view` points INTO the scene (the `perspectiveViewRay` convention). Both structs are
  exported alongside the fns and their names are part of the emitted WGSL ABI.

### lighting.cosinePalette
- **Import:** `import {lighting} from "@coreroot/gpu/kit"`
- **Signature:** body fn `(t: f32, phaseR: f32, phaseG: f32, phaseB: f32) → vec3f` —
  `cos((t + phase)·2π)·0.5 + 0.5` per channel (the IQ cosine palette).
- **Use when:** a cheap continuous spectrum from a scalar — iridescence, thin-film interference, a
  holographic sweep. **Do NOT hand-roll:** three phase-shifted cosines.
- **Used by:** `Holographic`, `effects/thinFilm.ts` (via `thinFilmRainbow`).
- **Example:**
  ```ts
  const rainbow = lighting.cosinePalette(t, 0.0, 1 / 3, 2 / 3)
  ```
- **Traps:** **The phases are arguments, not baked thirds**, because the fleet's copies disagree in
  the 4th decimal — Holographic uses exact `1/3, 2/3`, thinFilm the truncated `0.3333, 0.6667`.
  Unifying them would move pixels in one of the two, so each caller passes its own current values
  (the plan's §9 unification item stays deliberately deferred).

**Not extracted, deliberately** (each has ≥2 nominal consumers but the pairs differ in exponent
handling or in what they fold into the surrounding expression, so sharing them would be a pixel
change dressed as a refactor): `fresnelRim` — `glassFresnelRim` derives the view angle from the field
slope while Crystal builds its rim from the surface normal and Plastic/Holographic fold theirs into
their sheen terms. `dispersionTaps` — LiquidMetal's chromatic split is two studio evaluations at
rotated reflection vectors, Crystal's is per-facet trace offsets; shared intent, no shared shape.
`blinnPhongFresnel`, `metalTintRamp`, `beerLambertTransmit`, `noiseNormalPerturb` — candidates once
the radiance model settles. `studioEnv` — the four studio environments are genuinely different models
(tonemapped HDR bank set vs 3 banks over a squared sky vs 9 samples along a grain vs 5 inlined
banks), and any shared accumulation reorders float adds on an HDR quantity; the tractable slice if
someone picks it up is `studioBank(p, center, radiusMul, brightness, keyRad, drift, envStr) → f32`,
one softbox, byte-identical in all four. Goo's metaball-fill normal and Hologram's `.r`-only read are
permanent outliers.

---

## Wipes & transitions

*In this category when a prop called `progress` drives content disappearing. Every wipe is the same
shader with a different **coverage coordinate**: a per-pixel scalar in [0,1] answering "when does this
pixel go away". Pick a coverage coord, optionally modify it, hand it to `revealMask`. Do not write the
smoothstep tail yourself.*

**There is deliberately no `defineWipeShader`.** The kit fns carry the whole win; a factory would
have absorbed only the child guard and one `call(...)` line, and it fights three things — the prop
schemas don't converge (`feather` in six shaders, `softness` in five, four different defaults), the
mask fns take 7–9 args in shader-specific orders, and each shader's exported `…Mask` fn is its own
CPU-golden test surface. Revisit once a `wipePropConfig` prop-config factory exists.

### reveal.revealMask
- **Import:** `import {reveal} from "@coreroot/gpu/kit"`
- **Signature:** body fn `(coord: f32, progress: f32, feather: f32, invert: f32) → f32` — all runtime
- **Use when:** you have a coverage coordinate and need the alpha multiplier. **Do NOT hand-roll:**
  the 5-line `select(coord, 1-coord, invert>0)` / `max(feather, 1e-4)` /
  `front = progress*(1+2f)-f` / `smoothstep(front-f, front+f, coord)` tail — it was copy-pasted
  verbatim in all 11 wipes.
- **Used by:** `LinearWipe` (the two-line exemplar), `BlockDissolve`, `RadialWipe`.
- **Example:**
  ```ts
  const t = reveal.directionalCoord(uv, aspect, angleDeg)
  return reveal.applyReveal(color, reveal.revealMask(t, progress, feather, invert))
  ```
- **Traps:** `invert` is the `transformBoolean` f32 convention (`> 0` = on), and it flips the
  **coordinate**, not the progress, so the feather stays on one side of the front. The ±feather front
  remap is load-bearing: without it a feathered wipe never fully clears at progress 0 or 1.

### reveal.applyReveal
- **Import:** `import {reveal} from "@coreroot/gpu/kit"`
- **Signature:** body fn `(color: vec4f, reveal: f32) → vec4f` — all runtime
- **Use when:** scaling a STRAIGHT-alpha child's alpha by a mask, RGB preserved (the pointwise
  alpha-mask convention — no RTT, no unpremultiply). **Do NOT hand-roll:**
  `vec4f(color.x, color.y, color.z, color.w * reveal)`.
- **Used by:** all 11 wipes.
- **Traps:** Pointwise only. An RTT filter that samples its child must unpremultiply first; this fn
  assumes straight alpha.

### reveal.directionalCoord
- **Import:** `import {reveal} from "@coreroot/gpu/kit"`
- **Signature:** body fn `(uv: vec2f, aspect: f32, angleDeg: f32) → f32` — all runtime
- **Use when:** coverage should sweep 0→1 along an angle. **Do NOT hand-roll:** the `dir = (cos, sin)`
  / centered aspect-corrected `p` / `dot` / `ext = 0.5*(aspect*|dir.x| + |dir.y|)` block (verbatim in
  four shaders).
- **Used by:** `LinearWipe`, `BarnDoors`, `VenetianBlinds`, `RandomBars`.
- **Traps:** The half-extent normalization is what makes a diagonal reach exactly 0 and 1 at the frame
  corners — dividing by `aspect` instead gives a wipe that finishes early or late off-axis.

### reveal.radialCornerNormCoord
- **Import:** `import {reveal} from "@coreroot/gpu/kit"`
- **Signature:** body fn `(uv: vec2f, aspect: f32, center: vec2f) → f32` — all runtime
- **Use when:** coverage should grow outward from a point. **Do NOT hand-roll:** the farthest-corner
  normalization (`dx = max(cx, aspect - cx)` …), copied verbatim in two shaders.
- **Used by:** `IrisWipe`, `RippleWipe`.
- **Traps:** `center` is center-scaled (`center.x * aspect`, the canonical D-1 convention) and
  y-flipped (`1 - center.y`, the `transformPosition` convention). Normalizing by the **farthest**
  corner rather than a fixed radius is what makes progress 1 clear the frame for an off-center origin.

### reveal.angularCoord
- **Import:** `import {reveal} from "@coreroot/gpu/kit"`
- **Signature:** body fn `(uv: vec2f, aspect: f32, center: vec2f, startDeg: f32, mode: f32) → f32` —
  all runtime, but `mode` normally carries a compile-time choice baked as a literal
- **Use when:** a clock-hand sweep. `mode` 0 = clockwise, 1 = counter-clockwise, 2 = both wedges
  (folded and doubled).
- **Used by:** `RadialWipe`.
- **Traps:** The mode literal is the **caller's** job — a string prop must never be given a
  `transform` (the PagePeel-`corner` trap: it would write the string into an f32 field). RadialWipe
  maps `direction` → `0|1|2` in JS and passes `floatE(mode)`, which is also what makes `direction` a
  structural recompile input. Aspect handling here is UV-scaled (`(uv.x - center.x) * aspect`), NOT
  center-scaled like `radialCornerNormCoord` — preserved from RadialWipe as written; unifying it is a
  pixel change at non-square aspects.

### reveal.cellGrid
- **Import:** `import {reveal} from "@coreroot/gpu/kit"`
- **Signature:** body fn `(uv: vec2f, aspect: f32, size: f32) → CellGrid` — a struct
  `{cell: vec2f, local: vec2f, grid: vec2f}`; all runtime
- **Use when:** dicing the frame into square-ish cells `size` wide (a fraction of frame width) and you
  need any of the integer cell index (`cell`), the in-cell position (`local`, `fract` so [0,1)), or
  the grid resolution (`grid`, to normalize a cell's position). **Do NOT hand-roll:**
  `gridX = 1/max(size, 0.001)`, `gridY = max(gridX/aspect, 1)`, the `floor`/`fract` pair (verbatim
  three lines in three shaders).
- **Used by:** `BlockDissolve` (hashes `cell`), `DiamondWipe` (L1 from `local`), `CheckerWipe`
  (parity + diagonal from `cell`/`grid`).
- **Example:**
  ```ts
  const g = reveal.cellGrid(uv, aspect, blockSize)
  const r = noise.hash12(g.cell)
  return reveal.applyReveal(color, reveal.revealMask(r, progress, softness, invert))
  ```
- **Traps:** The `max(…, 1)` row floor is deliberate — a very wide frame would otherwise get zero
  rows. `size` is a **width** fraction; cell height follows from aspect. Returning all three members
  is the C1 answer to three consumers each needing a different subset; take what you need.

### reveal.foldAboutCenter / tilePhase
- **Import:** `import {reveal} from "@coreroot/gpu/kit"`
- **Signature:** body fns `foldAboutCenter(t: f32) → f32`; `tilePhase(t: f32, count: f32) → f32` —
  runtime
- **Use when:** one sweeping front should become two opening outward from the middle
  (`foldAboutCenter`), or every tile of a coverage coordinate should cross the front in lockstep
  (`tilePhase`, i.e. blinds).
- **Used by:** `BarnDoors`, `VenetianBlinds`.
- **Traps:** Both are coordinate modifiers — apply them to the coverage coord BEFORE `revealMask`,
  never to the mask.

**Deliberately not extracted** (single-consumer modifiers, one or two lines each over a kit coord,
and each is the shader's distinguishing identity): RippleWipe's ring quantization, RandomBars'
`hash11(idx + 0.5)` and BlockDissolve's `hash12(cell)` (the hash IS the primitive — a wrapper adds
nothing), DiamondWipe's L1 from the cell center, CheckerWipe's parity-plus-diagonal, NoiseDissolve's
3-octave seeded fbm. Recorded so a future second consumer knows where to look.

---

## Scaffolds (definition factories)

*In this category when your shader is a standard member of an established category — the factory
should write most of the file.*

Four scaffolds have landed, covering RTT distortions, 2D SDF shapes, and the two shapes of color
filter. Every scaffold is imported from `@coreroot/gpu/porters`, the same facade as the rest of the
contract — shaders do not import `@coreroot/gpu/scaffolds/<file>` directly.

### uvRemapShader
- **Import:** `import {uvRemapShader} from "@coreroot/gpu/porters"`
- **Signature:** `(source: UvMap | ((params) => UvMap), opts?) → Pick<GpuShaderDefinition, 'fragment' | 'uvRemap'>`.
  Spread the result into the definition. `opts`: `edges?: 'prop' | 'none' | number` (default
  `'prop'`), `resample?: 'catmullRom' | 'bilinear'` (default `'catmullRom'`),
  `requireChildMessage?: string`, `uvRemapIdentityWhen?: (params) => boolean`. All four are
  compile-time/structural — nothing here is read on the GPU.
- **Use when:** the shader is an RTT distortion — it takes a child and returns it resampled at a
  displaced coordinate. **Do NOT hand-roll:** the `fragment` + `uvRemap` hook PAIR. Writing both by
  hand is how the two drift apart, which was the biggest structural liability in the fleet.
- **Used by:** `Twirl` (the minimal case), `CornerPin` (coverage), `GridDistortion` (displacement
  texture), `Mirror` (`selectMap`), `PolarCoordinates` (`lerpToIdentity`).
- **Example:**
  ```ts
  ...uvRemapShader(
      ({uniforms}) => ({
          map: (uv, aspect) => call(twirlUV, 'twirlUV', [uniforms.center, uniforms.intensity, uv, aspect]),
      }),
      {requireChildMessage: 'You must pass a child component into the Twirl shader.'},
  )
  ```
- **Traps:**
  - `map` must be pure — it may not sample the child; the scaffold owns sampling and edge handling.
  - The map receives `uv` as an **argument** rather than reading it off params, because the two hooks
    pass different coordinates (`ctx.uv` vs the folded incoming UV). Reading `ctx.uv` inside the map
    would silently break the analytic path.
  - Read `animatedTime(params)` inside the FACTORY, not inside `map` — once per hook, not once per
    use of the coordinate.
  - `Expr` has no CSE: `member()` re-emits its whole subtree per use site, so taking two members off
    one `call(...)` emits that call twice in the WGSL. Pre-existing behavior, not something the
    scaffold introduced, but it means "compute once, use twice" is not a thing at this level — use
    `asLocal` if the subtree is expensive.
  - `edges: 'none'` is only sound when the map provably stays inside [0,1]. `Flip` is the only
    legitimate user.
  - `uvRemapIdentityWhen` is uvRemap-only and that asymmetry is deliberate, not a bug: a shader whose
    displacement comes from its own compute texture must keep the fragment path structurally
    identical whether or not the compute output exists (otherwise the pipeline hash changes when
    compute comes online), while the analytic path's whole job is to disappear when it has nothing to
    contribute. GridDistortion and Liquify are the consumers.

### UvMap / UvMapResult
- **Import:** `import type {UvMap, UvMapResult} from "@coreroot/gpu/porters"`
- **Signature:** `interface UvMap {map: (uv: Expr, aspect: Expr) => Expr | {uv: Expr; coverage?: Expr}}`
- **Use when:** describing a distortion's coordinate function. Return the bare `Expr` unless the
  shader also produces coverage. **Do NOT hand-roll:** a second closure for coverage.
- **Used by:** every shader in `category: "Distortions"`, plus `SliceWipe`, `GridDistortion`,
  `Liquify`.
- **Traps:** `coverage` is a 0..1 scalar; the scaffold multiplies it into sampled ALPHA on the
  fragment path and into the coverage MASK on the analytic path. Those are the two spellings of the
  same quantity — do not try to apply it yourself in the map. The union return type (rather than a
  sibling `coverage` closure, which is what the plan sketched) exists because CornerPin's coordinate
  and its `front` gate are the `.xy` and `.z` of one packed `vec3f`, and two closures would emit that
  homography twice. Generally: at the Expr level, "one computation, two outputs" must be one call
  returning a struct, because there is no CSE to deduplicate two calls.

### selectMap
- **Import:** `import {selectMap} from "@coreroot/gpu/porters"`
- **Signature:** `(a: (uv, aspect) => Expr, b: (uv, aspect) => Expr, pick: (uv, aspect) => Expr) → UvMap`
- **Use when:** a distortion picks between two coordinate functions by a HARD 0/1 selector (a `step`,
  a `sign`-derived gate). **Do NOT hand-roll:** sampling the child twice and mixing the two COLORS.
  That costs an extra texture fetch and, worse, it is a form the analytic path cannot use, so you end
  up writing two different hooks.
- **Used by:** `Mirror`.
- **Example:**
  ```ts
  selectMap(
      (uv) => uv,                                        // near side: identity
      (uv, aspect) => reflect(uv, aspect).member('xy'),  // far side: reflected
      (uv, aspect) => reflect(uv, aspect).member('z'),   // step(0, signedDistance)
  )
  ```
- **Traps:** `pick` MUST be exactly 0 or 1. Mixing coordinates and mixing colors agree only for a
  hard selector: at a fractional `pick` the coordinate form samples a point BETWEEN the two lookups,
  which is not the average of the two colors. If your selector is smooth you cannot use this — and
  you probably cannot use the analytic path at all.

### lerpToIdentity
- **Import:** `import {lerpToIdentity} from "@coreroot/gpu/porters"`
- **Signature:** `(inner: UvMap, amount: (uv, aspect) => Expr) → UvMap`
- **Use when:** the distortion has an `intensity`/`amount` prop that fades the whole warp in and out.
  **Do NOT hand-roll:** `mixExpr(uv, mapped, uniforms.intensity)` in both hooks.
- **Used by:** `PolarCoordinates`, `RectangularCoordinates`.
- **Traps:** Blends COORDINATES, not colors — intermediate values are a genuine partial warp, not a
  cross-fade between the undistorted and distorted images. That is what both consumers already did,
  but know it before reaching for this to implement a dissolve.

### defineSdfShapeShader
- **Import:** `import {defineSdfShapeShader} from "@coreroot/gpu/porters"`
- **Signature:** `(spec) => GpuShaderDefinition<T>`. Spec: `name`, `description`, `category?`
  (default `'Shapes'`), `shapeFn` + `shapeFnName` + `bodyArgs` (the body and its argument order),
  `colorDescription`, `centerDescription`, `centerLabel?`, `shapeProps`, `rotatable?` (default
  `true`), `rotationDescription?`, `bounds`, `stroke?`, `colorSpaceDescription?`, `propOverrides?`,
  `propOrder?`. All prop metadata is CPU-side/compile-time; nothing here is a runtime uniform.
- **Use when:** the shader is a filled 2D shape with a stroke — an SDF, a softness, a stroke, a fill.
  **Do NOT hand-roll:** the `origin` / `color` / `center` / `rotation` / softness+stroke / `colorSpace`
  prop block (≈90 lines), the `x`/`y`/`width`/`height`/`rotation` bounding-box bindings, or the
  fragment (`uvContext ?? ctx.uv` → `call(body)` → `mixColorsVariants` lookup → `vec4`).
- **Used by:** `Star` (the pure template), `Circle` (`rotatable: false` + `propOverrides` +
  `propOrder`), `Trapezoid` (`bounds: {custom}` with `computeBounds`/`writeBounds`/`freeResize`).
- **Example:**
  ```ts
  export const starShape = tgpu.fn([d.vec2f, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.vec2f, d.f32], sdf.ShapeMask)(
      (center, radius, sides, innerRatio, rotation, softness, strokeThickness, strokePosition, uv, aspect) => {
          'use gpu'
          const local = sdf.shapeLocalCoords(center, rotation, uv, aspect)
          const dist = sdf.starSdf(local.x, local.y, radius, sides, innerRatio)
          return sdf.strokeMaskFromSdf(dist, softness, strokeThickness, strokePosition)
      },
  )

  export const componentDefinition = defineSdfShapeShader<ComponentProps>({
      name: "Star",
      description: "Classic star polygon with straight sides and sharp pointed tips",
      shapeFn: starShape, shapeFnName: 'starShape',
      bodyArgs: ['radius', 'sides', 'innerRatio', 'rotation'],
      colorDescription: "Fill color of the star",
      centerDescription: "Center position of the star",
      bounds: {size: {width: 'radius', height: 'radius'}},
      shapeProps: {/* radius, sides, innerRatio */},
  })
  ```
- **Traps:**
  - **There is no `sdf:` callback and there will not be one for migrated shapes.** The body stays in
    the shader file because a TGSL body is transpiled from its literal source text and the emitted
    WGSL carries the source's identifiers and arity — a generated body could not match the fleet's.
    So a new shape copies the six-line body template (it's in the scaffold's file header) and writes
    ~20 lines of spec.
  - **The body's argument ORDER is a contract:** `center`, the `bodyArgs` (with `rotation` last among
    them for a rotatable shape), then `softness`, `strokeThickness`, `strokePosition`, `uv`, `aspect`.
    `bodyArgs` is a separate list from `shapeProps` because the orders can differ (Trapezoid declares
    `bottomWidth, topWidth, height`; its body takes `topWidth, bottomWidth, height`).
  - **`shapeFnName` must equal the fn's emitted WGSL name** — it is the `call` hint and it appears in
    the snapshots.
  - **Prop key order is load-bearing.** `createGpuUniformsMap` walks `props` to lay out the node's
    uniform struct, so reordering props reorders WGSL struct members; it also drives the settings
    panel order via the generated `shaderMetadata.ts`. The factory emits the fleet-canonical order;
    use `propOrder` only if a shader's existing order differs (only Circle's does).
  - **`bounds.size.as` defaults to `'half-canvas-height'`** because the fleet's size props are
    half-extents. Circle passes `'canvas-height'`: its `radius` names the full visual size and the
    shader halves it internally. Getting this wrong makes the overlay box half or double the shape.
  - **`rotatable: false` only changes what the FACTORY emits** (no `rotation` prop, no rotation box
    axis). Whether the body calls `shapeLocalCoords` or inlines the delta is the body's own business.
  - The generic parameter is documentation, not a check: props are assembled dynamically and cast, so
    a mismatch between `ComponentProps` and the actual props is not caught by the compiler. The
    prop-order test in `scaffold-sdfShape.test.ts` is the net.
  - Not for `Line`, `SineWave`, `Spiral`, `Blob` — structurally different, permanently out of scope.

### definePointwiseFilter
- **Import:** `import {definePointwiseFilter} from "@coreroot/gpu/porters"`
- **Signature:** `(config) => GpuShaderDefinition<T>`. Config = the whole definition minus
  `fragment`/`requiresChild`, plus: `body` (`{fn, hint}`, or `(propValues) => {fn, hint}` for a
  compile-time body pair), `args?` (`(params) => Expr[]` — everything after the child color),
  `compose?` (`(bodyResult, params) => Expr`, builder-level work on the result), `identity?`
  (`{props, when}`), `missingChildMessage?` (omit = fail silently), `setup?` (side effects, e.g.
  registering an `onBeforeRender`).
- **Use when:** the filter transforms the composed child color **per pixel** with no neighbour reads.
  **Do NOT hand-roll:** `requiresChild`, the `if (!childNode) {console.error(…); return ZERO}` guard,
  the `call(body, hint, [childNode, …])` wiring, or the `if (identityValue) return childNode` bypass.
- **Used by:** `Invert` (the minimal case: body only), `Vignette` (reads `ctx.uv`/`ctx.aspect` via
  `args`), `Tint` (compile-time body pair), `Duotone` / `GradientMap` (`compose` does color-space
  mixing / a palette branch the body cannot).
- **Example:**
  ```ts
  export const componentDefinition = definePointwiseFilter<ComponentProps>({
      name: "Exposure", category: "Adjustments", description: "…",
      props: {exposure: {default: 1, compileTimeWhen: (p, n) => (p === 1) !== (n === 1), ui: {…}}},
      body: {fn: exposureColor, hint: 'exposureColor'},
      args: ({uniforms}) => [uniforms.exposure],
      identity: {props: ['exposure'], when: (p) => (p.exposure as number) === 1},
      missingChildMessage: 'You must pass a child component into the Exposure shader.',
  })
  ```
- **Traps:**
  - `hint` lands in the emitted WGSL and therefore in the snapshot — keep it equal to the exported
    fn's variable name.
  - The child color is **always** the body's first argument; `args` supplies the rest.
  - Declaring `identity` is not enough: the prop also needs `compileTimeWhen` (or `compileTime`) so a
    value crossing the boundary recomposes. Without it the bypass sticks.
  - `identity.props` must list every prop `when` reads — the scaffold refuses the bypass when any of
    them carries a **map** driver. Mouse/auto drivers are still undetectable here; see the identity
    caveat at the end of this section.
  - `setup` does not run on the identity path.
  - Whether the missing-child guard logs is per-shader and user-facing; do not "normalize" it.
  - A bypass must be **provable**. `Tint` deliberately has none: at `amount === 0` the plain body is
    identity, but its DEFAULT body (`preserveLuminosity`) rescales by `originalLum / max(tintedLum,
    1e-4)`, which is exactly 1 only once luminance ≥ 1e-4 and darkens below it.

### defineRttFilter
- **Import:** `import {defineRttFilter} from "@coreroot/gpu/porters"`
- **Signature:** `(config) => GpuShaderDefinition<T>`. Config = the definition minus
  `fragment`/`requiresRTT`/`requiresChild`, plus: `build` (`(params) => Expr`, where `params` adds
  `texture` and `sampleStraight(uv)`), `resultAlpha?` (`'premultiplied'` default | `'straight'`),
  `identity?`, `missingChildMessage?`, `setup?`.
- **Use when:** the filter needs neighbour samples or any non-local read of the child (a convolution
  kernel, a dither tile, a halftone screen). **Do NOT hand-roll:** `convertToTexture(childNode)`, the
  `blend.unpremultiplyAlpha` tail, or the identity bypass.
- **Used by:** `Sharpness` (5-tap kernel). Designed for `Dither` / `Engraving` / `FilmStock`, which
  are deferred and still hand-roll the recipe.
- **Example:**
  ```ts
  export const componentDefinition = defineRttFilter<ComponentProps>({
      name: "Sharpness", props: {sharpness: {default: 0, compileTimeWhen: …, ui: {…}}},
      identity: {props: ['sharpness'], when: (p) => (p.sharpness as number) === 0},
      build: ({texture, ctx, uniforms}) => call(sharpnessCompose, 'sharpnessCompose', [
          texture.sample(ctx.uv), /* …4 tap samples… */ uniforms.sharpness]),
  })
  ```
- **Traps:**
  - **The child RTT is PREMULTIPLIED.** Sample `texture` directly for linear-in-RGB work (a
    convolution) and let the scaffold unpremultiply once at the end; use `sampleStraight(uv)` only
    when the math genuinely needs straight color (a luminance threshold, a quantiser), and then set
    `resultAlpha: 'straight'` so no second unpremultiply is applied.
  - The identity bypass **cannot** return `childNode` — that is the pre-RTT composed Expr and would
    double-composite the subtree at the wrong alpha. The scaffold samples the center texel and
    unpremultiplies. This is the single most valuable thing the scaffold encodes.
  - Identity still pays for the RTT pass; the saving is the kernel, not the boundary.

**Identity-bypass caveat, both filter scaffolds.** `propValues` is deliberately driver-unaware, so a
compile-time bypass keyed on a prop's base value **kills any mouse/auto driver on that prop** — a
Vignette with `intensity` mouse-driven from a base of 0 would never appear. The scaffolds close the
worst case (a per-pixel **map** driver) by refusing the bypass when `getMapInfo(prop)` is non-null,
but mouse/auto drivers cannot be detected from a fragment builder at all. Fixing that needs a
`GpuFragmentParams` addition and is tracked as a Phase 12 contract item.

---

## Simulation harnesses

*In this category when the shader carries state between frames — particles, fluids, feedback trails.*

### createStateBuffer
- **Import:** `import {createStateBuffer} from "@coreroot/gpu/porters"`
- **Signature:** `(root: TgpuRoot, schema: AnyWgslData, count: number) → TgpuBuffer<WgslArray> & StorageFlag`
- **Use when:** a simulation needs per-agent or per-cell state persisting across frames.
  **Do NOT hand-roll:** `root.createBuffer(d.arrayOf(...)).$usage('storage')` with its cast dance.
- **Used by:** `Boids`, `Particles`, `CursorRipples` (19 shaders).
- **Example:**
  ```ts
  const state = createStateBuffer(root, d.vec4f, particleCount)
  ```
- **Traps:** Bind to a `{storage, access: 'mutable'}` layout entry. Initialize (and read back) via
  `.write` / `.read`. The same module's `createPingPong` is **`@deprecated`** and has never had a
  consumer — it can express only one bind-group family, exposes no orientation query, and builds its
  groups eagerly. Use `createPingPongPair` below.

### createPingPongPair
- **Import:** `import {createPingPongPair} from "@coreroot/gpu/porters"` (it lives in `gpu/compute.ts`
  and is also re-exported from `scaffolds/feedbackSim`)
- **Signature:** `(a: TSlot, b: TSlot) → PingPongPair<TSlot>` with `groups(make(read, write)) → () => T`,
  `perSide(make(side)) → {read(), written()}`, `readSource()`, `writeTarget()`, `orientation()`,
  `swap()`, `reset()`. `TSlot` is one SIDE of the pair — a buffer, a texture, or a RECORD of
  resources that must swap together. All composition-time; nothing here is a uniform.
- **Use when:** any sim that reads last frame's state and writes this frame's. **Do NOT hand-roll:** a
  `let direction = 0` flag with `bgAB`/`bgBA` pairs and `direction = 1 - direction` at the end of
  `getComputeNodes` — that pattern was in six shaders.
- **Used by:** `CursorRipples` (buffers + a gradient pass on the written side), `ReactionDiffusion`
  (N swaps per frame, output copy on the read side), `Surface3D`, and every `createFeedbackTrailSim`
  consumer (textures).
- **Example:**
  ```ts
  const field = createPingPongPair(bufferA, bufferB)
  const propGroup = field.groups((readBuf, writeBuf) =>
      root.createBindGroup(propagateLayout, {readBuf, writeBuf, params: propParams.buffer}))
  const gradGroup = field.perSide((srcBuf) => root.createBindGroup(gradientLayout, {srcBuf, dispTex}))
  const nodes = [propagate.with(propGroup()), gradient.with(gradGroup.written())]
  field.swap()
  ```
- **Traps:** `groups()` may be called LATE (inside `bindInputs`) — that is the whole reason it is a
  method rather than a constructor argument. **`written()` vs `read()` is a real distinction:** an
  output/display copy that runs BEFORE the swap wants the side just written, while
  ReactionDiffusion's output runs AFTER its N swaps and wants `read()`. Getting it backwards costs
  one frame of latency, which is easy to miss and hard to spot later. A sim that re-seeds must
  `reset()`, because the seed kernel writes one fixed side. Because `TSlot` can be a record, lockstep
  slots (TimeTrail's prev-live pair) fall out for free — pass an object, not two pairs.

### createLateBoundChildInput
- **Import:** `import {createLateBoundChildInput} from "@coreroot/gpu/porters"`
- **Signature:** `(params: GpuFragmentParams, buildGroups: (childTexture) => TGroups) → {childTexture, bindInputs, bound(): TGroups | null, ready()} | null`.
  Returns `null` when the node has no child.
- **Use when:** a COMPUTE pass samples the child layer. **Do NOT hand-roll:** the `convertToTexture` +
  `resolve(key)` + `if (!src) return` + `as never` + `if (!bg) return null` block, which existed five
  times. Child-RTT compute inputs bind LATE, after the child target exists — that is what this owns.
- **Used by:** `ReactionDiffusion` (directly); `TimeTrail`, `DataMosh`, `KeyFrames` (through
  `createFeedbackTrailSim`).
- **Example:**
  ```ts
  const child = createLateBoundChildInput(params, (src) => ({
      bg: root.createBindGroup(layout, {src, prev: stateA, next: stateB, params: buf.buffer}),
  }))
  if (!child) return null
  return {
      outputs: {childTexture: child.childTexture, display},
      bindInputs: child.bindInputs,
      getComputeNodes: () => {
          const g = child.bound()
          if (!g) return null           // forward the null — do not dispatch unbound
          return [pipeline.with(g.bg)]
      },
  }
  ```
- **Traps:** `buildGroups` re-runs on every recompose, so it must be safe to rebuild. Put
  `childTexture` in `outputs` so the fragment samples the SAME RTT instead of registering a second
  one. The library's single audited `as never` lives inside this helper; the real fix is the
  contract's `bindInputs` typing (a Phase 12 item).

### createFeedbackTrailSim
- **Import:** `import {createFeedbackTrailSim} from "@coreroot/gpu/porters"`
- **Signature:** `(params, {size, format, extraSlots?, textures?, speedProp?, maxDeltaTime?, bindGroups, staticGroups?}) → {childTexture, display, pair, bindInputs, tick} | null`.
  Every option is structural. `tick(frameParams, frame)` IS the body of `getComputeNodes`.
- **Use when:** the shader keeps a fixed-resolution state texture that carries across frames plus a
  display copy the fragment samples. **Do NOT hand-roll:** the three texture allocations and their
  `onCleanup`, `registerComputeTexture`, the ping-pong flag, the null-until-bound guard, or
  `Math.min(deltaTime, 0.05) * speed`.
- **Used by:** `DataMosh` (the minimal shape), `TimeTrail` (adds a lockstep prev-live slot),
  `KeyFrames` (row-shaped state + a static feature-grid group).
- **Example:**
  ```ts
  const sim = createFeedbackTrailSim(params, {
      size: STATE_RES, format: 'rgba16float', speedProp: 'speed',
      bindGroups: ({childTexture, display}, read, write) => root.createBindGroup(moshLayout, {
          src: childTexture, prev: read.state, next: write.state, display, params: paramsBuf.buffer,
      } as never),
  })
  if (!sim) return null
  return {
      outputs: {childTexture: sim.childTexture, display: sim.display},
      bindInputs: sim.bindInputs,
      getComputeNodes: (fp) => sim.tick(fp, ({groups, dt, localTime}) => {
          paramsBuf.write({time: localTime, dt, /* … */})
          return [pipeline.with(groups as never)]
      }),
  }
  ```
- **Traps:** The pair swaps ONLY when `frame` returns steps, so an idle-skipped frame cannot desync
  the orientation. `localTime` advances BEFORE `frame` runs. **Kernels are not the scaffold's
  business** — compile-time kernel specialisation stays in the shader (TimeTrail picks four variants
  from `trailSource` × `tintMode`).

### buildStableFluidsKernels / createStableFluidsPasses
- **Import:** `import {buildStableFluidsKernels, createStableFluidsPasses} from "@coreroot/gpu/porters"`
- **Signature:** `buildStableFluidsKernels(layout: FluidSolverLayout, opts: StableFluidsOptions) → {curl, vorticity, divergence, jacobi, gradSubtract, advectVel, copyVel, advectDye?, copyDye?}`
  — call at MODULE scope, next to the layout. `createStableFluidsPasses(root, kernels, {n, bindGroup}) → {…passes, solveSteps({jacobiIters, vorticity?})}`
  — call per instance, inside the `compute` hook. Every option is structural; the only runtime values
  are the five uniform members below.
- **Use when:** a shader advances a velocity (and optionally dye) grid with Stable Fluids. **Do NOT
  hand-roll:** the 8–11 kernel chain (neighbour index, curl, vorticity confinement, divergence +
  pressure warm start, Jacobi, gradient subtract, semi-Lagrangian advection, ping-pong copies), which
  was copy-pasted verbatim in six shaders.
- **Used by:** `SmokeFlow` (simplest — clamped + densityAge), `Fog` (toroidal + velocity cap, no
  dissipation), `SmokeFill` (`solidMask`), `ParticleFlow` (`dye: 'none'` + `publishVelocityTexture`,
  substitutes its own vorticity pass).
- **Example:**
  ```ts
  const solverKernels = buildStableFluidsKernels(fluidLayout, {
      n: N, namePrefix: 'smokeFlow', boundary: 'clamped', dye: 'densityAge',
      dyeDissipation: true, ageAdvance: true,
  })
  export const {jacobi: jacobiKernel, /* … */} = solverKernels
  // per instance, in the compute hook:
  const solver = createStableFluidsPasses(root, solverKernels, {n: N, bindGroup: fluidBg})
  return [splat, ...solver.solveSteps({jacobiIters: 10}), output]
  ```
  The axes: `boundary: 'clamped' | 'toroidal'` (free-slip box vs wrapping neighbours AND wrapping
  advection backtrace — Fog, so its cloud has no visible frame) · `dye: 'none' | 'densityAge' | 'rgb'`
  (velocity only / density + age driving a fresh→aged color ramp / the dye IS the picture) ·
  `dyeDissipation` · `ageAdvance` · `solidMask` (clamped only) · `velocityCap` (only Fog needs it — a
  permanent field with no dissipation can accumulate enough energy over minutes to advect past a cell
  per step and go unstable) · `publishVelocityTexture` · `pressureDecay` (the warm start; 0.8 in all
  six).
- **Traps:** **The layout entry names are a fixed ABI** — the kernel bodies index them directly, so
  they must be exactly `velA, velB, dyeA, dyeB, pressure, divergence, maskBuf, velOutTex, params`.
  The shader still DECLARES the layout (C4), keeping its buffer/uniform surface visible in one place.
  **The params struct must expose `dt`, `curlStrength`, `velFade`** (+ `dyeFade` with
  `dyeDissipation`, + `colorDecay` with `ageAdvance`); extra members are free, but these names are not
  configurable because **TGSL cannot read a struct member by a computed name**. `namePrefix` must stay
  stable per consumer (kernel names appear in the emitted WGSL, and two solvers in one tree collide
  without distinct prefixes). `solveSteps` returns only the CORE chain: emission splats go before it,
  the output pass and any per-shader restoration after (Fog appends `colorRestore`); pass `vorticity`
  to substitute your own variant of that one pass and still share the other six. TypeScript cannot
  express "`dyeA` is required iff `dye !== 'none'`", so a layout missing an entry its options need
  fails at `tgpu.resolve` — which every consumer's resolve gate exercises.

### buildFluidOutputKernel / neighbourIndex
- **Import:** `import {buildFluidOutputKernel, neighbourIndex} from "@coreroot/gpu/porters"`
- **Signature:** `buildFluidOutputKernel(outputLayout: FluidOutputLayout, {n, namePrefix, dye})`;
  `neighbourIndex(n: number, boundary: FluidBoundary)` → the neighbour flat-index body fn.
- **Use when:** `buildFluidOutputKernel` writes the solved dye into the storage texture the fragment
  samples — separate from the kernel set because it reads a DIFFERENT bind group (dye read-only plus a
  write-only storage texture) and a kernel may reference only one layout. `neighbourIndex` is exported
  for a consumer whose OWN kernels need the same lookup (ParticleFlow's vorticity pass).
- **Used by:** all six fluid shaders; `neighbourIndex` directly by `ParticleFlow`.
- **Example:**
  ```ts
  export const outputKernel = buildFluidOutputKernel(outputLayout, {n: N, namePrefix: 'fog', dye: 'densityAge'})
  const nidx = neighbourIndex(N, 'clamped')   // in your own kernel
  ```
- **Traps:** `neighbourIndex` is **memoised per `(n, boundary)`** and every kernel in one consumer
  must share ONE instance, or the resolved WGSL grows `nidx`, `nidx_1`, `nidx_2` duplicates. Its
  emitted names are `nidx` (clamped) and `nw` (toroidal) — deliberately the names the hand-written
  copies used, so don't "improve" them without accepting snapshot churn across six shaders.
  `densityAge` writes `(density, age, 0, 0)` and `rgb` writes `(r, g, b, 1)`; both are `rgba16float`,
  which is what gives the fragment hardware-bilinear filtering free on the upsample. That format is
  effectively part of the solver's ABI — the working spelling for the texture type is
  `d.textureStorage2d<'rgba16float', 'write-only'>` (exported as `FluidStorageTexture`), because
  `d.WgslStorageTexture` is not assignable to `std.textureStore`'s parameter.

### gaussianBrushSq / gaussianBrushShiftedSq (+ vector forms, + the option factories)
- **Import:** `import {gaussianBrushShiftedSq, gaussianBrushFnSq} from "@coreroot/gpu/porters"`
- **Signature:** body fns `(distSq: f32, radiusSq: f32) → f32`; vector forms `gaussianBrushRaw` /
  `gaussianBrushShifted` take `(cellPos: vec2f, center: vec2f, radiusSq: f32)`. Factories
  `gaussianBrushFnSq({edgeShift, cutoffSigmas?})` / `gaussianBrushFn(...)` pick the form at
  composition time; `gaussianBrushShiftedSqFn(cutoffSigmas)` selects the edge constant directly.
- **Use when:** any cursor- or emitter-driven weight over a grid. **Prefer the SHIFTED form.** A raw
  `exp(-d²/r²)` truncated at the radius still carries 1/e (~37%) of its peak at the cut, and a cell
  lattice renders that step as a staircase ring. The shifted form,
  `(exp(-d²/r²) − e^−c²)/(1 − e^−c²)` clamped to [0, 1], is 1 at the center and exactly 0 at the edge.
- **Used by:** `SmokeFlow`, `Smoke`, `Fog`, `InkFlow`, `ParticleFlow` (the D-8 brush sites);
  `PixelThrow`'s CPU sim is where the shifted form originated.
- **Example:**
  ```ts
  const brush = gaussianBrushFnSq({edgeShift: true})          // 1σ cutoff — the fluid case
  const w = brush(distSq, std.max(radius * radius, 1.0))
  const brush3 = gaussianBrushShiftedSqFn(3.0)                // a 3σ-truncated consumer
  ```
- **Traps:** **`cutoffSigmas` must match where the CALLER stops evaluating**, because it sets the edge
  constant to `exp(-cutoffSigmas²)`. `radiusSq` is always σ² — the Gaussian's own scale — so passing a
  3σ consumer the default 1σ constant rescales the whole profile rather than just its edge (that is
  why CursorRipples and CursorTrail, which truncate at 3σ, need `cutoffSigmas: 3`). The factory
  memoises and `$name`s per cutoff, so two differently-truncated brushes in one tree don't collide.
  `radiusSq` must already be guarded by the caller — the grid-space consumers floor it at one cell
  (`max(r*r, 1.0)`), which is a different floor from what a UV-space consumer wants. Both fns are pure
  float and therefore CPU-executable under vitest with golden-value assertions (C8) — keep them that
  way.

### createAgentSystem
- **Import:** `import {createAgentSystem} from "@coreroot/gpu/porters"`
- **Signature:** `(params: GpuFragmentParams, config) → AgentSystem | null` (null = no GPU device; the
  caller returns `null` from its `compute` hook). Config: `layout` (the shader's module-scope
  `tgpu.bindGroupLayout` — **structural**, and it drives both the WGSL and the allocations), `params`
  (the per-frame uniform struct schema), `output` (`{key, name, size, format}`), `maxAgents`,
  `countCap?` (`{desktop, mobile}`), `externalKeys?`, `extraBindGroup?`, `pipelines`
  (`{[key]: {kernel, threads: 'agents' | 'max' | 'grid' | 'fixed', size?, extra?}}`), `program` (the
  per-frame step order), `initStep?`.
- **Use when:** a shader simulates N agents in storage buffers and renders them by additive splat into
  atomic accumulators. **Do NOT hand-roll:** one `createStateBuffer` per layout entry, the output
  texture + `registerComputeTexture` + `onCleanup`, the uniform, the bind group, the
  `createGuardedCompute` wrappers, the `initialized`/`lastSeed` latch, the mobile count cap, or the
  per-frame nodes array. A sixth agent shader is now ~15 declarative lines.
- **Used by:** `FloatingParticles` (the minimal case), `Boids` (adds `reseed`), `ParticleField` (2D
  `'grid'` dispatch + a late-bound child RTT via `externalKeys`), `Particles` (an `extraBindGroup`
  chained onto the per-particle steps only, and a 6-step program).
- **Example:**
  ```ts
  const sys = createAgentSystem(params, {
      layout: simLayout, params: SimParams, maxAgents: MAX_MOTES,
      output: {key: 'outTex', name: 'particleTexture', size: [RES, RES], format: STATE_FORMAT},
      countCap: {desktop: MAX_MOTES, mobile: 3000},
      pipelines: {
          init: {kernel: initKernel, threads: 'max'},
          update: {kernel: updateKernel, threads: 'agents'},
          splat: {kernel: splatKernel, threads: 'agents'},
          resolve: {kernel: resolveKernel, threads: 'fixed', size: [RES, RES]},
      },
      initStep: 'init', program: ['update', 'splat', 'resolve'],
  })
  if (!sys) return null
  return {outputs: sys.outputs, getComputeNodes: (fp) => {
      const {dt, aspect} = readAgentFrame(fp)
      sys.writeParams({/* … */})
      return sys.frame({count: sys.resolveCount(g('count', 1200))})
  }}
  ```
- **Traps:** **The kernels and the layout stay in the SHADER** — the harness never emits WGSL, so it
  can never rename anything, which is the whole reason the five migrations were snapshot-clean. There
  is deliberately no `namePrefix`: WGSL identifiers come from the layout's KEY NAMES and each kernel's
  `$name`, both of which the shader owns. Generalisation for any future sim scaffold: **put the layout
  and the kernels in the shader; put the allocation, binding and dispatch in the scaffold — the seam
  is the layout object.** `output.size` may be non-square and per-composition (ParticleField fits it
  to the frame aspect) but the ACCUMULATOR length comes from the module-scope layout, so it must cover
  the max. `externalKeys` makes `frame()` return `null` until `bindExternal` succeeds and the caller
  must forward that null. `extra: true` is per PIPELINE, not per system (Particles chains its SDF bind
  group onto the four per-particle steps and deliberately not onto the two full-target passes).

### worldSplatWindow / texelSplatWindow
- **Import:** `import {worldSplatWindow, texelSplatWindow, SplatWindow} from "@coreroot/gpu/porters"`
- **Signature:** body fns. `worldSplatWindow(pos: vec2f, reach: f32, aspect: f32, res: f32) → SplatWindow`;
  `texelSplatWindow(center: vec2f, rad: i32, last: vec2i) → SplatWindow`, where
  `SplatWindow = {x0, x1, y0, y1: i32}` — inclusive and already clipped.
- **Use when:** a splat kernel needs the texel range one agent can reach. `world*` for agents in
  screen-proportional world space (x ∈ [0, aspect], y ∈ [0, 1]) splatting into a square accumulator;
  `texel*` for agents already projected to texels with an isotropic radius. **Do NOT hand-roll:** a
  fixed ±N-texel box (it hard-clips large `arrow`/`streak`/`glow` shapes into squares — a correctness
  fix, not an optimisation), an unguarded `pos.x / aspect`, or a per-texel bounds test inside the loop.
- **Used by:** `Boids`, `MagneticFilings`, `FloatingParticles` (world); `Particles`, `ParticleField`
  (texel).
- **Example:**
  ```ts
  const win = worldSplatWindow(pos, bodyR * EXT + AA_W * 2.0, aspect, d.f32(RES))
  for (let py = win.y0; py <= win.y1; py++) {
      for (let px = win.x0; px <= win.x1; px++) { /* no bounds test needed */ }
  }
  ```
- **Traps:** **The X radius is `aspect`× NARROWER than the Y radius** — the accumulator is square but
  the world domain is `aspect` wide, so one x-texel covers `aspect` times as much world. This is the
  bug that is invisible at 1:1 and stretches every agent on a wide canvas. An off-screen agent clips
  to an EMPTY range, so that IS the off-target early-out and a separate `onTarget` test is optional.
  `last` is a parameter rather than derived because the target extent can be a per-frame uniform.

### energyCoverageAlpha / integrateSemiImplicitEuler
- **Import:** `import {energyCoverageAlpha, integrateSemiImplicitEuler} from "@coreroot/gpu/porters"`
- **Signature:** body fns. `energyCoverageAlpha(energy: f32, k: f32) → f32` = `1 − e^(−k·energy)`;
  `integrateSemiImplicitEuler(vel: vec3f, force: vec3f, dt: f32, dragMul: f32, maxSpeed: f32) → vec3f`.
- **Use when:** `energyCoverageAlpha` turns an accumulated fixed-point energy into a coverage alpha —
  saturating, so overlapping agents deepen toward opaque without clipping and empty texels give
  exactly 0. `integrateSemiImplicitEuler` integrates a force-based 3D agent. **Do NOT hand-roll:**
  explicit Euler (it gains energy and eventually explodes on an underdamped spring at real browser
  frame rates), a per-frame drag multiply instead of a CPU-computed `exp(−drag·dt)`, or omitting the
  speed clamp.
- **Used by:** all five agent shaders (`k` = 1.6 for the four energy accumulators, 2.6 for
  ParticleField's weight); the integrator by `Particles` and `ParticleField`.
- **Example:**
  ```ts
  const alpha = energyCoverageAlpha(energy, 1.6)
  const nextVel = integrateSemiImplicitEuler(vel, force, dt, Math.exp(-DRAG * dt), MAX_SPEED)
  ```
- **Traps:** `dragMul` is `Math.exp(-DRAG * dt)` computed on the CPU — passing a raw drag rate makes
  the decay frame-rate dependent. The speed clamp is a safety net, not a feel knob. The READ-THEN-ZERO
  next to `energyCoverageAlpha` is deliberately NOT extracted: WGSL cannot take a storage array as a
  function parameter, so `atomicLoad` + `atomicStore(0)` must stay in the shader beside the layout
  entry it names. (A whole-kernel `makeEnergyResolve` was investigated and closed as won't-do — the
  five resolve bodies genuinely differ, and it would need dynamic layout-key access inside a
  `'use gpu'` body, whose failure mode is silently wrong WGSL rather than an error.)

### FIXED_POINT_GAINS / resolveRenderRes / SIZE_REF_RES / R2_ALPHA / R3_ALPHA / readAgentFrame / makeCpuValueGetter
- **Import:** all from `"@coreroot/gpu/porters"`.
- **Signature:** `FIXED_POINT_GAINS = {HARD: 637, SOFT: 255}` · `resolveRenderRes({desktop, mobile}) → number`
  + `SIZE_REF_RES = 1024` · `R2_ALPHA` / `R3_ALPHA` (plastic-constant strides) ·
  `readAgentFrame(frameParams, fallbackAspect?) → AgentFrame` · `makeCpuValueGetter(getCpuValue) → g(key, fallback)`.
- **Use when:** the CPU half of an agent sim. **Do NOT hand-roll:** a fixed-point gain (a HARD shape's
  interior is coverage exactly 1, so at the SOFT gain it resolves to `1 − e^(−1.6)` ≈ 0.80 and a solid
  dot renders 80% gray), a device-tiered resolution, a low-discrepancy stride, a dt clamp, or an
  `undefined`-tolerant prop read.
- **Used by:** all five agent shaders. `ParticleFlow` still inlines its own `R2`/`R3` triples and
  should adopt (the values are byte-identical).
- **Example:**
  ```ts
  const RES = resolveRenderRes({desktop: 1024, mobile: 512})
  const sizeScale = RES / SIZE_REF_RES
  const {dt, aspect} = readAgentFrame(fp, 1)
  const g = makeCpuValueGetter(getCpuValue)
  ```
- **Traps:** **Always pair `resolveRenderRes` with `SIZE_REF_RES`** — the resolution is device-tiered
  but agent sizes must be authored against the FIXED reference, or the same preset renders a larger
  flock on a phone. `readAgentFrame`'s `fallbackAspect` is effectively required reading and has no safe
  default (16/9 for a viewport-filling generator, 1 for a square or refit domain — Particles and
  ParticleField pass it); a default would silently pick a winner, the same reasoning as D-1's `flipY`.
  `makeCpuValueGetter` exists because a dynamic prop (range map, mouse binding) can resolve to a
  non-number for a frame and a kernel uniform must never receive `undefined`. R2 is the per-AGENT
  stride (quasi-uniform coverage for any count prefix, so a count slider needs no re-seed); R3 is the
  per-FRAME one (dithering a voxel lattice out of moiré with a shape boundary).

### createGuardedCompute
- **Import:** `import {createGuardedCompute, type ComputeStep, type KitComputePipeline} from "@coreroot/gpu/porters"`
- **Signature:** `(root: TgpuRoot, callback: (...threads: number[]) => void, opts?: {size?: [number] | [number, number] | [number, number, number], bindGroup?: TgpuBindGroup}) → KitComputePipeline`
- **Use when:** a shader builds its own compute kernel pass (simulation step, blur-map fill,
  bright-extract pre-pass). **Do NOT hand-roll:** a manual `If(cy.lessThan(ch))` bounds guard or
  workgroup sizing — the guarded pipeline does both.
- **Used by:** `CursorRipples`, `Boids`, `ProgressiveBlur` (30 shaders).
- **Example:**
  ```ts
  const pass = createGuardedCompute(root, (cx, cy) => { 'use gpu'; kernel(cx, cy) },
      {size: [1024, 640], bindGroup: bg})
  pass.dispatch()
  ```
- **Traps:** The callback is a `'use gpu'` arrow **transpiled where you write it**, not inside the
  helper — the directive must be in your file. `dispatch()` throws unless `opts.size` was given;
  otherwise call `dispatchThreads(...)` explicitly. Shaders that only *consume* a kit blur
  (`Blur`, `ChannelBlur`) don't need this at all. Child-RTT compute inputs bind **late** via
  `bindInputs`, after the child target exists.

### agents.makeAgentWeightFn / makeAgentTexelWeightFn
- **Import:** `import {agents} from "@coreroot/gpu/kit"`
- **Signature:** factories. `makeAgentWeightFn(shape: string) → tgpu.fn([f32,f32,f32,f32,f32], f32)` as `(t, n, bodyR, aaW, soft) → coverage`; `makeAgentTexelWeightFn(shape) → tgpu.fn(...)` as `(q, t, n, bodyR, aaW) → coverage`
- **Use when:** rasterizing one particle/agent into a splat accumulator. `t` is along the agent's
  heading, `n` perpendicular (pass plain x/y for un-oriented agents). **Do NOT hand-roll:** an
  agent shape SDF or its AA/Gaussian falloff.
- **Used by:** `Boids`, `Particles`, `FloatingParticles`, `MagneticFilings`, `ParticleFlow`,
  `ParticleField`.
- **Example:**
  ```ts
  const weight = agents.makeAgentTexelWeightFn(shapeName)   // shapeName from propValues
  // inside the splat kernel, after the radial reject on q:
  const w = weight(q, t, n, bodyR, aaW)
  ```
- **Traps:** **Pick the right one.** `makeAgentTexelWeightFn` bakes softness as a compile-time
  constant and reuses the reject quotient `q` — use it unless softness is a live uniform (the
  Particles/FloatingParticles slider), in which case use `makeAgentWeightFn`. Both are **factories**
  (C3): call them once at composition time, outside the kernel, and note they `$name` their output —
  two configurations in one tree need distinct names. Hard-edged shapes return exactly 1 in the
  interior, so a consumer whose alpha curve tops out below opaque must over-drive its fixed-point
  gain (see Boids). Shape names come from `agents.AGENT_SHAPES` via
  `agents.resolveAgentShape(name, fallback)` — don't trust a raw string prop.

---

## Host/CPU lifecycle helpers

*In this category when the shader loads external resources, tracks pointer input, or rasterizes to a
canvas. These are plain CPU/TypeScript — no TGSL — and run from the `compute` hook or the definition's
lifecycle callbacks.*

*Three modules: `kit/host/pointer.ts` (pointer motion, idle gating, stamp ribbons),
`kit/host/mediaLifecycle.ts` (textures that swap, URL loaders, video elements) and
`kit/host/canvasRaster.ts` (2D-canvas raster targets and their invalidation).*

**The media recipe.** A media shader is `createSwappableMediaTexture` for the texture handle + one
source helper (`createUrlSourceLoader` + `decodeImageSource` for images, `createVideoElementSource` for
video/webcam) + `media.mediaSurface` in the fragment. A rastering shader is `createCanvasRasterTarget` +
`createRasterInvalidator` (+ `createFontDependentRaster` when a webfont is involved), writing into a
`createSwappableMediaTexture` — or, for text, into a slot of the composition's shared `glyphAtlasFor`.
Read `ImageTexture` and `Text`.

### createSwappableMediaTexture
- **Import:** `import {createSwappableMediaTexture} from "@coreroot/gpu/kit/host/mediaLifecycle"`
- **Signature:** `(params: GpuFragmentParams, {label, initial?, format?}) → {kit, ensureSize, write, unwrap, width, height, disposed}`
- **Use when:** a shader owns a media texture at all — whether it changes size (image at native
  resolution, re-rastered text, resized DOM capture) or never does (a fixed glyph atlas). **Do NOT
  hand-roll:** the `let current = createMediaTexture(...)` + `registerMediaTexture(() => current.texture)`
  + alloc/swap/destroy + `isDisposed` quartet.
- **Used by:** `ImageTexture`, `HTMLInCanvas`, `Ascii`, `ObjectTracker`.
- **Example:**
  ```ts
  const tex = createSwappableMediaTexture(params, {label: 'Image', initial: {width: 2, height: 2}})
  // …after rastering:
  tex.ensureSize(texW, texH)
  tex.write(canvas)
  // …in the builder:
  const sampled = tex.kit.sample(uv)
  ```
- **Traps:**
  - **Register `kit` once.** It is the getter the pass manager polls; a second `registerMediaTexture`
    allocates a second binding.
  - `ensureSize` at the same size is a deliberate NO-OP. Do not "force a fresh texture" to get a new bind
    group — a same-size re-`write` is the cheap path and is what you want.
  - Everything guards on disposal, including `write`. A fixed-size atlas shader gets that guard for free
    by using this instead of a bare `createMediaTexture`.
  - It generates a mip chain by default (`mipmaps ?? true`), and an implicit-LOD `sample` then picks a
    level from the on-screen size. Right for photos; wrong for a raster sampled at a known density
    (a supersampled glyph atlas), which should pass `mipmaps: false` and sample level 0.

### glyphAtlasFor
- **Import:** `import {glyphAtlasFor} from "@coreroot/gpu/kit/host/glyphAtlas"`
- **Signature:** `(params) → {kit, slot(onRect) → {set(canvas, w, h), clear()}}`
- **Use when:** several nodes of one composition upload coverage rasters (text). One `r8unorm`
  texture, no mips, one binding for all of them (via `params.shared`); each slot reserves bucketed
  room with a 4-texel gutter, re-uploads in place when it fits and repacks into a fresh texture when
  it outgrows its room. `onRect` reports the slot's place as atlas UV `[left, top, width, height]`;
  publish it as an extraField and map box-local UV into it.
- **Used by:** `Text`.
- **Traps:** write white glyphs on transparent (coverage lands in `.r` via premultiplied copy).
  Uploads are synchronous on purpose: captures re-raster inside the frame they draw.

### createUrlSourceLoader + decodeImageSource
- **Import:** `from "@coreroot/gpu/kit/host/mediaLifecycle"`
- **Signature:** `createUrlSourceLoader(params, {prop, load(url, ctx), onError?}) → {currentUrl, isLoading, request}`;
  `decodeImageSource(url) → Promise<{source, width, height, naturalWidth, naturalHeight, close}>`
- **Use when:** a shader loads anything from a URL prop. **Do NOT hand-roll:** the `setTimeout(0)`
  kickoff, the per-frame URL diff, or the in-flight guard.
- **Used by:** `ImageTexture`, and (through `createVideoElementSource`) `VideoTexture`.
- **Traps:**
  - **Check `ctx.isDisposed()` after every await, and call `ctx.commit()` only after the last one that
    can fail.** An uncommitted URL is retried every frame — that is how a broken URL behaves today, and
    it is the loader's retry policy, not a bug in your callback (a backoff is queued as a Phase 12 item).
  - `decodeImageSource` returns TWO size pairs. The raster pair sizes the texture; the natural pair goes
    to `registerNaturalSize`. For an SVG they differ by up to 64× and conflating them makes a 32px logo
    claim a 2048px layout slot.
  - Call `close()` on the decoded source once written — and on the dispose-during-await path too.

### createVideoElementSource
- **Import:** `from "@coreroot/gpu/kit/host/mediaLifecycle"`
- **Signature:** `(params, {source: {kind:'url',prop} | {kind:'webcam',constraints?}, loop?, naturalSizeKey?, metadataTimeoutMs?, autoplayRequired?, onError?}) → {getSource, element}`
- **Use when:** any shader that samples live video frames. Hand `getSource` straight to
  `registerExternalTexture`.
- **Used by:** `VideoTexture`, `WebcamTexture`.
- **Traps:**
  - The readiness gate (`readyState < 2 || videoWidth === 0 → null`) is REQUIRED: importing an
    undecodable frame is a validation error, and the pass manager's contract is that a null source skips
    the pass for that frame.
  - `onError` is a hook, not a message — the taxonomy stays per-shader (only `WebcamTexture` knows how to
    phrase a camera-permission denial), and the stages are `'acquire' | 'autoplay'` so a metadata failure
    is reported exactly once.
  - `autoplayRequired` exists because webcam and video genuinely disagree: a blocked autoplay still
    yields a decodable first video frame, but a rejected `play()` on a camera acquisition is fatal. It
    defaults to the forgiving behavior.
  - Firefox has no `importExternalTexture`. The fallback (per-frame `texture.write(video)` into a media
    texture) has exactly ONE marked site, in `readySource`. Do not scatter a second one.

### createCanvasRasterTarget / createRasterInvalidator / createFontDependentRaster
- **Import:** `from "@coreroot/gpu/kit/host/canvasRaster"`
- **Signature:** `createCanvasRasterTarget(params, {maxAxis?, supersample?, contextAttributes?}) → {context, canvas, rasterSize, resize, devicePixelRatio, disposed}`;
  `createRasterInvalidator(params, {key(), run(), minIntervalMs?, initialKey?})`;
  `createFontDependentRaster({key(), load(), onLoaded()}) → {ensure()}`
- **Use when:** a shader draws with a 2D context and uploads the result.
- **Used by:** `Text`, `Ascii`, `ObjectTracker`.
- **Traps:**
  - `rasterSize` is the whole point: `dpr × supersample`, clamped so NEITHER axis exceeds `maxAxis`,
    floored at 2. Degrading density is the correct failure mode; exceeding the cap throws at texture
    creation.
  - The canvas is created LAZILY and released by zeroing both axes. Dropping the reference alone leaves a
    multi-megabyte backing store to the GC.
  - `minIntervalMs` throttles the CHECK, not the raster — the check reads live prop handles.
  - Pass `initialKey` when the shader rastered eagerly at composition (`ObjectTracker`), or the first
    frame redoes it.
  - `createFontDependentRaster` memoizes per font key, so an already-loaded font does not re-draw. Put
    anything that must invalidate a MEASURE cache at the top of `onLoaded` — `measureText` keys on font
    availability, so metrics cached from the fallback have to be dropped before re-measuring (`Text` does
    this; `Ascii` and `ObjectTracker` do not need to).

**Related contract addition.** A media shader that should be measurable by the layout system declares
`naturalSizeKey?: {fromProp: string} | {fixed: string}` on its definition; `utilities/measure.ts` reads
it off the generated registry instead of the shader-name switch it used to hard-code.

### createPointerVelocityTracker
- **Import:** `import {createPointerVelocityTracker} from "@coreroot/gpu/kit/host/pointer"`
- **Signature:** `(opts?: {smoothing?, teleportGuard?, minDrag?, initialX?, initialY?}) → {update(pointer, dt) → PointerFrame}`,
  where `PointerFrame` carries `x/y`, `prevX/prevY`, `dx/dy`, `dragDist`, `velX/velY` (per second),
  `smoothVelX/Y`, `smoothSpeed`, `teleport`, `moving`. Defaults: `smoothing` 0.2, `teleportGuard` 0.25
  UV, `minDrag` 0.0006, position center.
- **Use when:** a compute hook turns pointer motion into a force. **Do NOT hand-roll:** the
  `prevMx/prevMy` differencing + `Math.max(dt, 0.001)` divide + exponential smoothing + the teleport
  guard. Ten shaders had their own copy and only two had the guard.
- **Used by:** `CursorRipples`, `CursorTrail`, `Fog`, `Smoke`, `SmokeFlow`, `SmokeFill`, `InkFlow`,
  `ParticleFlow`, `ReactionDiffusion` (all nine).
- **Example:**
  ```ts
  const pointer = createPointerVelocityTracker()
  // per frame, in getComputeNodes:
  const pf = pointer.update(frameParams.pointer, dt)
  if (pf.moving) { /* inject force at pf.x, pf.y using pf.smoothVelX/Y */ }
  ```
- **Traps:** **The teleport guard is why this exists.** Entering the canvas, a tab switch, or a window
  drag jumps the pointer, and an ungated delta fires one enormous impulse — a popped ripple, a painted
  stripe, a bloomed blob. On a teleport frame the tracker still ADVANCES its previous position (the
  jump is absorbed, not deferred) and reports `dx/dy/velX/velY` as zero, so the smoothed velocity
  decays rather than spiking. `dragDist` is the RAW distance and is still reported on such a frame —
  **gate on `moving`, not on `dragDist`.**

### createIdleGate / decayFadeSeconds
- **Import:** `import {createIdleGate, decayFadeSeconds} from "@coreroot/gpu/kit/host/pointer"`
- **Signature:** `createIdleGate({warmupFrames?}?) → {markActive(now), tickFrame(), shouldSkip(now, fadeSeconds, driving?), neverActive, warmedUp}`
  (the last two are getters); `decayFadeSeconds(ratio, rate, minRate = 0.05) → ln(ratio) / max(rate, minRate)`
  seconds.
- **Use when:** a sim should stop dispatching once it has settled — the last rendered texture persists
  on screen, so a settled sim costs zero GPU time. **Do NOT hand-roll:** a "has the mouse moved
  lately" timer, and do not gate on pointer speed.
- **Used by:** `ParticleFlow`, `SmokeFlow`, `InkFlow`.
- **Example:**
  ```ts
  const idle = createIdleGate({warmupFrames: WARMUP_FRAMES + 1})
  if (pf.moving) idle.markActive(now)
  if (idle.shouldSkip(now, decayFadeSeconds(255, dissipationRate))) return null
  idle.tickFrame()
  ```
- **Traps:** **Pointer speed being zero is NOT a valid skip condition** — a stationary cursor over an
  evolving field must keep dispatching. The gate is about elapsed time since the last INPUT.
  `warmupFrames` exists because a sim whose first frames initialize and settle state must not freeze
  mid-warm-up, and its semantics are off-by-one-sensitive: `warmedUp` is `framesTicked >= warmupFrames`,
  so an original `warmup > WARMUP_FRAMES` maps to `warmupFrames: WARMUP_FRAMES + 1`. Get it wrong and
  you silently disable (or never enable) idle freezing. `decayFadeSeconds(255, rate)` means "wait until
  the field is below one 8-bit level"; the `minRate` floor (0.05) stops a near-zero dissipation from
  opening an infinite window.

### pathStampRibbon
- **Import:** `import {pathStampRibbon} from "@coreroot/gpu/kit/host/pointer"`
- **Signature:** `(nodes: ComputeStep[], {fromX, fromY, dx, dy, dragDist, stepSize, maxSteps, scale?, prepare?, write, pass}) → stampCount`.
  Appends alternating write-thunk / dispatch pairs to `nodes`. Stamp count is
  `min(maxSteps, max(1, ceil(dragDist / stepSize)))`.
- **Use when:** a drag should lay a continuous ribbon of splats instead of dotted gaps on a fast flick.
  **Do NOT hand-roll:** the sub-step loop with its per-stamp uniform write. The pass manager runs each
  thunk then its dispatch in `device.queue` order, which is what lets ONE uniform buffer carry a
  different value per stamp.
- **Used by:** `InkFlow` (ink splats), `ParticleFlow` (cursor force wake), `CursorTrail` (trail dots).
- **Example:**
  ```ts
  const stamps = pathStampRibbon(nodes, {
      fromX: pf.prevX, fromY: pf.prevY, dx: pf.dx, dy: pf.dy,
      dragDist: pf.dragDist, stepSize: radius * 0.4, maxSteps: 8,
      write: (posX, posY) => stampParams.write({cursorX: posX, cursorY: posY, /* … */}),
      pass: stampPass.with(stampBg),
  })
  ```
- **Traps:** **`write` runs at DISPATCH time.** Anything that must advance once per stamp regardless of
  dispatch goes in `prepare(t)`, which runs at BUILD time in stamp order and whose result is handed to
  `write` — InkFlow's color cycle uses it, so its ribbon hues are laid down in stroke order either
  way. Stamps sit at cell centres (`t = (s + 0.5)/n`), so a one-stamp frame lands mid-stroke rather
  than at an endpoint. `maxSteps` is a hard cap so a huge flick cannot blow out the dispatch count.

---

## Cross-cutting: per-node animated time

### animatedTime
- **Import:** `import {animatedTime} from "@coreroot/gpu/porters"`
- **Signature:** `(params: {props, uniforms}, seedName?: string, fieldName?: string) → Expr`
- **Use when:** the shader has a continuously advancing clock (phase, drift, evolution).
  **Do NOT hand-roll:** a time accumulator, and do not read a raw wall-clock uniform.
- **Used by:** `Aurora`, `PerlinNoise`, `FlowField` (49 shaders).
- **Example:**
  ```ts
  // definition: animatedTime: {speed: 'speed'}
  const t = animatedTime(params)
  const body = call(myField, 'myField', [uv, t])
  ```
- **Traps:** The accumulation is the **renderer's** job — you must declare
  `animatedTime: {speed: '<speedProp>'}` on the definition or the field never advances. `speed = 0`
  pauses without rewinding. Pass `seedName` to add a seed uniform on the GPU (the seed stays an
  ordinary uniform, not part of the declaration). For a second independent clock, declare
  `extraAnimatedTimes: {<key>: '<speedProp>'}` and read it with
  `animatedTime(params, undefined, '_animTime_<key>')` (FlowField's evolution clock). A `speed` prop
  feeding `animatedTime` is a rate, so it should *not* get the `['range', 'map']` dynamic-driver UI.

---

## Cross-cutting: prop-config factories

These live in `src/utilities/` (CPU-side data builders, not GPU code) and each returns a fresh
`PropConfig` equal to the block the fleet inlines today, so adopting one changes no default, no UI
metadata and no emitted WGSL (Gate A).

**Why so many required arguments.** Anything that genuinely varies across the fleet is a REQUIRED
parameter with no default — `edges` defaults split three ways, every `center` carries its own
description, and descriptions are user-visible tooltip text in the Design Editor. A convenient default
would let a migration silently change a shader's default value or its metadata, which is exactly what
the extraction exists to make impossible. **Pass the shader's current values verbatim.** Likewise
optional keys are omitted rather than set to `undefined`, because emitting `units: undefined` would
change the shape of the prop metadata the design editor generates from it.

### edgesPropConfig
- **Import:** `import {edgesPropConfig, type EdgeMode} from "@coreroot/utilities/propConfigs"`
- **Signature:** `(defaultMode: EdgeMode, description: string, overrides?: {label?, group?}) → PropConfig<string>`.
  `EdgeMode = 'stretch' | 'transparent' | 'mirror' | 'wrap'`.
- **Use when:** the shader samples outside its source and needs the standard 4-option edge select
  (i.e. any distortion). **Do NOT hand-roll:** the option list, `transform: transformEdges`, or the
  `compileTime: true` marker.
- **Used by:** `Twirl`, `Kaleidoscope`, `Bend` (22 shaders inline this identically apart from three
  fields).
- **Example:**
  ```ts
  props: {edges: edgesPropConfig('mirror', 'How to handle samples that fall outside the source')}
  ```
- **Traps:** `defaultMode` is required because the fleet genuinely disagrees — `mirror`, `stretch` and
  `transparent` are all in real use — so pass the shader's current default, don't pick a nice one. The
  prop is `compileTime`, which is why `edges` can never take the `['range', 'map']` driver UI. NOT a
  consumer: `Form3D.uvMode`, which reuses `transformEdges` but is a different prop (three options, no
  `transparent`, no `compileTime`).

### centerPropConfig
- **Import:** `import {centerPropConfig} from "@coreroot/utilities/propConfigs"`
- **Signature:** `(description: string, overrides?: {default?: {x, y}, label?, group?, units?: string[]}) → PropConfig<Parameters<typeof transformPosition>[0]>`
- **Use when:** the shader has a `center` position prop. **Do NOT hand-roll:** the
  `transform: transformPosition` + `ui.type: 'position'` block.
- **Used by:** `Twirl`, `Bulge`, `Kaleidoscope` (9 shaders directly, plus every 2D shape via
  `defineSdfShapeShader`).
- **Traps:** The returned prop type is **wider** than most shaders declared by hand — numbers OR px
  strings, matching what `transformPosition` actually accepts. Type your `ComponentProps.center` as
  `Parameters<typeof transformPosition>[0]` (the distortion fleet was widened to this); declaring
  `{x: number, y: number} | string` produces an assignability error. `units: ['%', 'px']` is present on
  the shape/material family and absent on the distortions, so it is opt-in and the key is omitted when
  not requested.

### originProp
- **Import:** `import {originProp} from "@coreroot/utilities/propConfigs"`
- **Signature:** `() → PropConfig<BoundingBoxOrigin>` — no arguments
- **Use when:** the shape's center position should be measurable from a chosen reference edge.
  **Do NOT hand-roll:** it is byte-identical in all 36 files that declare it, which is why this one
  takes no arguments.
- **Used by:** every 2D shape, via `defineSdfShapeShader`.
- **Traps:** No `transform` — the string is read by the bounds layer, not the GPU. Don't add one.

### shapeStrokeProps
- **Import:** `import {shapeStrokeProps} from "@coreroot/utilities/propConfigs"`
- **Signature:** `(overrides?: {softnessDescription?, strokeThicknessMax?, strokeColorDescription?, strokePositionDescription?}) → {softness, strokeThickness, strokeColor, strokePosition}`
- **Use when:** a 2D shape needs the standard softness + stroke block. **Do NOT hand-roll:** all four
  props (byte-identical across 13 shape files).
- **Used by:** the 2D shape fleet, via `defineSdfShapeShader` (`Ring` passes
  `strokeThicknessMax: 0.1` and its own softness wording).
- **Traps:** The baseline is the FLEET standard: plain `range` inputs, softness max 0.1 / step 0.001,
  stroke thickness max 0.2 / step 0.005. **`Circle` is not that block** — its `softness` and
  `strokeThickness` are `['range', 'map']` typed with max 1 / 0.5 and a `dimensional: 'canvas-height'`
  marker, so it spreads its two over the factory result rather than adopting them.

### shapeColorSpaceProp
- **Import:** `import {shapeColorSpaceProp} from "@coreroot/utilities/propConfigs"`
- **Signature:** `(overrides?: {description?: string}) → PropConfig<string>`
- **Use when:** a shape's fill → stroke blend needs a color-space choice. **Do NOT hand-roll:** the
  `transformColorSpace` + `colorSpaceOptions` + `compileTime: true` block.
- **Used by:** the 2D shape fleet, via `defineSdfShapeShader`.
- **Traps:** `compileTime` — the mode selects a `mixColors` variant at composition, so a change
  recompiles and only the chosen space's math is emitted. This is the shape-specific wording ("Color
  Blending", group 'Colors'); a generator blending arbitrary colors wants the generic `colorSpace`
  prop convention instead.

### objectFitProp
- **Import:** `import {objectFitProp} from "@coreroot/utilities/propConfigs"`
- **Signature:** `(defaultValue: string, description: string, options: {allowNone: boolean}) → PropConfig<string>`
- **Use when:** a texture shader (image / video / webcam) needs the object-fit select.
  **Do NOT hand-roll:** the string→mode table or the option list.
- **Used by:** `ImageTexture`, `VideoTexture` (`allowNone: false`), `WebcamTexture` (`allowNone: true`).
- **Traps:** `allowNone` picks between two **deliberately different** tables, and they must not be
  unified because the mode numbers are baked into shipped presets. `allowNone: false` (Image, Video)
  retires `none`, coalescing it to fill (2) so old presets keep filling their box rather than
  re-cropping, and falls back to 2. `allowNone: true` (Webcam) keeps `none` as its own mode 4 (natural
  pixel size), offers it in the select, and falls back to cover (0).

### colorStopsPropConfig
- **Import:** `import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"`
- **Signature:** `() → PropConfig<ColorStop[] | null>`
- **Use when:** the shader's color count is open-ended (a gradient). **Do NOT hand-roll:** the
  `stops` prop's default, transform marker, `compileTimeWhen`, or UI metadata — and do not add
  `colorC`, `colorD`, … props instead.
- **Used by:** `LinearGradient`, `Voronoi`, `Plasma` (17 shaders).
- **Example:**
  ```ts
  props: {
      colorA: {...}, colorB: {...},
      stops: colorStopsPropConfig(),
  }
  // in fragment: branch on (propValues.stopCount as number) > 1 →
  //   kitColorStops.mixColorStopsRuntime(t, {...}, colorSpaceMode), else the 2-color path.
  //   Or just call noiseColor.mixStopsOrColorsExpr(params, t), which does that branch for you.
  ```
- **Traps:** Import from `@coreroot/utilities/colorStops` — `gpu/kit/colorStops.ts` exports a
  same-named duplicate, and shaders use the utilities one. Default is `null` so legacy
  `colorA`/`colorB` presets keep working; don't default it to an array. Fixed MAX of 8 stops. The
  `compileTimeWhen` recompiles only when the *active stop count* changes (or presence toggles) —
  same-count color edits update in place, so don't mark the prop `compileTime: true`.


---

## Addendum — concept-word sweep additions (2026-09-01)

New generic vocabulary from the look-dissolution sweep (Blob/Aurora/Mesh/Flowing/
Studio/Marble/Strands). Std words first (reach for these); the kit kernel behind each
is noted for machinery work only.

- `std/paint/noise`: **`waves`/`wavyLine`/`organicWaves`** (sine-interference engine,
  schedule as data — pure Expr, no kernel); **`ribbons`** (gradient-colored ribbon
  stack between two anchors → `kit/noisePaints.ribbonsField` + `ribbonPathFrame`,
  `toneLift`).
- `std/paint/fields`: **`scatterField`** (nearness-blended drifting anchor field →
  `kit/fields.scatterField` + `spiralScatter`); **`warpedPoint`** (the warped COORD from
  the shared `warpDomain` instance); **`warpStep`** (one chained IQ warp level →
  `kit/fields.chainedWarpStep`, carries the domainWarp2 Gate-C note); **`wrapRamp`**
  (palette wrapping → `kit/gradientPaints.rampWrap`); **`colorLadder3`**, **`quadBlend`**,
  **`standardPalette`**, **`mixColorsIn`** (palette words, moved here).
- `std/paint/light`: **`rayBands`** (banded shimmer along an axis); **`domeNormal`** +
  **`shine`** (fake puffy surface + specular lobe — compose, never fuse);
  **`glowSpot`**/**`glowAt`** (anisotropic / positioned gaussian lobes over
  `kit/lightfields`).
- `std/paint/compose`: **`layered`** (weighted variation stacks → weightedSum/
  totalWeight); **`dithered`** (anti-banding output → `kit/tone.rgbDither`).
- `std/mask`: **`softBand`** (asymmetric band around a curve), **`softDisc`**
  (gamma-curved soft disc).
- `std/frames`: **`rawSurfaceOf`** (raw-canvas surface), **`centredFrame`** (delta +
  dist about a position prop).
- `std/math`: **`rotate2`**, **`hashPhase`**, **`reflect`/`refract`/`exp2`** (promoted).
- `std/signal`: **`driveUnitDirection`** (CPU-normalized direction → extraFields),
  **`cpuHash01`**.
- Also generic in kit (behind existing words): `segmentArcPerp` (shapePaints),
  `godraysRayLayer` (gradientPaints), `strandsPathFrame` → `ribbonPathFrame`.
