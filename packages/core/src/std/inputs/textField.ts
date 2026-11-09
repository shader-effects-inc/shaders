/**
 * The distance field of a `text` input: for every point, the nearest point on the glyph outlines,
 * the signed distance to it in canvas pixels, and which letter that point belongs to. Built on the
 * GPU by jump flooding from the drawn coverage, and rebuilt only when the text is redrawn.
 */
// Maintainer notes. Three raw compute modules over two rgba32float textures at a working size
// (the canvas, capped at FIELD_MAX on its long side; seeds are placed sub-texel, so the field stays
// accurate to about a canvas pixel). Texel layout everywhere: (signed distance px, nearest uv x,
// nearest uv y, letter index). `seed` marks texels the 0.5 coverage contour crosses, moved onto the
// contour along the coverage gradient, and tags each with its letter (the letter box it sits
// deepest in, rotated with the text). `flood` runs jump sizes N/2 … 1 plus one extra 1 (JFA+1).
// `finish` writes the signed distance from each texel centre to its nearest seed. The jump size
// is a uniform written before each dispatch; queue order keeps the writes and dispatches in step.
import {createComputeModule, createStorageTexture, destroyTexture, type ComputeModule} from './module'
import {moduleFor, texelPrelude, uniformBlock} from './common'
import type {ComputeStep} from '../../gpu/compute'
import type {InputBuildContext} from './types'

/** Longest side of the field, in texels. */
const FIELD_MAX = 1024

/** No outline anywhere (empty text): the distance reads as this. */
export const FIELD_FAR = 1e6

export interface TextFieldSources {
    /** The drawn coverage (alpha), at canvas size, filterable. */
    coverage: () => unknown
    /** The letter boxes (rgba32float, one texel per letter: centre uv, half size px). */
    letters: () => unknown
    /** [letter count, rotation in radians, canvas width px, canvas height px]. */
    info: () => readonly [number, number, number, number]
}

export interface TextField {
    /** The finished field (swaps between the two textures). */
    current(): unknown
    size(): {width: number; height: number}
    /** Rebuild on the next frame its modules are ready (the text was redrawn or resized). */
    invalidate(): void
    steps(): ComputeStep[]
}

const LETTER_AT = `fn _letterAt(s: vec2f) -> f32 {
  let n = i32(_txt.x);
  let cs = vec2f(cos(_txt.y), sin(_txt.y));
  var best = -1.0;
  var bestScore = 1e9;
  for (var i = 0; i < n; i++) {
    let box = textureLoad(_letters, vec2i(i, 0), 0);
    let d = (s - box.xy) * _txt.zw;
    // Into the text's own frame (it is drawn rotated by _txt.y, clockwise in y-down pixels).
    let local = vec2f(d.x * cs.x + d.y * cs.y, -d.x * cs.y + d.y * cs.x);
    let q = abs(local) / max(box.zw, vec2f(0.5));
    let score = max(q.x, q.y);
    if (score < bestScore) { bestScore = score; best = f32(i); }
  }
  return best;
}`

export function createTextField(ctx: InputBuildContext, src: TextFieldSources): TextField {
    const mc = moduleFor(ctx)
    const sizeFor = (c: {width: number; height: number}) => {
        const s = Math.min(1, FIELD_MAX / Math.max(1, c.width, c.height))
        return {width: Math.max(1, Math.round(c.width * s)), height: Math.max(1, Math.round(c.height * s))}
    }
    let size = sizeFor(ctx.canvas())
    const make = (i: number) => createStorageTexture(ctx.params, size, 'rgba32float', `${ctx.componentName}:${ctx.name} (distance ${i})`)
    let tex: [unknown, unknown] = [make(0), make(1)]
    ctx.onCleanup(() => tex.forEach(destroyTexture))
    let dirty = true
    // Which texture each pass reads and writes; the finished field is `tex[done]`.
    let readIdx = 0
    let done = 1

    const txt = uniformBlock(ctx, 1, `${ctx.name}:text-field`)
    const jfa = uniformBlock(ctx, 1, `${ctx.name}:text-field-jump`)
    const txtDecl = {kind: 'uniform' as const, name: '_txt', type: 'vec4f', buffer: () => txt.buffer}

    const seed: ComputeModule = createComputeModule(mc, {
        label: `text '${ctx.name}' distance (seed)`,
        decls: LETTER_AT,
        segments: [],
        includeShared: false,
        workgroup: [8, 8],
        resources: [
            txtDecl,
            {kind: 'texture', name: '_cov', texture: src.coverage},
            {kind: 'texture', name: '_letters', texture: src.letters, unfilterable: true},
            {kind: 'storageTexture', name: '_out', format: 'rgba32float', texture: () => tex[0]},
        ],
        entry: `${texelPrelude('_out')}
  let px = 1.0 / vec2f(_size);
  let a = textureSampleLevel(_cov, _samp, uv, 0.0).a;
  let ax0 = textureSampleLevel(_cov, _samp, uv - vec2f(px.x, 0.0), 0.0).a;
  let ax1 = textureSampleLevel(_cov, _samp, uv + vec2f(px.x, 0.0), 0.0).a;
  let ay0 = textureSampleLevel(_cov, _samp, uv - vec2f(0.0, px.y), 0.0).a;
  let ay1 = textureSampleLevel(_cov, _samp, uv + vec2f(0.0, px.y), 0.0).a;
  let lo = min(a, min(min(ax0, ax1), min(ay0, ay1)));
  let hi = max(a, max(max(ax0, ax1), max(ay0, ay1)));
  var o = vec4f(1000000.0, -1.0, -1.0, -1.0);
  if (lo < 0.5 && hi >= 0.5) {
    // Onto the 0.5 contour along the coverage gradient, at most a texel away.
    let g = vec2f(ax1 - ax0, ay1 - ay0) / (2.0 * px);
    let gg = dot(g, g);
    var s = uv;
    if (gg > 1e-6) { s = uv - (a - 0.5) * g / gg; }
    s = uv + clamp(s - uv, -px, px);
    o = vec4f(0.0, s, _letterAt(s));
  }
  textureStore(_out, gid.xy, o);`,
    })

    const flood: ComputeModule = createComputeModule(mc, {
        label: `text '${ctx.name}' distance (flood)`,
        segments: [],
        includeShared: false,
        workgroup: [8, 8],
        resources: [
            txtDecl,
            {kind: 'uniform', name: '_jfa', type: 'vec4f', buffer: () => jfa.buffer},
            {kind: 'texture', name: '_src', texture: () => tex[readIdx], unfilterable: true},
            {kind: 'storageTexture', name: '_out', format: 'rgba32float', texture: () => tex[1 - readIdx]},
        ],
        entry: `${texelPrelude('_out')}
  let k = i32(_jfa.x);
  let m = vec2i(_size) - vec2i(1);
  var best = textureLoad(_src, vec2i(gid.xy), 0);
  var bestD = 1e30;
  if (best.y >= 0.0) { let e = (best.yz - uv) * _txt.zw; bestD = dot(e, e); }
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let c = textureLoad(_src, clamp(vec2i(gid.xy) + vec2i(dx, dy) * k, vec2i(0), m), 0);
      if (c.y < 0.0) { continue; }
      let e = (c.yz - uv) * _txt.zw;
      let dd = dot(e, e);
      if (dd < bestD) { bestD = dd; best = c; }
    }
  }
  textureStore(_out, gid.xy, best);`,
    })

    const finish: ComputeModule = createComputeModule(mc, {
        label: `text '${ctx.name}' distance (finish)`,
        segments: [],
        includeShared: false,
        workgroup: [8, 8],
        resources: [
            txtDecl,
            {kind: 'texture', name: '_cov', texture: src.coverage},
            {kind: 'texture', name: '_src', texture: () => tex[readIdx], unfilterable: true},
            {kind: 'storageTexture', name: '_out', format: 'rgba32float', texture: () => tex[1 - readIdx]},
        ],
        entry: `${texelPrelude('_out')}
  let c = textureLoad(_src, vec2i(gid.xy), 0);
  var o = vec4f(1000000.0, -1.0, -1.0, -1.0);
  if (c.y >= 0.0) {
    let d = length((uv - c.yz) * _txt.zw);
    let inside = textureSampleLevel(_cov, _samp, uv, 0.0).a >= 0.5;
    o = vec4f(select(d, -d, inside), c.yzw);
  }
  textureStore(_out, gid.xy, o);`,
    })

    ctx.onResize((canvas) => {
        const next = sizeFor(canvas)
        if (next.width === size.width && next.height === size.height) return
        const old = tex
        size = next
        tex = [make(0), make(1)]
        old.forEach(destroyTexture)
        dirty = true
    })

    const threads = () => [size.width, size.height] as const
    const seedStep = seed.step(threads)
    const floodStep = flood.step(threads)
    const finishStep = finish.step(threads)

    return {
        current: () => tex[done],
        size: () => size,
        invalidate() {
            dirty = true
        },
        steps() {
            if (!dirty || !seed.ready() || !flood.ready() || !finish.ready()) return []
            dirty = false
            const steps: ComputeStep[] = []
            steps.push(() => {
                txt.write([...src.info()])
                readIdx = 0
            })
            steps.push(seedStep)
            let k = 1
            while (k * 2 < Math.max(size.width, size.height)) k *= 2
            const jumps: number[] = []
            for (; k >= 1; k = Math.floor(k / 2)) jumps.push(k)
            jumps.push(1)
            for (const j of jumps) {
                steps.push(() => jfa.write([j]))
                steps.push(floodStep)
                steps.push(() => {
                    readIdx = 1 - readIdx
                })
            }
            steps.push(finishStep)
            steps.push(() => {
                done = 1 - readIdx
            })
            return steps
        },
    }
}
