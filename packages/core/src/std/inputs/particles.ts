/**
 * `simulate.particles`: thousands of points that move and are drawn into a texture the body reads
 * by name (`stars(uv)`). Give it forces as data, or write the motion yourself; or skip motion and
 * place every point with a formula (`draw` only). The engine owns the buffers, spawning, ageing,
 * the splat into the texture and the phone cap.
 */
// Maintainer notes. State: one storage array of `Particle`. Per frame: spawn (dead or expired
// slots), update (forces → your `step` → integrate), clear the accumulation buffer, splat (each
// point adds its colour into fixed-point atomics, 1 px or a soft footprint), resolve (atomics →
// rgba16float). `neighbors` adds a coarse count grid the update reads the next frame as
// `crowd(p)` / `crowdSlope(p)`. Without spawn/step/forces the field is stateless: `draw(index)`
// places every point each frame, and no particle buffer is allocated.
import type {WgslBody} from '../wgsl'
import type {ComputeStep} from '../../gpu/compute'
import {createComputeModule, rewriteForModule, type ModuleResource} from './module'
import {bodyText, canvasTexture, moduleFor, siblingReads} from './common'
import {isPropRef, scalarValue, withBuilder, type InputNode, type InputSource, type ScalarArg} from './types'
import type {PropRef} from '../values'

/** A number, or a prop read live each frame (`p('pull')`): what a force's strength and sizes take. */
export type ForceArg = number | PropRef

/**
 * A force, as data. Each object has one key, the force's name, set to its strength (a number or a
 * prop, so a slider can drive it live).
 */
export type ParticleForce =
    | {readonly gravity: ForceArg}
    | {readonly drag: ForceArg}
    | {readonly cursor: ForceArg; readonly radius?: ForceArg}
    | {readonly vortex: ForceArg; readonly center?: readonly [number, number]}
    | {readonly orbit: ForceArg; readonly center?: readonly [number, number]}
    | {readonly wind: ForceArg; readonly scale?: ForceArg}
    | {readonly wander: ForceArg}
    | {readonly spring: ForceArg}
    | {readonly separate: ForceArg}

/** Where new particles appear. */
export type ParticleSpawn =
    | 'scatter'
    | 'ring'
    | 'edge'
    | 'atPointer'
    | {readonly fromShape: InputSource}

export interface ParticlesInput extends InputNode {
    readonly kind: 'simulate.particles'
    readonly count: ScalarArg
    readonly max: number
    readonly mobileMax: number
    readonly spawn: ParticleSpawn | string
    readonly step: string | null
    readonly forces: readonly ParticleForce[]
    readonly lifetime: {readonly seconds: number; readonly jitter: number} | null
    readonly draw: string | null
    readonly splat: 'point' | 'soft'
    readonly normalize: boolean
    readonly resolution: number
    readonly neighbors: {readonly radius: number} | null
}

const PARTICLE_STRUCT = `struct Particle {
  pos: vec2f,
  vel: vec2f,
  home: vec2f,
  age: f32,
  life: f32,
  seed: f32,
  alive: f32,
  data: vec2f,
  color: vec4f,
}`
const PARTICLE_STRIDE = 64

const POINT_STRUCT = `struct Point { pos: vec2f, size: f32, color: vec4f }`
/** What `draw` sees besides `p` and `index`: the size, in pixels, of the texture points land in
 *  (the canvas times `resolution`). Size points from it, not from `viewport`. */
const DRAW_PRELUDE = 'let targetSize = vec2f(f32(_a.y), f32(_a.z));'

/** Fixed-point scale for the additive splat (colour units per 1.0). */
// 4096: 1/4096 steps in linear light keep faint stars and soft edges from banding near black
// (256 left ~13 sRGB levels there). Headroom: 2^32 / (64 × 4096) = 16k maximal splats per pixel.
const FIX = 4096

function isMobile(): boolean {
    return typeof navigator !== 'undefined' && /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent)
}

/**
 * Thousands of moving points drawn into a texture, read by name in WGSL.
 *
 * - `count` (number or prop) up to `max`; `mobile: {max}` caps it on phones.
 * - `forces`: data, e.g. `[{gravity: 0.3}, {drag: 0.5}, {cursor: 1}, {wind: 0.4}]`. Every strength (and
 *   `radius`, `scale`) takes a number or a prop, `{spring: p('pull')}`, read live, so sliders drive the
 *   physics. Forces: `gravity`, `drag` (settling), `spring` (back to the spawn point), `cursor`
 *   (pushed from the pointer while one is over the canvas), `wander` (each particle jiggles on its
 *   own), `wind` (a shared flow), `vortex`, `orbit`, `separate`.
 * - `spawn`: `'scatter'` (default), `'ring'`, `'edge'`, `'atPointer'`, `{fromShape: source}` (a
 *   texture's opaque area, e.g. `samplePoints` or a layer prop), or a WGSL body returning a
 *   `Particle` (it sees `index` and `rand: vec4f`).
 * - `step`: WGSL run per particle per frame with `var p: Particle` in scope (and `index`, `rand`).
 *   Forces are applied before it; position integrates after it.
 * - `lifetime: {seconds, jitter}` respawns particles as they age out.
 * - `draw`: WGSL returning `Point(pos, sizePx, color)` per particle. It sees `p` (with state),
 *   `index`, and `targetSize` (vec2f): the pixel size of the texture points land in, which is what
 *   `sizePx` is measured against. Scale sizes by `targetSize.y / 1080.0` so a field looks the same
 *   at every canvas size and in captures.
 *   Without a state (no spawn, step or forces) it sees only `index`: the formula-driven mode.
 * - `splat`: `'point'` (1 px) or `'soft'` (a round footprint `size` pixels wide, at most 16).
 *   Points add up (light, alpha summed). A soft point's `color` is its peak, so its total light
 *   grows with size²; `normalize: true` keeps the total the same at every size.
 * - `neighbors: {radius}` lets `step` read `crowd(p)` and `crowdSlope(p)`: how crowded a spot is.
 *
 * @example
 * ```ts
 * inputs: {
 *   motes: simulate.particles({count: 20000, forces: [{wind: 0.4}, {cursor: 1}, {drag: 0.6}], lifetime: {seconds: 6, jitter: 0.5}, splat: 'soft'}),
 * },
 * paint: wgsl`return motes(uv);`,
 * ```
 */
export function particlesInput(opts: {
    count: ScalarArg
    max?: number
    mobile?: {max: number}
    spawn?: ParticleSpawn | WgslBody | string
    step?: WgslBody | string
    forces?: readonly ParticleForce[]
    lifetime?: {seconds: number; jitter?: number}
    draw?: WgslBody | string
    splat?: 'point' | 'soft'
    /** Soft points keep the same total light at every size (`color` is then the light of a 1-pixel point). Default false: `color` is the peak. */
    normalize?: boolean
    resolution?: number
    neighbors?: {radius: number}
}): ParticlesInput {
    const max = Math.max(1, Math.min(1 << 20, Math.floor(opts.max ?? (typeof opts.count === 'number' ? opts.count : 65536))))
    const spawnArg = opts.spawn
    const spawn: ParticleSpawn | string =
        spawnArg === undefined ? 'scatter'
            : typeof spawnArg === 'string' && ['scatter', 'ring', 'edge', 'atPointer'].includes(spawnArg) ? (spawnArg as ParticleSpawn)
                : typeof spawnArg === 'object' && spawnArg !== null && 'fromShape' in spawnArg ? (spawnArg as ParticleSpawn)
                    : bodyText(spawnArg as WgslBody | string)
    const node = {
        kind: 'simulate.particles' as const,
        count: opts.count,
        max,
        mobileMax: Math.min(max, opts.mobile?.max ?? max),
        spawn,
        step: opts.step ? bodyText(opts.step) : null,
        forces: opts.forces ?? [],
        lifetime: opts.lifetime ? {seconds: opts.lifetime.seconds, jitter: opts.lifetime.jitter ?? 0} : null,
        draw: opts.draw ? bodyText(opts.draw) : null,
        splat: opts.splat ?? 'soft',
        normalize: opts.normalize ?? false,
        resolution: Math.min(1, Math.max(0.25, opts.resolution ?? 1)),
        neighbors: opts.neighbors ?? null,
        // A cursor force or pointer spawn is data, invisible to the code scan: declare it.
        readsPointer: (opts.forces ?? []).some((f) => 'cursor' in f) || spawn === 'atPointer',
    }
    return withBuilder(node, (ctx) => {
        const stateless = opts.spawn === undefined && !node.step && node.forces.length === 0 && !node.lifetime && !!node.draw
        const cap = isMobile() ? node.mobileMax : node.max
        const live = () => Math.max(0, Math.min(cap, Math.floor(scalarValue(node.count, ctx.params, cap))))
        const target = canvasTexture(ctx, {scale: node.resolution, label: `${ctx.name} (particles)`})
        const mod = moduleFor(ctx)
        const allCode = [typeof spawn === 'string' ? spawn : '', node.step ?? '', node.draw ?? ''].join('\n')
        const reads = siblingReads(ctx, allCode)
        const shapeSource = typeof spawn === 'object' ? ctx.source(spawn.fromShape) : null
        if (typeof spawn === 'object' && !shapeSource) ctx.report('spawn.fromShape: the source has no texture yet')
        if (shapeSource) reads.push({call: '_shape', texture: shapeSource})

        // Accumulation: 4 u32 per pixel, fixed point. Sized to the target; reallocated on resize.
        let accumPixels = target.width() * target.height()
        let accum = ctx.device.createBuffer({size: accumPixels * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: `${ctx.name}:accum`})
        ctx.onResize(() => {
            const px = target.width() * target.height()
            if (px === accumPixels) return
            accum.destroy()
            accumPixels = px
            accum = ctx.device.createBuffer({size: px * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: `${ctx.name}:accum`})
        })
        ctx.onCleanup(() => accum.destroy())
        const state = stateless ? null : ctx.device.createBuffer({size: Math.max(16, cap * PARTICLE_STRIDE), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: `${ctx.name}:particles`})
        if (state) ctx.onCleanup(() => state.destroy())
        const args = ctx.device.createBuffer({size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: `${ctx.name}:args`})
        ctx.onCleanup(() => args.destroy())

        const CROWD = 128
        const crowdCount = node.neighbors ? ctx.device.createBuffer({size: CROWD * CROWD * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: `${ctx.name}:crowd`}) : null
        if (crowdCount) ctx.onCleanup(() => crowdCount.destroy())
        const crowd = node.neighbors ? canvasTexture(ctx, {fixed: {width: CROWD, height: CROWD}, label: `${ctx.name} (crowd)`}) : null

        // _a.x = live count, _a.y = target width, _a.z = target height, _a.w = first frame (1/0)
        const common: ModuleResource[] = [{kind: 'uniform', name: '_a', type: 'vec4u', buffer: () => args}]
        const stateRes: ModuleResource[] = state ? [{kind: 'storage', name: '_ps', type: 'array<Particle>', access: 'read_write', buffer: () => state}] : []
        const accumRes: ModuleResource = {kind: 'storage', name: '_acc', type: 'array<atomic<u32>>', access: 'read_write', buffer: () => accum}
        const crowdRes: ModuleResource[] = crowdCount ? [{kind: 'storage', name: '_crowd', type: 'array<atomic<u32>>', access: 'read_write', buffer: () => crowdCount}] : []
        const crowdRead: ModuleResource[] = crowd ? [{kind: 'texture', name: '_crowdTex', texture: crowd.current}] : []
        const decls = [PARTICLE_STRUCT, POINT_STRUCT, `fn _rand4(i: u32, salt: u32) -> vec4f {
  var h = vec4u(i * 1664525u + salt * 22695477u, i ^ 0x9e3779b9u, i * 2246822519u + salt, salt * 3266489917u + i);
  h = h * 1664525u + 1013904223u; h.x += h.y * h.w; h.y += h.z * h.x; h.z += h.x * h.y; h.w += h.y * h.z;
  h = h ^ (h >> vec4u(16u)); h.x += h.y * h.w; h.y += h.z * h.x; h.z += h.x * h.y; h.w += h.y * h.z;
  return vec4f(h) / 4294967296.0;
}`].join('\n')
        const crowdFns = crowd ? `fn crowd(q: vec2f) -> f32 { return textureSampleLevel(_crowdTex, _samp, q, 0.0).x; }
fn crowdSlope(q: vec2f) -> vec2f {
  let e = 1.0 / ${CROWD}.0;
  return vec2f(crowd(q + vec2f(e, 0.0)) - crowd(q - vec2f(e, 0.0)), crowd(q + vec2f(0.0, e)) - crowd(q - vec2f(0.0, e))) * 0.5;
}` : ''

        const spawnFn = (() => {
            if (typeof spawn === 'string' && !['scatter', 'ring', 'edge', 'atPointer'].includes(spawn)) {
                return `fn _spawn(index: u32, rand: vec4f) -> Particle {\n${spawn}\n}`
            }
            const place = typeof spawn === 'object' && shapeSource
                ? `var q = rand.xy;
  for (var t = 0u; t < 16u; t++) {
    let c = _shape(q);
    if (c.a > 0.5) { break; }
    q = _rand4(index, 97u + t).xy;
  }
  let pos = q;`
                : spawn === 'ring' ? `let a = rand.x * 6.2831853; let pos = vec2f(0.5) + vec2f(cos(a), sin(a)) * (0.3 + rand.y * 0.02);`
                    : spawn === 'edge' ? `let side = u32(rand.x * 4.0); let t = rand.y;
  let pos = select(select(select(vec2f(t, 0.0), vec2f(1.0, t), side == 1u), vec2f(t, 1.0), side == 2u), vec2f(0.0, t), side == 3u);`
                        : spawn === 'atPointer' ? `let pos = pointer + (rand.xy - 0.5) * 0.02;`
                            : `let pos = rand.xy;`
            // Spawning from a shape can miss it (16 random tries cover a word's ~10% of the frame
            // only most of the time, and none on frames before the shape exists), leaving a
            // particle with a home off the shape: a stray dot its spring holds forever. _rehome
            // keeps looking, a few tries a frame, so every particle settles onto the shape.
            const rehome = typeof spawn === 'object' && shapeSource
                ? `fn _rehome(index: u32, p: Particle) -> Particle {
  if (_shape(p.home).a > 0.5) { return p; }
  var q = p;
  for (var t = 0u; t < 4u; t++) {
    let c = _rand4(index, u32(_f.frame) * 13u + 211u + t).xy;
    if (_shape(c).a > 0.5) { q.home = c; break; }
  }
  return q;
}`
                : `fn _rehome(index: u32, p: Particle) -> Particle { return p; }`
            return `fn _spawn(index: u32, rand: vec4f) -> Particle {
  ${place}
  return Particle(pos, vec2f(0.0), pos, 0.0, 0.0, rand.w, 1.0, vec2f(0.0), vec4f(1.0));
}
${rehome}`
        })()

        // A force argument in WGSL: a float literal, or the prop's name (the step function is
        // rewritten to read props from the props uniform).
        const arg = (v: unknown, what: string): string => {
            if (isPropRef(v)) {
                if (!ctx.props[v.name]) throw new Error(`simulate.particles: ${what}: p('${v.name}') is not a prop of this component`)
                return `f32(${v.name})`
            }
            const n = Number(v)
            if (!Number.isFinite(n)) throw new Error(`simulate.particles: ${what} must be a number or p('prop')`)
            return Number.isInteger(n) ? `${n}.0` : String(n)
        }
        const forceCode = node.forces.map((f) => {
            const [name, raw] = Object.entries(f)[0] as [string, unknown]
            const k = arg(raw, name)
            switch (name) {
                case 'gravity': return `p.vel.y += ${k} * dt;`
                case 'drag': return `p.vel *= exp(-${k} * dt * 4.0);`
                case 'cursor': {
                    const r = arg((f as {radius?: unknown}).radius ?? 0.15, 'cursor radius')
                    // Only while a real pointer is over the canvas: with none, `pointer` is just a
                    // remembered spot (the centre at first) and would hold a hole open forever.
                    return `{ let d = p.pos - pointer; let l = max(length(d), 0.0001); p.vel += normalize(d) * ${k} * exp(-(l * l) / (${r} * ${r})) * (0.5 + length(pointerVelocity)) * pointerOver * dt * 4.0; }`
                }
                case 'vortex': {
                    const c = (f as {center?: readonly number[]}).center ?? [0.5, 0.5]
                    return `{ let d = p.pos - vec2f(${c[0]}, ${c[1]}); p.vel += vec2f(-d.y, d.x) * ${k} * dt; }`
                }
                case 'orbit': {
                    const c = (f as {center?: readonly number[]}).center ?? [0.5, 0.5]
                    return `{ let d = vec2f(${c[0]}, ${c[1]}) - p.pos; p.vel += normalize(d) * ${k} * dt / max(length(d), 0.05); }`
                }
                case 'wind': {
                    const s = arg((f as {scale?: unknown}).scale ?? 3, 'wind scale')
                    return `{ let q = p.pos * ${s} + vec2f(time * 0.05, 0.0);
    let n = vec2f(sin(q.y * 1.7 + time * 0.3) + sin(q.x * 2.3 - time * 0.2), cos(q.x * 1.9 + time * 0.25) + cos(q.y * 2.1 + time * 0.15));
    p.vel += n * ${k} * dt * 0.2; }`
                }
                // Each particle jiggles on its own (unlike wind, a shared flow): restless, alive.
                case 'wander': return `p.vel += (rand.xy - 0.5) * ${k} * sqrt(max(dt, 0.0)) * 4.0;`
                case 'spring': return `p.vel += (p.home - p.pos) * ${k} * dt * 4.0;`
                case 'separate': return crowd ? `p.vel -= crowdSlope(p.pos) * ${k} * dt * 4.0;` : ''
                default: throw new Error(`simulate.particles: unknown force '${name}'`)
            }
        }).join('\n  ')

        const lifeCode = node.lifetime
            ? `let expired = p.age > p.life;`
            : `let expired = false;`
        const lifeInit = node.lifetime ? `p.life = ${node.lifetime.seconds} * (1.0 + (rand.z - 0.5) * 2.0 * ${node.lifetime.jitter});` : `p.life = 1e9;`

        const drawFn = node.draw
            ? (stateless
                ? `fn _draw(index: u32) -> Point {\n${DRAW_PRELUDE}\n${node.draw}\n}`
                : `fn _draw(p: Particle, index: u32) -> Point {\n${DRAW_PRELUDE}\n${node.draw}\n}`)
            : `fn _draw(p: Particle, index: u32) -> Point { return Point(p.pos, 2.0, p.color); }`

        const userSegments = [
            {label: `${ctx.name} spawn`, text: rewriteForModule(spawnFn, mod, reads), offset: 1},
            {label: `${ctx.name} draw`, text: rewriteForModule(drawFn, mod, reads), offset: node.draw ? 2 : 1},
        ]
        const stepFn = `fn _step(index: u32, rand: vec4f, p0: Particle) -> Particle {
  var p = p0;
  ${forceCode}
  ${node.step ?? ''}
  p.pos += p.vel * dt;
  p.age += dt;
  return p;
}`

        const build = (label: string, entry: string, segments: {label: string; text: string; offset?: number}[], resources: ModuleResource[], wg: [number, number]) =>
            createComputeModule(mod, {
                label: `${ctx.name}: ${label}`,
                decls: decls + '\n' + (resources.some((r) => r.name === '_crowdTex') ? crowdFns : ''),
                segments,
                entry,
                resources: [...common, ...resources],
                textureReads: reads,
                workgroup: wg,
            })

        const steps: ComputeStep[] = []
        const threads1 = () => [live(), 1] as const
        // The module that seeds particles on the first frame (`_a.w == 1`), if any.
        let seeder: {ready(): boolean} | null = null
        if (state) {
            const update = build('update', `  let index = gid.x;
  if (index >= _a.x) { return; }
  var p = _ps[index];
  let rand = _rand4(index, u32(_f.frame));
  ${lifeCode}
  if (p.alive < 0.5 || expired || _a.w == 1u) {
    let r0 = _rand4(index, select(u32(_f.frame) * 7u + 3u, 0u, _a.w == 1u));
    p = _spawn(index, r0);
    ${lifeInit.replace('rand.z', 'r0.z')}
    p.age = select(0.0, r0.z * p.life, _a.w == 1u && p.life < 1e8);
    p.alive = 1.0;
  }
  p = _rehome(index, p);
  _ps[index] = _step(index, rand, p);`, [...userSegments.slice(0, 1), {label: `${ctx.name} step`, text: rewriteForModule(stepFn, mod, reads), offset: node.step ? stepFn.slice(0, stepFn.indexOf(node.step)).split('\n').length - 1 : 0}], [...stateRes, ...crowdRead], [64, 1])
            seeder = update
            steps.push(update.step(threads1))
        }
        const clear = () => {
            const enc = ctx.device.createCommandEncoder({label: `${ctx.name}: clear`})
            enc.clearBuffer(accum)
            if (crowdCount) enc.clearBuffer(crowdCount)
            ctx.device.queue.submit([enc.finish()])
        }
        steps.push(clear)
        const splatBody = node.splat === 'point'
            ? `  let px = vec2i(pt.pos * vec2f(f32(_a.y), f32(_a.z)));
  if (px.x < 0 || px.y < 0 || px.x >= i32(_a.y) || px.y >= i32(_a.z)) { return; }
  let k = (u32(px.y) * _a.y + u32(px.x)) * 4u;
  let c = vec4u(clamp(pt.color, vec4f(0.0), vec4f(64.0)) * ${FIX}.0);
  atomicAdd(&_acc[k], c.x); atomicAdd(&_acc[k + 1u], c.y); atomicAdd(&_acc[k + 2u], c.z); atomicAdd(&_acc[k + 3u], c.w);`
            : `  let size = vec2f(f32(_a.y), f32(_a.z));
  let center = pt.pos * size;
  let r = clamp(pt.size * 0.5, 0.5, 8.0);
  let ri = i32(ceil(r));
  for (var j = -ri; j <= ri; j++) { for (var i = -ri; i <= ri; i++) {
    let px = vec2i(floor(center)) + vec2i(i, j);
    if (px.x < 0 || px.y < 0 || px.x >= i32(_a.y) || px.y >= i32(_a.z)) { continue; }
    let d = length(vec2f(px) + 0.5 - center) / r;
    if (d > 1.0) { continue; }
    let w = (1.0 - d * d) * (1.0 - d * d)${node.normalize ? ' / max(1.0472 * r * r, 1.0)' : ''};
    let k = (u32(px.y) * _a.y + u32(px.x)) * 4u;
    let c = vec4u(clamp(pt.color * w, vec4f(0.0), vec4f(64.0)) * ${FIX}.0);
    atomicAdd(&_acc[k], c.x); atomicAdd(&_acc[k + 1u], c.y); atomicAdd(&_acc[k + 2u], c.z); atomicAdd(&_acc[k + 3u], c.w);
  } }`
        const crowdSplat = crowdCount ? `
  let cq = vec2i(clamp(pt.pos, vec2f(0.0), vec2f(0.999)) * ${CROWD}.0);
  atomicAdd(&_crowd[u32(cq.y) * ${CROWD}u + u32(cq.x)], 1u);` : ''
        const splatModule = build('splat', `  let index = gid.x;
  if (index >= _a.x) { return; }
  ${stateless ? 'let pt = _draw(index);' : 'let p = _ps[index];\n  if (p.alive < 0.5) { return; }\n  let pt = _draw(p, index);'}
${splatBody}${crowdSplat}`, userSegments.slice(1), [...stateRes, accumRes, ...crowdRes], [64, 1])
        steps.push(splatModule.step(threads1))
        const resolve = build('resolve', `  let size = vec2u(_a.y, _a.z);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let k = (gid.y * size.x + gid.x) * 4u;
  let c = vec4f(f32(atomicLoad(&_acc[k])), f32(atomicLoad(&_acc[k + 1u])), f32(atomicLoad(&_acc[k + 2u])), f32(atomicLoad(&_acc[k + 3u]))) / ${FIX}.0;
  textureStore(_out, gid.xy, vec4f(c.rgb, min(c.a, 1.0)));`, [], [accumRes, {kind: 'storageTexture', name: '_out', format: 'rgba16float', texture: target.current}], [8, 8])
        steps.push(resolve.step(() => [target.width(), target.height()]))
        if (crowd && crowdCount) {
            const crowdResolve = build('crowd', `  if (gid.x >= ${CROWD}u || gid.y >= ${CROWD}u) { return; }
  let n = f32(atomicLoad(&_crowd[gid.y * ${CROWD}u + gid.x]));
  textureStore(_out, gid.xy, vec4f(n * ${(CROWD * CROWD) / Math.max(1, cap)} * ${node.neighbors!.radius * 10}, 0.0, 0.0, 1.0));`, [], [...crowdRes, {kind: 'storageTexture', name: '_out', format: 'rgba16float', texture: crowd.current}], [8, 8])
            steps.push(crowdResolve.step(() => [CROWD, CROWD]))
        }

        const argData = new Uint32Array(8)
        let first = true
        // No re-seed on resize: positions and homes are in uv, so they hold at any resolution.
        // (Re-seeding here restarted the swarm whenever the editor changed a frame's render size,
        // which it does on every select and deselect.) The splat buffer reallocates above.
        return {
            outputs: {'': target},
            usesPointer: node.forces.some((f) => 'cursor' in f) || spawn === 'atPointer',
            steps() {
                argData.set([live(), target.width(), target.height(), first ? 1 : 0])
                // The first-frame scatter only counts once the update pipeline has compiled.
                if (!seeder || seeder.ready()) first = false
                ctx.device.queue.writeBuffer(args, 0, argData)
                return steps
            },
        }
    })
}

export type {InputNode}
