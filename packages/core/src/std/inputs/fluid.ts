/**
 * `simulate.fluid`: a real incompressible fluid (stable fluids: advect, force, project) carrying
 * coloured dye. You write the forces and where dye comes in; the engine runs the solve. Read the
 * dye as `ink(uv)` and the velocity as `ink.velocity(uv)`.
 */
// Maintainer notes. Velocity, pressure and divergence share the velocity resolution; dye has its
// own. All rgba16float. Per frame: advect velocity → add forces (author hook + pointer stir) →
// boundaries → divergence → N Jacobi pressure iterations → subtract gradient → advect dye →
// inject (author hook) → drain → (OKLab) display conversion. Every pass is a raw module over a
// ping-pong pair; inline rotate steps swap the pairs between dispatches.
import type {WgslBody} from '../wgsl'
import type {ComputeStep} from '../../gpu/compute'
import {createComputeModule, createStorageTexture, destroyTexture, rewriteForModule, type ModuleResource} from './module'
import {bodyText, moduleFor, siblingReads} from './common'
import {isPropRef, withBuilder, type InputNode} from './types'
import type {PropRef} from '../values'

export interface FluidInput extends InputNode {
    readonly kind: 'simulate.fluid'
    readonly velocity: number
    readonly dye: {readonly res: number; readonly space: 'linear' | 'oklab'}
    readonly pressureIterations: number
    readonly boundary: {readonly floor: 'solid' | 'open'; readonly walls: 'solid' | 'open'}
    readonly drain: {readonly floor: FluidArg}
    readonly forces: string | null
    readonly inject: string | null
    readonly init: string | null
    readonly dissipation: FluidArg
    readonly stir: FluidArg
}

const F: GPUTextureFormat = 'rgba16float'

/** A number, or a prop read live each frame (`p('smear')`). */
type FluidArg = number | PropRef

/** WGSL for an argument: a float literal, or the prop's uniform. */
function argWgsl(arg: FluidArg, props: Readonly<Record<string, unknown>>, what: string): string {
    if (isPropRef(arg)) {
        if (!props[arg.name]) throw new Error(`simulate.fluid: ${what}: p('${arg.name}') is not a prop of this component`)
        return arg.name
    }
    const n = Number(arg)
    return Number.isInteger(n) ? `${n}.0` : String(n)
}

/**
 * A fluid solve with coloured dye.
 *
 * - `velocity` / `dye.res`: grid sizes (cells on the long side). 256 / 512 is a good start.
 * - `dye.space: 'oklab'` carries the dye in OKLab, so colours mix cleanly instead of going muddy.
 * - `forces`: WGSL returning a `vec2f` acceleration for this cell (uv per second²). It sees
 *   `cellUV`, `velocity` (this cell's), `dye` (this cell's colour), and everything a body sees.
 * - `inject`: WGSL returning a `vec4f` of dye to add this frame at this cell (premultiplied: the
 *   alpha is how much). `isNew(period)` is true on the first frame of each `period`-second slot:
 *   use it to add a drop once instead of every frame.
 * - `stir`: how hard the pointer pushes the fluid (0 turns it off; default 1).
 * - `boundary.floor: 'solid'` makes the bottom a wall dye piles up against; `drain.floor` removes
 *   that much dye per second near the floor so the canvas never fills.
 *
 * @example
 * ```ts
 * inputs: {
 *   ink: simulate.fluid({
 *     velocity: 256, dye: {res: 512, space: 'oklab'}, boundary: {floor: 'solid'}, drain: {floor: 0.05},
 *     forces: wgsl`return vec2f(0.0, gravity * dye.a);`,
 *     inject: wgsl`
 *       let id = floor(time * drops);
 *       let at = vec2f(hash11(id), 0.08);
 *       let k = select(0.0, 1.0, isNew(1.0 / drops)) * (1.0 - smoothstep(0.0, 0.03, distance(cellUV, at)));
 *       return colors(hash11(id + 1.0)) * k;
 *     `,
 *   }),
 * },
 * paint: wgsl`return ink(uv);`,
 * ```
 */
export function fluidInput(opts: {
    velocity?: number
    dye?: {res?: number; space?: 'linear' | 'oklab'}
    pressureIterations?: number
    boundary?: {floor?: 'solid' | 'open'; walls?: 'solid' | 'open'}
    drain?: {floor?: FluidArg}
    forces?: WgslBody | string
    inject?: WgslBody | string
    init?: WgslBody | string
    dissipation?: FluidArg
    stir?: FluidArg
} = {}): FluidInput {
    const node = {
        kind: 'simulate.fluid' as const,
        outputNames: ['velocity'],
        velocity: Math.max(32, Math.min(1024, Math.round(opts.velocity ?? 256))),
        dye: {res: Math.max(32, Math.min(2048, Math.round(opts.dye?.res ?? 512))), space: opts.dye?.space ?? 'linear'},
        pressureIterations: Math.max(1, Math.min(64, Math.round(opts.pressureIterations ?? 20))),
        boundary: {floor: opts.boundary?.floor ?? 'open', walls: opts.boundary?.walls ?? 'open'},
        drain: {floor: opts.drain?.floor ?? 0},
        forces: opts.forces ? bodyText(opts.forces) : null,
        inject: opts.inject ? bodyText(opts.inject) : null,
        init: opts.init ? bodyText(opts.init) : null,
        dissipation: isPropRef(opts.dissipation) ? opts.dissipation : Math.max(0, opts.dissipation ?? 0.2),
        stir: opts.stir ?? 1,
    }
    return withBuilder(node, (ctx) => {
        const canvas = ctx.canvas()
        const aspect = canvas.width / Math.max(1, canvas.height)
        const sizeFor = (long: number) => aspect >= 1
            ? {width: long, height: Math.max(16, Math.round(long / aspect))}
            : {width: Math.max(16, Math.round(long * aspect)), height: long}
        const vSize = sizeFor(node.velocity)
        const dSize = sizeFor(node.dye.res)
        const tex = (size: {width: number; height: number}, label: string) => {
            const t = createStorageTexture(ctx.params, size, F, `${ctx.name} ${label}`)
            ctx.onCleanup(() => destroyTexture(t))
            return t
        }
        const vel = [tex(vSize, 'velocity a'), tex(vSize, 'velocity b')]
        const prs = [tex(vSize, 'pressure a'), tex(vSize, 'pressure b')]
        const div = tex(vSize, 'divergence')
        const dye = [tex(dSize, 'dye a'), tex(dSize, 'dye b')]
        const display = node.dye.space === 'oklab' ? tex(dSize, 'dye display') : null
        let v = 0
        let p = 0
        let dy = 0
        const swapV: ComputeStep = () => { v = 1 - v }
        const swapP: ComputeStep = () => { p = 1 - p }
        const swapD: ComputeStep = () => { dy = 1 - dy }

        const mod = moduleFor(ctx)
        // A prop argument reads the props uniform (`_p.<name>`) live each frame.
        const arg = (a: FluidArg, what: string): string => {
            const w = argWgsl(a, ctx.props, what)
            return isPropRef(a) ? `_p.${w}` : w
        }
        const reads = siblingReads(ctx, `${node.forces ?? ''}\n${node.inject ?? ''}\n${node.init ?? ''}`)
        const solid = (side: 'floor' | 'walls') => node.boundary[side] === 'solid'
        const common = `fn _load(t: texture_2d<f32>, c: vec2i) -> vec4f { return textureLoad(t, clamp(c, vec2i(0), vec2i(textureDimensions(t)) - vec2i(1)), 0); }
fn _sample(t: texture_2d<f32>, uv: vec2f) -> vec4f { return textureSampleLevel(t, _samp, uv, 0.0); }
fn isNew(period: f32) -> bool { return fract(_f.time / max(period, 0.0001)) * period < max(_f.dt, 0.0001); }
fn _toOklab(c: vec3f) -> vec3f {
  let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
  let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
  let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
  let lc = pow(max(vec3f(l, m, s), vec3f(0.0)), vec3f(1.0 / 3.0));
  return vec3f(0.2104542553 * lc.x + 0.7936177850 * lc.y - 0.0040720468 * lc.z,
               1.9779984951 * lc.x - 2.4285922050 * lc.y + 0.4505937099 * lc.z,
               0.0259040371 * lc.x + 0.7827717662 * lc.y - 0.8086757660 * lc.z);
}
fn _fromOklab(c: vec3f) -> vec3f {
  let l = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
  let m = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
  let s = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
  let l3 = l * l * l; let m3 = m * m * m; let s3 = s * s * s;
  return vec3f(4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3,
              -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
              -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3);
}
// A stored dye cell as straight linear colour (alpha = how much dye), whatever the dye space.
fn _straight(c: vec4f) -> vec4f {
  if (c.a <= 0.0001) { return vec4f(0.0); }
  return vec4f(${node.dye.space === 'oklab' ? '_fromOklab(c.rgb / c.a)' : 'c.rgb / c.a'}, c.a);
}
// Straight linear colour to a stored dye cell.
fn _stored(c: vec4f) -> vec4f {
  let a = clamp(c.a, 0.0, 1.0);
  return vec4f(${node.dye.space === 'oklab' ? '_toOklab(c.rgb)' : 'c.rgb'} * a, a);
}`
        // Every module the solve runs: frames wait until all have compiled, since the ping-pong
        // swaps between passes still happen when a pass isn't ready, which would wipe the state.
        const modules: {ready(): boolean}[] = []
        let ready = false
        const track = <M extends {ready(): boolean}>(m: M): M => {
            modules.push(m)
            return m
        }
        const pass = (label: string, entry: string, resources: ModuleResource[], size: {width: number; height: number}, segments: {label: string; text: string; offset?: number}[] = []) =>
            track(createComputeModule(mod, {
                label: `${ctx.name}: ${label}`,
                decls: common,
                segments,
                entry: `  let gs = vec2i(${size.width}, ${size.height});
  if (i32(gid.x) >= gs.x || i32(gid.y) >= gs.y) { return; }
  let cell = vec2i(gid.xy);
  let cellUV = (vec2f(gid.xy) + 0.5) / vec2f(gs);
${entry}`,
                resources,
                textureReads: reads,
                workgroup: [8, 8],
            })).step(() => [size.width, size.height])
        const t = (name: string, get: () => unknown): ModuleResource => ({kind: 'texture', name, texture: get})
        const o = (name: string, get: () => unknown): ModuleResource => ({kind: 'storageTexture', name, format: F, texture: get})

        const advectVel = pass('advect velocity', `  let u = _load(_v, cell).xy;
  let back = cellUV - u * _f.dt;
  let w = _sample(_v, back).xy * exp(-${arg(node.dissipation, 'dissipation')} * 0.2 * _f.dt);
  textureStore(_out, gid.xy, vec4f(w, 0.0, 1.0));`, [t('_v', () => vel[v]), o('_out', () => vel[1 - v])], vSize)
        const forcesFn = node.forces
            ? [{label: `${ctx.name} forces`, offset: 1, text: rewriteForModule(`fn _forces(cell: vec2i, cellUV: vec2f, velocity: vec2f, dye: vec4f) -> vec2f {\n${node.forces}\n}`, mod, reads)}]
            : []
        const addForces = pass('forces', `  var u = _load(_v, cell).xy;
  let d = _sample(_d, cellUV);
  ${node.forces ? 'u += _forces(cell, cellUV, u, d) * _f.dt;' : ''}
  ${node.stir !== 0 ? `let pd = length((cellUV - _f.pointer) * vec2f(${aspect}, 1.0));
  u += _f.pointerVelocity * ${arg(node.stir, 'stir')} * exp(-pd * pd * 900.0) * 0.6;` : ''}
  ${solid('floor') ? 'if (cell.y >= gs.y - 1) { u.y = min(u.y, 0.0); }' : ''}
  ${solid('walls') ? 'if (cell.x <= 0) { u.x = max(u.x, 0.0); } if (cell.x >= gs.x - 1) { u.x = min(u.x, 0.0); } if (cell.y <= 0) { u.y = max(u.y, 0.0); }' : ''}
  textureStore(_out, gid.xy, vec4f(u, 0.0, 1.0));`, [t('_v', () => vel[v]), t('_d', () => dye[dy]), o('_out', () => vel[1 - v])], vSize, forcesFn)
        const divergence = pass('divergence', `  let l = _load(_v, cell - vec2i(1, 0)).x;
  let r = _load(_v, cell + vec2i(1, 0)).x;
  let b = _load(_v, cell - vec2i(0, 1)).y;
  let tp = _load(_v, cell + vec2i(0, 1)).y;
  textureStore(_out, gid.xy, vec4f((r - l + tp - b) * 0.5, 0.0, 0.0, 1.0));`, [t('_v', () => vel[v]), o('_out', () => div)], vSize)
        const jacobi = pass('pressure', `  let d = _load(_div, cell).x;
  let s = _load(_prs, cell - vec2i(1, 0)).x + _load(_prs, cell + vec2i(1, 0)).x + _load(_prs, cell - vec2i(0, 1)).x + _load(_prs, cell + vec2i(0, 1)).x;
  textureStore(_out, gid.xy, vec4f((s - d) * 0.25, 0.0, 0.0, 1.0));`, [t('_prs', () => prs[p]), t('_div', () => div), o('_out', () => prs[1 - p])], vSize)
        const subtract = pass('project', `  let u = _load(_v, cell).xy;
  let gx = _load(_prs, cell + vec2i(1, 0)).x - _load(_prs, cell - vec2i(1, 0)).x;
  let gy = _load(_prs, cell + vec2i(0, 1)).x - _load(_prs, cell - vec2i(0, 1)).x;
  textureStore(_out, gid.xy, vec4f(u - vec2f(gx, gy) * 0.5, 0.0, 1.0));`, [t('_v', () => vel[v]), t('_prs', () => prs[p]), o('_out', () => vel[1 - v])], vSize)
        const injectFn = node.inject
            ? [{label: `${ctx.name} inject`, offset: 1, text: rewriteForModule(`fn _inject(cell: vec2i, cellUV: vec2f, dye: vec4f) -> vec4f {\n${node.inject}\n}`, mod, reads)}]
            : []
        const oklab = node.dye.space === 'oklab'
        const advectDye = pass('advect dye', `  let u = _sample(_v, cellUV).xy;
  var c = _sample(_d, cellUV - u * _f.dt);
  ${node.inject ? `let add = _inject(cell, cellUV, _straight(c));
  let addC = ${oklab ? 'vec4f(_toOklab(add.rgb / max(add.a, 0.0001)) * add.a, add.a)' : 'add'};
  c = c * (1.0 - clamp(add.a, 0.0, 1.0)) + addC;` : ''}
  ${node.drain.floor !== 0 ? `let nearFloor = smoothstep(0.9, 1.0, cellUV.y);
  c *= 1.0 - clamp(${arg(node.drain.floor, 'drain.floor')} * nearFloor * _f.dt, 0.0, 1.0);` : ''}
  textureStore(_out, gid.xy, c);`, [t('_v', () => vel[v]), t('_d', () => dye[dy]), o('_out', () => dye[1 - dy])], dSize, injectFn)
        const toDisplay = display
            ? pass('display', `  let c = _load(_d, cell);
  let rgb = select(_fromOklab(c.rgb / max(c.a, 0.0001)), vec3f(0.0), c.a <= 0.0001);
  textureStore(_out, gid.xy, vec4f(max(rgb, vec3f(0.0)), clamp(c.a, 0.0, 1.0)));`, [t('_d', () => dye[dy]), o('_out', () => display)], dSize)
            : null

        // `init`: the dye's first state (a prefilled pool), written once when everything has
        // compiled. Straight linear colour in, stored in the dye's own space.
        const initStep = node.init
            ? pass('init', `  textureStore(_out, gid.xy, _stored(_init(cell, cellUV)));`, [o('_out', () => dye[dy])], dSize,
                [{label: `${ctx.name} init`, offset: 1, text: rewriteForModule(`fn _init(cell: vec2i, cellUV: vec2f) -> vec4f {\n${node.init}\n}`, mod, reads)}])
            : null
        let seeded = false

        const plan: ComputeStep[] = [advectVel, swapV, addForces, swapV, divergence]
        for (let i = 0; i < node.pressureIterations; i++) plan.push(jacobi, swapP)
        plan.push(subtract, swapV, advectDye, swapD)
        if (toDisplay) plan.push(toDisplay)

        return {
            outputs: {
                '': {current: () => display ?? dye[dy], premultiplied: !display, format: F, size: () => dSize},
                velocity: {current: () => vel[v], premultiplied: false, format: F, size: () => vSize},
            },
            usesPointer: node.stir !== 0,
            steps: () => {
                if (!ready) {
                    if (!modules.every((m) => m.ready())) return []
                    ready = true
                }
                if (!seeded) {
                    seeded = true
                    if (initStep) return [initStep, ...plan]
                }
                return plan
            },
        }
    })
}

export {}
