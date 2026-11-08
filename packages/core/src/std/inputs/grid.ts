/**
 * The grid simulation on the GPU: a field of `vec4f` cells advanced once per frame by a list of
 * steps (built-in ops and your own kernels), with named outputs derived from it. Read the field
 * by its input name (`field(uv)`) and each output by `field.slope(uv)`.
 */
// Maintainer notes. State lives in rgba16float textures (filterable, so a body samples the field
// smoothly). Four slots: `hist` (the state at the end of the frame before last: `readPrev`), `cur`
// (the state at the end of last frame, and then the latest step's output: `read`), and two work
// slots the steps ping-pong through, so neither `hist` nor this frame's starting state is
// overwritten mid-frame. At the end of a frame the starting state becomes `hist`. Each step is
// one compute module per op; inline steps move the slot indices between dispatches, so the
// module's resource getters (read at dispatch time) see the right textures. Ops are data
// (std/ops.ts); their WGSL is generated here, the one place that needs it.
import type {PropConfig} from '../../types'
import type {GridDeriveOp, GridStepOp, OpArg} from '../ops'
import type {GridSimConfig} from '../sim'
import {createComputeModule, createStorageTexture, destroyTexture, rewriteForModule} from './module'
import {bodyText, moduleFor, siblingReads} from './common'
import type {ComputeStep} from '../../gpu/compute'
import type {InputBuildContext, InputInstance, InputTexture} from './types'

/** Cell storage: 16-bit floats by default; `precision: 'full'` stores 32-bit (exact positions). */
function stateFormat(config: GridSimConfig): GPUTextureFormat {
    return config.precision === 'full' ? 'rgba32float' : 'rgba16float'
}

function argExpr(arg: OpArg | undefined, props: Readonly<Record<string, PropConfig<unknown>>>, fallback: number): string {
    if (arg === undefined) return float(fallback)
    if (typeof arg === 'number') return float(arg)
    if (!props[arg.name]) throw new Error(`p('${arg.name}') is not a prop of this component`)
    return arg.name
}

function float(n: number): string {
    const s = String(n)
    return /[.eE]/.test(s) ? s : `${s}.0`
}

/** A grid step's per-cell WGSL body (returns the cell's new vec4f). */
function stepBody(op: GridStepOp, props: Readonly<Record<string, PropConfig<unknown>>>): string {
    const lap = `(read(cell + vec2i(1, 0)) + read(cell - vec2i(1, 0)) + read(cell + vec2i(0, 1)) + read(cell - vec2i(0, 1)) - 4.0 * read(cell))`
    switch (op.kind) {
        case 'op.wave':
            return `let c = read(cell);
  let avg = (read(cell + vec2i(1, 0)).x + read(cell - vec2i(1, 0)).x + read(cell + vec2i(0, 1)).x + read(cell - vec2i(0, 1)).x) * 0.5;
  let next = (avg - readPrev(cell).x) * (1.0 - clamp(${argExpr(op.damping, props, 10)}, 0.0, 20.0) * 0.004);
  return vec4f(next, c.yzw);`
        case 'op.splat': {
            const max = float(op.amount.max ?? 2)
            return `let c = read(cell);
  let aspectFix = vec2f(f32(gridSize.x) / f32(gridSize.y), 1.0);
  let r = max(${argExpr(op.radius, props, 0.5)} * 0.05, 0.001);
  let d = length((cellUV - pointer) * aspectFix) / r;
  let speed = min(length(pointerVelocity) * dt * 20.0, ${max});
  let push = select(0.0, exp(-d * d * 4.0) * speed, d < 3.0);
  return vec4f(c.x + push, c.yzw);`
        }
        case 'op.diffuse':
            return `return read(cell) + ${lap} * clamp(${argExpr(op.rate, props, 0.5)} * dt * 4.0, 0.0, 0.24);`
        case 'op.decay':
            return `return read(cell) * exp2(-dt / max(${argExpr(op.halfLife, props, 1)}, 0.001));`
        case 'op.reactDiffuse':
            return `let c = read(cell);
  let l = ${lap};
  let a = c.x; let b = c.y;
  let k = clamp(${argExpr(op.speed, props, 1)}, 0.0, 4.0);
  let na = a + (1.0 * l.x - a * b * b + ${argExpr(op.feed, props, 0.055)} * (1.0 - a)) * k;
  let nb = b + (0.5 * l.y + a * b * b - (${argExpr(op.kill, props, 0.062)} + ${argExpr(op.feed, props, 0.055)}) * b) * k;
  return vec4f(clamp(na, 0.0, 1.0), clamp(nb, 0.0, 1.0), c.zw);`
        case 'op.automaton': {
            const hit = (list: readonly number[]) => list.length ? list.map((n) => `n == ${Math.round(n)}`).join(' || ') : 'false'
            return `let c = read(cell);
  var n = 0;
  for (var j = -1; j <= 1; j++) { for (var i = -1; i <= 1; i++) {
    if (i == 0 && j == 0) { continue; }
    n += select(0, 1, read(cell + vec2i(i, j)).x > 0.5);
  } }
  let alive = c.x > 0.5;
  let next = select(${hit(op.birth)}, ${hit(op.survive)}, alive);
  return vec4f(select(0.0, 1.0, next), c.yzw);`
        }
        case 'op.heat':
            return `let c = read(cell);
  let spread = ${lap}.x * clamp(${argExpr(op.spread, props, 0.3)} * dt * 4.0, 0.0, 0.24);
  return vec4f(max(0.0, (c.x + spread) * exp(-${argExpr(op.cool, props, 0.4)} * dt)), c.yzw);`
        case 'op.growthFront':
            return `let c = read(cell);
  let reverse = ${argExpr(op.reverse, props, 0)} > 0.5;
  let n = max(max(read(cell + vec2i(1, 0)).x, read(cell - vec2i(1, 0)).x), max(read(cell + vec2i(0, 1)).x, read(cell - vec2i(0, 1)).x));
  let m = min(min(read(cell + vec2i(1, 0)).x, read(cell - vec2i(1, 0)).x), min(read(cell + vec2i(0, 1)).x, read(cell - vec2i(0, 1)).x));
  let h = fract(sin(dot(vec2f(cell) + time * 0.123, vec2f(12.9898, 78.233))) * 43758.5453);
  let rough = mix(1.0, h * 2.0, clamp(${argExpr(op.roughness, props, 0.5)}, 0.0, 1.0));
  let chance = clamp(${argExpr(op.rate, props, 6)} * dt * rough, 0.0, 1.0);
  let grown = select(c.x, 1.0, !reverse && c.x < 0.5 && n > 0.5 && h < chance);
  let shrunk = select(grown, 0.0, reverse && c.x > 0.5 && m < 0.5 && h < chance);
  return vec4f(shrunk, c.yzw);`
        case 'op.occupancy':
            return `let c = read(cell);
  if (c.w > 0.0) { return c; }
  var claim = 0.0;
  for (var j = -1; j <= 1; j++) { for (var i = -1; i <= 1; i++) {
    let w = read(cell + vec2i(i, j)).w;
    claim = select(claim, w, claim == 0.0 && w > 0.0);
  } }
  return vec4f(c.xyz, claim);`
        case 'op.kernel':
            return op.body
    }
}

function deriveBody(op: GridDeriveOp): string {
    if (op.kind === 'op.gradient') {
        return `return vec4f((read(cell + vec2i(1, 0)).x - read(cell - vec2i(1, 0)).x) * 0.5, (read(cell + vec2i(0, 1)).x - read(cell - vec2i(0, 1)).x) * 0.5, 0.0, 1.0);`
    }
    return op.body
}

/** The cell-step prelude every grid module gets: read/readPrev/cellUV/gridSize. */
const GRID_DECLS = `fn read(c: vec2i) -> vec4f { return textureLoad(_cur, clamp(c, vec2i(0), vec2i(textureDimensions(_cur)) - vec2i(1)), 0); }
fn readPrev(c: vec2i) -> vec4f { return textureLoad(_prev, clamp(c, vec2i(0), vec2i(textureDimensions(_prev)) - vec2i(1)), 0); }
fn readLinear(p: vec2f) -> vec4f {
  let q = p * vec2f(textureDimensions(_cur)) - 0.5;
  let i = vec2i(floor(q));
  let f = q - floor(q);
  return mix(mix(read(i), read(i + vec2i(1, 0)), f.x), mix(read(i + vec2i(0, 1)), read(i + vec2i(1, 1)), f.x), f.y);
}`

export function gridSize(config: GridSimConfig): {width: number; height: number} {
    const r = config.resolution
    const [w, h] = Array.isArray(r) ? r : [r as number, r as number]
    const clampSide = (n: number) => Math.max(2, Math.min(2048, Math.round(n)))
    return {width: clampSide(w), height: clampSide(h)}
}

/** Build a grid simulation input. */
export function buildGrid(ctx: InputBuildContext, config: GridSimConfig): InputInstance | null {
    const size = gridSize(config)
    const STATE_FORMAT = stateFormat(config)
    const unfilterable = config.precision === 'full'
    const slots = [0, 1, 2, 3].map((i) => createStorageTexture(ctx.params, size, STATE_FORMAT, `${ctx.name} state ${i}`))
    ctx.onCleanup(() => slots.forEach(destroyTexture))
    // `prev` is what readPrev() sees (last frame's starting state), `cur` what read() sees.
    let prev = 0
    let cur = 1
    /** True while `init` runs (frame one, slots 0 and 1 being written). */
    let seeding = false
    let start = 1
    let out = 2
    const free = (): number[] => [0, 1, 2, 3].filter((i) => i !== prev && i !== start)
    /** Frame start: the state at the end of last frame is this frame's start; work slots reset. */
    const beginFrame = () => {
        start = cur
        const [a] = free()
        out = a
    }
    /** After a step: its output is the latest; the next step writes the other work slot. */
    const advance = () => {
        cur = out
        const [a, b] = free()
        out = cur === a ? b : a
    }
    /** Frame end: this frame's start becomes next frame's history. */
    const endFrame = () => {
        prev = start
    }
    const mod = moduleFor(ctx)
    const allCode = [
        ...config.step.map((s) => (s.kind === 'op.kernel' ? s.body : '')),
        ...Object.values(config.derive).map((d) => (d.kind === 'op.kernel' ? d.body : '')),
        config.init ? bodyText(config.init) : '',
    ].join('\n')
    const reads = siblingReads(ctx, allCode)
    const cellModule = (label: string, body: string, target: () => unknown) => {
        const fnName = `_cell_${label.replace(/\W/g, '_')}`
        return createComputeModule(mod, {
            label: `${ctx.name}: ${label}`,
            decls: GRID_DECLS,
            segments: [{
                label: `${ctx.name} ${label}`,
                text: rewriteForModule(`fn ${fnName}(cell: vec2i, cellUV: vec2f, gridSize: vec2i) -> vec4f {\n  ${body}\n}`, mod, reads),
                offset: 1,
            }],
            entry: `  let gs = vec2i(textureDimensions(_out));
  if (i32(gid.x) >= gs.x || i32(gid.y) >= gs.y) { return; }
  let cell = vec2i(gid.xy);
  let cellUV = (vec2f(gid.xy) + 0.5) / vec2f(gs);
  textureStore(_out, gid.xy, ${fnName}(cell, cellUV, gs));`,
            resources: [
                // While `init` writes the starting slots, reads point at the two work slots instead:
                // a texture can't be bound for reading and writing in the same pass.
                {kind: 'texture', name: '_cur', texture: () => slots[seeding ? 2 : cur], unfilterable},
                {kind: 'texture', name: '_prev', texture: () => slots[seeding ? 3 : prev], unfilterable},
                {kind: 'storageTexture', name: '_out', format: STATE_FORMAT, texture: target},
            ],
            textureReads: reads,
            workgroup: [8, 8],
        })
    }
    const threads = () => [size.width, size.height] as const
    const stepPlan: ComputeStep[] = []
    /** Every module the grid runs: a frame waits until all have compiled (see `steps`). */
    const modules: {ready(): boolean}[] = []
    config.step.forEach((op, i) => {
        const module = cellModule(`step ${i} (${op.kind.replace('op.', '')})`, stepBody(op, ctx.props), () => slots[out])
        modules.push(module)
        const step = module.step(threads)
        const times = op.kind === 'op.kernel' ? op.iterations : 1
        for (let k = 0; k < times; k++) stepPlan.push(step, advance)
    })
    const initModule = config.init ? cellModule('init', bodyText(config.init), () => slots[out]) : null
    if (initModule) modules.push(initModule)
    const initStep = initModule ? initModule.step(threads) : null
    let initialised = false
    let ready = false

    const derived = Object.entries(config.derive).map(([name, op]) => {
        const target = createStorageTexture(ctx.params, size, STATE_FORMAT, `${ctx.name}.${name}`)
        ctx.onCleanup(() => destroyTexture(target))
        const module = cellModule(`derive ${name}`, deriveBody(op), () => target)
        modules.push(module)
        return {name, target, step: module.step(threads)}
    })

    const field: InputTexture = {current: () => slots[cur], premultiplied: false, format: STATE_FORMAT, size: () => size}
    const outputs: Record<string, InputTexture> = {'': field}
    for (const d of derived) outputs[d.name] = {current: () => d.target, premultiplied: false, format: STATE_FORMAT, size: () => size}

    return {
        outputs,
        usesPointer: config.step.some((s) => s.kind === 'op.splat'),
        steps() {
            const steps: ComputeStep[] = []
            // Pipelines compile asynchronously. Until every one of the grid's modules can run,
            // hold the simulation still: a step that isn't ready skips its write, but the slots
            // still rotate, so the state would be swapped for an unwritten texture (wiped), and a
            // step that runs once (init) would lose its only chance.
            if (!ready) {
                if (!modules.every((m) => m.ready())) return []
                ready = true
            }
            if (!initialised) {
                initialised = true
                // Seed both the starting state and the history, so the first frame has no velocity.
                if (initStep) steps.push(() => { seeding = true; out = prev }, initStep, () => { out = cur }, initStep, () => { seeding = false })
            }
            steps.push(beginFrame, ...stepPlan, endFrame)
            for (const d of derived) steps.push(d.step)
            return steps
        },
    }
}
