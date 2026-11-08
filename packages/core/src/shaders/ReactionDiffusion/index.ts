import type {GpuShaderDefinition, GpuFragmentParams, KitTexture, Expr} from "@coreroot/gpu/porters"
import {
    call, ZERO, createStateBuffer, type ComputeStep, createPingPongPair, createLateBoundChildInput,
    createGridKernelPass, buildGrayScottStep, buildScatterSeedKernel, buildPairPublishKernel,
} from "@coreroot/gpu/porters"
import {defineShader, schema as d} from "@coreroot/std"
import {
    add, clamp, cos, div, dot, gt, local, max, min, mix, mul, neg, normalize, pow, select, sin,
    smoothstep, splat3, sub, vec2, vec3, vec4,
} from "@coreroot/std/math"
import {gridSim, stage} from "@coreroot/std/sim/grids"
import {createPointerVelocityTracker} from "@coreroot/gpu/kit/host/pointer"
import {colorMixing, constants} from "@coreroot/gpu/kit"
import {transformColor, transformColorSpace, transformBoolean, colorSpaceOptions} from "@coreroot/utilities/transformations"

const cm = colorMixing

export interface ComponentProps {
    preset: string
    feed: number
    kill: number
    diffusionRatio: number
    featureSize: number
    speed: number
    brushSize: number
    brushStrength: number
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    colorC: Parameters<typeof transformColor>[0]
    contrast: number
    threshold: number
    colorSpace: string
    relief: number
    lightAngle: number
    childInfluence: number
    childContrast: number
    childThreshold: number
    childInvert: boolean
}

// ── Simulation grid ──────────────────────────────────────────────────────────────────────────
// A fixed square grid holds the two Gray-Scott chemicals U,V (vec2f per cell). The whole layer is
// filled by the sim (no shape mask). Reflective NO-FLUX boundaries at the grid edges keep it stable
// — obtained "for free" by clamp-to-edge neighbour fetch (a boundary cell's off-grid neighbour
// clamps back to itself → zero gradient). The fragment maps the square field onto the layer with a
// COVER + aspect fit so cells stay square on any aspect ratio.
// 512² gives fine detail headroom — the finest on-screen feature is bounded by the grid resolution
// (at featureSize = 1 the whole grid fills the layer), so more cells = smaller features are possible.
const N = 512
const COUNT = N * N
// Output texture the fragment samples (U in .r, V in .g). rgba16float — NOT rg16float — because
// rg16float STORAGE requires the `texture-formats-tier1` feature, which many devices lack and which
// gpu/root.ts does not request at device creation; rgba16float is a CORE storage + linear-filterable
// format (the same one CursorRipples/DataMosh use). The output kernel already writes a full
// vec4f(U,V,0,0) and the fragment reads .g, so behavior is identical (the two extra channels are
// free); the ping-pong sim STATE lives in vec2f storage BUFFERS, unaffected by this.
const STATE_FORMAT = 'rgba16float' as const

// Du folds to 1.0 in the kit's Gray-Scott step (the stability ceiling for its explicit scheme);
// dt bakes to 1.0 (evolution SPEED is the per-frame iteration count instead). Dv arrives as a
// uniform (diffusion ratio), the "character" knob.
const MAX_ITERATIONS = 16
// Fraction of cells scattered with a V seed at (re)initialisation — enough to bloom the pattern
// across the whole field quickly rather than crawling from a single point.
const SEED_DENSITY = 0.06

// ── Surface relief lighting ──────────────────────────────────────────────────────────────────────
// The V field is treated as a HEIGHT-FIELD and lit in the fragment: a normal is reconstructed from
// the local V gradient (central differences over a ~texel step), then shaded with one directional
// light (Lambert + a tight Blinn specular sheen). `relief` cross-fades flat → fully-shaded so the
// classic 2-tone ramp is exactly recovered at 0. Constants are visual-tuning only.
const RELIEF_EPS = 1.5 / N        // gradient sample step, in field-UV (≈1.5 grid cells)
const RELIEF_STRENGTH = 6.0       // height gain — how steeply the V gradient tilts the normal
const RELIEF_AMBIENT = 0.42       // valley (unlit) brightness floor
const RELIEF_LIGHT_XY = 0.85      // light direction in-plane reach (elevation is the rest)
const RELIEF_LIGHT_Z = 0.55       // light elevation above the surface
const RELIEF_SHININESS = 18.0     // specular exponent — higher = tighter sheen
const RELIEF_SPEC = 0.35          // specular sheen strength
const DEG2RAD = constants.DEG_TO_RAD

// Named Gray-Scott regimes → (feed, kill) pairs (tuned for Du=1, Dv=0.5). Selecting a preset drives
// feed/kill from here; "custom" falls through to the feed/kill sliders. Classic coral/mitosis values
// plus the mrob/pmneila xmorphia atlas points for the rest.
const PRESETS: Record<string, {feed: number; kill: number}> = {
    coral: {feed: 0.0545, kill: 0.0620},    // branching coral growth
    mitosis: {feed: 0.0367, kill: 0.0649},  // endlessly dividing cells
    spots: {feed: 0.0300, kill: 0.0620},    // stable round spots / solitons
    maze: {feed: 0.0290, kill: 0.0570},     // dense labyrinth
    worms: {feed: 0.0780, kill: 0.0610},    // wandering worms & loops
    fingerprints: {feed: 0.0370, kill: 0.0600}, // fine parallel ridges (dermatoglyphs)
    holes: {feed: 0.0390, kill: 0.0580},    // negative spots (holes / bubbles)
    solitons: {feed: 0.0740, kill: 0.0640}, // stable pulsing dots that never merge
    bubbles: {feed: 0.0980, kill: 0.0555},  // large lazy negative bubbles
    waves: {feed: 0.0140, kill: 0.0450},    // travelling waves / solitons
    flowers: {feed: 0.0620, kill: 0.06093}, // u-skate world (gliders / flowers)
}

// ── Child-driven modulation (Jason Webb "style map") ───────────────────────────────────────────────
// When a child layer is nested, its per-pixel luminance spatially biases the operating point: the
// PRESET sets the pattern's character, the CHILD sets where it grows dense vs. sparse. Brightness
// (after contrast/threshold) shifts feed up / kill down along the density axis, scaled by influence
// and the child's alpha coverage. Ranges clamp the shifted point inside the stable Gray-Scott band.
// Constants are visual-tuning only.
const CHILD_CONTRAST_MAX = 8.0   // contrast slider 0..1 → luma gain 1..9 (→ near-binary edges at 1)
const CHILD_DFEED = 0.022        // max feed shift the child can apply
const CHILD_DKILL = 0.014        // max kill shift the child can apply
const CHILD_FEED_MIN = 0.008
const CHILD_FEED_MAX = 0.11
const CHILD_KILL_MIN = 0.03
const CHILD_KILL_MAX = 0.075

// ── The kernel set: the kit's Gray-Scott stencil step (two variants — the drive variant reads the
// child RTT's luminance to spatially bias feed/kill), plus the scatter seed and the state publish. ──
const grayScott = buildGrayScottStep({n: N, namePrefix: 'reactionDiffusion'})
const grayScottChild = buildGrayScottStep({
    n: N, namePrefix: 'reactionDiffusion',
    drive: {
        contrastMax: CHILD_CONTRAST_MAX,
        dFeed: CHILD_DFEED, dKill: CHILD_DKILL,
        feedMin: CHILD_FEED_MIN, feedMax: CHILD_FEED_MAX,
        killMin: CHILD_KILL_MIN, killMax: CHILD_KILL_MAX,
    },
})
const seedSet = buildScatterSeedKernel({n: N, density: SEED_DENSITY, namePrefix: 'reactionDiffusion'})
const outputSet = buildPairPublishKernel({n: N, format: STATE_FORMAT, namePrefix: 'reactionDiffusion'})

export const reactionKernel = grayScott.kernel
export const reactionKernelChild = grayScottChild.kernel
export const seedKernel = seedSet.kernel
export const outputKernel = outputSet.kernel

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "ReactionDiffusion",
    role: 'simulation',
    species: 'custom',
    boundingBoxDeclaration: {aspectRatio: null},
    category: "Interactive",
    description: "A living Gray-Scott reaction-diffusion pattern that fills the layer and blooms wherever you drag the cursor",
    requiresRTT: false,
    requiresChild: false,
    // Works standalone, but optionally consumes a nested child (its luminance biases the sim). This
    // makes it a valid nest target in the editor without ghosting it as "needs input" when childless.
    acceptsOptionalChild: true,
    usesPointer: true,
    props: {
        preset: {
            default: 'coral',
            description: "Named Gray-Scott regime — drives the feed & kill rates for a whole family of patterns. Choose Custom to dial feed & kill by hand",
            ui: {
                type: 'select',
                options: [
                    {label: 'Coral', value: 'coral'},
                    {label: 'Mitosis', value: 'mitosis'},
                    {label: 'Spots', value: 'spots'},
                    {label: 'Maze', value: 'maze'},
                    {label: 'Worms', value: 'worms'},
                    {label: 'Fingerprints', value: 'fingerprints'},
                    {label: 'Holes', value: 'holes'},
                    {label: 'Solitons', value: 'solitons'},
                    {label: 'Bubbles', value: 'bubbles'},
                    {label: 'Waves', value: 'waves'},
                    {label: 'Flowers', value: 'flowers'},
                    {label: 'Custom', value: 'custom'},
                ],
                label: 'Pattern',
                group: 'Pattern'
            }
        },
        feed: {
            default: 0.0545,
            description: "Feed rate — how fast chemical U is replenished (used when Pattern = Custom)",
            ui: {type: 'range', min: 0.01, max: 0.1, step: 0.001, label: 'Feed', group: 'Pattern', condition: {preset: 'custom'}}
        },
        kill: {
            default: 0.062,
            description: "Kill rate — how fast chemical V is removed (used when Pattern = Custom)",
            ui: {type: 'range', min: 0.03, max: 0.07, step: 0.001, label: 'Kill', group: 'Pattern', condition: {preset: 'custom'}}
        },
        diffusionRatio: {
            default: 0.5,
            description: "Ratio of the two chemicals' diffusion rates — shifts the pattern character (lower = tighter, more filament-like)",
            ui: {type: 'range', min: 0.2, max: 0.9, step: 0.01, label: 'Diffusion Ratio', group: 'Pattern'}
        },
        featureSize: {
            default: 3,
            description: "Size of the pattern features — lower shows finer, smaller cells; higher magnifies them into larger cells",
            ui: {type: 'range', min: 1, max: 6, step: 0.1, label: 'Feature Size', group: 'Pattern'}
        },
        speed: {
            default: 6,
            description: "Simulation steps per frame — higher evolves the pattern faster",
            ui: {type: 'range', min: 1, max: MAX_ITERATIONS, step: 1, label: 'Speed', group: 'Pattern'}
        },
        brushSize: {
            default: 0.04,
            description: "Radius of the cursor brush that seeds new reagent as you drag",
            ui: {type: 'range', min: 0.01, max: 0.3, step: 0.01, label: 'Brush Size', group: 'Interaction'}
        },
        brushStrength: {
            default: 0.6,
            description: "How much reagent the cursor injects while dragging",
            ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Brush Strength', group: 'Interaction'}
        },
        colorA: {
            default: "transparent",
            transform: transformColor,
            description: "Background color (empty regions between cells)",
            ui: {type: 'color', label: 'Background', group: 'Colors'}
        },
        colorC: {
            default: "#38bdf8",
            transform: transformColor,
            description: "Accent color at the cell walls (mid pattern values)",
            ui: {type: 'color', label: 'Wall Color', group: 'Colors'}
        },
        colorB: {
            default: "#e2f5ff",
            transform: transformColor,
            description: "Color of the dense cell interiors (high pattern values)",
            ui: {type: 'color', label: 'Cell Color', group: 'Colors'}
        },
        contrast: {
            default: 0.5,
            description: "Sharpness of the cell walls — higher gives crisper, more graphic edges",
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Contrast', group: 'Colors'}
        },
        threshold: {
            default: 0.28,
            description: "Pattern level the colors are centered on — shifts the balance of background vs. cells",
            ui: {type: ['range', 'map'], min: 0.05, max: 0.6, step: 0.01, label: 'Threshold', group: 'Colors'}
        },
        colorSpace: {
            default: 'oklch',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for color interpolation',
            ui: {type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors'}
        },
        relief: {
            default: 0.4,
            description: "Treats the pattern as a raised surface and shades it with a light — higher gives a more sculpted, three-dimensional look (0 for a flat, graphic look)",
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Relief', group: 'Lighting'}
        },
        lightAngle: {
            default: 135,
            description: "Direction the surface light comes from, in degrees",
            ui: {type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Light Angle', group: 'Lighting'}
        },
        childInfluence: {
            default: 0.6,
            description: "When a child layer is nested, how strongly its brightness biases the pattern — the preset sets the character, the child steers where it grows dense vs. sparse (0 ignores the child)",
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Influence', group: 'Child'}
        },
        childContrast: {
            default: 0.5,
            description: "Contrast applied to the child's brightness before it drives the pattern — higher pushes toward a crisp two-tone split (edges), lower keeps smooth tonal shading",
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Contrast', group: 'Child'}
        },
        childThreshold: {
            default: 0.5,
            description: "Brightness level of the child that reads as neutral — shifts which tones grow the pattern vs. clear it",
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Threshold', group: 'Child'}
        },
        childInvert: {
            default: false,
            transform: transformBoolean,
            description: "Flip which tones of the child grow the pattern (bright vs. dark areas)",
            ui: {type: 'checkbox', label: 'Invert', group: 'Child'}
        }
    },

    // WebGPU compute: a fullscreen Gray-Scott reaction-diffusion sim. Ping-pong over two vec2f (U,V)
    // state buffers with a 9-point Laplacian and reflective (clamp-to-edge) grid boundaries. Per frame:
    // [seed on (re)init] → reaction × speed → output. `preset` drives feed/kill CPU-side; re-seeds on
    // first frame and on preset change. OPTIONAL CHILD: when a child layer is nested, a second kernel
    // variant reads the child's rasterised luminance (bound LATE via bindInputs, DataMosh-style) and
    // spatially biases feed/kill so the pattern grows shaped by the child. Adding/removing a child
    // changes the tree → forces a recompose → this builder re-runs with the right variant.
    ...gridSim((params, root) => {
        const {getCpuValue, registerComputeTexture, onCleanup, childNode} = params

        const hasChild = !!childNode

        const bufferA = createStateBuffer(root, d.vec2f, COUNT)
        const bufferB = createStateBuffer(root, d.vec2f, COUNT)

        const outTex = root.createTexture({size: [N, N], format: STATE_FORMAT}).$usage('storage', 'sampled')
        onCleanup(() => outTex.destroy())
        const rdTexture = registerComputeTexture(outTex)

        const variant = hasChild ? grayScottChild : grayScott
        const paramsU = root.createUniform(variant.Params)
        const size: [number, number] = [N, N]

        // The state buffers ping-pong: each reaction step reads one and writes the other, and the
        // output copy then reads whichever side holds the CURRENT state (the one just written).
        const state = createPingPongPair(bufferA, bufferB)
        const output = createGridKernelPass(root, outputSet.kernel, {size})
        const outGroup = state.perSide((srcBuf) => output.with(root.createBindGroup(outputSet.layout, {srcBuf, outTex})))

        // The seed kernel writes bufferA, so a re-seed also resets the ping-pong orientation.
        const seed = createGridKernelPass(root, seedSet.kernel, {
            size, bindGroup: root.createBindGroup(seedSet.layout, {stateBuf: bufferA}),
        })

        // Reaction pipeline — the noun's baked variant selection. The generator variant binds its
        // ping-pong groups immediately; the child variant reads the drive texture and must bind LATE
        // (its RTT texture isn't allocated until after composition), so its groups arrive via the
        // late-bound-child scaffold and the ready gate below waits for them.
        const reaction = createGridKernelPass(root, variant.kernel, {size})
        const generatorGroup = hasChild ? null : state.groups((readBuf, writeBuf) =>
            reaction.with(root.createBindGroup(grayScott.layout, {readBuf, writeBuf, params: paramsU.buffer})))
        const child = hasChild
            ? createLateBoundChildInput(params, (childTex) => state.groups((readBuf, writeBuf) =>
                reaction.with(root.createBindGroup(grayScottChild.layout, {readBuf, writeBuf, params: paramsU.buffer, driveTex: childTex} as never))))
            : null

        let needsSeed = true
        let lastPreset = ''
        // Pointer tracking with the shared teleport guard (the brush is velocity-scaled below).
        const pointer = createPointerVelocityTracker()
        let reactGroup: (() => ComputeStep) | null = null
        let iters = 1

        return {
            outputs: {rdTexture},
            ...(child ? {bindInputs: child.bindInputs} : {}),
            clampDt: 0.05, // the sim's own step is the per-frame iteration count; dt is unused
            stages: [
                stage.readyWhen(() => {
                    reactGroup = child ? child.bound() : generatorGroup
                    return !!reactGroup // wait for the late child bind
                }),
                stage.values('feed/kill+brush', (f) => {
                    const fp = f.frameParams
                    const g = f.num
                    const hgt = fp.dimensions?.height ?? 0
                    const aspect = hgt > 0 ? (fp.dimensions?.width ?? 0) / hgt : 1

                    // Preset → feed/kill (Custom falls through to the sliders). Re-seed on first frame
                    // and when the preset switches, so each named regime starts fresh.
                    const preset = (getCpuValue('preset') as string) || 'coral'
                    if (preset !== lastPreset) { lastPreset = preset; needsSeed = true }
                    const pk = PRESETS[preset]
                    const feed = pk ? pk.feed : g('feed', 0.055)
                    const kill = pk ? pk.kill : g('kill', 0.062)

                    // Map the screen pointer into field space with the SAME cover + aspect +
                    // feature-size fit the fragment uses, then into grid-cell coordinates, so the
                    // brush lands under the cursor. The child map uses the inverse (1/(s·view)) so it
                    // aligns with the display too.
                    const featureSize = Math.max(g('featureSize', 3), 0.0001)
                    const view = 1 / featureSize
                    const sx = Math.min(aspect, 1)
                    const sy = Math.min(1 / aspect, 1)
                    // The tracker applies the teleport guard: a jump the user never dragged (entering
                    // the canvas, a tab switch) zeroes dx/dy, so no blob blooms where the pointer reappeared.
                    const move = pointer.update(fp.pointer, 0.016)
                    const fieldX = (move.x - 0.5) * sx * view + 0.5
                    const fieldY = (move.y - 0.5) * sy * view + 0.5
                    const cursorSpeed = Math.sqrt(move.dx * move.dx + move.dy * move.dy)
                    // Velocity-scaled injection: a fast drag blooms strongly, a gentle graze barely
                    // tints, and a still cursor injects nothing — the brush tracks how you're moving.
                    const moving = cursorSpeed > 0.0005 && g('brushStrength', 0.6) > 0
                    const velScale = moving ? Math.min(cursorSpeed / 0.02, 1) : 0

                    const brushRadGrid = g('brushSize', 0.04) * N * view
                    const invRaw = getCpuValue('childInvert')
                    iters = Math.min(MAX_ITERATIONS, Math.max(1, Math.round(g('speed', 6))))
                    paramsU.write({
                        feed,
                        kill,
                        dv: g('diffusionRatio', 0.5),
                        brushRadSq: brushRadGrid * brushRadGrid,
                        brushStrength: g('brushStrength', 0.6),
                        cursorX: fieldX * N,
                        cursorY: fieldY * N,
                        cursorActive: velScale,
                        driveInfluence: g('childInfluence', 0.6),
                        driveContrast: g('childContrast', 0.5),
                        driveThreshold: g('childThreshold', 0.5),
                        driveInvert: (invRaw === true || invRaw === 1) ? 1 : 0,
                        driveMapX: 1 / (sx * view),
                        driveMapY: 1 / (sy * view),
                        _pad0: 0,
                        _pad1: 0,
                    })
                }),
                stage.seedOnce({pass: seed, staleWhen: () => needsSeed, onSeed: () => { needsSeed = false; state.reset() }}),
                stage.iterate({count: () => iters, step: (_i, nodes) => {
                    nodes.push(reactGroup!())
                    state.swap()
                }}),
                stage.publish(() => outGroup.read()),
            ],
        }
    }),

    gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const {ctx, computeOutputs, propValues, uniforms} = params
        const rdField = computeOutputs?.rdTexture as KitTexture | undefined
        if (!rdField) return ZERO // GPU-free (no device) → transparent.

        // Cover + aspect + feature-size fit → square cells that fill the layer. `view` zooms the
        // field (higher featureSize = show less of it = bigger features).
        const view = local(div(1, uniforms.featureSize), 'view')
        const sx = min(ctx.aspect, 1)
        const sy = min(div(1, ctx.aspect), 1)
        const sampleUV = local(vec2(
            add(mul(mul(sub(ctx.uv.member('x'), 0.5), sx), view), 0.5),
            add(mul(mul(sub(ctx.uv.member('y'), 0.5), sy), view), 0.5),
        ), 'sampleUV')
        const V = rdField.sample(sampleUV, 'linearClamp').member('y')

        // Map V through the 3-color ramp (background → wall → cell) in the chosen color space.
        // `contrast` narrows the smoothstep band around `threshold` (→ crisp cell walls);
        // `threshold` picks the V level the mid color lands on. Segment A→C covers t∈[0,0.5],
        // C→B covers t∈[0.5,1] (both equal color C at the midpoint, so the ramp is continuous).
        const w = local(max(sub(0.6, mul(uniforms.contrast, 0.55)), 0.02), 'rampW')
        const t = local(smoothstep(sub(uniforms.threshold, w), add(uniforms.threshold, w), V), 'rampT')
        const colorSpaceMode = (propValues.colorSpace as number) ?? 0
        const variant = cm.mixColorsVariants[colorSpaceMode as keyof typeof cm.mixColorsVariants] ?? cm.mixColorsLinear
        const lowSeg = call(variant, 'mixColors', [uniforms.colorA, uniforms.colorC, clamp(mul(t, 2), 0, 1)])
        const highSeg = call(variant, 'mixColors', [uniforms.colorC, uniforms.colorB, clamp(sub(mul(t, 2), 1), 0, 1)])
        const color = local(select(gt(t, 0.5), highSeg, lowSeg), 'rampColor')

        // Relief lighting: treat V as a height-field. Fetch the four neighbours (linear-sampled at
        // a sub-texel step → smooth normals, central differences → surface normal) and shade the
        // ramp color with one directional light (Lambert + a tight Blinn specular sheen).
        // relief=0 recovers the flat look exactly.
        const vL = rdField.sample(add(sampleUV, vec2(-RELIEF_EPS, 0)), 'linearClamp').member('y')
        const vR = rdField.sample(add(sampleUV, vec2(RELIEF_EPS, 0)), 'linearClamp').member('y')
        const vT = rdField.sample(add(sampleUV, vec2(0, -RELIEF_EPS)), 'linearClamp').member('y')
        const vB = rdField.sample(add(sampleUV, vec2(0, RELIEF_EPS)), 'linearClamp').member('y')
        // UV +y runs down the field, so +gy tilts "down".
        const gx = mul(sub(vR, vL), RELIEF_STRENGTH)
        const gy = mul(sub(vB, vT), RELIEF_STRENGTH)
        const n = local(normalize(vec3(neg(gx), neg(gy), 1)), 'reliefN')
        const ang = local(mul(uniforms.lightAngle, DEG2RAD), 'lightAng')
        const lightDir = local(normalize(vec3(
            mul(cos(ang), RELIEF_LIGHT_XY), mul(sin(ang), RELIEF_LIGHT_XY), RELIEF_LIGHT_Z,
        )), 'lightDir')
        const diff = clamp(dot(n, lightDir), 0, 1)
        const shade = add(RELIEF_AMBIENT, mul(1 - RELIEF_AMBIENT, diff))
        // Blinn-Phong sheen (view straight-on → half-vector is light + +Z).
        const halfVec = normalize(add(lightDir, vec3(0, 0, 1)))
        const spec = mul(pow(clamp(dot(n, halfVec), 0, 1), RELIEF_SHININESS), RELIEF_SPEC)
        const litRgb = add(mul(color.member('rgb'), shade), splat3(spec))
        const outRgb = mix(color.member('rgb'), litRgb, uniforms.relief)

        return vec4(outRgb, color.member('a'))
    }}
})

export default componentDefinition
