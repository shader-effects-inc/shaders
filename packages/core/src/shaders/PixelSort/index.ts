import type {GpuShaderDefinition, GpuFragmentParams, KitTexture, Expr} from "@coreroot/gpu/porters"
import {call, ZERO, createStateBuffer, createGridKernelPass, buildOddEvenSortSet} from "@coreroot/gpu/porters"
import {defineShader, schema as d} from "@coreroot/std"
import {vec2} from "@coreroot/std/math"
import {gridSim, stage, trackedViewport} from "@coreroot/std/sim/grids"
import {blend} from "@coreroot/gpu/kit"

// Fixed square working grid. Sorting happens along one axis; the result is sampled in normalised UV
// so it adapts to any canvas aspect. Higher = finer sort (and slower convergence — the "sorts bit by
// bit" feel).
const WORK = 512
const CELL_COUNT = WORK * WORK
const STATE_FORMAT = 'rgba16float' as const

export interface ComponentProps {
    radius: number
    falloff: number
    strength: number
    decay: number
    axis: string
    direction: string
}

/**
 * The kernel set at this effect's grid: the kit's odd-even transposition displacement sort
 * (luma-key prepass, brush-gated compare-swap pair, sorted-coordinate publish), with the
 * compile-time `axis`/`direction` baked in. Rebuilt per compose (cheap; recomposes on change).
 */
export const makePixelSortKernels = (vertical: boolean, direction: 1 | -1) => buildOddEvenSortSet({
    n: WORK, format: STATE_FORMAT, namePrefix: 'pixelSort', vertical, direction,
})

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "PixelSort",
    role: 'simulation',
    species: 'custom',
    category: "Interactive",
    description: "Pixels sort by brightness around the cursor and keep their sorted position, optionally decaying back over time",
    requiresRTT: true,
    requiresChild: true,
    boundingBoxDeclaration: { aspectRatio: null },
    usesPointer: true,
    props: {
        radius: {
            default: 0.4,
            description: "Size of the cursor brush that sorts pixels",
            ui: { type: ['range', 'map'], min: 0.03, max: 0.6, step: 0.01, label: 'Radius', group: 'Effect' }
        },
        falloff: {
            default: 1,
            description: "Softness of the brush edge (0 = hard circle, 1 = very soft)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Falloff', group: 'Effect' }
        },
        strength: {
            default: 0.1,
            description: "How fast pixels sort — runs more sorting passes per frame",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Strength', group: 'Effect' }
        },
        decay: {
            default: 0.1,
            description: "How quickly sorted pixels melt back to their original position (0 = permanent)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Decay', group: 'Effect' }
        },
        // axis/direction are compile-time enums read RAW from propValues (no transform — an inline
        // string transform would make the bridge write the string into an f32 field: the GradientMap
        // `palette` / transformStrokePosition trap). Mapped string→behaviour in JS below.
        axis: {
            default: "vertical",
            compileTime: true,
            description: "Axis pixels are sorted along",
            ui: {
                type: 'select',
                options: [
                    { label: 'Horizontal', value: 'horizontal' },
                    { label: 'Vertical', value: 'vertical' }
                ],
                label: 'Axis',
                group: 'Effect'
            }
        },
        direction: {
            default: "descending",
            compileTime: true,
            description: "Sort order by brightness",
            ui: {
                type: 'select',
                options: [
                    { label: 'Dark → Light', value: 'ascending' },
                    { label: 'Light → Dark', value: 'descending' }
                ],
                label: 'Direction',
                group: 'Effect'
            }
        }
    },

    // WebGPU compute: an odd-even transposition sort over a persistent per-cell source-coordinate map.
    // Two ping-pong offset buffers + a cached-luma buffer (attributeArray → createStateBuffer). Each
    // frame: a luma prepass, then `1 + round(strength·4)` compare-swap passes (parity + read/write
    // buffer alternate per pass — the CursorTrail ordered-multi-dispatch pattern), then an output pass
    // writing the sorted source coordinate into a state texture the fragment samples.
    ...gridSim((params, root) => {
        const {childNode, convertToTexture, registerComputeTexture, propValues, onCleanup} = params
        if (!childNode) return null

        const vertical = propValues.axis === 'vertical'
        const dir = propValues.direction === 'descending' ? -1 : 1

        const childTexture = convertToTexture(childNode)

        // Persistent sim state (WebGPU zero-initialises buffers → offsets start at identity 0).
        const bufferA = createStateBuffer(root, d.f32, CELL_COUNT)
        const bufferB = createStateBuffer(root, d.f32, CELL_COUNT)
        const lumaBuf = createStateBuffer(root, d.f32, CELL_COUNT)

        const stateTex = root.createTexture({size: [WORK, WORK], format: STATE_FORMAT}).$usage('storage', 'sampled')
        onCleanup(() => stateTex.destroy())
        const state = registerComputeTexture(stateTex)

        const set = makePixelSortKernels(vertical, dir)
        const lumaParams = root.createUniform(set.LumaParams)
        const swapParams = root.createUniform(set.SwapParams)

        const viewport = trackedViewport(params)

        const size: [number, number] = [WORK, WORK]
        // Luma pipeline: input RTT bound LATE (allocated after composition).
        let lumaPipeline = createGridKernelPass(root, set.key, {size})
        let lumaReady = false

        const swapBgAB = root.createBindGroup(set.swapLayout, {readBuf: bufferA, writeBuf: bufferB, lumaBuf, params: swapParams.buffer})
        const swapBgBA = root.createBindGroup(set.swapLayout, {readBuf: bufferB, writeBuf: bufferA, lumaBuf, params: swapParams.buffer})
        const swap0 = createGridKernelPass(root, set.swap0, {size})
        const swap1 = createGridKernelPass(root, set.swap1, {size})

        const outputBgA = root.createBindGroup(set.outputLayout, {srcBuf: bufferA, stateTex})
        const outputBgB = root.createBindGroup(set.outputLayout, {srcBuf: bufferB, stateTex})
        const outputPass = createGridKernelPass(root, set.output, {size})

        let frame = 0
        let passes = 1
        // Parity-alternating compare-swap chain; the persistent tick parity is harness policy.
        const sort = stage.sortPass({
            passes: () => passes,
            pass: (parity, side) => (parity ? swap1 : swap0).with(side === 'A' ? swapBgAB : swapBgBA),
        })

        return {
            outputs: {state, childTexture},
            bindInputs: (resolve) => {
                const src = resolve(childTexture.key)
                if (src) {
                    const lumaBg = root.createBindGroup(set.lumaLayout, {input: src.texture as never, lumaBuf, params: lumaParams.buffer})
                    lumaPipeline = lumaPipeline.with(lumaBg)
                    lumaReady = true
                }
            },
            clampDt: 0.05,
            stages: [
                stage.readyWhen(() => lumaReady),
                stage.values('brush+decay', (f) => {
                    const fp = f.frameParams
                    const safe = viewport.safe(fp)
                    const tracked = viewport.tracked()
                    const strength = f.num('strength', 0.1)
                    const decayProp = f.num('decay', 0.1)
                    passes = 1 + Math.round(strength * 4)
                    swapParams.write({
                        mouseX: fp.pointer?.x ?? 0.5,
                        mouseY: fp.pointer?.y ?? 0.5,
                        radius: f.num('radius', 0.4),
                        falloff: f.num('falloff', 1),
                        decay: (decayProp * f.dt * 2.0) / passes, // spread decay across passes → rate independent of strength
                        aspect: safe.width / safe.height,
                        seed: frame % 1024,
                    })
                    lumaParams.write({inputWidth: tracked.width, inputHeight: tracked.height})
                }),
                stage.cache(() => lumaPipeline),
                sort,
                stage.publish(() => outputPass.with(sort.side() === 'A' ? outputBgA : outputBgB)),
                stage.values('frameAdvance', () => { frame++ }),
            ],
        }
    }),

    gpu: {
        fragment: ({childNode, ctx, computeOutputs, propValues, convertToTexture}: GpuFragmentParams): Expr => {
            if (!childNode) return ZERO
            const state = computeOutputs?.state as KitTexture | undefined
            const childTex = computeOutputs?.childTexture as KitTexture | undefined
            if (!state || !childTex) {
                // Compute unavailable (no GPU device): passthrough, unpremultiplied.
                const tex = convertToTexture(childNode)
                return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [tex.sample(ctx.uv)])
            }
            const vertical = propValues.axis === 'vertical'
            // State stores discrete cell coords → nearest sampling. Then look up the child at the sorted UV.
            const srcCoord = state.sample(ctx.uv, 'nearestClamp').member('r')
            const childUV = vertical
                ? vec2(ctx.uv.member('x'), srcCoord)
                : vec2(srcCoord, ctx.uv.member('y'))
            return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [childTex.sample(childUV)])
        }
    }
})

export default componentDefinition
