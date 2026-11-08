import type {GpuShaderDefinition, GpuFragmentParams, Expr} from "@coreroot/gpu/porters"
import {call, ZERO} from "@coreroot/gpu/porters"
import {defineShader, p as prop} from "@coreroot/std"
import {mirrorAcrossRow, depthRampBlur, planarReflection} from "@coreroot/std/effects/blurs"
import {blend} from "@coreroot/gpu/kit"
import {transformEdges} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    height: number
    distance: number
    falloff: number
    blur: number
    blurDistance: number
    edges: Parameters<typeof transformEdges>[0]
}

// Compute-blur tuning. halfKernel=30 → 61 taps per pass; at blur=5 (12px×5 = 60 source px)
// tap spacing stays ~2 source px.
const HALF_KERNEL = 30

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "ReflectivePlane",
    role: 'filter',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Reflective floor that mirrors the content above it",
    requiresRTT: true,
    requiresChild: true,
    props: {
        height: {
            default: 0.7,
            description: "Vertical position of the reflective surface",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Floor Height', group: 'Plane' }
        },
        distance: {
            default: 0.5,
            description: "How far below the floor the reflection remains visible before fully fading to transparent.",
            ui: { type: ['range', 'map'], min: 0.01, max: 1, step: 0.01, label: 'Reflection Distance', group: 'Plane' }
        },
        falloff: {
            default: 0.5,
            description: "Width of the fade zone, as a fraction of reflection distance.",
            ui: { type: ['range', 'map'], min: 0, max: 3, step: 0.01, label: 'Reflection Falloff', group: 'Plane' }
        },
        blur: {
            default: 3,
            description: "Maximum blur applied to the reflection far from the surface.",
            ui: { type: ['range', 'map'], min: 0, max: 5, step: 0.01, label: 'Blur', group: 'Surface' }
        },
        blurDistance: {
            default: 0.3,
            description: "How far below the surface the blur takes to ramp from sharp to maximum.",
            ui: { type: 'range', min: 0.01, max: 1, step: 0.01, label: 'Blur Distance', group: 'Surface' }
        },
        edges: {
            default: 'stretch',
            description: "How to handle reflected samples that fall outside the source content.",
            transform: transformEdges,
            compileTime: true,
            ui: {
                type: 'select',
                options: [
                    {label: 'Stretch', value: 'stretch'},
                    {label: 'Transparent', value: 'transparent'},
                    {label: 'Mirror', value: 'mirror'},
                    {label: 'Wrap', value: 'wrap'}
                ],
                label: 'Edges',
                group: 'Surface'
            }
        }
    },

    // Stage 2 (compute): the progressive blur-by-depth of the mirror image.
    ...depthRampBlur({line: prop('height'), blur: prop('blur'), blurDistance: prop('blurDistance'), halfKernel: HALF_KERNEL}),

    // The composed fragment: mirrorAcrossRow → the blurred source below the floor, sharp content
    // above it → planarReflection composite. `sharp`/`blurred` are the RTT + the compute output;
    // when compute is unavailable (no GPU root) both fall back to the sharp child RTT (reflection
    // renders without the depth-blur). Both stores are PREMULTIPLIED → unpremultiply on the way
    // out (Twirl trap #2).
    gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const {childNode, ctx, computeOutputs, convertToTexture} = params
        if (!childNode) return ZERO

        const blurred = computeOutputs?.blurredTexture as ReturnType<typeof convertToTexture> | undefined
        const sharp = computeOutputs?.childTexture as ReturnType<typeof convertToTexture> | undefined
        const source = sharp ?? convertToTexture(childNode)
        const reflectionTex = blurred ?? source

        const composed = planarReflection(
            {line: prop('height'), distance: prop('distance'), falloff: prop('falloff'), edges: prop('edges')},
            source.sample(ctx.uv),
            (uv) => reflectionTex.sample(uv),
            mirrorAcrossRow(prop('height')),
        )(params)
        return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [composed])
    }}
})

export default componentDefinition
