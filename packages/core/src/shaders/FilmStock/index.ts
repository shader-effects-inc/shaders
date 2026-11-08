import type {GpuShaderDefinition, GpuFragmentParams, KitTexture, Expr} from "@coreroot/gpu/porters"
import {call, ZERO} from "@coreroot/gpu/porters"
import {defineShader, crosses, p as prop, schema} from "@coreroot/std"
import {lutAtlasGrade} from "@coreroot/std/effects/color"
import {screenedBloom, frameSway} from "@coreroot/std/effects/blurs"
import {blend} from "@coreroot/gpu/kit"
import {LUT_SIZE, STOCK_OPTIONS, decodeStockLut} from "./stockLuts"

// The stock color grades are real measured film-emulation LUTs from the RawTherapee Film
// Simulation Collection by Pat David, Pavlov Dmitry and Michael Ezra (CC BY-SA 4.0,
// https://rawpedia.rawtherapee.com/Film_Simulation), downsampled one-time to 17³ in
// stockLuts.ts (whose header records each stock's source file). The LUT data remains
// CC BY-SA 4.0; our stock names are aliases for the classic stocks each LUT approximates.

export interface ComponentProps {
    stock: string
    strength: number
    halation: number
    halationRadius: number
    weave: number
}

/** Halation tint — the red-orange of emulsion backscatter (fixed; real stocks all glow warm). */
const HAL_TINT: [number, number, number] = [1.0, 0.38, 0.16]
/** Highlight-extraction threshold for halation (the red-orange emulsion backscatter). */
const HALATION_THRESHOLD = 0.62

// The recipe's three stages, as std vocabulary: gate weave (frameSway — jitter the sample UV on
// a gated clock) → LUT grade (lutAtlasGrade over the stock's measured 17³ LUT) → halation
// (screenedBloom — screen the warm highlight bloom back in). Module-level so the compute hook
// and the fragment read the SAME part.
const weaveStage = frameSway({amount: prop('weave'), field: 'weaveTime'})
const gradeStage = lutAtlasGrade({
    select: prop('stock'), strength: prop('strength'),
    size: LUT_SIZE, decode: decodeStockLut, fallback: 'portrait400', label: 'filmstock-lut',
})
const halationStage = screenedBloom({
    strength: prop('halation'), radius: prop('halationRadius'),
    tint: HAL_TINT, threshold: HALATION_THRESHOLD,
    output: 'halationTexture', extractName: 'filmStockHalationExtract',
})

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "FilmStock",
    role: 'filter',
    species: 'custom',
    boundingBoxDeclaration: {aspectRatio: null},
    category: "Adjustments",
    description: "Real analog film color from measured film-emulation LUTs — ten classic stock looks with emulsion halation and projector gate weave",
    requiresRTT: true,
    requiresChild: true,
    // Per-frame gate-weave drift accumulator (advances only while enabled) — driven by the
    // frameSway stage's onBeforeRender hook.
    extraFields: {
        weaveTime: {schema: schema.f32, initial: 0},
    },
    props: {
        stock: {
            default: 'portrait400',
            description: 'Film stock — each look is a real measured film-emulation color LUT',
            compileTime: true,
            ui: {
                type: 'select',
                options: [...STOCK_OPTIONS],
                label: 'Stock',
                group: 'Stock',
            },
        },
        strength: {
            default: 1,
            description: "How strongly the stock's color grade is applied",
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Strength', group: 'Stock'},
        },
        halation: {
            default: 0.4,
            // The screen-blend strength: slider 1 = internal 10 (the un-scaled glow was too subtle
            // to read as emulsion bloom at sane slider values).
            transform: (value: number) => value * 10,
            description: 'Strength of the red-orange highlight glow (film emulsion backscatter)',
            // Recompose only when halation crosses 0 (adds/removes the highlight-bloom compute pass).
            recompile: crosses(0),
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Halation', group: 'Halation'},
        },
        halationRadius: {
            default: 28,
            description: 'Spread of the halation glow in pixels',
            ui: {type: 'range', min: 0, max: 100, step: 1, label: 'Halation Radius', group: 'Halation'},
        },
        weave: {
            default: 0,
            description: 'Gate weave — subtle projector-like frame jitter and rotation',
            ui: {type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Gate Weave', group: 'Weave'},
        },
    },

    // Stage 3's compute half — the halation highlight-bloom pre-pass.
    compute: halationStage.compute,

    // The composed fragment: gate weave → LUT grade → halation screen. The child RTT is
    // premultiplied → recover straight alpha at the woven UV before grading.
    gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const {childNode, computeOutputs, convertToTexture} = params
        if (!childNode) {
            console.error('You must pass a child component into the FilmStock shader.')
            return ZERO
        }

        // Child RTT: the compute pass's copy when halation ran, else our own boundary (weave still needs it).
        const childTex = (computeOutputs?.childTexture as KitTexture | undefined) ?? convertToTexture(childNode)

        const wovenUv = weaveStage(params)
        const straight = call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [childTex.sample(wovenUv)])
        return halationStage.screen(gradeStage(straight, params), wovenUv, params)
    }},
})

export default componentDefinition
