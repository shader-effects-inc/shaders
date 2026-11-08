import {defineShader, p, crosses} from "@coreroot/std"
import {bleedWarp, kuwahara, paperGrain, wash} from "@coreroot/std/effects/stylize"
import {transformColor} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    radius: number
    bleed: number
    strength: number
    paper: number
    paperColor: Parameters<typeof transformColor>[0]
}

// The bleed frame: the time-wobbled tap position, shared by the original sample and every
// brush tap.
const wet = bleedWarp({amount: p('bleed')})

// std gather filter — painterly Kuwahara flattening over the child's render-to-texture, plus
// paper grain and a bleed-wobbled edge break. `radius` is a compileTime brush size (the tap loop
// is unrolled in JS), so the builder re-runs per radius; `bleed`/`paper`/`strength` at 0 each
// compile their work away (`recompile: crosses(0)` recompiles on the 0 crossing).
export const componentDefinition = defineShader<ComponentProps>({
    name: "Watercolor",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Painterly watercolor look — Kuwahara flattening, pigment edge darkening, paper grain and bleeding",
    props: {
        radius: {
            default: 3,
            // Loop bounds are unrolled in JS at build time, so the brush radius must recompile when it
            // changes. Hidden — fixed brush size.
            compileTime: true,
            description: "Brush size — radius of the painterly flattening",
            ui: { type: 'range', min: 1, max: 8, step: 1, label: 'Brush', group: 'Effect', hidden: true }
        },
        bleed: {
            default: 1,
            // At 0 the noise wobble is dead — the base UV collapses to the screen UV.
            recompile: crosses(0),
            description: "Amount the pigment bleeds and wobbles past hard edges",
            ui: { type: ['range', 'map'], min: 0, max: 3, step: 0.01, label: 'Bleed', group: 'Effect' }
        },
        strength: {
            default: 1,
            // At 0 the compose mixes all the way back to the child — the whole gather is dead.
            recompile: crosses(0),
            description: "Blend between the original image (0) and the full watercolor effect (1)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Strength', group: 'Effect' }
        },
        paper: {
            default: 0.35,
            // At 0 the grain multiplies to 1 and the tint weight is 0 — the grain noise is dead.
            recompile: crosses(0),
            description: "Strength of the paper grain the pigment settles into",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Paper', group: 'Effect' }
        },
        paperColor: {
            default: "#fbf7ec",
            transform: transformColor,
            description: "Tint of the paper grain",
            ui: { type: 'color', label: 'Paper Color', group: 'Colors' }
        }
    },

    // The recipe: bleed-warped taps → Kuwahara brush (4 overlapping quadrants, lowest-variance
    // mean) → paper grain → wash compose mixing back toward the original by strength.
    effect: wash({
        at: wet,
        brush: kuwahara({at: wet, radius: p('radius')}),
        grain: paperGrain({amount: p('paper')}),
        paper: p('paper'),
        paperColor: p('paperColor'),
        strength: p('strength'),
    }),
})

export default componentDefinition
