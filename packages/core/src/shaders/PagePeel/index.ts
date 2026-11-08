import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {gatherStack, peelFrame, peelCompose, curlShading, curlSheen, overhangShadow, revealShadow} from "@coreroot/std/effects/blurs"

export interface ComponentProps {
    corner: string
    amount: number
    radius: number
    shading: number
    highlight: number
    highlightSoftness: number
    shadow: number
}

// The peel geometry (curl UV, fold angle, crease distances + shadow reaches) every part reads.
const peel = peelFrame({corner: p('corner'), amount: p('amount'), radius: p('radius')})

// The curl composites TWO child samples (curl UV + flat screen UV) per pixel, which a
// single-coordinate warp map can't express — hence a gather effect, not a warp.
export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "PagePeel",
    role: 'filter',
    species: 'gather',
    category: "Transitions",
    description: "Curl the content up from a corner like a peeling page",
    boundingBoxDeclaration: { aspectRatio: null },
    props: {
        corner: {
            default: 'bottom-right',
            description: "Which corner the page peels from",
            // NOTE: corner carries no `transform` — it is a cpu-only compile-time string (an inline
            // transform would make the bridge write the string into an f32 field).
            compileTime: true,
            ui: {
                type: 'select',
                options: [
                    {label: 'Top Left', value: 'top-left'},
                    {label: 'Top Right', value: 'top-right'},
                    {label: 'Bottom Left', value: 'bottom-left'},
                    {label: 'Bottom Right', value: 'bottom-right'}
                ],
                label: 'Corner',
                group: 'Effect'
            }
        },
        amount: {
            default: 0.2,
            description: "How far the peel has progressed across the page (0 = flat, 1 = fully peeled)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Amount', group: 'Effect' }
        },
        radius: {
            default: 0.2,
            description: "Tightness of the curl (fraction of the page diagonal)",
            ui: { type: ['range', 'map'], min: 0.02, max: 0.4, step: 0.01, label: 'Curl Radius', group: 'Effect' }
        },
        shading: {
            default: 0.55,
            description: "How much the underside of the curl is shaded for depth",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Shading', group: 'Effect' }
        },
        highlight: {
            default: 0.4,
            description: "Strength of the specular sheen running along the curl",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Highlight', group: 'Effect' }
        },
        highlightSoftness: {
            default: 0.2,
            description: "Width of the specular highlight band (low = tight gloss, high = broad satin)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Highlight Softness', group: 'Effect' }
        },
        shadow: {
            default: 1,
            description: "Strength of the contact shadow the curl casts on the page",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Shadow', group: 'Effect' }
        }
    },

    // The fold composite — crook shading and specular sheen on the curl, the two contact shadows
    // either side of the crease, layered back to front. Silent on a missing child, as this
    // shader has always been.
    effect: gatherStack(
        peelCompose(peel, {
            shade: curlShading(peel, {shading: p('shading')}),
            sheen: curlSheen(peel, {highlight: p('highlight'), softness: p('highlightSoftness')}),
            overhang: overhangShadow(peel, {amount: p('amount'), shadow: p('shadow')}),
            reveal: revealShadow(peel, {amount: p('amount'), shadow: p('shadow')}),
        }),
        [],
    ),
})

export default componentDefinition
