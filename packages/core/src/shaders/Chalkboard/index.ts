import {defineShader, p} from "@coreroot/std"
import {chalkSketch, sobelTaps} from "@coreroot/std/effects/stylize"
import {transformColor} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    boardColor: Parameters<typeof transformColor>[0]
    chalkColor: Parameters<typeof transformColor>[0]
    edgeSensitivity: number
    edgeThickness: number
    shading: number
    hatchScale: number
    grain: number
}

// std gather filter: samples the composed child at the 8 Sobel taps + centre (straight alpha),
// luminance-reduces, then composes the chalk drawing. Multi-tap → not a uvRemap candidate.
export const componentDefinition = defineShader<ComponentProps>({
    name: "Chalkboard",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Renders content as a chalk drawing on a blackboard, with edge strokes and cross-hatch shading",
    props: {
        boardColor: {
            default: "#000000",
            transform: transformColor,
            description: "Background board color",
            ui: { type: 'color', label: 'Board', group: 'Colors' }
        },
        chalkColor: {
            default: "#eceadb",
            transform: transformColor,
            description: "Chalk stroke color",
            ui: { type: 'color', label: 'Chalk', group: 'Colors' }
        },
        edgeSensitivity: {
            default: 0.5,
            description: "How readily edges are detected and drawn as chalk strokes",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Edge Sensitivity', group: 'Effect' }
        },
        edgeThickness: {
            default: 1.5,
            description: "Thickness of the chalk outline strokes",
            ui: { type: ['range', 'map'], min: 0.5, max: 4, step: 0.1, label: 'Edge Thickness', group: 'Effect' }
        },
        shading: {
            default: 0.05,
            description: "Amount of cross-hatch shading filling darker regions",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Shading', group: 'Effect' }
        },
        hatchScale: {
            default: 16,
            description: "Spacing of the cross-hatch lines in pixels",
            ui: { type: ['range', 'map'], min: 3, max: 24, step: 0.5, label: 'Hatch Scale', group: 'Effect' }
        },
        grain: {
            default: 0.45,
            description: "Chalk dustiness / grain that breaks up the strokes",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Grain', group: 'Effect' }
        }
    },

    // The recipe: the Sobel tap ring's edge magnitude becomes the stroke outline; the compose
    // fills darker regions with cross-hatch line families and breaks the strokes with value-noise
    // dust over the board color.
    effect: chalkSketch({
        edges: sobelTaps({spacing: p('edgeThickness')}),
        sensitivity: p('edgeSensitivity'),
        hatchScale: p('hatchScale'),
        shading: p('shading'),
        grain: p('grain'),
        board: p('boardColor'),
        chalk: p('chalkColor'),
    }),
})

export default componentDefinition
