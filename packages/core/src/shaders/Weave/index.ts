import {defineShader, p} from "@coreroot/std"
import {cellFrame, weaveThreads} from "@coreroot/std/paint/patterns"
import {transformColor} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    colorA: Parameters<typeof transformColor>[0]
    colorB: Parameters<typeof transformColor>[0]
    cells: number
    gap: number
    rotation: number
}

// std generator: paints from coordinates, no child consumed.
export const componentDefinition = defineShader<ComponentProps>({
    name: "Weave",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "Interlaced textile weave pattern with two thread colors going over and under each other",
    // When a UV-propagating distortion wraps this generator, the composer supplies a per-pixel
    // `uvContext` and Weave rasterises at full canvas resolution against that distorted UV.
    acceptsUVContext: true,
    props: {
        colorA: {
            default: "#c4c4c4",
            transform: transformColor,
            description: "Horizontal thread color",
            ui: { type: 'color', label: 'X Color', group: 'Colors' }
        },
        colorB: {
            default: "#4d4d4d",
            transform: transformColor,
            description: "Vertical thread color",
            ui: { type: 'color', label: 'Y Color', group: 'Colors' }
        },
        cells: {
            default: 10,
            description: "Number of threads across the shortest canvas edge",
            ui: { type: ['range', 'map'], min: 2, max: 40, step: 1, label: 'Cells', group: 'Effect' }
        },
        gap: {
            default: 0.25,
            description: "Gap between threads (0 = no gap, 0.5 = maximum gap)",
            ui: { type: ['range', 'map'], min: 0, max: 0.45, step: 0.01, label: 'Gap', group: 'Effect' }
        },
        rotation: {
            default: 0,
            description: "Rotation of the weave pattern in degrees",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Rotation', group: 'Effect' }
        }
    },

    // The recipe: a plain lattice, interlaced horizontal/vertical thread bands drawn in it with
    // a checkerboard over-under rule, composited by per-thread alpha weight.
    paint: weaveThreads({
        frame: cellFrame({cells: p('cells'), rotation: p('rotation'), convention: 'plain'}),
        gap: p('gap'),
        colors: [p('colorA'), p('colorB')],
    })
})

export default componentDefinition
