import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {slicedSlide} from "@coreroot/std/warps"

export interface ComponentProps {
    progress: number
    angle: number
    sliceCount: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "SliceWipe",
    role: 'warp',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Transitions",
    description: "Slice the content into strips that slide away in alternating directions",
    props: {
        progress: {
            default: 0.5,
            description: "How far the strips have slid (0 = fully visible, 1 = fully wiped away)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Progress', group: 'Transition' }
        },
        angle: {
            default: 0,
            description: "Orientation of the strips in degrees (0 = vertical strips sliding up and down)",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Transition' }
        },
        sliceCount: {
            default: 8,
            description: "Number of strips across the frame",
            ui: { type: ['range', 'map'], min: 2, max: 60, step: 1, label: 'Slices', group: 'Transition' }
        }
    },

    // `edges: 1` bakes TRANSPARENT as a fixed mode rather than reading a prop — clipping the
    // vacated space is what makes this a wipe, so there is no user-facing choice.
    // `resample: 'bilinear'` keeps the single tap it has always used.
    map: slicedSlide({angle: p('angle'), count: p('sliceCount'), progress: p('progress')}),
    edges: 1,
    resample: 'bilinear',
    missingChildMessage: 'You must pass a child component into the Slice Wipe shader.',
})

export default componentDefinition
