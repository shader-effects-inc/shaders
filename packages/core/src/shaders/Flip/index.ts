import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {flip} from "@coreroot/std/warps"
import {transformBoolean} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    flipX: boolean
    flipY: boolean
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Flip",
    role: 'warp',
    category: "Distortions",
    description: "Mirror content horizontally, vertically, or both",
    boundingBoxDeclaration: { aspectRatio: null },
    props: {
        flipX: {
            default: false,
            transform: transformBoolean,
            description: "Mirror the content horizontally (left ↔ right)",
            ui: { type: 'checkbox', label: 'Flip X', group: 'Effect' }
        },
        flipY: {
            default: false,
            transform: transformBoolean,
            description: "Mirror the content vertically (top ↔ bottom)",
            ui: { type: 'checkbox', label: 'Flip Y', group: 'Effect' }
        }
    },

    // `edges: 'none'` is the degenerate case of the warp role and Flip is its only legitimate
    // user: the flip maps [0,1] onto [0,1], so no lookup can ever land outside the source and
    // there is nothing for an edge mode to decide. The fragment samples straight and the
    // analytic path passes the coverage mask through untouched.
    map: flip({flipX: p('flipX'), flipY: p('flipY')}),
    edges: 'none',
    missingChildMessage: 'You must pass a child component into the Flip shader.',
})

export default componentDefinition
