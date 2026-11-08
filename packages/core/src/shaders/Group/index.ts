import type {GpuShaderDefinition, GpuFragmentParams, Expr} from "@coreroot/gpu/porters"
import {ZERO} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"

export interface ComponentProps {
    // Groups don't need any props - they just combine children
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Group",
    role: 'structural',
    species: 'custom',
    category: "Utilities",
    description: "Container for organizing and composing child effects — supports flex-like flow layout (column/row stacking via the flow prop)",
    requiresChild: true,
    // A Group is an image-like viewport over its composited children: an active bounding box
    // resamples the composite INTO the box (scale/position/rotate + corner-radius clip), with
    // px x/y/w/h, origin anchors and aspect-lock — the generic (no-propBindings) bbox path.
    // Identity (zero RTT) until the box is non-default. The resample is entirely the composer's
    // boxResamplesContent path (applyBoxTransform); the fragment just returns the composite.
    boundingBoxDeclaration: { aspectRatio: null, boxResamplesContent: true },
    props: {
        // No props needed for Group
    },
    // ATOMIC-structural: Group IS the minimal compositor citizen — the flow layout, bbox
    // resample and child blending are all composer-owned, so the only shader-side code is the
    // custom tier's mandatory fragment, a child passthrough. Nothing here decomposes further.
    gpu: {fragment: ({childNode}: GpuFragmentParams): Expr => {
        // Return the composited children if present, otherwise transparent.
        return childNode ?? ZERO
    }},
})

export default componentDefinition
