import {defineShader, p} from "@coreroot/std"
import {gatherStack, childTap, shadowOffset, silhouetteCoverage, shadowComposite} from "@coreroot/std/effects/blurs"
import {transformColor, transformBoolean} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    color: Parameters<typeof transformColor>[0]
    distance: number
    angle: number
    blur: number
    intensity: number
    cutout: boolean
}

export const componentDefinition = defineShader<ComponentProps>({
    name: "DropShadow",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Adds a soft shadow behind the child content based on its alpha silhouette",
    props: {
        color: {
            default: '#000000',
            transform: transformColor,
            description: 'Shadow color',
            ui: { type: 'color', label: 'Color', group: 'Colors' }
        },
        distance: {
            default: 0.1,
            description: 'How far the shadow is offset from the content',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.005, label: 'Distance', group: 'Effect' }
        },
        angle: {
            default: 135,
            description: 'Direction the shadow is cast (compass degrees: 0=up, 90=right, 135=lower-right, 180=down)',
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Effect' }
        },
        blur: {
            default: 5,
            description: 'Shadow softness (blur radius in pixels)',
            ui: { type: ['range', 'map'], min: 0, max: 20, step: 0.5, label: 'Blur', group: 'Effect' }
        },
        intensity: {
            default: 0.5,
            description: 'Shadow intensity — how strong/visible the shadow is',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Intensity', group: 'Effect' }
        },
        cutout: {
            default: false,
            transform: transformBoolean,
            compileTime: true,
            description: 'Hide the original layer and show only the shadow',
            ui: { type: 'checkbox', label: 'Cutout', group: 'Effect' }
        }
    },

    // Blur the child's alpha silhouette at the compass-angle offset, tint it, and composite it
    // behind the child (or shadow-only in cutout mode).
    effect: gatherStack(childTap(), [
        shadowComposite({
            coverage: silhouetteCoverage({
                at: shadowOffset({angle: p('angle'), distance: p('distance')}),
                blur: p('blur'),
            }),
            color: p('color'),
            intensity: p('intensity'),
            cutout: p('cutout'),
        }),
    ]),
})

export default componentDefinition
