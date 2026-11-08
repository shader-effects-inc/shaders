import {defineShader, p} from "@coreroot/std"
import {gatherStack, sphereFrame, crispTap, rimLit} from "@coreroot/std/effects/blurs"
import {transformPosition, transformColor} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    radius: number
    depth: number
    center: Parameters<typeof transformPosition>[0]
    lightPosition: Parameters<typeof transformPosition>[0]
    lightIntensity: number
    lightSoftness: number
    lightColor: Parameters<typeof transformColor>[0]
}

// The sphere-bulge geometry (bulged UV + boundary coverage + surface normal) both stages read.
const sphere = sphereFrame({center: p('center'), radius: p('radius'), depth: p('depth')})

export const componentDefinition = defineShader<ComponentProps>({
    name: "Spherize",
    role: 'filter',
    species: 'gather',
    category: "Distortions",
    description: "Map content onto a 3D sphere surface with depth distortion",
    boundingBoxDeclaration: { aspectRatio: null },
    props: {
        radius: {
            default: 1,
            description: 'Radius of the sphere (1 = half viewport height)',
            ui: { type: ['range', 'map'], min: 0.1, max: 3, step: 0.1, label: 'Radius', group: 'Effect' }
        },
        depth: {
            default: 1,
            description: 'How much the sphere bulges toward viewer (0 = flat, higher = more bulge)',
            ui: { type: ['range', 'map'], min: 0, max: 3, step: 0.1, label: 'Depth', group: 'Effect' }
        },
        center: {
            default: {
                x: 0.5,
                y: 0.5
            },
            transform: transformPosition,
            description: "The center point of the sphere",
            ui: { type: 'position', label: 'Center', group: 'Position' }
        },
        lightPosition: {
            default: {
                x: 0.3,
                y: 0.3
            },
            transform: transformPosition,
            description: "Position of the specular light source",
            ui: { type: 'position', label: 'Light Position', group: 'Position' }
        },
        lightIntensity: {
            default: 0.5,
            description: "Intensity of the rim light (0 = off)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Light Intensity', group: 'Effect' }
        },
        lightSoftness: {
            default: 0.5,
            description: "Softness of the rim light falloff (0 = hard edge, 1 = soft glow)",
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Light Softness', group: 'Effect' }
        },
        lightColor: {
            default: "#ffffff",
            transform: transformColor,
            description: "Color of the specular highlight",
            ui: { type: 'color', label: 'Light Color', group: 'Effect' }
        }
    },

    // Geometry → one crisp Catmull-Rom tap at the bulged UV → the fresnel rim. No uvRemap: the
    // rim ADDS color, which a uv+mask analytic fold can't carry — so Spherize stays on the RTT
    // fragment path to keep the specular highlight.
    effect: gatherStack(
        crispTap(sphere),
        [rimLit(sphere, {
            position: p('lightPosition'),
            intensity: p('lightIntensity'),
            softness: p('lightSoftness'),
            color: p('lightColor'),
        })],
        {resultAlpha: 'straight'},
    ),
})

export default componentDefinition
