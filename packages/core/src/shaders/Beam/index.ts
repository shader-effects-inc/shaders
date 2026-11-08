import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {beam, beamPreconvertedFields} from "@coreroot/std/paint/gradients"
import {transformPosition, transformColor, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    startPosition: Parameters<typeof transformPosition>[0]
    endPosition: Parameters<typeof transformPosition>[0]
    startThickness: number
    endThickness: number
    startSoftness: number
    endSoftness: number
    insideColor: Parameters<typeof transformColor>[0],
    outsideColor: Parameters<typeof transformColor>[0],
    colorSpace: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Beam",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Textures",
    description: "A beam of light from one point to another.",
    acceptsUVContext: true,
    // The beam paint preconverts the endpoint colors on the CPU each frame (non-linear
    // color spaces) into these fields — see beamPreconvertedFields.
    extraFields: beamPreconvertedFields,
    props: {
        startPosition: {
            default: {
                x: 0.2,
                y: 0.5
            },
            transform: transformPosition,
            description: "Starting point of the beam",
            ui: { type: 'position', label: 'Start Position', group: 'Position' }
        },
        endPosition: {
            default: {
                x: 0.8,
                y: 0.5
            },
            transform: transformPosition,
            description: "Ending point of the beam",
            ui: { type: 'position', label: 'End Position', group: 'Position' }
        },
        startThickness: {
            default: 0.2,
            description: 'Thickness at the start of the beam',
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.1, label: 'Start Thickness', group: 'Effect' }
        },
        endThickness: {
            default: 0.2,
            description: 'Thickness at the end of the beam',
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.1, label: 'End Thickness', group: 'Effect' }
        },
        startSoftness: {
            default: 0.5,
            description: 'Edge softness at the start of the beam',
            ui: { type: ['range', 'map'], min: 0, max: 50, step: 0.1, label: 'Start Softness', group: 'Effect' }
        },
        endSoftness: {
            default: 0.5,
            description: 'Edge softness at the end of the beam',
            ui: { type: ['range', 'map'], min: 0, max: 20, step: 0.1, label: 'End Softness', group: 'Effect' }
        },
        insideColor: {
            default: "#FF0000",
            transform: transformColor,
            description: "Color at the center of the beam",
            ui: { type: 'color', label: 'Inside Color', group: 'Colors' }
        },
        outsideColor: {
            default: "#0000FF",
            transform: transformColor,
            description: "Color at the edges of the beam",
            ui: { type: 'color', label: 'Outside Color', group: 'Colors' }
        },
        colorSpace: {
            default: 'linear',
            transform: transformColorSpace,
            compileTime: true,
            description: 'Color space for color interpolation',
            ui: {
                type: 'select',
                options: colorSpaceOptions,
                label: 'Color Space',
                group: 'Colors'
            }
        }
    },
    paint: beam({
        from: p('startPosition'),
        to: p('endPosition'),
        thickness: {start: p('startThickness'), end: p('endThickness')},
        softness: {start: p('startSoftness'), end: p('endSoftness')},
        colors: {inside: p('insideColor'), outside: p('outsideColor')},
        space: p('colorSpace'),
    })
})

export default componentDefinition
