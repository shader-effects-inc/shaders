import {defineShader, p, uniformOf} from "@coreroot/std"
import {ZERO} from "@coreroot/gpu/porters"
import {motionBlur} from "@coreroot/gpu/kit"

export interface ComponentProps {
    intensity: number
    tileCount: number
    rotation: number
    roundness: number
}

// std custom filter: a fragment + analytic-uvRemap pair. Not a warp — the fragment samples with
// Catmull-Rom and NO edge mode (plain clamp sampler) while the remap clamps via edgeClampUV, a
// combination the warp role's edge options don't spell.
export const componentDefinition = defineShader<ComponentProps>({
    name: "GlassTiles",
    role: 'filter',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Distortions",
    description: "Refraction-like distortion in a tile grid pattern",
    requiresRTT: true,
    requiresChild: true,
    props: {
        intensity: {
            default: 2.0,
            description: 'The intensity of the glass tiles effect',
            ui: { type: ['range', 'map'], min: 0, max: 10, step: 0.1, label: 'Intensity', group: 'Effect' }
        },
        tileCount: {
            default: 20,
            description: 'Number of tiles across the longest dimension',
            ui: { type: ['range', 'map'], min: 5, max: 50, step: 1, label: 'Tile Count', group: 'Effect' }
        },
        rotation: {
            default: 0,
            description: 'Rotation angle of the tile grid in degrees',
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Rotation', group: 'Effect' }
        },
        roundness: {
            default: 0,
            description: 'Makes tiles more circular instead of square',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.1, label: 'Roundness', group: 'Effect' }
        }
    },

    gpu: {
        fragment: (params) => {
            if (!params.childNode) {
                console.error('You must pass a child component into the Glass Tiles shader.')
                return ZERO
            }
            return motionBlur.glassTilesFragment({
                uv: params.ctx.uv,
                aspect: params.ctx.aspect,
                intensity: uniformOf(p('intensity'), params),
                tileCount: uniformOf(p('tileCount'), params),
                rotation: uniformOf(p('rotation'), params),
                roundness: uniformOf(p('roundness'), params),
                texture: params.convertToTexture(params.childNode),
            })
        },
        uvRemap: ({uv, mask, uniforms, aspect}) => ({
            uv: motionBlur.glassTilesRemapUV({
                uv,
                aspect,
                intensity: uniformOf(p('intensity'), {uniforms}),
                tileCount: uniformOf(p('tileCount'), {uniforms}),
                rotation: uniformOf(p('rotation'), {uniforms}),
                roundness: uniformOf(p('roundness'), {uniforms}),
            }),
            mask,
        }),
    },
})

export default componentDefinition
