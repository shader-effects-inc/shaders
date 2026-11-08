import {defineShader} from "@coreroot/std"
import {call, vec4, ZERO, type Expr, type GpuFragmentParams} from "@coreroot/gpu/porters"
import {blend, noisePaints} from "@coreroot/gpu/kit"

export interface ComponentProps {
    roughness: number
    grainScale: number
    displacement: number
    seed: number
}

// std custom-tier filter: an RTT filter sampling the child at a curl-displaced UV (fiber-direction
// micro-roughness), then modulating the straight rgb by the fibrous-grain brightness.
export const componentDefinition = defineShader<ComponentProps>({
    name: "Paper",
    role: 'filter',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Applies realistic paper grain and surface roughness to child content",
    requiresRTT: true,
    requiresChild: true,
    props: {
        roughness: {
            default: 0.3,
            description: 'Surface roughness — higher values create more pronounced brightness variation',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Roughness', group: 'Texture' }
        },
        grainScale: {
            default: 1,
            description: 'Scale of the paper grain — lower = coarser, higher = finer',
            ui: { type: 'range', min: 0.1, max: 3, step: 0.01, label: 'Grain Scale', group: 'Texture' }
        },
        displacement: {
            default: 0.15,
            description: 'Surface micro-roughness — shifts pixels at grain scale like real paper fiber bumps',
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Displacement', group: 'Texture' }
        },
        seed: {
            default: 0,
            description: 'Random seed for pattern variation',
            ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Seed', group: 'Texture' }
        }
    },

    // Sample the child at a UV micro-displaced along the curl fiber flow, unpremultiply (the
    // RTT stores premultiplied alpha), then modulate the straight rgb by the fibrous-grain
    // brightness; the final pass re-premultiplies globally. The surface body stays fused —
    // one grain UV feeds both the displacement (xy) and the brightness (z).
    gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const {childNode, ctx, uniforms, convertToTexture} = params
        if (!childNode) {
            console.error('You must pass a child component into the Paper shader.')
            return ZERO
        }
        const texture = convertToTexture(childNode)
        const surface = call(noisePaints.paperSurface, 'paperSurface', [
            ctx.uv, ctx.aspect, uniforms.grainScale, uniforms.seed,
            uniforms.displacement, uniforms.roughness,
        ])
        const sampled = texture.sample(surface.member('xy'))
        const straight = call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [sampled])
        return vec4(straight.member('rgb').mul(surface.member('z')), straight.member('a'))
    }}
})

export default componentDefinition
