import type {GpuComputeNode, GpuFragmentParams, GpuShaderDefinition, KitTexture} from "@coreroot/gpu/porters"
import {defineStd, recompileWhen} from "@coreroot/std"
import {shapedSurface} from "@coreroot/std/paint/materials"
import {blur, effects} from "@coreroot/gpu/kit"
import {transformPosition, transformColor, transformBoolean} from "@coreroot/utilities/transformations"
import {glassShellProps} from "@coreroot/utilities/propConfigs"
import {shapeEffectBoundingBoxDeclaration} from "@coreroot/utilities/shapeEffectBounds"
import type {BoundingBoxOrigin} from "@coreroot/types"

const {applyGlassEffect} = effects.glass

// Default shape configuration
const DEFAULT_SHAPE_CONFIG = JSON.stringify({ type: 'sphere3D', radius: 0.35 })

/** Resolve the active analytic shape type: the compile-time `shapeType` prop, else the shape JSON's
 *  `type`, else circleSDF. */
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Frosted-blur prepass (Glass-only, so composed here rather than a spine axis): RTT the child
 * once — shared as the blur input AND the fragment's base/lens sample — and run kit/blur's
 * separable Gaussian at canvas resolution. The child RTT binds LATE (`bindInputs`) since the
 * pass manager allocates it after composition. The `blur` prop carries a crosses-0 recompile
 * rule so the pass's presence recomposes on toggle.
 */
function frostBlurCompute(params: GpuFragmentParams) {
    const {childNode, gpu, convertToTexture, registerComputeTexture, getCpuValue, onCleanup, onResize, dimensions} = params
    const root = gpu?.root
    const blurValue = getCpuValue('blur')
    if (!childNode || !root || (typeof blurValue === 'number' && blurValue <= 0)) return null

    const childTexture = convertToTexture(childNode)
    let curWidth = Math.max(1, Math.round(dimensions.width))
    let curHeight = Math.max(1, Math.round(dimensions.height))
    const gaussian = blur.createGaussianBlurCompute(root, null, curWidth, curHeight, onCleanup)
    const blurredTexture = registerComputeTexture(gaussian.outputTexture)
    onResize(({width, height}) => {
        curWidth = Math.max(1, Math.round(width))
        curHeight = Math.max(1, Math.round(height))
        gaussian.setInputDimensions(curWidth, curHeight)
    })
    return {
        outputs: {childTexture, blurredTexture},
        bindInputs: (resolve: (key: string) => {texture: unknown} | undefined) => {
            const src = resolve(childTexture.key)
            if (src) gaussian.setInputTexture(src.texture as never)
        },
        getComputeNodes: () => {
            // Gaussian sigma from the live blur prop, scaled to a pixel radius (blur × 2).
            const b = getCpuValue('blur')
            if (typeof b === 'number') gaussian.updateRadius(b * 2.0)
            return gaussian.computeSteps
        },
    }
}

const spine = shapedSurface({
    chord: 'firstLobe',
    gradSampler: 'same',
    bakedGradients: 'all',
    stencil: 'none',
    child: 'required',
    surface: (frame, params) =>
        applyGlassEffect(params, frame.sampler, frame.childTexture!,
            params.computeOutputs?.blurredTexture as KitTexture | undefined, {
                volumetric: frame.volumetric,
                bakedGradients: frame.bakedGradients,
            }),
})

/** The spine's volumetric-field compute merged with the frosted-blur prepass. */
const glassCompute: GpuComputeNode = (params) => {
    const fieldResult = spine.compute(params)
    const blurResult = frostBlurCompute(params)
    if (!fieldResult && !blurResult) return null
    return {
        outputs: {...blurResult?.outputs, ...fieldResult?.outputs},
        bindInputs: blurResult?.bindInputs,
        getComputeNodes: (frameParams: unknown) => {
            const nodes = [
                ...(blurResult?.getComputeNodes() ?? []),
                ...(fieldResult?.getComputeNodes(frameParams as never) ?? []),
            ]
            return nodes.length ? nodes : null
        },
    }
}

export interface ComponentProps {
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    scale: number
    rotation: number
    refraction: number
    edgeSoftness: number
    blur: number
    thickness: number
    aberration: number
    cutout: boolean
    highlight: number
    innerZoom: number
    highlightColor: Parameters<typeof transformColor>[0]
    highlightSoftness: number
    tintColor: Parameters<typeof transformColor>[0]
    tintIntensity: number
    tintPreserveLuminosity: boolean
    lightAngle: number
    fresnel: number
    fresnelSoftness: number
    fresnelColor: Parameters<typeof transformColor>[0]
    shape: string | Record<string, unknown>
    shapeSdfUrl: string
    shapeType: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineStd<ComponentProps>({
    name: "Glass",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: shapeEffectBoundingBoxDeclaration,
    category: "Shape Effects",
    description: "Optically realistic glass lens driven in a custom shape",
    requiresRTT: true,
    requiresChild: true,
    // The shape-effect spine with the Glass-specific slots:
    //  - `stencil: 'none'` + `bakedGradients: 'all'` — the lens builder samples for itself: the
    //    compute path's GRAD sampler returns the bicubic field plus its analytic derivative in
    //    `.g/.b` from ONE 16-load tap, and the flat-SVG field carries CPU-baked forward
    //    differences in the same channels, so the effect skips its two finite-difference taps
    //    (48 loads → 16). The texel-localised gradient also kills the wide smeared halos the
    //    0.01-UV FD stencil painted around chord discontinuities (overlapping lobes). The
    //    analytic path reports `bakedGradients: false` and takes its own taps.
    //  - `chord: 'firstLobe'` + `pattern: 'none'` — the chord measures only the FIRST solid lobe
    //    along the ray, so overlapping lobes (gyroscope rings, separated metaballs) don't step
    //    the field mid-surface and refraction stays clean where they cross; only Crystal reads
    //    the surface-locked pattern coords.
    // The material is the kit's golden-tested glass lens builder: sdfUV/taps → glassLensUVs (tap
    // stage) → blur/aberration tap tree → glassComposite (combine stage). The frosted-blur prepass
    // is merged into the spine's compute half below; the fragment reads the blurred buffer off
    // `computeOutputs` (and the frame's childTexture picks up the prepass's shared child RTT).
    ...spine,
    compute: glassCompute,
    props: {
        origin: {
            default: 'center',
            description: "Reference edge the center position is measured from (center default). Lets you pin the shape relative to a corner or the canvas centre.",
            ui: { type: 'origin', label: 'Origin', group: 'Position' }
        },
        center: {
            default: { x: 0.5, y: 0.5 },
            transform: transformPosition,
            description: 'Center position of the glass shape',
            ui: { type: 'position', label: 'Center', group: 'Position', units: ['%', 'px'] }
        },
        scale: {
            default: 1,
            description: 'Scale of the glass shape (1 = default size)',
            ui: { type: ['range', 'map'], min: 0.1, max: 3, step: 0.01, label: 'Scale', group: 'Position' }
        },
        rotation: {
            default: 0,
            description: 'Rotation of the glass shape in degrees',
            ui: { type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position' }
        },
        cutout: {
            default: false,
            description: 'Cut out the alpha outside the glass shape',
            transform: transformBoolean,
            // compileTime: toggling cutout while a bounding box is active changes how the
            // compositor blends this layer (replace vs Porter-Duff-over), decided at compose
            // time — so the toggle must force a recompose.
            compileTime: true,
            ui: { type: 'checkbox', label: 'Cutout', group: 'Glass' }
        },
        ...glassShellProps({
            refraction: {
                default: 1,
                description: 'Lens refraction — how aggressively the edges warp content beneath (0 = none, 1 = max)',
                ui: {max: 2},
            },
            edgeSoftness: {
                default: 0.1,
                description: 'Edge softness — higher values give a wider, softer fade at the glass boundary',
            },
        }),
        blur: {
            default: 0,
            recompile: recompileWhen((prev, next) => ((prev as number) > 0) !== ((next as number) > 0)),
            description: 'Frosted blur amount — 0 = clear glass, higher = frosted/diffuse',
            ui: { type: 'range', min: 0, max: 20, step: 0.1, label: 'Blur', group: 'Glass' }
        },
        ...glassShellProps({
            thickness: {
                transform: (v: number) => v * 0.5,
                description: 'Glass depth — how far inward from the edge the refraction extends',
            },
        }),
        aberration: {
            default: 0.5,
            recompile: recompileWhen((prev, next) => ((prev as number) > 0) !== ((next as number) > 0)),
            description: 'Chromatic aberration — splits RGB channels along the refraction vector',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Aberration', group: 'Glass' }
        },
        innerZoom: {
            default: 1,
            description: 'Inner zoom level — magnifies content seen through the glass',
            ui: { type: ['range', 'map'], min: 0.5, max: 3, step: 0.01, label: 'Inner Zoom', group: 'Glass' }
        },
        ...glassShellProps({
            lightAngle: {default: 300, description: 'Light angle in degrees', ui: {group: 'Highlight'}},
            highlight: {
                default: 0.05,
                description: 'Directional edge highlight — bright rim on the light-facing boundary',
            },
            highlightColor: {description: 'Color of the directional edge highlight and specular glint'},
            highlightSoftness: {description: 'Specular highlight softness'},
            fresnel: {description: 'Fresnel rim glow — a soft luminous halo around the glass boundary'},
            fresnelSoftness: true,
            fresnelColor: true,
        }),
        tintColor: {
            default: '#ffffff',
            transform: transformColor,
            description: 'Color tint applied to the internal directional gradient',
            ui: { type: 'color', label: 'Tint Color', group: 'Tint' }
        },
        tintIntensity: {
            default: 0,
            description: 'Intensity of the color tint applied to the glass interior',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Tint Intensity', group: 'Tint' }
        },
        tintPreserveLuminosity: {
            default: true,
            transform: transformBoolean,
            description: 'Preserve original brightness when tinting',
            ui: { type: 'checkbox', label: 'Preserve Luminosity', group: 'Tint' }
        },
        shape: {
            default: DEFAULT_SHAPE_CONFIG,
            description: 'Serialized shape configuration (JSON)',
            ui: { type: 'shape', label: 'Shape', group: 'Shape' }
        },
        shapeSdfUrl: {
            default: '',
            compileTime: true,
            description: 'URL to a pre-generated SDF .bin file — when non-empty, activates SVG mode and triggers a shader recompile'
        },
        shapeType: {
            default: '',
            compileTime: true,
            description: 'Active SDF shape type — triggers recompile when shape is switched. When empty, derived from shape JSON at mount time.'
        }
    },

})

export default componentDefinition
