import {defineShader, p} from "@coreroot/std"
import {asLocal} from "@coreroot/gpu/porters"
import {
    flareFrame, lensGhost, flareHalo, flareStarburst, flareStreak, flareGlare, flareCore,
    flareComposite, additive,
} from "@coreroot/std/paint/light"
import {transformPosition} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    lightPosition: Parameters<typeof transformPosition>[0]
    intensity: number
    ghostIntensity: number
    ghostSpread: number
    ghostChroma: number
    haloIntensity: number
    haloRadius: number
    haloChroma: number
    haloSoftness: number
    starburstIntensity: number
    starburstPoints: number
    streakIntensity: number
    streakLength: number
    glareIntensity: number
    glareSize: number
    edgeFade: number
    speed: number
}

// std generator: paints from coordinates (no child), honouring the UV context when provided. The
// additive light stack (ghosts, halo, starburst, streak, glare, core) shimmers on the node's
// accumulated time.
export const componentDefinition = defineShader<ComponentProps>({
    name: "LensFlare",
    role: 'generator',
    boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
    category: "Stylize",
    description: "Realistic camera lens flare with artifacts.",
    acceptsUVContext: true,
    animatedTime: { speed: 'speed' },
    props: {
        lightPosition: {
            default: { x: 0.3, y: 0.3 },
            transform: transformPosition,
            description: 'Position of the light source',
            ui: { type: 'position', label: 'Light Position', group: 'Light' }
        },
        intensity: {
            default: 0.5,
            description: 'Master brightness of the entire lens flare effect',
            ui: { type: ['range', 'map'], min: 0, max: 2, step: 0.01, label: 'Intensity', group: 'Light' }
        },
        ghostIntensity: {
            default: 0.4,
            description: 'Brightness of internal reflection ghost discs along the flare axis',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Ghosts', group: 'Reflections' }
        },
        ghostSpread: {
            default: 0.7,
            description: 'Spacing between ghost reflections along the flare axis',
            ui: { type: 'range', min: 0.1, max: 2, step: 0.01, label: 'Ghost Spread', group: 'Reflections' }
        },
        ghostChroma: {
            default: 0.3,
            description: 'Rainbow chromatic fringing around ghost element edges',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Ghost Chroma', group: 'Reflections' }
        },
        haloIntensity: {
            default: 0.4,
            description: 'Brightness of the circular halo ring from internal reflection',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Halo', group: 'Reflections' }
        },
        haloRadius: {
            default: 0.6,
            description: 'Radius of the halo ring',
            ui: { type: 'range', min: 0.1, max: 1, step: 0.01, label: 'Halo Size', group: 'Reflections' }
        },
        haloChroma: {
            default: 0.6,
            description: 'Spectral dispersion on the halo creating rainbow color separation',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Halo Chroma', group: 'Reflections' }
        },
        haloSoftness: {
            default: 0.8,
            description: 'Thickness and softness of the halo ring',
            ui: { type: ['range', 'map'], min: 0.01, max: 3, step: 0.01, label: 'Halo Softness', group: 'Reflections' }
        },
        starburstIntensity: {
            default: 0.3,
            description: 'Brightness of diffraction spikes radiating from the light source',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Starburst', group: 'Effects' }
        },
        starburstPoints: {
            default: 6,
            description: 'Number of starburst spikes (simulates aperture blade count)',
            ui: { type: ['range', 'map'], min: 4, max: 16, step: 1, label: 'Blades', group: 'Effects' }
        },
        streakIntensity: {
            default: 0.15,
            description: 'Brightness of horizontal anamorphic light streak',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Streak', group: 'Effects' }
        },
        streakLength: {
            default: 0.5,
            description: 'Horizontal extent of the anamorphic streak',
            ui: { type: ['range', 'map'], min: 0.1, max: 1, step: 0.01, label: 'Streak Length', group: 'Effects' }
        },
        glareIntensity: {
            default: 0.2,
            description: 'Soft veiling glare that washes out contrast around the light',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Glare', group: 'Effects' }
        },
        glareSize: {
            default: 0.5,
            description: 'Size of the soft glare glow',
            ui: { type: ['range', 'map'], min: 0.1, max: 1, step: 0.01, label: 'Glare Size', group: 'Effects' }
        },
        edgeFade: {
            default: 0.2,
            description: 'How much the flare fades when the light source is near the screen edge (0 = no fade, 1 = heavy fade)',
            ui: { type: ['range', 'map'], min: 0, max: 1, step: 0.01, label: 'Edge Fade', group: 'Light' }
        },
        speed: {
            default: 0.5,
            description: 'Speed of subtle flare shimmer and starburst rotation',
            ui: { type: 'range', min: 0, max: 3, step: 0.1, label: 'Speed', group: 'Animation' }
        }
    },
    // The recipe: the shared flare frame (light position, flare axis, master fade, polar
    // coordinates about the light) is the composition root; the additive light stack — the seven
    // internal-reflection ghosts, the chromatic halo, the diffraction starburst, the anamorphic
    // streak, the veiling glare and the bright core — sums over it, faded by the composite.
    paint: (params) => {
        const flare = asLocal(flareFrame({
            position: p('lightPosition'), intensity: p('intensity'), edgeFade: p('edgeFade'),
        })(params), 'flare')

        // The seven ghosts, unrolled at the call site: disc = offset/size/bright/hollow,
        // tint = r/g/b/gPhase — the baked table IS the lens.
        const ghost = lensGhost({intensity: p('ghostIntensity'), spread: p('ghostSpread'), chroma: p('ghostChroma')})
        const ghosts = additive(
            ghost(flare, [0.15, 0.20, 0.30, 0.10], [1.0, 0.95, 0.90, 0.0], params),
            ghost(flare, [0.30, 0.12, 0.25, 0.65], [1.0, 0.85, 0.55, 2.17], params),
            ghost(flare, [0.50, 0.08, 0.35, 0.15], [0.62, 0.82, 1.0, 4.34], params),
            ghost(flare, [0.72, 0.14, 0.18, 0.75], [0.75, 1.0, 0.70, 6.51], params),
            ghost(flare, [0.95, 0.06, 0.30, 0.00], [1.0, 0.60, 0.85, 8.68], params),
            ghost(flare, [1.25, 0.10, 0.15, 0.80], [0.55, 0.88, 1.0, 10.85], params),
            ghost(flare, [1.55, 0.07, 0.20, 0.25], [1.0, 0.92, 0.60, 13.02], params),
        )

        const total = additive(
            ghosts,
            flareHalo({intensity: p('haloIntensity'), radius: p('haloRadius'), chroma: p('haloChroma'), softness: p('haloSoftness')})(flare, params),
            flareStarburst({intensity: p('starburstIntensity'), points: p('starburstPoints')})(flare, params),
            flareStreak({intensity: p('streakIntensity'), length: p('streakLength')})(flare, params),
            flareGlare({intensity: p('glareIntensity'), size: p('glareSize')})(flare, params),
            flareCore({intensity: p('intensity')})(flare, params),
        )
        return flareComposite(total, flare)
    },
})

export default componentDefinition
