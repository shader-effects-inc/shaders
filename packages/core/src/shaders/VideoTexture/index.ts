import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {videoMedia} from "@coreroot/std/paint/media"
import {objectFitProp} from "@coreroot/utilities/propConfigs"

export interface ComponentProps {
    url: string
    objectFit: string
    loop: boolean
}

// The video's pixel dimensions come from `textureDimensions(externalTexture)` — read on the GPU
// every frame — so there are no CPU-tracked `videoWidth`/`videoHeight`/`videoAspect` uniforms; the
// fit math derives the aspect from those dimensions. The viewport it fits into is the EFFECTIVE one
// (device px, or the bounding box's pixel size in resize-fit mode).

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "VideoTexture",
    role: 'media',
    species: 'custom',
    boundingBoxDeclaration: {aspectRatio: null, supportsResizeFit: true},
    category: "Textures",
    description: "Display a video with customizable playback and object-fit modes",
    acceptsUVContext: true,
    naturalSizeKey: {fromProp: 'url'},
    props: {
        url: {
            default: "https://shaders.com/sample.mp4",
            description: "Upload a video or provide a URL",
            ui: {
                type: 'video-upload',
                label: 'Video',
                group: 'Media'
            }
        },
        objectFit: objectFitProp(
            'fill',
            'How the video should be sized within the viewport',
            {allowNone: false},
        ),
        loop: {
            default: true,
            description: 'Loop the video playback',
            ui: {
                type: 'checkbox',
                label: 'Loop',
                group: 'Playback'
            }
        }
    },

    // Media (external-texture) generator: the shared video-element lifecycle feeding a per-frame
    // zero-copy external-texture sample through the shared object-fit surface. A blocked autoplay
    // still yields a decodable first frame, so it is reported but not fatal.
    ...videoMedia({
        src: p('url'),
        fit: p('objectFit'),
        loop: p('loop'),
        decode: 'srgb-linear-alpha-cut',
        metadataTimeoutMs: 10000,
        onError: (error, {stage}) => {
            if (stage === 'autoplay') console.warn('[VideoTexture] Autoplay failed (browser policy):', error)
            else console.error('[VideoTexture] Failed to load video:', error)
        },
    }),
})

export default componentDefinition

// WebcamTexture owns the Firefox / non-external copy-path pattern — a media texture written each
// frame via `texture.write(video)` (textures.ts createMediaTexture + per-frame `.write`), sampled
// as a regular `tex.$.<key>` texture instead of an external one. This path is Chromium-only
// (importExternalTexture, zero-copy); the copy-path fallback is intentionally omitted, and its one
// implementation site is marked in `kit/host/mediaLifecycle.ts`.
