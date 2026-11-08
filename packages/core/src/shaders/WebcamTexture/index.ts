import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {webcamMedia} from "@coreroot/std/paint/media"
import {objectFitProp} from "@coreroot/utilities/propConfigs"

export interface ComponentProps {
    objectFit: string
    mirror: boolean
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "WebcamTexture",
    role: 'media',
    species: 'custom',
    boundingBoxDeclaration: {aspectRatio: null, supportsResizeFit: true},
    category: "Textures",
    description: "Display a live webcam feed with customizable object-fit modes",
    acceptsUVContext: true,
    // The webcam has no URL — layout measures it under this fixed natural-size key.
    naturalSizeKey: {fixed: 'webcam'},
    props: {
        objectFit: objectFitProp(
            'cover',
            'How the webcam feed should be sized within the viewport',
            {allowNone: true},
        ),
        mirror: {
            default: true,
            description: 'Mirror the webcam feed horizontally (selfie mode)',
            ui: {
                type: 'checkbox',
                label: 'Mirror',
                group: 'Effect'
            }
        }
    },

    // Media (external-texture) generator — the exact VideoTexture path, only the source is a
    // webcam (getUserMedia), plus the runtime selfie mirror on the fitted UV. The error taxonomy
    // stays here by design: only this shader knows how to phrase a camera permission denial.
    ...webcamMedia({
        fit: p('objectFit'),
        mirror: p('mirror'),
        decode: 'srgb-linear-alpha-cut',
        constraints: {
            video: {width: {ideal: 1280}, height: {ideal: 720}, facingMode: 'user'},
            audio: false
        },
        naturalSizeKey: 'webcam',
        onError: (error) => {
            if (error instanceof DOMException) {
                if (error.name === 'NotAllowedError') console.error('[WebcamTexture] Camera permission denied by user')
                else if (error.name === 'NotFoundError') console.error('[WebcamTexture] No camera found on this device')
                else if (error.name === 'NotReadableError') console.error('[WebcamTexture] Camera is already in use by another application')
                else console.error('[WebcamTexture] Camera error:', error.message)
            } else {
                console.error('[WebcamTexture] Failed to start webcam:', error)
            }
        },
    }),
})

export default componentDefinition
