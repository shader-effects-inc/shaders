import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {imageMedia} from "@coreroot/std/paint/media"
import {objectFitProp} from "@coreroot/utilities/propConfigs"

export interface ComponentProps {
    url: string
    objectFit: string
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "ImageTexture",
    role: 'media',
    species: 'custom',
    boundingBoxDeclaration: {aspectRatio: null, supportsResizeFit: true},
    category: "Textures",
    description: "Display an image with customizable object-fit modes",
    acceptsUVContext: true,
    naturalSizeKey: {fromProp: 'url'},
    props: {
        url: {
            default: "https://shaders.com/sample.jpg",
            description: "Upload an image or provide a URL",
            ui: {
                type: 'image-upload',
                label: 'Image',
                group: 'Media'
            }
        },
        objectFit: objectFitProp(
            'fill',
            'How the image should be sized within the viewport',
            {allowNone: false},
        )
    },

    // Media (static texture) generator: the URL-loader lifecycle + the shared object-fit sample
    // surface. Only the error phrasing stays here — the taxonomy is this shader's.
    ...imageMedia({
        src: p('url'),
        fit: p('objectFit'),
        decode: 'srgb-linear-alpha-cut',
        label: 'ImageTexture',
        onError(error, url) {
            const isRelative = !url.startsWith('http://') && !url.startsWith('https://') && !url.startsWith('data:')
            console.error(
                `[ImageTexture] Failed to load image\n` +
                `  Requested URL: ${url}\n` +
                `  Path type: ${isRelative ? 'relative' : 'absolute'}\n` +
                `  Error:`, error
            )
        },
    }),
})

export default componentDefinition
