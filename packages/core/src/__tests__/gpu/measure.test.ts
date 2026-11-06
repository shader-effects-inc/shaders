/**
 * Layout measurement gates — measureNode's four answer paths + out-of-flow nulls.
 * Pure CPU (no GPU device): measurement must work identically in any host.
 */
import { describe, it, expect } from 'vitest'
import { measureNode } from '../../utilities/measure'
import { registerNaturalSize } from '../../utilities/naturalSize'
import { componentDefinition as Circle } from '../../shaders/Circle'
import { componentDefinition as Ellipse } from '../../shaders/Ellipse'
import { componentDefinition as Blob } from '../../shaders/Blob'
import { componentDefinition as Trapezoid } from '../../shaders/Trapezoid'
import { componentDefinition as Saturation } from '../../shaders/Saturation'
import { componentDefinition as ImageTexture } from '../../shaders/ImageTexture'
import { componentDefinition as VideoTexture } from '../../shaders/VideoTexture'
import { componentDefinition as WebcamTexture } from '../../shaders/WebcamTexture'

const CW = 1920
const CH = 1080
// The caller hands in the definition's naturalSizeKey (the renderer holds the definition); the
// helper mirrors that by resolving the test's own imported definitions, never the registry.
const DEFS: Record<string, { naturalSizeKey?: any }> = { ImageTexture, VideoTexture, WebcamTexture }
const measure = (type: string, decl: any, props: Record<string, any>, boundingBox?: any) =>
    measureNode({ type, decl, naturalSizeKey: DEFS[type]?.naturalSizeKey, props, boundingBox, availableWidthPx: CW }, CW, CH)

describe('measureNode — propBindings sizes', () => {
    it('Circle: radius (canvas-height) → square box', () => {
        const m = measure('Circle', Circle.boundingBoxDeclaration, { radius: 0.5 })
        expect(m).toEqual({ widthPx: 0.5 * CH, heightPx: 0.5 * CH })
    })

    it('Circle: px DimensionalValue radius carries the full pixel size', () => {
        const m = measure('Circle', Circle.boundingBoxDeclaration, { radius: { value: 240, unit: 'px' } })
        expect(m).toEqual({ widthPx: 240, heightPx: 240 })
    })

    it('Ellipse: independent radiusX/radiusY (half-canvas-height on BOTH axes)', () => {
        const m = measure('Ellipse', Ellipse.boundingBoxDeclaration, { radiusX: 0.4, radiusY: 0.2 })
        expect(m).toEqual({ widthPx: 0.4 * 2 * CH, heightPx: 0.2 * 2 * CH })
    })
})

describe('measureNode — computeBounds shapes', () => {
    it('Blob: box hugs size (diameter), softness excluded by design', () => {
        const m = measure('Blob', Blob.boundingBoxDeclaration, { size: 0.25, center: { x: 0.5, y: 0.5 }, softness: 1 })
        expect(m).toEqual({ widthPx: 0.25 * 2 * CH, heightPx: 0.25 * 2 * CH })
    })

    it('Trapezoid: width is the WIDER edge (topWidth > bottomWidth)', () => {
        const m = measure('Trapezoid', Trapezoid.boundingBoxDeclaration, {
            bottomWidth: 0.2, topWidth: 0.4, height: 0.25, center: { x: 0.5, y: 0.5 }
        })
        expect(m).toEqual({ widthPx: 0.4 * 2 * CH, heightPx: 0.25 * 2 * CH })
    })

    it('Trapezoid: px bottomWidth resolves as a full pixel size', () => {
        const m = measure('Trapezoid', Trapezoid.boundingBoxDeclaration, {
            bottomWidth: { value: 900, unit: 'px' }, topWidth: 0.1, height: 0.25, center: { x: 0.5, y: 0.5 }
        })
        // 900px full width (half = 900/(2·1080) UV) vs topWidth 0.1 (216px) → bottom wins
        expect(m!.widthPx).toBeCloseTo(900)
    })
})

describe('measureNode — explicit box and media', () => {
    it('explicit authored box (both dims set) answers for any layer', () => {
        const m = measure('ImageTexture', ImageTexture.boundingBoxDeclaration, { url: 'https://x/a.jpg' }, {
            width: { value: 0.5, unit: 'uv' }, height: { value: 200, unit: 'px' }
        })
        expect(m).toEqual({ widthPx: 0.5 * CW, heightPx: 200 })
    })

    it('media natural size answers when no explicit box (the <img naturalWidth> analog)', () => {
        registerNaturalSize('https://x/photo.jpg', 800, 600)
        const m = measure('ImageTexture', ImageTexture.boundingBoxDeclaration, { url: 'https://x/photo.jpg' })
        expect(m).toEqual({ widthPx: 800, heightPx: 600 })
    })

    it('unloaded media with no box is unmeasurable (null, out of flow until it loads)', () => {
        const m = measure('ImageTexture', ImageTexture.boundingBoxDeclaration, { url: 'https://x/not-loaded.jpg' })
        expect(m).toBeNull()
    })
})

/**
 * The media tier reads the `naturalSizeKey` declaration the caller passes in, instead of the
 * hard-coded three-name switch it used to be. These assert the three declarations reproduce the old
 * behaviour exactly, and that a non-media type still answers null.
 */
describe('measureNode — declaration-driven natural-size keys', () => {
    it('VideoTexture keys on its url prop, like ImageTexture', () => {
        registerNaturalSize('https://x/clip.mp4', 1920, 1080)
        const m = measure('VideoTexture', VideoTexture.boundingBoxDeclaration, { url: 'https://x/clip.mp4' })
        expect(m).toEqual({ widthPx: 1920, heightPx: 1080 })
    })

    it('WebcamTexture keys on the fixed "webcam" key (it has no url)', () => {
        registerNaturalSize('webcam', 1280, 720)
        const m = measure('WebcamTexture', WebcamTexture.boundingBoxDeclaration, {})
        expect(m).toEqual({ widthPx: 1280, heightPx: 720 })
    })

    it('an empty or non-string url is unmeasurable rather than keyed on ""', () => {
        expect(measure('ImageTexture', ImageTexture.boundingBoxDeclaration, { url: '' })).toBeNull()
        expect(measure('ImageTexture', ImageTexture.boundingBoxDeclaration, { url: 42 })).toBeNull()
    })

    it('a shader with no naturalSizeKey is never media-measured', () => {
        registerNaturalSize('https://x/photo.jpg', 800, 600)
        // Same props as the ImageTexture case above, but Blob declares no media key.
        expect(measure('Blob', undefined, { url: 'https://x/photo.jpg' })).toBeNull()
    })

    it('the three media shaders declare their keys (the switch is gone, not just bypassed)', () => {
        expect(ImageTexture.naturalSizeKey).toEqual({ fromProp: 'url' })
        expect(VideoTexture.naturalSizeKey).toEqual({ fromProp: 'url' })
        expect(WebcamTexture.naturalSizeKey).toEqual({ fixed: 'webcam' })
    })
})

describe('measureNode — out of flow', () => {
    it('filters have no size by nature', () => {
        expect(measure('Saturation', Saturation.boundingBoxDeclaration, { intensity: 1 })).toBeNull()
    })

    it('a partial explicit box (one axis) does not answer', () => {
        const m = measure('Saturation', Saturation.boundingBoxDeclaration, {}, { width: { value: 0.5, unit: 'uv' } })
        expect(m).toBeNull()
    })
})

describe('measureNode — no registry dependency', () => {
    it('measure.ts does not import the shader registry (it pulls every shader into user bundles)', async () => {
        const { readFileSync } = await import('node:fs')
        const { resolve } = await import('node:path')
        // vitest runs from packages/core (jsdom env: import.meta.url is not a file URL)
        const src = readFileSync(resolve(process.cwd(), 'src/utilities/measure.ts'), 'utf8')
        expect(src).not.toMatch(/shaderRegistry/)
    })
})
