import type {GpuShaderDefinition, GpuFragmentParams, KitTexture, Expr} from "@coreroot/gpu/porters"
import {asLocal, call, vec4} from "@coreroot/gpu/porters"
import {defineShader, schema} from "@coreroot/std"
import {orientedBoxField} from "@coreroot/std/effects/stylize"
import {sampling, tgpu, d, std} from "@coreroot/gpu/kit"
import {glyphAtlasFor, type GlyphAtlas} from "@coreroot/gpu/kit/host/glyphAtlas"
import {createCanvasRasterTarget, createFontDependentRaster, createRasterInvalidator} from "@coreroot/gpu/kit/host/canvasRaster"
import type {BoundingBoxOrigin, DimensionalString, DimensionalValue} from "@coreroot/types"
import {transformColor, transformPosition} from "@coreroot/utilities/transformations"
import {loadGoogleFont} from "@coreroot/utilities/fonts"
import {bumpMeasureGeneration, measureTextBlock, supportsCanvasLetterSpacing, type TextTransformMode} from "@coreroot/utilities/textMeasure"
import {typographyBounds, typographyProps} from "@coreroot/utilities/typography"

/** The shared typography prop definitions, with the Text component's defaults. */
const TYPOGRAPHY = typographyProps()

export interface ComponentProps {
    text: string
    fontFamily: string
    fontWeight: number
    italic: boolean
    fontSize: number | DimensionalValue | DimensionalString
    letterSpacing: number
    lineHeight: number
    textAlign: string
    width: number | DimensionalValue | DimensionalString
    textTransform: string
    color: Parameters<typeof transformColor>[0]
    origin: BoundingBoxOrigin
    center: Parameters<typeof transformPosition>[0]
    rotation: number
}

const MAX_TEXTURE_AXIS = 4096
/**
 * Raster density over the canvas's own pixel density. The sampler area-filters it back down
 * (exact coverage at any sub-pixel position or rotation), so 2× is all a static layer needs;
 * the headroom also keeps text sharp under a distortion that magnifies it up to 2×, past which
 * the sampler switches to a cubic reconstruction.
 */
const TEXT_SUPERSAMPLE = 2
/**
 * Edge contrast. Coverage is blended in linear light, where antialiased edges of dark text on a
 * light ground come out thin and grey and light text on dark comes out heavy — browsers blend
 * text closer to gamma space. The tint reshapes coverage toward that: dark text gets
 * 1 − (1 − a)^γ (fuller edges), light text a^γ (leaner), mid tones in between. γ = 1 is off;
 * 2.2 matches gamma-space blending against the opposite extreme exactly.
 */
const TEXT_EDGE_GAMMA = 1.8
/** 1 / 2.2, pre-folded (no Math in a 'use gpu' body): linear luminance → perceived lightness. */
const INV_DISPLAY_GAMMA = 0.4545454545

// ═══════════════════════════════════════════════════════════════════════════════════════
// Host recipe: the glyph atlas — raster target + swappable glyph texture + font lifecycle +
// re-raster invalidation. Pure CPU machinery; the GPU tail below only samples its texture.
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * Create the glyph raster host: measure the text block, draw white glyphs into a supersampled
 * canvas raster, upload it into a swappable media texture (a size change swaps in a new texture —
 * the pass manager rebuilds the sampling bind group, no recompose), publish the padded raster box
 * half-extents through the `halfW`/`halfH` extraFields, raster with the fallback font immediately
 * and re-raster when the real Google font lands, and re-raster on any glyph-affecting prop or
 * canvas-size change (color/center/rotation are pure uniforms — they never re-raster).
 */
function createGlyphRasterHost(params: GpuFragmentParams): GlyphAtlas {
    const {dimensions, getCpuValue, setExtraField} = params

    // ── Glyph raster target ───────────────────────────────────────────────
    // 3× supersampling for crisp glyph edges, degrading (never failing) past 4096 per axis —
    // see the raster note below for why the headroom matters.
    const target = createCanvasRasterTarget(params, {maxAxis: MAX_TEXTURE_AXIS, supersample: TEXT_SUPERSAMPLE})
    const rasterCtx = target.context()

    // This layer's raster lives in the composition's shared glyph atlas (one texture, one
    // binding for every Text layer); the atlas reports where, as the `glyphRect` extraField.
    const atlas = glyphAtlasFor(params)
    const slot = atlas.slot(rect => setExtraField('glyphRect', rect))

    const defaults = componentDefinition.props
    const readProps = () => ({
        text: String(getCpuValue('text') ?? defaults.text.default),
        fontFamily: String(getCpuValue('fontFamily') ?? defaults.fontFamily.default),
        fontWeight: Number(getCpuValue('fontWeight') ?? defaults.fontWeight.default),
        italic: Number(getCpuValue('italic') ?? 0) > 0.5,
        // Dimensional (px) fontSize is resolved to a UV number before uniforms are built
        fontSizeUV: Number(getCpuValue('fontSize') ?? defaults.fontSize.default),
        letterSpacing: Number(getCpuValue('letterSpacing') ?? defaults.letterSpacing.default),
        lineHeight: Number(getCpuValue('lineHeight') ?? defaults.lineHeight.default) || defaults.lineHeight.default,
        textAlign: String(getCpuValue('textAlign') ?? defaults.textAlign.default),
        // Dimensional (px) width is resolved to a canvas-width UV number before uniforms are built
        widthUV: Number(getCpuValue('width') ?? defaults.width.default),
        textTransform: String(getCpuValue('textTransform') ?? defaults.textTransform.default) as TextTransformMode
    })

    const raster = () => {
        if (!rasterCtx || target.disposed) return
        const p = readProps()
        const dimW = dimensions.width || 1
        const dimH = dimensions.height || 1

        const fontSizePx = p.fontSizeUV * dimH
        const wrapWidthPx = p.widthUV * dimW

        if (!p.text.trim() || fontSizePx <= 0) {
            slot.clear()
            setExtraField('halfW', 0.001)
            setExtraField('halfH', 0.001)
            return
        }

        // Block measurement: explicit \n breaks + optional wrapping at the authored width
        // (expressed at the 100px reference so wrap points are scale-invariant — see
        // measureTextBlock). Single-line text with no width reduces to the old behaviour.
        const m = measureTextBlock({
            text: p.text,
            fontFamily: p.fontFamily,
            fontWeight: p.fontWeight,
            italic: p.italic,
            letterSpacingEm: p.letterSpacing,
            textTransform: p.textTransform
        }, p.lineHeight, wrapWidthPx > 0 ? wrapWidthPx * 100 / fontSizePx : 0)

        const f = fontSizePx / 100
        const contentWPx = Math.max(1, (wrapWidthPx > 0 ? Math.max(wrapWidthPx * 100 / fontSizePx, m.widthPer100) : m.widthPer100) * f)
        const pitchPx = m.pitchPer100 * f
        const lineBoxPx = Math.max(1, fontSizePx * p.lineHeight)
        // Font-box block height: ascent + (n−1)·pitch + descent, min one line box.
        const blockHPx = Math.max(lineBoxPx, m.blockHeightPer100 * f)
        const ascentPx = m.ascentPer100 * f
        const descentPx = m.descentPer100 * f
        // Padding for italic overhangs and ascenders/descenders that escape a tight line box
        const padPx = Math.max(0.25 * fontSizePx, (ascentPx + descentPx - lineBoxPx) / 2 + 0.25 * fontSizePx)

        const cssW = contentWPx + padPx * 2
        const cssH = blockHPx + padPx * 2

        // Supersampled for the sampler's area filter; degrade instead of failing past 4096/axis
        const {scale, texW, texH} = target.rasterSize(cssW, cssH)

        target.resize(texW, texH)
        rasterCtx.fillStyle = '#ffffff'
        rasterCtx.font = `${p.italic ? 'italic ' : ''}${p.fontWeight} ${fontSizePx * scale}px "${p.fontFamily}", sans-serif`
        if (supportsCanvasLetterSpacing()) {
            ;(rasterCtx as any).letterSpacing = `${p.letterSpacing * fontSizePx * scale}px`
        }
        rasterCtx.textAlign = 'left'
        rasterCtx.textBaseline = 'alphabetic'
        // Centre the font-box BLOCK on the canvas centre (single line: identical to the old
        // optical centring), then lay baselines at `lineHeight` pitch. Horizontal placement
        // per line by textAlign within the content width.
        const blockTopY = texH / 2 - (blockHPx / 2) * scale
        const contentLeftX = (texW - contentWPx * scale) / 2
        for (let i = 0; i < m.lines.length; i++) {
            const line = m.lines[i]
            if (!line.trim()) continue
            const lineWPx = m.lineWidthsPer100[i] * f
            const drawX = p.textAlign === 'left' ? contentLeftX
                : p.textAlign === 'right' ? contentLeftX + (contentWPx - lineWPx) * scale
                : contentLeftX + ((contentWPx - lineWPx) / 2) * scale
            const baselineY = blockTopY + (ascentPx + i * pitchPx) * scale
            rasterCtx.fillText(line, drawX, baselineY)
        }

        const canvas = target.canvas()
        if (canvas) slot.set(canvas, texW, texH)

        setExtraField('halfW', Math.max(1e-4, (cssW / 2) / dimH))
        setExtraField('halfH', Math.max(1e-4, (cssH / 2) / dimH))
    }

    // ── Font loading: raster with the fallback immediately, re-raster when loaded ──
    const font = createFontDependentRaster({
        key: () => {
            const p = readProps()
            return `${p.fontFamily}|${p.fontWeight}|${p.italic}`
        },
        load: () => {
            const p = readProps()
            return loadGoogleFont(p.fontFamily, p.fontWeight, p.italic)
        },
        onLoaded: () => {
            // measureText keys on font availability, but bump the local epoch too so a
            // same-availability edge (e.g. fonts cleared) can't serve stale metrics
            bumpMeasureGeneration()
            raster()
        },
    })

    // Wait a tick for uniforms to populate (Ascii pattern), then raster + load the real font
    setTimeout(() => {
        raster()
        font.ensure()
    }, 0)

    // Re-raster when any glyph-affecting prop or the canvas size changes. Color, center and
    // rotation are pure uniforms — they never re-raster.
    createRasterInvalidator(params, {
        key: () => {
            const p = readProps()
            return `${p.text}|${p.fontFamily}|${p.fontWeight}|${p.italic}|${p.fontSizeUV}|${p.letterSpacing}|${p.lineHeight}|${p.textAlign}|${p.widthUV}|${p.textTransform}|${dimensions.width}x${dimensions.height}|${params.canvas.width}`
        },
        run: () => {
            raster()
            font.ensure()
        },
    })

    return atlas
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// GPU tail parts: fit frame → sample → post.
// ═══════════════════════════════════════════════════════════════════════════════════════

/** Fit frame: the oriented-box placement field (box-local UV + inside mask) from the raster's
 *  extraFields half-extents and the center/rotation uniforms. */
const glyphFrame = ({uniforms, ctx, uvContext}: GpuFragmentParams): Expr =>
    asLocal(call(orientedBoxField, 'orientedBoxField', [uvContext ?? ctx.uv, ctx.aspect, uniforms.center, uniforms.rotation, uniforms.halfW, uniforms.halfH]), 'glyphFrame')

/**
 * Sample: glyph coverage at the frame's box-local UV, mapped into this layer's rectangle of the
 * shared atlas (`glyphRect` = left, top, width, height in atlas UV). NO V-flip: the atlas is
 * written from the raster canvas via `copyExternalImageToTexture` (top-left origin), so the UV
 * maps straight through. The coverage sampler area-filters the supersampled raster (crisp, exact
 * edges), switches to Catmull-Rom only where a distortion magnifies the glyphs, and reads nothing
 * outside the glyph box.
 */
const glyphCoverage = (atlas: KitTexture, frame: Expr, rect: Expr): Expr =>
    sampling.sampleCoverageExpr(atlas, rect.member('xy').add(frame.member('xy').mul(rect.member('zw'))), frame.member('z'))

/**
 * Post: tint via the color uniform (so color edits never re-raster), with edge contrast matched
 * to how browsers blend text (TEXT_EDGE_GAMMA): dark text fuller, light text leaner, weighted by
 * the colour's perceived lightness. The coverage is already zero outside the glyph box.
 */
const shapeCoverage = tgpu.fn([d.f32, d.vec3f], d.f32)((coverage, rgb) => {
    'use gpu'
    const lightness = std.pow(std.saturate(std.dot(rgb, d.vec3f(0.2126, 0.7152, 0.0722))), INV_DISPLAY_GAMMA)
    const fuller = 1.0 - std.pow(1.0 - coverage, TEXT_EDGE_GAMMA)
    const leaner = std.pow(coverage, TEXT_EDGE_GAMMA)
    return std.mix(fuller, leaner, lightness)
})

const tintGlyph = (coverage: Expr, color: Expr): Expr =>
    vec4(color.member('rgb'), call(shapeCoverage, 'shapeCoverage', [coverage, color.member('rgb')]).mul(color.member('a')))

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Text",
    role: 'media',
    species: 'custom',
    category: "Textures",
    description: "Text with any Google font — multi-line with wrapping, alignment and line height — rendered crisp via a glyph texture",
    acceptsUVContext: true,
    // Padded raster box half-extents in canvas-height UV units, derived CPU-side after each raster
    // and read on the GPU as uniforms.halfW / uniforms.halfH — the declared extraFields contract,
    // patched per-frame without recomposing.
    extraFields: {
        halfW: {schema: schema.f32, initial: 0.001},
        halfH: {schema: schema.f32, initial: 0.001},
        // This layer's rectangle in the shared glyph atlas (atlas UV: left, top, width, height)
        glyphRect: {schema: schema.vec4f, initial: [0, 0, 0, 0]},
    },
    // The box hugs the ink of the measured block, and a corner drag scales the font: shared with
    // every component that draws text through a typography `text()` input (utilities/typography).
    boundingBoxDeclaration: typographyBounds((props) => String(props.text ?? '')),
    props: {
        text: {
            default: "Hello World",
            description: "The text to display (single line)",
            // Edited inline on the canvas (double-click the layer) — not shown in the settings panel
            ui: {type: 'text', label: 'Text', group: 'Text', hidden: true}
        },
        // Font, weight, italic, case, size, tracking, line height, alignment and wrap width: the
        // shared typography props (utilities/typography), so every text-drawing component matches.
        fontFamily: TYPOGRAPHY.fontFamily as never,
        fontWeight: TYPOGRAPHY.fontWeight as never,
        italic: TYPOGRAPHY.italic as never,
        textTransform: TYPOGRAPHY.textTransform as never,
        fontSize: TYPOGRAPHY.fontSize as never,
        letterSpacing: TYPOGRAPHY.letterSpacing as never,
        lineHeight: TYPOGRAPHY.lineHeight as never,
        textAlign: TYPOGRAPHY.textAlign as never,
        width: TYPOGRAPHY.width as never,
        color: {
            default: "#ffffff",
            transform: transformColor,
            description: "Text color",
            ui: {type: 'color', label: 'Color', group: 'Colors'}
        },
        origin: TYPOGRAPHY.origin as never,
        center: TYPOGRAPHY.center as never,
        rotation: TYPOGRAPHY.rotation as never
    },

    // The glyph raster host (CPU) + the three-part GPU tail: placement frame → filtered glyph
    // coverage from the shared atlas → color tint. All glyph layout lives in the host recipe above.
    gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const atlas = createGlyphRasterHost(params)
        const frame = glyphFrame(params)
        return tintGlyph(glyphCoverage(atlas.kit, frame, params.uniforms.glyphRect), params.uniforms.color)
    }}
})

export default componentDefinition
