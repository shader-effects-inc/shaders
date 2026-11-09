/**
 * Typography shared by everything that draws text: the Text component, and any component whose
 * `text()` input draws a word (a particle logo, an ASCII field). One set of prop definitions (so
 * the editor shows the same font picker and controls everywhere), one way to measure a block, and
 * one bounding box that hugs the ink and turns a corner drag into a font-size change.
 */
import {sizeValuePx} from './dimensionalProps'
import {measureTextBlock, type TextBlockMeasure, type TextTransformMode} from './textMeasure'
import {transformPosition} from './transformations'
import type {BoundingBoxDeclaration} from '../types'

/** The typography controls a text-drawing component can carry, by prop name. */
export const TYPOGRAPHY_FIELDS = ['fontFamily', 'fontWeight', 'italic', 'textTransform', 'fontSize', 'letterSpacing', 'lineHeight', 'textAlign', 'width', 'origin', 'center', 'rotation'] as const
export type TypographyField = (typeof TYPOGRAPHY_FIELDS)[number]

/** Initial values a component can give its typography (the props' defaults). */
export interface TypographyDefaults {
    fontFamily?: string
    fontWeight?: number
    italic?: boolean
    fontSize?: number
    letterSpacing?: number
    lineHeight?: number
    textAlign?: 'left' | 'center' | 'right'
    center?: {x: number; y: number}
}

/** The typography prop definitions, exactly as the Text component declares them. */
export function typographyProps(defaults: TypographyDefaults = {}): Record<TypographyField, Record<string, unknown>> {
    return {
        fontFamily: {
            default: defaults.fontFamily ?? 'Inter',
            description: 'Google Fonts family name',
            ui: {type: 'font-family', label: 'Font', group: 'Text'},
        },
        fontWeight: {
            default: defaults.fontWeight ?? 400,
            description: 'Font weight (only weights published for the family render true; others are synthesized)',
            ui: {type: 'font-weight', label: 'Weight', group: 'Text'},
        },
        italic: {
            default: defaults.italic ?? false,
            description: 'Italic style',
            ui: {type: 'checkbox', label: 'Italic', group: 'Text'},
            transform: (value: boolean) => (value ? 1.0 : 0.0),
        },
        textTransform: {
            default: 'none',
            description: 'Case transformation applied to the text',
            ui: {
                type: 'select',
                options: [
                    {label: 'None', value: 'none'},
                    {label: 'Uppercase', value: 'uppercase'},
                    {label: 'Lowercase', value: 'lowercase'},
                    {label: 'Capitalize', value: 'capitalize'},
                ],
                label: 'Transform',
                group: 'Text',
            },
        },
        fontSize: {
            default: defaults.fontSize ?? 0.07,
            description: 'Font size as a fraction of canvas height',
            ui: {type: 'range', min: 0.01, max: 0.5, step: 0.005, label: 'Size', group: 'Typography', dimensional: 'canvas-height'},
        },
        letterSpacing: {
            default: defaults.letterSpacing ?? 0,
            description: 'Letter spacing in em units',
            ui: {type: 'range', min: -0.1, max: 0.5, step: 0.005, label: 'Letter Spacing', group: 'Typography'},
        },
        lineHeight: {
            default: defaults.lineHeight ?? 1.2,
            description: 'Line height as a multiple of font size (multi-line text)',
            ui: {type: 'range', min: 0.8, max: 2.5, step: 0.05, label: 'Line Height', group: 'Paragraph', hidden: true},
        },
        textAlign: {
            default: defaults.textAlign ?? 'center',
            description: 'Horizontal alignment of lines within the text block',
            ui: {
                type: 'select',
                options: [
                    {label: 'Left', value: 'left'},
                    {label: 'Center', value: 'center'},
                    {label: 'Right', value: 'right'},
                ],
                label: 'Align',
                group: 'Paragraph',
                hidden: true,
            },
        },
        width: {
            default: 0,
            description: "Wrap width — text wraps to fit; 0 = auto (hug contents, single line per explicit break). Edited via the Typography panel's Auto/Fixed width toggle and the side drag handles, not a slider.",
            ui: {type: 'range', min: 0, max: 1, step: 0.005, label: 'Max Width', group: 'Paragraph', units: ['%', 'px'], dimensional: 'canvas-width', hidden: true},
        },
        // What `center` is measured from. Without it the editor treats `center` as the box's
        // top-left, so its marker and drags sit half a box off.
        origin: {
            default: 'center',
            description: 'Reference edge the center position is measured from (center default). Lets you pin the text relative to a corner or the canvas centre.',
            ui: {type: 'origin', label: 'Origin', group: 'Position'},
        },
        center: {
            default: defaults.center ?? {x: 0.5, y: 0.5},
            transform: transformPosition,
            description: 'Center position of the text',
            ui: {type: 'position', label: 'Center', group: 'Position', units: ['%', 'px']},
        },
        rotation: {
            default: 0,
            description: 'Rotation in degrees',
            ui: {type: 'range', min: 0, max: 360, step: 1, label: 'Rotation', group: 'Position'},
        },
    }
}

/** Resolve a raw fontSize prop (UV number or a unit-carrying value) to pixels. 0.1 UV by default. */
export const fontSizePxOf = (v: unknown, canvasWidth: number, canvasHeight: number): number =>
    sizeValuePx(v as never, canvasHeight, 'none', canvasWidth, canvasHeight) ?? 0.1 * canvasHeight

/** Resolve the optional wrap width prop (0/absent = auto/hug) to pixels. */
export const wrapWidthPxOf = (v: unknown, canvasWidth: number, canvasHeight: number): number =>
    sizeValuePx(v as never, canvasWidth, 'x', canvasWidth, canvasHeight) ?? 0

/** What a block measurement needs, read from a component's props. */
export interface TypographyValues {
    text: string
    fontFamily: string
    fontWeight: number
    italic: boolean
    letterSpacing: number
    lineHeight: number
    textTransform: TextTransformMode
}

/** Read the typography values from raw props (missing ones take the Text component's defaults). */
export function typographyValuesOf(props: Record<string, unknown>, text: string): TypographyValues {
    return {
        text,
        fontFamily: String(props.fontFamily ?? 'Inter'),
        fontWeight: Number(props.fontWeight ?? 400),
        italic: !!props.italic && props.italic !== 0,
        letterSpacing: Number(props.letterSpacing ?? 0),
        lineHeight: Number(props.lineHeight ?? 1.2) || 1.2,
        textTransform: (props.textTransform ?? 'none') as TextTransformMode,
    }
}

/** The shared measure call: block measurement with wrapping expressed at the 100px reference. */
export function measureTypographyBlock(props: Record<string, unknown>, text: string, cw: number, ch: number): {fontSizePx: number; widthPx: number; m: TextBlockMeasure} {
    const fontSizePx = fontSizePxOf(props.fontSize, cw, ch)
    const widthPx = wrapWidthPxOf(props.width, cw, ch)
    const v = typographyValuesOf(props, text)
    return {
        fontSizePx,
        widthPx,
        m: measureTextBlock({
            text: v.text,
            fontFamily: v.fontFamily,
            fontWeight: v.fontWeight,
            italic: v.italic,
            letterSpacingEm: v.letterSpacing,
            textTransform: v.textTransform,
        }, v.lineHeight, fontSizePx > 0 && widthPx > 0 ? widthPx * 100 / fontSizePx : 0),
    }
}

/** A downward shift of `px` in the text's own frame, turned by `deg` (clockwise, y-down pixels). */
function rotatedShift(px: number, deg: number): [number, number] {
    const a = (deg || 0) * Math.PI / 180
    return [-px * Math.sin(a), px * Math.cos(a)]
}

/**
 * The bounding box of a text block: it hugs the INK of the measured block, and a corner drag
 * scales the font (a side drag rewraps a fixed-width block). `textOf` reads the string from the
 * component's props (the Text component's `text`, a particle logo's `label`, a literal). The box
 * is for selection and layout only: it never clips what the component draws.
 */
export function typographyBounds(textOf: (props: Record<string, unknown>) => string): BoundingBoxDeclaration {
    return {
        propBindings: {
            x: {prop: 'center', as: 'position-x'},
            y: {prop: 'center', as: 'position-y'},
            rotation: {prop: 'rotation', as: 'degrees'},
        },

        // The box hugs the INK of the measured block (actual glyph extents: first line's ink
        // ascent to last line's ink descent, lines pitched at `lineHeight`) — so stacked text in
        // a layout column sits tight at gap 0 and the overlay box wraps the visible glyphs.
        // Width: the authored wrap `width` when set (fixed-width mode), else the widest line.
        // The text is drawn with its FONT-box block centred on `center`, so the ink box's centre is
        // OFFSET from the stored centre: computeBounds returns that shifted centre (consumers derive
        // the shift as `centerYPx − cy·ch` and invert it on write).
        computeBounds(props: Record<string, any>, cw: number, ch: number) {
            const {fontSizePx, widthPx, m} = measureTypographyBlock(props, textOf(props), cw, ch)
            const f = fontSizePx / 100
            const cx = typeof props.center?.x === 'number' ? props.center.x : 0.5
            const cy = typeof props.center?.y === 'number' ? props.center.y : 0.5
            // Block ink centre vs font-box block centre (generalised single-line formula with
            // first-line/last-line inks — the (n−1)·pitch middle cancels).
            const inkShiftPx = ((m.ascentPer100 - m.descentPer100) + (m.inkDescentLastPer100 - m.inkAscentFirstPer100)) / 2 * f
            // The block rotates about `center`, so the shift (straight down in the text's own
            // frame) turns with it: clockwise in y-down pixels, matching how the text is drawn.
            const [sx, sy] = rotatedShift(inkShiftPx, (props.rotation as number) ?? 0)
            return {
                centerXPx: cx * cw + sx,
                centerYPx: cy * ch + sy,
                widthPx: Math.max(0, widthPx > 0 ? widthPx : m.widthPer100 * f),
                heightPx: Math.max(0, m.inkHeightPer100 * f),
                rotationDeg: (props.rotation as number) ?? 0,
            }
        },

        // The scale-vs-reflow split:
        //   scale = newHeight / oldInkHeight — corner drags change the height, so fontSize (and
        //   the wrap width with it, keeping the ratio → IDENTICAL line breaks) scales;
        //   a W-only edit leaves the height alone → scale 1 → fontSize untouched, width rewraps.
        // Preserves px-unit object shapes. bounds.centerYPx arrives in ink-box space (matching
        // computeBounds); the shift is inverted back to the stored font-box centre.
        writeBounds(bounds: {centerXPx: number; centerYPx: number; widthPx: number; heightPx: number; rotationDeg: number}, currentProps: Record<string, any>, cw: number, ch: number) {
            const {fontSizePx: oldFontPx, widthPx: oldWidthPx, m} = measureTypographyBlock(currentProps, textOf(currentProps), cw, ch)
            const oldInkPx = Math.max(1, m.inkHeightPer100 * oldFontPx / 100)
            // Fixed-width mode disambiguates the gesture by comparing per-axis ratios:
            //   proportional (corner/H edits: sW ≈ sH) → scale font + width together;
            //   width-only (side handles) → font untouched, width rewraps.
            let scale = bounds.heightPx / oldInkPx
            if (oldWidthPx > 0) {
                const sW = bounds.widthPx / Math.max(1, oldWidthPx)
                const sH = bounds.heightPx / oldInkPx
                scale = Math.abs(sW - sH) < 0.02 ? sH : 1
            }
            const newFontPx = Math.max(1, oldFontPx * scale)
            const inkShiftPx = ((m.ascentPer100 - m.descentPer100) + (m.inkDescentLastPer100 - m.inkAscentFirstPer100)) / 2 * (newFontPx / 100)
            const rawFont = currentProps.fontSize
            const fontSize = rawFont && typeof rawFont === 'object' && rawFont.unit === 'px'
                ? {value: Math.max(1, Math.round(newFontPx)), unit: 'px'}
                : Math.max(0.001, ch > 0 ? newFontPx / ch : 0.001)
            const [sx, sy] = rotatedShift(inkShiftPx, bounds.rotationDeg)
            const updates: Record<string, any> = {
                center: {...(currentProps.center ?? {x: 0.5, y: 0.5}), x: (bounds.centerXPx - sx) / cw, y: (bounds.centerYPx - sy) / ch},
                fontSize,
                rotation: bounds.rotationDeg,
            }
            // Fixed-width mode: the box width IS the wrap width prop (unit-preserving).
            if (oldWidthPx > 0) {
                const rawW = currentProps.width
                updates.width = rawW && typeof rawW === 'object' && rawW.unit === 'px'
                    ? {value: Math.max(1, Math.round(bounds.widthPx)), unit: 'px'}
                    : Math.max(0.001, cw > 0 ? bounds.widthPx / cw : 0.001)
            }
            return updates
        },
    } as BoundingBoxDeclaration
}
