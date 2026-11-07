import {describe, it, expect} from 'vitest'
import {isSvgShape, svgShapeBoundsUrl, svgUrlOfShape} from '../utilities/svgShape'
import {isNonUniformShapeEffect} from '../utilities/shapeEffectBounds'

/**
 * A custom SVG shape is identified by the `svgUrl` in its shape config (generated in real time);
 * a legacy `.bin` (`shapeSdfUrl`) still counts, as the fallback for presets saved before.
 */
describe('svgShape routing', () => {
    const svgOnly = JSON.stringify({type: 'svg', svgUrl: 'https://example.com/logo.svg'})

    it('an svgUrl alone makes an SVG shape (new uploads have no .bin)', () => {
        expect(isSvgShape('', svgOnly)).toBe(true)
        expect(isSvgShape(undefined, {type: 'svg', svgUrl: 'https://example.com/a.svg'})).toBe(true)
        expect(svgUrlOfShape(svgOnly)).toBe('https://example.com/logo.svg')
    })

    it('a legacy .bin alone still makes an SVG shape', () => {
        expect(isSvgShape('https://example.com/a_sdf.bin', JSON.stringify({type: 'svg'}))).toBe(true)
    })

    it('analytic shapes are not SVG shapes', () => {
        expect(isSvgShape('', JSON.stringify({type: 'circleSDF', radius: 0.3}))).toBe(false)
        expect(isSvgShape('', 'not json')).toBe(false)
    })

    it('bounds scan shapeSdfUrl (an SVG override or the authoritative .bin) when set, else svgUrl', () => {
        expect(svgShapeBoundsUrl('https://example.com/a_sdf.bin', svgOnly)).toBe('https://example.com/a_sdf.bin')
        expect(svgShapeBoundsUrl('https://example.com/override.svg', svgOnly)).toBe('https://example.com/override.svg')
        expect(svgShapeBoundsUrl('', svgOnly)).toBe('https://example.com/logo.svg')
    })

    it('shape-effect bounds treat an svgUrl-only shape as SVG (uniform box)', () => {
        expect(isNonUniformShapeEffect({shape: svgOnly, shapeSdfUrl: ''})).toBe(false)
    })
})
