// Host-machinery exceptions to the zero-kit-import rule: `d` is needed for the extraFields
// schemas, the host/* modules for the CPU glyph-atlas raster lifecycle, and GpuFragmentParams to
// type the atlas setup hook. The GPU bodies live in the kit; the composition is the `gpu` block.
import type {GpuFragmentParams, KitTexture} from "@coreroot/gpu/porters"
import {defineShader, p, schema} from "@coreroot/std"
import {cellSample, charGrid, glyphFor, glyphTint} from "@coreroot/std/effects/stylize"
import {createSwappableMediaTexture} from "@coreroot/gpu/kit/host/mediaLifecycle"
import {createCanvasRasterTarget, createFontDependentRaster, createRasterInvalidator} from "@coreroot/gpu/kit/host/canvasRaster"
import {transformBoolean} from "@coreroot/utilities/transformations"

export interface ComponentProps {
    characters: string
    cellSize: number
    fontFamily: string
    spacing: number
    gamma: number
    alphaThreshold: number
    preserveAlpha: boolean
}

const ATLAS_TEXTURE_SIZE = 2048

const GOOGLE_FONTS_MAP: Record<string, string> = {
    'JetBrains Mono': 'JetBrains+Mono', 'Fira Code': 'Fira+Code', 'Source Code Pro': 'Source+Code+Pro',
    'IBM Plex Mono': 'IBM+Plex+Mono', 'Space Mono': 'Space+Mono', 'Roboto Mono': 'Roboto+Mono',
    'Courier Prime': 'Courier+Prime', 'Geist Mono': 'Geist+Mono', 'VT323': 'VT323', 'Press Start 2P': 'Press+Start+2P',
    'Silkscreen': 'Silkscreen', 'Major Mono Display': 'Major+Mono+Display', 'Syne Mono': 'Syne+Mono',
    'Nova Mono': 'Nova+Mono', 'Xanh Mono': 'Xanh+Mono', 'Cutive Mono': 'Cutive+Mono', 'Share Tech Mono': 'Share+Tech+Mono',
    'Martian Mono': 'Martian+Mono', 'Azeret Mono': 'Azeret+Mono'
}
const loadedFonts = new Set<string>()
async function loadAsciiFont(fontFamily: string): Promise<void> {
    if (loadedFonts.has(fontFamily) || !GOOGLE_FONTS_MAP[fontFamily]) return
    const googleFontName = GOOGLE_FONTS_MAP[fontFamily]
    if (!document.querySelector(`link[href*="${googleFontName}"]`)) {
        const link = document.createElement('link')
        link.rel = 'stylesheet'
        link.href = `https://fonts.googleapis.com/css2?family=${googleFontName}:wght@400&display=swap`
        document.head.appendChild(link)
    }
    try {
        if (document.fonts && document.fonts.load) {
            for (const spec of [`400 12px "${fontFamily}"`, `12px "${fontFamily}"`, `400 12px ${fontFamily}`, `12px ${fontFamily}`]) {
                try { await document.fonts.load(spec); break } catch { /* try next */ }
            }
            await new Promise((r) => setTimeout(r, 500))
        } else {
            await new Promise((r) => setTimeout(r, 1000))
        }
        loadedFonts.add(fontFamily)
    } catch { /* fallback font already applies */ }
}

const num = (v: unknown, f: number): number => (typeof v === 'number' ? v : f)

// The CPU glyph-atlas host lifecycle, handed to the glyph lookup's `atlas` slot: a fixed-size
// RGBA atlas media texture re-written in place on prop change, a lazily-created raster canvas,
// font loading with fallback-then-real-font re-raster, and prop-keyed invalidation. Runs its side
// effects at composition time and returns the atlas texture the GPU tail samples.
const setupGlyphAtlas = (params: GpuFragmentParams): KitTexture => {
    const {getCpuValue, setExtraField} = params

    // Fixed-size RGBA glyph atlas media texture, re-written in place on prop change (never
    // resized, so `ensureSize` is never called — what the handle buys here is the dispose guard).
    const glyphTex = createSwappableMediaTexture(params, {
        label: 'Ascii:atlas',
        initial: {width: ATLAS_TEXTURE_SIZE, height: ATLAS_TEXTURE_SIZE},
    })

    // The atlas canvas is created on first build — a 2048² backing store is not worth allocating
    // for a shader whose `characters` prop is empty.
    const target = createCanvasRasterTarget(params, {contextAttributes: {willReadFrequently: true}})

    const buildAtlas = (characters: string, spacing: number, fontFamily: string): void => {
        if (glyphTex.disposed || !characters || characters.length === 0) return
        const charCount = characters.length
        const atlasSize = Math.max(2, Math.ceil(Math.sqrt(charCount)))
        const baseAtlasCellSize = 128
        const spacingMultiplier = Math.max(1, 2 / spacing)
        const actualCellSize = Math.min(baseAtlasCellSize * spacingMultiplier, ATLAS_TEXTURE_SIZE / atlasSize)
        const fontSize = actualCellSize * 0.75

        const c = target.canvas() ? target.context() : target.resize(ATLAS_TEXTURE_SIZE, ATLAS_TEXTURE_SIZE)
        if (!c) return
        c.clearRect(0, 0, ATLAS_TEXTURE_SIZE, ATLAS_TEXTURE_SIZE)
        c.fillStyle = '#ffffff'
        c.font = `${fontSize}px "${fontFamily}", ${fontFamily}, monospace`
        if (!c.font.includes(fontFamily) && !c.font.includes('IBM') && !c.font.includes('Plex')) {
            c.font = `${fontSize}px ${fontFamily}, monospace`
        }
        c.textAlign = 'center'
        c.textBaseline = 'middle'
        for (let i = 0; i < charCount; i++) {
            const row = Math.floor(i / atlasSize)
            const col = i % atlasSize
            c.fillText(characters[i], col * actualCellSize + actualCellSize / 2, row * actualCellSize + actualCellSize / 2)
        }
        glyphTex.write(target.canvas())
        const uvScale = (atlasSize * actualCellSize) / ATLAS_TEXTURE_SIZE
        setExtraField('_charCount', charCount)
        setExtraField('_atlasScale', uvScale)
        setExtraField('_atlasSize', atlasSize)
    }

    const readProps = () => ({
        characters: String(getCpuValue('characters') ?? componentDefinition.props.characters.default),
        fontFamily: String(getCpuValue('fontFamily') ?? componentDefinition.props.fontFamily.default),
        // spacing carries the `v => 0.1 + v*1.4` transform, so getCpuValue returns the
        // post-transform value (0.1–1.5); fall back to the transformed default (1.5).
        spacing: num(getCpuValue('spacing'), 1.5),
    })

    const rebuild = () => { const p = readProps(); buildAtlas(p.characters, p.spacing, p.fontFamily) }
    // Unlike Text, no measure cache to bump — the atlas is a fixed grid, so the font load only
    // has to re-draw. `loadAsciiFont` dedupes internally; the key memo here means an already
    // loaded font also skips the redundant re-draw.
    const font = createFontDependentRaster({
        key: () => readProps().fontFamily,
        load: () => loadAsciiFont(readProps().fontFamily),
        onLoaded: rebuild,
    })

    // Wait a tick for uniforms, raster with the fallback font, then load the real font and re-raster.
    setTimeout(() => {
        rebuild()
        font.ensure()
    }, 0)

    createRasterInvalidator(params, {
        key: () => {
            const p = readProps()
            return `${p.characters}|${p.fontFamily}|${p.spacing}`
        },
        run: () => {
            rebuild()
            font.ensure()
        },
    })

    return glyphTex.kit
}

// The character grid frame and the per-cell child color — shared by the glyph lookup and the
// tint compose.
const grid = charGrid({cellSize: p('cellSize'), spacing: p('spacing')})
const cell = cellSample({grid})

// std custom-tier filter: the GPU tail is the grid → glyph lookup → tint composition below; the
// CPU glyph-atlas host lifecycle above is handed in through the `atlas` slot.
export const componentDefinition = defineShader<ComponentProps>({
    name: "Ascii",
    role: 'filter',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Convert imagery to ASCII character art",
    requiresRTT: true,
    requiresChild: true,
    // Atlas-derived values patched per raster (charCount / atlasScale / atlasSize).
    extraFields: {
        _charCount: {schema: schema.f32, initial: 9},
        _atlasScale: {schema: schema.f32, initial: 1},
        _atlasSize: {schema: schema.f32, initial: 3},
    },
    props: {
        characters: {
            default: "@%#*+=-:.",
            description: 'Characters ordered from dense to sparse. First character is used for bright areas, last for dark areas.',
            ui: { type: 'text', label: 'Characters', group: 'Effect' }
        },
        cellSize: {
            default: 30,
            description: 'Size of each ASCII character cell (normalized to 1080p reference, scales proportionally at other resolutions)',
            ui: { type: 'range', min: 8, max: 100, step: 1, label: 'Cell Size', group: 'Effect' }
        },
        fontFamily: {
            default: "JetBrains Mono",
            description: 'Font family for characters',
            ui: {
                type: 'select',
                options: [
                    {label: 'Azeret Mono', value: 'Azeret Mono'}, {label: 'Courier Prime', value: 'Courier Prime'},
                    {label: 'Cutive Mono', value: 'Cutive Mono'}, {label: 'Fira Code', value: 'Fira Code'},
                    {label: 'Geist Mono', value: 'Geist Mono'}, {label: 'IBM Plex Mono', value: 'IBM Plex Mono'},
                    {label: 'JetBrains Mono', value: 'JetBrains Mono'}, {label: 'Major Mono Display', value: 'Major Mono Display'},
                    {label: 'Martian Mono', value: 'Martian Mono'}, {label: 'Nova Mono', value: 'Nova Mono'},
                    {label: 'Press Start 2P', value: 'Press Start 2P'}, {label: 'Roboto Mono', value: 'Roboto Mono'},
                    {label: 'Share Tech Mono', value: 'Share Tech Mono'}, {label: 'Silkscreen', value: 'Silkscreen'},
                    {label: 'Source Code Pro', value: 'Source Code Pro'}, {label: 'Space Mono', value: 'Space Mono'},
                    {label: 'Syne Mono', value: 'Syne Mono'}, {label: 'VT323', value: 'VT323'}, {label: 'Xanh Mono', value: 'Xanh Mono'}
                ],
                label: 'Font Family', group: 'Font'
            }
        },
        spacing: {
            default: 1.0,
            description: 'Character size within each cell (1.0 = optimal size, 0.0 = smallest)',
            ui: { type: ['range', 'map'], min: 0.0, max: 1.0, step: 0.01, label: 'Character Size', group: 'Effect' },
            transform: (value: number) => 0.1 + (value * 1.4)
        },
        gamma: {
            default: 1.0,
            description: 'Brightness curve adjustment. <1 brightens darks (more light characters), >1 darkens midtones (more dark characters). Use to better fit characters to image brightness range.',
            ui: { type: ['range', 'map'], min: 0.25, max: 3.0, step: 0.01, label: 'Gamma', group: 'Effect' }
        },
        alphaThreshold: {
            default: 0.0,
            description: 'Pixels with alpha below this threshold become fully transparent.',
            ui: { type: 'range', min: 0.0, max: 1.0, step: 0.01, label: 'Alpha Threshold', group: 'Effect' }
        },
        preserveAlpha: {
            default: true,
            description: 'When enabled, output alpha matches input alpha. When disabled, pixels above the alpha threshold become fully opaque.',
            ui: { type: 'checkbox', label: 'Preserve Alpha', group: 'Effect' },
            transform: transformBoolean
        }
    },

    // The recipe: the cell's child brightness picks a glyph from the CPU-rasterised atlas, the
    // glyph is tinted by the cell color, and background / out-of-bounds / below-threshold pixels
    // turn transparent.
    gpu: {
        fragment: glyphTint({
            grid,
            cell,
            glyph: glyphFor({grid, cell, gamma: p('gamma'), atlas: setupGlyphAtlas}),
            alphaThreshold: p('alphaThreshold'),
            preserveAlpha: p('preserveAlpha'),
        })
    }
})

export default componentDefinition
