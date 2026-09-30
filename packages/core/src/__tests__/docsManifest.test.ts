import {describe, it, expect} from 'vitest'
import path from 'path'
import {fileURLToPath} from 'url'
import {buildDocsManifest, parseCuration, renderCategoryMarkdown, renderLlmsText} from '../../scripts/docsManifest'

/**
 * The docs manifest is the one description of the std vocabulary every reference surface
 * renders from. This gate keeps it honest:
 *
 *  - every std module is categorised and every public word lands in the manifest;
 *  - documentation coverage can only go UP (a ratchet on the undocumented count — lower the
 *    baseline when you document words, never raise it);
 *  - a curated category's word order and reach table name real words (the builder throws);
 *  - the parts renderers consume (signatures, imports, sources) are present for every word.
 */
const coreDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const manifest = buildDocsManifest({coreDir, engineVersion: 'test'})

/**
 * Every exported std word carries a summary (the vocabulary-wide rewrite of 2026-09-30 took
 * this to zero). A new word without a doc comment fails here; document it or mark it
 * `@internal`. Raising this number is a review-blocking regression.
 */
const UNDOCUMENTED_BASELINE = 0

describe('docs manifest', () => {
    it('covers every category with at least one word and no orphan words', () => {
        const categoryIds = new Set(Object.keys(manifest.categories))
        for (const group of manifest.groups) for (const id of group.categories) expect(categoryIds.has(id), id).toBe(true)
        for (const category of Object.values(manifest.categories)) {
            expect(category.words.length, `${category.id} has words`).toBeGreaterThan(0)
            for (const wid of category.words) expect(manifest.words[wid], wid).toBeDefined()
        }
        for (const word of Object.values(manifest.words)) {
            expect(categoryIds.has(word.category), `${word.id} category`).toBe(true)
            expect(manifest.categories[word.category].words, `${word.id} listed`).toContain(word.id)
        }
    })

    it('every word has a signature, an import path and a source location', () => {
        for (const word of Object.values(manifest.words)) {
            expect(word.signature.length, word.id).toBeGreaterThan(0)
            expect(word.import.statement).toMatch(/^import \{[A-Za-z0-9_]+\} from 'shaders\/std'$/)
            expect(word.import.usage.length).toBeGreaterThan(0)
            expect(word.source.line).toBeGreaterThan(0)
        }
    })

    it('documentation coverage does not regress', () => {
        const {undocumented, words, documented} = manifest.coverage
        expect(documented + undocumented.length).toBe(words)
        expect(undocumented.length, `undocumented words:\n${undocumented.join('\n')}`).toBeLessThanOrEqual(UNDOCUMENTED_BASELINE)
    })

    it('overloads merge into one word listing each overload signature', () => {
        const wgsl = manifest.words['define.wgsl']
        expect(wgsl).toBeDefined()
        expect(wgsl.signature.split('\n')).toHaveLength(3)
        expect(wgsl.signature).toContain('wgsl(spec: WgslSpec): WgslBody')
        expect(wgsl.signature).toContain('...values: unknown[]')
        expect(Object.keys(manifest.words).filter((id) => id === 'define.wgsl')).toHaveLength(1)
    })

    it('every summary is a single sentence (a long first paragraph is split, the rest kept)', () => {
        const sentenceBreak = /[.!?]\s+(?=[A-Z`(\[])/
        for (const w of Object.values(manifest.words)) {
            if (!w.summary) continue
            expect(sentenceBreak.test(w.summary), `${w.id}: "${w.summary}"`).toBe(false)
        }
        // A word whose comment is one long paragraph keeps the remainder as its description.
        const w = manifest.words['paint.rampOver']
        expect(w.summary.length).toBeGreaterThan(0)
        expect(w.summary.length + w.description.length).toBeGreaterThan(w.summary.length)
    })

    it('object-grouped words are qualified and reachable through their parent', () => {
        const radial = manifest.words['paint.dist.radial']
        expect(radial.qualifiedName).toBe('dist.radial')
        expect(radial.import.usage).toBe('paint.dist.radial')
        expect(manifest.words['paint.dist'].members).toContain('paint.dist.radial')
    })

    it('used-by is computed from the library shaders', () => {
        expect(manifest.words['paint.rampOver'].usedBy).toContain('LinearGradient')
        expect(manifest.words['paint.dist.radial'].usedBy).toContain('RadialGradient')
    })

    it('curated categories apply their order and validate their names', () => {
        expect(manifest.coverage.curatedCategories).toEqual(expect.arrayContaining(['paint', 'light']))
        expect(manifest.categories.paint.words[0]).toBe('paint.rampOver')
        expect(manifest.categories.light.words[0]).toBe('light.glowAt')
        expect(manifest.categories.light.reach?.length).toBeGreaterThan(5)
        expect(manifest.categories.paint.example).toContain('defineShader')
    })

    it('parseCuration reads the heading convention', () => {
        const c = parseCuration([
            '# Title',
            'Intro line one.',
            '',
            'Intro line two.',
            '## Reach for it when',
            '| When | Use |',
            '|---|---|',
            '| a glow | `glowAt` or `glowPoint` |',
            '## Order',
            '- glowAt',
            '- `glowPoint`',
            '## Example',
            '```ts',
            'const x = 1',
            '```',
        ].join('\n'))
        expect(c.title).toBe('Title')
        expect(c.intro).toBe('Intro line one.\n\nIntro line two.')
        expect(c.reach).toEqual([{when: 'a glow', use: '`glowAt` or `glowPoint`'}])
        expect(c.order).toEqual(['glowAt', 'glowPoint'])
        expect(c.example).toBe('const x = 1')
    })

    it('renders category markdown and llms.txt without throwing', () => {
        const md = renderCategoryMarkdown(manifest, 'light')
        expect(md).toContain('# Light')
        expect(md).toContain('### glowAt')
        expect(md).toContain('## Reach for it when')
        const llms = renderLlmsText(manifest)
        expect(llms).toContain('## Painting')
        expect(llms).toContain('`rampOver(')
    })
})
