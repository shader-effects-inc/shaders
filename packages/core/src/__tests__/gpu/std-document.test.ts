import {describe, it, expect} from 'vitest'
import * as fs from 'node:fs'
import * as path from 'node:path'
import ts from 'typescript'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, call} from '@coreroot/gpu/composer'
import {tgpu as kitTgpu, d} from '@coreroot/gpu/kit'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import * as std from '@coreroot/std'
import * as library from '@coreroot/std/library/index'
import {parseComponentSource, componentSourceForExport, evaluateComponentIR, defineStoredComponent, sourceHash, type StoredComponent} from '@coreroot/std/document/index'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * The agent format's document layer: every example in examples/agent is parsed WITHOUT running it,
 * its IR evaluated back into a definition, and that definition compared with the one the file
 * produces when it really runs (the drift guard). Then each one composes to WGSL.
 */

const EXAMPLES = path.resolve(__dirname, '../../../examples/agent')
const files = fs.readdirSync(EXAMPLES).filter((f) => f.endsWith('.ts'))

/** Run a component file for real: transpile it and resolve its imports to the engine's std. */
function runModule(source: string): Record<string, unknown> {
    const js = ts.transpileModule(source, {compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022}}).outputText
    const module = {exports: {} as Record<string, unknown>}
    const require = (id: string) => {
        if (id === 'shaders/std') return std
        if (id === 'shaders/std/wgsl') return library
        throw new Error(`unexpected import ${id}`)
    }
    new Function('exports', 'module', 'require', js)(module.exports, module, require)
    return module.exports
}
function runFile(source: string): GpuShaderDefinition {
    const exports = runModule(source)
    return (exports.default ?? Object.values(exports)[0]) as GpuShaderDefinition
}

/** What must match between the parsed-and-evaluated definition and the run one. */
function shape(def: GpuShaderDefinition) {
    return {
        name: def.name,
        revision: def.revision,
        props: Object.fromEntries(Object.entries(def.props).map(([k, v]) => [k, JSON.stringify((v as {default: unknown}).default)])),
        requiresChild: !!def.requiresChild,
        usesPointer: !!def.usesPointer,
        animatedTime: def.animatedTime,
        extraFields: Object.keys(def.extraFields ?? {}).sort(),
        compute: typeof def.compute,
        uvRemap: typeof def.uvRemap,
    }
}

const genBody = kitTgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Generator: GpuShaderDefinition = {
    name: 'Generator',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

describe('agent-format examples', () => {
    it('finds the examples', () => {
        expect(files.length).toBeGreaterThanOrEqual(12)
    })

    for (const file of files) {
        const source = fs.readFileSync(path.join(EXAMPLES, file), 'utf8')
        describe(file, () => {
            it('parses without running, and the IR matches running the file', () => {
                const parsed = parseComponentSource(source, ts, file)
                if (!parsed.ok) throw new Error(parsed.errors.map((e) => `${e.line}:${e.column} ${e.message}`).join('\n'))
                // The IR is plain JSON.
                const ir = JSON.parse(JSON.stringify(parsed.ir))
                const fromIR = evaluateComponentIR(ir)
                const fromRun = runFile(source)
                expect(shape(fromIR)).toEqual(shape(fromRun))
            })

            it('exports under a new name and identifier, still a valid component file', () => {
                const out = componentSourceForExport(source, ts, {identifier: 'Exported', name: 'Renamed_1234abcd', header: ['from a test']})
                expect(out).not.toBeNull()
                expect(out!.startsWith('// from a test\n')).toBe(true)
                const reparsed = parseComponentSource(out!, ts, file)
                if (!reparsed.ok) throw new Error(reparsed.errors.map((e) => `${e.line}:${e.column} ${e.message}`).join('\n'))
                expect(reparsed.ir.name).toBe('Renamed_1234abcd')
                const exports = runModule(out!)
                expect((exports.Exported as GpuShaderDefinition).name).toBe('Renamed_1234abcd')
                expect(exports.default ?? exports.Exported).toBeTruthy()
            })

            it('composes to WGSL', () => {
                const parsed = parseComponentSource(source, ts, file)
                if (!parsed.ok) throw new Error('parse failed')
                const stored: StoredComponent = {kind: 'source', source, ir: parsed.ir, irVersion: parsed.ir.v, parserVersion: parsed.ir.parser, hash: sourceHash(source)}
                const def = defineStoredComponent(stored)
                const wantsChild = def.requiresChild || !!def.uvRemap
                const {registry} = buildRegistry([
                    {id: 'root', def: RootContainer, parentId: null},
                    {id: 'c', def, parentId: 'root', metadata: {renderOrder: 0}},
                    ...(wantsChild ? [{id: 'gen', def: Generator, parentId: 'c', metadata: {renderOrder: 0}}] : []),
                ])
                const ir = composeNodeTree(registry)
                const text = tgpu.resolve([ir.finalPass.entry], {names: 'strict'})
                expect(text.length).toBeGreaterThan(100)
            })
        })
    }
})

describe('the reader refuses what a component file may not hold', () => {
    const refuse = (source: string) => {
        const r = parseComponentSource(source, ts)
        expect(r.ok).toBe(false)
        return r.ok ? [] : r.errors
    }

    it('names the line and column', () => {
        const errors = refuse(`import {defineShader, wgsl} from 'shaders/std'
const k = () => 3
export default defineShader({name: 'X', props: {}, paint: wgsl\`return vec4f(1.0);\`})`)
        expect(errors[0]).toMatchObject({line: 2, column: 11})
        expect(errors[0].message).toMatch(/functions cannot be part of a component/)
    })

    it('refuses other imports, let, loops and unknown words', () => {
        expect(refuse(`import fs from 'node:fs'\nimport {defineShader} from 'shaders/std'`)[0].message).toMatch(/imports may only come from/)
        expect(refuse(`import {defineShader} from 'shaders/std'\nlet x = 1`)[0].message).toMatch(/use const/)
        expect(refuse(`import {defineShader} from 'shaders/std'\nfor (;;) {}`)[0].message).toMatch(/a component file holds/)
        expect(refuse(`import {defineShader, registerShader} from 'shaders/std'`)[0].message).toMatch(/cannot be used in a component file/)
        expect(refuse(`import {defineShader} from 'shaders/std'\nimport {nope} from 'shaders/std/wgsl'`)[0].message).toMatch(/not in the WGSL library/)
    })

    it('folds arithmetic and inlines const numbers into WGSL', () => {
        const r = parseComponentSource(`import {defineShader, wgsl} from 'shaders/std'
const N = 4 * 2
export default defineShader({name: 'Folded', props: {k: {default: 1 / 2}}, paint: wgsl\`for (var i = 0; i < \${N}; i++) {} return vec4f(1.0);\`})`, ts)
        expect(r.ok).toBe(true)
        if (!r.ok) return
        expect((r.ir.definition.props as Record<string, {default: number}>).k.default).toBe(0.5)
        expect(JSON.stringify(r.ir.definition.paint)).toContain('i < 8')
    })

    it('keeps older envelopes working', () => {
        const def = defineStoredComponent({kind: 'wgsl', name: 'OldOne', props: {}, body: 'return vec4f(uv, 0.0, 1.0);'})
        expect(def.name).toBe('OldOne')
    })
})

describe('helpers in a fragment slot', () => {
    const compose = (def: GpuShaderDefinition) => {
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'c', def, parentId: 'root', metadata: {renderOrder: 0}},
        ])
        return tgpu.resolve([composeNodeTree(registry).finalPass.entry], {names: 'strict'})
    }

    it('inlines a const used only inside a helper, including consts built from consts', () => {
        const def = std.defineShader({
            name: 'ConstHelper',
            props: {},
            helpers: std.wgsl(`
                const BASE = 0.25;
                const GAIN = BASE * 3.0;
                const K: f32 = 2;
                fn shade(t: f32) -> f32 { return t * GAIN + K * 0.0; }
            `),
            paint: std.wgsl('return vec4f(vec3f(shade(uv.x)), 1.0);'),
        } as never)
        const text = compose(def)
        expect(text).not.toMatch(/\bGAIN\b|\bBASE\b/)
        expect(text).toContain('0.25')
    })

    it('lets a helper local shadow a built-in name', () => {
        const def = std.defineShader({
            name: 'LocalUv',
            props: {},
            helpers: std.wgsl(`
                fn wobble(p: vec2f) -> vec2f {
                    var uv = p;
                    uv.x += 0.1;
                    return uv;
                }
            `),
            paint: std.wgsl('return vec4f(wobble(uv), 0.0, 1.0);'),
        } as never)
        const text = compose(def)
        expect(text).toMatch(/fn wobble\(p: vec2f\)/)
    })

    it('lets a body own a built-in name it declares, nested or at the top (old envelopes wrote `let dt`)', () => {
        const nested = compose(std.defineShader({
            name: 'NestedDt',
            props: {},
            paint: std.wgsl('var acc = 0.0;\nfor (var s = 0; s < 4; s++) { let dt = 0.1 * f32(s); acc += dt; }\nreturn vec4f(vec3f(acc), 1.0);'),
        } as never))
        expect(nested.length).toBeGreaterThan(100)
        const top = compose(std.defineShader({
            name: 'TopDt',
            props: {},
            paint: std.wgsl('let dt = 0.5;\nlet frame = 2.0;\nreturn vec4f(vec3f(dt * frame), 1.0);'),
        } as never))
        expect(top).not.toMatch(/_b_dt|_b_frame/)
    })
})

describe('built-ins read inside input code', () => {
    it('sees pointer use after a // comment in a simulation step', () => {
        const make = (body: string) => std.defineShader({
            name: 'PointerAfterComment',
            props: {},
            inputs: {g: std.simulate.grid({resolution: 32, step: [std.op.kernel(std.wgsl(body))]})},
            paint: std.wgsl('return g(uv);'),
        } as never) as GpuShaderDefinition
        const commented = make("// the hand's push\nlet c = read(cell);\nreturn c + vec4f(pointer * dt, 0.0, 0.0);")
        expect(commented.usesPointer).toBe(true)
        expect(Object.keys(commented.extraFields ?? {})).toContain('_b_dt')
    })
})

describe('child in a filter that samples childTexture', () => {
    it('keeps child (the pixel beneath) available alongside childTexture', () => {
        const def = std.defineShader({
            name: 'ChildAndTexture',
            props: {},
            effect: std.wgsl('let n = textureSample(childTexture, childSampler, uv + vec2f(0.01, 0.0)); return mix(child, n, 0.5);'),
        } as never) as GpuShaderDefinition
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'c', def, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'c', metadata: {renderOrder: 0}},
        ])
        const text = tgpu.resolve([composeNodeTree(registry).finalPass.entry], {names: 'strict'})
        expect(text.length).toBeGreaterThan(100)
    })
})

describe('alpha on an effect that samples childTexture', () => {
    const composeFile = (alphaLine: string) => {
        const r = parseComponentSource(`import {defineShader, wgsl} from 'shaders/std'
export default defineShader({name: 'AlphaMode', props: {}, ${alphaLine} effect: wgsl\`return textureSample(childTexture, childSampler, uv);\`})`, ts)
        if (!r.ok) throw new Error(JSON.stringify(r.errors))
        const def = evaluateComponentIR(r.ir)
        const {registry} = buildRegistry([
            {id: 'root', def: RootContainer, parentId: null},
            {id: 'c', def, parentId: 'root', metadata: {renderOrder: 0}},
            {id: 'gen', def: Generator, parentId: 'c', metadata: {renderOrder: 0}},
        ])
        return tgpu.resolve([composeNodeTree(registry).finalPass.entry], {names: 'strict'})
    }

    it("a top-level alpha: 'straight' changes how the result is treated", () => {
        expect(composeFile("alpha: 'straight',")).not.toEqual(composeFile(''))
    })
})
