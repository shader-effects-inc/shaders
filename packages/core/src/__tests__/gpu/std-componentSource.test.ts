import {afterAll, describe, expect, it} from 'vitest'
import {mkdirSync, rmSync, writeFileSync} from 'node:fs'
import {resolve} from 'node:path'
import {tgpu} from '@coreroot/gpu/kit'
import {composeNodeTree, call} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import {defineWgslEnvelope, wgslComponentSource, type WgslEnvelope} from '@coreroot/std'
import {d} from '@coreroot/gpu/kit'
import {buildRegistry, RootContainer} from './_patternHarness'

/**
 * Code export writes a custom component (a `wgsl` envelope) as a `defineShader` source file.
 * The gate: importing that file gives the same shader the editor builds from the envelope:
 * the same WGSL, the same props, the same clock and the same kind of layer.
 */

const STD_ENTRY = resolve(__dirname, '../../std/index.ts')
const TMP = resolve(__dirname, '.component-source-tmp')

afterAll(() => rmSync(TMP, {recursive: true, force: true}))

/** Write the emitted source with `shaders/std` pointed at this package's std, and import it. */
async function importSource(source: string, file: string): Promise<GpuShaderDefinition> {
    mkdirSync(TMP, {recursive: true})
    const path = resolve(TMP, `${file}.ts`)
    writeFileSync(path, source.replace(`from 'shaders/std'`, `from '${STD_ENTRY}'`))
    const module = await import(/* @vite-ignore */ path)
    return Object.values(module)[0] as GpuShaderDefinition
}

const genBody = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
    'use gpu'
    return d.vec4f(uv.x, uv.y, 0.5, 1.0)
})
const Child: GpuShaderDefinition = {
    name: 'Child',
    props: {} as never,
    fragment: ({ctx}: GpuFragmentParams): Expr => call(genBody, 'genBody', [ctx.uv]),
}

function resolveWgsl(def: GpuShaderDefinition): string {
    const filter = def.requiresChild || def.requiresRTT || !def.boundingBoxDeclaration
    const {registry} = buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        {id: 'x', def, parentId: 'root', metadata: {renderOrder: 0}},
        ...(filter ? [{id: 'c', def: Child, parentId: 'x', metadata: {renderOrder: 0}}] : []),
    ])
    const ir = composeNodeTree(registry)
    const entries = [...ir.rttPasses.map((pass) => pass.fragment.entry), ir.finalPass.entry]
    // Whitespace aside: the file re-indents the body
    return entries.map((entry) => tgpu.resolve([entry as never], {names: 'strict'})).join('\n').replace(/\s+/g, ' ')
}

/** What the renderer and editor read from a definition, minus functions compared by identity. */
function shapeOf(def: GpuShaderDefinition) {
    const props = Object.fromEntries(Object.entries(def.props as Record<string, {default: unknown; transform?: unknown; ui?: unknown; compileTime?: boolean}>)
        .map(([name, config]) => [name, {default: config.default, transform: config.transform, ui: config.ui, compileTime: config.compileTime}]))
    return {
        name: def.name,
        props,
        animatedTime: def.animatedTime,
        acceptsUVContext: def.acceptsUVContext,
        boundingBoxDeclaration: def.boundingBoxDeclaration,
        requiresChild: def.requiresChild,
        requiresRTT: def.requiresRTT,
    }
}

const ENVELOPES: Record<string, WgslEnvelope> = {
    paint: {
        kind: 'wgsl',
        name: 'Halo_1a2b3c4d',
        description: 'Rings that breathe outward',
        props: {
            inner: {kind: 'color', default: '#ffd166', label: 'Inner'},
            outer: {kind: 'color', default: '#073b4c'},
            center: {kind: 'position', default: {x: 0.5, y: 0.5}},
            radius: {kind: 'float', default: 0.6, min: 0.05, max: 2, description: 'How far the rings reach'},
            spin: {kind: 'angle', default: 30},
            invert: {kind: 'bool', default: false},
        },
        body: `
            let q = (uv - center) * vec2f(aspect, 1.0);
            let a = radians(spin);
            let dd = length(q) / radius + sin(time + a) * 0.05;
            let t = smoothstep(0.0, 1.0, dd);
            let c = mix(inner.rgb, outer.rgb, select(t, 1.0 - t, invert > 0.5));
            return vec4f(c, 1.0);`,
    },
    pointwise: {
        kind: 'wgsl',
        name: 'Tint_00ff00aa',
        props: {color: {kind: 'color', default: '#ff0066'}, amount: {kind: 'float', default: 0.5, min: 0, max: 1}},
        body: 'return vec4f(mix(child.rgb, color.rgb * child.a, amount), child.a);',
    },
    gather: {
        kind: 'wgsl',
        name: 'Smear_12345678',
        props: {reach: {kind: 'float', default: 0.01, min: 0, max: 0.1}},
        body: `
\tvar sum = vec4f(0.0);
\tfor (var i = 0; i < 4; i++) {
\t\tsum += textureSample(childTexture, childSampler, uv + vec2f(f32(i) * reach, 0.0));
\t}
\treturn sum / 4.0;`,
    },
}

describe('wgslComponentSource', () => {
    for (const [kind, envelope] of Object.entries(ENVELOPES)) {
        it(`a ${kind} component's file is the shader the editor builds`, async () => {
            const source = wgslComponentSource(envelope, {identifier: 'Exported', header: ['Generated by Shaders']})
            const exported = await importSource(source, kind)
            const editor = defineWgslEnvelope(envelope)
            expect(shapeOf(exported)).toEqual(shapeOf(editor))
            expect(resolveWgsl(exported)).toBe(resolveWgsl(editor))
        })
    }

    it('reads like hand-written code', () => {
        const source = wgslComponentSource(ENVELOPES.paint!, {identifier: 'Halo', header: ['Generated by Shaders']})
        expect(source).toMatch(/^\/\/ Generated by Shaders\nimport \{ defineShader, wgsl, transformBool01, transformColor, transformPosition \} from 'shaders\/std'/)
        expect(source).toContain(`export const Halo = defineShader({\n  name: 'Halo_1a2b3c4d',`)
        expect(source).toContain(`animatedTime: { speed: 'speed' },`)
        expect(source).toContain(`  paint: wgsl\`\n    let q = (uv - center) * vec2f(aspect, 1.0);`)
        // An authored description is kept; one that only repeats the label is not
        expect(source).toContain(`description: 'How far the rings reach'`)
        expect(source).not.toMatch(/inner: \{[^\n]*description/)
    })

    it('rejects an identifier that is not one', () => {
        expect(() => wgslComponentSource(ENVELOPES.paint!, {identifier: 'my halo'})).toThrow(/not a valid identifier/)
    })
})
