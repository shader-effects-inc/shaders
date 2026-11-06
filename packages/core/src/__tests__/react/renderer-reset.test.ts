import {readFileSync} from 'node:fs'
import {resolve} from 'node:path'
import {createRequire} from 'node:module'
import ts from 'typescript'
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest'

// Use the React package's own peer dependencies; no GPU is needed to exercise registration.
const requireReact = createRequire(resolve(__dirname, '../../../../react/package.json'))
const React = requireReact('react')
const {createRoot} = requireReact('react-dom/client')
const {act, createElement: h} = React
const definition = {name: 'TestLayer', props: {gain: {default: 1}}, fragment: () => ({})}
const nodes = new Map<string, {parent: string | null, uniforms: Record<string, {value: unknown}>}>()
let initialized = false
const renderer = {
    initialize: vi.fn(async () => { initialized = true }),
    isInitialized: () => initialized,
    getFailureReason: () => null,
    setOnReady: () => {}, setOnUnavailable: () => {},
    cleanup: () => { nodes.clear(); initialized = false },
    registerNode: (id: string, _fragment: unknown, parent: string | null, _metadata: unknown, uniforms: Record<string, {value: unknown}>) => { nodes.set(id, {parent, uniforms}) },
    removeNode: (id: string) => { nodes.delete(id) },
    updateNodeMetadata: () => {},
    updateUniformValue: (id: string, name: string, value: unknown) => { const entry = nodes.get(id)?.uniforms[name]; if (entry) entry.value = value }
}
let shaderModule: Record<string, any>
function load(name: string) {
    const source = readFileSync(resolve(__dirname, `../../../../react/src/engine/${name}`), 'utf8')
        .replace('__MAPPABLE_PROP_DECLS__', '').replace('__OMIT_TYPE__', 'never')
    const output: Record<string, any> = {}
    const require = (id: string) => {
        if (id === 'react' || id === 'react-dom') return requireReact(id)
        if (id.endsWith('/Shader')) return shaderModule
        if (id === 'shaders-core/telemetry') return {isExternalUser: () => false}
        if (id === 'shaders-core/utilities/transformations') return {setColorSpaceMode: () => {}}
        if (id.startsWith('shaders-core/')) return {componentDefinition: definition}
        if (id === 'shaders-core') return {
            shaderRendererGPU: () => renderer, rootPassthrough: definition, debugWarn: () => {},
            registerShader: () => {}, resolveBoundingBox: (value: unknown) => value,
            createGpuUniformsMap: (_definition: unknown, values: Record<string, unknown>) => Object.fromEntries(Object.entries(values).map(([key, value]) => [key, {value}]))
        }
        throw new Error(`Unexpected import: ${id}`)
    }
    new Function('exports', 'require', ts.transpileModule(source, {
        compilerOptions: {module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React, esModuleInterop: true}
    }).outputText)(output, require)
    return output
}
shaderModule = load('Shader.tsx')
const {Shader} = shaderModule
const {ShaderComponent: Layer} = load('component.template.tsx')
const {CustomShader} = load('CustomShader.tsx')
let root: ReturnType<typeof createRoot>
let container: HTMLDivElement

beforeEach(() => {
    nodes.clear(); initialized = false; renderer.initialize.mockClear()
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('IntersectionObserver', class { observe() {} disconnect() {} })
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({width: 320, height: 200} as DOMRect)
    container = document.createElement('div'); document.body.appendChild(container)
    root = createRoot(container)
})
afterEach(async () => {
    await act(async () => root.unmount())
    container.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals()
})

const composition = (settings: Record<string, string>, gain = 2) => h(Shader, {...settings, disableTelemetry: true},
    h(Layer, {id: 'outer', gain}, h(Layer, {id: 'nested', gain: 3})),
    h(CustomShader, {id: 'custom', src: definition, gain: 4}))

function expectLayers(gain: number) {
    expect(nodes.has('outer')).toBe(true)
    expect(nodes.get('nested')?.parent).toBe('outer')
    expect(nodes.get('outer')?.uniforms.gain.value).toBe(gain)
    expect(nodes.get('custom')?.uniforms.gain.value).toBe(4)
    expect(nodes.size).toBe(4) // root, parent, nested layer and custom shader
}

describe('React renderer reset', () => {
    it.each([{toneMapping: 'aces'}, {colorSpace: 'srgb'}, {toneMapping: 'aces', colorSpace: 'srgb'}])('keeps nested and custom layers for %j', async settings => {
        await act(async () => root.render(composition({})))
        expectLayers(2)
        const marker = container.querySelector('[data-shader-id="outer"]')
        await act(async () => root.render(composition(settings)))
        expectLayers(2)
        expect(container.querySelector('[data-shader-id="outer"]')).toBe(marker)
        await act(async () => root.render(composition(settings, 7)))
        expectLayers(7)
        await act(async () => root.render(composition({} , 7)))
        expectLayers(7)
    })
})
