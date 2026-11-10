import {describe, it, expect, vi, beforeAll, afterAll} from 'vitest'
import {tgpu, d} from '@coreroot/gpu/kit'
import {createUniformStore, type UniformStore, type FieldInit} from '@coreroot/gpu/uniformStore'
import {SystemUniforms} from '@coreroot/gpu/kit/coords'
import {composeNodeTree, type ComposeOptions} from '@coreroot/gpu/composer'
import type {GpuShaderDefinition, RegistryView, RegistryNode, GpuFragmentParams, Expr} from '@coreroot/gpu/contract'
import type {NodeMetadata} from '@coreroot/types'
import HTMLInCanvas from '@coreroot/shaders/HTMLInCanvas/index'
import {findShaderCanvas, findCaptureHost, supportsHtmlInCanvas} from '@coreroot/utilities/htmlInCanvas'

/**
 * HTMLInCanvas gate. The real capture (WICG html-in-canvas, Chrome 157+) can't run headless, so
 * this covers what can be checked without a GPU:
 *   - the composition is a plain media-texture sample of ctx.uv with an sRGB decode;
 *   - the capture host drives the html-in-canvas API correctly against stubs: marks the <Shader>
 *     canvas `content="drawable"`, copies only after a `paint` that concerns its wrapper, copies at
 *     the canvas's backing size via `drawElementImageToTexture({source}, {texture, size})`, recopies
 *     on resize, syncs hit-test geometry once, and undoes everything on cleanup;
 *   - it's a silent no-op where the API is missing (every non-Chromium browser today).
 */

beforeAll(() => vi.useFakeTimers())
afterAll(() => vi.useRealTimers())

function mockRoot() {
    const buffer = {
        patch: vi.fn(),
        write: vi.fn(),
        destroy: vi.fn(),
        $usage: vi.fn(function (this: unknown) {
            return buffer
        }),
    }
    return {createBuffer: vi.fn(() => buffer), createBindGroup: vi.fn(() => ({}))} as never
}

const RootContainer: GpuShaderDefinition = {
    name: 'Root',
    props: {} as never,
    fragment: ({childNode}: GpuFragmentParams): Expr => childNode ?? ({_emit: () => 'vec4f(0.0)'} as Expr),
}

interface NodeSpec {
    id: string
    def: GpuShaderDefinition
    parentId: string | null
    metadata?: Partial<NodeMetadata>
    domElement?: HTMLElement
}

function buildRegistry(specs: NodeSpec[]): {registry: RegistryView; store: UniformStore} {
    const store = createUniformStore(mockRoot(), {systemSchema: SystemUniforms})
    const handlesById: Record<string, Record<string, {accessorPath: string; cpu: boolean; value: unknown}>> = {}
    for (const s of specs) {
        const synthetic: FieldInit[] = [{name: '_opacity', schema: d.f32, initial: s.metadata?.opacity ?? 1}]
        handlesById[s.id] = store.defineNode(s.id, synthetic) as never
    }
    store.finalize()

    const nodes = new Map<string, RegistryNode>()
    const childrenByParent = new Map<string, RegistryNode[]>()
    for (const s of specs) {
        nodes.set(s.id, {
            id: s.id,
            componentName: s.def.name,
            parentId: s.parentId,
            definition: s.def,
            metadata: {blendMode: 'normal', opacity: undefined, renderOrder: 0, ...s.metadata} as NodeMetadata,
            handles: handlesById[s.id],
            domElement: s.domElement,
        })
    }
    for (const s of specs) {
        if (s.parentId) {
            const arr = childrenByParent.get(s.parentId) ?? []
            arr.push(nodes.get(s.id)!)
            childrenByParent.set(s.parentId, arr)
        }
    }
    const root = specs.find((s) => s.parentId === null)!
    const registry: RegistryView = {
        rootId: root.id,
        getNode: (id) => nodes.get(id),
        getChildren: (parentId) => childrenByParent.get(parentId) ?? [],
        resolveCustomId: () => null,
        store,
    }
    return {registry, store}
}

/** A media-texture stub that remembers every texture it hands out (so swaps are observable). */
function mediaTextureFactory() {
    const made: {width: number; height: number; raw: object; destroy: ReturnType<typeof vi.fn>}[] = []
    const create: ComposeOptions['createMediaTexture'] = (opts) => {
        const raw = {label: `tex${made.length}`}
        const entry = {width: opts.width, height: opts.height, raw, destroy: vi.fn()}
        made.push(entry)
        return {texture: {}, width: opts.width, height: opts.height, write: vi.fn(), unwrap: () => raw, generateMipmaps: vi.fn(), destroy: entry.destroy} as never
    }
    return {made, create}
}

function compose(options: ComposeOptions, domElement?: HTMLElement) {
    const {registry} = buildRegistry([
        {id: 'root', def: RootContainer, parentId: null},
        {id: 'dom', def: HTMLInCanvas as GpuShaderDefinition, parentId: 'root', metadata: {renderOrder: 0}, domElement},
    ])
    const ir = composeNodeTree(registry, options)
    return {ir, wgsl: () => tgpu.resolve([ir.finalPass.entry], {names: 'strict'})}
}

const runAll = (callbacks: ((p: unknown) => void)[]): void => {
    for (const cb of callbacks) cb({})
}

/** A <Shader>-like canvas carrying the html-in-canvas API stubs, with a drawable wrapper inside. */
function drawableSetup(width = 800, height = 600) {
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const requestPaint = vi.fn()
    const updateElementGeometry = vi.fn()
    Object.assign(canvas, {requestPaint, updateElementGeometry})
    const wrapper = document.createElement('div')
    wrapper.setAttribute('drawable', '')
    canvas.appendChild(wrapper)
    document.body.appendChild(canvas)
    const drawElementImageToTexture = vi.fn()
    const textures = mediaTextureFactory()
    const options: ComposeOptions = {
        canvas,
        createMediaTexture: textures.create,
        gpu: {device: {queue: {drawElementImageToTexture}} as never, root: {} as never},
    }
    const paint = (changedElements?: Element[]): void => {
        const event = new Event('paint')
        if (changedElements) Object.defineProperty(event, 'changedElements', {value: changedElements})
        canvas.dispatchEvent(event)
    }
    return {canvas, wrapper, requestPaint, updateElementGeometry, drawElementImageToTexture, textures, options, paint}
}

describe('HTMLInCanvas — media-texture sample of the captured DOM', () => {
    it('registers a media texture, samples ctx.uv, decodes sRGB, no external/RTT', () => {
        const {made, create} = mediaTextureFactory()
        const {ir, wgsl} = compose({createMediaTexture: create, gpu: {device: {queue: {}} as never, root: {} as never}})
        const mediaTex = ir.textures.filter((t) => t.kind === 'media')
        expect(mediaTex.length).toBe(1)
        expect(mediaTex[0].key).toBe('media_0')
        expect(ir.externalTextures.length).toBe(0)
        expect(ir.rttPasses.length).toBe(0)
        const code = wgsl()
        expect(code).toMatch(/textureSample\(media_0/)
        expect(code).toMatch(/srgbToLinear/)
        expect(code).toMatchSnapshot('final-pass')
        expect(made.length).toBe(1)
    })
})

describe('HTMLInCanvas — capture host', () => {
    it('is a silent no-op when the API is missing (non-Chromium browsers)', () => {
        const {canvas, wrapper, options} = drawableSetup()
        const plainCanvas = document.createElement('canvas')
        plainCanvas.appendChild(wrapper)
        const {ir} = compose({...options, canvas: plainCanvas, gpu: {device: {queue: {}} as never, root: {} as never}}, wrapper)
        expect(() => {
            runAll(ir.onBeforeRender)
            runAll(ir.onCleanup)
        }).not.toThrow()
        expect(plainCanvas.hasAttribute('content')).toBe(false)
        canvas.remove()
    })

    it('stays inert when the wrapper is not inside the <Shader> canvas', () => {
        const {canvas, options, requestPaint, drawElementImageToTexture} = drawableSetup()
        const stray = document.createElement('div')
        document.body.appendChild(stray)
        const {ir} = compose(options, stray)
        runAll(ir.onBeforeRender)
        expect(canvas.hasAttribute('content')).toBe(false)
        expect(requestPaint).not.toHaveBeenCalled()
        expect(drawElementImageToTexture).not.toHaveBeenCalled()
        stray.remove()
        canvas.remove()
    })

    it('marks the canvas drawable, requests a paint, and copies only after one', () => {
        const s = drawableSetup(800, 600)
        const {ir} = compose(s.options, s.wrapper)
        expect(s.canvas.getAttribute('content')).toBe('drawable')
        expect(s.requestPaint).toHaveBeenCalledTimes(1)

        // No snapshot yet: drawing now would throw "no cached paint record".
        runAll(ir.onBeforeRender)
        expect(s.drawElementImageToTexture).not.toHaveBeenCalled()

        // The requested paint arrives with nothing changed; it still means a snapshot exists.
        s.paint([])
        runAll(ir.onBeforeRender)
        expect(s.drawElementImageToTexture).toHaveBeenCalledTimes(1)
        const [source, destination] = s.drawElementImageToTexture.mock.calls[0]
        expect(source).toEqual({source: s.wrapper})
        expect(destination.size).toEqual({width: 800, height: 600})
        expect(destination.texture).toBe(s.textures.made[0].raw)
        // Hit-test / accessibility geometry: identity, synced once.
        expect(s.updateElementGeometry).toHaveBeenCalledTimes(1)
        expect(s.updateElementGeometry.mock.calls[0][0]).toBe(s.wrapper)

        // Nothing new painted → no copy.
        runAll(ir.onBeforeRender)
        expect(s.drawElementImageToTexture).toHaveBeenCalledTimes(1)

        // Another drawable changed, not ours → no copy.
        s.paint([document.createElement('div')])
        runAll(ir.onBeforeRender)
        expect(s.drawElementImageToTexture).toHaveBeenCalledTimes(1)

        // Ours changed → copy.
        s.paint([s.wrapper])
        runAll(ir.onBeforeRender)
        expect(s.drawElementImageToTexture).toHaveBeenCalledTimes(2)
        expect(s.updateElementGeometry).toHaveBeenCalledTimes(1)

        runAll(ir.onCleanup)
        s.canvas.remove()
    })

    it('reallocates and recopies at the new backing size when the canvas resizes', () => {
        const s = drawableSetup(400, 300)
        const {ir} = compose(s.options, s.wrapper)
        s.paint([s.wrapper])
        runAll(ir.onBeforeRender)
        expect(s.drawElementImageToTexture).toHaveBeenCalledTimes(1)

        s.canvas.width = 1200
        s.canvas.height = 900
        runAll(ir.onBeforeRender)
        expect(s.drawElementImageToTexture).toHaveBeenCalledTimes(2)
        const [, destination] = s.drawElementImageToTexture.mock.calls[1]
        expect(destination.size).toEqual({width: 1200, height: 900})
        const latest = s.textures.made[s.textures.made.length - 1]
        expect(destination.texture).toBe(latest.raw)
        expect(latest.width).toBe(1200)
        expect(s.textures.made[0].destroy).toHaveBeenCalled()

        runAll(ir.onCleanup)
        s.canvas.remove()
    })

    it('swallows a failed copy and retries on the next paint', () => {
        const s = drawableSetup()
        const {ir} = compose(s.options, s.wrapper)
        s.drawElementImageToTexture.mockImplementationOnce(() => {
            throw new DOMException('No cached paint record for element.', 'InvalidStateError')
        })
        s.paint([s.wrapper])
        expect(() => runAll(ir.onBeforeRender)).not.toThrow()
        expect(s.updateElementGeometry).not.toHaveBeenCalled()
        runAll(ir.onBeforeRender)
        expect(s.drawElementImageToTexture).toHaveBeenCalledTimes(1)
        s.paint([s.wrapper])
        runAll(ir.onBeforeRender)
        expect(s.drawElementImageToTexture).toHaveBeenCalledTimes(2)
        expect(s.updateElementGeometry).toHaveBeenCalledTimes(1)
        runAll(ir.onCleanup)
        s.canvas.remove()
    })

    it('keeps the canvas drawable while any composition holds it, restores it after the last', () => {
        const s = drawableSetup()
        s.canvas.setAttribute('content', 'fallback')
        const first = compose(s.options, s.wrapper)
        const second = compose(s.options, s.wrapper) // a recompose: the old one stays cached
        expect(s.canvas.getAttribute('content')).toBe('drawable')
        runAll(first.ir.onCleanup)
        expect(s.canvas.getAttribute('content')).toBe('drawable')
        runAll(second.ir.onCleanup)
        expect(s.canvas.getAttribute('content')).toBe('fallback')

        // Cleaned-up hosts stop listening.
        s.paint([s.wrapper])
        runAll(second.ir.onBeforeRender)
        expect(s.drawElementImageToTexture).not.toHaveBeenCalled()
        s.canvas.remove()
    })
})

describe('findShaderCanvas', () => {
    const shaderMarkup = (): {host: HTMLDivElement; canvas: HTMLCanvasElement} => {
        const host = document.createElement('div')
        host.className = 'shader'
        const canvas = document.createElement('canvas')
        canvas.setAttribute('data-renderer', 'shaders')
        host.appendChild(canvas)
        document.body.appendChild(host)
        return {host, canvas}
    }

    it('finds the enclosing renderer canvas (Vue renders its slot inside the canvas)', () => {
        const {host, canvas} = shaderMarkup()
        const marker = document.createElement('span')
        canvas.appendChild(marker)
        expect(findShaderCanvas(marker)).toBe(canvas)
        host.remove()
    })

    it('finds the sibling renderer canvas (React, Svelte, Solid render children beside it)', () => {
        const {host, canvas} = shaderMarkup()
        const group = document.createElement('span')
        const marker = document.createElement('span')
        group.appendChild(marker)
        host.appendChild(group)
        expect(findShaderCanvas(marker)).toBe(canvas)
        host.remove()
    })

    it('returns null outside a <Shader>', () => {
        expect(findShaderCanvas(document.createElement('span'))).toBeNull()
        expect(findShaderCanvas(null)).toBeNull()
    })
})

describe('findCaptureHost', () => {
    const shaderMarkup = (): {host: HTMLDivElement; canvas: HTMLCanvasElement; marker: HTMLSpanElement} => {
        const host = document.createElement('div')
        host.className = 'shader'
        const canvas = document.createElement('canvas')
        canvas.setAttribute('data-renderer', 'shaders')
        const marker = document.createElement('span')
        host.append(canvas, marker)
        document.body.appendChild(host)
        return {host, canvas, marker}
    }

    it('falls back to the .shader container (plain HTML overlay) without html-in-canvas', () => {
        expect(supportsHtmlInCanvas()).toBe(false) // happy-dom: no requestPaint, no GPUQueue
        const {host, marker} = shaderMarkup()
        expect(findCaptureHost(marker)).toEqual({element: host, drawable: false})
        // A static container gets positioned so the overlay has something to sit in…
        expect(host.style.position).toBe('relative')
        host.remove()
    })

    it('leaves an author-positioned container alone', () => {
        const {host, marker} = shaderMarkup()
        host.style.position = 'fixed'
        findCaptureHost(marker)
        expect(host.style.position).toBe('fixed')
        host.remove()
    })

    it('targets the canvas itself (drawable) when html-in-canvas is available', () => {
        const proto = HTMLCanvasElement.prototype as unknown as Record<string, unknown>
        proto.requestPaint = () => {}
        vi.stubGlobal('GPUQueue', class {drawElementImageToTexture() {}})
        try {
            expect(supportsHtmlInCanvas()).toBe(true)
            const {host, canvas, marker} = shaderMarkup()
            expect(findCaptureHost(marker)).toEqual({element: canvas, drawable: true})
            expect(host.style.position).toBe('')
            host.remove()
        } finally {
            delete proto.requestPaint
            vi.unstubAllGlobals()
        }
    })

    it('returns null outside a <Shader>', () => {
        expect(findCaptureHost(document.createElement('span'))).toBeNull()
    })
})
