/**
 * `rasterize`: draw a grid simulation's cells as a deforming sheet of triangles into an
 * offscreen texture, so a body can read what the sheet shows at each pixel: which part of the
 * sheet (`sheet(uv).xy`), how deep it is (`.z`), whether it covers the pixel (`.w`), and which way
 * it faces (`sheet.normal(uv).xyz`). Cloth, folding paper, peeling layers.
 *
 * The component's own output is still one full-screen quad: the triangles only ever land in this
 * texture. No camera, models or scene.
 */
// Maintainer notes. The grid's cells hold positions: `.xy` in uv (y down), `.z` depth in 0–1
// (0 nearest). Every cell is two triangles (`holes` can skip some). One raw render pass per frame
// with a depth buffer; the vertex stage reads the grid with textureLoad. Normals come from the
// neighbouring positions, flipped to face the viewer.
import type {GridSim} from '../sim'
import {canvasTexture} from './common'
import {rawTimestampWrites} from '../../gpu/gpuTimer'
import {authorError} from '../../gpu/support'
import {withBuilder, type InputNode} from './types'

export interface RasterizeInput extends InputNode {
    readonly kind: 'rasterize'
    readonly grid: GridSim | InputNode
    readonly scale: number
    readonly holes: string | null
}

const SHADER = (holes: string | null) => `
@group(0) @binding(0) var grid: texture_2d<f32>;
@group(0) @binding(1) var<uniform> _view: vec4f; // x: the sheet's aspect (width / height)
struct V { @builtin(position) pos: vec4f, @location(0) sheetUV: vec2f, @location(1) n: vec3f, @location(2) keep: f32 }
fn at(c: vec2i) -> vec3f {
  let d = vec2i(textureDimensions(grid));
  return textureLoad(grid, clamp(c, vec2i(0), d - vec2i(1)), 0).xyz;
}
fn keepCell(cell: vec2i, cellUV: vec2f) -> bool { ${holes ? holes : 'return true;'} }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
  let d = vec2i(textureDimensions(grid));
  let quads = u32(d.x - 1);
  let q = vi / 6u;
  let k = vi % 6u;
  let cx = i32(q % quads);
  let cy = i32(q / quads);
  var off = vec2i(0, 0);
  if (k == 1u || k == 4u) { off = vec2i(1, 0); }
  if (k == 2u || k == 3u) { off = vec2i(0, 1); }
  if (k == 5u) { off = vec2i(1, 1); }
  let c = vec2i(cx, cy) + off;
  let p = at(c);
  // Tangents in true proportions: uv x is stretched by the aspect, so scale it back.
  let sx = vec3f(_view.x, 1.0, 1.0);
  let tx = (at(c + vec2i(1, 0)) - at(c - vec2i(1, 0))) * sx;
  let ty = (at(c + vec2i(0, 1)) - at(c - vec2i(0, 1))) * sx;
  var n = normalize(cross(vec3f(tx.xy, -tx.z), vec3f(ty.xy, -ty.z)));
  n = select(n, -n, n.z < 0.0);
  var o: V;
  o.pos = vec4f(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, clamp(p.z, 0.0, 1.0), 1.0);
  o.sheetUV = vec2f(c) / vec2f(d - vec2i(1));
  o.n = n;
  o.keep = select(0.0, 1.0, keepCell(vec2i(cx, cy), vec2f(f32(cx), f32(cy)) / vec2f(d - vec2i(1))));
  return o;
}
struct Out { @location(0) sheet: vec4f, @location(1) normal: vec4f }
@fragment fn fs(v: V) -> Out {
  if (v.keep < 0.5) { discard; }
  var o: Out;
  o.sheet = vec4f(v.sheetUV, v.pos.z, 1.0);
  o.normal = vec4f(normalize(v.n), 1.0);
  return o;
}`

/**
 * Draw a grid's cells as a sheet of triangles into a texture. Each cell of `grid` holds a
 * position: `.xy` in uv, `.z` depth (0 nearest, 1 farthest). Read `name(uv)`: `.xy` the sheet's
 * own coordinate at this pixel (0–1 across the grid: cell `c` sits at `vec2f(c) / vec2f(gridSize - 1)`),
 * `.z` depth, `.w` 1 where the sheet covers the pixel; and `name.normal(uv).xyz`, the facing
 * direction (toward the viewer, in true proportions on any frame shape). Give the grid
 * `precision: 'full'` when cells hold positions: half precision rounds a cloth's small per-frame
 * moves to nothing. `holes` is WGSL returning
 * false for cells to leave out (tears, cut-outs; it sees `cell` and `cellUV`).
 *
 * @example
 * ```ts
 * const cloth = simulate.grid({resolution: [64, 48], history: 2, init: wgsl`return vec4f(cellUV, 0.5, 0.0);`, step: [op.kernel(verlet)]})
 * inputs: {cloth, sheet: rasterize(cloth)},
 * effect: wgsl`
 *   let s = sheet(uv);
 *   if (s.w < 0.5) { return vec4f(0.0); }
 *   return textureSample(childTexture, childSampler, s.xy);
 * `,
 * ```
 */
export function rasterize(grid: GridSim | InputNode, opts: {scale?: number; holes?: string} = {}): RasterizeInput {
    const node = {
        kind: 'rasterize' as const,
        grid,
        scale: Math.min(1, Math.max(0.25, opts.scale ?? 1)),
        holes: opts.holes ?? null,
        outputNames: ['normal'],
    }
    return withBuilder(node, (ctx) => {
        const source = ctx.source(grid as InputNode)
        if (!source) {
            ctx.report('rasterize: list the grid in inputs before the sheet')
            return null
        }
        const sheet = canvasTexture(ctx, {scale: node.scale, label: `${ctx.name} (sheet)`, render: true})
        const normal = canvasTexture(ctx, {scale: node.scale, label: `${ctx.name} (normal)`, render: true})
        let depth: GPUTexture | null = null
        let depthSize = {width: 0, height: 0}
        ctx.onCleanup(() => depth?.destroy())
        const {device} = ctx
        device.pushErrorScope('validation')
        const module = device.createShaderModule({code: SHADER(node.holes), label: `${ctx.componentName}: ${ctx.name} (rasterize)`})
        const layout = device.createBindGroupLayout({entries: [
            {binding: 0, visibility: GPUShaderStage.VERTEX, texture: {sampleType: 'unfilterable-float'}},
            {binding: 1, visibility: GPUShaderStage.VERTEX, buffer: {type: 'uniform'}},
        ]})
        const view = device.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: `${ctx.componentName}: ${ctx.name} (rasterize view)`})
        ctx.onCleanup(() => view.destroy())
        void device.popErrorScope().then((e) => e && authorError(`[shaders] ${ctx.componentName}: rasterize '${ctx.name}': ${e.message.split('\n')[0]}`))
        let pipeline: GPURenderPipeline | null = null
        device.createRenderPipelineAsync({
            layout: device.createPipelineLayout({bindGroupLayouts: [layout]}),
            vertex: {module, entryPoint: 'vs'},
            fragment: {module, entryPoint: 'fs', targets: [{format: 'rgba16float'}, {format: 'rgba16float'}]},
            primitive: {topology: 'triangle-list'},
            depthStencil: {format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less'},
            label: `${ctx.componentName}: ${ctx.name} (rasterize)`,
        }).then((p) => {
            pipeline = p
        }).catch((e: unknown) => authorError(`[shaders] ${ctx.componentName}: rasterize '${ctx.name}' could not be built: ${e instanceof Error ? e.message : String(e)}`))
        const raw = (t: unknown) => ctx.root.unwrap(t as never) as unknown as GPUTexture
        return {
            outputs: {
                '': sheet,
                normal,
            },
            steps() {
                return [(timer) => {
                    const gridTex = source.current()
                    if (!pipeline || !gridTex) return
                    const w = sheet.width()
                    const h = sheet.height()
                    if (!depth || depthSize.width !== w || depthSize.height !== h) {
                        depth?.destroy()
                        depth = device.createTexture({size: [w, h], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT})
                        depthSize = {width: w, height: h}
                    }
                    const g = source.size()
                    const quads = Math.max(0, (g.width - 1) * (g.height - 1))
                    device.queue.writeBuffer(view, 0, new Float32Array([w / Math.max(1, h), 0, 0, 0]))
                    const bind = device.createBindGroup({layout, entries: [{binding: 0, resource: raw(gridTex).createView()}, {binding: 1, resource: {buffer: view}}]})
                    const enc = device.createCommandEncoder({label: `${ctx.name} (rasterize)`})
                    const pass = enc.beginRenderPass({
                        colorAttachments: [
                            {view: raw(sheet.current()).createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 1, 0]},
                            {view: raw(normal.current()).createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 1, 0]},
                        ],
                        depthStencilAttachment: {view: depth.createView(), depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'store'},
                        timestampWrites: rawTimestampWrites(ctx.root, timer, `${ctx.name} (rasterize)`),
                    })
                    pass.setPipeline(pipeline)
                    pass.setBindGroup(0, bind)
                    pass.draw(quads * 6)
                    pass.end()
                    device.queue.submit([enc.finish()])
                }]
            },
        }
    })
}
