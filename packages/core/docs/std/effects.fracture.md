# Fracture

Broken glass. The canvas splits into **shards** around a set of random sites; a pixel belongs
to the nearest site. Where a shard has moved, a **crack** opens along its border with the
next-nearest shard, and the layer inside bends and colour-splits across it like light through
a prism. `voronoiRegionField` is the compute hook (for `compute:`) that finds each pixel's
shard and the next-nearest one; it only recomputes when the seed changes. The crack words run
in the fragment: `crackGeom` measures the crack at a pixel, `crackRefractUV` bends one colour
channel across it, and `crackShardCompose` joins the three channels with a little light on
tilted shards.

Each shard's position and displacement reach the fragment through a one-row data texture you
fill on the CPU (the Shatter shader moves them with pointer physics every frame). Lay it out as
four values per shard and read it with `cellLaneUVFor`. The crack words are GPU functions:
invoke one with `call(fn, 'name', [args])`. Samples come from the layer's texture, so
unpremultiply the final colour.

## Reach for it when

| When | Use |
|---|---|
| splitting the canvas into shards that follow a seed | `voronoiRegionField` |
| reading one shard's position or displacement from your data texture | `cellLaneUVFor` |
| how wide the crack is at a pixel and where the shard has gone | `crackGeom` |
| bending and colour-splitting the layer across the crack | `crackRefractUV` |
| the final glass colour with shard lighting | `crackShardCompose` |

## Order

- voronoiRegionField
- crackGeom
- crackRefractUV
- crackShardCompose
- cellLaneUVFor

## Example

```ts
import {defineShader, effects, math, call} from 'shaders/std'
import type {Expr, GpuFragmentParams, KitTexture} from 'shaders/std'

const {voronoiRegionField, cellLaneUVFor, crackGeom, crackRefractUV, crackShardCompose} = effects.fracture
const {vec4, div, max, sub, mul, float} = math

const SHARDS = 16
const cellLaneUV = cellLaneUVFor(SHARDS * 4)

// Deterministic shard sites for a seed, in uv. Shared by the region field and the data texture.
const rand = (n: number): number => {
  const x = Math.sin(n) * 10000
  return x - Math.floor(x)
}
const sites = (seed: number): Float32Array => {
  const out = new Float32Array(SHARDS * 2)
  for (let i = 0; i < SHARDS; i++) {
    out[i * 2] = rand(seed + i * 2)
    out[i * 2 + 1] = rand(seed + i * 2 + 1)
  }
  return out
}

// Straight alpha from a premultiplied sample.
const unpremultiply = (c: Expr): Expr => vec4(div(c.member('rgb'), max(c.member('a'), 0.0001)), c.member('a'))

// Frozen shatter: every shard sits a little away from the centre, so the cracks stay open.
export const FrozenShatter = defineShader({
  name: 'FrozenShatter',
  requiresRTT: true,
  requiresChild: true,
  props: {
    crackWidth: {default: 1},
    spread: {default: 1, compileTime: true},
    refractionStrength: {default: 5},
    chromaticSplit: {default: 1},
    shardLighting: {default: 0.1},
    seed: {default: 2, compileTime: true},
  },
  compute: voronoiRegionField({count: SHARDS, size: 1024, sites, seedProp: 'seed', seedDefault: 2, output: 'shards'}),
  gpu: {fragment: (params: GpuFragmentParams): Expr => {
    const {childNode, ctx, uniforms, computeOutputs, convertToTexture, getCpuValue, createDataTexture, registerMediaTexture, onCleanup} = params
    if (!childNode) return vec4(0, 0, 0, 0)
    const child = convertToTexture(childNode)
    const field = computeOutputs?.shards as KitTexture | undefined
    if (!field) return unpremultiply(child.sample(ctx.uv))

    // One row, four values per shard: x, y, then the displacement encoded as byte = d / 0.1 + 0.5.
    const seed = (getCpuValue('seed') as number) ?? 2
    const spread = (getCpuValue('spread') as number) ?? 1
    const pos = sites(seed)
    const bytes = new Uint8Array(SHARDS * 4 * 4)
    for (let i = 0; i < SHARDS; i++) {
      const dx = (pos[i * 2] - 0.5) * 0.05 * spread
      const dy = (pos[i * 2 + 1] - 0.5) * 0.05 * spread
      const lane = [pos[i * 2], pos[i * 2 + 1], dx / 0.1 + 0.5, dy / 0.1 + 0.5]
      for (let f = 0; f < 4; f++) bytes[(i * 4 + f) * 4] = Math.round(Math.min(1, Math.max(0, lane[f])) * 255)
    }
    const data = createDataTexture({width: SHARDS * 4, height: 1, format: 'rgba8unorm', data: bytes})
    onCleanup(() => data.destroy())
    const dataKit = registerMediaTexture(() => data.texture)
    const at = (idx: Expr, f: number): Expr => dataKit.sample(call(cellLaneUV, 'cellLaneUV', [idx, float(f)]), 'nearestClamp').member('r')
    const dispAt = (idx: Expr, f: number): Expr => mul(sub(at(idx, f), 0.5), 0.1)

    const cell = field.sample(ctx.uv, 'nearestClamp')
    const nearest = cell.member('r')
    const second = cell.member('g')
    const geom = call(crackGeom, 'crackGeom', [
      ctx.uv, at(nearest, 0), at(nearest, 1), at(second, 0), at(second, 1), dispAt(nearest, 2), dispAt(nearest, 3), uniforms.crackWidth,
    ])
    const refracted = (channel: number): Expr => child.sample(call(crackRefractUV, 'crackRefractUV', [
      geom.member('displacedUV'), geom.member('edgeNormal'), geom.member('crackIntensity'), uniforms.refractionStrength, uniforms.chromaticSplit, float(channel),
    ]))
    const glass = call(crackShardCompose, 'crackShardCompose', [
      child.sample(geom.member('displacedUV')), refracted(1), refracted(0), refracted(-1), geom.member('crackIntensity'), geom.member('disp'), uniforms.shardLighting,
    ])
    return unpremultiply(glass)
  }},
})
```
