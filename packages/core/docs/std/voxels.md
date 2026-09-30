# Voxels

One word, `voxelSurface`, rebuilds any shape out of blocks. A flat shape becomes chunky
pixel art, an SVG a slab of bricks, a 3D shape a lit voxel model. It asks you for one thing:
the color of a single block. For that it hands your `surface` a **voxel frame**, everything
a material needs already worked out: the block's normal, how open its surroundings are, how
much shadow other blocks cast on it, its cell position, a random number that stays fixed to
that block, how high and how deep it sits, and how close the pixel is to the face's edge.

Spread `...voxelSurface({...})` into a shape-effect definition next to the standard shape
props. The geometry knobs (block size, style, gaps, bevel, camera and light direction) are
prop refs made with `p('name')`, and changing them rebuilds the model. Your `surface` lights
the block with the `materials` words and returns `vec4(rgb, vox.coverage)`. The engine
resolves which block each pixel sees, anti-aliases the edges and clips to the silhouette.

## Reach for it when

| When | Use |
|---|---|
| a shape as pixel art, bricks or a voxel model | `voxelSurface` |
| per-block color variation | the frame's per-block random number inside `voxelSurface` |
| a gradient by height or depth across the model | the frame's height and depth inside `voxelSurface` |

## Order

- voxelSurface

## Example

```ts
import {defineShader, p, materials, voxels, math, transformColor, transformPosition} from 'shaders/std'

const {lambert} = materials
const {voxelSurface} = voxels
const {add, cos, local, mix, mul, neg, normalize, sin, sub, vec3, vec4} = math

const DEG = Math.PI / 180

// Any shape as lit blocks: a key light you can orbit, shadows between the blocks, a sky fill in the creases.
export const Blocks = defineShader({
  name: 'Blocks',
  role: 'shapeEffect',
  species: 'custom',
  props: {
    center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
    scale: {default: 1},
    rotation: {default: 0},
    voxelSize: {default: 0.035},
    voxelShape: {default: 'cube', compileTime: true, ui: {type: 'select', options: [{label: 'Cube', value: 'cube'}, {label: 'Rounded', value: 'rounded'}, {label: 'Sphere', value: 'sphere'}]}},
    gridSpace: {default: 'shape', compileTime: true, ui: {type: 'select', options: [{label: 'Model', value: 'shape'}, {label: 'Screen', value: 'view'}]}},
    voxelScale: {default: 1},
    fill: {default: 0},
    bevel: {default: 0},
    depth: {default: 0.12},
    lightAngle: {default: 225},
    lightElevation: {default: 45},
    shadows: {default: 0.7},
    shadowSoftness: {default: 0.3},
    color: {default: '#e8a24a', transform: transformColor},
    sky: {default: '#6f86c9', transform: transformColor},
    shape: {default: JSON.stringify({type: 'sphere3D', radius: 0.35}), ui: {type: 'shape', label: 'Shape'}},
    shapeSdfUrl: {default: '', compileTime: true},
    shapeType: {default: '', compileTime: true},
  },
  ...voxelSurface({
    style: p('voxelShape'),
    gridSpace: p('gridSpace'),
    voxelSize: p('voxelSize'),
    fill: p('fill'),
    voxelScale: p('voxelScale'),
    bevel: p('bevel'),
    shadowSoftness: p('shadowSoftness'),
    lightAngle: p('lightAngle'),
    lightElevation: p('lightElevation'),
    depth: p('depth'),
    surface: (vox, _frame, params) => {
      const u = params.uniforms
      // The key light as a vector, from its angle around the canvas and its elevation.
      const az = mul(u.lightAngle, DEG)
      const el = mul(u.lightElevation, DEG)
      const L = local(normalize(vec3(mul(cos(az), cos(el)), mul(sin(az), cos(el)), neg(sin(el)))), 'L')
      // Key light, cut by the shadow other blocks cast; sky fill, darkened in the creases.
      const lit = sub(1, mul(vox.shadow, u.shadows))
      const key = mul(lambert(vox.normal, L, {wrap: 0.12}), lit)
      const fill = mul(u.sky.member('rgb'), mul(0.5, vox.ao))
      // A little per-block variation so the bricks read as individual.
      const tone = mix(0.85, 1.15, vox.cellHash)
      const rgb = mul(add(mul(u.color.member('rgb'), key), fill), tone)
      return vec4(rgb, vox.coverage)
    },
  }),
})
```
