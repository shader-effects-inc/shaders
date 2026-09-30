# Feedback

Effects that remember the last frame. A feedback simulation keeps a picture of its own
previous output and advances it by one step of your rule every frame: read last frame's state,
apply the rule, write the new state. Temporal echoes, codec smears and decaying trails all live
here. `feedbackSim` keeps the two state textures and swaps them, keeps a display copy the layer
samples, binds the child's picture once it exists, and runs a clock you can scale with a speed
prop.

You bring one **step** (a GPU function run once per state cell that reads the previous state
and writes the next) and the `values` it reads each frame. The build callback runs once per
instance of the layer; pick the step's variant there when a compile-time prop changes it. The
fragment samples `display` and mixes it with the live child however the effect wants. Without a
GPU nothing runs, so the fragment should fall back to the child.

The simulation needs a child. For a wave field or a displacement, `simulate.grid` says it
without GPU code; for a rule that runs many iterations per frame, see `sim.grids`.

## Reach for it when

| When | Use |
|---|---|
| a codec smear, a temporal echo, a decaying trail of the layer inside | `feedbackSim` |
| the effect should pause or run faster from a prop | the speedProp option of `feedbackSim` |
| a second texture that must swap with the state | the extraSlots option of `feedbackSim` |
| skip the whole step while a prop is 0 | the skip option of `feedbackSim` |

## Order

- feedbackSim

## Example

```ts
import {defineShader, sim} from 'shaders/std'
import {buildMacroblockAdvectKernel, call, mixExpr, ZERO} from '@coreroot/gpu/porters'
import type {GpuFragmentParams, KitTexture, Expr} from '@coreroot/gpu/porters'
import {blend} from '@coreroot/gpu/kit'

const {feedbackSim} = sim.feedback

const STATE_RES = 768

// The rule: each block of the picture either refreshes from the live child or holds and drags
// its previous contents along a motion vector, like a video stream that lost its keyframes.
const decoder = buildMacroblockAdvectKernel({
  res: STATE_RES,
  format: 'rgba16float',
  namePrefix: 'dataMosh',
  blockMix: {coarseScale: 2.6, coarsePick: 0.35},
  generationLoss: {levels: 18, amount: 0.12, decay: 0.9985},
})

// DataMosh: corrupted-codec smearing of the layer inside.
export const DataMosh = defineShader({
  name: 'DataMosh',
  requiresRTT: true,
  requiresChild: true,
  props: {
    intensity: {default: 0.7},
    blockSize: {default: 48},
    drift: {default: 0.35},
    churn: {default: 0.4},
    blend: {default: 1},
    speed: {default: 1},
    seed: {default: 0},
  },
  ...feedbackSim((params, root) => ({
    size: STATE_RES,
    format: 'rgba16float',
    speedProp: 'speed',
    paramsSchema: decoder.Params,
    // Hand the step the child, last frame's state, this frame's state, the display copy and the values.
    bindGroups: ({childTexture, display, paramsBuffer}, read, write) => root.createBindGroup(decoder.layout, {
      src: childTexture, prev: read.state, next: write.state, display, params: paramsBuffer,
    } as never),
    step: decoder.kernel,
    // The simulation's own clock stops with speed 0, so the smear freezes in place.
    values: ({dt, localTime}) => ({
      time: localTime,
      dt,
      seed: (params.getCpuValue('seed') as number) ?? 0,
      intensity: (params.getCpuValue('intensity') as number) ?? 0.7,
      blockSize: (params.getCpuValue('blockSize') as number) ?? 48,
      drift: (params.getCpuValue('drift') as number) ?? 0.35,
      churn: (params.getCpuValue('churn') as number) ?? 0.4,
    }),
  })),
  gpu: {
    fragment: ({childNode, ctx, computeOutputs, uniforms, convertToTexture}: GpuFragmentParams): Expr => {
      if (!childNode) return ZERO
      const display = computeOutputs?.display as KitTexture | undefined
      const childTex = (computeOutputs?.childTexture as KitTexture | undefined) ?? convertToTexture(childNode)
      const live = childTex.sample(ctx.uv)
      if (!display) return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [live]) // no GPU: the child untouched
      const moshed = display.sample(ctx.uv, 'linearClamp')
      return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [mixExpr(live, moshed, uniforms.blend)])
    },
  },
})
```
