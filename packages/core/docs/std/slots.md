# Identity & recompile rules

Two small kinds of rule that make a filter cheap. An **identity rule** sits on a filter
definition's `identityWhen:` and says when the filter does nothing: while it holds, the layer
inside passes through untouched and the effect costs no GPU time. A **recompile rule** sits
on a prop's `recompile:` and says which changes rebuild the shader rather than just update a
value.

They pair up. The skip is decided when the shader is built, so a prop the identity rule reads
needs `recompile: crosses(value)` at the same value. Cross it and the shader rebuilds with
the filter switched on or off. A prop that is mapped per pixel by a driver disables the skip
on its own, since its value then varies across the canvas.

## Reach for it when

| When | Use |
|---|---|
| skip a filter while a prop is 0 | `isZero`, with `crosses` on the prop |
| skip while a prop sits at its neutral value (1 for a gain) | `isValue`, with `crosses` on the prop |
| skip only when several props are all at rest | `allOf` |
| a no-op condition the rules cannot express | `identityWhenever` |
| a rebuild threshold that is not a single value | `recompileWhen` |

## Order

- crosses
- isZero
- isValue
- allOf
- identityWhenever
- recompileWhen

## Example

```ts
import {defineShader, p, crosses, isValue, effects} from 'shaders/std'

// Saturation: free while intensity sits at 1, rebuilt only when it leaves or returns.
export const Saturation = defineShader({
  name: 'Saturation',
  props: {
    intensity: {
      default: 1,
      recompile: crosses(1),
      ui: {type: 'range', min: 0, max: 3, step: 0.1, label: 'Intensity'},
    },
  },
  effect: effects.color.saturate(p('intensity')),
  identityWhen: isValue('intensity', 1),
  missingChildMessage: 'Nest a layer inside Saturation to adjust its color.',
})
```
