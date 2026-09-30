# Shapes

The 2D shapes: a circle, a star, a heart and a dozen more, each as a function of a point
that tells you how far that point is from the shape's edge. The number is a **signed
distance**: negative inside, zero on the edge, positive outside. Sizes are in uv, where 1 is
the canvas height.

A shape becomes a component through the `shape:` field of `defineShader`. Its `distance`
receives the shape-local point (already centred on the `center` prop, aspect-corrected,
rotated and y down) and `u('name')` for each of your size props, and returns one of these
words. `shapeProps` declares those size props. `bounds` names the props that size the
selection box in the editor. The engine adds the `center`, `rotation`, `color`, stroke and
softness props, the bounding box and the anti-aliased pixel mask, so a shape shader is a
few lines. Shapes also combine with `math`: `mix` two distances to morph between them,
`min` for a union, `max` for an intersection.

## Reach for it when

| When | Use |
|---|---|
| a disc | `circle` |
| an oval | `ellipse` |
| a card, a button, a pill | `roundedRect` |
| a triangle, a hexagon, any regular polygon | `polygon` |
| a star or a flower | `star`, `flower` |
| a donut | `ring` |
| a pie slice or a gauge arc | `arc` with `APERTURE_TO_HALF_ANGLE` |
| a heart, a drop, a lens, a moon | `heart`, `teardrop`, `vesica`, `crescent` |
| a slanted or tapered box | `parallelogram`, `trapezoid` |
| a plus sign | `cross` |
| rounded corners on a sharp shape | `mix` toward `circle` |

## Order

- circle
- ellipse
- roundedRect
- polygon
- star
- flower
- heart
- ring
- arc
- APERTURE_TO_HALF_ANGLE
- vesica
- crescent
- teardrop
- cross
- trapezoid
- parallelogram

## Example

```ts
import {defineShader, shape, math} from 'shaders/std'

const {polygon, circle} = shape

// A regular polygon whose corners round off toward a circle.
export const Hexagon = defineShader({
  role: 'shape',
  name: 'Hexagon',
  description: 'Regular polygon with adjustable sides and corner rounding',
  shape: {
    distance: ({x, y}, u) =>
      math.mix(polygon(x, y, u('radius'), u('sides')), circle(x, y, u('radius')), u('rounding')),
    colorDescription: 'Fill color of the polygon',
    centerDescription: 'Center position of the polygon',
    bounds: {size: {width: 'radius', height: 'radius'}},
    shapeProps: {
      radius: {
        default: 0.4,
        description: 'Distance from the center to the middle of each side',
        ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Radius', group: 'Shape'},
      },
      sides: {
        default: 6,
        description: 'Number of sides (3 = triangle, 4 = square, 6 = hexagon)',
        ui: {type: 'range', min: 3, max: 12, step: 1, label: 'Sides', group: 'Shape'},
      },
      rounding: {
        default: 0,
        description: 'Corner rounding: 0 is sharp, 1 becomes a circle',
        ui: {type: 'range', min: 0, max: 1, step: 0.01, label: 'Rounding', group: 'Shape'},
      },
    },
  },
})
```
