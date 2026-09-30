# Overlays

Heads-up-display overlays: whole fragments that look at the layer inside, find things in it
and draw chrome on top. `detectionOverlay` cuts the canvas into cells, finds the bright,
dark, colored or opaque content in each and draws a bounding box with an optional label
around it, the computer-vision look. `motionTrackerHud` draws tracker gizmos that chase
content over time, dropping keyframe diamonds behind them along their paths.

Both words return a fragment for a `gpu:` definition rather than an `effect:`; the child is
read as a texture and shows through underneath. `motionTrackerHud` only draws: the trackers
come from the tracker pursuit simulation, which the same definition runs as its `compute:`
pass. Sizes are in pixels on a 1080-pixel-tall canvas and scale with the canvas height.
Colors are color props.

## Reach for it when

| When | Use |
|---|---|
| bounding boxes and labels around whatever is in the layer | `detectionOverlay` |
| trackers that follow moving content and leave keyframe trails | `motionTrackerHud` |

## Order

- detectionOverlay
- motionTrackerHud

## Example

```ts
import {defineShader, p, effects, transformColor, transformBoolean} from 'shaders/std'

const {detectionOverlay} = effects.overlay

// Object detection HUD: corner-bracket boxes around bright content, with a dimensions label.
export const Detector = defineShader({
  name: 'Detector',
  role: 'overlay',
  requiresRTT: true,
  requiresChild: true,
  props: {
    detectionMode: {default: 'bright', compileTime: true, ui: {type: 'select', options: [{label: 'Bright', value: 'bright'}, {label: 'Dark', value: 'dark'}, {label: 'Opaque', value: 'alpha'}]}},
    threshold: {default: 0.25, ui: {type: 'range', min: 0, max: 1, step: 0.01}},
    layout: {default: 'mosaic', compileTime: true, ui: {type: 'select', options: [{label: 'Grid', value: 'grid'}, {label: 'Quadtree', value: 'quadtree'}, {label: 'Mosaic', value: 'mosaic'}]}},
    cellSize: {default: 400, ui: {type: 'range', min: 24, max: 800, step: 1}},
    maxDepth: {default: 2, compileTime: true, ui: {type: 'range', min: 1, max: 5, step: 1}},
    boxStyle: {default: 'corners', compileTime: true, ui: {type: 'select', options: [{label: 'Full', value: 'full'}, {label: 'Corners', value: 'corners'}]}},
    lineWidth: {default: 1.5, ui: {type: 'range', min: 0.5, max: 12, step: 0.5}},
    cornerRadius: {default: 0, ui: {type: 'range', min: 0, max: 40, step: 1}},
    strokeColor: {default: '#ffffff', transform: transformColor},
    fillColor: {default: '#5a79911a', transform: transformColor},
    labelColor: {default: '#000000', transform: transformColor},
    labelBackgroundColor: {default: '#ffffff', transform: transformColor},
    labelMode: {default: 'dimensions', compileTime: true, ui: {type: 'select', options: [{label: 'None', value: 'none'}, {label: 'Dimensions', value: 'dimensions'}, {label: 'Percentage', value: 'percentage'}]}},
    labelPosition: {default: 'bottom-right', compileTime: true, ui: {type: 'select', options: [{label: 'Bottom left', value: 'bottom-left'}, {label: 'Bottom right', value: 'bottom-right'}, {label: 'Top left', value: 'top-left'}, {label: 'Top right', value: 'top-right'}]}},
    labelInset: {default: true, transform: transformBoolean, ui: {type: 'checkbox'}},
    labelRadius: {default: 0, ui: {type: 'range', min: 0, max: 30, step: 1}},
    fontFamily: {default: 'Inter', ui: {type: 'font-family'}},
    fontWeight: {default: 500, ui: {type: 'font-weight'}},
    fontSize: {default: 0.015, ui: {type: 'range', min: 0.01, max: 0.5, step: 0.005}},
    letterSpacing: {default: 0, ui: {type: 'range', min: -0.1, max: 0.5, step: 0.005}},
  },
  gpu: {fragment: detectionOverlay({
    detectionMode: p('detectionMode'), threshold: p('threshold'),
    layout: p('layout'), cellSize: p('cellSize'), maxDepth: p('maxDepth'),
    boxStyle: p('boxStyle'), lineWidth: p('lineWidth'), cornerRadius: p('cornerRadius'),
    strokeColor: p('strokeColor'), fillColor: p('fillColor'),
    labelColor: p('labelColor'), labelBackgroundColor: p('labelBackgroundColor'),
    labelMode: p('labelMode'), labelPosition: p('labelPosition'),
    labelInset: p('labelInset'), labelRadius: p('labelRadius'),
    fontFamily: p('fontFamily'), fontWeight: p('fontWeight'),
    fontSize: p('fontSize'), letterSpacing: p('letterSpacing'),
  })},
})
```
