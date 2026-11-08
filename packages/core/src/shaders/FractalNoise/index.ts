import {defineShader, p} from "@coreroot/std"
import {fractalNoise} from "@coreroot/std/paint/noise"
import {transformColor, transformAngle, transformColorSpace, colorSpaceOptions} from "@coreroot/utilities/transformations"
import {colorStopsPropConfig, type ColorStop} from "@coreroot/utilities/colorStops"

export type {ColorStop}

export interface ComponentProps {
  colorA: Parameters<typeof transformColor>[0]
  colorB: Parameters<typeof transformColor>[0]
  stops: ColorStop[] | null
  octaves: number
  detail: number
  contrast: number
  speed: number
  angle: Parameters<typeof transformAngle>[0]
  seed: number
  colorSpace: string
}

// std generator: a gated 8-octave golden-angle fBm — each octave drifts in its own direction so
// the octaves interfere rather than translating together, which is what makes this MORPH rather
// than scroll. `octaves` is compile-time but gates the fixed loop at runtime, so the slider does
// not recompile the pipeline.
export const componentDefinition = defineShader<ComponentProps>({
  name: "FractalNoise",
  role: 'generator',
  boundingBoxDeclaration: { aspectRatio: null, supportsResizeFit: true },
  category: "Textures",
  description: "Multi-octave fractal Brownian motion noise texture with true noise evolution",
  acceptsUVContext: true,
  animatedTime: { speed: 'speed' },
  props: {
    colorA: {
      default: "#000000",
      transform: transformColor,
      description: "First color",
      ui: { type: 'color', label: 'Color A', group: 'Colors' }
    },
    colorB: {
      default: "#ffffff",
      transform: transformColor,
      description: "Second color",
      ui: { type: 'color', label: 'Color B', group: 'Colors' }
    },
    stops: colorStopsPropConfig(),
    octaves: {
      default: 4,
      compileTime: true,
      description: "Number of noise octaves (more = more detail)",
      ui: { type: 'range', min: 1, max: 8, step: 1, label: 'Octaves', group: 'Effect' }
    },
    detail: {
      default: 2.0,
      description: "How much finer each successive octave becomes",
      ui: { type: ['range', 'map'], min: 1, max: 4, step: 0.1, label: 'Detail', group: 'Effect' }
    },
    contrast: {
      default: 0.5,
      description: "How strongly finer octaves contribute — higher values create more texture contrast",
      ui: { type: ['range', 'map'], min: 0.1, max: 1, step: 0.01, label: 'Contrast', group: 'Effect' }
    },
    speed: {
      default: 0.15,
      description: "Speed at which the noise pattern evolves in place",
      ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Speed', group: 'Animation' }
    },
    angle: {
      default: 0,
      transform: transformAngle,
      description: "Rotation angle in degrees",
      ui: { type: ['range', 'map'], min: -180, max: 180, step: 1, label: 'Angle', group: 'Effect' }
    },
    seed: {
      default: 0,
      description: "Random seed for pattern variation",
      ui: { type: ['range', 'map'], min: 0, max: 100, step: 1, label: 'Seed', group: 'Effect' }
    },
    colorSpace: {
      default: 'linear',
      transform: transformColorSpace,
      compileTime: true,
      description: 'Color space for interpolation',
      ui: { type: 'select', options: colorSpaceOptions, label: 'Color Space', group: 'Colors' }
    }
  },
  paint: fractalNoise({
    angle: p('angle'),
    detail: p('detail'),
    contrast: p('contrast'),
    octaves: p('octaves'),
    seed: p('seed'),
    space: p('colorSpace'),
  })
})

export default componentDefinition
