<script setup lang="ts">
import { computed, h, ref, onMounted, watch, type Component } from 'vue'

import Shader from './Shader.vue'
// <<< SHADERS_PREVIEW_MAP:START >>>
import AngularBlur from '../components/AngularBlur.vue'
import Arc from '../components/Arc.vue'
import Ascii from '../components/Ascii.vue'
import Aurora from '../components/Aurora.vue'
import BarShift from '../components/BarShift.vue'
import BarnDoors from '../components/BarnDoors.vue'
import Beam from '../components/Beam.vue'
import Bend from '../components/Bend.vue'
import Blob from '../components/Blob.vue'
import BlockDissolve from '../components/BlockDissolve.vue'
import BlockNoise from '../components/BlockNoise.vue'
import BlueNoise from '../components/BlueNoise.vue'
import Blur from '../components/Blur.vue'
import Boids from '../components/Boids.vue'
import BokehBlur from '../components/BokehBlur.vue'
import BrickPattern from '../components/BrickPattern.vue'
import BrightnessContrast from '../components/BrightnessContrast.vue'
import BrushedMetal from '../components/BrushedMetal.vue'
import Bulge from '../components/Bulge.vue'
import CRTScreen from '../components/CRTScreen.vue'
import CarbonFiber from '../components/CarbonFiber.vue'
import Chalkboard from '../components/Chalkboard.vue'
import ChannelBlur from '../components/ChannelBlur.vue'
import CheckerWipe from '../components/CheckerWipe.vue'
import Checkerboard from '../components/Checkerboard.vue'
import Chevron from '../components/Chevron.vue'
import ChromaFlow from '../components/ChromaFlow.vue'
import ChromaticAberration from '../components/ChromaticAberration.vue'
import Chrome from '../components/Chrome.vue'
import Circle from '../components/Circle.vue'
import ColorWheel from '../components/ColorWheel.vue'
import CompressionArtifacts from '../components/CompressionArtifacts.vue'
import ConcentricSpin from '../components/ConcentricSpin.vue'
import ConicGradient from '../components/ConicGradient.vue'
import ContourLines from '../components/ContourLines.vue'
import CornerPin from '../components/CornerPin.vue'
import Crescent from '../components/Crescent.vue'
import Cross from '../components/Cross.vue'
import Crystal from '../components/Crystal.vue'
import CurlNoise from '../components/CurlNoise.vue'
import CursorRipples from '../components/CursorRipples.vue'
import CursorTrail from '../components/CursorTrail.vue'
import DataMosh from '../components/DataMosh.vue'
import DiamondGradient from '../components/DiamondGradient.vue'
import DiamondWipe from '../components/DiamondWipe.vue'
import DiffuseBlur from '../components/DiffuseBlur.vue'
import DisplacementMap from '../components/DisplacementMap.vue'
import Dither from '../components/Dither.vue'
import DotGrid from '../components/DotGrid.vue'
import DropShadow from '../components/DropShadow.vue'
import Duotone from '../components/Duotone.vue'
import Ellipse from '../components/Ellipse.vue'
import Emboss from '../components/Emboss.vue'
import Engraving from '../components/Engraving.vue'
import ErosionNoise from '../components/ErosionNoise.vue'
import Exposure from '../components/Exposure.vue'
import FallingLines from '../components/FallingLines.vue'
import FilmGrain from '../components/FilmGrain.vue'
import FilmStock from '../components/FilmStock.vue'
import Flip from '../components/Flip.vue'
import FloatingParticles from '../components/FloatingParticles.vue'
import FlowField from '../components/FlowField.vue'
import Flower from '../components/Flower.vue'
import FlowingGradient from '../components/FlowingGradient.vue'
import FlutedGlass from '../components/FlutedGlass.vue'
import Fog from '../components/Fog.vue'
import Form3D from '../components/Form3D.vue'
import FractalNoise from '../components/FractalNoise.vue'
import Frost from '../components/Frost.vue'
import GaborNoise from '../components/GaborNoise.vue'
import Glass from '../components/Glass.vue'
import GlassTiles from '../components/GlassTiles.vue'
import Glitch from '../components/Glitch.vue'
import Glow from '../components/Glow.vue'
import Godrays from '../components/Godrays.vue'
import Goo from '../components/Goo.vue'
import GradientMap from '../components/GradientMap.vue'
import Grayscale from '../components/Grayscale.vue'
import Grid from '../components/Grid.vue'
import GridDistortion from '../components/GridDistortion.vue'
import Group from '../components/Group.vue'
import HTMLInCanvas from '../components/HTMLInCanvas.vue'
import Halftone from '../components/Halftone.vue'
import Heart from '../components/Heart.vue'
import Heatmap from '../components/Heatmap.vue'
import HexGrid from '../components/HexGrid.vue'
import Hologram from '../components/Hologram.vue'
import Holographic from '../components/Holographic.vue'
import HueShift from '../components/HueShift.vue'
import ImageTexture from '../components/ImageTexture.vue'
import InkFlow from '../components/InkFlow.vue'
import Invert from '../components/Invert.vue'
import IrisWipe from '../components/IrisWipe.vue'
import Irradiance from '../components/Irradiance.vue'
import IsometricCubes from '../components/IsometricCubes.vue'
import Kaleidoscope from '../components/Kaleidoscope.vue'
import KeyFrames from '../components/KeyFrames.vue'
import LensDistortion from '../components/LensDistortion.vue'
import LensFlare from '../components/LensFlare.vue'
import LightEdge from '../components/LightEdge.vue'
import LightLeak from '../components/LightLeak.vue'
import Line from '../components/Line.vue'
import LinearBlur from '../components/LinearBlur.vue'
import LinearGradient from '../components/LinearGradient.vue'
import LinearWipe from '../components/LinearWipe.vue'
import LiquidMetal from '../components/LiquidMetal.vue'
import Liquify from '../components/Liquify.vue'
import MagneticFilings from '../components/MagneticFilings.vue'
import Marble from '../components/Marble.vue'
import MeshGradient from '../components/MeshGradient.vue'
import Mirror from '../components/Mirror.vue'
import MultiPointGradient from '../components/MultiPointGradient.vue'
import Nebula from '../components/Nebula.vue'
import Neon from '../components/Neon.vue'
import NoiseDissolve from '../components/NoiseDissolve.vue'
import ObjectTracker from '../components/ObjectTracker.vue'
import Obsidian from '../components/Obsidian.vue'
import PagePeel from '../components/PagePeel.vue'
import Paper from '../components/Paper.vue'
import Parallelogram from '../components/Parallelogram.vue'
import ParticleField from '../components/ParticleField.vue'
import ParticleFlow from '../components/ParticleFlow.vue'
import Particles from '../components/Particles.vue'
import PerlinNoise from '../components/PerlinNoise.vue'
import Perspective from '../components/Perspective.vue'
import PixelSort from '../components/PixelSort.vue'
import PixelThrow from '../components/PixelThrow.vue'
import Pixelate from '../components/Pixelate.vue'
import Plasma from '../components/Plasma.vue'
import Plastic from '../components/Plastic.vue'
import PolarCoordinates from '../components/PolarCoordinates.vue'
import Polygon from '../components/Polygon.vue'
import Posterize from '../components/Posterize.vue'
import Prism from '../components/Prism.vue'
import ProgressiveBlur from '../components/ProgressiveBlur.vue'
import RadialGradient from '../components/RadialGradient.vue'
import RadialWipe from '../components/RadialWipe.vue'
import RandomBars from '../components/RandomBars.vue'
import ReactionDiffusion from '../components/ReactionDiffusion.vue'
import RectangularCoordinates from '../components/RectangularCoordinates.vue'
import ReflectivePlane from '../components/ReflectivePlane.vue'
import Repeater from '../components/Repeater.vue'
import Ring from '../components/Ring.vue'
import RippleWipe from '../components/RippleWipe.vue'
import Ripples from '../components/Ripples.vue'
import RoundedRect from '../components/RoundedRect.vue'
import Saturation from '../components/Saturation.vue'
import Scratches from '../components/Scratches.vue'
import Sharpness from '../components/Sharpness.vue'
import Shatter from '../components/Shatter.vue'
import SimplexNoise from '../components/SimplexNoise.vue'
import SineWave from '../components/SineWave.vue'
import SliceWipe from '../components/SliceWipe.vue'
import Smoke from '../components/Smoke.vue'
import SmokeFill from '../components/SmokeFill.vue'
import SmokeFlow from '../components/SmokeFlow.vue'
import Solarize from '../components/Solarize.vue'
import SolidColor from '../components/SolidColor.vue'
import Sparkle from '../components/Sparkle.vue'
import Spherize from '../components/Spherize.vue'
import Spiral from '../components/Spiral.vue'
import Star from '../components/Star.vue'
import Stone from '../components/Stone.vue'
import Strands from '../components/Strands.vue'
import Stretch from '../components/Stretch.vue'
import Stripes from '../components/Stripes.vue'
import StudioBackground from '../components/StudioBackground.vue'
import SunBurst from '../components/SunBurst.vue'
import Surface3D from '../components/Surface3D.vue'
import Swirl from '../components/Swirl.vue'
import Teardrop from '../components/Teardrop.vue'
import Text from '../components/Text.vue'
import ThinFilm from '../components/ThinFilm.vue'
import TiltShift from '../components/TiltShift.vue'
import TimeTrail from '../components/TimeTrail.vue'
import Tint from '../components/Tint.vue'
import Trapezoid from '../components/Trapezoid.vue'
import TriangularGrid from '../components/TriangularGrid.vue'
import Tritone from '../components/Tritone.vue'
import Truchet from '../components/Truchet.vue'
import Twirl from '../components/Twirl.vue'
import VHS from '../components/VHS.vue'
import VenetianBlinds from '../components/VenetianBlinds.vue'
import Vesica from '../components/Vesica.vue'
import Vibrance from '../components/Vibrance.vue'
import VideoTexture from '../components/VideoTexture.vue'
import Vignette from '../components/Vignette.vue'
import Voronoi from '../components/Voronoi.vue'
import Voxels from '../components/Voxels.vue'
import Water from '../components/Water.vue'
import Watercolor from '../components/Watercolor.vue'
import WaveDistortion from '../components/WaveDistortion.vue'
import Waveform from '../components/Waveform.vue'
import WaveletNoise from '../components/WaveletNoise.vue'
import Weave from '../components/Weave.vue'
import WebcamTexture from '../components/WebcamTexture.vue'
import Wool from '../components/Wool.vue'
import WorleyNoise from '../components/WorleyNoise.vue'
import ZoomBlur from '../components/ZoomBlur.vue'

// --- Component Map ---

const componentMap: Record<string, Component> = {
  AngularBlur,
  Arc,
  Ascii,
  Aurora,
  BarShift,
  BarnDoors,
  Beam,
  Bend,
  Blob,
  BlockDissolve,
  BlockNoise,
  BlueNoise,
  Blur,
  Boids,
  BokehBlur,
  BrickPattern,
  BrightnessContrast,
  BrushedMetal,
  Bulge,
  CRTScreen,
  CarbonFiber,
  Chalkboard,
  ChannelBlur,
  CheckerWipe,
  Checkerboard,
  Chevron,
  ChromaFlow,
  ChromaticAberration,
  Chrome,
  Circle,
  ColorWheel,
  CompressionArtifacts,
  ConcentricSpin,
  ConicGradient,
  ContourLines,
  CornerPin,
  Crescent,
  Cross,
  Crystal,
  CurlNoise,
  CursorRipples,
  CursorTrail,
  DataMosh,
  DiamondGradient,
  DiamondWipe,
  DiffuseBlur,
  DisplacementMap,
  Dither,
  DotGrid,
  DropShadow,
  Duotone,
  Ellipse,
  Emboss,
  Engraving,
  ErosionNoise,
  Exposure,
  FallingLines,
  FilmGrain,
  FilmStock,
  Flip,
  FloatingParticles,
  FlowField,
  Flower,
  FlowingGradient,
  FlutedGlass,
  Fog,
  Form3D,
  FractalNoise,
  Frost,
  GaborNoise,
  Glass,
  GlassTiles,
  Glitch,
  Glow,
  Godrays,
  Goo,
  GradientMap,
  Grayscale,
  Grid,
  GridDistortion,
  Group,
  HTMLInCanvas,
  DOMTexture: HTMLInCanvas,
  Halftone,
  Heart,
  Heatmap,
  HexGrid,
  Hologram,
  Holographic,
  HueShift,
  ImageTexture,
  InkFlow,
  Invert,
  IrisWipe,
  Irradiance,
  IsometricCubes,
  Kaleidoscope,
  KeyFrames,
  LensDistortion,
  LensFlare,
  LightEdge,
  LightLeak,
  Line,
  LinearBlur,
  LinearGradient,
  LinearWipe,
  LiquidMetal,
  Liquify,
  MagneticFilings,
  Marble,
  MeshGradient,
  Mirror,
  MultiPointGradient,
  Nebula,
  Neon,
  NoiseDissolve,
  ObjectTracker,
  Obsidian,
  PagePeel,
  Paper,
  Parallelogram,
  ParticleField,
  ParticleFlow,
  Particles,
  PerlinNoise,
  Perspective,
  PixelSort,
  PixelThrow,
  Pixelate,
  Plasma,
  Plastic,
  PolarCoordinates,
  Polygon,
  Posterize,
  Prism,
  ProgressiveBlur,
  RadialGradient,
  RadialWipe,
  RandomBars,
  ReactionDiffusion,
  RectangularCoordinates,
  ReflectivePlane,
  Repeater,
  Ring,
  RippleWipe,
  Ripples,
  RoundedRect,
  Saturation,
  Scratches,
  Sharpness,
  Shatter,
  SimplexNoise,
  SineWave,
  SliceWipe,
  Smoke,
  SmokeFill,
  SmokeFlow,
  Solarize,
  SolidColor,
  Sparkle,
  Spherize,
  Spiral,
  Star,
  Stone,
  Strands,
  Stretch,
  Stripes,
  StudioBackground,
  SunBurst,
  Surface3D,
  Swirl,
  Teardrop,
  Text,
  ThinFilm,
  TiltShift,
  TimeTrail,
  Tint,
  Trapezoid,
  TriangularGrid,
  Tritone,
  Truchet,
  Twirl,
  VHS,
  VenetianBlinds,
  Vesica,
  Vibrance,
  VideoTexture,
  Vignette,
  Voronoi,
  Voxels,
  Water,
  Watercolor,
  WaveDistortion,
  Waveform,
  WaveletNoise,
  Weave,
  WebcamTexture,
  Wool,
  WorleyNoise,
  ZoomBlur,
}
// <<< SHADERS_PREVIEW_MAP:END >>>

// --- Inlined Types ---

export interface ShaderComponent {
  type: string
  id?: string
  props?: Record<string, any>
  children?: ShaderComponent[]
  blendMode?: string
  opacity?: number
  renderOrder?: number
}

export interface ShaderDefinition {
  components: ShaderComponent[]
  structureVersion?: number
  colorSpace?: 'p3-linear' | 'srgb'
  devicePreview?: string
}

export interface KeyPropTarget {
  component_id: string
  prop: string
}

export interface KeyProp {
  label: string
  type: 'color' | 'range' | 'logo' | 'image'
  category?: string
  targets: KeyPropTarget[]
}

// --- Preview Decode Logic ---
// Preview definitions use simple XOR + Base64 with NO name mapping.
// The JSON contains readable component/prop names directly.
// This makes Preview components version-proof — adding/removing shaders
// on the server won't break deployed client components.

function xorCrypt(data: Uint8Array, key: string): Uint8Array {
  const keyBytes = new TextEncoder().encode(key)
  const result = new Uint8Array(data.length)
  for (let i = 0; i < data.length; i++) {
    result[i] = data[i] ^ keyBytes[i % keyBytes.length]
  }
  return result
}

function decodePreviewDefinition(encoded: string, key: string): ShaderDefinition {
  const binaryString = atob(encoded)
  const encrypted = new Uint8Array(binaryString.length)
  for (let i = 0; i < binaryString.length; i++) {
    encrypted[i] = binaryString.charCodeAt(i)
  }

  const decrypted = xorCrypt(encrypted, key)
  const json = new TextDecoder().decode(decrypted)

  try {
    return JSON.parse(json)
  } catch {
    throw new Error('Failed to decode shader definition: invalid data or incorrect key')
  }
}

// --- Configuration apply ---
// Mirrors the key-prop identification the design editor uses when it saves a
// preset. Inlined per framework because the build copies each framework dist
// into the published npm package and won't carry a cross-package import.

const RESERVED_IDENTIFIERS = new Set([
  'break', 'case', 'catch', 'class', 'const', 'continue', 'default',
  'delete', 'do', 'else', 'export', 'extends', 'finally', 'for',
  'function', 'if', 'import', 'in', 'instanceof', 'new', 'return',
  'super', 'switch', 'this', 'throw', 'try', 'typeof', 'var', 'void',
  'while', 'with', 'yield',
])

function slugifyIdentifier(label: string, fallback: string): string {
  const cleaned = label.replace(/[^A-Za-z0-9]+/g, ' ').trim()
  if (!cleaned) return fallback
  const parts = cleaned.split(/\s+/)
  const camel = parts
    .map((part, i) => i === 0 ? part.charAt(0).toLowerCase() + part.slice(1) : part.charAt(0).toUpperCase() + part.slice(1))
    .join('')
  const safe = /^[A-Za-z_$]/.test(camel) ? camel : `_${camel}`
  return RESERVED_IDENTIFIERS.has(safe) ? `_${safe}` : safe
}

function buildKeyPropIdentifierMap(keyProps: KeyProp[] | null | undefined): Map<string, KeyProp> {
  const used = new Set<string>(['style'])
  const map = new Map<string, KeyProp>()
  if (!keyProps?.length) return map
  keyProps.forEach((kp, i) => {
    if (!kp.targets?.length) return
    const base = slugifyIdentifier(kp.label, `keyProp${i}`)
    let ident = base
    let suffix = 1
    while (used.has(ident)) {
      suffix++
      ident = `${base}${suffix}`
    }
    used.add(ident)
    map.set(ident, kp)
  })
  return map
}

function findInTree(id: string, list: ShaderComponent[]): ShaderComponent | null {
  for (const c of list) {
    if (c.id === id) return c
    if (c.children?.length) {
      const hit = findInTree(id, c.children)
      if (hit) return hit
    }
  }
  return null
}

function applyConfiguration(
  definition: ShaderDefinition,
  configuration: Record<string, unknown> | null | undefined,
  keyProps: KeyProp[] | null | undefined
): ShaderDefinition {
  if (!configuration || !keyProps?.length) return definition
  const entries = Object.entries(configuration)
  if (entries.length === 0) return definition
  const identMap = buildKeyPropIdentifierMap(keyProps)
  // Deep-clone components only; preserves top-level identity sharing of
  // colorSpace/structureVersion. Returning a fresh object lets the computed
  // re-fire on configuration changes without refetching.
  const cloned = JSON.parse(JSON.stringify(definition.components)) as ShaderComponent[]
  for (const [ident, value] of entries) {
    const kp = identMap.get(ident)
    if (!kp) continue
    for (const t of kp.targets ?? []) {
      const comp = findInTree(t.component_id, cloned)
      if (comp?.props && t.prop in comp.props) comp.props[t.prop] = value
    }
  }
  return { ...definition, components: cloned }
}

// --- Props ---

export interface PreviewProps {
  shader?: string
  presetId?: string
  apiBaseUrl?: string
  obfuscationKey?: string
  watermarkText?: string
  watermarkLink?: string
  /**
   * Map of camelCase key_prop identifier → user value, layered on top of the
   * preset's authored defaults. Identifiers are derived from `key_prop.label`
   * via the same slugify rules the Pro Framer exporter uses, so identifiers
   * generated by codegen and identifiers expected here always match.
   *
   * Only applies on the `presetId` path — `shader` token previews don't carry
   * key_props and silently ignore this prop.
   */
  configuration?: Record<string, unknown>
  /**
   * Lock the preview to a specific snapshot of the preset, addressed by
   * content hash. Customized snippets pin a version so that subsequent
   * upstream edits (label renames, key_prop regenerations) don't silently
   * break the configuration. Server falls back to latest with the
   * `X-Shaders-Version-Fallback` header when the requested snapshot is
   * missing. Only meaningful on the `presetId` path.
   */
  version?: string
}

const props = withDefaults(defineProps<PreviewProps>(), {
  apiBaseUrl: 'https://shaders.com',
  obfuscationKey: 'shaders-preview-key',
  watermarkText: 'Unlock your Shaders Pro license',
  watermarkLink: 'https://shaders.com/dashboard?pricing=true'
})

// --- Decode & Resolve ---

// State for async-fetched definition (shader token or presetId)
const fetchedDefinition = ref<ShaderDefinition | null>(null)
const fetchedKeyProps = ref<KeyProp[] | null>(null)

const resolvedDefinition = computed<ShaderDefinition | null>(() => {
  if (!fetchedDefinition.value) return null
  return applyConfiguration(fetchedDefinition.value, props.configuration, fetchedKeyProps.value)
})

// Fetch from API and decode the response
async function fetchAndDecode(url: string) {
  try {
    const response = await fetch(url)
    if (!response.ok) throw new Error(`Failed to fetch: ${response.status}`)
    const data = await response.json()
    const item = data.preset || data.shader
    if (item?.definition && typeof item.definition === 'string') {
      fetchedDefinition.value = decodePreviewDefinition(item.definition, props.obfuscationKey)
      fetchedKeyProps.value = Array.isArray(item.key_props) ? item.key_props : null
    }
  } catch (e) {
    console.error('[Shaders Preview] Failed to fetch preview:', e)
  }
}

// Watch shader prop (token UUID) — fetch from API
watch(() => props.shader, (token) => {
  if (token) {
    fetchedDefinition.value = null
    fetchedKeyProps.value = null
    fetchAndDecode(`${props.apiBaseUrl}/api/preview/shader/${token}`)
  } else if (!props.presetId) {
    fetchedDefinition.value = null
    fetchedKeyProps.value = null
  }
}, { immediate: true })

// Watch presetId + version — refetch when either changes. Version pins the
// preview to a snapshot; when omitted, the server returns latest.
watch(() => [props.presetId, props.version] as const, ([id, version]) => {
  if (id) {
    fetchedDefinition.value = null
    fetchedKeyProps.value = null
    const versionParam = version ? `?version=${encodeURIComponent(version)}` : ''
    fetchAndDecode(`${props.apiBaseUrl}/api/preview/preset/${id}${versionParam}`)
  } else if (!props.shader) {
    fetchedDefinition.value = null
    fetchedKeyProps.value = null
  }
}, { immediate: true })

// --- Licensing Warning ---

onMounted(() => {
  console.warn(
    '[Shaders] Preview component is intended for use with a Shaders license. ' +
    'Visit https://shaders.com for more information.'
  )
})

</script>

<script lang="ts">
// Recursive component for rendering nested shader components
const RenderComponent = {
  name: 'RenderComponent',
  props: ['config', 'componentMap', 'structureVersion'],
  setup(props: { config: ShaderComponent; componentMap: Record<string, Component>; structureVersion?: number }) {
    return () => {
      const { config, componentMap } = props
      const Component = componentMap[config.type]

      if (!Component) return null

      const componentProps = config.props ? { ...config.props } : {}

      const childVNodes = config.children?.map((child: ShaderComponent, index: number) =>
        h(RenderComponent, {
          key: `${child.id || 'child'}_${index}_v${props.structureVersion || 0}`,
          config: { ...child, renderOrder: child.renderOrder ?? index },
          componentMap,
          structureVersion: props.structureVersion
        })
      )

      const slots = childVNodes && childVNodes.length > 0
        ? { default: () => childVNodes }
        : undefined

      return h(Component, {
        id: config.id,
        ...componentProps,
        renderOrder: config.renderOrder
      }, slots)
    }
  }
}
</script>

<template>
  <div v-if="resolvedDefinition" class="shaders-preview" style="position: relative; width: 100%; height: 100%;">
    <Shader
:color-space="resolvedDefinition.colorSpace || 'p3-linear'"
      :enable-performance-tracking="true"
      :is-preview="true"
      style="width: 100%; height: 100%;"
      v-bind="$attrs"
    >
      <RenderComponent
        v-for="(config, index) in resolvedDefinition.components"
        :key="`${config.id || 'root'}_${index}_v${resolvedDefinition.structureVersion || 0}`"
        :config="{ ...config, renderOrder: config.renderOrder ?? index }"
        :component-map="componentMap"
        :structure-version="resolvedDefinition.structureVersion"
      />
    </Shader>
    <a
      v-if="watermarkText"
      :href="watermarkLink"
      target="_blank"
      rel="noopener noreferrer"
      style="
        position: absolute;
        bottom: 8px;
        right: 12px;
        font-size: 11px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: rgba(255, 255, 255, 0.5);
        text-decoration: none;
        pointer-events: auto;
        z-index: 10;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
        transition: color 0.2s ease;
      "
      @mouseenter="($event.target as HTMLElement).style.color = 'rgba(255, 255, 255, 0.8)'"
      @mouseleave="($event.target as HTMLElement).style.color = 'rgba(255, 255, 255, 0.5)'"
    >
      {{ watermarkText }}
    </a>
  </div>
</template>
