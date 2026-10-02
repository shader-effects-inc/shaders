<script lang="ts">
import { onMount } from 'svelte'
import type { Snippet } from 'svelte'

import Shader from './Shader.svelte'
import RenderComponent from './PreviewRenderComponent.svelte'
// <<< SHADERS_PREVIEW_MAP:START >>>
import AngularBlur from '../components/AngularBlur.svelte'
import Arc from '../components/Arc.svelte'
import Ascii from '../components/Ascii.svelte'
import Aurora from '../components/Aurora.svelte'
import BarShift from '../components/BarShift.svelte'
import BarnDoors from '../components/BarnDoors.svelte'
import Beam from '../components/Beam.svelte'
import Bend from '../components/Bend.svelte'
import Blob from '../components/Blob.svelte'
import BlockDissolve from '../components/BlockDissolve.svelte'
import BlockNoise from '../components/BlockNoise.svelte'
import BlueNoise from '../components/BlueNoise.svelte'
import Blur from '../components/Blur.svelte'
import Boids from '../components/Boids.svelte'
import BokehBlur from '../components/BokehBlur.svelte'
import BrickPattern from '../components/BrickPattern.svelte'
import BrightnessContrast from '../components/BrightnessContrast.svelte'
import BrushedMetal from '../components/BrushedMetal.svelte'
import Bulge from '../components/Bulge.svelte'
import CRTScreen from '../components/CRTScreen.svelte'
import CarbonFiber from '../components/CarbonFiber.svelte'
import Chalkboard from '../components/Chalkboard.svelte'
import ChannelBlur from '../components/ChannelBlur.svelte'
import CheckerWipe from '../components/CheckerWipe.svelte'
import Checkerboard from '../components/Checkerboard.svelte'
import Chevron from '../components/Chevron.svelte'
import ChromaFlow from '../components/ChromaFlow.svelte'
import ChromaticAberration from '../components/ChromaticAberration.svelte'
import Chrome from '../components/Chrome.svelte'
import Circle from '../components/Circle.svelte'
import ColorWheel from '../components/ColorWheel.svelte'
import CompressionArtifacts from '../components/CompressionArtifacts.svelte'
import ConcentricSpin from '../components/ConcentricSpin.svelte'
import ConicGradient from '../components/ConicGradient.svelte'
import ContourLines from '../components/ContourLines.svelte'
import CornerPin from '../components/CornerPin.svelte'
import Crescent from '../components/Crescent.svelte'
import Cross from '../components/Cross.svelte'
import Crystal from '../components/Crystal.svelte'
import CurlNoise from '../components/CurlNoise.svelte'
import CursorRipples from '../components/CursorRipples.svelte'
import CursorTrail from '../components/CursorTrail.svelte'
import DataMosh from '../components/DataMosh.svelte'
import DiamondGradient from '../components/DiamondGradient.svelte'
import DiamondWipe from '../components/DiamondWipe.svelte'
import DiffuseBlur from '../components/DiffuseBlur.svelte'
import DisplacementMap from '../components/DisplacementMap.svelte'
import Dither from '../components/Dither.svelte'
import DotGrid from '../components/DotGrid.svelte'
import DropShadow from '../components/DropShadow.svelte'
import Duotone from '../components/Duotone.svelte'
import Ellipse from '../components/Ellipse.svelte'
import Emboss from '../components/Emboss.svelte'
import Engraving from '../components/Engraving.svelte'
import ErosionNoise from '../components/ErosionNoise.svelte'
import Exposure from '../components/Exposure.svelte'
import FallingLines from '../components/FallingLines.svelte'
import FilmGrain from '../components/FilmGrain.svelte'
import FilmStock from '../components/FilmStock.svelte'
import Flip from '../components/Flip.svelte'
import FloatingParticles from '../components/FloatingParticles.svelte'
import FlowField from '../components/FlowField.svelte'
import Flower from '../components/Flower.svelte'
import FlowingGradient from '../components/FlowingGradient.svelte'
import FlutedGlass from '../components/FlutedGlass.svelte'
import Fog from '../components/Fog.svelte'
import Form3D from '../components/Form3D.svelte'
import FractalNoise from '../components/FractalNoise.svelte'
import Frost from '../components/Frost.svelte'
import GaborNoise from '../components/GaborNoise.svelte'
import Glass from '../components/Glass.svelte'
import GlassTiles from '../components/GlassTiles.svelte'
import Glitch from '../components/Glitch.svelte'
import Glow from '../components/Glow.svelte'
import Godrays from '../components/Godrays.svelte'
import Goo from '../components/Goo.svelte'
import GradientMap from '../components/GradientMap.svelte'
import Grayscale from '../components/Grayscale.svelte'
import Grid from '../components/Grid.svelte'
import GridDistortion from '../components/GridDistortion.svelte'
import Group from '../components/Group.svelte'
import HTMLInCanvas from '../components/HTMLInCanvas.svelte'
import Halftone from '../components/Halftone.svelte'
import Heart from '../components/Heart.svelte'
import Heatmap from '../components/Heatmap.svelte'
import HexGrid from '../components/HexGrid.svelte'
import Hologram from '../components/Hologram.svelte'
import Holographic from '../components/Holographic.svelte'
import HueShift from '../components/HueShift.svelte'
import ImageTexture from '../components/ImageTexture.svelte'
import InkFlow from '../components/InkFlow.svelte'
import Invert from '../components/Invert.svelte'
import IrisWipe from '../components/IrisWipe.svelte'
import Irradiance from '../components/Irradiance.svelte'
import IsometricCubes from '../components/IsometricCubes.svelte'
import Kaleidoscope from '../components/Kaleidoscope.svelte'
import KeyFrames from '../components/KeyFrames.svelte'
import LensDistortion from '../components/LensDistortion.svelte'
import LensFlare from '../components/LensFlare.svelte'
import LightEdge from '../components/LightEdge.svelte'
import LightLeak from '../components/LightLeak.svelte'
import Line from '../components/Line.svelte'
import LinearBlur from '../components/LinearBlur.svelte'
import LinearGradient from '../components/LinearGradient.svelte'
import LinearWipe from '../components/LinearWipe.svelte'
import LiquidMetal from '../components/LiquidMetal.svelte'
import Liquify from '../components/Liquify.svelte'
import MagneticFilings from '../components/MagneticFilings.svelte'
import Marble from '../components/Marble.svelte'
import MeshGradient from '../components/MeshGradient.svelte'
import Mirror from '../components/Mirror.svelte'
import MultiPointGradient from '../components/MultiPointGradient.svelte'
import Nebula from '../components/Nebula.svelte'
import Neon from '../components/Neon.svelte'
import NoiseDissolve from '../components/NoiseDissolve.svelte'
import ObjectTracker from '../components/ObjectTracker.svelte'
import Obsidian from '../components/Obsidian.svelte'
import PagePeel from '../components/PagePeel.svelte'
import Paper from '../components/Paper.svelte'
import Parallelogram from '../components/Parallelogram.svelte'
import ParticleField from '../components/ParticleField.svelte'
import ParticleFlow from '../components/ParticleFlow.svelte'
import Particles from '../components/Particles.svelte'
import PerlinNoise from '../components/PerlinNoise.svelte'
import Perspective from '../components/Perspective.svelte'
import PixelSort from '../components/PixelSort.svelte'
import PixelThrow from '../components/PixelThrow.svelte'
import Pixelate from '../components/Pixelate.svelte'
import Plasma from '../components/Plasma.svelte'
import Plastic from '../components/Plastic.svelte'
import PolarCoordinates from '../components/PolarCoordinates.svelte'
import Polygon from '../components/Polygon.svelte'
import Posterize from '../components/Posterize.svelte'
import Prism from '../components/Prism.svelte'
import ProgressiveBlur from '../components/ProgressiveBlur.svelte'
import RadialGradient from '../components/RadialGradient.svelte'
import RadialWipe from '../components/RadialWipe.svelte'
import RandomBars from '../components/RandomBars.svelte'
import ReactionDiffusion from '../components/ReactionDiffusion.svelte'
import RectangularCoordinates from '../components/RectangularCoordinates.svelte'
import ReflectivePlane from '../components/ReflectivePlane.svelte'
import Repeater from '../components/Repeater.svelte'
import Ring from '../components/Ring.svelte'
import RippleWipe from '../components/RippleWipe.svelte'
import Ripples from '../components/Ripples.svelte'
import RoundedRect from '../components/RoundedRect.svelte'
import Saturation from '../components/Saturation.svelte'
import Scratches from '../components/Scratches.svelte'
import Sharpness from '../components/Sharpness.svelte'
import Shatter from '../components/Shatter.svelte'
import SimplexNoise from '../components/SimplexNoise.svelte'
import SineWave from '../components/SineWave.svelte'
import SliceWipe from '../components/SliceWipe.svelte'
import Smoke from '../components/Smoke.svelte'
import SmokeFill from '../components/SmokeFill.svelte'
import SmokeFlow from '../components/SmokeFlow.svelte'
import Solarize from '../components/Solarize.svelte'
import SolidColor from '../components/SolidColor.svelte'
import Sparkle from '../components/Sparkle.svelte'
import Spherize from '../components/Spherize.svelte'
import Spiral from '../components/Spiral.svelte'
import Star from '../components/Star.svelte'
import Stone from '../components/Stone.svelte'
import Strands from '../components/Strands.svelte'
import Stretch from '../components/Stretch.svelte'
import Stripes from '../components/Stripes.svelte'
import StudioBackground from '../components/StudioBackground.svelte'
import SunBurst from '../components/SunBurst.svelte'
import Surface3D from '../components/Surface3D.svelte'
import Swirl from '../components/Swirl.svelte'
import Teardrop from '../components/Teardrop.svelte'
import Text from '../components/Text.svelte'
import ThinFilm from '../components/ThinFilm.svelte'
import TiltShift from '../components/TiltShift.svelte'
import TimeTrail from '../components/TimeTrail.svelte'
import Tint from '../components/Tint.svelte'
import Trapezoid from '../components/Trapezoid.svelte'
import TriangularGrid from '../components/TriangularGrid.svelte'
import Tritone from '../components/Tritone.svelte'
import Truchet from '../components/Truchet.svelte'
import Twirl from '../components/Twirl.svelte'
import VHS from '../components/VHS.svelte'
import VenetianBlinds from '../components/VenetianBlinds.svelte'
import Vesica from '../components/Vesica.svelte'
import Vibrance from '../components/Vibrance.svelte'
import VideoTexture from '../components/VideoTexture.svelte'
import Vignette from '../components/Vignette.svelte'
import Voronoi from '../components/Voronoi.svelte'
import Voxels from '../components/Voxels.svelte'
import Water from '../components/Water.svelte'
import Watercolor from '../components/Watercolor.svelte'
import WaveDistortion from '../components/WaveDistortion.svelte'
import Waveform from '../components/Waveform.svelte'
import WaveletNoise from '../components/WaveletNoise.svelte'
import Weave from '../components/Weave.svelte'
import WebcamTexture from '../components/WebcamTexture.svelte'
import Wool from '../components/Wool.svelte'
import WorleyNoise from '../components/WorleyNoise.svelte'
import ZoomBlur from '../components/ZoomBlur.svelte'

// --- Component Map ---

const componentMap: Record<string, any> = {
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

interface ShaderComponent {
  type: string
  id?: string
  props?: Record<string, any>
  children?: ShaderComponent[]
  blendMode?: string
  opacity?: number
  renderOrder?: number
}

interface ShaderDefinition {
  components: ShaderComponent[]
  structureVersion?: number
  colorSpace?: 'p3-linear' | 'srgb'
  devicePreview?: string
}

interface KeyPropTarget {
  component_id: string
  prop: string
}

interface KeyProp {
  label: string
  type: 'color' | 'range' | 'logo' | 'image'
  category?: string
  targets: KeyPropTarget[]
}

// --- Preview Decode Logic ---
// No name mapping — preview definitions contain readable JSON.

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

// Fetch from API and decode response, returning both the definition and key_props
interface FetchResult {
  definition: ShaderDefinition | null
  keyProps: KeyProp[] | null
}

function fetchAndDecode(url: string, key: string): Promise<FetchResult> {
  return fetch(url)
    .then(res => {
      if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`)
      return res.json()
    })
    .then(data => {
      const item = data.preset || data.shader
      if (item?.definition && typeof item.definition === 'string') {
        return {
          definition: decodePreviewDefinition(item.definition, key),
          keyProps: Array.isArray(item.key_props) ? item.key_props : null,
        }
      }
      return { definition: null, keyProps: null }
    })
    .catch(err => {
      console.error('[Shaders Preview] Failed to fetch preview:', err)
      return { definition: null, keyProps: null }
    })
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

interface Props {
  shader?: string
  presetId?: string
  apiBaseUrl?: string
  obfuscationKey?: string
  watermarkText?: string
  watermarkLink?: string
  style?: string
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
   * content hash. Customized snippets pin a version so subsequent upstream
   * edits don't silently break the configuration. Server falls back to
   * latest with the `X-Shaders-Version-Fallback` header when the requested
   * snapshot is missing. Only meaningful on the `presetId` path.
   */
  version?: string
  [key: string]: any
}

const {
  shader = undefined,
  presetId = undefined,
  apiBaseUrl = 'https://shaders.com',
  obfuscationKey = 'shaders-preview-key',
  watermarkText = 'Unlock your Shaders Pro license',
  watermarkLink = 'https://shaders.com/dashboard?pricing=true',
  style = undefined,
  configuration = undefined,
  version = undefined,
  ...restProps
}: Props = $props()

// --- Decode & Resolve ---

let fetchedDefinition: ShaderDefinition | null = $state(null)
let fetchedKeyProps: KeyProp[] | null = $state(null)
let watermarkHovered = $state(false)

let resolvedDefinition: ShaderDefinition | null = $derived(
  fetchedDefinition ? applyConfiguration(fetchedDefinition, configuration, fetchedKeyProps) : null
)

// Fetch shader by token
$effect(() => {
  if (shader) {
    fetchedDefinition = null
    fetchedKeyProps = null
    fetchAndDecode(`${apiBaseUrl}/api/preview/shader/${shader}`, obfuscationKey)
      .then(result => { fetchedDefinition = result.definition; fetchedKeyProps = result.keyProps })
  } else if (!presetId) {
    fetchedDefinition = null
    fetchedKeyProps = null
  }
})

// Fetch preset by ID. Version pins the preview to a snapshot; when omitted,
// the server returns latest.
$effect(() => {
  if (presetId) {
    fetchedDefinition = null
    fetchedKeyProps = null
    const versionParam = version ? `?version=${encodeURIComponent(version)}` : ''
    fetchAndDecode(`${apiBaseUrl}/api/preview/preset/${presetId}${versionParam}`, obfuscationKey)
      .then(result => { fetchedDefinition = result.definition; fetchedKeyProps = result.keyProps })
  } else if (!shader) {
    fetchedDefinition = null
    fetchedKeyProps = null
  }
})

// --- Licensing Warning ---

onMount(() => {
  console.warn(
    '[Shaders] Preview component is intended for use with a Shaders license. ' +
    'Visit https://shaders.com for more information.'
  )
})

</script>

{#if resolvedDefinition}
  <div class="shaders-preview" style="position: relative; width: 100%; height: 100%;{style ? ' ' + style : ''}" {...restProps}>
    <Shader
      colorSpace={resolvedDefinition.colorSpace || 'p3-linear'}
      isPreview={true}
      enablePerformanceTracking={true}
      style="width: 100%; height: 100%;"
    >
      {#each resolvedDefinition.components as config, index (config.id || `root_${index}_v${resolvedDefinition.structureVersion || 0}`)}
        <RenderComponent
          config={{ ...config, renderOrder: config.renderOrder ?? index }}
          {componentMap}
          structureVersion={resolvedDefinition.structureVersion}
        />
      {/each}
    </Shader>
    {#if watermarkText}
      <a
        href={watermarkLink}
        target="_blank"
        rel="noopener noreferrer"
        style="
          position: absolute;
          bottom: 8px;
          right: 12px;
          font-size: 11px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          color: {watermarkHovered ? 'rgba(255, 255, 255, 0.8)' : 'rgba(255, 255, 255, 0.5)'};
          text-decoration: none;
          pointer-events: auto;
          z-index: 10;
          text-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
          transition: color 0.2s ease;
        "
        onmouseenter={() => watermarkHovered = true}
        onmouseleave={() => watermarkHovered = false}
      >
        {watermarkText}
      </a>
    {/if}
  </div>
{/if}
