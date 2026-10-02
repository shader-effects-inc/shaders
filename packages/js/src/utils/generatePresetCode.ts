interface ComponentConfig {
  type: string
  id?: string
  props?: Record<string, any>
  children?: ComponentConfig[]
}

interface PresetConfig {
  components: ComponentConfig[]
}

// Default transform values
const DEFAULT_TRANSFORM = {
  offsetX: 0,
  offsetY: 0,
  rotation: 0,
  scale: 1,
  anchorX: 0.5,
  anchorY: 0.5,
  edges: 'transparent'
}

// Bounding-box reduction helper (reduceBoundingBoxForExport) is injected here from generate-components.ts:
// Reduce a boundingBox prop for export: drop keys still at their full-frame defaults and strip
// editor-only fields (lockAspect — a Design Editor resize hint the renderer ignores). Returns
// null when every key is default: the box is the full-frame identity and the prop should be
// omitted entirely. The runtime wrappers merge partial boxes back with the same defaults, so
// emitting only the changed keys round-trips exactly.
function reduceBoundingBoxForExport(bb: any): Record<string, any> | null {
  if (!bb || typeof bb !== 'object') return null
  // A zero offset is zero in any unit; a full-frame extent is only knowable in UV ('%' is its UI alias)
  const isUVUnit = (u: any) => u === undefined || u === 'uv' || u === '%'
  const isZeroDim = (d: any) => !!d && typeof d === 'object' && Number(d.value) === 0
  const isFullDim = (d: any) => !!d && typeof d === 'object' && Number(d.value) === 1 && isUVUnit(d.unit)
  const out: Record<string, any> = {}
  if (bb.x !== undefined && !isZeroDim(bb.x)) out.x = bb.x
  if (bb.y !== undefined && !isZeroDim(bb.y)) out.y = bb.y
  if (bb.width !== undefined && !isFullDim(bb.width)) out.width = bb.width
  if (bb.height !== undefined && !isFullDim(bb.height)) out.height = bb.height
  if (bb.origin !== undefined && bb.origin !== 'top-left') out.origin = bb.origin
  if (bb.rotation !== undefined && Number(bb.rotation) !== 0) out.rotation = bb.rotation
  if (bb.cornerRadius !== undefined && !isZeroDim(bb.cornerRadius)) out.cornerRadius = bb.cornerRadius
  return Object.keys(out).length > 0 ? out : null
}

// @ts-ignore
const shaderMetadata: Record<string, Record<string, any>> = {
  "AngularBlur": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 20,
    "center": {
      "x": 0.5,
      "y": 0.5
    }
  },
  "Arc": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "color": "#ffffff",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "radius": 0.38,
    "aperture": 270,
    "rotation": 0,
    "softness": 0,
    "strokeThickness": 0,
    "strokeColor": "#000000",
    "strokePosition": "center",
    "colorSpace": "linear"
  },
  "Ascii": {
    "opacity": 1,
    "blendMode": "normal",
    "characters": "@%#*+=-:.",
    "cellSize": 30,
    "fontFamily": "JetBrains Mono",
    "spacing": 1,
    "gamma": 1,
    "alphaThreshold": 0,
    "preserveAlpha": true
  },
  "Aurora": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#a533f8",
    "colorB": "#22ee88",
    "colorC": "#1694e8",
    "colorSpace": "linear",
    "balance": 50,
    "intensity": 80,
    "curtainCount": 4,
    "speed": 5,
    "waviness": 50,
    "rayDensity": 20,
    "height": 120,
    "center": {
      "x": 0.5,
      "y": 0
    },
    "seed": 0
  },
  "BarnDoors": {
    "opacity": 1,
    "blendMode": "normal",
    "progress": 0.5,
    "angle": 0,
    "feather": 0.1,
    "invert": false
  },
  "BarShift": {
    "opacity": 1,
    "blendMode": "normal",
    "count": 6,
    "angle": 0,
    "intensity": 0.15,
    "seed": 0,
    "speed": 0,
    "edges": "mirror"
  },
  "Beam": {
    "opacity": 1,
    "blendMode": "normal",
    "startPosition": {
      "x": 0.2,
      "y": 0.5
    },
    "endPosition": {
      "x": 0.8,
      "y": 0.5
    },
    "startThickness": 0.2,
    "endThickness": 0.2,
    "startSoftness": 0.5,
    "endSoftness": 0.5,
    "insideColor": "#FF0000",
    "outsideColor": "#0000FF",
    "colorSpace": "linear"
  },
  "Bend": {
    "opacity": 1,
    "blendMode": "normal",
    "strength": 0.5,
    "falloff": 0,
    "angle": 0,
    "edges": "transparent"
  },
  "Blob": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "colorA": "#ff6b35",
    "colorB": "#e91e63",
    "stops": null,
    "size": 0.5,
    "deformation": 0.5,
    "softness": 0.5,
    "highlightIntensity": 0.5,
    "highlightX": 0.3,
    "highlightY": -0.3,
    "highlightZ": 0.4,
    "highlightColor": "#ffe11a",
    "speed": 0.5,
    "seed": 1,
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "colorSpace": "linear"
  },
  "BlockDissolve": {
    "opacity": 1,
    "blendMode": "normal",
    "progress": 0.5,
    "blockSize": 0.08,
    "softness": 0.15,
    "invert": false
  },
  "BlockNoise": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#ffffff",
    "colorB": "#000000",
    "stops": null,
    "colorSpace": "linear",
    "scale": 2,
    "contrast": 0,
    "balance": 0,
    "seed": 0,
    "speed": 1
  },
  "BlueNoise": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#ffffff",
    "colorB": "#000000",
    "stops": null,
    "colorSpace": "linear",
    "grain": 1,
    "contrast": 0,
    "balance": 0,
    "seed": 0
  },
  "Blur": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 50
  },
  "Boids": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#8ec5ff",
    "colorB": "#ff7ad9",
    "colorSpace": "oklab",
    "agentShape": "arrow",
    "count": 2000,
    "speed": 2,
    "seed": 0,
    "size": 1.5,
    "trails": 0,
    "separation": 1.7,
    "alignment": 1.5,
    "cohesion": 1,
    "perception": 0.12,
    "cursorMode": "repel",
    "cursorStrength": 1.5
  },
  "BokehBlur": {
    "opacity": 1,
    "blendMode": "normal",
    "radius": 50,
    "highlightGain": 4,
    "highlightThreshold": 0.6,
    "bladeShape": "blades",
    "bladeCount": 6,
    "bladeRotation": 0,
    "chromaticFringe": 0.2
  },
  "BrickPattern": {
    "opacity": 1,
    "blendMode": "normal",
    "colorBrick": "#000000",
    "colorMortar": "#ffffff",
    "cellsX": 8,
    "cellsY": 10,
    "mortar": 0.05,
    "softness": 0,
    "variation": 0,
    "rotation": 0,
    "speed": 0,
    "offset": 0,
    "speedVariance": 0,
    "seed": 0,
    "colorSpace": "linear"
  },
  "BrightnessContrast": {
    "opacity": 1,
    "blendMode": "normal",
    "brightness": 0,
    "contrast": 0
  },
  "BrushedMetal": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "lightColor": "#e9ebef",
    "darkColor": "#26282d",
    "brushAngle": 0,
    "anisotropy": 0.4,
    "grain": 0.4,
    "grainScale": 2,
    "roughness": 0.5,
    "environment": 1.4,
    "envRotation": 0,
    "lightAngle": 215,
    "speed": 0.4,
    "bevelWidth": 0.05,
    "bevelShape": 0,
    "edgeSoftness": 0.05,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "Bulge": {
    "opacity": 1,
    "blendMode": "normal",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "strength": 1,
    "radius": 1,
    "falloff": 0.5,
    "edges": "stretch"
  },
  "CarbonFiber": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "lightColor": "#b9bdc6",
    "darkColor": "#0b0c0e",
    "weaveStyle": "twill",
    "weaveScale": 30,
    "weaveAngle": 0,
    "relief": 0.5,
    "fiberSheen": 1.2,
    "roughness": 0.5,
    "clearcoat": 1.2,
    "environment": 2,
    "envRotation": 0,
    "lightAngle": 215,
    "bevelWidth": 0.05,
    "bevelShape": 0,
    "edgeSoftness": 0.05,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "Chalkboard": {
    "opacity": 1,
    "blendMode": "normal",
    "boardColor": "#000000",
    "chalkColor": "#eceadb",
    "edgeSensitivity": 0.5,
    "edgeThickness": 1.5,
    "shading": 0.05,
    "hatchScale": 16,
    "grain": 0.45
  },
  "ChannelBlur": {
    "opacity": 1,
    "blendMode": "normal",
    "redIntensity": 0,
    "greenIntensity": 20,
    "blueIntensity": 40
  },
  "Checkerboard": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#cccccc",
    "colorB": "#999999",
    "cells": 8,
    "softness": 0,
    "colorSpace": "linear"
  },
  "CheckerWipe": {
    "opacity": 1,
    "blendMode": "normal",
    "progress": 0.5,
    "blockSize": 0.1,
    "softness": 0.15,
    "invert": false
  },
  "Chevron": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#000000",
    "colorB": "#ffffff",
    "count": 5,
    "angle": 0,
    "balance": 0.5,
    "softness": 0,
    "speed": 0,
    "offset": 0,
    "colorSpace": "linear"
  },
  "ChromaFlow": {
    "opacity": 1,
    "blendMode": "normal",
    "baseColor": "#0066ff",
    "upColor": "#00ff00",
    "downColor": "#ff0000",
    "leftColor": "#0000ff",
    "rightColor": "#ffff00",
    "intensity": 1,
    "radius": 3,
    "momentum": 30
  },
  "ChromaticAberration": {
    "opacity": 1,
    "blendMode": "normal",
    "strength": 0.2,
    "angle": 0,
    "redOffset": -1,
    "greenOffset": 0,
    "blueOffset": 1
  },
  "Chrome": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "tint": "#ffffff",
    "warmColor": "#ffa94d",
    "coolColor": "#82a7e6",
    "bevelWidth": 0.028,
    "bevelShape": 0.55,
    "curvature": 0.5,
    "waviness": 0.15,
    "environment": 1,
    "envRotation": 0,
    "softness": 0.3,
    "spectral": 0.9,
    "dispersion": 0.3,
    "shadows": 0.7,
    "speed": 0.5,
    "edgeSoftness": 0.05,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "Circle": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "color": "#ffffff",
    "radius": 1,
    "softness": 0,
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "strokeThickness": 0,
    "strokeColor": "#000000",
    "strokePosition": "center",
    "colorSpace": "linear"
  },
  "ColorWheel": {
    "opacity": 1,
    "blendMode": "normal",
    "mode": "rainbow",
    "colorA": "#ff0000",
    "colorB": "#00ff88",
    "colorC": "#0066ff",
    "scale": 1,
    "angle": 0,
    "speed": 0.05,
    "colorSpace": "oklch"
  },
  "CompressionArtifacts": {
    "opacity": 1,
    "blendMode": "normal",
    "quality": 12
  },
  "ConcentricSpin": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 20,
    "rings": 8,
    "smoothness": 0.03,
    "seed": 0,
    "speed": 0.1,
    "speedRandomness": 0.5,
    "edges": "mirror",
    "center": {
      "x": 0.5,
      "y": 0.5
    }
  },
  "ConicGradient": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#FF0080",
    "colorB": "#00BFFF",
    "stops": null,
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "rotation": 0,
    "repeat": 1,
    "colorSpace": "linear"
  },
  "ContourLines": {
    "opacity": 1,
    "blendMode": "normal",
    "levels": 5,
    "lineWidth": 2,
    "softness": 0,
    "gamma": 0.5,
    "invert": false,
    "source": "luminance",
    "colorMode": "source",
    "lineColor": "#000000",
    "backgroundColor": "transparent"
  },
  "CornerPin": {
    "opacity": 1,
    "blendMode": "normal",
    "topLeft": {
      "x": 0,
      "y": 0
    },
    "topRight": {
      "x": 1,
      "y": 0
    },
    "bottomLeft": {
      "x": 0,
      "y": 1
    },
    "bottomRight": {
      "x": 1,
      "y": 1
    },
    "amount": 1,
    "edges": "transparent"
  },
  "Crescent": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "color": "#ffffff",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "radius": 0.3,
    "innerRatio": 0.8,
    "offset": 0.2,
    "rotation": 0,
    "softness": 0,
    "strokeThickness": 0,
    "strokeColor": "#000000",
    "strokePosition": "center",
    "colorSpace": "linear"
  },
  "Cross": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "color": "#ffffff",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "radius": 0.35,
    "thickness": 0.08,
    "rounding": 0,
    "rotation": 0,
    "softness": 0,
    "strokeThickness": 0,
    "strokeColor": "#000000",
    "strokePosition": "center",
    "colorSpace": "linear"
  },
  "CRTScreen": {
    "opacity": 1,
    "blendMode": "normal",
    "pixelSize": 128,
    "colorShift": 1,
    "scanlineIntensity": 0.3,
    "scanlineFrequency": 200,
    "brightness": 1,
    "contrast": 1,
    "vignetteIntensity": 1,
    "vignetteRadius": 0.5
  },
  "Crystal": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "cutout": false,
    "refraction": 0.5,
    "dispersion": 0.5,
    "facets": 5,
    "fresnel": 0.05,
    "fresnelSoftness": 1,
    "fresnelColor": "#ffffff",
    "edgeSoftness": 0,
    "innerZoom": 1.5,
    "lightAngle": 270,
    "highlights": 0.5,
    "shadows": 0.3,
    "brightness": 1.2,
    "tintColor": "#e8e0ff",
    "tintIntensity": 0,
    "tintPreserveLuminosity": true,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "CurlNoise": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#ffffff",
    "colorB": "#000000",
    "stops": null,
    "colorSpace": "linear",
    "scale": 2,
    "contrast": 0,
    "balance": 0,
    "seed": 0,
    "speed": 1
  },
  "CursorRipples": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 10,
    "decay": 10,
    "radius": 0.5,
    "chromaticSplit": 1,
    "edges": "stretch"
  },
  "CursorTrail": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#00aaff",
    "colorB": "#ff00aa",
    "stops": null,
    "radius": 0.5,
    "length": 0.5,
    "shrink": 1,
    "softness": 0,
    "colorSpace": "linear"
  },
  "DataMosh": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 0.7,
    "blockSize": 48,
    "drift": 0.35,
    "churn": 0.4,
    "blend": 1,
    "speed": 1,
    "seed": 0
  },
  "DiamondGradient": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#4ffb4a",
    "colorB": "#4f1238",
    "stops": null,
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "size": 0.7,
    "rotation": 0,
    "repeat": 1,
    "roundness": 0,
    "colorSpace": "linear"
  },
  "DiamondWipe": {
    "opacity": 1,
    "blendMode": "normal",
    "progress": 0.5,
    "size": 0.15,
    "feather": 0.1,
    "invert": false
  },
  "DiffuseBlur": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 30,
    "edges": "stretch"
  },
  "DisplacementMap": {
    "opacity": 1,
    "blendMode": "normal",
    "source": "",
    "amount": 0.3,
    "channelMode": "twoAxis",
    "angle": 0,
    "edges": "mirror"
  },
  "Dither": {
    "opacity": 1,
    "blendMode": "normal",
    "pattern": "bayer4",
    "pixelSize": 4,
    "threshold": 0.5,
    "spread": 1,
    "colorMode": "custom",
    "colorA": "transparent",
    "colorB": "#ffffff"
  },
  "DotGrid": {
    "opacity": 1,
    "blendMode": "normal",
    "color": "#ffffff",
    "density": 30,
    "dotSize": 0.3,
    "offset": 0,
    "speed": 0,
    "speedVariance": 0.3,
    "twinkle": 0
  },
  "DropShadow": {
    "opacity": 1,
    "blendMode": "normal",
    "color": "#000000",
    "distance": 0.1,
    "angle": 135,
    "blur": 5,
    "intensity": 0.5,
    "cutout": false
  },
  "Duotone": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#ff0000",
    "colorB": "#023af4",
    "blend": 0.5,
    "colorSpace": "linear"
  },
  "Ellipse": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "color": "#ffffff",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "radiusX": 0.35,
    "radiusY": 0.2,
    "rotation": 0,
    "softness": 0,
    "strokeThickness": 0,
    "strokeColor": "#000000",
    "strokePosition": "center",
    "colorSpace": "linear"
  },
  "Emboss": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "depth": -0.5,
    "lightAngle": 260,
    "lightIntensity": 0.6,
    "shadowIntensity": 0.3,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "Engraving": {
    "opacity": 1,
    "blendMode": "normal",
    "style": "crosshatch",
    "frequency": 90,
    "angle": 8,
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "relief": 0.6,
    "waviness": 0.35,
    "contrast": 1.15,
    "inkColor": "#1a1410",
    "paperColor": "#f4eee2"
  },
  "ErosionNoise": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#ffffff",
    "colorB": "#000000",
    "stops": null,
    "colorSpace": "linear",
    "scale": 1.5,
    "contrast": 0,
    "balance": 0,
    "seed": 0
  },
  "Exposure": {
    "opacity": 1,
    "blendMode": "normal",
    "exposure": 1
  },
  "FallingLines": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#ffffff",
    "colorB": "#ffffff00",
    "colorSpace": "linear",
    "angle": 90,
    "speed": 0.5,
    "speedVariance": 0.3,
    "density": 15,
    "trailLength": 0.35,
    "balance": 0.5,
    "strokeWidth": 0.15,
    "rounding": 1
  },
  "FilmGrain": {
    "opacity": 1,
    "blendMode": "normal",
    "strength": 0.5,
    "bias": 2,
    "animated": false
  },
  "FilmStock": {
    "opacity": 1,
    "blendMode": "normal",
    "stock": "portrait400",
    "strength": 1,
    "halation": 0.4,
    "halationRadius": 28,
    "weave": 0
  },
  "Flip": {
    "opacity": 1,
    "blendMode": "normal",
    "flipX": false,
    "flipY": false
  },
  "FloatingParticles": {
    "opacity": 1,
    "blendMode": "normal",
    "particleColor": "#ffffff",
    "shape": "dot",
    "count": 1200,
    "particleSize": 1.2,
    "softness": 0.1,
    "speed": 0.25,
    "angle": 90,
    "speedVariance": 0.3,
    "angleVariance": 30,
    "randomness": 0.25,
    "twinkle": 0.5,
    "cursorStrength": 0
  },
  "Flower": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "color": "#ffffff",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "radius": 0.4,
    "sides": 5,
    "innerRatio": 0.4,
    "rotation": 0,
    "softness": 0,
    "strokeThickness": 0,
    "strokeColor": "#000000",
    "strokePosition": "center",
    "colorSpace": "linear"
  },
  "FlowField": {
    "opacity": 1,
    "blendMode": "normal",
    "strength": 0.15,
    "detail": 2,
    "speed": 0,
    "evolutionSpeed": 0,
    "seed": 0,
    "edges": "mirror"
  },
  "FlowingGradient": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#0a0015",
    "colorB": "#6b17e6",
    "colorC": "#ff4d6a",
    "colorD": "#ff6b35",
    "colorSpace": "oklch",
    "speed": 1,
    "distortion": 0.5,
    "seed": 0
  },
  "FlutedGlass": {
    "opacity": 1,
    "blendMode": "normal",
    "shape": "bars",
    "angle": 0,
    "frequency": 10,
    "softness": 0.5,
    "waveAmplitude": 0.06,
    "waveFrequency": 1.5,
    "speed": 0,
    "refraction": 1.5,
    "aberration": 0.2,
    "lightAngle": 30,
    "highlight": 0.2,
    "highlightSoftness": 0.3,
    "highlightColor": "#ffffff",
    "edges": "mirror"
  },
  "Fog": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#e0e0e0",
    "colorB": "#888888",
    "seed": 0,
    "speed": 1,
    "turbulence": 1,
    "detail": 15,
    "blending": 0.3,
    "mouseInfluence": 0.1,
    "mouseRadius": 0.1,
    "colorSpace": "linear"
  },
  "Form3D": {
    "opacity": 1,
    "blendMode": "normal",
    "shape3d": "{\"type\":\"ribbon\",\"angle\":0,\"twist\":50,\"width\":40,\"thickness\":20,\"seed\":0}",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "zoom": 50,
    "glossiness": 50,
    "lighting": 50,
    "uvMode": "stretch",
    "speed": 1
  },
  "FractalNoise": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#000000",
    "colorB": "#ffffff",
    "stops": null,
    "octaves": 4,
    "detail": 2,
    "contrast": 0.5,
    "speed": 0.15,
    "angle": 0,
    "seed": 0,
    "colorSpace": "linear"
  },
  "Frost": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "iceColor": "#0597fc",
    "ambient": 0,
    "edgeSoftness": 0.05,
    "density": 5,
    "absorption": 1,
    "scatter": 0.5,
    "frostAmount": 0.8,
    "frostDepth": 0.4,
    "frostScale": 1,
    "frostRoughness": 3,
    "sparkle": 0.5,
    "gloss": 0.6,
    "fresnel": 0.02,
    "lightAngle": 300,
    "speed": 0.3,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "GaborNoise": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#ffffff",
    "colorB": "#000000",
    "stops": null,
    "colorSpace": "linear",
    "scale": 1.5,
    "frequency": 8,
    "contrast": 0,
    "balance": 0,
    "seed": 0,
    "speed": 1
  },
  "Glass": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "cutout": false,
    "refraction": 1,
    "edgeSoftness": 0.1,
    "blur": 0,
    "thickness": 0.2,
    "aberration": 0.5,
    "innerZoom": 1,
    "lightAngle": 300,
    "highlight": 0.05,
    "highlightColor": "#ffffff",
    "highlightSoftness": 0.5,
    "fresnel": 0.1,
    "fresnelSoftness": 0.1,
    "fresnelColor": "#ffffff",
    "tintColor": "#ffffff",
    "tintIntensity": 0,
    "tintPreserveLuminosity": true,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "GlassTiles": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 2,
    "tileCount": 20,
    "rotation": 0,
    "roundness": 0
  },
  "Glitch": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 0.5,
    "speed": 1,
    "rgbShift": 5,
    "blockDensity": 10,
    "colorBarIntensity": 0.2,
    "mirrorAmount": 0.3,
    "scanlineIntensity": 0.2
  },
  "Glow": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 1,
    "threshold": 0.5,
    "size": 25
  },
  "Godrays": {
    "opacity": 1,
    "blendMode": "normal",
    "center": {
      "x": 0,
      "y": 0
    },
    "density": 0.3,
    "intensity": 0.8,
    "spotty": 1,
    "speed": 0.5,
    "rayColor": "#4283fb",
    "backgroundColor": "transparent"
  },
  "Goo": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "gooColor": "#37c95a",
    "translucency": 0.6,
    "absorb": 2,
    "containment": 0.9,
    "edgeSoftness": 0.06,
    "blobScale": 0.3,
    "spread": 1,
    "merge": 0.6,
    "bulge": 1,
    "threshold": 0.5,
    "lightAngle": 300,
    "wetness": 2,
    "fresnel": 0.25,
    "specColor": "#ffffff",
    "ambient": 0.25,
    "speed": 0.5,
    "wobble": 0.5,
    "breathe": 0.3,
    "seed": 1,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "GradientMap": {
    "opacity": 1,
    "blendMode": "normal",
    "palette": "rainbow",
    "colorLow": "#1a0b2e",
    "colorMid": "#e94560",
    "colorHigh": "#f9ed69",
    "speed": 0.15,
    "contrast": 1,
    "blackPoint": 0,
    "whitePoint": 1,
    "strength": 1,
    "colorSpace": "oklch"
  },
  "Grayscale": {
    "opacity": 1,
    "blendMode": "normal"
  },
  "Grid": {
    "opacity": 1,
    "blendMode": "normal",
    "color": "#ffffff",
    "cellColor": "transparent",
    "cells": 10,
    "thickness": 1,
    "rotation": 0,
    "softness": 0,
    "variation": 0,
    "colorSpace": "linear"
  },
  "GridDistortion": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 1,
    "decay": 3,
    "radius": 1,
    "gridSize": 20,
    "edges": "stretch"
  },
  "Group": {
    "opacity": 1,
    "blendMode": "normal"
  },
  "Halftone": {
    "opacity": 1,
    "blendMode": "normal",
    "style": "classic",
    "frequency": 100,
    "angle": 45,
    "cyanAngle": 15,
    "magentaAngle": 75,
    "yellowAngle": 0,
    "blackAngle": 45,
    "misprint": 0,
    "misprintAngle": 0,
    "paperColor": "#ffffff",
    "cyanColor": "#00ffff",
    "magentaColor": "#ff00ff",
    "yellowColor": "#ffff00",
    "blackColor": "#000000"
  },
  "Heart": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "color": "#ffffff",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "radius": 0.32,
    "rotation": 0,
    "softness": 0,
    "strokeThickness": 0,
    "strokeColor": "#000000",
    "strokePosition": "center",
    "colorSpace": "linear"
  },
  "Heatmap": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "colorA": "#02010f",
    "colorB": "#f9e25f",
    "stops": [
      {
        "color": "#02010f",
        "position": 0
      },
      {
        "color": "#2a0a8a",
        "position": 0.2
      },
      {
        "color": "#a41c9b",
        "position": 0.45
      },
      {
        "color": "#e8632b",
        "position": 0.7
      },
      {
        "color": "#f9e25f",
        "position": 0.9
      },
      {
        "color": "#ffffff",
        "position": 1
      }
    ],
    "colorSpace": "oklab",
    "innerGlow": 0.4,
    "outerGlow": 0.2,
    "contour": 0.5,
    "angle": 90,
    "speed": 1,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "HexGrid": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#000000",
    "colorB": "#ffffff",
    "cells": 8,
    "thickness": 1,
    "rotation": 0,
    "softness": 0,
    "variation": 0,
    "colorSpace": "linear"
  },
  "Hologram": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "color": "#5ec8ff",
    "brightness": 2.5,
    "edgeSoftness": 0.04,
    "fill": 0.45,
    "edgeGlow": 1,
    "depthLines": 0.5,
    "depthScale": 26,
    "scanlines": 0.55,
    "scanlineScale": 130,
    "scanlineSpeed": 1,
    "sweep": 0.1,
    "flicker": 0.3,
    "distortion": 0.1,
    "grain": 0.7,
    "speed": 1,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "Holographic": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "hueShift": 0,
    "foilScale": 1.5,
    "saturation": 0.75,
    "roughness": 0.25,
    "speed": 1,
    "crinkle": 0.5,
    "crinkleScale": 1,
    "sparkle": 0.4,
    "edgeSoftness": 0.05,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "HTMLInCanvas": {
    "opacity": 1,
    "blendMode": "normal"
  },
  "HueShift": {
    "opacity": 1,
    "blendMode": "normal",
    "shift": 0
  },
  "ImageTexture": {
    "opacity": 1,
    "blendMode": "normal",
    "url": "https://shaders.com/sample.jpg",
    "objectFit": "fill"
  },
  "InkFlow": {
    "opacity": 1,
    "blendMode": "normal",
    "colorMode": "rainbow",
    "colorSpeed": 1,
    "color": "#ff2d7e",
    "color1": "#4338ff",
    "color2": "#ff2d7e",
    "color3": "#19e3ff",
    "colorSpace": "oklab",
    "radius": 0.3,
    "force": 1,
    "curl": 0,
    "decay": 0.5,
    "momentum": 0.6
  },
  "Invert": {
    "opacity": 1,
    "blendMode": "normal"
  },
  "IrisWipe": {
    "opacity": 1,
    "blendMode": "normal",
    "progress": 0.5,
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "feather": 0.1,
    "invert": false
  },
  "Irradiance": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "lights": [
      {
        "position": {
          "x": 0.3,
          "y": 0.3
        },
        "color": "#ffb347",
        "intensity": 3
      },
      {
        "position": {
          "x": 0.72,
          "y": 0.7
        },
        "color": "#7a5cff",
        "intensity": 2
      }
    ],
    "lightHeight": 0.5,
    "lightRange": 1.2,
    "reach": 3,
    "core": 1,
    "wrap": 0.1,
    "shadows": true,
    "shadowSoftness": 0.15,
    "bodyColor": "#000000",
    "bodyLight": 0.6,
    "bevelWidth": 0.01,
    "edgeSoftness": 0.05,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "IsometricCubes": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#7c5cff",
    "colorB": "#ff5c9d",
    "lineColor": "#000000",
    "cells": 6,
    "thickness": 1,
    "rotation": 0,
    "softness": 0,
    "colorVariation": 1,
    "colorSpace": "linear"
  },
  "Kaleidoscope": {
    "opacity": 1,
    "blendMode": "normal",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "segments": 6,
    "angle": 0,
    "edges": "mirror"
  },
  "KeyFrames": {
    "opacity": 1,
    "blendMode": "normal",
    "trackers": 12,
    "detect": "bright",
    "threshold": 0.15,
    "variance": 0.8,
    "agility": 0.5,
    "lifespan": 6,
    "trail": 1,
    "markerSize": 28,
    "lineWidth": 1.5,
    "markerColor": "#54d66b",
    "keyframeColor": "#ffcf33",
    "pathColor": "#ffffffb8"
  },
  "LensDistortion": {
    "opacity": 1,
    "blendMode": "normal",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "spread": 0.6,
    "angle": 0,
    "perspective": 0.1,
    "bias": 1,
    "count": 35,
    "dispersion": 1,
    "dispersionShift": 0,
    "dispersionColor": 0.6,
    "focusCenter": 0.8,
    "focusEdges": 1,
    "lensBulge": 0,
    "lensCircle": 0,
    "swirl": 0.35,
    "noise": 0,
    "noiseFrequency": 0.25,
    "noiseOffset": 0,
    "grainMixer": 0,
    "grainOverlay": 0
  },
  "LensFlare": {
    "opacity": 1,
    "blendMode": "normal",
    "lightPosition": {
      "x": 0.3,
      "y": 0.3
    },
    "intensity": 0.5,
    "ghostIntensity": 0.4,
    "ghostSpread": 0.7,
    "ghostChroma": 0.3,
    "haloIntensity": 0.4,
    "haloRadius": 0.6,
    "haloChroma": 0.6,
    "haloSoftness": 0.8,
    "starburstIntensity": 0.3,
    "starburstPoints": 6,
    "streakIntensity": 0.15,
    "streakLength": 0.5,
    "glareIntensity": 0.2,
    "glareSize": 0.5,
    "edgeFade": 0.2,
    "speed": 0.5
  },
  "LightEdge": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "colorA": "#7b2ff7",
    "colorB": "#00e0ff",
    "stops": [
      {
        "color": "#7b2ff7",
        "position": 0
      },
      {
        "color": "#00e0ff",
        "position": 0.5
      },
      {
        "color": "#ff3d81",
        "position": 1
      }
    ],
    "colorSpace": "linear",
    "thickness": 0.25,
    "softness": 0.5,
    "intensity": 0.6,
    "bloom": 0.3,
    "spots": 3,
    "spotSize": 0.5,
    "pulse": 0,
    "smoke": 0.3,
    "smokeSize": 0.5,
    "speed": 1,
    "seed": 0,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "LightLeak": {
    "opacity": 1,
    "blendMode": "normal",
    "position": {
      "x": 0.95,
      "y": 0.4
    },
    "spread": 0.55,
    "intensity": 0.15,
    "streaks": 0.6,
    "colorHot": "#fff3c4",
    "colorMid": "#ff7a2f",
    "colorFringe": "#a63d8f",
    "flicker": 0.35,
    "speed": 1,
    "seed": 0
  },
  "Line": {
    "opacity": 1,
    "blendMode": "normal",
    "color": "#ffffff",
    "pointA": {
      "x": 0.2,
      "y": 0.5
    },
    "pointB": {
      "x": 0.8,
      "y": 0.5
    },
    "thickness": 0.01,
    "style": "solid",
    "dashLength": 0.05,
    "gapLength": 0.025,
    "capStart": "rounded",
    "capEnd": "rounded"
  },
  "LinearBlur": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 30,
    "angle": 0
  },
  "LinearGradient": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#1aff00",
    "colorB": "#0000ff",
    "stops": null,
    "start": {
      "x": 0,
      "y": 0.5
    },
    "end": {
      "x": 1,
      "y": 0.5
    },
    "angle": 0,
    "edges": "stretch",
    "colorSpace": "linear"
  },
  "LinearWipe": {
    "opacity": 1,
    "blendMode": "normal",
    "progress": 0.5,
    "angle": 0,
    "feather": 0.1,
    "invert": false
  },
  "LiquidMetal": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "lightColor": "#eef0f6",
    "darkColor": "#141414",
    "turbulence": 1,
    "ripple": 4,
    "warp": 1.5,
    "speed": 0.5,
    "environment": 1.5,
    "envRotation": 0,
    "sharpness": 0.6,
    "dispersion": 0.25,
    "lightAngle": 265,
    "bevelWidth": 0.05,
    "bevelShape": 0,
    "edgeSoftness": 0.05,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "Liquify": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 10,
    "stiffness": 3,
    "damping": 3,
    "radius": 1,
    "edges": "stretch"
  },
  "MagneticFilings": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#be1ef9",
    "colorB": "#6326ff",
    "colorSpace": "oklab",
    "shape": "dot",
    "count": 8000,
    "size": 1,
    "restOrientation": "random",
    "fieldType": "dipole",
    "strength": 1.7,
    "reach": 0.6,
    "response": 0.5
  },
  "Marble": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#ffffff",
    "colorB": "#3a2d54",
    "colorC": "#0f0f0f",
    "scale": 2,
    "turbulence": 10,
    "speed": 0.05,
    "seed": 0,
    "colorSpace": "linear"
  },
  "MeshGradient": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#1a0533",
    "colorB": "#ffdf8e",
    "stops": [
      {
        "color": "#1a0533",
        "position": 0
      },
      {
        "color": "#6d2fd1",
        "position": 0.26
      },
      {
        "color": "#e04b9e",
        "position": 0.52
      },
      {
        "color": "#ff8c42",
        "position": 0.76
      },
      {
        "color": "#ffdf8e",
        "position": 1
      }
    ],
    "colorSpace": "oklab",
    "count": 5,
    "smoothness": 2,
    "variation": 0.35,
    "swirl": 0.3,
    "drift": 0.5,
    "wrapping": 0,
    "speed": 1,
    "seed": 0
  },
  "Mirror": {
    "opacity": 1,
    "blendMode": "normal",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "angle": 0,
    "edges": "mirror"
  },
  "MultiPointGradient": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#4776E6",
    "positionA": {
      "x": 0.2,
      "y": 0.2
    },
    "colorB": "#C44DFF",
    "positionB": {
      "x": 0.8,
      "y": 0.2
    },
    "colorC": "#1ABC9C",
    "positionC": {
      "x": 0.2,
      "y": 0.8
    },
    "colorD": "#F8BBD9",
    "positionD": {
      "x": 0.8,
      "y": 0.8
    },
    "colorE": "#FF8C42",
    "positionE": {
      "x": 0.5,
      "y": 0.5
    },
    "colorSpace": "linear",
    "smoothness": 2
  },
  "Nebula": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "coreColor": "#ffd9b0",
    "gasColor": "#ff5e7a",
    "veilColor": "#3a5bdb",
    "colorSpace": "oklab",
    "density": 1,
    "cavity": 0,
    "dust": 0.55,
    "gasScale": 0.4,
    "billow": 0.6,
    "glow": 0,
    "seed": 0,
    "stars": 0.07,
    "starScale": 1,
    "twinkle": 0.35,
    "refraction": 0.5,
    "environment": 0.7,
    "highlight": 1.15,
    "highlightSoftness": 0.5,
    "lightAngle": 315,
    "edgeSoftness": 0.05,
    "speed": 0.3,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "Neon": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "color": "#00ddff",
    "secondaryColor": "#ff00aa",
    "secondaryBlend": 0.5,
    "glowColor": "#00ddff",
    "tubeThickness": 0.2,
    "intensity": 1.5,
    "hotCoreIntensity": 0.6,
    "glowIntensity": 0.6,
    "glowRadius": 0.25,
    "lightAngle": 300,
    "specularIntensity": 0.5,
    "specularSize": 0.5,
    "cornerSmoothing": 0.15,
    "flickerSpeed": 0,
    "flickerAmount": 0.2,
    "flowSpeed": 0,
    "flowAmount": 0.3,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "NoiseDissolve": {
    "opacity": 1,
    "blendMode": "normal",
    "progress": 0.5,
    "scale": 3,
    "softness": 0.25,
    "seed": 0,
    "invert": false
  },
  "ObjectTracker": {
    "opacity": 1,
    "blendMode": "normal",
    "detectionMode": "bright",
    "threshold": 0.25,
    "layout": "mosaic",
    "cellSize": 400,
    "maxDepth": 2,
    "boxStyle": "corners",
    "lineWidth": 1.5,
    "cornerRadius": 0,
    "strokeColor": "#ffffff",
    "fillColor": "#5a79911a",
    "labelColor": "#000000",
    "labelBackgroundColor": "#ffffff",
    "labelMode": "none",
    "labelPosition": "bottom-right",
    "labelRadius": 0,
    "labelInset": true,
    "fontFamily": "Inter",
    "fontWeight": 500,
    "fontSize": 0.015,
    "letterSpacing": 0
  },
  "Obsidian": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "bodyColor": "#06071a",
    "colorA": "#ff2f7a",
    "colorB": "#5fe6ff",
    "stops": [
      {
        "color": "#ff2f7a",
        "position": 0
      },
      {
        "color": "#2a2cff",
        "position": 0.3
      },
      {
        "color": "#5fe6ff",
        "position": 0.55
      },
      {
        "color": "#a6ffd8",
        "position": 0.75
      },
      {
        "color": "#ffb3e6",
        "position": 1
      }
    ],
    "colorSpace": "oklab",
    "iridescence": 1,
    "rimWidth": 0.55,
    "rimSoftness": 0.25,
    "flowAngle": 30,
    "flowScale": 1,
    "speed": 0.5,
    "gloss": 0.35,
    "envRotation": 0,
    "bevelWidth": 0.08,
    "bevelShape": 0,
    "edgeSoftness": 0.05,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "PagePeel": {
    "opacity": 1,
    "blendMode": "normal",
    "corner": "bottom-right",
    "amount": 0.2,
    "radius": 0.2,
    "shading": 0.55,
    "highlight": 0.4,
    "highlightSoftness": 0.2,
    "shadow": 1
  },
  "Paper": {
    "opacity": 1,
    "blendMode": "normal",
    "roughness": 0.3,
    "grainScale": 1,
    "displacement": 0.15,
    "seed": 0
  },
  "Parallelogram": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "color": "#ffffff",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "width": 0.32,
    "height": 0.22,
    "skew": 0.15,
    "rotation": 0,
    "softness": 0,
    "strokeThickness": 0,
    "strokeColor": "#000000",
    "strokePosition": "center",
    "colorSpace": "linear"
  },
  "ParticleField": {
    "opacity": 1,
    "blendMode": "normal",
    "count": 15000,
    "depthSource": "luminance",
    "depth": 0.7,
    "particleShape": "dot",
    "particleSize": 0.75,
    "depthShading": 0.6,
    "wobble": 0.35,
    "zoom": 1,
    "rotationX": 0,
    "rotationY": 0,
    "rotationZ": 0,
    "offsetX": 0,
    "offsetY": 0,
    "cursorMode": "push",
    "cursorStrength": 0.8,
    "cursorRadius": 0.2
  },
  "ParticleFlow": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#8ec5ff",
    "colorB": "#ffd98e",
    "colorSpace": "oklab",
    "shape": "streak",
    "count": 6000,
    "size": 1.2,
    "trails": 0,
    "speed": 1,
    "force": 1,
    "swirl": 25,
    "momentum": 0.6,
    "ambient": 0.15
  },
  "Particles": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "colorA": "#8ec5ff",
    "colorB": "#ff7ad9",
    "count": 4000,
    "size": 1.5,
    "particleShape": "dot",
    "spread": 1,
    "agitation": 0.12,
    "damping": 0.4,
    "gravity": 0,
    "mouseInfluence": 2,
    "mouseRadius": 0.2,
    "exposure": 1,
    "softness": 0.1,
    "depth": 0.18,
    "colorSpace": "oklab",
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "PerlinNoise": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#ffffff",
    "colorB": "#000000",
    "stops": null,
    "colorSpace": "linear",
    "scale": 2,
    "contrast": 0,
    "balance": 0,
    "seed": 0,
    "speed": 1
  },
  "Perspective": {
    "opacity": 1,
    "blendMode": "normal",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "pan": 0,
    "tilt": 0,
    "fov": 60,
    "zoom": 1,
    "offset": {
      "x": 0.5,
      "y": 0.5
    },
    "edges": "transparent"
  },
  "Pixelate": {
    "opacity": 1,
    "blendMode": "normal",
    "scale": 50,
    "gap": 0,
    "roundness": 0
  },
  "PixelSort": {
    "opacity": 1,
    "blendMode": "normal",
    "radius": 0.4,
    "falloff": 1,
    "strength": 0.1,
    "decay": 0.1,
    "axis": "vertical",
    "direction": "descending"
  },
  "PixelThrow": {
    "opacity": 1,
    "blendMode": "normal",
    "throwKey": "luminance",
    "strength": 0.25,
    "keyInfluence": 0.8,
    "radius": 0.2,
    "friction": 0.3,
    "momentum": 0.5,
    "edges": "stretch"
  },
  "Plasma": {
    "opacity": 1,
    "blendMode": "normal",
    "density": 2,
    "speed": 2,
    "intensity": 1.5,
    "warp": 0.4,
    "contrast": 1,
    "balance": 50,
    "colorA": "#7018be",
    "colorB": "#000000",
    "stops": null,
    "colorSpace": "linear"
  },
  "Plastic": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "colorA": "#f5f5f7",
    "colorB": "#d9dade",
    "stops": null,
    "gradientAngle": 90,
    "colorSpace": "linear",
    "thickness": 0.5,
    "roughness": 0.12,
    "reflectivity": 1,
    "crumple": 0.25,
    "crumpleScale": 1,
    "crumpleCoverage": 1,
    "seed": 0,
    "edgeSoftness": 0.05,
    "lightAngle": 315,
    "speed": 1,
    "shading": 0.35,
    "rim": 0.25,
    "rimColor": "#ffffff",
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "PolarCoordinates": {
    "opacity": 1,
    "blendMode": "normal",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "wrap": 1,
    "radius": 1,
    "intensity": 1,
    "edges": "transparent"
  },
  "Polygon": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "color": "#ffffff",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "radius": 0.4,
    "sides": 6,
    "rounding": 0,
    "rotation": 0,
    "softness": 0,
    "strokeThickness": 0,
    "strokeColor": "#000000",
    "strokePosition": "center",
    "colorSpace": "linear"
  },
  "Posterize": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 5
  },
  "Prism": {
    "opacity": 1,
    "blendMode": "normal",
    "position": {
      "x": 0.18,
      "y": 0.18
    },
    "beamWidth": 0.04,
    "intensity": 1.6,
    "beamColor": "#ffffff",
    "startFalloff": 0.15,
    "endFalloff": 0.6,
    "splitPosition": {
      "x": 0.5,
      "y": 0.42
    },
    "spread": 0.7,
    "softness": 0.12,
    "saturation": 0.95,
    "speed": 0.1
  },
  "ProgressiveBlur": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 50,
    "angle": 0,
    "center": {
      "x": 0,
      "y": 0.5
    },
    "falloff": 1
  },
  "RadialGradient": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#ff0000",
    "colorB": "#0000ff",
    "stops": null,
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "radius": 1,
    "repeat": 1,
    "aspect": 1,
    "skewAngle": 0,
    "colorSpace": "linear"
  },
  "RadialWipe": {
    "opacity": 1,
    "blendMode": "normal",
    "progress": 0.5,
    "startAngle": 0,
    "direction": "cw",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "feather": 0.1,
    "invert": false
  },
  "RandomBars": {
    "opacity": 1,
    "blendMode": "normal",
    "progress": 0.5,
    "angle": 0,
    "barCount": 12,
    "softness": 0.15,
    "invert": false
  },
  "ReactionDiffusion": {
    "opacity": 1,
    "blendMode": "normal",
    "preset": "coral",
    "feed": 0.0545,
    "kill": 0.062,
    "diffusionRatio": 0.5,
    "featureSize": 3,
    "speed": 6,
    "brushSize": 0.04,
    "brushStrength": 0.6,
    "colorA": "transparent",
    "colorC": "#38bdf8",
    "colorB": "#e2f5ff",
    "contrast": 0.5,
    "threshold": 0.28,
    "colorSpace": "oklch",
    "relief": 0.4,
    "lightAngle": 135,
    "childInfluence": 0.6,
    "childContrast": 0.5,
    "childThreshold": 0.5,
    "childInvert": false
  },
  "RectangularCoordinates": {
    "opacity": 1,
    "blendMode": "normal",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "intensity": 1,
    "edges": "transparent"
  },
  "ReflectivePlane": {
    "opacity": 1,
    "blendMode": "normal",
    "height": 0.7,
    "distance": 0.5,
    "falloff": 0.5,
    "blur": 3,
    "blurDistance": 0.3,
    "edges": "stretch"
  },
  "Repeater": {
    "opacity": 1,
    "blendMode": "normal",
    "mode": "grid",
    "cropLeft": 0,
    "cropRight": 0,
    "cropTop": 0,
    "cropBottom": 0,
    "columns": 3,
    "rows": 3,
    "gapX": 0.05,
    "gapY": 0.05,
    "stagger": 0,
    "flip": "none",
    "count": 8,
    "radius": 0.3,
    "startAngle": 0,
    "sweep": 360,
    "faceCenter": false,
    "direction": 0,
    "spacing": 0.2,
    "instanceScale": 1,
    "instanceRotation": 0,
    "instanceOpacity": 1,
    "hueShift": 0,
    "phase": 0,
    "zOrder": "forward",
    "jitterPosition": 0,
    "jitterRotation": 0,
    "jitterScale": 0,
    "jitterOpacity": 0,
    "seed": 0
  },
  "Ring": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "color": "#ffffff",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "radius": 0.3,
    "thickness": 0.07,
    "softness": 0,
    "strokeThickness": 0,
    "strokeColor": "#000000",
    "strokePosition": "center",
    "colorSpace": "linear"
  },
  "Ripples": {
    "opacity": 1,
    "blendMode": "normal",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "colorA": "#ffffff",
    "colorB": "#000000",
    "speed": 1,
    "frequency": 20,
    "softness": 0,
    "thickness": 0.5,
    "phase": 0
  },
  "RippleWipe": {
    "opacity": 1,
    "blendMode": "normal",
    "progress": 0.5,
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "rings": 8,
    "feather": 0.2,
    "invert": false
  },
  "RoundedRect": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "color": "#ffffff",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "width": 0.35,
    "height": 0.25,
    "rounding": 0.05,
    "rotation": 0,
    "softness": 0,
    "strokeThickness": 0,
    "strokeColor": "#000000",
    "strokePosition": "center",
    "colorSpace": "linear"
  },
  "Saturation": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 1
  },
  "Scratches": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#ffffff",
    "colorB": "#000000",
    "scale": 2,
    "thickness": 1,
    "seed": 0,
    "speed": 1
  },
  "Sharpness": {
    "opacity": 1,
    "blendMode": "normal",
    "sharpness": 0
  },
  "Shatter": {
    "opacity": 1,
    "blendMode": "normal",
    "crackWidth": 1,
    "intensity": 4,
    "radius": 0.4,
    "decay": 1,
    "seed": 2,
    "chromaticSplit": 1,
    "refractionStrength": 5,
    "shardLighting": 0.1,
    "edges": "mirror"
  },
  "SimplexNoise": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#ffffff",
    "colorB": "#000000",
    "stops": null,
    "colorSpace": "linear",
    "scale": 2,
    "balance": 0,
    "contrast": 0,
    "seed": 0,
    "speed": 1
  },
  "SineWave": {
    "opacity": 1,
    "blendMode": "normal",
    "color": "#ffffff",
    "amplitude": 0.15,
    "frequency": 1,
    "speed": 1,
    "angle": 0,
    "position": {
      "x": 0.5,
      "y": 0.5
    },
    "thickness": 0.2,
    "softness": 0.4
  },
  "SliceWipe": {
    "opacity": 1,
    "blendMode": "normal",
    "progress": 0.5,
    "angle": 0,
    "sliceCount": 8
  },
  "Smoke": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#fc83f9",
    "colorB": "#c21c79",
    "stops": null,
    "emitFrom": {
      "x": 0.5,
      "y": 1
    },
    "direction": 0,
    "speed": 20,
    "spread": 60,
    "emitRadius": 0.08,
    "intensity": 1,
    "dissipation": 0.2,
    "detail": 25,
    "gravity": 0.5,
    "colorDecay": 0.4,
    "mouseInfluence": 0.1,
    "mouseRadius": 0.1,
    "colorSpace": "linear"
  },
  "SmokeFill": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#8cf3ff",
    "colorB": "#04a0d6",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "emitFrom": {
      "x": 0.5,
      "y": 0.5
    },
    "direction": 0,
    "speed": 10,
    "spread": 60,
    "emitRadius": 0.03,
    "intensity": 1,
    "dissipation": 0.3,
    "detail": 25,
    "gravity": 0.5,
    "colorDecay": 0.4,
    "mouseInfluence": 0.1,
    "mouseRadius": 0.1,
    "colorSpace": "linear",
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "SmokeFlow": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#e29c8b",
    "colorB": "#d517f9",
    "stops": null,
    "intensity": 1,
    "emitRadius": 0.07,
    "momentum": 20,
    "dissipation": 0.5,
    "detail": 10,
    "gravity": 2,
    "colorDecay": 0.5,
    "colorSpace": "OKLAB"
  },
  "Solarize": {
    "opacity": 1,
    "blendMode": "normal",
    "threshold": 0.5,
    "strength": 1
  },
  "SolidColor": {
    "opacity": 1,
    "blendMode": "normal",
    "color": "#5b18ca"
  },
  "Spherize": {
    "opacity": 1,
    "blendMode": "normal",
    "radius": 1,
    "depth": 1,
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "lightPosition": {
      "x": 0.3,
      "y": 0.3
    },
    "lightIntensity": 0.5,
    "lightSoftness": 0.5,
    "lightColor": "#ffffff"
  },
  "Spiral": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#000000",
    "colorB": "#ffffff",
    "strokeWidth": 0.5,
    "strokeFalloff": 0,
    "softness": 0,
    "speed": 1,
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "colorSpace": "linear"
  },
  "Star": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "color": "#ffffff",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "radius": 0.4,
    "sides": 5,
    "innerRatio": 0.4,
    "rotation": 0,
    "softness": 0,
    "strokeThickness": 0,
    "strokeColor": "#000000",
    "strokePosition": "center",
    "colorSpace": "linear"
  },
  "Stone": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 0.5,
    "scale": 1,
    "contrast": 0,
    "distortion": 0.15,
    "seed": 0
  },
  "Strands": {
    "opacity": 1,
    "blendMode": "normal",
    "speed": 0.5,
    "amplitude": 2,
    "frequency": 0.3,
    "lineCount": 8,
    "lineWidth": 0.05,
    "softness": 0.05,
    "spread": 0.2,
    "pinEdges": false,
    "stops": [
      {
        "color": "#00e5ff",
        "position": 0
      },
      {
        "color": "#5b8cff",
        "position": 0.34
      },
      {
        "color": "#b06bff",
        "position": 0.67
      },
      {
        "color": "#ff5fa2",
        "position": 1
      }
    ],
    "colorSpace": "oklab",
    "colorScale": 1,
    "colorVariance": 1,
    "colorSpeed": 1,
    "start": {
      "x": 0,
      "y": 0.5
    },
    "end": {
      "x": 1,
      "y": 0.5
    }
  },
  "Stretch": {
    "opacity": 1,
    "blendMode": "normal",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "strength": 1,
    "angle": 0,
    "falloff": 0,
    "edges": "stretch"
  },
  "Stripes": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#000000",
    "colorB": "#ffffff",
    "angle": 45,
    "density": 5,
    "balance": 0.5,
    "softness": 0,
    "speed": 0.2,
    "offset": 0,
    "colorSpace": "linear"
  },
  "StudioBackground": {
    "opacity": 1,
    "blendMode": "normal",
    "color": "#d8dbec",
    "keyColor": "#d5e4ea",
    "keyIntensity": 40,
    "keySoftness": 50,
    "fillColor": "#d5e4ea",
    "fillIntensity": 10,
    "fillSoftness": 70,
    "fillAngle": 70,
    "backColor": "#c8d4e8",
    "backIntensity": 20,
    "backSoftness": 80,
    "brightness": 20,
    "vignette": 0,
    "center": {
      "x": 0.5,
      "y": 0.8
    },
    "lightTarget": 100,
    "wallCurvature": 10,
    "ambientIntensity": 50,
    "ambientSpeed": 2,
    "seed": 0
  },
  "SunBurst": {
    "opacity": 1,
    "blendMode": "normal",
    "color": "#ffdd88",
    "background": "#000000",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "rayCount": 12,
    "softness": 0.3,
    "radius": 0.8,
    "feather": 0.5,
    "speed": 0.2
  },
  "Surface3D": {
    "opacity": 1,
    "blendMode": "normal",
    "amplitude": 0.3,
    "waveType": "fractal",
    "frequency": 1.5,
    "octaves": 2,
    "seed": 0,
    "speed": 0.5,
    "tilt": 35,
    "roll": 0,
    "height": 0,
    "zoom": 1,
    "nearCutoff": 0,
    "farCutoff": 1,
    "edgePinning": 0,
    "edges": "mirror",
    "lighting": 30,
    "glossiness": 0,
    "highlights": 15,
    "lightX": 0.4,
    "lightY": -0.6,
    "lightZ": 0.7,
    "lightColor": "#ffffff",
    "cursorIntensity": 1,
    "cursorSpeed": 0.5
  },
  "Swirl": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#1275d8",
    "colorB": "#e19136",
    "stops": null,
    "speed": 1,
    "detail": 1,
    "blend": 50,
    "colorSpace": "linear"
  },
  "Teardrop": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "color": "#ffffff",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "radius": 0.22,
    "height": 0.4,
    "rotation": 0,
    "softness": 0,
    "strokeThickness": 0,
    "strokeColor": "#000000",
    "strokePosition": "center",
    "colorSpace": "linear"
  },
  "Text": {
    "opacity": 1,
    "blendMode": "normal",
    "text": "Hello World",
    "fontFamily": "Inter",
    "fontWeight": 400,
    "italic": false,
    "textTransform": "none",
    "fontSize": 0.07,
    "letterSpacing": 0,
    "lineHeight": 1.2,
    "textAlign": "center",
    "width": 0,
    "color": "#ffffff",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "rotation": 0
  },
  "ThinFilm": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "intensity": 1,
    "rimWidth": 1,
    "edgeSoftness": 0.3,
    "thickness": 0.5,
    "dispersion": 0.5,
    "saturation": 1,
    "hueShift": 0,
    "lightAngle": 300,
    "mode": "rainbow",
    "colorA": "#2b6fff",
    "colorB": "#ffffff",
    "colorC": "#ff7a21",
    "colorSpace": "oklch",
    "speed": 0.1,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "TiltShift": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 50,
    "width": 0.3,
    "falloff": 0.3,
    "angle": 0,
    "center": {
      "x": 0.5,
      "y": 0.5
    }
  },
  "TimeTrail": {
    "opacity": 1,
    "blendMode": "normal",
    "trailSource": "motion",
    "motionThreshold": 0.06,
    "trailLength": 0.6,
    "diffusion": 1,
    "trailOpacity": 1,
    "trailBlend": "normal",
    "driftX": 0,
    "driftY": 0,
    "zoom": 1,
    "tintMode": "none",
    "tint": "#33aaff",
    "speed": 1
  },
  "Tint": {
    "opacity": 1,
    "blendMode": "normal",
    "color": "#ff8800",
    "amount": 0.5,
    "preserveLuminosity": true
  },
  "Trapezoid": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "color": "#ffffff",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "bottomWidth": 0.35,
    "topWidth": 0.2,
    "height": 0.25,
    "rotation": 0,
    "softness": 0,
    "strokeThickness": 0,
    "strokeColor": "#000000",
    "strokePosition": "center",
    "colorSpace": "linear"
  },
  "TriangularGrid": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#1a1a1a",
    "colorB": "#ffffff",
    "cells": 8,
    "thickness": 1,
    "rotation": 0,
    "softness": 0,
    "variation": 0,
    "speed": 0,
    "speedVariance": 0.3,
    "colorSpace": "linear"
  },
  "Tritone": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#ce1bea",
    "colorB": "#2fff00",
    "colorC": "#ffff00",
    "blendMid": 0.5,
    "colorSpace": "linear"
  },
  "Truchet": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#000000",
    "colorB": "#ffffff",
    "cells": 10,
    "thickness": 2,
    "rotation": 0,
    "softness": 0,
    "seed": 0,
    "colorSpace": "linear"
  },
  "Twirl": {
    "opacity": 1,
    "blendMode": "normal",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "intensity": 1,
    "edges": "stretch"
  },
  "VenetianBlinds": {
    "opacity": 1,
    "blendMode": "normal",
    "progress": 0.5,
    "angle": 0,
    "stripCount": 5,
    "feather": 0.15,
    "invert": false
  },
  "Vesica": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "color": "#ffffff",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "radius": 0.35,
    "spread": 0.5,
    "rotation": 0,
    "softness": 0,
    "strokeThickness": 0,
    "strokeColor": "#000000",
    "strokePosition": "center",
    "colorSpace": "linear"
  },
  "VHS": {
    "opacity": 1,
    "blendMode": "normal",
    "wobble": 1,
    "scanlineNoise": 0.6,
    "smear": 0.2,
    "speed": 1
  },
  "Vibrance": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 0
  },
  "VideoTexture": {
    "opacity": 1,
    "blendMode": "normal",
    "url": "https://shaders.com/sample.mp4",
    "objectFit": "fill",
    "loop": true
  },
  "Vignette": {
    "opacity": 1,
    "blendMode": "normal",
    "color": "#000000",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "radius": 0.5,
    "falloff": 0.5,
    "intensity": 1
  },
  "Voronoi": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#3186cf",
    "colorB": "#fc02dd",
    "stops": null,
    "colorBorder": "#000000",
    "scale": 6,
    "speed": 0.5,
    "seed": 0,
    "edgeIntensity": 0.5,
    "edgeSoftness": 0.05,
    "colorSpace": "linear"
  },
  "Voxels": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "voxelSize": 0.035,
    "voxelShape": "cube",
    "voxelScale": 1,
    "fill": 0,
    "bevel": 0.08,
    "depth": 0.12,
    "gridSpace": "shape",
    "colorA": "#6ea8ff",
    "colorB": "#1d2b4f",
    "colorMode": "height",
    "colorVariation": 0.25,
    "colorSpace": "oklab",
    "lightAngle": 225,
    "lightElevation": 42,
    "lightColor": "#fff2df",
    "lightIntensity": 1.1,
    "ambientColor": "#9fb4d8",
    "ambient": 0.55,
    "shadows": 0.85,
    "shadowSoftness": 0.35,
    "ao": 1,
    "glossiness": 0.35,
    "specular": 0.6,
    "seams": 0.2,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "Water": {
    "opacity": 1,
    "blendMode": "normal",
    "origin": "center",
    "center": {
      "x": 0.5,
      "y": 0.5
    },
    "scale": 1,
    "rotation": 0,
    "waterColor": "#00bfeb",
    "clarity": 0.4,
    "depth": 5,
    "shallows": 0.5,
    "caustics": 0.15,
    "causticScale": 6,
    "choppiness": 0.8,
    "waveScale": 3.5,
    "swirl": 0.6,
    "speed": 0.5,
    "reflection": 2,
    "envRotation": 0,
    "sharpness": 0.1,
    "lightAngle": 30,
    "foam": 0.3,
    "edgeSoftness": 0.05,
    "shape": "{\"type\":\"sphere3D\",\"radius\":0.35}"
  },
  "Watercolor": {
    "opacity": 1,
    "blendMode": "normal",
    "radius": 3,
    "bleed": 1,
    "strength": 1,
    "paper": 0.35,
    "paperColor": "#fbf7ec"
  },
  "WaveDistortion": {
    "opacity": 1,
    "blendMode": "normal",
    "strength": 0.3,
    "frequency": 1,
    "speed": 1,
    "angle": 0,
    "waveType": "sine",
    "edges": "stretch"
  },
  "Waveform": {
    "opacity": 1,
    "blendMode": "normal",
    "style": "bars",
    "colorA": "#3b5bff",
    "colorB": "#ff3bd4",
    "stops": null,
    "colorSpace": "oklab",
    "from": {
      "x": 0,
      "y": 0.5
    },
    "to": {
      "x": 1,
      "y": 0.5
    },
    "amplitude": 1,
    "frequency": 1,
    "height": 0.6,
    "align": "mirrored",
    "count": 32,
    "barWidth": 0.6,
    "rounding": 1,
    "dotSize": 0.7,
    "lineWidth": 0.006,
    "softness": 0,
    "speed": 1,
    "seed": 1
  },
  "WaveletNoise": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#ffffff",
    "colorB": "#000000",
    "stops": null,
    "colorSpace": "linear",
    "scale": 1.5,
    "detail": 1.24,
    "contrast": 0,
    "balance": 0,
    "seed": 0,
    "speed": 1
  },
  "Weave": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#c4c4c4",
    "colorB": "#4d4d4d",
    "cells": 10,
    "gap": 0.25,
    "rotation": 0
  },
  "WebcamTexture": {
    "opacity": 1,
    "blendMode": "normal",
    "objectFit": "cover",
    "mirror": true
  },
  "Wool": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 0.5,
    "scale": 4,
    "contrast": -0.5,
    "distortion": 0.15,
    "seed": 0
  },
  "WorleyNoise": {
    "opacity": 1,
    "blendMode": "normal",
    "colorA": "#ffffff",
    "colorB": "#000000",
    "colorSpace": "linear",
    "scale": 6,
    "mode": "f1",
    "distance": "euclidean",
    "octaves": 1,
    "lacunarity": 2,
    "persistence": 0.5,
    "jitter": 1,
    "contrast": 1,
    "balance": 0,
    "seed": 0,
    "speed": 0.5
  },
  "ZoomBlur": {
    "opacity": 1,
    "blendMode": "normal",
    "intensity": 30,
    "center": {
      "x": 0.5,
      "y": 0.5
    }
  }
}

type PropMapValue = { type: 'map'; source: string; [key: string]: unknown }

function isPropMapValue(value: unknown): value is PropMapValue {
  return typeof value === 'object' && value !== null && 'type' in value && (value as Record<string, unknown>).type === 'map'
}

function escapeString(str: string): string {
  return str
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t')
}

function formatValue(value: any, indent: string): string {
  if (typeof value === 'string') {
    // Detect JSON-encoded object strings and format as objects
    if (value.startsWith('{') && value.endsWith('}')) {
      try {
        const parsed = JSON.parse(value)
        if (typeof parsed === 'object' && parsed !== null) {
          return formatValue(parsed, indent)
        }
      } catch { /* not JSON, treat as string */ }
    }
    return `'${escapeString(value)}'`
  }
  if (typeof value === 'boolean' || typeof value === 'number') {
    return String(value)
  }
  if (value === null || value === undefined) {
    return String(value)
  }
  if (Array.isArray(value)) {
    const items = value.map(v => formatValue(v, indent + '  ')).join(', ')
    return `[${items}]`
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value)
    if (entries.length === 0) return '{}'
    const lines = entries.map(([k, v]) => `${indent}  ${k}: ${formatValue(v, indent + '  ')}`)
    return `{\n${lines.join(',\n')}\n${indent}}`
  }
  return String(value)
}

function shouldIncludeProp(key: string, value: any, componentType: string, props: Record<string, any>): boolean {
  // Skip maskType if it's the default 'alpha'
  if (key === 'maskType' && value === 'alpha') return false

  // When a custom SDF shape URL is set, `shape` and `shapeType` are both
  // editor-only metadata — the renderer only needs `shapeSdfUrl`.
  if ((key === 'shape' || key === 'shapeType') && props.shapeSdfUrl && props.shapeType !== 'svgExtrude3D') return false

  // Special handling for transform
  if (key === 'transform' && typeof value === 'object') {
    const allDefaults = Object.keys(DEFAULT_TRANSFORM).every(
      k => value[k] === DEFAULT_TRANSFORM[k as keyof typeof DEFAULT_TRANSFORM]
    )
    if (allDefaults) return false
  }

  // Special handling for boundingBox — drop it entirely when at full-frame defaults
  if (key === 'boundingBox' && typeof value === 'object') {
    return reduceBoundingBoxForExport(value) !== null
  }

  // Group flow layout: drop when inactive; absolute: drop when falsy
  if (key === 'flow' && (!value || typeof value !== 'object' || (value as any).mode === 'none' || !(value as any).mode)) return false
  if (key === 'absolute' && !value) return false

  // Get component-specific defaults from metadata
  const componentDefaults = shaderMetadata[componentType] || {}

  // Skip if this prop matches its default value
  if (componentDefaults.hasOwnProperty(key)) {
    const defaultValue = componentDefaults[key]

    if (value != null && defaultValue != null && typeof value === 'object' && typeof defaultValue === 'object') {
      return JSON.stringify(value) !== JSON.stringify(defaultValue)
    }

    return value !== defaultValue
  }

  // Skip universal default values (fallback)
  if (key === 'opacity' && value === 1) return false
  if (key === 'blendMode' && value === 'normal') return false

  return true
}

function generateComponentObject(config: ComponentConfig, allComponents: ComponentConfig[], indent: string): string {
  const lines: string[] = []
  lines.push(`${indent}{`)

  // Type
  lines.push(`${indent}  type: '${config.type}',`)

  // Always include ID for JS (needed for shader.update())
  if (config.id) {
    lines.push(`${indent}  id: '${config.id}',`)
  }

  // Props
  const filteredProps: [string, any][] = []
  if (config.props) {
    for (const [key, value] of Object.entries(config.props).sort(([a], [b]) => a.localeCompare(b))) {
      if (shouldIncludeProp(key, value, config.type, config.props)) {
        // Handle transform — only include non-default keys
        if (key === 'transform' && typeof value === 'object') {
          const nonDefaultKeys: Record<string, any> = {}
          for (const k in value) {
            if (value[k] !== DEFAULT_TRANSFORM[k as keyof typeof DEFAULT_TRANSFORM]) {
              nonDefaultKeys[k] = value[k]
            }
          }
          if (Object.keys(nonDefaultKeys).length > 0) {
            filteredProps.push([key, nonDefaultKeys])
          }
        } else if (key === 'boundingBox' && typeof value === 'object') {
          // boundingBox passed shouldIncludeProp with at least one non-default key — emit only those keys
          const reduced = reduceBoundingBoxForExport(value)
          if (reduced) {
            filteredProps.push([key, reduced])
          }
        } else if (typeof value === 'object' && value !== null && 'x' in value && typeof value.x === 'number') {
          // Round x/y position values
          const rounded = { ...value }
          if (typeof rounded.x === 'number') rounded.x = Math.round(rounded.x * 100) / 100
          if (typeof rounded.y === 'number') rounded.y = Math.round(rounded.y * 100) / 100
          filteredProps.push([key, rounded])
        } else {
          filteredProps.push([key, value])
        }
      }
    }
  }

  if (filteredProps.length > 0) {
    lines.push(`${indent}  props: {`)
    for (const [key, value] of filteredProps) {
      lines.push(`${indent}    ${key}: ${formatValue(value, indent + '    ')},`)
    }
    lines.push(`${indent}  },`)
  }

  // Children
  if (config.children && config.children.length > 0) {
    lines.push(`${indent}  children: [`)
    for (const child of config.children) {
      lines.push(generateComponentObject(child, allComponents, indent + '    ') + ',')
    }
    lines.push(`${indent}  ],`)
  }

  lines.push(`${indent}}`)
  return lines.join('\n')
}

// Number-rounding helpers (roundPresetForExport) are injected here from generate-components.ts:
// Round every number in the preset (incl. numbers inside the JSON-encoded `shape` string) to 4
// decimals — well below any visible threshold for shader props — so exported code stays clean.
function roundForExportDeep(value: any): any {
  if (typeof value === 'number') return Number.isFinite(value) ? Math.round(value * 1e4) / 1e4 : value
  if (typeof value === 'string') {
    if (value.startsWith('{') && value.endsWith('}')) {
      try {
        const parsed = JSON.parse(value)
        if (parsed && typeof parsed === 'object') return JSON.stringify(roundForExportDeep(parsed))
      } catch { /* not JSON — leave as-is */ }
    }
    return value
  }
  if (Array.isArray(value)) return value.map(roundForExportDeep)
  if (value && typeof value === 'object') {
    const out: Record<string, any> = {}
    for (const k in value) out[k] = roundForExportDeep(value[k])
    return out
  }
  return value
}
function roundPresetForExport(preset: PresetConfig): PresetConfig {
  const mapComp = (c: ComponentConfig): ComponentConfig => ({
    ...c,
    props: c.props ? roundForExportDeep(c.props) : c.props,
    children: c.children ? c.children.map(mapComp) : c.children
  })
  return { components: preset.components.map(mapComp) }
}

export function generatePresetCode(preset: PresetConfig, colorSpace?: 'p3-linear' | 'srgb', toneMapping?: 'linear' | 'reinhard' | 'cineon' | 'aces' | 'agx' | 'neutral' | 'hable' | 'unreal'): string {
  preset = roundPresetForExport(preset)
  const componentsStr = preset.components
    .map(config => generateComponentObject(config, preset.components, '    '))
    .join(',\n')

  const optionsLines: string[] = []
  if (colorSpace && colorSpace !== 'p3-linear') {
    optionsLines.push(`  colorSpace: '${colorSpace}',`)
  }
  if (toneMapping && toneMapping !== 'linear') {
    optionsLines.push(`  toneMapping: '${toneMapping}',`)
  }
  const optionsStr = optionsLines.length > 0
    ? `, {\n${optionsLines.join('\n')}\n}`
    : ''

  return `import { createShader } from 'shaders/js'

const shader = await createShader(document.getElementById("canvas"), {
  components: [
${componentsStr}
  ]
}${optionsStr})`
}

// Available components (auto-generated)
export const availableComponents = [
  // @ts-ignore - replaced at build time
    'AngularBlur',
  'Arc',
  'Ascii',
  'Aurora',
  'BarShift',
  'BarnDoors',
  'Beam',
  'Bend',
  'Blob',
  'BlockDissolve',
  'BlockNoise',
  'BlueNoise',
  'Blur',
  'Boids',
  'BokehBlur',
  'BrickPattern',
  'BrightnessContrast',
  'BrushedMetal',
  'Bulge',
  'CRTScreen',
  'CarbonFiber',
  'Chalkboard',
  'ChannelBlur',
  'CheckerWipe',
  'Checkerboard',
  'Chevron',
  'ChromaFlow',
  'ChromaticAberration',
  'Chrome',
  'Circle',
  'ColorWheel',
  'CompressionArtifacts',
  'ConcentricSpin',
  'ConicGradient',
  'ContourLines',
  'CornerPin',
  'Crescent',
  'Cross',
  'Crystal',
  'CurlNoise',
  'CursorRipples',
  'CursorTrail',
  'DataMosh',
  'DiamondGradient',
  'DiamondWipe',
  'DiffuseBlur',
  'DisplacementMap',
  'Dither',
  'DotGrid',
  'DropShadow',
  'Duotone',
  'Ellipse',
  'Emboss',
  'Engraving',
  'ErosionNoise',
  'Exposure',
  'FallingLines',
  'FilmGrain',
  'FilmStock',
  'Flip',
  'FloatingParticles',
  'FlowField',
  'Flower',
  'FlowingGradient',
  'FlutedGlass',
  'Fog',
  'Form3D',
  'FractalNoise',
  'Frost',
  'GaborNoise',
  'Glass',
  'GlassTiles',
  'Glitch',
  'Glow',
  'Godrays',
  'Goo',
  'GradientMap',
  'Grayscale',
  'Grid',
  'GridDistortion',
  'Group',
  'HTMLInCanvas',
  'Halftone',
  'Heart',
  'Heatmap',
  'HexGrid',
  'Hologram',
  'Holographic',
  'HueShift',
  'ImageTexture',
  'InkFlow',
  'Invert',
  'IrisWipe',
  'Irradiance',
  'IsometricCubes',
  'Kaleidoscope',
  'KeyFrames',
  'LensDistortion',
  'LensFlare',
  'LightEdge',
  'LightLeak',
  'Line',
  'LinearBlur',
  'LinearGradient',
  'LinearWipe',
  'LiquidMetal',
  'Liquify',
  'MagneticFilings',
  'Marble',
  'MeshGradient',
  'Mirror',
  'MultiPointGradient',
  'Nebula',
  'Neon',
  'NoiseDissolve',
  'ObjectTracker',
  'Obsidian',
  'PagePeel',
  'Paper',
  'Parallelogram',
  'ParticleField',
  'ParticleFlow',
  'Particles',
  'PerlinNoise',
  'Perspective',
  'PixelSort',
  'PixelThrow',
  'Pixelate',
  'Plasma',
  'Plastic',
  'PolarCoordinates',
  'Polygon',
  'Posterize',
  'Prism',
  'ProgressiveBlur',
  'RadialGradient',
  'RadialWipe',
  'RandomBars',
  'ReactionDiffusion',
  'RectangularCoordinates',
  'ReflectivePlane',
  'Repeater',
  'Ring',
  'RippleWipe',
  'Ripples',
  'RoundedRect',
  'Saturation',
  'Scratches',
  'Sharpness',
  'Shatter',
  'SimplexNoise',
  'SineWave',
  'SliceWipe',
  'Smoke',
  'SmokeFill',
  'SmokeFlow',
  'Solarize',
  'SolidColor',
  'Spherize',
  'Spiral',
  'Star',
  'Stone',
  'Strands',
  'Stretch',
  'Stripes',
  'StudioBackground',
  'SunBurst',
  'Surface3D',
  'Swirl',
  'Teardrop',
  'Text',
  'ThinFilm',
  'TiltShift',
  'TimeTrail',
  'Tint',
  'Trapezoid',
  'TriangularGrid',
  'Tritone',
  'Truchet',
  'Twirl',
  'VHS',
  'VenetianBlinds',
  'Vesica',
  'Vibrance',
  'VideoTexture',
  'Vignette',
  'Voronoi',
  'Voxels',
  'Water',
  'Watercolor',
  'WaveDistortion',
  'Waveform',
  'WaveletNoise',
  'Weave',
  'WebcamTexture',
  'Wool',
  'WorleyNoise',
  'ZoomBlur'
]
