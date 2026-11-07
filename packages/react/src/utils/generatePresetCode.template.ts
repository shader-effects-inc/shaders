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

// Convert a value to a JS literal string (recursive)
function toJSLiteral(value: any): string {
  if (typeof value === 'string') return JSON.stringify(value)
  if (typeof value === 'boolean' || typeof value === 'number' || value === null || value === undefined) return String(value)
  if (Array.isArray(value)) return `[${value.map(v => toJSLiteral(v)).join(', ')}]`
  if (typeof value === 'object') {
    const entries = Object.entries(value).map(([k, v]) => `${k}: ${toJSLiteral(v)}`)
    return `{ ${entries.join(', ')} }`
  }
  return String(value)
}

// JSX attributes decode entities and normalize newlines; use a JS literal when needed.
function toJSXStringAttribute(key: string, value: string): string {
  return /["&\r\n]/.test(value) ? `${key}={${toJSLiteral(value)}}` : `${key}="${value}"`
}

// Convert object to JS object literal string (for React/JSX)
function toObjectLiteral(obj: Record<string, any>): string {
  const entries = Object.entries(obj).map(([key, value]) => {
    return `${key}: ${toJSLiteral(value)}`
  })
  return `{ ${entries.join(', ')} }`
}

// Bounding-box reduction helper (reduceBoundingBoxForExport) is injected here from generate-components.ts:
// __BOUNDING_BOX_EXPORT__

// @ts-ignore - replaced at build time
const shaderMetadata: Record<string, Record<string, any>> = __SHADER_METADATA__

type PropMapValue = { type: 'map'; source: string; [key: string]: unknown }

function isPropMapValue(value: unknown): value is PropMapValue {
  return typeof value === 'object' && value !== null && 'type' in value && (value as Record<string, unknown>).type === 'map'
}

function generatePropString(props: Record<string, any>, componentType: string, indent: string = '  '): string {
  return Object.entries(props)
    .sort(([a], [b]) => a.localeCompare(b))
    .filter(([key, value]) => {
      // Skip maskType if it's the default 'alpha'
      if (key === 'maskType' && value === 'alpha') return false

      // When a custom SDF shape URL is set, `shape` and `shapeType` are both
      // editor-only metadata — the renderer only needs `shapeSdfUrl`. Omit
      // them from generated code to keep the output focused on what the
      // shader actually consumes.
      if ((key === 'shape' || key === 'shapeType') && props.shapeSdfUrl && props.shapeType !== 'svgExtrude3D') return false

      // Special handling for transform
      if (key === 'transform' && typeof value === 'object') {
        // Check if all transform values match defaults
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

        // Deep comparison for objects
        if (value != null && defaultValue != null && typeof value === 'object' && typeof defaultValue === 'object') {
          return JSON.stringify(value) !== JSON.stringify(defaultValue)
        }

        // Simple comparison for primitives
        return value !== defaultValue
      }

      // Skip universal default values (fallback)
      if (key === 'opacity' && value === 1) return false
      if (key === 'blendMode' && value === 'normal') return false

      return true
    })
    // boundingBox survived the filter with at least one non-default key — emit only those keys
    .map(([key, value]): [string, any] =>
      key === 'boundingBox' && value && typeof value === 'object'
        ? [key, reduceBoundingBoxForExport(value) ?? {}]
        : [key, value])
    .map(([key, value]) => {
      // Special handling for transform - only include non-default keys
      if (key === 'transform' && typeof value === 'object') {
        const nonDefaultKeys: Record<string, any> = {}
        for (const k in value) {
          if (value[k] !== DEFAULT_TRANSFORM[k as keyof typeof DEFAULT_TRANSFORM]) {
            nonDefaultKeys[k] = value[k]
          }
        }

        // Format transform across multiple lines if it has multiple keys
        const keys = Object.keys(nonDefaultKeys).sort()
        if (keys.length > 1) {
          const entries = keys.map(k => `${k}: ${toJSLiteral(nonDefaultKeys[k])}`)
          return `${key}={{\n${indent}    ${entries.join(`,\n${indent}    `)}\n${indent}  }}`
        } else if (keys.length === 1) {
          const k = keys[0]
          const v = nonDefaultKeys[k]
          return `${key}={{ ${k}: ${toJSLiteral(v)} }}`
        }
      }

      // PropDriver (map config) — format across multiple lines for readability
      if (isPropMapValue(value)) {
        const entries = Object.keys(value).map(k => {
          const v = value[k]
          return `${k}: ${toJSLiteral(v)}`
        })
        return `${key}={{\n${indent}    ${entries.join(`,\n${indent}    `)}\n${indent}  }}`
      }

      if (typeof value === 'string') {
        // Configuration strings represent objects, but text must keep its literal content.
        if (key !== 'text' && value.startsWith('{') && value.endsWith('}')) {
          try {
            const parsed = JSON.parse(value)
            if (typeof parsed === 'object' && parsed !== null) {
              const keys = Object.keys(parsed)
              if (keys.length > 1) {
                const entries = keys.map(k => `${k}: ${toJSLiteral(parsed[k])}`)
                return `${key}={{\n${indent}    ${entries.join(`,\n${indent}    `)}\n${indent}  }}`
              }
              return `${key}={${toObjectLiteral(parsed)}}`
            }
          } catch { /* not JSON, treat as string */ }
        }
        return toJSXStringAttribute(key, value)
      } else if (Array.isArray(value)) {
        // Array prop (e.g. gradient stops) — pass as a JS array literal.
        return `${key}={${toJSLiteral(value)}}`
      } else if (value !== null && typeof value === 'object') {
        // Round x/y position values to 2 decimal places
        const roundedValue = { ...value }
        if ('x' in roundedValue && typeof roundedValue.x === 'number') {
          roundedValue.x = Math.round(roundedValue.x * 100) / 100
        }
        if ('y' in roundedValue && typeof roundedValue.y === 'number') {
          roundedValue.y = Math.round(roundedValue.y * 100) / 100
        }
        // Multi-line for objects with 2+ keys
        const keys = Object.keys(roundedValue)
        if (keys.length > 1) {
          const entries = keys.map(k => `${k}: ${toJSLiteral(roundedValue[k])}`)
          return `${key}={{\n${indent}    ${entries.join(`,\n${indent}    `)}\n${indent}  }}`
        }
        return `${key}={${toObjectLiteral(roundedValue)}}`
      } else {
        return `${key}={${value}}`
      }
    })
    .join('\n' + indent + '  ')
}

// Check if an ID is referenced by any component (as mask source or prop map source)
function isIdReferenced(id: string, allComponents: ComponentConfig[]): boolean {
  const flatComponents: ComponentConfig[] = []

  function flattenComponents(components: ComponentConfig[]) {
    for (const component of components) {
      flatComponents.push(component)
      if (component.children) {
        flattenComponents(component.children)
      }
    }
  }

  flattenComponents(allComponents)

  return flatComponents.some(component => {
    if (component.props?.maskSource === id) return true
    // DisplacementMap's own built-in layer source-selector (ui.type: 'layer') — a bare
    // string prop, not a PropMapValue object, so the loop below wouldn't catch it.
    if (component.type === 'DisplacementMap' && component.props?.source === id) return true
    if (component.props) {
      for (const value of Object.values(component.props)) {
        if (isPropMapValue(value) && value.source === id) return true
      }
    }
    return false
  })
}

// Number-rounding helpers (roundPresetForExport) are injected here from generate-components.ts:
// __PRESET_EXPORT_ROUNDING__

export function generatePresetCode(preset: PresetConfig, colorSpace?: 'p3-linear' | 'srgb', toneMapping?: 'linear' | 'reinhard' | 'cineon' | 'aces' | 'agx' | 'neutral' | 'hable' | 'unreal'): string {
  preset = roundPresetForExport(preset)
  // Collect unique component types
  const componentTypes = new Set<string>()

  function collectTypes(components: ComponentConfig[]) {
    for (const config of components) {
      componentTypes.add(config.type)
      if (config.children) collectTypes(config.children)
    }
  }
  collectTypes(preset.components)

  const generateComponentString = (config: ComponentConfig, indent: string = '      '): string => {
    const propString = config.props ? generatePropString(config.props, config.type, indent) : ''

    // Only include ID if it's referenced by another component (mask source or prop map source)
    let idString = ''
    if (config.id) {
      const isReferenced = isIdReferenced(config.id, preset.components)

      if (isReferenced) {
        idString = toJSXStringAttribute('id', config.id)
      }
    }

    const attributes = [idString, propString].filter(Boolean).join('\n' + indent + '  ')

    // HTMLInCanvas: inject sample HTML so users see how to swap in their own content
    const sampleChildren = (config.type === 'HTMLInCanvas' || config.type === 'DOMTexture') && (!config.children || config.children.length === 0)
      ? `${indent}  <h1>Your HTML content here</h1>`
      : null

    if (sampleChildren) {
      if (attributes) {
        return `${indent}<${config.type}\n${indent}  ${attributes}>\n${sampleChildren}\n${indent}</${config.type}>`
      } else {
        return `${indent}<${config.type}>\n${sampleChildren}\n${indent}</${config.type}>`
      }
    } else if (config.children && config.children.length > 0) {
      const childrenString = config.children
        .map(child => generateComponentString(child, indent + '  '))
        .join('\n')

      if (attributes) {
        return `${indent}<${config.type}\n${indent}  ${attributes}>\n${childrenString}\n${indent}</${config.type}>`
      } else {
        return `${indent}<${config.type}>\n${childrenString}\n${indent}</${config.type}>`
      }
    } else {
      if (attributes) {
        return `${indent}<${config.type}\n${indent}  ${attributes} />`
      } else {
        return `${indent}<${config.type} />`
      }
    }
  }

  const componentStrings = preset.components
    .map(config => generateComponentString(config))
    .join('\n')

  // Add colorSpace and toneMapping props to Shader when non-default
  const colorSpaceProp = colorSpace && colorSpace !== 'p3-linear' ? ` colorSpace="${colorSpace}"` : ''
  const toneMappingProp = toneMapping && toneMapping !== 'linear' ? ` toneMapping="${toneMapping}"` : ''
  const shaderProps = colorSpaceProp + toneMappingProp

  // Build import list
  const sortedTypes = Array.from(componentTypes).sort()
  const allImports = ['Shader', ...sortedTypes]

  const importStatement = allImports.length > 3
    ? `import {\n  ${allImports.join(',\n  ')},\n} from 'shaders/react'`
    : `import { ${allImports.join(', ')} } from 'shaders/react'`

  return `${importStatement}

export default function ShaderEffect() {
  return (
    <Shader${shaderProps}>
${componentStrings}
    </Shader>
  )
}`
}

// Available components (auto-generated)
export const availableComponents = [
  // @ts-ignore - replaced at build time
  __COMPONENT_LIST__
]
