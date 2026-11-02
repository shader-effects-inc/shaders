interface ComponentConfig {
  type: string
  id?: string
  props?: Record<string, any>
  children?: ComponentConfig[]
}

interface PresetConfig {
  components: ComponentConfig[]
}

function camelToKebab(str: string): string {
  return str.replace(/([a-z0-9])([A-Z])/g, '$1-$2').toLowerCase()
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
// __BOUNDING_BOX_EXPORT__

// __CUSTOM_COMPONENTS_EXPORT__

// @ts-ignore
const shaderMetadata: Record<string, Record<string, any>> = __SHADER_METADATA__

type PropMapValue = { type: 'map'; source: string; [key: string]: unknown }

function isPropMapValue(value: unknown): value is PropMapValue {
  return typeof value === 'object' && value !== null && 'type' in value && (value as Record<string, unknown>).type === 'map'
}

// Recursively format any value as a JS literal (single-quote strings, objects, primitives)
// Escapes a string for a single-quoted JS literal embedded in a double-quoted Vue
// attribute (e.g. :foo="{ k: 'v' }"). Single quotes / backslashes / newlines would
// otherwise produce invalid output; a literal double-quote is HTML-encoded so it can't
// close the attribute (Vue decodes it back before parsing the expression). Vue's
// single-quoted-attribute convention is why this can't use JSON.stringify like the
// react/solid/svelte generators do.
function escapeAttrString(s: string): string {
  return s
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t')
    .replace(/"/g, '&quot;')
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return String(v)
  if (typeof v === 'string') return `'${escapeAttrString(v)}'`
  if (typeof v === 'number' || typeof v === 'boolean') return String(v)
  if (Array.isArray(v)) return `[${v.map(formatValue).join(', ')}]`
  if (typeof v === 'object') {
    const obj = v as Record<string, unknown>
    const ks = Object.keys(obj)
    if (ks.length === 0) return '{}'
    const entries = ks.map(k => `${k}: ${formatValue(obj[k])}`)
    return `{ ${entries.join(', ')} }`
  }
  return String(v)
}

function generatePropString(props: Record<string, any>, componentType: string, indent: string = '  '): string {
  return Object.entries(props)
    .sort(([a], [b]) => a.localeCompare(b))
    .filter(([key, value]) => {
      // Skip maskType if it's the default 'alpha'
      if (key === 'maskType' && value === 'alpha') return false

      // When a custom SDF shape URL is set, `shape` and `shapeType` are both
      // editor-only metadata — the renderer only needs `shapeSdfUrl`.
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
      const componentDefaults = defaultsFor(componentType)

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
      const kebabKey = camelToKebab(key)

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
          const entries = keys.map(k => `${k}: ${typeof nonDefaultKeys[k] === 'string' ? `'${nonDefaultKeys[k]}'` : nonDefaultKeys[k]}`)
          return `:${kebabKey}="{\n${indent}    ${entries.join(`,\n${indent}    `)}\n${indent}  }"`
        } else if (keys.length === 1) {
          const k = keys[0]
          const v = nonDefaultKeys[k]
          return `:${kebabKey}="{ ${k}: ${typeof v === 'string' ? `'${v}'` : v} }"`
        }
      }

      // PropDriver (map config) — format across multiple lines for readability
      if (isPropMapValue(value)) {
        const entries = Object.keys(value).map(k => {
          const v = value[k]
          return `${k}: ${formatValue(v)}`
        })
        return `:${kebabKey}="{\n${indent}    ${entries.join(`,\n${indent}    `)}\n${indent}  }"`
      }

      if (typeof value === 'string') {
        // Detect JSON-encoded object strings and treat as objects
        if (value.startsWith('{') && value.endsWith('}')) {
          try {
            const parsed = JSON.parse(value)
            if (typeof parsed === 'object' && parsed !== null) {
              const keys = Object.keys(parsed)
              if (keys.length === 0) return `:${kebabKey}="{}"`
              if (keys.length > 1) {
                const entries = keys.map(k => `${k}: ${formatValue(parsed[k])}`)
                return `:${kebabKey}="{\n${indent}    ${entries.join(`,\n${indent}    `)}\n${indent}  }"`
              }
              const [pk] = keys
              return `:${kebabKey}="{ ${pk}: ${formatValue(parsed[pk])} }"`
            }
          } catch { /* not JSON, treat as string */ }
        }
        return `${kebabKey}="${value}"`
      } else if (Array.isArray(value)) {
        // Array prop (e.g. gradient stops) — bind as a JS array literal.
        return `:${kebabKey}="${formatValue(value)}"`
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
        if (keys.length === 0) return `:${kebabKey}="{}"`
        if (keys.length > 1) {
          const entries = keys.map(k => `${k}: ${formatValue(roundedValue[k])}`)
          return `:${kebabKey}="{\n${indent}    ${entries.join(`,\n${indent}    `)}\n${indent}  }"`
        }
        const [rk] = keys
        return `:${kebabKey}="{ ${rk}: ${formatValue(roundedValue[rk])} }"`
      } else {
        return `:${kebabKey}="${value}"`
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

export function generatePresetCode(preset: PresetConfig, colorSpace?: 'p3-linear' | 'srgb', toneMapping?: 'linear' | 'reinhard' | 'cineon' | 'aces' | 'agx' | 'neutral' | 'hable' | 'unreal', options: GeneratePresetCodeOptions = {}): string {
  customExports = options.customComponents ?? {}
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

  const generateComponentString = (config: ComponentConfig, indent: string = '    '): string => {
    const propString = config.props ? generatePropString(config.props, config.type, indent) : ''

    // Only include ID if it's referenced by another component (mask source or prop map source)
    let idString = ''
    if (config.id) {
      const isReferenced = isIdReferenced(config.id, preset.components)

      // Include ID only if it's referenced as a mask source
      if (isReferenced) {
        idString = `id="${config.id}"`
      }
    }

    // A custom component mounts through CustomShader, its definition passed as src
    const custom = customExportFor(config.type)
    const tag = custom ? 'CustomShader' : config.type
    const srcString = custom ? `:src="${custom.identifier}"` : ''

    const attributes = [srcString, idString, propString].filter(Boolean).join('\n' + indent + '  ')

    // HTMLInCanvas: inject sample HTML so users see how to swap in their own content
    const sampleChildren = (config.type === 'HTMLInCanvas' || config.type === 'DOMTexture') && (!config.children || config.children.length === 0)
      ? `${indent}  <h1>Your HTML content here</h1>`
      : null

    if (sampleChildren) {
      if (attributes) {
        return `${indent}<${tag}\n${indent}  ${attributes}>\n${sampleChildren}\n${indent}</${tag}>`
      } else {
        return `${indent}<${tag}>\n${sampleChildren}\n${indent}</${tag}>`
      }
    } else if (config.children && config.children.length > 0) {
      const childrenString = config.children
        .map(child => generateComponentString(child, indent + '  '))
        .join('\n')

      if (attributes) {
        return `${indent}<${tag}\n${indent}  ${attributes}>\n${childrenString}\n${indent}</${tag}>`
      } else {
        return `${indent}<${tag}>\n${childrenString}\n${indent}</${tag}>`
      }
    } else {
      if (attributes) {
        return `${indent}<${tag}\n${indent}  ${attributes}/>`
      } else {
        return `${indent}<${tag}/>`
      }
    }
  }

  const componentStrings = preset.components
    .map(config => generateComponentString(config))
    .join('\n')

  // Add colorSpace and toneMapping props to Shader when non-default
  const colorSpaceProp = colorSpace && colorSpace !== 'p3-linear' ? ` color-space="${colorSpace}"` : ''
  const toneMappingProp = toneMapping && toneMapping !== 'linear' ? ` tone-mapping="${toneMapping}"` : ''
  const shaderProps = colorSpaceProp + toneMappingProp

  // Build import list
  // Custom components come from their own files; the library supplies CustomShader to mount them
  const libraryTypes = Array.from(componentTypes).filter(type => !customExportFor(type))
  const usesCustom = libraryTypes.length !== componentTypes.size
  const sortedTypes = [...libraryTypes, ...(usesCustom ? ['CustomShader'] : [])].sort()
  const customImports = customImportLines(componentTypes)
  const allImports = ['Shader', ...sortedTypes]
  const uniqueImports = [...new Set(allImports)]

  const imports = uniqueImports.length > 3
    ? uniqueImports.join(',\n  ')
    : uniqueImports.join(', ')

  const importStatement = uniqueImports.length > 3
    ? `import {\n  ${imports}\n} from 'shaders/vue'`
    : `import { ${imports} } from 'shaders/vue'`

  // Build full Vue SFC
  const lines = [
    '<script setup lang="ts">',
    importStatement,
    ...customImports,
    '</script>',
    '',
    '<template>',
    `  <Shader${shaderProps}>`,
    componentStrings,
    '  </Shader>',
    '</template>'
  ]

  return lines.join('\n')
}

// Available components (auto-generated)
export const availableComponents = [
  // @ts-ignore - replaced at build time
  __COMPONENT_LIST__
]