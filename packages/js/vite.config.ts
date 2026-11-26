import { defineConfig } from 'vite'
import dts from 'vite-plugin-dts'
import { externalizeDeps } from 'vite-plugin-externalize-deps'
import { resolve } from 'path'
import fs from 'fs'

function getPackageVersion(): string {
  const shadersPkgPath = resolve(__dirname, '../shaders/package.json')
  const frameworkPkgPath = resolve(__dirname, 'package.json')
  if (fs.existsSync(shadersPkgPath)) {
    try {
      const shadersPkg = JSON.parse(fs.readFileSync(shadersPkgPath, 'utf-8'))
      return shadersPkg.version
    } catch (_) { /* fall through */ }
  }
  try {
    return JSON.parse(fs.readFileSync(frameworkPkgPath, 'utf-8')).version
  } catch (_) {
    return '0.0.0'
  }
}

export default defineConfig({
  plugins: [
    externalizeDeps({
      except: ['colorjs.io']
    }),
    dts({
      include: ['src'],
      beforeWriteFile: (filePath, content) => {
        const transformedPath = filePath.replace('/src/', '/')
        const afterDist = transformedPath.split('/dist/')[1] || ''
        const depth = afterDist.split('/').length - 1
        const relativePath = '../' + '../'.repeat(depth) + 'core'

        content = content.replace(
          /from ['"]shaders-core\/telemetry['"]/g,
          `from '${relativePath}/telemetry'`
        )
        content = content.replace(
          /from ['"]shaders-core\/registry['"]/g,
          `from '${relativePath}/registry'`
        )
        content = content.replace(
          /from ['"]shaders-core\/([^'"]+)['"]/g,
          `from '${relativePath}/shaders/$1'`
        )
        content = content.replace(
          /from ['"]shaders-core['"]/g,
          `from '${relativePath}'`
        )
        content = content.replace(
          /import\(['"]shaders-core\/telemetry['"]\)/g,
          `import('${relativePath}/telemetry')`
        )
        content = content.replace(
          /import\(['"]shaders-core\/registry['"]\)/g,
          `import('${relativePath}/registry')`
        )
        content = content.replace(
          /import\(['"]shaders-core\/([^'"]+)['"]\)/g,
          `import('${relativePath}/shaders/$1')`
        )
        content = content.replace(
          /import\(['"]shaders-core['"]\)/g,
          `import('${relativePath}')`
        )

        return {
          filePath: transformedPath,
          content,
        }
      },
    }),
  ],
  build: {
    minify: false,
    lib: {
      entry: {
        index: resolve(__dirname, 'src/index.ts'),
        'utils/generatePresetCode': resolve(__dirname, 'src/utils/generatePresetCode.ts')
      },
      formats: ['es'],
      fileName: (_format, entryName) => {
        if (entryName === 'utils/generatePresetCode') {
          return 'utils/generatePresetCode.js'
        }
        return `${entryName}.js`
      }
    },
    rollupOptions: {
      output: {
        preserveModules: true,
        preserveModulesRoot: 'src',
        exports: 'named'
      }
    },
    rolldownOptions: {
      output: {
          // Rolldown 1.x keeps JSDoc in the JS output; rolldown-vite 7 dropped it (types carry the docs).
          comments: { legal: true, annotation: true, jsdoc: false },
        preserveModules: true,
        preserveModulesRoot: 'src',
        exports: 'named'
      }
    },
  },
  define: {
    __SHADERS_VERSION__: JSON.stringify(getPackageVersion())
  },
})
