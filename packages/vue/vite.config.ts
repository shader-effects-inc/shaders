import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import dts from 'vite-plugin-dts';
import {externalizeDeps} from 'vite-plugin-externalize-deps';
import { resolve } from 'path';
import fs from 'fs';

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

// Workaround for rolldown bug rolldown/rolldown#8761: in preserveModules mode
// with object-form lib.entry, the default `output.sanitizeFileName` is bypassed
// for chunks whose source ID carries a query (Vue's `Aurora.vue?vue&type=script…`).
// Until the fix in rolldown PR #9289 ships, derive the entry filename ourselves so
// `?`, `&`, `=` are replaced with `_` (the shape rolldown emitted up to beta.53
// and the same shape we shipped in shaders v2.5.111). Drop this when we land on a
// rolldown release that includes #9289.
const sanitizeChunkName = (name: string) => name.replace(/[?&=]/g, '_');

// rolldown-vite 7 emitted every preserved (non-entry) module flat at the dist root by its basename
// (`components/Aurora.vue` → `Aurora.js`); Vite 8 / Rolldown 1.x keeps the source folders. Keep the
// published layout: real lib entries keep their path, everything else is flattened as before.
const entryFileName = (chunk: { name: string; isEntry: boolean }) => {
    const name = sanitizeChunkName(chunk.name)
    return chunk.isEntry ? `${name}.js` : `${name.split('/').pop()!.replace(/\.vue$/, '')}.js`
};

export default defineConfig({
    plugins: [
        externalizeDeps({
            except: ['colorjs.io'] // Bundled in core, not a peer dependency
        }),
        vue(),
        dts({
            include: ['src'],
            beforeWriteFile: (filePath, content) => {
                // Calculate the correct relative path to core based on file depth
                // Calculate the correct relative path to core based on file depth
                // Files in dist/vue/ need ../core, files in dist/vue/components/ need ../../core
                const transformedPath = filePath.replace('/src/', '/')
                const afterDist = transformedPath.split('/dist/')[1] || ''
                const depth = afterDist.split('/').length - 1 // -1 for the filename itself
                const relativePath = '../' + '../'.repeat(depth) + 'core'

                // Rewrite shaders-core imports to relative paths
                // Note: shaders-core/ShaderName maps to dist/shaders/ShaderName in core package
                // but shaders-core and shaders-core/telemetry map directly to dist/
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
                    return 'utils/generatePresetCode.js';
                }
                return `${entryName}.js`;
            }
        },
        rollupOptions: {
            output: {
                preserveModules: true,
                preserveModulesRoot: 'src',
                exports: 'named',
                entryFileNames: entryFileName
            }
        },
        rolldownOptions: {
            output: {
                // Rolldown 1.x keeps JSDoc in the JS output; rolldown-vite 7 dropped it (types carry the docs).
                comments: { legal: true, annotation: true, jsdoc: false },
                preserveModules: true,
                preserveModulesRoot: 'src',
                exports: 'named',
                entryFileNames: entryFileName,
                globals: {
                    vue: 'Vue'
                }
            }
        },
    },
    define: {
        __SHADERS_VERSION__: JSON.stringify(getPackageVersion())
    },
});
