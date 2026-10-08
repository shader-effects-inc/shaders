// Shader code may only import from the shader kit (@coreroot/gpu/kit). Direct `typegpu` / `unplugin-typegpu` imports
// are restricted to src/gpu/**. This flat config exists solely to enforce that boundary;
// run it with `pnpm lint:facade`.
import tsParser from '@typescript-eslint/parser'

const message =
    'Import from @coreroot/gpu/kit — direct typegpu imports are restricted to src/gpu/'

export default [
    {
        ignores: ['dist/**', 'node_modules/**'],
    },
    {
        files: ['src/**/*.ts'],
        // The facade's own home — the one place allowed to import typegpu directly — and the
        // tests, which deliberately reach into typegpu internals to verify the facade.
        ignores: ['src/gpu/**', 'src/__tests__/**'],
        languageOptions: {
            parser: tsParser,
            parserOptions: {
                ecmaVersion: 'latest',
                sourceType: 'module',
            },
        },
        rules: {
            'no-restricted-imports': [
                'error',
                {
                    patterns: [
                        {
                            group: ['typegpu', 'typegpu/*', 'unplugin-typegpu', 'unplugin-typegpu/*'],
                            message,
                        },
                    ],
                },
            ],
        },
    },
]
