/** The symbol a `displaceBy` effect carries its own lowering under (see std/displaceBy.ts). */
import type {GpuShaderDefinition} from '../gpu/contract'

export const LOWER_DISPLACE: unique symbol = Symbol.for('shaders.std.lowerDisplaceBy') as never

/** The lowering helpers `defineShader` hands the word, so the word's module needs no import of lower.ts. */
export interface DisplaceLoweringHelpers {
    lowerProps: (props: never) => unknown
    lowerIdentity: (rule: never, props: never) => unknown
}

export type DisplaceLowering = (definition: unknown, helpers: DisplaceLoweringHelpers) => GpuShaderDefinition
