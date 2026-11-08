/**
 * The record of a component's own rendered output, shared by the body lowering (which renders
 * the body to a texture when the component reads `previousFrame`) and the history copy.
 */
import type {GpuFragmentParams} from '../../gpu/contract'

export interface SelfOutput {
    /** The render-to-texture key of the body's output this composition, or null. */
    key: string | null
    /** The bound texture, once the composition has allocated it. */
    texture: unknown
}

const selfOutputs = new WeakMap<object, SelfOutput>()

export function selfOutput(params: GpuFragmentParams): SelfOutput {
    let s = selfOutputs.get(params)
    if (!s) selfOutputs.set(params, (s = {key: null, texture: null}))
    return s
}

const premultipliedOutputs = new WeakMap<object, Set<string>>()

/** The compute output keys of a component's inputs that hold premultiplied alpha. */
export function premultipliedKeys(params: GpuFragmentParams): Set<string> {
    let s = premultipliedOutputs.get(params)
    if (!s) premultipliedOutputs.set(params, (s = new Set()))
    return s
}
