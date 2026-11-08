/**
 * Input words: what the engine computes for a component before its WGSL runs. List them in a
 * definition's `inputs` and read each by name in WGSL.
 */
export {child, isInputNode, type InputNode, type InputSource, type ChildSource, type ScalarArg} from './types'
export {blur, type BlurInput} from './blur'
export {bloom, type BloomInput} from './bloom'
export {pass, downsample, directionalBlur, type PassInput, type PassFormat, type DownsampleInput, type DirectionalBlurInput} from './passes'
export {previousFrame, type PreviousFrameInput} from './history'
export {buffer, kernel, struct, type BufferInput, type KernelInput, type StructType, type StructFieldType} from './compute'
export {rasterize, type RasterizeInput} from './rasterize'
export {events, type EventsInput} from './events'
export {pointerTrail, type PointerTrailInput} from './trail'
export {text, samplePoints, type TextInput} from './text'
export type {ParticlesInput, ParticleForce, ParticleSpawn} from './particles'
export type {FluidInput} from './fluid'
