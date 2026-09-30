import React, { useContext, useEffect, useId, useMemo, useRef } from 'react';
import {
    createGpuUniformsMap,
    resolveBoundingBox,
    registerShader,
    type GpuShaderDefinition,
    type GpuUniformsMap,
    type BlendMode,
    type NodeMetadata,
    type PropConfig,
    type MaskConfig,
    type PropDriver,
    type TransformConfig,
    type BoundingBoxConfig,
    type LayoutConfig
} from 'shaders-core';
import {setColorSpaceMode} from 'shaders-core/utilities/transformations';
import { ShaderContext } from './Shader';

/**
 * `<CustomShader src={definition} …props />`
 *
 * Mounts a user-defined shader (a `defineShader` result from `shaders/std`) as a node in the
 * shader tree — the same registration, uniform bridge, metadata and child slot the generated
 * library components use, with the definition supplied at runtime instead of imported at
 * build time. Shader props are typed from the definition (`src={Halo}` → `radius`, `inner`, …).
 *
 * Swapping `src` for a new definition object re-registers the node in place (children stay
 * attached); the definition's `revision` makes the renderer recompose.
 */

export interface CustomShaderLayerProps {
    children?: React.ReactNode;
    blendMode?: BlendMode;
    opacity?: number;
    visible?: boolean;
    id?: string;
    maskSource?: string;
    maskType?: string;
    renderOrder?: number;
    transform?: Partial<TransformConfig>;
    boundingBox?: Partial<BoundingBoxConfig>;
    flow?: LayoutConfig;
    absolute?: boolean;
}

export type CustomShaderProps<T extends Record<string, any>> = CustomShaderLayerProps & {
    /** The shader definition (`defineShader({...})`). */
    src: GpuShaderDefinition<T>;
} & {
    [K in keyof T]?: T[K] | PropDriver;
};

const LAYER_KEYS = new Set([
    'children', 'src', 'blendMode', 'opacity', 'visible', 'id', 'maskSource', 'maskType',
    'renderOrder', 'transform', 'boundingBox', 'flow', 'absolute', 'ref', 'key'
]);

function isPropDriver(value: unknown): value is PropDriver {
    return typeof value === 'object' && value !== null && 'type' in value &&
        ((value as any).type === 'map' || (value as any).type === 'mouse' || (value as any).type === 'mouse-position' || (value as any).type === 'auto-animate');
}

const MARKER_STYLE = { display: 'contents' } as const;

const DEFAULT_TRANSFORM: TransformConfig = {
    offsetX: 0,
    offsetY: 0,
    rotation: 0,
    scale: 1,
    anchorX: 0.5,
    anchorY: 0.5,
    edges: 'transparent'
};

/** The definition's props → their current authored values (props over defaults), drivers included. */
function readShaderProps(definition: GpuShaderDefinition<any>, props: Record<string, unknown>): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const [key, config] of Object.entries(definition.props as Record<string, PropConfig<unknown>>)) {
        const value = LAYER_KEYS.has(key) ? undefined : props[key];
        out[key] = value === undefined ? config.default : value;
    }
    return out;
}

function uniformValues(definition: GpuShaderDefinition<any>, values: Record<string, unknown>): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(values)) {
        out[key] = isPropDriver(value) ? (definition.props as Record<string, PropConfig<unknown>>)[key]?.default : value;
    }
    return out;
}

function mapsOf(values: Record<string, unknown>): Record<string, PropDriver> | undefined {
    const maps: Record<string, PropDriver> = {};
    for (const [key, value] of Object.entries(values)) if (isPropDriver(value)) maps[key] = value;
    return Object.keys(maps).length > 0 ? maps : undefined;
}

export function CustomShader<T extends Record<string, any> = Record<string, any>>(props: CustomShaderProps<T>): React.ReactElement {
    const context = useContext(ShaderContext);
    if (!context) {
        throw new Error('<CustomShader> must be used inside a <Shader> component or another shader component');
    }
    const {
        shaderParentId: parentId,
        shaderNodeRegister: parentRegister,
        shaderUniformUpdate: parentUniformUpdate,
        shaderMetadataUpdate: parentMetadataUpdate,
        shaderColorSpace
    } = context;

    const definition = props.src as GpuShaderDefinition<any>;
    if (!definition || typeof definition.fragment !== 'function') {
        throw new Error('<CustomShader> needs a `src` shader definition (the result of defineShader())');
    }

    const reactId = useId();
    const instanceId = useMemo(() => {
        return (props.id ? props.id.replace(/[^a-zA-Z0-9_]/g, '_') : null) || reactId.replace(/[^a-zA-Z0-9_]/g, '');
    }, [props.id, reactId]);

    // Authored shader-prop values for this render (drivers included).
    const values = useMemo(() => readShaderProps(definition, props as Record<string, unknown>), [definition, props]);
    const maps = useMemo(() => mapsOf(values), [values]);

    const effectiveTransform = useMemo<TransformConfig>(() => ({...DEFAULT_TRANSFORM, ...props.transform}), [props.transform]);
    const effectiveBoundingBox = useMemo<BoundingBoxConfig | undefined>(() => resolveBoundingBox(props.boundingBox), [props.boundingBox]);

    const markerRef = useRef<HTMLSpanElement | null>(null);
    const detectedOrderRef = useRef<number | undefined>(undefined);

    // One uniform map per definition object. Rebuilt (and the node re-registered) when `src` changes.
    const uniformsRef = useRef<{definition: GpuShaderDefinition<any>; uniforms: GpuUniformsMap} | null>(null);
    if (uniformsRef.current === null || uniformsRef.current.definition !== definition) {
        setColorSpaceMode(shaderColorSpace);
        uniformsRef.current = {
            definition,
            uniforms: createGpuUniformsMap(definition as any, uniformValues(definition, values), instanceId)
        };
    }
    const uniforms = uniformsRef.current.uniforms;

    const childContextValue = useMemo(() => ({...context, shaderParentId: instanceId}), [context, instanceId]);

    const metadataFor = (): NodeMetadata => ({
        blendMode: props.blendMode || 'normal',
        opacity: props.opacity,
        visible: props.visible === false ? false : true,
        id: props.id,
        mask: props.maskSource ? {source: props.maskSource, type: props.maskType || 'alpha'} as MaskConfig : undefined,
        maps,
        renderOrder: props.renderOrder ?? detectedOrderRef.current,
        transform: effectiveTransform,
        boundingBox: effectiveBoundingBox,
        flow: props.flow,
        absolute: props.absolute
    } as NodeMetadata);
    const metadataRef = useRef(metadataFor);
    metadataRef.current = metadataFor;

    // Register (and re-register on a new definition). The node id is stable, so children stay
    // attached across a `src` swap; unregister only on unmount.
    const prevValuesRef = useRef<Record<string, unknown>>({});
    useEffect(() => {
        registerShader(definition);
        parentRegister(instanceId, definition.fragment, parentId, metadataRef.current(), uniforms, definition);
        // Clear, don't seed: the uniform map is cached per DEFINITION, so a re-registration
        // caused by a new instanceId or parentId hands the renderer values seeded at first
        // mount. Clearing makes the update effect forward every current value once — a few
        // redundant patches on a fresh mount, correct values after an id/parent change.
        prevValuesRef.current = {};
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [instanceId, parentId, parentRegister, definition, uniforms]);

    useEffect(() => {
        return () => {
            parentRegister(instanceId, null, null, null, null);
        };
    }, [instanceId, parentRegister]);

    // Detect DOM position among siblings for render order.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    useEffect(() => {
        if (props.renderOrder === undefined && markerRef.current) {
            const parent = markerRef.current.parentElement;
            if (parent) {
                const siblings = parent.querySelectorAll(':scope > [data-shader-id]');
                const position = Array.from(siblings).indexOf(markerRef.current);
                if (position >= 0) {
                    detectedOrderRef.current = position;
                    parentMetadataUpdate(instanceId, { renderOrder: position } as NodeMetadata);
                }
            }
        }
    }, []);

    // Forward changed shader-prop values as uniform updates.
    useEffect(() => {
        for (const [key, value] of Object.entries(values)) {
            if (isPropDriver(value)) continue;
            const entry = uniforms[key];
            if (!entry || entry.value === undefined) continue;
            if (prevValuesRef.current[key] !== value) {
                parentUniformUpdate(instanceId, key, value);
                prevValuesRef.current[key] = value;
            }
        }
    }, [values, uniforms, instanceId, parentUniformUpdate]);

    // Layer metadata (blend, opacity, masks, transform, drivers).
    useEffect(() => {
        parentMetadataUpdate(instanceId, metadataRef.current());
    }, [props.blendMode, props.opacity, props.visible, props.maskSource, props.maskType, props.renderOrder, props.id, maps,
        effectiveTransform, effectiveBoundingBox, props.flow, props.absolute, instanceId, parentMetadataUpdate]);

    return (
        <ShaderContext.Provider value={childContextValue}>
            <span ref={markerRef} style={MARKER_STYLE} data-shader-id={instanceId}>
                {props.children}
            </span>
        </ShaderContext.Provider>
    );
}

export default CustomShader;
