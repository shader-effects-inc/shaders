import { createMemo, createEffect, on, onMount, onCleanup, splitProps, createUniqueId, type JSX } from 'solid-js';
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
import { ShaderContext, type ShaderContextValue, useShaderContext } from './Shader';

/**
 * `<CustomShader src={definition} …props />`
 *
 * Mounts a user-defined shader (a `defineShader` result from `shaders/std`) as a node in the
 * shader tree — the same registration, uniform bridge, metadata and child slot the generated
 * library components use, with the definition supplied at runtime instead of imported at
 * build time. Shader props are typed from the definition.
 *
 * Swapping `src` for a new definition object re-registers the node in place (children stay
 * attached); the definition's `revision` makes the renderer recompose.
 */

export interface CustomShaderLayerProps {
    children?: JSX.Element;
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

function isPropDriver(value: unknown): value is PropDriver {
    return typeof value === 'object' && value !== null && 'type' in value &&
        ((value as any).type === 'map' || (value as any).type === 'mouse' || (value as any).type === 'mouse-position' || (value as any).type === 'auto-animate');
}

const DEFAULT_TRANSFORM: TransformConfig = {
    offsetX: 0,
    offsetY: 0,
    rotation: 0,
    scale: 1,
    anchorX: 0.5,
    anchorY: 0.5,
    edges: 'transparent'
};

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

export default function CustomShader<T extends Record<string, any> = Record<string, any>>(props: CustomShaderProps<T>) {
    // Split children first so the getter is not evaluated before this component's Provider is active.
    const [local, shaderProps] = splitProps(props, [
        'children', 'src', 'blendMode', 'opacity', 'visible', 'id', 'maskSource', 'maskType',
        'renderOrder', 'transform', 'boundingBox', 'flow', 'absolute'
    ]);

    const context = useShaderContext();
    const {
        shaderParentId: parentId,
        shaderNodeRegister: parentRegister,
        shaderUniformUpdate: parentUniformUpdate,
        shaderMetadataUpdate: parentMetadataUpdate,
        shaderColorSpace
    } = context;

    const instanceId = (local.id ? local.id.replace(/[^a-zA-Z0-9_]/g, '_') : null) || createUniqueId();

    /** The definition's props → their current authored values (props over defaults), drivers included. */
    const values = createMemo(() => {
        const definition = local.src as GpuShaderDefinition<any>;
        const out: Record<string, unknown> = {};
        for (const [key, config] of Object.entries(definition.props as Record<string, PropConfig<unknown>>)) {
            const value = (shaderProps as Record<string, unknown>)[key];
            out[key] = value === undefined ? config.default : value;
        }
        return out;
    });
    const maps = createMemo(() => mapsOf(values()));

    const effectiveTransform = createMemo<TransformConfig>(() => ({...DEFAULT_TRANSFORM, ...local.transform}));
    const effectiveBoundingBox = createMemo<BoundingBoxConfig | undefined>(() => resolveBoundingBox(local.boundingBox));

    const childContextValue = createMemo<ShaderContextValue>(() => ({...context, shaderParentId: instanceId}));

    let markerRef: HTMLSpanElement | undefined;
    let detectedRenderOrder: number | undefined = undefined;
    let isRegistered = false;
    let uniforms: GpuUniformsMap | null = null;
    let lastValues: Record<string, unknown> = {};

    const metadata = (): NodeMetadata => ({
        blendMode: local.blendMode || 'normal',
        opacity: local.opacity,
        visible: local.visible === false ? false : true,
        id: local.id,
        mask: local.maskSource ? {source: local.maskSource, type: local.maskType || 'alpha'} as MaskConfig : undefined,
        maps: maps(),
        renderOrder: local.renderOrder ?? detectedRenderOrder,
        transform: effectiveTransform(),
        boundingBox: effectiveBoundingBox(),
        flow: local.flow,
        absolute: local.absolute
    } as NodeMetadata);

    const register = (definition: GpuShaderDefinition<any>) => {
        if (!definition || typeof definition.fragment !== 'function') {
            console.error('<CustomShader> needs a `src` shader definition (the result of defineShader())');
            return;
        }
        const current = values();
        setColorSpaceMode(shaderColorSpace);
        uniforms = createGpuUniformsMap(definition as any, uniformValues(definition, current), instanceId);
        lastValues = current;
        registerShader(definition);
        try {
            parentRegister(instanceId, definition.fragment, parentId, metadata(), uniforms, definition);
            isRegistered = true;
        } catch (error) {
            console.error('Error registering shader node:', error);
        }
    };

    onMount(() => {
        register(local.src as GpuShaderDefinition<any>);
        if (local.renderOrder === undefined && markerRef) {
            const parent = markerRef.parentElement;
            if (parent) {
                const siblings = parent.querySelectorAll(':scope > [data-shader-id]');
                const position = Array.from(siblings).indexOf(markerRef);
                if (position >= 0) {
                    detectedRenderOrder = position;
                    parentMetadataUpdate(instanceId, { renderOrder: position } as NodeMetadata);
                }
            }
        }
    });

    // A new definition object → rebuild uniforms and re-register the same node id.
    createEffect(on(() => local.src, (definition, previous) => {
        if (!isRegistered || definition === previous) return;
        register(definition as GpuShaderDefinition<any>);
    }, { defer: true }));

    onCleanup(() => {
        isRegistered = false;
        parentRegister(instanceId, null, null, null, null);
    });

    // Forward changed shader-prop values as uniform updates (drivers → metadata).
    createEffect(() => {
        const current = values();
        if (!isRegistered || !uniforms) return;
        let mapsChanged = false;
        for (const [key, value] of Object.entries(current)) {
            const prev = lastValues[key];
            if (value === prev) continue;
            if (isPropDriver(value) || isPropDriver(prev)) mapsChanged = true;
            else if (uniforms[key] && uniforms[key].value !== undefined) parentUniformUpdate(instanceId, key, value);
        }
        lastValues = current;
        if (mapsChanged) parentMetadataUpdate(instanceId, metadata());
    });

    // Layer metadata (blend, opacity, masks, transform).
    createEffect(() => {
        const next = metadata();
        if (!isRegistered) return;
        parentMetadataUpdate(instanceId, next);
    });

    return (
        <ShaderContext.Provider value={childContextValue()}>
            <span ref={markerRef} style={{display: 'contents'}} data-shader-id={instanceId}>
                {local.children}
            </span>
        </ShaderContext.Provider>
    );
}
