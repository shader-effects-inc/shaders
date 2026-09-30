<script lang="ts" generics="T extends Record<string, any> = Record<string, any>">
    /**
     * <CustomShader src={definition} …props />
     *
     * Mounts a user-defined shader (a `defineShader` result from `shaders/std`) as a node in
     * the shader tree — the same registration, uniform bridge, metadata and child slot the
     * generated library components use, with the definition supplied at runtime instead of
     * imported at build time. Shader props are passed as ordinary props, typed from the
     * definition.
     *
     * Swapping `src` for a new definition object re-registers the node in place (children
     * stay attached); the definition's `revision` makes the renderer recompose.
     */
    import { getContext, setContext, onMount, onDestroy, untrack } from 'svelte';
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

    type Props = {
        /** The shader definition (`defineShader({...})`). */
        src: GpuShaderDefinition<T>;
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
        children?: import('svelte').Snippet;
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

    const {
        src,
        blendMode = 'normal',
        opacity,
        visible = true,
        id,
        maskSource,
        maskType,
        renderOrder,
        transform,
        boundingBox,
        flow,
        absolute,
        children,
        ...shaderProps
    }: Props = $props();

    /** The definition's props → their current authored values (props over defaults), drivers included. */
    function readShaderProps(definition: GpuShaderDefinition<any>): Record<string, unknown> {
        const out: Record<string, unknown> = {};
        for (const [key, config] of Object.entries(definition.props as Record<string, PropConfig<unknown>>)) {
            const value = (shaderProps as Record<string, unknown>)[key];
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

    const effectiveTransform = $derived({...DEFAULT_TRANSFORM, ...transform});
    const effectiveBoundingBox = $derived(resolveBoundingBox(boundingBox));

    const parentId = getContext<string>('shaderParentId');
    if (parentId === undefined) {
        throw new Error('<CustomShader> must be used inside a <Shader> component or another shader component');
    }
    // svelte-ignore state_referenced_locally
    const instanceId = (id ? id.replace(/[^a-zA-Z0-9_]/g, '_') : null) || Math.random().toString(36).substring(2, 10);
    setContext('shaderParentId', instanceId);

    const shaderColorSpace = getContext<() => 'p3-linear' | 'srgb'>('shaderColorSpace');
    const parentRegister = getContext<(id: string, fragmentNodeFunc: any, parentId: string | null, metadata: NodeMetadata | null, uniforms: GpuUniformsMap | null, componentDefinition: any, domCanvas?: HTMLCanvasElement) => void>('shaderNodeRegister');
    const parentUniformUpdate = getContext<(nodeId: string, uniformName: string, value: any) => void>('shaderUniformUpdate');
    const parentMetadataUpdate = getContext<(nodeId: string, metadata: NodeMetadata) => void>('shaderMetadataUpdate');
    if (!parentRegister || !parentUniformUpdate || !parentMetadataUpdate) {
        throw new Error('<CustomShader> must be used inside a <Shader> component or another shader component');
    }

    let orderMarker: HTMLSpanElement;
    let detectedRenderOrder: number | undefined = undefined;
    let isRegistered = $state(false);
    let definition: GpuShaderDefinition<any> | null = null;
    let uniforms: GpuUniformsMap | null = null;
    let lastValues: Record<string, unknown> = {};

    function metadata(values: Record<string, unknown>): NodeMetadata {
        return {
            blendMode,
            opacity,
            visible: visible === false ? false : true,
            id,
            mask: maskSource ? {source: maskSource, type: (maskType || 'alpha') as MaskConfig['type']} : undefined,
            maps: mapsOf(values),
            renderOrder: renderOrder ?? detectedRenderOrder,
            transform: effectiveTransform,
            boundingBox: effectiveBoundingBox,
            flow,
            absolute
        } as NodeMetadata;
    }

    function register(next: GpuShaderDefinition<any>) {
        if (!next || typeof next.fragment !== 'function') {
            console.error('<CustomShader> needs a `src` shader definition (the result of defineShader())');
            return;
        }
        definition = next;
        const values = readShaderProps(next);
        if (shaderColorSpace) setColorSpaceMode(shaderColorSpace());
        uniforms = createGpuUniformsMap(next as any, uniformValues(next, values), instanceId);
        lastValues = values;
        registerShader(next);
        parentRegister(instanceId, next.fragment, parentId, metadata(values), uniforms, next);
        isRegistered = true;
    }

    // Register on mount and again whenever a NEW definition object arrives (same node id).
    $effect(() => {
        const next = src;
        untrack(() => register(next));
    });

    // Forward changed shader-prop values as uniform updates (drivers → metadata).
    $effect(() => {
        if (!isRegistered || !definition || !uniforms) return;
        const values = readShaderProps(definition);
        untrack(() => {
            let mapsChanged = false;
            for (const [key, value] of Object.entries(values)) {
                const prev = lastValues[key];
                if (value === prev) continue;
                if (isPropDriver(value) || isPropDriver(prev)) mapsChanged = true;
                else if (uniforms![key] && uniforms![key].value !== undefined) parentUniformUpdate(instanceId, key, value);
            }
            lastValues = values;
            if (mapsChanged) parentMetadataUpdate(instanceId, metadata(values));
        });
    });

    // Layer metadata (blend, opacity, masks, transform).
    $effect(() => {
        const next = metadata(lastValues);
        if (!isRegistered) return;
        parentMetadataUpdate(instanceId, next);
    });

    onMount(() => {
        if (renderOrder === undefined && orderMarker) {
            const parent = orderMarker.parentElement;
            if (parent) {
                const siblings = parent.querySelectorAll(':scope > [data-shader-id]');
                const position = Array.from(siblings).indexOf(orderMarker);
                if (position >= 0) {
                    detectedRenderOrder = position;
                    parentMetadataUpdate(instanceId, { renderOrder: position } as NodeMetadata);
                }
            }
        }
    });

    onDestroy(() => {
        isRegistered = false;
        parentRegister(instanceId, null, null, null, null, null);
    });
</script>

<span bind:this={orderMarker} style="display:contents" data-shader-id={instanceId}>
    {@render children?.()}
</span>
