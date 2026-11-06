<script lang="ts">
    import { getContext, setContext, onMount, onDestroy, untrack } from 'svelte';
    import {
        createGpuUniformsMap,
        resolveBoundingBox,
        type GpuUniformsMap,
        type BlendMode,
        type NodeMetadata,
        type PropConfig,
        type MaskConfig,
        type PropDriver,
        type TransformConfig,
        type BoundingBoxConfig,
        type BoundingBoxInput,
        type LayoutConfig
    } from 'shaders-core';
    import {setColorSpaceMode} from 'shaders-core/utilities/transformations';

    // @ts-ignore - this import is replaced at build time
    import { componentDefinition, type ComponentProps } from 'shaders-core/__SHADER_NAME__';

    // Warn once per page load if this is an experimental component
    let _experimentalWarnedOnce = false;
    if ((componentDefinition as any).experimental && !_experimentalWarnedOnce) {
        _experimentalWarnedOnce = true;
        const _e = (componentDefinition as any).experimental;
        console.info(`%c⚠ [Shaders] ${componentDefinition.name} is experimental: ${_e.message}`, 'color: #f59e0b; font-weight: bold');
    }

    /**
     * Define component props including blend mode, opacity, visibility, masking, and transformation
     */
    interface ExtendedComponentProps extends Partial<ComponentProps> {
__MAPPABLE_PROP_DECLS__        blendMode?: BlendMode;
        opacity?: number;
        visible?: boolean;
        id?: string;
        maskSource?: string;
        maskType?: string;
        renderOrder?: number;
        transform?: Partial<TransformConfig>;
        boundingBox?: BoundingBoxInput;
        flow?: LayoutConfig;
        absolute?: boolean;
        children?: import('svelte').Snippet;
    }

    function isPropDriver(value: unknown): value is PropDriver {
        return typeof value === 'object' && value !== null && 'type' in value &&
            ((value as any).type === 'map' || (value as any).type === 'mouse' || (value as any).type === 'mouse-position' || (value as any).type === 'auto-animate');
    }

    /**
     * Default transform configuration (optimized for zero overhead)
     */
    const DEFAULT_TRANSFORM: TransformConfig = {
        offsetX: 0,
        offsetY: 0,
        rotation: 0,
        scale: 1,
        anchorX: 0.5,
        anchorY: 0.5,
        edges: 'transparent'
    };

    // Define the component props and their default values from the shader definition
    const componentDefaults = {
        blendMode: 'normal' as BlendMode,
        visible: true,
        // opacity intentionally has no default - handled by renderer
        // transform intentionally has no default - handled by effectiveTransform
        ...Object.entries(componentDefinition.props).reduce(
            (acc, [key, config]) => {
                acc[key] = (config as unknown as PropConfig<typeof config>).default;
                return acc;
            },
            {} as Record<string, any>
        )
    };

    // Declare props using Svelte 5's syntax
    const props: ExtendedComponentProps = $props();

    // Apply defaults manually since Svelte 5 doesn't have withDefaults equivalent
    // Use $derived so the metadata $effect re-runs when any of these change at runtime
    const blendMode = $derived(props.blendMode ?? componentDefaults.blendMode);
    const opacity = $derived(props.opacity); // No default - handled by renderer
    const visible = $derived(props.visible ?? componentDefaults.visible); // Default to true
    const id = $derived(props.id);
    const maskSource = $derived(props.maskSource);
    const maskType = $derived(props.maskType);
    const renderOrder = $derived(props.renderOrder);
    const { children } = props;

    // Collect PropDriver values from shader props into the maps metadata structure
    const mapsFromProps = $derived.by(() => {
        const maps: Record<string, PropDriver> = {};
        for (const key of Object.keys(componentDefinition.props)) {
            const val = (props as any)[key];
            if (isPropDriver(val)) maps[key] = val as PropDriver;
        }
        return Object.keys(maps).length > 0 ? maps : undefined;
    });

    /**
     * Computed transform that merges user-provided values with defaults
     */
    const effectiveTransform = $derived({
        ...DEFAULT_TRANSFORM,
        ...props.transform
    });

    /**
     * Effective bounding box — partial configs merge with full-frame defaults,
     * absent stays undefined so the bbox render path is not activated
     */
    const effectiveBoundingBox = $derived(resolveBoundingBox(props.boundingBox));

    /**
     * FIRST: Get the parent ID from context BEFORE setting our own context
     */
    const parentId = getContext<string>('shaderParentId');
    if (parentId === undefined) {
        throw new Error('Shader components must be used inside an <Shader> component or another shader component');
    }

    /**
     * Use the provided ID or generate a unique identifier for this component instance
     */
    const instanceId = (id ? id.replace(/[^a-zA-Z0-9_]/g, '_') : null) || Math.random().toString(36).substring(2, 10);

    /**
     * THEN: Provide our unique identifier to child components
     */
    setContext('shaderParentId', instanceId);

    /**
     * Creates a non-reactive object containing only props that differ from defaults
     * This optimization prevents unnecessary GPU uniform updates for unchanged values
     * Special props like blendMode and opacity are handled separately
     */
    const shaderReadyProps = $derived.by(() => {
        let baseProps = { ...componentDefaults };

        // Only include props that differ from defaults (excluding special props and PropDrivers)
        for (const key in props) {
            if (key !== 'blendMode' && key !== 'opacity' && key !== 'visible' &&
                key !== 'id' && key !== 'maskSource' && key !== 'maskType' && key !== 'renderOrder' &&
                key !== 'transform' && key !== 'boundingBox' && key !== 'flow' && key !== 'absolute' && key !== 'children') {
                const propValue = (props as any)[key];
                if (isPropDriver(propValue)) continue; // PropDrivers go to metadata.maps, not uniforms
                const defaultValue = (componentDefaults as any)[key];
                if (propValue !== undefined && propValue !== defaultValue) {
                    (baseProps as any)[key] = propValue;
                }
            }
        }
        return baseProps;
    });

    /**
     * Get the color space from the root Shader component.
     * Used to set the global color space mode before creating uniforms.
     */
    const shaderColorSpace = getContext<() => 'p3-linear' | 'srgb'>('shaderColorSpace');

    /**
     * (Re)build the GPU uniforms map under the current color-space mode.
     * A renderer reset (e.g. colorSpace change) re-initializes the renderer with a new
     * mode; a cached map would keep color stops converted for the old mode.
     */
    let uniforms: GpuUniformsMap;
    function buildUniforms() {
        if (shaderColorSpace) {
            setColorSpaceMode(shaderColorSpace());
        }
        uniforms = createGpuUniformsMap(componentDefinition, shaderReadyProps, instanceId);
    }
    buildUniforms();

    /**
     * Get the node registration function from parent context
     */
    const parentRegister = getContext<(id: string, fragmentNodeFunc: any, parentId: string | null, metadata: NodeMetadata | null, uniforms: GpuUniformsMap | null, componentDefinition: any, domCanvas?: HTMLCanvasElement) => void>('shaderNodeRegister');
    if (parentRegister === undefined) {
        throw new Error('Shader components must be used inside an <Shader> component or another shader component');
    }

    /**
     * Get the uniform update function from parent context
     */
    const parentUniformUpdate = getContext<(nodeId: string, uniformName: string, value: any) => void>('shaderUniformUpdate');
    if (parentUniformUpdate === undefined) {
        throw new Error('Shader components require shaderUniformUpdate from parent');
    }

    /**
     * Get the metadata update function from parent context
     */
    const parentMetadataUpdate = getContext<(nodeId: string, metadata: NodeMetadata) => void>('shaderMetadataUpdate');
    if (parentMetadataUpdate === undefined) {
        throw new Error('Shader components require shaderMetadataUpdate from parent');
    }

    // capturesDOM — canvas layoutsubtree portal for HTMLInCanvas-style shaders
    const isCapturesDOM = !!(componentDefinition as any).capturesDOM;
    let captureCanvas: HTMLCanvasElement | undefined;
    let captureW = $state(typeof window !== 'undefined' ? Math.round(window.innerWidth * Math.min(window.devicePixelRatio, 2)) : 0);
    let captureH = $state(typeof window !== 'undefined' ? Math.round(window.innerHeight * Math.min(window.devicePixelRatio, 2)) : 0);

    function teleportToBody(node: HTMLElement) {
        document.body.appendChild(node);
        return { destroy() { node.remove(); } };
    }

    if (isCapturesDOM) {
        const onWinResize = () => {
            const d = Math.min(window.devicePixelRatio, 2);
            captureW = Math.round(window.innerWidth * d);
            captureH = Math.round(window.innerHeight * d);
        };
        onMount(() => { window.addEventListener('resize', onWinResize); });
        onDestroy(() => { window.removeEventListener('resize', onWinResize); });
    }

    // DOM marker ref for determining render order from template position
    let orderMarker: HTMLSpanElement;

    // Stores the DOM-detected render order
    let detectedRenderOrder: number | undefined = undefined;

    // Flag to track when component is registered
    let isRegistered = $state(false);

    // Setup uniform watchers with registration guard
    Object.entries(uniforms).forEach(([propName, entry]) => {
        $effect(() => {
            // Only run after component is registered
            if (!isRegistered) return;

            // GPU bridge entries carry `.value` (no v1 `.uniform` node). colorStops expands into
            // extra array fields (colorsArray/…) that aren't component props — their `props[name]`
            // is always undefined, so the guard below skips them.
            if (entry && entry.value !== undefined) {
                const newValue = (props as any)[propName];
                if (newValue !== undefined && !isPropDriver(newValue)) {
                    // Send raw value - renderer will handle transformation
                    // PropDriver values go to metadata.maps, not uniforms
                    parentUniformUpdate(instanceId, propName, newValue);
                }
            }
        });
    });

    // Watch blend mode, opacity, visibility, masking, transformations, and prop maps changes
    $effect(() => {
        // Only run after component is registered
        if (!isRegistered) return;

        const metadata: NodeMetadata = {
            blendMode,
            opacity,
            visible: visible === false ? false : true,
            id,
            mask: maskSource ? {
                source: maskSource,
                type: (maskType || 'alpha') as MaskConfig['type']
            } : undefined,
            maps: mapsFromProps,
            renderOrder: renderOrder ?? detectedRenderOrder,
            transform: effectiveTransform,
            boundingBox: effectiveBoundingBox,
            flow: props.flow,
            absolute: props.absolute
        };
        parentMetadataUpdate(instanceId, metadata);
    });

    /**
     * Get the renderer reset signal from the root Shader component.
     * When it bumps, the renderer has been re-initialized (cleanup() cleared the node
     * registry) and this component must register again.
     */
    const rendererResetSignal = getContext<() => number>('shaderRendererResetSignal');

    /**
     * Register this component with the renderer
     */
    function registerWithRenderer() {
        // Register this component with safety check
        if (componentDefinition && typeof componentDefinition.fragment === 'function') {
            parentRegister(
                instanceId,
                componentDefinition.fragment,
                parentId,
                {
                    blendMode,
                    opacity,
                    visible: visible !== false ? true : false,
                    id,
                    mask: maskSource ? {
                        source: maskSource,
                        type: (maskType || 'alpha') as MaskConfig['type']
                    } as MaskConfig : undefined,
                    maps: mapsFromProps,
                    renderOrder: renderOrder ?? detectedRenderOrder,
                    transform: effectiveTransform,
                    boundingBox: effectiveBoundingBox,
                    flow: props.flow,
                    absolute: props.absolute
                },
                uniforms,
                componentDefinition,
                isCapturesDOM ? captureCanvas : undefined
            );

            // Set flag to enable effects after successful registration
            isRegistered = true;
        } else {
            console.error('componentDefinition.fragment is not a function:', {
                componentDefinition,
                fragment: componentDefinition?.fragment,
                type: typeof componentDefinition?.fragment
            });
        }
    }

    // Register this component after mount to ensure parent is ready
    onMount(() => {
        registerWithRenderer();

        // Detect DOM position for correct render ordering
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

    // Re-register after renderer resets (e.g. colorSpace/toneMapping change tore the
    // renderer down and re-initialized it). The signal starts at 0 so the initial
    // registration above is not duplicated by this effect.
    $effect(() => {
        const signal = rendererResetSignal?.();
        if (signal === undefined || signal === 0 || !isRegistered) return;
        untrack(() => {
            // Rebuild uniforms under the (possibly new) color-space mode before
            // re-registering; the cached map holds color stops for the old mode.
            buildUniforms();
            registerWithRenderer();
        });
    });

    // Clean up node from registry when component is unmounted
    onDestroy(() => {
        isRegistered = false;
        parentRegister(instanceId, null, null, null, null);
    });
</script>

<span bind:this={orderMarker} style="display:contents" data-shader-id={instanceId}>
    {#if !isCapturesDOM}
        {@render children?.()}
    {/if}
</span>
{#if isCapturesDOM}
    <canvas use:teleportToBody bind:this={captureCanvas} layoutsubtree
            width={captureW} height={captureH}
            style="position:fixed;inset:0;width:100vw;height:100vh;z-index:-9999">
        {@render children?.()}
    </canvas>
{/if}