import { createMemo, createEffect, on, onMount, onCleanup, splitProps, createUniqueId, createSignal, type JSX } from 'solid-js';
import { Portal } from 'solid-js/web';
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
// @ts-ignore - replaced at build time
import { componentDefinition, type ComponentProps } from 'shaders-core/__SHADER_NAME__';

// Re-export the ComponentProps type for better TypeScript inference
export type { ComponentProps };

// Warn once per page load if this is an experimental component
let _experimentalWarnedOnce = false;
if ((componentDefinition as any).experimental && !_experimentalWarnedOnce) {
    _experimentalWarnedOnce = true;
    const _e = (componentDefinition as any).experimental;
    console.info(`%c⚠ [Shaders] ${componentDefinition.name} is experimental: ${_e.message}`, 'color: #f59e0b; font-weight: bold');
}
// @ts-ignore - correct path at build time
import { ShaderContext, type ShaderContextValue, useShaderContext } from '../engine/Shader';

/**
 * Base props that all shader components have
 */
interface BaseShaderProps {
    children?: JSX.Element;
    blendMode?: BlendMode;
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
}

function isPropDriver(value: unknown): value is PropDriver {
    return typeof value === 'object' && value !== null && 'type' in value &&
        ((value as any).type === 'map' || (value as any).type === 'mouse' || (value as any).type === 'mouse-position' || (value as any).type === 'auto-animate');
}

/**
 * Component-specific props that merge base props with shader-specific props
 * Note: ComponentProps are made optional since they have defaults from the shader definition
 */
type ShaderComponentProps = BaseShaderProps & Partial<Omit<ComponentProps, __OMIT_TYPE__>> & {
__MAPPABLE_PROP_DECLS__};

// Default transform configuration (optimized for zero overhead)
const DEFAULT_TRANSFORM: TransformConfig = {
    offsetX: 0,
    offsetY: 0,
    rotation: 0,
    scale: 1,
    anchorX: 0.5,
    anchorY: 0.5,
    edges: 'transparent'
};

// Extract default props from shader definition
const defaultProps: Record<string, any> = {
    blendMode: 'normal',
    visible: true,
    // opacity intentionally has no default - handled by renderer
    // transform intentionally has no default - handled by effectiveTransform
};

try {
    // Safely extract default values from shader definition
    if (componentDefinition && componentDefinition.props) {
        Object.entries(componentDefinition.props).forEach(([key, config]) => {
            const propConfig = config as PropConfig<any>;
            if (propConfig && typeof propConfig === 'object' && 'default' in propConfig) {
                defaultProps[key] = propConfig.default;
            }
        });
    }
} catch (e) {
    console.warn('Error extracting default props:', e);
}

/**
 * The main Solid wrapper component for Shader shader nodes
 */
export default function ShaderComponent(props: ShaderComponentProps) {
    // CRITICAL: Split children from other props FIRST to avoid triggering children getter
    // In SolidJS, props is a reactive proxy - iterating with Object.entries() would evaluate
    // the children getter, causing children to be created before this component's Provider is active
    const [local, otherProps] = splitProps(props, ['children']);

    // Get Shader context from parent
    const context = useShaderContext();

    const {
        shaderParentId: parentId,
        shaderNodeRegister: parentRegister,
        shaderUniformUpdate: parentUniformUpdate,
        shaderMetadataUpdate: parentMetadataUpdate,
        shaderColorSpace,
        shaderRendererResetSignal
    } = context;

    // Create instance ID from name or generate a deterministic one (SSR-safe)
    const instanceId = (props.id ? props.id.replace(/[^a-zA-Z0-9_]/g, '_') : null) || createUniqueId();

    // Compute final props, merging defaults with user props
    // Uses otherProps (which excludes children) to avoid early children evaluation
    const effectiveProps = createMemo(() => {
        let baseProps = { ...defaultProps };

        for (const [key, value] of Object.entries(otherProps)) {
            if (value !== undefined && !isPropDriver(value)) {
                baseProps[key] = value;
            }
        }

        return baseProps;
    });

    // Compute effective transform, merging user-provided values with defaults
    const effectiveTransform = createMemo<TransformConfig>(() => ({
        ...DEFAULT_TRANSFORM,
        ...props.transform
    }));

    // Compute effective bounding box — partial configs merge with full-frame defaults,
    // absent stays undefined so the bbox render path is not activated
    const effectiveBoundingBox = createMemo<BoundingBoxConfig | undefined>(() => resolveBoundingBox(props.boundingBox));

    // Collect PropDriver values from shader props into the maps metadata structure
    const mapsFromProps = createMemo(() => {
        const maps: Record<string, PropDriver> = {};
        for (const [key, value] of Object.entries(otherProps)) {
            if (isPropDriver(value)) maps[key] = value as PropDriver;
        }
        return Object.keys(maps).length > 0 ? maps : undefined;
    });

    // Create uniforms map once
    let uniformsMap: GpuUniformsMap | null = null;

    // Initialize uniforms on first access
    // Set the global color space mode before creating uniforms so colors are transformed correctly
    const getUniformsMap = () => {
        if (!uniformsMap) {
            setColorSpaceMode(shaderColorSpace());
            uniformsMap = createGpuUniformsMap(componentDefinition, effectiveProps(), instanceId);
        }
        return uniformsMap;
    };

    // Create context for children with this component as their parent
    const childContextValue = createMemo<ShaderContextValue>(() => ({
        ...context,
        shaderParentId: instanceId
    }));

    // capturesDOM — canvas layoutsubtree portal for HTMLInCanvas-style shaders
    const isCapturesDOM = !!(componentDefinition as any).capturesDOM;
    let captureCanvasEl: HTMLCanvasElement | undefined;
    const [captureSize, setCaptureSize] = createSignal({
        w: typeof window !== 'undefined' ? Math.round(window.innerWidth * Math.min(window.devicePixelRatio, 2)) : 0,
        h: typeof window !== 'undefined' ? Math.round(window.innerHeight * Math.min(window.devicePixelRatio, 2)) : 0
    });

    if (isCapturesDOM) {
        const onWinResize = () => {
            const d = Math.min(window.devicePixelRatio, 2);
            setCaptureSize({ w: Math.round(window.innerWidth * d), h: Math.round(window.innerHeight * d) });
        };
        onMount(() => window.addEventListener('resize', onWinResize));
        onCleanup(() => window.removeEventListener('resize', onWinResize));
    }

    // DOM marker ref for determining render order from template position
    let markerRef: HTMLSpanElement | undefined;

    // Stores the DOM-detected render order
    let detectedRenderOrder: number | undefined = undefined;

    // Track registration state
    let isRegistered = false;

    // Register this node with the parent renderer
    const registerWithRenderer = () => {
        const uniforms = getUniformsMap();
        if (!uniforms) return;

        try {
            // Register this node with parent component
            const metadata: NodeMetadata = {
                blendMode: props.blendMode || 'normal',
                opacity: props.opacity,
                visible: props.visible === false ? false : true,
                id: props.id,
                mask: props.maskSource ? {
                    source: props.maskSource,
                    type: props.maskType || 'alpha'
                } as MaskConfig : undefined,
                maps: mapsFromProps(),
                renderOrder: props.renderOrder ?? detectedRenderOrder,
                transform: effectiveTransform(),
                boundingBox: effectiveBoundingBox(),
                flow: props.flow,
                absolute: props.absolute
            };

            parentRegister(
                instanceId,
                componentDefinition.fragment,
                parentId,
                metadata,
                uniforms,
                componentDefinition,
                isCapturesDOM ? captureCanvasEl : undefined
            );

            isRegistered = true;
        } catch (error) {
            console.error('Error registering shader node:', error);
        }
    };

    // Register with parent when mounted (ONCE - no re-registration on metadata changes)
    onMount(() => {
        registerWithRenderer();

        // Detect DOM position for correct render ordering
        if (props.renderOrder === undefined && markerRef) {
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

    // Re-register after renderer resets (e.g. colorSpace/toneMapping change tore the
    // renderer down and re-initialized it): cleanup() clears the node registry and
    // initializeRenderer() only re-registers the root node. The signal starts at 0 so
    // the initial registration above is not duplicated by this effect.
    createEffect(on(() => shaderRendererResetSignal?.(), (signal) => {
        // `on` watches ONLY the reset signal: registerWithRenderer reads props/
        // memos, and those must not re-trigger a full re-registration (metadata
        // and uniform updates have their own dedicated effects).
        if (signal === undefined || signal === 0 || !isRegistered) return;
        // The renderer was re-initialized, possibly with a new colorSpace: drop the
        // cached uniforms map so it is rebuilt under the current color-space mode
        // (createGpuUniformsMap precomputes color stops for the active mode).
        uniformsMap = null;
        registerWithRenderer();
    }));

    // Cleanup on unmount - unregister node
    onCleanup(() => {
        isRegistered = false;
        parentRegister(instanceId, null, null, null, null);
    });

    // Update uniform values when specific props change (performance optimization)
    createEffect(() => {
        if (!isRegistered) return;

        const uniforms = getUniformsMap();
        if (!uniforms) return;

        try {
            const props_snapshot = effectiveProps();

            // Update all uniforms based on current prop values
            Object.entries(uniforms).forEach(([propName, uniformData]) => {
                if (!uniformData || typeof uniformData !== 'object') return;

                // GPU bridge entries carry `.value` (no v1 `.uniform` node). `propName in props_snapshot`
                // filters out colorStops' expanded array fields (colorsArray/…), which aren't props.
                if (uniformData.value !== undefined && propName in props_snapshot) {
                    const newValue = props_snapshot[propName];
                    // Send raw value - renderer will handle transformation
                    parentUniformUpdate(instanceId, propName, newValue);
                }
            });
        } catch (error) {
            console.error('Error updating uniforms:', error);
        }
    });

    // Update metadata when blend mode, opacity, visibility, masking, or transformations change
    createEffect(() => {
        if (!isRegistered) return;

        try {
            const metadata: NodeMetadata = {
                blendMode: props.blendMode || 'normal',
                opacity: props.opacity,
                visible: props.visible === false ? false : true,
                id: props.id,
                mask: props.maskSource ? {
                    source: props.maskSource,
                    type: props.maskType || 'alpha'
                } as MaskConfig : undefined,
                maps: mapsFromProps(),
                renderOrder: props.renderOrder ?? detectedRenderOrder,
                transform: effectiveTransform(),
                boundingBox: effectiveBoundingBox(),
                flow: props.flow,
                absolute: props.absolute
            };
            parentMetadataUpdate(instanceId, metadata);
        } catch (error) {
            console.error('Error updating metadata:', error);
        }
    });

    // Guard DOM-only APIs from SSR
    const canUseDOM = typeof document !== 'undefined' && typeof window !== 'undefined';

    // Return provider with children
    // Uses local.children which was safely split at the beginning of the component
    return (
        <ShaderContext.Provider value={childContextValue()}>
            <span ref={markerRef} style={{display: 'contents'}} data-shader-id={instanceId}>
                {/* Render children in span on server (SSR) or for non-capturesDOM components */}
                {(!isCapturesDOM || !canUseDOM) && local.children}
            </span>
            {isCapturesDOM && canUseDOM && (
                <Portal>
                    <canvas ref={captureCanvasEl} {...({'layoutsubtree': ''} as any)}
                            width={captureSize().w} height={captureSize().h}
                            style="position:fixed;inset:0;width:100vw;height:100vh;z-index:-9999">
                        {local.children}
                    </canvas>
                </Portal>
            )}
        </ShaderContext.Provider>
    );
}
