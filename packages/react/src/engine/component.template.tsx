import React, { useContext, useEffect, useId, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
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

// @ts-ignore - correct path at build time
import { ShaderContext } from '../engine/Shader';

/**
 * Base props that all shader components have
 */
interface BaseShaderProps {
    children?: React.ReactNode;
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
    ref?: React.Ref<any>;
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

/**
 * Merges the default props and any explicit user props together into one object.
 */
function computeEffectiveProps(
    props: ShaderComponentProps,
    defaultProps: Record<string, any>
): Record<string, any> {
    let baseProps = { ...defaultProps };

    for (const [key, value] of Object.entries(props)) {
        if (key !== 'children' && key !== 'ref' && value !== undefined && !isPropDriver(value)) {
            baseProps[key] = value;
        }
    }

    return baseProps;
}

// Stable style reference so React skips style diffing via reference equality
const MARKER_STYLE = { display: 'contents' } as const;

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
 * The main React wrapper component for Shader shader nodes
 */
export const ShaderComponent: React.FC<ShaderComponentProps> = (props) => {
    // Get Shader context from parent
    const context = useContext(ShaderContext);
    if (!context) {
        throw new Error('Shader components must be used inside an <Shader> component or another shader component');
    }

    const {
        shaderParentId: parentId,
        shaderNodeRegister: parentRegister,
        shaderUniformUpdate: parentUniformUpdate,
        shaderMetadataUpdate: parentMetadataUpdate,
        shaderColorSpace
    } = context;

    // Create instance ID from name or generate a deterministic one (SSR-safe)
    const reactId = useId();
    const instanceId = useMemo(() => {
        return (props.id ? props.id.replace(/[^a-zA-Z0-9_]/g, '_') : null) || reactId.replace(/[^a-zA-Z0-9_]/g, '');
    }, [props.id, reactId]);

    // Compute final props, merging defaults with user props
    const effectiveProps = useMemo(() => {
        return computeEffectiveProps(props, defaultProps);
    }, [props]);

    // Compute effective transform, merging user-provided values with defaults
    const effectiveTransform = useMemo<TransformConfig>(() => ({
        ...DEFAULT_TRANSFORM,
        ...props.transform
    }), [props.transform]);

    // Compute effective bounding box — partial configs merge with full-frame defaults,
    // absent stays undefined so the bbox render path is not activated
    const effectiveBoundingBox = useMemo<BoundingBoxConfig | undefined>(
        () => resolveBoundingBox(props.boundingBox),
        [props.boundingBox]
    );

    // DOM marker ref for determining render order from template position
    const markerRef = useRef<HTMLSpanElement | null>(null);

    // capturesDOM — canvas layoutsubtree portal for HTMLInCanvas-style shaders
    const isCapturesDOM = !!(componentDefinition as any).capturesDOM;
    const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const [domMounted, setDomMounted] = useState(false);
    const [captureSize, setCaptureSize] = useState(() => ({
        w: typeof window !== 'undefined' ? Math.round(window.innerWidth * Math.min(window.devicePixelRatio, 2)) : 0,
        h: typeof window !== 'undefined' ? Math.round(window.innerHeight * Math.min(window.devicePixelRatio, 2)) : 0
    }));

    useEffect(() => {
        if (!isCapturesDOM) return;
        setDomMounted(true);
        const onResize = () => {
            const d = Math.min(window.devicePixelRatio, 2);
            setCaptureSize({ w: Math.round(window.innerWidth * d), h: Math.round(window.innerHeight * d) });
        };
        window.addEventListener('resize', onResize);
        return () => window.removeEventListener('resize', onResize);
    }, [isCapturesDOM]);

    // Stores the DOM-detected render order
    const detectedOrderRef = useRef<number | undefined>(undefined);

    // Create a ref to store the uniform map
    const uniformsRef = useRef<GpuUniformsMap | null>(null);

    // Initialize uniforms map once
    // Set the global color space mode before creating uniforms so colors are transformed correctly
    if (uniformsRef.current === null) {
        setColorSpaceMode(shaderColorSpace);
        uniformsRef.current = createGpuUniformsMap(componentDefinition, effectiveProps, instanceId);
    }

    // Create context for children with this component as their parent
    const childContextValue = useMemo(() => {
        return {
            ...context,
            shaderParentId: instanceId
        };
    }, [context, instanceId]);

    // Register with parent when mounted (ONCE - no re-registration on metadata changes)
    useEffect(() => {
        if (!uniformsRef.current) return;

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
                maps: mapsFromProps,
                renderOrder: props.renderOrder ?? detectedOrderRef.current,
                transform: effectiveTransform,
                boundingBox: effectiveBoundingBox,
                flow: props.flow,
                absolute: props.absolute
            };

            parentRegister(
                instanceId,
                componentDefinition.fragment,
                parentId,
                metadata,
                uniformsRef.current,
                componentDefinition,
                isCapturesDOM ? (captureCanvasRef.current ?? undefined) : undefined
            );

            // Cleanup on unmount - unregister node
            return () => {
                parentRegister(instanceId, null, null, null, null);
            };
        } catch (error) {
            console.error('Error registering shader node:', error);
            return () => {};
        }
    }, [instanceId, parentId, parentRegister, domMounted]); // domMounted ensures captureCanvasRef is populated before registering for capturesDOM

    // On mount, detect this component's position among shader siblings in the DOM
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

    // Track previous prop values to detect changes
    const prevPropsRef = useRef<Record<string, any>>({});

    // Collect PropDriver values from shader props into the maps metadata structure.
    // Must scan props directly (not effectiveProps) since effectiveProps filters PropDrivers out.
    const mapsFromProps = useMemo(() => {
        const maps: Record<string, PropDriver> = {};
        for (const [key, value] of Object.entries(props)) {
            if (key !== 'children' && key !== 'ref' && isPropDriver(value)) {
                maps[key] = value as PropDriver;
            }
        }
        return Object.keys(maps).length > 0 ? maps : undefined;
    }, [props]);

    // Update uniform values only when specific props change (performance optimization)
    useEffect(() => {
        if (!uniformsRef.current) return;

        try {
            // Only update uniforms for props that actually changed
            Object.entries(uniformsRef.current).forEach(([propName, uniformData]) => {
                if (!uniformData || typeof uniformData !== 'object') return;

                // GPU bridge entries carry `.value` (no v1 `.uniform` node). `propName in effectiveProps`
                // filters out colorStops' expanded array fields (colorsArray/…), which aren't props.
                if (uniformData.value !== undefined && propName in effectiveProps) {
                    const newValue = effectiveProps[propName];
                    // Skip PropDriver values - they are handled via metadata.maps, not uniforms
                    if (isPropDriver(newValue)) return;
                    const prevValue = prevPropsRef.current[propName];

                    // Only update if value actually changed
                    if (newValue !== prevValue) {
                        // Send raw value - renderer will handle transformation
                        parentUniformUpdate(instanceId, propName, newValue);
                        // Track this value
                        prevPropsRef.current[propName] = newValue;
                    }
                }
            });
        } catch (error) {
            console.error('Error updating uniforms:', error);
        }
    }, [effectiveProps, instanceId, parentUniformUpdate]);

    // Update metadata when blend mode, opacity, visibility, masking, transformations, or prop maps change
    useEffect(() => {
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
                maps: mapsFromProps,
                renderOrder: props.renderOrder ?? detectedOrderRef.current,
                transform: effectiveTransform,
                boundingBox: effectiveBoundingBox,
                flow: props.flow,
                absolute: props.absolute
            };
            parentMetadataUpdate(instanceId, metadata);
        } catch (error) {
            console.error('Error updating metadata:', error);
        }
    }, [props.blendMode, props.opacity, props.visible, props.maskSource, props.maskType, props.renderOrder, props.id, mapsFromProps,
        effectiveTransform, effectiveBoundingBox, props.flow, props.absolute, instanceId, parentMetadataUpdate]);

    // Handle React 18 compatibility with refs
    if (props.ref && typeof props.ref === 'function') {
        try {
            props.ref(null); // We don't have a DOM element to expose
        } catch (e) {
            // Silently ignore ref errors
        }
    }

    // Return provider with children
    const captureCanvas = isCapturesDOM && domMounted ? createPortal(
        <canvas ref={captureCanvasRef} {...({'layoutsubtree': ''} as any)}
                width={captureSize.w} height={captureSize.h}
                style={{position: 'fixed', inset: 0, width: '100vw', height: '100vh', zIndex: -9999} as React.CSSProperties}>
            {props.children}
        </canvas>,
        document.body
    ) : null;

    return (
        <ShaderContext.Provider value={childContextValue}>
            <span ref={markerRef} style={MARKER_STYLE} data-shader-id={instanceId}>
                {!isCapturesDOM && props.children}
            </span>
            {captureCanvas}
        </ShaderContext.Provider>
    );
};

export default ShaderComponent;
