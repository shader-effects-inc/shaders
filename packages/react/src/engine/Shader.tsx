import React, {useRef, useEffect, useId, createContext, useMemo, useState} from 'react';
import {shaderRendererGPU, rootPassthrough, debugWarn} from 'shaders-core';
import type {GpuFailureReason} from 'shaders-core';
import {isExternalUser, startTelemetry} from 'shaders-core/telemetry';
import {setColorSpaceMode} from 'shaders-core/utilities/transformations';

declare const __SHADERS_VERSION__: string;

// Define the context value type
export interface ShaderContextValue {
    shaderParentId: string;
    shaderNodeRegister: (id: string, fragmentNodeFunc: any, parentId: string | null, metadata: any, uniforms: any, componentDefinition?: any, domCanvas?: HTMLCanvasElement) => void;
    shaderUniformUpdate: (nodeId: string, uniformName: string, value: any) => void;
    shaderMetadataUpdate: (nodeId: string, metadata: any) => void;
    shaderColorSpace: 'p3-linear' | 'srgb';
    shaderRendererResetSignal: number;
}

export const ShaderContext = createContext<ShaderContextValue | null>(null);

interface ShaderProps {
    children?: React.ReactNode;
    disableTelemetry?: boolean;
    colorSpace?: 'p3-linear' | 'srgb';
    toneMapping?: 'linear' | 'reinhard' | 'cineon' | 'aces' | 'agx' | 'neutral' | 'hable' | 'unreal';
    isPreview?: boolean;
    onReady?: () => void;
    /**
     * This browser/GPU cannot run the shader, so the canvas will stay transparent for good.
     * Fires at most once, and nothing is written to the console — render your own static
     * fallback (a gradient, an image) from this callback if you want one.
     */
    onUnavailable?: (reason: GpuFailureReason) => void;
    style?: React.CSSProperties;
    className?: string;

    [key: string]: any;
}

/**
 * Root Shader component that initializes the renderer and provides context to children
 */
export const Shader: React.FC<ShaderProps> = ({
                                                children,
                                                disableTelemetry = false,
                                                colorSpace = 'p3-linear',
                                                toneMapping = 'linear',
                                                isPreview = false,
                                                onReady,
                                                onUnavailable,
                                                style = {},
                                                className = '',
                                                ...rest
                                            }) => {
    // Use refs for the container and canvas elements
    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Unique ID for this root component (SSR-safe)
    const reactId = useId();
    const rootId = useMemo(() => 'shader-root-' + reactId.replace(/[^a-zA-Z0-9_]/g, ''), [reactId]);

    // Get renderer instance
    const rendererRef = useRef<any>(null);
    if (rendererRef.current === null) {
        rendererRef.current = shaderRendererGPU();
    }

    // Keep the callback refs fresh every render so we never invoke a stale closure
    const onReadyRef = useRef(onReady);
    onReadyRef.current = onReady;
    const onUnavailableRef = useRef(onUnavailable);
    onUnavailableRef.current = onUnavailable;

    // Terminal GPU failure (no WebGPU, no adapter, device refused/lost, GPU unusable). The
    // renderer has already released everything — a canvas that never drew is transparent —
    // so we latch it here so none of the re-entry points below (visibility observer,
    // colorSpace/toneMapping effects) try again.
    const gpuUnavailableRef = useRef<GpuFailureReason | null>(null);

    // Register the onReady bridge in an effect (not during render) to avoid
    // mutating renderer state as a side-effect of rendering.
    useEffect(() => {
        rendererRef.current.setOnReady(() => onReadyRef.current?.());
        return () => {
            rendererRef.current.setOnReady(null);
        };
    }, []);

    // Telemetry collector reference for cleanup
    const telemetryCollectorRef = useRef<any>(null);
    const telemetryStartTimeoutRef = useRef<number | null>(null);
    const shouldSendTelemetryRef = useRef<boolean | null>(null);

    // Define stable callback functions
    const nodeRegister = useMemo(() => {
        return (id: string, fragmentNodeFunc: any, parentId: string | null, metadata: any, uniforms: any = null, componentDefinition: any = null, domCanvas?: HTMLCanvasElement) => {
            // Handle node removal (cleanup)
            if (fragmentNodeFunc === null) {
                try {
                    rendererRef.current.removeNode(id);
                } catch (err) {
                    console.warn("Error removing node:", err);
                }
                return;
            }

            try {
                // Handle normal node registration
                rendererRef.current.registerNode(id, fragmentNodeFunc, parentId, metadata, uniforms, componentDefinition, domCanvas);
            } catch (err) {
                console.error("Error registering node:", err, {id, parentId, metadata});
            }
        };
    }, []);

    const uniformUpdate = useMemo(() => {
        return (nodeId: string, uniformName: string, value: any) => {
            try {
                rendererRef.current.updateUniformValue(nodeId, uniformName, value);
            } catch (err) {
                console.warn("Error updating uniform:", err);
            }
        };
    }, []);

    const metadataUpdate = useMemo(() => {
        return (nodeId: string, metadata: any) => {
            try {
                rendererRef.current.updateNodeMetadata(nodeId, metadata);
            } catch (err) {
                console.warn("Error updating metadata:", err);
            }
        };
    }, []);

    const [rendererResetSignal, setRendererResetSignal] = useState(0);

    // Create a stable context value
    const contextValue = useMemo(() => {
        return {
            shaderParentId: rootId,
            shaderNodeRegister: nodeRegister,
            shaderUniformUpdate: uniformUpdate,
            shaderMetadataUpdate: metadataUpdate,
            shaderColorSpace: colorSpace,
            shaderRendererResetSignal: rendererResetSignal
        };
    }, [rootId, nodeRegister, uniformUpdate, metadataUpdate, colorSpace, rendererResetSignal]);

    // Track whether the component has been initialized (Strict Mode protection)
    const isInitializedRef = useRef(false);
    const isInitializingRef = useRef(false);
    const isCleanedUpRef = useRef(false);
    const wasVisibleRef = useRef(false);
    const visibilityObserverRef = useRef<IntersectionObserver | null>(null);

    // Wait for active rendering before starting telemetry collection.
    // Bounded: on a browser that can't run WebGPU fps never leaves 0, and an unbounded
    // 500ms recursion would tick for the life of the page against a renderer that will
    // never draw.
    const MAX_TELEMETRY_POLLS = 40; // 40 × 500ms = 20s
    const startTelemetryWhenReady = () => {
        let polls = 0;
        const checkRendering = () => {
            // Early-return if component has been unmounted to prevent scheduling timeouts after cleanup
            if (isCleanedUpRef.current || !rendererRef.current || gpuUnavailableRef.current || ++polls > MAX_TELEMETRY_POLLS) {
                telemetryStartTimeoutRef.current = null;
                return;
            }

            const stats = rendererRef.current.getPerformanceStats();
            if (stats.fps > 0) {
                const version = typeof __SHADERS_VERSION__ !== 'undefined' ? __SHADERS_VERSION__ : 'unknown';
                telemetryCollectorRef.current = startTelemetry(
                    rendererRef.current,
                    version,
                    disableTelemetry,
                    isPreview
                );
                if (telemetryCollectorRef.current) {
                    telemetryCollectorRef.current.start();
                }
                telemetryStartTimeoutRef.current = null;
            } else {
                telemetryStartTimeoutRef.current = window.setTimeout(checkRendering, 500);
            }
        };

        telemetryStartTimeoutRef.current = window.setTimeout(checkRendering, 500);
    };

    // Function to initialize renderer with visibility check.
    //
    // Never rejects. `initialize()` resolves even when WebGPU is unavailable (it reports via
    // setOnUnavailable and leaves the canvas transparent), and the remaining steps are
    // guarded so a failure here can never surface as an unhandled promise rejection.
    const initializeRenderer = async () => {
        const canvas = canvasRef.current;
        if (!canvas || isInitializedRef.current || isInitializingRef.current || gpuUnavailableRef.current) {
            return;
        }

        isInitializingRef.current = true;

        try {
            const renderer = rendererRef.current;

            // Subscribe before initialize so a start-up failure is caught on the first attempt.
            renderer.setOnUnavailable((reason: GpuFailureReason) => {
                gpuUnavailableRef.current = reason;
                try {
                    onUnavailableRef.current?.(reason);
                } catch (err) {
                    debugWarn('[Shaders] onUnavailable callback threw:', err);
                }
            });

            // Check if renderer is already initialized to avoid double initialization
            if (!renderer.isInitialized()) {
                await renderer.initialize({
                    canvas: canvas,
                    colorSpace: colorSpace,
                    toneMapping: toneMapping,
                });
            }

            // The renderer reports unavailability asynchronously (via a microtask), so
            // re-check synchronously here too — nothing below is worth doing against a
            // renderer that has already given up.
            if (renderer.getFailureReason()) {
                isInitializingRef.current = false;
                return;
            }

            // Register the root node. The GPU renderer requires a component definition on every
            // node — rootPassthrough supplies the shared "composite children / transparent when
            // empty" root (replacing the v1 bare vec4 fn).
            renderer.registerNode(
                rootId,
                rootPassthrough.fragment,
                null, // No parent (this is the root)
                null, // No metadata to pass
                {},
                rootPassthrough
            );

            isInitializedRef.current = true;
            isInitializingRef.current = false;
            // cleanup() clears child nodes; mounted layers must register again after initialization.
            setRendererResetSignal(signal => signal + 1);

            // Compute sampling decision once (includes random sampling roll)
            if (shouldSendTelemetryRef.current === null) {
                shouldSendTelemetryRef.current = isExternalUser()
            }

            if (shouldSendTelemetryRef.current && !telemetryCollectorRef.current) {
                startTelemetryWhenReady();
            }

        } catch (err) {
            debugWarn('[Shaders] renderer initialization failed:', err);
            isInitializingRef.current = false;
        }
    };

    // Setup visibility observer for container hide/show detection
    const setupVisibilityObserver = () => {
        const container = containerRef.current;
        if (!container || visibilityObserverRef.current) return;

        visibilityObserverRef.current = new IntersectionObserver((entries) => {
            const entry = entries[0];
            if (!entry) return;

            const rect = container.getBoundingClientRect();
            const isCurrentlyVisible = entry.isIntersecting && rect && rect.width > 0 && rect.height > 0;
            
            if (isCurrentlyVisible && !wasVisibleRef.current) {
                // Canvas became visible - resume animation
                if (rendererRef.current.isInitialized()) {
                    rendererRef.current.startAnimation();
                    // Start telemetry if conditions are met and not already started
                    if (shouldSendTelemetryRef.current && !telemetryCollectorRef.current && !telemetryStartTimeoutRef.current) {
                        startTelemetryWhenReady();
                    }
                } else {
                    // First time visible, need to initialize
                    void initializeRenderer();
                }
                wasVisibleRef.current = true;
            } else if (!isCurrentlyVisible && wasVisibleRef.current) {
                // Canvas became hidden - pause animation but keep renderer alive
                rendererRef.current.stopAnimation();
                wasVisibleRef.current = false;
            }
        }, { threshold: 0 });

        visibilityObserverRef.current.observe(container);
    };

    // Initialize renderer on mount - with visibility awareness
    useEffect(() => {
        // Reset cleanup flag on mount to handle StrictMode re-mounting
        isCleanedUpRef.current = false;

        const container = containerRef.current;
        if (!container) {
            return;
        }

        // Check if container is visible on mount
        const rect = container.getBoundingClientRect();
        const isVisible = rect.width > 0 && rect.height > 0;

        if (isVisible) {
            // Container is visible, initialize immediately
            void initializeRenderer();
            wasVisibleRef.current = true;
        } else {
            // Container is hidden, set up observer for when it becomes visible
            wasVisibleRef.current = false;
        }

        // Always set up observer to handle show/hide cycles
        setupVisibilityObserver();

        // Cleanup on unmount
        return () => {
            // Prevent double cleanup in React Strict Mode
            if (isCleanedUpRef.current) {
                return;
            }
            isCleanedUpRef.current = true;

            // Stop telemetry collection if active
            if (telemetryCollectorRef.current) {
                telemetryCollectorRef.current.stop();
                telemetryCollectorRef.current = null;
            }

            // Clear telemetry start timeout if pending
            if (telemetryStartTimeoutRef.current !== null) {
                clearTimeout(telemetryStartTimeoutRef.current);
                telemetryStartTimeoutRef.current = null;
            }

            // Clean up visibility observer
            if (visibilityObserverRef.current) {
                visibilityObserverRef.current.disconnect();
                visibilityObserverRef.current = null;
            }

            // Only cleanup if renderer was initialized
            if (rendererRef.current && isInitializedRef.current) {
                try {
                    rendererRef.current.cleanup();
                } catch (err) {
                    console.warn("[Shader] Error during cleanup:", err);
                }
            }

            // Reset state flags
            isInitializedRef.current = false;
            isInitializingRef.current = false;
        };
    }, []); // Empty deps - only run on mount/unmount. Note: disableTelemetry is intentionally evaluated only on mount.

    // Watch for colorSpace changes and update the renderer
    // Skip the initial execution since colorSpace is set during initialization
    const isFirstRenderRef = useRef(true);
    useEffect(() => {
        if (isFirstRenderRef.current) {
            isFirstRenderRef.current = false;
            return;
        }

        if (rendererRef.current && isInitializedRef.current) {
            // Update the global color space mode
            setColorSpaceMode(colorSpace);

            // Force re-registration by cleaning up and re-initializing
            try {
                rendererRef.current.cleanup();
                isInitializedRef.current = false;
                void initializeRenderer();
            } catch (err) {
                console.error("[Shader] Error updating colorSpace:", err);
            }
        }
    }, [colorSpace]);

    // Watch for toneMapping changes — re-initialize to apply the new mode
    const isFirstToneMappingRef = useRef(true);
    useEffect(() => {
        if (isFirstToneMappingRef.current) {
            isFirstToneMappingRef.current = false;
            return;
        }

        if (rendererRef.current && isInitializedRef.current) {
            try {
                rendererRef.current.cleanup();
                isInitializedRef.current = false;
                void initializeRenderer();
            } catch (err) {
                console.error("[Shader] Error updating toneMapping:", err);
            }
        }
    }, [toneMapping]);

    return (
        <ShaderContext.Provider value={contextValue}>
            <div
                ref={containerRef}
                className={'shader' + (className ? ' ' + className : '')}
                style={style}
                {...rest}
            >
                <canvas
                    data-renderer="shaders"
                    ref={canvasRef}
                    style={{width: '100%', height: '100%', display: 'block'}}
                />
                {children}
            </div>
        </ShaderContext.Provider>
    );
};

export default Shader;