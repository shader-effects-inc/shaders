import { createContext, useContext, onMount, onCleanup, createMemo, createEffect, createSignal, splitProps, createUniqueId, type JSX } from 'solid-js';
import { shaderRendererGPU, rootPassthrough, debugWarn } from 'shaders-core';
import type { GpuFailureReason } from 'shaders-core';
import { isExternalUser, startTelemetry } from 'shaders-core/telemetry';
import { setColorSpaceMode } from 'shaders-core/utilities/transformations';

declare const __SHADERS_VERSION__: string;

// Define the context value type
export interface ShaderContextValue {
    shaderParentId: string;
    shaderNodeRegister: (id: string, fragmentNodeFunc: any, parentId: string | null, metadata: any, uniforms: any, componentDefinition?: any, domCanvas?: HTMLCanvasElement) => void;
    shaderUniformUpdate: (nodeId: string, uniformName: string, value: any) => void;
    shaderMetadataUpdate: (nodeId: string, metadata: any) => void;
    shaderColorSpace: () => 'p3-linear' | 'srgb';
    shaderRendererResetSignal: () => number;
}

export const ShaderContext = createContext<ShaderContextValue>();

interface ShaderProps {
    children?: JSX.Element;
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
    style?: JSX.CSSProperties;
    class?: string;
    [key: string]: any;
}

/**
 * Root Shader component that initializes the renderer and provides context to children
 */
export default function Shader(allProps: ShaderProps) {
    // Split component-specific props from DOM-safe props
    const [props, domProps] = splitProps(allProps, ['children', 'disableTelemetry', 'colorSpace', 'toneMapping', 'isPreview', 'onReady', 'onUnavailable', 'style', 'class']);
    // Refs for DOM elements
    let containerRef: HTMLDivElement | undefined;
    let canvasRef: HTMLCanvasElement | undefined;

    // Unique ID for this root component (SSR-safe)
    const rootId = 'shader-root-' + createUniqueId();

    // Get renderer instance
    const rendererInstance = shaderRendererGPU();

    // Wire up onReady callback
    rendererInstance.setOnReady(() => props.onReady?.());

    // Terminal GPU failure (no WebGPU, no adapter, device refused/lost, GPU unusable). The
    // renderer has already released everything — a canvas that never drew is transparent — so
    // we latch it here so none of the re-entry points below (visibility observer, colorSpace /
    // toneMapping effects) try again, and hand it to the host for a static fallback.
    let gpuUnavailable: GpuFailureReason | null = null;
    rendererInstance.setOnUnavailable((reason: GpuFailureReason) => {
        gpuUnavailable = reason;
        try {
            props.onUnavailable?.(reason);
        } catch (err) {
            debugWarn('[Shaders] onUnavailable callback threw:', err);
        }
    });

    // Telemetry collector reference for cleanup
    let telemetryCollector: any = null;
    let telemetryStartTimeout: number | null = null;
    let shouldSendTelemetry: boolean | null = null;

    // Visibility tracking
    let wasVisible = false;
    let visibilityObserver: IntersectionObserver | null = null;

    // Initialization tracking
    let isInitialized = false;
    let isInitializing = false;

    // Bumped after every renderer (re-)initialization so mounted child components can
    // re-register: cleanup() clears the node registry and initializeRenderer() only
    // re-registers the root node (mirrors the Vue adapter's `shaderRendererResetSignal`).
    const [rendererResetSignal, setRendererResetSignal] = createSignal(0);

    // Define stable callback functions
    const nodeRegister = (id: string, fragmentNodeFunc: any, parentId: string | null, metadata: any, uniforms: any = null, componentDefinition: any = null, domCanvas?: HTMLCanvasElement) => {
        // Handle node removal (cleanup)
        if (fragmentNodeFunc === null) {
            try {
                rendererInstance.removeNode(id);
            } catch (err) {
                console.warn("Error removing node:", err);
            }
            return;
        }

        try {
            // Handle normal node registration
            rendererInstance.registerNode(id, fragmentNodeFunc, parentId, metadata, uniforms, componentDefinition, domCanvas);
        } catch (err) {
            console.error("Error registering node:", err, {id, parentId, metadata});
        }
    };

    const uniformUpdate = (nodeId: string, uniformName: string, value: any) => {
        try {
            rendererInstance.updateUniformValue(nodeId, uniformName, value);
        } catch (err) {
            console.warn("Error updating uniform:", err);
        }
    };

    const metadataUpdate = (nodeId: string, metadata: any) => {
        try {
            rendererInstance.updateNodeMetadata(nodeId, metadata);
        } catch (err) {
            console.warn("Error updating metadata:", err);
        }
    };

    // Create a stable context value
    // Note: shaderColorSpace and shaderRendererResetSignal are passed as accessors
    // (not invoked), so this memo does not re-track the signal or props.colorSpace —
    // children always read the current values when they call the accessors.
    const contextValue = createMemo<ShaderContextValue>(() => ({
        shaderParentId: rootId,
        shaderNodeRegister: nodeRegister,
        shaderUniformUpdate: uniformUpdate,
        shaderMetadataUpdate: metadataUpdate,
        shaderColorSpace: () => props.colorSpace || 'p3-linear',
        shaderRendererResetSignal: rendererResetSignal
    }));

    // Wait for active rendering before starting telemetry collection.
    // Bounded: on a browser that can't run WebGPU fps never leaves 0, and an unbounded 500ms
    // recursion would tick for the life of the page against a renderer that will never draw.
    const MAX_TELEMETRY_POLLS = 40; // 40 × 500ms = 20s
    const startTelemetryWhenReady = () => {
        let polls = 0;
        const checkRendering = () => {
            if (gpuUnavailable || ++polls > MAX_TELEMETRY_POLLS) {
                telemetryStartTimeout = null;
                return;
            }
            const stats = rendererInstance.getPerformanceStats();
            if (stats.fps > 0) {
                const version = typeof __SHADERS_VERSION__ !== 'undefined' ? __SHADERS_VERSION__ : 'unknown';
                telemetryCollector = startTelemetry(
                    rendererInstance,
                    version,
                    props.disableTelemetry,
                    props.isPreview
                );
                if (telemetryCollector) {
                    telemetryCollector.start();
                }
                telemetryStartTimeout = null;
            } else {
                telemetryStartTimeout = window.setTimeout(checkRendering, 500);
            }
        };

        telemetryStartTimeout = window.setTimeout(checkRendering, 500);
    };

    // Function to initialize renderer with visibility check.
    //
    // Never rejects. `initialize()` resolves even when WebGPU is unavailable (it reports via
    // setOnUnavailable and leaves the canvas transparent), and the remaining steps are guarded
    // so a failure here can never surface as an unhandled promise rejection.
    const initializeRenderer = async () => {
        const canvas = canvasRef;
        if (!canvas || isInitialized || isInitializing || gpuUnavailable) {
            return;
        }

        isInitializing = true;

        try {
            // Check if renderer is already initialized to avoid double initialization
            if (!rendererInstance.isInitialized()) {
                await rendererInstance.initialize({
                    canvas: canvas,
                    colorSpace: props.colorSpace || 'p3-linear',
                    toneMapping: props.toneMapping || 'linear',
                });
            }

            // The renderer reports unavailability asynchronously (via a microtask), so
            // re-check synchronously here too — nothing below is worth doing against a
            // renderer that has already given up.
            if (rendererInstance.getFailureReason()) {
                isInitializing = false;
                return;
            }

            // Register the root node. The GPU renderer requires a component definition on every
            // node — rootPassthrough supplies the shared "composite children / transparent when
            // empty" root (replacing the v1 bare vec4 fn).
            rendererInstance.registerNode(
                rootId,
                rootPassthrough.fragment,
                null, // No parent (this is the root)
                null, // No metadata to pass
                {},
                rootPassthrough
            );

            isInitialized = true;
            isInitializing = false;

            // cleanup() clears child nodes; mounted layers must register again after initialization.
            setRendererResetSignal((signal) => signal + 1);

            // Compute sampling decision once (includes random sampling roll)
            if (shouldSendTelemetry === null) {
                shouldSendTelemetry = isExternalUser()
            }

            if (shouldSendTelemetry && !telemetryCollector) {
                startTelemetryWhenReady();
            }

        } catch (err) {
            debugWarn('[Shaders] renderer initialization failed:', err);
            isInitializing = false;
        }
    };

    // Setup visibility observer for container hide/show detection
    const setupVisibilityObserver = () => {
        const container = containerRef;
        if (!container || visibilityObserver) return;

        visibilityObserver = new IntersectionObserver((entries) => {
            const entry = entries[0];
            if (!entry) return;

            const rect = container.getBoundingClientRect();
            const isCurrentlyVisible = entry.isIntersecting && rect && rect.width > 0 && rect.height > 0;

            if (isCurrentlyVisible && !wasVisible) {
                // Canvas became visible - resume animation
                if (rendererInstance.isInitialized()) {
                    rendererInstance.startAnimation();
                    // Start telemetry if conditions are met and not already started
                    if (shouldSendTelemetry && !telemetryCollector && !telemetryStartTimeout) {
                        startTelemetryWhenReady();
                    }
                } else {
                    // First time visible, need to initialize
                    void initializeRenderer();
                }
                wasVisible = true;
            } else if (!isCurrentlyVisible && wasVisible) {
                // Canvas became hidden - pause animation but keep renderer alive
                rendererInstance.stopAnimation();
                wasVisible = false;
            }
        }, { threshold: 0 });

        visibilityObserver.observe(container);
    };

    // Initialize renderer on mount - with visibility awareness
    onMount(async () => {
        const container = containerRef;
        if (!container) {
            debugWarn('[Shaders] container ref is null in Shader onMount');
            return;
        }

        // Check if container is visible on mount
        const rect = container.getBoundingClientRect();
        const isVisible = rect.width > 0 && rect.height > 0;

        if (isVisible) {
            // Container is visible, initialize immediately
            await initializeRenderer();
            wasVisible = true;
        } else {
            // Container is hidden, set up observer for when it becomes visible
            wasVisible = false;
        }

        // Always set up observer to handle show/hide cycles
        setupVisibilityObserver();
    });

    // Cleanup on component destroy
    onCleanup(() => {
        // Stop telemetry collection if active
        if (telemetryCollector) {
            telemetryCollector.stop();
            telemetryCollector = null;
        }

        // Clear telemetry start timeout if pending
        if (telemetryStartTimeout !== null) {
            clearTimeout(telemetryStartTimeout);
            telemetryStartTimeout = null;
        }

        // Clean up visibility observer
        if (visibilityObserver) {
            visibilityObserver.disconnect();
            visibilityObserver = null;
        }

        // Only cleanup if renderer was initialized
        if (rendererInstance && isInitialized) {
            try {
                rendererInstance.cleanup();
            } catch (err) {
                console.warn('[Shader] Error during cleanup:', err);
            }
        }
    });

    // Watch for colorSpace changes and update the renderer
    // Skip the initial execution since colorSpace is set during initialization
    let isFirstEffect = true;
    createEffect(() => {
        const currentColorSpace = props.colorSpace || 'p3-linear';

        if (isFirstEffect) {
            isFirstEffect = false;
            return;
        }

        if (rendererInstance.isInitialized()) {
            // Update the global color space mode
            setColorSpaceMode(currentColorSpace);

            // Force re-initialization to apply color space changes
            rendererInstance.cleanup();
            isInitialized = false;
            void initializeRenderer();
        }
    });

    // Watch for toneMapping changes — re-initialize to apply the new mode
    let isFirstToneMappingEffect = true;
    createEffect(() => {
        void (props.toneMapping || 'linear'); // access for reactivity

        if (isFirstToneMappingEffect) {
            isFirstToneMappingEffect = false;
            return;
        }

        if (rendererInstance.isInitialized()) {
            rendererInstance.cleanup();
            isInitialized = false;
            void initializeRenderer();
        }
    });

    return (
        <ShaderContext.Provider value={contextValue()}>
            <div
                ref={containerRef}
                class={'shader' + (props.class ? ' ' + props.class : '')}
                style={props.style}
                {...domProps}
            >
                <canvas
                    data-renderer="shaders"
                    ref={canvasRef}
                    style={{width: '100%', height: '100%', display: 'block'}}
                />
                {props.children}
            </div>
        </ShaderContext.Provider>
    );
}

// Export the context and hook for use in child components
export function useShaderContext() {
    const ctx = useContext(ShaderContext);
    if (!ctx) {
        throw new Error('Shader components must be used inside a <Shader> component');
    }
    return ctx;
}
