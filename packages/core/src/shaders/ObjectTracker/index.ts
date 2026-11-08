import type {GpuShaderDefinition} from "@coreroot/gpu/porters"
import {defineShader, p} from "@coreroot/std"
import {detectionOverlay} from "@coreroot/std/effects/overlay"
import {transformColor, transformBoolean} from "@coreroot/utilities/transformations"

/**
 * ObjectTracker — a computer-vision "object detection" overlay. Each pixel walks a spatial
 * partition of the canvas to the leaf cell it belongs to, scans that leaf for content (coverage
 * + a content-tight bounding box), and — if the leaf holds an object — composites a bounding box
 * + an optional label over the source. The whole fragment is the std detectionOverlay recipe:
 * analysis core (partition walk + content bbox) → detection box → glyph-pill label, over the
 * kit's detection parts and the shared overlay vocabulary.
 */

export interface ComponentProps {
    detectionMode: string
    threshold: number
    layout: string
    cellSize: number
    maxDepth: number
    boxStyle: string
    lineWidth: number
    cornerRadius: number
    strokeColor: Parameters<typeof transformColor>[0]
    fillColor: Parameters<typeof transformColor>[0]
    labelColor: Parameters<typeof transformColor>[0]
    labelBackgroundColor: Parameters<typeof transformColor>[0]
    labelMode: string
    labelPosition: string
    labelInset: boolean
    labelRadius: number
    fontFamily: string
    fontWeight: number
    fontSize: number
    letterSpacing: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "ObjectTracker",
    role: 'overlay',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Computer-vision style object detection overlay — draws bounding boxes and labels around detected regions of the content below, with grid, quadtree and mosaic layouts.",
    requiresRTT: true,
    requiresChild: true,
    props: {
        detectionMode: {
            default: "bright", compileTime: true, description: "What counts as a detectable object.",
            ui: { type: 'select', options: [
                { label: 'Transparency', value: 'alpha' }, { label: 'Brightness', value: 'bright' }, { label: 'Darkness', value: 'dark' },
                { label: 'Red', value: 'red' }, { label: 'Green', value: 'green' }, { label: 'Blue', value: 'blue' }
            ], label: 'Detect', group: 'Detection' }
        },
        threshold: {
            default: 0.25, description: "Detection cut-off.",
            ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Threshold', group: 'Detection' }
        },
        layout: {
            default: "mosaic", compileTime: true, description: "How the canvas is partitioned into detection cells.",
            ui: { type: 'select', options: [{ label: 'Grid', value: 'grid' }, { label: 'Quadtree', value: 'quadtree' }, { label: 'Mosaic', value: 'mosaic' }], label: 'Layout', group: 'Detection' }
        },
        cellSize: {
            default: 400, description: "Base cell size in pixels",
            ui: { type: 'range', min: 24, max: 800, step: 1, label: 'Base Cell Size', group: 'Detection' }
        },
        maxDepth: {
            default: 2, compileTime: true, description: "Maximum subdivision levels for Quadtree/Mosaic layouts (each level can quarter a cell).",
            ui: { type: 'range', min: 1, max: 5, step: 1, label: 'Subdivision', group: 'Detection', condition: { layout: ['quadtree', 'mosaic'] } }
        },
        boxStyle: {
            default: "corners", compileTime: true, description: "Bounding box outline style",
            ui: { type: 'select', options: [{ label: 'Full', value: 'full' }, { label: 'Corners', value: 'corners' }], label: 'Box Style', group: 'Boxes' }
        },
        lineWidth: {
            default: 1.5, description: "Thickness of the box outline in pixels.",
            ui: { type: 'range', min: 0.5, max: 12, step: 0.5, label: 'Line Width', group: 'Boxes' }
        },
        cornerRadius: {
            default: 0, description: "Corner radius of the box outline in pixels",
            ui: { type: 'range', min: 0, max: 40, step: 1, label: 'Corner Radius', group: 'Boxes', condition: { boxStyle: 'full' } }
        },
        strokeColor: {
            default: "#ffffff", transform: transformColor, description: "Bounding box outline color.",
            ui: { type: 'color', label: 'Stroke', group: 'Colors' }
        },
        fillColor: {
            default: "#5a79911a", transform: transformColor, description: "Bounding box fill color.",
            ui: { type: 'color', label: 'Fill', group: 'Colors' }
        },
        labelColor: {
            default: "#000000", transform: transformColor, description: "Label text color.",
            ui: { type: 'color', label: 'Label', group: 'Colors', condition: { labelMode: ['dimensions', 'percentage'] } }
        },
        labelBackgroundColor: {
            default: "#ffffff", transform: transformColor, description: "Label pill background color.",
            ui: { type: 'color', label: 'Label Background', group: 'Colors', condition: { labelMode: ['dimensions', 'percentage'] } }
        },
        labelMode: {
            default: "none", compileTime: true, description: "What each label shows.",
            ui: { type: 'select', options: [{ label: 'None', value: 'none' }, { label: 'Dimensions', value: 'dimensions' }, { label: 'Percentage', value: 'percentage' }], label: 'Label', group: 'Labels' }
        },
        labelPosition: {
            default: "bottom-right", compileTime: true,
            description: "Which corner of the box the label anchors to.",
            ui: { type: 'select', options: [
                { label: 'Bottom Left', value: 'bottom-left' }, { label: 'Bottom Right', value: 'bottom-right' },
                { label: 'Top Left', value: 'top-left' }, { label: 'Top Right', value: 'top-right' }
            ], label: 'Label Position', group: 'Labels', condition: { labelMode: ['dimensions', 'percentage'] } }
        },
        labelRadius: {
            default: 0, description: "Corner radius of the label pill in pixels.",
            ui: { type: 'range', min: 0, max: 30, step: 1, label: 'Border Radius', group: 'Labels', condition: { labelMode: ['dimensions', 'percentage'] } }
        },
        labelInset: {
            default: true, transform: transformBoolean, description: "Place the label inside or outside the box.",
            ui: { type: 'checkbox', label: 'Label Inset', group: 'Labels', condition: { labelMode: ['dimensions', 'percentage'] } }
        },
        fontFamily: {
            default: "Inter", description: "Google Fonts family used for label text.",
            ui: { type: 'font-family', label: 'Font', group: 'Typography', condition: { labelMode: ['dimensions', 'percentage'] } }
        },
        fontWeight: {
            default: 500, description: "Font weight for label text.",
            ui: { type: 'font-weight', label: 'Weight', group: 'Typography', condition: { labelMode: ['dimensions', 'percentage'] } }
        },
        fontSize: {
            default: 0.015, description: "Label text size as a fraction of canvas height.",
            ui: { type: 'range', min: 0.01, max: 0.5, step: 0.005, label: 'Size', group: 'Typography', dimensional: 'canvas-height', condition: { labelMode: ['dimensions', 'percentage'] } }
        },
        letterSpacing: {
            default: 0, description: "Letter spacing for label text, in em units.",
            ui: { type: 'range', min: -0.1, max: 0.5, step: 0.005, label: 'Letter Spacing', group: 'Typography', condition: { labelMode: ['dimensions', 'percentage'] } }
        }
    },

    gpu: {fragment: detectionOverlay({
        detectionMode: p('detectionMode'), threshold: p('threshold'),
        layout: p('layout'), cellSize: p('cellSize'), maxDepth: p('maxDepth'),
        boxStyle: p('boxStyle'), lineWidth: p('lineWidth'), cornerRadius: p('cornerRadius'),
        strokeColor: p('strokeColor'), fillColor: p('fillColor'),
        labelColor: p('labelColor'), labelBackgroundColor: p('labelBackgroundColor'),
        labelMode: p('labelMode'), labelPosition: p('labelPosition'),
        labelInset: p('labelInset'), labelRadius: p('labelRadius'),
        fontFamily: p('fontFamily'), fontWeight: p('fontWeight'),
        fontSize: p('fontSize'), letterSpacing: p('letterSpacing'),
    })},
})

export default componentDefinition
