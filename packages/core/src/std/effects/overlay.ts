/**
 * std/effects/overlay — HUD overlays: whole fragments that draw detection boxes or motion trackers over the layer inside.
 *
 * Both words return a fragment for a `gpu:` definition. They read the child as a texture,
 * look for content in it (bright, dark, colored or opaque areas) and draw chrome over it:
 * bounding boxes and labels for `detectionOverlay`, pursuing tracker gizmos with keyframe
 * trails for `motionTrackerHud`. Sizes are in pixels on a 1080-pixel-tall canvas and scale
 * with the canvas height. Colors are color props (`transform: transformColor`).
 */
// Maintainer notes: the HUD-drawing vocabulary shared by the overlay-role shaders (ObjectTracker,
// KeyFrames): premultiplied "over" compositing, round-rect signed distance, the corner-bracket
// stroke mask, and the premultiplied→straight output convention. The part BODIES live in
// kit/overlayParts (statement-level string builders; see the rationale there); this module is
// the std-facing surface shader definitions import. Only multi-consumer vocabulary lives here
// (the pass-4 rule: single-consumer stages stay in their shader file as named parts).
//
// The four re-exports below are raw-WGSL statement builders for maintainers writing new overlay
// recipes; they are not author-facing words.
export {overPremul, roundRectSD, cornerBracketMask, straightFromPremul} from '../../gpu/kit/overlayParts'

// ── Detection overlay recipe ────────────────────────────────────────────────────────────────
import type {Expr as ExprT, GpuFragmentParams, EmitContext} from '../../gpu/contract'
import {Expr, formatFloat, ZERO} from '../../gpu/porters'
import {blend, overlayParts} from '../../gpu/kit/index'
import {straightFromPremul} from '../../gpu/kit/overlayParts'
import type {PropRef} from '../values'

const num = (v: unknown, f: number): number => (typeof v === 'number' ? v : f)
const DETECTION_REFERENCE_HEIGHT = 1080.0

/** The props `detectionOverlay` reads. Every field is a prop ref (`p('name')`). Props marked "select" are declared `compileTime: true`. */
export interface DetectionOverlaySlots {
    /** Select: what counts as an object, `'alpha' | 'bright' | 'dark' | 'red' | 'green' | 'blue'`. */
    detectionMode: PropRef
    /** 0–1, how strong the content must be to count. */
    threshold: PropRef
    /** Select: how the canvas is cut into cells, `'grid' | 'quadtree' | 'mosaic'`. */
    layout: PropRef
    /** Base cell size in pixels (24–800). */
    cellSize: PropRef
    /** Select (integer 1–5): how many times a quadtree or mosaic cell may split in four. Ignored by `'grid'`. */
    maxDepth: PropRef
    /** Select: `'full'` box outline or `'corners'` brackets. */
    boxStyle: PropRef
    /** Outline thickness in pixels. */
    lineWidth: PropRef
    /** Outline corner radius in pixels (full boxes only). */
    cornerRadius: PropRef
    /** Box outline color. */
    strokeColor: PropRef
    /** Box fill color; use a translucent one. */
    fillColor: PropRef
    /** Label text color. */
    labelColor: PropRef
    /** Label pill color. */
    labelBackgroundColor: PropRef
    /** Select: what the label says, `'none' | 'dimensions' | 'percentage'`. */
    labelMode: PropRef
    /** Select: the box corner the label sits at, `'bottom-left' | 'bottom-right' | 'top-left' | 'top-right'`. */
    labelPosition: PropRef
    /** Boolean prop (`transform: transformBoolean`): label inside the box (on) or outside (off). */
    labelInset: PropRef
    /** Label pill corner radius in pixels. */
    labelRadius: PropRef
    /** Google Fonts family for the label (a `font-family` prop). */
    fontFamily: PropRef
    /** Font weight for the label (a `font-weight` prop). */
    fontWeight: PropRef
    /** Label text size as a fraction of canvas height (0.01–0.5). */
    fontSize: PropRef
    /** Label letter spacing in em. */
    letterSpacing: PropRef
}

/**
 * Draw computer-vision style detection boxes over whatever the child contains.
 *
 * A whole fragment for a `gpu:` definition. The canvas is cut into cells (`layout`,
 * `cellSize`, `maxDepth`); each cell that holds content of the chosen kind (`detectionMode`
 * above `threshold`) gets a box drawn tight around that content, with an optional label
 * showing its size or coverage. The child shows through underneath. Every slot is a prop
 * ref; the field list on `DetectionOverlaySlots` gives units and options.
 *
 * @example
 * ```ts
 * gpu: {fragment: detectionOverlay({detectionMode: p('detectionMode'), threshold: p('threshold'), layout: p('layout'), cellSize: p('cellSize'), maxDepth: p('maxDepth'), boxStyle: p('boxStyle'), lineWidth: p('lineWidth'), cornerRadius: p('cornerRadius'), strokeColor: p('strokeColor'), fillColor: p('fillColor'), labelColor: p('labelColor'), labelBackgroundColor: p('labelBackgroundColor'), labelMode: p('labelMode'), labelPosition: p('labelPosition'), labelInset: p('labelInset'), labelRadius: p('labelRadius'), fontFamily: p('fontFamily'), fontWeight: p('fontWeight'), fontSize: p('fontSize'), letterSpacing: p('letterSpacing')})}
 * ```
 * @tip Cost grows with `maxDepth`; 2 is plenty for most layouts. Set `labelMode` to `'none'` to skip the text pass entirely.
 * @see motionTrackerHud
 */
export function detectionOverlay(slots: DetectionOverlaySlots): (params: GpuFragmentParams) => ExprT {
    // Each pixel walks a spatial partition of the canvas to its leaf cell, scans that leaf for
    // content (coverage + a content-tight bbox), and, if the leaf holds an object, composites a
    // bounding box + an optional glyph-pill label over the source. One raw-WGSL builder over the
    // kit's detection parts: the walk is JS-unrolled over compile-time `maxDepth`, each cell scan
    // a JS-unrolled N×N textureSampleLevel loop, bbox accumulation via `select`-folds (uniform
    // control flow, AA stays valid). Compile-time branches (detectionMode / layout / maxDepth /
    // boxStyle / labelMode / labelPosition) read propValues.
    return (params) => {
        const {uniforms, ctx, childNode, convertToTexture, propValues} = params
        if (!childNode) return ZERO
        const childTex = convertToTexture(childNode)
        const texKey = childTex.key

        // ── Compile-time options ──
        const modeMap: Record<string, number> = {alpha: 0, bright: 1, dark: 2, red: 3, green: 4, blue: 5}
        const layoutMap: Record<string, number> = {grid: 0, quadtree: 1, mosaic: 2}
        const labelMap: Record<string, number> = {none: 0, dimensions: 1, percentage: 2}
        const posMap: Record<string, number> = {'bottom-left': 0, 'bottom-right': 1, 'top-left': 2, 'top-right': 3}
        const rawMode = propValues[slots.detectionMode.name]
        const modeStr = (['alpha', 'bright', 'dark', 'red', 'green', 'blue'] as const)[
            typeof rawMode === 'number' ? rawMode : (modeMap[rawMode as string] ?? 1)
        ]
        const rawLayout = propValues[slots.layout.name]
        const layoutStr = (['grid', 'quadtree', 'mosaic'] as const)[
            typeof rawLayout === 'number' ? rawLayout : (layoutMap[rawLayout as string] ?? 2)
        ]
        let D = Math.round(num(propValues[slots.maxDepth.name], 2))
        if (!Number.isFinite(D)) D = 2
        D = Math.max(0, Math.min(5, D))
        if (layoutStr === 'grid') D = 0
        const rawBox = propValues[slots.boxStyle.name]
        const cornerStyle = (typeof rawBox === 'string' ? rawBox : 'corners') === 'corners'
        const rawLabel = propValues[slots.labelMode.name]
        const labelStr = (['none', 'dimensions', 'percentage'] as const)[
            typeof rawLabel === 'number' ? rawLabel : (labelMap[rawLabel as string] ?? 0)
        ]
        const labelsOn = labelStr !== 'none'
        const rawPos = propValues[slots.labelPosition.name]
        const labelPos = typeof rawPos === 'number' ? rawPos : (posMap[rawPos as string] ?? 1)

        // ── Label glyph atlas (only when labels on) — the host recipe part ──
        const atlasKey = labelsOn
            ? overlayParts.createGlyphStripAtlas(params, {
                label: 'DetectionOverlay:atlas',
                familyProp: slots.fontFamily.name, weightProp: slots.fontWeight.name,
            }).key
            : undefined

        return new Expr((ec: EmitContext) => {
            const p = ec.freshLocal('ot')
            const s: string[] = []
            const f: overlayParts.DetectionEmitFrame = {
                p,
                U: (name: string): string => uniforms[slots[name as keyof DetectionOverlaySlots].name]._emit(ec),
                L: (n: number): string => formatFloat(n),
                s,
                sampleChild: (uvWgsl: string): string => `textureSampleLevel(tex.$.${texKey}, samp.$.linearClamp, ${uvWgsl}, 0.0)`,
                UN: ec.external(blend.unpremultiplyAlpha, 'unpremultiplyAlpha'),
                modeStr,
            }
            const {U, L} = f

            // ── Shared HUD frame: logical-pixel coords, AA width, scaled line/box metrics ──
            s.push(`let ${p}_res = ${ctx.logicalViewportSize._emit(ec)};`)
            s.push(`let ${p}_scaleFactor = ${p}_res.y / ${L(DETECTION_REFERENCE_HEIGHT)};`)
            const uvE = ctx.uv._emit(ec)
            s.push(`let ${p}_fragPx = vec2f((${uvE}).x * ${p}_res.x, (${uvE}).y * ${p}_res.y);`)
            s.push(`let ${p}_aa = max(max(fwidth(${p}_fragPx.x), fwidth(${p}_fragPx.y)), ${L(0.0001)});`)
            s.push(`let ${p}_cellPx = max(${U('cellSize')} * ${p}_scaleFactor, ${L(4)});`)
            s.push(`let ${p}_minSize = ${L(3)} * ${p}_scaleFactor;`)
            s.push(`let ${p}_lineHalf = max(${U('lineWidth')} * ${p}_scaleFactor * ${L(0.5)}, ${L(0.5)});`)
            s.push(`let ${p}_cornerRadiusPx = ${U('cornerRadius')} * ${p}_scaleFactor;`)
            s.push(`let ${p}_boxInset = ${p}_lineHalf + ${p}_scaleFactor * ${L(1.5)};`)
            s.push(`let ${p}_thr = ${U('threshold')};`)

            // ── Analysis core (atomic — see the irreducibility note on the parts) ──
            overlayParts.partitionWalkStmts(f, layoutStr, D)
            overlayParts.contentBBoxStmts(f)

            s.push(`var ${p}_dst = ${f.sampleChild(`(${uvE})`)};`)

            // ── Boxes / labels drawn by parts ──
            overlayParts.detectionBoxStmts(f, cornerStyle)
            if (labelsOn && atlasKey) overlayParts.glyphPillLabelStmts(f, {labelStr, labelPos, atlasKey})

            for (const line of s) ec.statement(line)
            // Premultiplied → straight for output.
            return straightFromPremul(`${p}_dst`)
        })
    }
}

// ── Motion-tracker HUD recipe ───────────────────────────────────────────────────────────────
import type {KitTexture} from '../../gpu/contract'
import {trackerSim} from '../../gpu/kit/index'
import {diamondStamp, pathSegmentStamp, ringTrailStmts, bracketGizmo, type HudEmitFrame} from '../../gpu/kit/overlayParts'

/** The props `motionTrackerHud` reads. Every field is a prop ref (`p('name')`). */
export interface MotionTrackerHudSlots {
    /** How many trackers to draw, 1–64. Must be the same prop the tracker simulation reads. */
    trackers: PropRef
    /** 0–1, how visible the keyframe trail behind each tracker is. */
    trail: PropRef
    /** Gizmo size in pixels (8–80). */
    markerSize: PropRef
    /** Stroke width in pixels. */
    lineWidth: PropRef
    /** Color of the tracker gizmo (brackets, crosshair, dot). */
    markerColor: PropRef
    /** Color of the keyframe diamonds along the trail. */
    keyframeColor: PropRef
    /** Color of the motion path between keyframes. */
    pathColor: PropRef
}

/**
 * Draw motion-tracker gizmos that chase content across the child, leaving keyframe trails.
 *
 * A whole fragment for a `gpu:` definition. It only draws; the trackers themselves come from
 * the tracker pursuit simulation, which the same definition must run as its `compute:` pass
 * (`trackerSim.createTrackerPursuitSim` from the kit, as the KeyFrames shader does). Without
 * that pass the child passes through untouched. Every slot is a prop ref; the field list on
 * `MotionTrackerHudSlots` gives units.
 *
 * @example
 * ```ts
 * gpu: {fragment: motionTrackerHud({trackers: p('trackers'), trail: p('trail'), markerSize: p('markerSize'), lineWidth: p('lineWidth'), markerColor: p('markerColor'), keyframeColor: p('keyframeColor'), pathColor: p('pathColor')})}
 * ```
 * @tip Copy the KeyFrames shader as a whole; the compute pass and this fragment share the `trackers` prop and the detection props.
 * @see detectionOverlay
 */
export function motionTrackerHud(slots: MotionTrackerHudSlots): (params: GpuFragmentParams) => ExprT {
    // ONE copy of the draw body in a runtime loop over the tracker count, reading the pursuit
    // sim's state rows (`trackState` compute output). Per pixel each tracker costs 2
    // cache-resident loads (head + trail AABB) unless the pixel is inside its padded box, where
    // the trail-ring stamps (keyframe diamond + path segment) and the bracket gizmo run. The AABB
    // cull is an atomic raw-WGSL core: a runtime-count loop whose draw body sits behind a
    // per-tracker early-out, which no Expr-graph fold can express inside emitted control flow.
    return (params) => {
        const {uniforms, ctx, childNode, convertToTexture, computeOutputs} = params
        if (!childNode) return ZERO
        const childTex = (computeOutputs?.childTexture as KitTexture | undefined) ?? convertToTexture(childNode)
        const texKey = childTex.key
        const stateKit = computeOutputs?.trackState as KitTexture | undefined

        return new Expr((ec: EmitContext) => {
            const p = ec.freshLocal('kft')
            const s: string[] = []
            const f: HudEmitFrame = {
                p,
                U: (name: string): string => uniforms[slots[name as keyof MotionTrackerHudSlots].name]._emit(ec),
                L: (n: number): string => formatFloat(n),
                s,
            }
            const {U, L} = f
            const uvE = ctx.uv._emit(ec)

            // ── Shared HUD frame: logical-pixel coords, AA width, scaled stroke/gizmo metrics ──
            s.push(`let ${p}_res = ${ctx.logicalViewportSize._emit(ec)};`)
            s.push(`let ${p}_sf = ${p}_res.y / ${L(DETECTION_REFERENCE_HEIGHT)};`)
            s.push(`let ${p}_px = vec2f((${uvE}).x * ${p}_res.x, (${uvE}).y * ${p}_res.y);`)
            s.push(`let ${p}_aa = max(max(fwidth(${p}_px.x), fwidth(${p}_px.y)), ${L(0.0001)});`)
            s.push(`let ${p}_lw = max(${U('lineWidth')} * ${p}_sf, ${L(0.5)});`)
            s.push(`let ${p}_ms = max(${U('markerSize')} * ${p}_sf, ${L(4)});`)
            s.push(`var ${p}_dst = textureSampleLevel(tex.$.${texKey}, samp.$.linearClamp, (${uvE}), 0.0);`)

            if (stateKit) {
                const stateKey = stateKit.key
                const load = (col: string): string => `textureLoad(tex.$.${stateKey}, vec2u(${col}, ti), 0)`

                // Runtime tracker loop with the per-tracker AABB early-out.
                s.push(`let ${p}_n = u32(clamp(${U('trackers')}, ${L(1)}, ${L(trackerSim.TRACKER_MAX)}));`)
                s.push(`for (var ti = 0u; ti < ${p}_n; ti = ti + 1u) {`)
                s.push(`let head = ${load('0u')};`)
                s.push(`if (head.w < 0.5) { continue; }`)
                // Per-pixel early-out on the kernel-published trail AABB, padded by gizmo reach.
                s.push(`let bb = ${load(`${trackerSim.TRACKER_AABB_COL}u`)};`)
                s.push(`let pad = ${p}_ms + ${p}_lw * ${L(4)} + ${p}_aa * ${L(2)};`)
                s.push(`let bbMin = vec2f(bb.x * ${p}_res.x, bb.y * ${p}_res.y) - vec2f(pad);`)
                s.push(`let bbMax = vec2f(bb.z * ${p}_res.x, bb.w * ${p}_res.y) + vec2f(pad);`)
                s.push(`if (all(${p}_px >= bbMin) && all(${p}_px <= bbMax)) {`)
                // NOTE: raw-WGSL locals must dodge WGSL reserved words (`meta`, `half`, …).
                s.push(`let mta = ${load('1u')};`)
                s.push(`let tp = vec2f(head.x * ${p}_res.x, head.y * ${p}_res.y);`)
                s.push(`let searching = select(${L(1)}, ${L(0.45)}, mta.x > ${L(4)});`)

                // ── trail ring frame → diamond + path-segment stamps → bracket gizmo ──
                ringTrailStmts(f, {
                    len: trackerSim.TRACKER_TRAIL_LEN, every: trackerSim.TRACKER_TRAIL_EVERY,
                    load, stages: [diamondStamp, pathSegmentStamp(trackerSim.TRACKER_TRAIL_EVERY)],
                })
                bracketGizmo(f)

                s.push(`}`) // AABB early-out
                s.push(`}`) // for ti
            }

            for (const line of s) ec.statement(line)
            // Premultiplied → straight for output.
            return straightFromPremul(`${p}_dst`)
        })
    }
}
