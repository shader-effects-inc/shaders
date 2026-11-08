import {defineShader, p} from "@coreroot/std"
import {chosenBy, cmykPress, dotScreen, inkPlate} from "@coreroot/std/paint/patterns"
import {transformAngle, transformColor} from "@coreroot/utilities/transformations"

// Style selector: 0 = simple (single-plate dot pattern), 1 = cmyk. Any non-'cmyk' → 0.
const transformStyle = (value: string): number => (value === 'cmyk' ? 1 : 0)

// Compile-time reader mirroring the transform — robust to a raw string (a preset loaded before
// the transform ran) as well as the bridge-mapped number.
const styleOf = (raw: unknown): 'classic' | 'cmyk' => (raw === 'cmyk' || raw === 1 ? 'cmyk' : 'classic')

export interface ComponentProps{
    style: string
    frequency: number
    angle: Parameters<typeof transformAngle>[0]
    cyanAngle: Parameters<typeof transformAngle>[0]
    magentaAngle: Parameters<typeof transformAngle>[0]
    yellowAngle: Parameters<typeof transformAngle>[0]
    blackAngle: Parameters<typeof transformAngle>[0]
    misprint: number
    misprintAngle: Parameters<typeof transformAngle>[0]
    paperColor: Parameters<typeof transformColor>[0]
    cyanColor: Parameters<typeof transformColor>[0]
    magentaColor: Parameters<typeof transformColor>[0]
    yellowColor: Parameters<typeof transformColor>[0]
    blackColor: Parameters<typeof transformColor>[0]
}

// std gather filter: samples the child's render-to-texture per plate offset.
export const componentDefinition = defineShader<ComponentProps>({
    name: "Halftone",
    role: 'filter',
    species: 'gather',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Stylize",
    description: "Halftone dot pattern effect for printing aesthetics",
    // Stylization: composite the print over the child so transparent output reveals it.
    // blendWithChildren is false → behavior is "replace": transparent output reveals the
    // background, not the child. Flip to true to reveal the child through transparent output.
    blendWithChildren: false,
    props: {
        style: {
            default: "classic",
            transform: transformStyle,
            compileTime: true,
            description: "Halftone rendering style",
            ui: {
                type: 'select',
                options: [
                    { label: 'Simple', value: 'classic' },
                    { label: 'CMYK Print', value: 'cmyk' }
                ],
                label: 'Style',
                group: 'Effect'
            }
        },
        frequency: {
            default: 100,
            description: "Frequency of the halftone dots",
            ui: { type: ['range', 'map'], min: 10, max: 300, step: 1, label: 'Frequency', group: 'Effect' }
        },
        angle: {
            default: 45,
            transform: transformAngle,
            description: "Rotation angle of the pattern (in degrees)",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Angle', group: 'Effect', condition: { style: 'classic' } }
        },
        cyanAngle: {
            default: 15,
            transform: transformAngle,
            description: "Screen angle for the cyan plate (in degrees)",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Cyan Angle', group: 'Plate Angles', condition: { style: 'cmyk' } }
        },
        magentaAngle: {
            default: 75,
            transform: transformAngle,
            description: "Screen angle for the magenta plate (in degrees)",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Magenta Angle', group: 'Plate Angles', condition: { style: 'cmyk' } }
        },
        yellowAngle: {
            default: 0,
            transform: transformAngle,
            description: "Screen angle for the yellow plate (in degrees)",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Yellow Angle', group: 'Plate Angles', condition: { style: 'cmyk' } }
        },
        blackAngle: {
            default: 45,
            transform: transformAngle,
            description: "Screen angle for the black plate (in degrees)",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Black Angle', group: 'Plate Angles', condition: { style: 'cmyk' } }
        },
        misprint: {
            default: 0,
            description: "Simulated mis-registration between plates. Plates are offset around the misprint angle, producing color fringing at the edges of inked regions.",
            ui: { type: ['range', 'map'], min: 0, max: 0.01, step: 0.0005, label: 'Misprint', group: 'Effect', condition: { style: 'cmyk' } }
        },
        misprintAngle: {
            default: 0,
            transform: transformAngle,
            description: "Direction the plates drift apart. Rotating this rotates the color-fringing pattern.",
            ui: { type: ['range', 'map'], min: 0, max: 360, step: 1, label: 'Misprint Angle', group: 'Effect', condition: { style: 'cmyk' } }
        },
        paperColor: {
            default: "#ffffff",
            transform: transformColor,
            description: "Paper/substrate color shown where no ink lands",
            ui: { type: 'color', label: 'Paper', group: 'Inks', condition: { style: 'cmyk' } }
        },
        cyanColor: {
            default: "#00ffff",
            transform: transformColor,
            description: "Cyan ink color",
            ui: { type: 'color', label: 'Cyan', group: 'Inks', condition: { style: 'cmyk' } }
        },
        magentaColor: {
            default: "#ff00ff",
            transform: transformColor,
            description: "Magenta ink color",
            ui: { type: 'color', label: 'Magenta', group: 'Inks', condition: { style: 'cmyk' } }
        },
        yellowColor: {
            default: "#ffff00",
            transform: transformColor,
            description: "Yellow ink color",
            ui: { type: 'color', label: 'Yellow', group: 'Inks', condition: { style: 'cmyk' } }
        },
        blackColor: {
            default: "#000000",
            transform: transformColor,
            description: "Black (key) ink color",
            ui: { type: 'color', label: 'Black', group: 'Inks', condition: { style: 'cmyk' } }
        }
    },

    // The recipe, branched at compile time on `style`: classic modulates a single rotated dot
    // plate by the child's brightness; cmyk lays four subtractive ink plates down in order, each
    // registered a quarter turn further around `misprintAngle`.
    effect: chosenBy(p('style'), styleOf, {
        classic: dotScreen({angle: p('angle'), frequency: p('frequency')}),
        cmyk: cmykPress({
            paper: p('paperColor'),
            frequency: p('frequency'),
            misprint: p('misprint'),
            misprintAngle: p('misprintAngle'),
            plates: [
                inkPlate({channel: 'cyan', screenAngle: p('cyanAngle'), ink: p('cyanColor')}),
                inkPlate({channel: 'magenta', screenAngle: p('magentaAngle'), ink: p('magentaColor')}),
                inkPlate({channel: 'yellow', screenAngle: p('yellowAngle'), ink: p('yellowColor')}),
                inkPlate({channel: 'black', screenAngle: p('blackAngle'), ink: p('blackColor')}),
            ],
        }),
    }),
})

export default componentDefinition
