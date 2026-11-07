import {readFileSync} from 'node:fs'
import {resolve} from 'node:path'
import ts from 'typescript'
import {describe, expect, it} from 'vitest'
import {generatePresetCodeSource} from '../../../scripts/generate-components'

// Generate the utility exactly as the build does, including real rounding and bounding-box
// helpers. This also works before CI builds the framework packages.
const template = readFileSync(resolve(__dirname, '../../../../react/src/utils/generatePresetCode.template.ts'), 'utf8')
const generated = generatePresetCodeSource(template, '', {})
const exports: Record<string, any> = {}
new Function('exports',
    ts.transpileModule(generated, {compilerOptions: {module: ts.ModuleKind.CommonJS}}).outputText
)(exports)

function roundTrip(components: unknown[]) {
    const code = exports.generatePresetCode({components})
    const result = ts.transpileModule(code, {
        fileName: 'preset.tsx', reportDiagnostics: true,
        compilerOptions: {jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS}
    })
    expect(result.diagnostics?.filter(d => d.category === ts.DiagnosticCategory.Error), code).toEqual([])
    const output: Record<string, any> = {}
    const jsx = (type: unknown, props: unknown) => ({type, props})
    const require = (name: string) => name === 'react/jsx-runtime'
        ? {jsx, jsxs: jsx}
        : new Proxy({}, {get: (_, key) => key})
    new Function('exports', 'require', result.outputText)(output, require)
    return output.default().props.children
}

describe('React preset string serialization', () => {
    it.each(['He said "hello" and <bye/>', 'Line one\nLine two\t\\path', 'Line one\rLine two', 'Line one\r\nLine two', '&quot; &amp;', '{"a":"b"}', '{}', '[1,2]', 'plain text', '\t\\path'])('preserves text %j', text => {
        expect(roundTrip([{type: 'Text', props: {text}}]).props.text).toBe(text)
    })

    it.each(['{"a": "b"}', '{ "radius": 0.123456789, "value": 1.230000e-7 }'])('preserves JSON text spacing and number formatting in nested components: %j', text => {
        const components = [{type: 'Group', children: [{type: 'Text', props: {text}}]}]
        expect(roundTrip(components).props.children.props.text).toBe(text)
    })

    it.each(['{"name":"layer"}', '{ "name": "layer", "value": 0.123456789 }'])('preserves JSON-looking references %j', id => {
        const children = roundTrip([
            {type: 'Text', id, props: {text: 'source'}},
            {type: 'Text', props: {maskSource: id, opacity: {type: 'map', source: id, min: 0, max: 1}}},
            {type: 'DisplacementMap', props: {source: id}}
        ])
        expect(children[0].props.id).toBe(id)
        expect(children[1].props.maskSource).toBe(id)
        expect(children[1].props.opacity.source).toBe(id)
        expect(children[2].props.source).toBe(id)
    })

    it.each(['shape', 'shape3d'])('still converts and rounds %s configuration and numeric props', key => {
        const props = {[key]: '{"type":"circleSDF","radius":0.123456789}', rotation: 12.3456789}
        const child = roundTrip([{type: key === 'shape' ? 'Glass' : 'Form3D', props}])
        expect(child.props[key]).toEqual({type: 'circleSDF', radius: 0.1235})
        expect(child.props.rotation).toBe(12.3457)
        expect(props[key]).toBe('{"type":"circleSDF","radius":0.123456789}')
    })

    it('keeps simple string props and referenced ids as quoted JSX attributes', () => {
        const components = [
            {type: 'Text', id: 'source-layer', props: {text: 'plain text', color: '#ff0000'}},
            {type: 'Text', props: {maskSource: 'source-layer'}}
        ]
        const code = exports.generatePresetCode({components})
        expect(code).toContain('color="#ff0000"')
        expect(code).toContain('text="plain text"')
        expect(code).toContain('id="source-layer"')
        expect(code).toContain('maskSource="source-layer"')
        const children = roundTrip(components)
        expect(children[0].props.color).toBe('#ff0000')
        expect(children[0].props.id).toBe('source-layer')
        expect(children[1].props.maskSource).toBe('source-layer')
    })

    it.each(['say "hello"', 'A&B', 'line\nbreak', 'line\rbreak'])('uses JS string expressions for special characters in %j', value => {
        const components = [
            {type: 'Text', id: value, props: {text: value}},
            {type: 'Text', props: {maskSource: value}}
        ]
        const code = exports.generatePresetCode({components})
        for (const key of ['text', 'id', 'maskSource']) {
            expect(code).toContain(`${key}={${JSON.stringify(value)}}`)
        }
        const children = roundTrip(components)
        expect(children[0].props.text).toBe(value)
        expect(children[0].props.id).toBe(value)
        expect(children[1].props.maskSource).toBe(value)
    })

    it('keeps referenced ids and string driver values intact', () => {
        const id = 'layer "one" &amp;'
        const children = roundTrip([
            {type: 'Text', id, props: {text: 'source'}},
            {type: 'Text', props: {maskSource: id, opacity: {type: 'map', source: id, min: 0, max: 1}}}
        ])
        expect(children[0].props.id).toBe(id)
        expect(children[1].props.maskSource).toBe(id)
        expect(children[1].props.opacity.source).toBe(id)
    })

    it.each(['{}', '{"radius":0.3}', '{"type":"circleSDF","radius":0.3}', '{"type":"circleSDF","options":{"label":"say \\"hi\\" &amp;","stops":[0,1]}}'])('converts JSON configuration %j to a readable object literal', shape => {
        const components = [{type: 'Glass', props: {shape}}]
        expect(roundTrip(components).props.shape).toEqual(JSON.parse(shape))
        const code = exports.generatePresetCode({components})
        expect(code).toContain('shape={{')
        expect(code).not.toContain('shape={"')
        if (shape.includes('circleSDF')) expect(code).toContain('type: "circleSDF"')
    })

    it('preserves malformed JSON configuration as a string', () => {
        const shape = '{not "JSON"}'
        expect(roundTrip([{type: 'Glass', props: {shape}}]).props.shape).toBe(shape)
    })

    it('leaves real objects and arrays as objects', () => {
        const shape = {type: 'circleSDF', radius: 0.3}
        const stops = [{color: '#fff', position: 0}, {color: '#000', position: 1}]
        const child = roundTrip([{type: 'Glass', props: {shape, colorStops: stops, transform: {edges: 'wrap', rotation: 30}}}])
        expect(child.props.shape).toEqual(shape)
        expect(child.props.colorStops).toEqual(stops)
        expect(child.props.transform).toEqual({edges: 'wrap', rotation: 30})
    })

    it.each([{edges: 'say "hi" &amp;'}, {edges: 'say "hi" &amp;', rotation: 30}])('preserves transform string values in %j', transform => {
        expect(roundTrip([{type: 'Text', props: {transform}}]).props.transform).toEqual(transform)
    })
})
