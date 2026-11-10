::shader-preview{component="HTMLInCanvas"}
<div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; font-family: system-ui, -apple-system, sans-serif; color: white;">
  <h1 style="font-size: 48px; font-weight: 800; margin: 0;">Your HTML content here</h1>
</div>
::

## Usage

Put the HTML you want as a layer inside `<HTMLInCanvas>`:

::code-group{:tabs='["Vue", "React", "Svelte", "Solid", "JS"]'}
```vue-html
<Shader>
  <HTMLInCanvas>
    <div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; font-family: system-ui, sans-serif; color: white;">
      <h1>Your HTML content here</h1>
    </div>
  </HTMLInCanvas>
</Shader>
```

```jsx
<Shader>
  <HTMLInCanvas>
    <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'system-ui, sans-serif', color: 'white' }}>
      <h1>Your HTML content here</h1>
    </div>
  </HTMLInCanvas>
</Shader>
```

```tsx
<Shader>
  <HTMLInCanvas>
    <div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; font-family: system-ui, sans-serif; color: white;">
      <h1>Your HTML content here</h1>
    </div>
  </HTMLInCanvas>
</Shader>
```

```tsx
<Shader>
  <HTMLInCanvas>
    <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'system-ui, sans-serif', color: 'white' }}>
      <h1>Your HTML content here</h1>
    </div>
  </HTMLInCanvas>
</Shader>
```

```javascript
// HTMLInCanvas requires DOM children and is not supported via the JS API.
// Use the Vue, React, Svelte, or Solid packages instead.
```
::

Effects placed after `<HTMLInCanvas>` apply to the HTML like any other layer.

## Layout and resolution

The children lay out inside the canvas, at the canvas's CSS size. A child with `width: 100%; height: 100%` fills the shader, and `100%` means the canvas, not the viewport. They inherit fonts and colors from the page, as normal HTML does.

The children sit in a size container that matches the canvas, so container units measure the shader. `font-size: 10cqmin` keeps a headline proportional at any shader size.

The HTML is redrawn into the layer at the canvas's pixel resolution, so text stays sharp on high-density screens. It updates whenever its rendering changes: framework updates, CSS animations and transitions, hover states, the text caret, images finishing loading.

## Interactivity

The HTML stays interactive where it appears. Links, buttons, inputs and text selection work, and screen readers see the content in place.

Clicks go to the HTML's undistorted position. If a later effect moves pixels (a warp or a ripple), the visible content and the clickable areas drift apart. To make the layer purely visual, add `pointer-events: none` to your HTML, or `inert` to also remove it from the tab order.

## Browser support

`<HTMLInCanvas>` uses the html-in-canvas API, which shipped in Chrome 157. Firefox, Safari and older Chromium browsers don't have it. There, the HTML is shown as plain HTML on top of the shader, at the same size and position. It stays readable and clickable, but effects don't apply to it. The rest of the shader renders as usual.

## Limitations

To keep page pixels private, the browser leaves some content out of the layer:

- **Cross-origin content**: images, iframes, CSS `url()` backgrounds and SVG references from another origin. Serve assets from your own origin to keep them in the layer.
- **Browser state**: visited link colors, spelling and grammar markers, autofill previews, system colors, and video captions and controls.

A child with its own `drawable` attribute is drawn separately, so it's missing from this layer.
