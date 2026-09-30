# Media

Pixels that come from an image, a video or the user's camera instead of from math. Each word
loads its source, fits it into the layer the way CSS `object-fit` fits an image into a box,
and returns the GPU half of a definition. Spread it into `defineShader({...})` next to your
props, with `role: 'media'` and `species: 'custom'`.

Every media word takes a `fit` prop (`'cover'`, `'contain'`, `'fill'` or `'scale-down'`,
marked `compileTime`) and `decode: 'srgb-linear-alpha-cut'`, which reads the source as sRGB
and makes the letterbox around a contained image transparent. Set `naturalSizeKey` on the
definition so layout can measure the media. Each word takes an `onError` callback so you can
phrase failures in your own words.

## Reach for it when

| When | Use |
|---|---|
| a photo or logo from a URL | `imageMedia` |
| a looping video from a URL | `videoMedia` |
| the user's camera, with a selfie mirror | `webcamMedia` |

## Order

- imageMedia
- videoMedia
- webcamMedia

## Example

```ts
import {defineShader, p, media} from 'shaders/std'

// An image from a URL, fitted like CSS object-fit. The fit prop is a plain string.
export const Photo = defineShader({
  name: 'Photo',
  role: 'media',
  species: 'custom',
  naturalSizeKey: {fromProp: 'url'},
  props: {
    url: {default: 'https://shaders.com/sample.jpg', ui: {type: 'image-upload', label: 'Image'}},
    objectFit: {
      default: 'cover',
      compileTime: true,
      ui: {type: 'select', options: [{label: 'Cover', value: 'cover'}, {label: 'Contain', value: 'contain'}, {label: 'Fill', value: 'fill'}, {label: 'Scale down', value: 'scale-down'}]},
    },
  },
  ...media.imageMedia({
    src: p('url'),
    fit: p('objectFit'),
    decode: 'srgb-linear-alpha-cut',
    label: 'Photo',
    onError: (error, url) => console.error(`Photo: could not load ${url}`, error),
  }),
})
```
