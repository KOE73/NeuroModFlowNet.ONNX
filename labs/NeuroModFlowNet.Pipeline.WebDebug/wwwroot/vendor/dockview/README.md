# Dockview Core 8.2.0

Local redistribution of `dockview-core` 8.2.0 (MIT, see `LICENSE.md`).
Source package: the same version installed in the owner's SeMaps editor.

- `dockview.mjs`: official `dist/package/main.esm.min.mjs`, unchanged.
- `dockview.css`: stylesheet extracted unchanged from the official UMD build's
  `style.textContent` payload. Loaded as a separate local file to preserve the
  debugger's Content Security Policy without permitting injected inline styles.
- Application colors and layout adjustments live outside this directory, in
  `workspace.css` and `theme-debug-extensions.css`.

No runtime network dependencies or package installation are required.
