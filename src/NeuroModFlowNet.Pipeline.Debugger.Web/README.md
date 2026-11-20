# VM debugger workspace

Embeddable browser workspace for a single VM target at a time. It contains instruction navigation, breakpoint and step controls, register inspection, and dockable observation panels. The containing application supplies the target selector, navigation, lifecycle of the underlying pipeline, and transport.

## Embed

Reference this Razor Class Library. Load `debugger/icons/tabler-icons.css`, `debugger/vendor/dockview/dockview.css`, `debugger/workspace.css` and `debugger/panels.css` from the application's static assets. The panels support the application's inherited `--lu-*` theme tokens and default to the cream palette.

```js
import { mountDebuggerWorkspace } from './debugger/debugger-workspace.js';

const workspace = await mountDebuggerWorkspace(root, { session, onError, onRefresh });
await workspace.selectTarget(targetId);
workspace.update(snapshot);
```

Give `root` a bounded height or place it in a flex column with available height. Call `update(snapshot)` with the selected target's debug snapshot, or with `undefined` when unavailable. `selectTarget` clears the previous target's instructions and register state. Call `dispose()` on unmount to stop polling and release the layout and previews.

The optional callbacks control translation (`t`), persistence (`read`, `save`), active page and shortcuts, diagnostics, help, and opening a detached viewer. The host can bind its own controls to `showPanel`, `resetLayout` and `openViewer`.

## Session adapter

Each asynchronous method receives an opaque target ID and an abort signal as the last argument. The workspace calls methods on this object and consumes values directly. The adapter owns endpoints, authorization and serialization.

| Method | Value or request |
| --- | --- |
| `getInstructions(id, signal)` | Ordered instructions with program-local indices and flow metadata |
| `sendCommand(id, command, signal)` | Pause, continue, step, next frame, run-to or detach. Commands that depend on a paused position carry its revision. |
| `setBreakpoints(id, breakpoints, signal)` | Program and index pairs |
| `pollRegisterCatalog(id, signal)` | Observed register metadata with discovery order and capture points |
| `getObservation(id, signal)` | Current observation configuration, generation and samples |
| `setObservation(id, configuration, signal)` | New configuration, returning the updated observation snapshot |
| `pollObservation(id, acknowledgement, signal)` | Sample batches after the acknowledged sequence numbers |
| `inspect(id, inspection, signal)` | A PNG `Blob` or typed value as JSON text for the paused revision |
| `getResultImage(id, signal)` | A PNG `Blob` for the current or next result frame |

See the [WebDebug adapter](../../labs/NeuroModFlowNet.Pipeline.WebDebug/wwwroot/debugger-session.js) and [server contracts](../../labs/NeuroModFlowNet.Pipeline.WebDebug/Debugging/DebugContracts.cs) for the reference host.
