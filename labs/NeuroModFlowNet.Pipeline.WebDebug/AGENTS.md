# Web VM debugger

Read and apply [docs/ui-principles.md](docs/ui-principles.md) and [REQUIREMENTS.md](REQUIREMENTS.md) before any UI work. The complete owner standard is mandatory.

Never show automatic notification banners, toasts or alert blocks, including API errors and successful actions. Messages must not shift or cover the workspace. Use the fixed diagnostics control; details open only on explicit user action. See the mandatory message rule in docs/ui-principles.md.

Keep this an opt-in laboratory host. Reuse MultiCamera sources, programs and endpoints. Do not add debugger branches or dependencies to the VM/ONNX core. Paused contexts never escape to HTTP threads: register inspection runs on the waiting VM task. Never read a GPU pointer from managed code. GPU download is explicit and requested only while paused.

No port 443. Default address: http://127.0.0.1:5264. Remote binding requires `WebDebug:AccessToken`. Do not commit local paths, credentials or tokens. One in-flight run per camera in this debugger is deliberate; the normal host retains its configured concurrency.
