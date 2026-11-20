# Web VM debugger

Read and apply [docs/ui-principles.md](docs/ui-principles.md) and [REQUIREMENTS.md](REQUIREMENTS.md) before any UI work. The complete owner standard is mandatory.

Never show automatic notification banners, toasts or alert blocks, including API errors and successful actions. Messages must not shift or cover the workspace. Use the fixed diagnostics control; details open only on explicit user action. See the mandatory message rule in docs/ui-principles.md.

Keep this an opt-in laboratory host. Reuse MultiCamera sources, programs and endpoints. Do not add debugger branches or dependencies to the VM/ONNX core. Live contexts never escape to HTTP or encoder threads: manual inspection runs on the waiting VM task, and explicitly selected observations capture at a stable VM instruction boundary. Never read a GPU pointer from managed code. Observation reserves a bounded slot before GPU preparation/download. PNG encoding receives owned CPU data. The explicitly enabled H.264 encoder receives its own FFmpeg CUDA surfaces filled through ONNX Runtime bound outputs. Disabled observation must not prepare, copy, serialize or encode registers.

No port 443. Default address: http://127.0.0.1:5264. Remote binding requires `WebDebug:AccessToken`. Do not commit local paths, credentials or tokens. One in-flight run per camera in this debugger is deliberate; the normal host retains its configured concurrency.

## Универсальность общего кода — указание владельца 2026-11-16

В обобщённых проектах, библиотеках и компонентах запрещена привязка к конкретному прикладному проекту, его камерам, моделям, именам регистров или этапам программы. Если для изменения кажется необходимой такая специфика, сначала уточнить и согласовать её с владельцем. Нельзя молча добавлять специальные кнопки, endpoints или ветви исполнения. Просмотр и кодирование данных регистров подключать через существующий универсальный механизм выбора регистров и точки наблюдения. Возможности определять по явному контракту данных, а не по имени регистра.
