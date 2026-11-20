const camera = document.querySelector('#camera');
const status = document.querySelector('#status');
const sampleButton = document.querySelector('#sample');
const continuous = document.querySelector('#continuous');
const registers = document.querySelector('#observed-registers');
const registerPanels = new Map();

function showRegisters(values) {
  for (const [name,content] of registerPanels) {
    if (Object.hasOwn(values,name)) continue;
    content.parentElement.remove();
    registerPanels.delete(name);
  }
  for (const [name,value] of Object.entries(values)) {
    let content = registerPanels.get(name);
    if (!content) {
      const pane = document.createElement('section');
      pane.className = 'workspace-pane panel';
      const title = document.createElement('h2');
      title.className = 'panel-header';
      title.textContent = name;
      content = document.createElement('pre');
      pane.append(title, content);
      registers.append(pane);
      registerPanels.set(name,content);
    }
    content.textContent = JSON.stringify(value, null, 2);
  }
}
let active = false;

async function capture() {
  if (active || !camera.value || document.hidden) return;
  active = true;
  sampleButton.disabled = true;
  const source = camera.value;
  try {
    const response = await fetch(`/api/observation/sources/${encodeURIComponent(source)}/next`);
    if (source !== camera.value) return;
    if (response.status === 204) { status.textContent = 'Нет нового кадра'; return; }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const frame = await response.json();
    if (source !== camera.value) return;
    status.textContent = `RunId ${frame.runId} · кадр ${frame.frameNumber}`;
    showRegisters(frame.registers);
  } catch (error) {
    status.textContent = `Связь со сборщиком: ${error.message}`;
  } finally {
    active = false;
    sampleButton.disabled = false;
  }
}

camera.addEventListener('change', () => {
  showRegisters({});
  status.textContent = '';
});
sampleButton.addEventListener('click', capture);
setInterval(() => { if (continuous.checked) void capture(); }, 250);

try {
  const response = await fetch('/api/observation/sources');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const { cameras } = await response.json();
  for (const source of cameras) camera.add(new Option(source, source));
  const requested = new URLSearchParams(location.search).get('source');
  if (cameras.includes(requested)) camera.value = requested;
  sampleButton.disabled = continuous.disabled = cameras.length === 0;
} catch (error) {
  status.textContent = `Связь со сборщиком: ${error.message}`;
}
