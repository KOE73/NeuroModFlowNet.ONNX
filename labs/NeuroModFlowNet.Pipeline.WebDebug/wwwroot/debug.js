import './shell/ui-tooltip.js';
import { ObservationView } from './observation.js';
import { DebugWorkspace } from './workspace.js';
import { memoryKind,sortRegisters } from './register-table.js';
import { BatchView,batchWords } from './batches.js';

const $ = selector => document.querySelector(selector);
const shell = $('#shell');
const storage = 'nmfn.webdebug.ui.';
const parameters = new URLSearchParams(location.search);
const viewerId = parameters.get('viewer') || 'main';
const viewerMode = viewerId !== 'main';
let workspace = null;
let registerSort = {column:'key',direction:1};
const read = (key, fallback) => { try { return localStorage.getItem(storage + key) ?? fallback; } catch { return fallback; } };
const save = (key, value) => { try { localStorage.setItem(storage + key, value); } catch { /* Preferences are optional. */ } };
let viewerBootstrap = null;
try { viewerBootstrap = JSON.parse(read(`viewer.${viewerId}`,'null')); registerSort = JSON.parse(read('registerSort','null')) ?? registerSort; } catch { }
if(!['key','memory','shape'].includes(registerSort.column) || ![1,-1].includes(registerSort.direction)) registerSort = {column:'key',direction:1};
shell.classList.toggle('viewer-mode',viewerMode);
if(viewerMode) { window.opener = null; $('#viewer-camera').hidden = false; }
let language = read('language', 'ru');
let theme = read('theme', 'cream');
let density = read('density', 'nano');
let panelScheme = read('panelScheme','compact');
if(!['compact','rounded','square'].includes(panelScheme)) panelScheme = 'compact';
let token = sessionStorage.getItem('nmfn.webdebug.token') || '';
let selectedCamera = '', selectedRegister = '', selectedInstruction = null, current = null;
let catalog = [], lastRender = '', busy = false, polling = false, imageUrl = null, pinnedUrl = null;
let selectingCamera = '';
let imageCaption = '', modalReturnFocus = null;
let imageInfo = null, pinnedInfo = null;
let lastError = '';
let focusedActionKey = '';
const words = {
  ...batchWords,
  sections:['Разделы','Sections'], overview:['Обзор камер','Camera overview'], settings:['Настройки','Settings'],
  heading:['Виртуальные машины · камеры','Virtual machines · cameras'], start:['Запустить','Start'],stop:['Остановить всё','Stop all'],
  cameras:['Камеры','Cameras'],camera:['Камера','Camera'],state:['Состояние','State'],decoder:['Декодер','Decoder'],frames:['Кадры','Frames'],runms:['Прогон, мс','Run, ms'],tracks:['Треки','Tracks'],failures:['Ошибки','Errors'],text:['Распознанный текст','Recognized text'],
  pause:['Пауза','Pause'],continue:['Продолжить','Continue'],step:['Шаг','Step'],nextframe:['Следующий кадр','Next frame'],detach:['Отсоединить','Detach'],
  debugcontrols:['Управление отладкой выбранной камеры','Selected camera debug controls'],
  panels:['Панели','Panels'],resetlayout:['Сбросить расположение','Reset layout'],newviewer:['Открыть окно наблюдения','Open observation window'],
  floatpanel:['Плавающая панель / вернуть в раскладку','Float panel / dock back'],returngallery:['Вернуть в обзор','Return to overview'],
  closepanel:['Закрыть панель','Close panel'],
  panelscheme:['Границы панелей','Panel framing'],panelcompact:['Слитно','Joined'],panelrounded:['Скруглённые','Rounded'],panelsquare:['Прямые','Square'],
  layoutinvalid:['Не удалось восстановить расположение','Could not restore layout'],popupblocked:['Браузер не открыл окно. Разрешите всплывающие окна для этого адреса.','The browser did not open the window. Allow popups for this site.'],
  instructions:['Программы и инструкции','Programs and instructions'],instruction:['Инструкция','Instruction'],breakpoint:['Т.О.','B.P.'],ms:['мс','ms'],registers:['Регистры','Registers'],register:['Регистр','Register'],memory:['Память','Memory'],shape:['Форма','Shape'],
  inspect:['Значение','Value'],preview:['Изображение','Image'],image:['Просмотр данных','Data inspection'],pin:['Закрепить вход','Pin input'],resultframe:['Кадр результата','Result frame'],
  imageempty:['На паузе выберите изображение или значение значком. Для динамики отметьте регистры галочками и продолжите VM.','On pause, use an icon to inspect an image or value. For live observation, check registers and resume the VM.'],
  menu:['Свернуть / развернуть меню','Collapse / expand navigation'],help:['Справка об отладке камер','Camera debugger help'],close:['Закрыть окно','Close dialog'],
  pausetip:['Остановиться перед следующей инструкцией выбранной камеры. Другие камеры продолжают работу.','Stop before the next instruction of this camera. Other cameras continue.'],
  continuetip:['Продолжить до точки останова или запроса паузы. F8.','Continue until a breakpoint or pause request. F8.'],
  steptip:['Выполнить ровно одну инструкцию и остановиться после неё. F6.','Execute exactly one instruction and stop after it. F6.'],
  nextframetip:['Закончить текущий кадр и остановиться перед первой инструкцией следующего. Точки останова сохраняются.','Finish this frame and stop before the first instruction of the next. Breakpoints remain active.'],
  detachtip:['Снять паузу и отключить остановки этой камеры. Повторная пауза подключит отладчик.','Release the pause and disable stops for this camera. Pause attaches the debugger again.'],
  previewtip:['Явно уменьшить и выгрузить выбранный GPU-тензор в PNG на остановленной VM. Это добавляет работу только во время просмотра.','Explicitly resize and download the selected GPU tensor to PNG on the paused VM. This adds work only during inspection.'],
  paused:['Пауза','Paused'],pausing:['Запрос паузы','Pause requested'],running:['Работает','Running'],starting:['Запуск','Starting'],stopped:['Остановлено','Stopped'],waiting:['Прогрев','Warmup'],failed:['Ошибка','Failed'],detached:['Без отладки','Detached'],
  before:['до','before'],after:['после','after'],warmup:['прогрев','warmup'],empty:['Нет запущенных камер','No cameras running'],noRegisters:['Регистры доступны на паузе','Registers are available while paused'],
  color:['Цветовая схема','Color scheme'],cream:['Кремовая','Cream'],dark:['Тёмная','Dark'],emerald:['Изумрудная','Emerald'],light:['Светлая','Light'],density:['Плотность','Density'],nano:['Ультракомпактная','Ultra compact'],mini:['Компактная','Compact'],norm:['Обычная','Normal'],tooltips:['Подсказки','Tooltips'],showtips:['Показывать подсказки','Show tooltips'],
  connection:['Подключение к API','API connection'],connected:['API подключён','API connected'],disconnected:['API недоступен','API unavailable'],unauthorized:['Нужен токен доступа. Откройте настройки.','Access token required. Open settings.'],
  accesstoken:['Токен доступа к удалённому отладчику','Remote debugger access token'],apply:['Применить','Apply'],tokenhint:['Токен хранится только в этой вкладке браузера. Адрес API совпадает с адресом страницы.','The token is stored only in this browser tab. The API uses the page address.'],
  diagnostics:['Диагностика','Diagnostics'],nodiagnostics:['Диагностика: сообщений нет','Diagnostics: no messages'],hasdiagnostics:['Есть сообщение. Открыть диагностику','A message is available. Open diagnostics'],
  tensorvalue:['Тип и форма тензора показаны в таблице. Для просмотра картинки используйте «Изображение».','Tensor type and shape are shown in the table. Use Image to inspect its pixels.'],
  currentinstruction:['Текущая инструкция','Current instruction'],runtoshort:['До','To'],runto:['Выполнить до','Run to'],
  runtotip:['Продолжить выбранную камеру и остановиться перед этой инструкцией. Постоянные точки останова сохраняются.','Continue this camera and stop before this instruction. Persistent breakpoints remain active.'],
  valueaction:['Показать значение:','Show value:'],imageaction:['Показать изображение:','Show image:'],
  resizeinstructions:['Ширина панелей инструкций и регистров','Instructions and registers panel widths'],resizeregisters:['Ширина панелей регистров и просмотра','Registers and inspection panel widths'],
  resizetip:['Потяните разделитель мышью. Стрелки ← → меняют ширину с клавиатуры. Двойной щелчок сбрасывает ширины панелей.','Drag to resize. Use ← → to resize with the keyboard. Double-click to reset panel widths.'],
  watch:['Наблюдать','Watch'],watchregister:['Наблюдать регистр:','Watch register:'],watchoff:['Наблюдение выключено','Observation disabled'],
  watchfps:['Максимальная частота наблюдения на регистр','Maximum observation frequency per register'],watchsize:['Максимальная сторона изображения, пикселей','Maximum image edge, pixels'],
  watchfpstip:['Максимум снимков в секунду на каждый регистр. Медленный браузер или кодировщик снижает фактическую частоту.','Maximum samples per second for each register. A slow browser or encoder reduces the actual rate.'],
  watchsizetip:['Уменьшение изображения выполняется до выгрузки с GPU. Размер задаёт максимальную сторону PNG.','Images are resized before GPU download. This limits the longest PNG edge.'],
  watchtip:['Выберите до 8 регистров и продолжите VM. Изображения и JSON снимаются после записи регистра; исходные входы — перед первой инструкцией. Снимки ограничены частотой и готовностью браузера. Без подключения браузера наблюдение засыпает через 5 секунд.','Select up to 8 registers and resume the VM. Images and JSON are captured after the register is written; source inputs are captured before the first instruction. Sampling is limited by frequency and browser readiness. Observation sleeps after 5 seconds without a browser.'],
  watchwaiting:['Ожидание выполнения VM','Waiting for VM execution'],watcherror:['Ошибка наблюдения · подробности в подсказке','Observation error · hover for details'],
  watchreceived:['Снимки','Samples'],watchskipped:['пропуски','skipped'],
  watchcountertip:['С момента изменения настроек: подготовленные снимки (включая ошибки) и прохождения точки съёмки, пропущенные до подготовки/GPU-выгрузки из-за лимита частоты или занятого слота.','Since configuration changed: prepared samples (including errors), and capture-point visits skipped before preparation/GPU download due to frequency limits or a busy slot.'],
  registermetadatatip:['Во время работы показаны метаданные последней паузы. Выбранные значения обновляются в обзоре наблюдения.','While running, this table retains metadata from the last pause. Selected values update in the observation overview.'],
  registerhelp:['«Значение» (значок списка в строке регистра) показывает число, текст, типизированный массив или данные обратного преобразования в JSON. «Изображение» (значок глаза) явно получает PNG из поддерживаемого Mat/GPU-тензора на паузе. Иконки присутствуют только у доступных действий. «Выполнить до» в строке инструкции продолжает камеру до остановки перед этой командой; текущая команда после остановки ещё не выполнена. Если выбранная строка уже пройдена, цель может быть достигнута в следующем кадре. Обычные точки останова продолжают действовать.','Value (the list icon in a register row) shows a scalar, text, typed array or back-transform data as JSON. Image (the eye icon) explicitly obtains a PNG from a supported Mat/GPU tensor while paused. Icons appear only for available actions. Run to in an instruction row continues this camera and stops before that instruction; it has not executed at that stop. A previously passed target may be reached on the next frame. Regular breakpoints remain active.'],
  helpbody:[
    '<p>Отладчик запускает отдельный экземпляр существующего MultiCamera pipeline: NVDEC → GPU NV12 → BGR → преобразования → детектор → трекинг → OCR. На камеру — один VmController и один кадр в работе. Прогрев выполняется без остановок, первый рабочий кадр останавливается перед первой инструкцией.</p><p>Камеры, программа, регистры и данные доступны в одном обзоре. Выберите камеру строкой таблицы или в меню. Галочка у инструкции задаёт точку останова. «До» — инструкция ещё не выполнена, «после» — её результаты уже в регистрах. F6 — один шаг, F8 — продолжить.</p><p>Просмотр картинки выполняется явно на паузе. Вход можно закрепить над следующим результатом и сравнить этапы преобразования. Нажатие картинки открывает полный PNG. Поддержаны NV12 поверхности, BGR U8 NHWC и одиночные RGB FP16/FP32 NCHW 0..1. Массивы batch показываются как метаданные; интерпретация неподдержанных форматов не подменяется.</p><p>Время инструкции исключает ожидание паузы и отладочный просмотр. Время прогона включает паузы. Захват продолжает читать источник, очередь хранит два последних кадра с вытеснением старых; «следующий кадр» — следующий принятый VM кадр. Это не покадровый проигрыватель всех кадров видео.</p><p>Закрытие вкладки сохраняет паузу. Переподключитесь или используйте API, чтобы продолжить. «Отсоединить» снимает паузу выбранной камеры. «Остановить всё» освобождает ресурсы отладочного запуска. Рабочие службы и другие процессы не управляются этим UI.</p>',
    '<p>The debugger starts a separate instance of the existing MultiCamera pipeline: NVDEC → GPU NV12 → BGR → transforms → detection → tracking → OCR. Each camera has one VmController and one in-flight frame. Warmup runs without stops; the first working frame stops before its first instruction.</p><p>Cameras, programs, registers and inspected data share one overview. Select a camera in the table or navigation. Check an instruction to set a breakpoint. Before means it has not executed; after means its results are already in the registers. F6 steps one instruction; F8 continues.</p><p>Image inspection is explicit and requires a pause. Pin an input above the next result to compare transform stages. Click an image to open its full PNG. NV12 surfaces, BGR U8 NHWC and single RGB FP16/FP32 NCHW 0..1 images are supported. Batch tensors show metadata; unsupported formats are never reinterpreted.</p><p>Instruction timing excludes debugger waits and inspection. Run timing includes pauses. Capture continues reading into a two-frame drop-oldest queue; next frame means the next frame accepted by the VM, not every consecutive frame of a video.</p><p>Closing the tab preserves the pause. Reconnect or use the API to resume. Detach releases this camera. Stop all releases the debug run resources. This UI does not control production services or other processes.</p>'
  ],
  reads:['Читает','Reads'],writes:['Пишет','Writes'],input:['Вход','Input'],frame:['Кадр','Frame'],run:['Прогон','Run'],requesting:['Получение данных…','Loading data…'],
  totalframes:['Количество декодированных кадров с начала этого запуска. Вытесненные кадры также посчитаны.','Decoded frames since this run started, including frames dropped before VM submission.'],
  fpshint:['Сглаженная частота завершения VM-прогонов, включая время пауз. Не частота декодирования.','Smoothed completed VM runs per second, including pauses. This is not decoder FPS.'],
  runhint:['Сглаженная длительность VM-прогона, включая паузы. Времена отдельных команд справа исключают паузы.','Smoothed VM run duration including pauses. Individual instruction times exclude pauses.'],
};
const t = key => words[key]?.[language === 'ru' ? 0 : 1] ?? key;
const esc = value => String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
const number = (value, digits = 0) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const cameraPath = () => `/api/cameras/${encodeURIComponent(selectedCamera)}`;
const observation = new ObservationView({api,t,esc,read,save,bootstrap:viewerBootstrap,
  onTile:(key,tile)=>workspace?.attachTile(key,tile),onRemove:key=>workspace?.removeTile(key),
  onFloat:(key,tile)=>workspace?.floatTile(key,tile),onPopup:key=>openViewer(key),
  onSync:keys=>workspace?.pruneTiles(keys),
  onError:showError,onChange:() => { if(viewerMode && observation.snapshot) { save(`viewer.${viewerId}`,JSON.stringify({camera:observation.camera,configuration:observation.configuration})); if(workspace && observation.selected.length === 0 && workspace.dock.panels.length === 0) workspace.show('registers'); } lastRender = ''; renderSelected(); }});
function updateDiagnostics() {
  $('#diagnostics').classList.toggle('error',Boolean(lastError));
  $('#diagnostics').dataset.tooltip = t(lastError ? 'hasdiagnostics':'nodiagnostics');
  $('#diagnostics').setAttribute('aria-label',t(lastError ? 'hasdiagnostics':'diagnostics'));
}
function showError(error) {
  const message = error.message ?? String(error);
  if(message !== lastError) $('#message-announcement').textContent = t('hasdiagnostics');
  lastError = message; updateDiagnostics();
}
function clearError() { lastError = ''; $('#message-announcement').textContent = ''; updateDiagnostics(); }
const batches = new BatchView({api,t,esc,read,save,onError:showError,showDialog});
async function api(path, options = {}) {
  if(/\/watch(?:\/poll)?$/.test(path)) path += `?viewer=${encodeURIComponent(viewerId)}`;
  const headers = { 'X-VM-Debug':'1', ...(token ? {Authorization:`Bearer ${token}`} : {}), ...options.headers };
  if(options.body) headers['Content-Type'] = 'application/json';
  const response = await fetch(path, { ...options, headers, signal:options.signal ?? AbortSignal.timeout(15000) });
  if(!response.ok) {
    if(response.status === 401) throw new Error(t('unauthorized'));
    let detail; try { detail = await response.json(); } catch { }
    throw new Error(detail?.error ?? `HTTP ${response.status}`);
  }
  return response;
}

function appearance() {
  shell.dataset.theme = theme; shell.dataset.density = density; shell.dataset.panelScheme = panelScheme;
  workspace?.applyAppearance();
  shell.dataset.tooltips = read('tooltips','true');
  shell.classList.toggle('menu-collapsed', read('menu','expanded') === 'collapsed');
  $('#appearance-label').textContent = `${t(theme)} · ${density.toUpperCase()} · ${t('panel' + panelScheme)}`;
  const themes = [['light','sun'],['cream','ice-cream'],['emerald','trees'],['dark','moon']];
  const densities = [['nano','baseline-density-small'],['mini','baseline-density-medium'],['norm','baseline-density-large']];
  $('#appearance-menu').innerHTML = `<strong>${t('color')}</strong>` + themes.map(([id,icon]) => `<button data-theme="${id}" aria-pressed="${theme === id}" class="${theme === id ? 'selected':''}"><i class="ti ti-${icon}" aria-hidden="true"></i> ${t(id)} ${theme === id ? '✓':''}</button>`).join('')
    + `<strong>${t('density')}</strong>` + densities.map(([id,icon]) => `<button data-density="${id}" aria-pressed="${density === id}" class="${density === id ? 'selected':''}"><i class="ti ti-${icon}" aria-hidden="true"></i> ${t(id)} ${density === id ? '✓':''}</button>`).join('')
    + `<strong>${t('panelscheme')}</strong>` + ['compact','rounded','square'].map(id => `<button data-panel-scheme="${id}" aria-pressed="${panelScheme === id}" class="${panelScheme === id ? 'selected':''}"><span class="panel-scheme-sample" data-sample="${id}" aria-hidden="true"><span></span><span></span></span> ${t('panel' + id)} ${panelScheme === id ? '✓':''}</button>`).join('')
    + `<strong>${t('tooltips')}</strong><button id="toggle-tips" role="switch" aria-checked="${shell.dataset.tooltips}"><i class="ti ti-eye${shell.dataset.tooltips === 'true' ? '':'-off'}" aria-hidden="true"></i> ${t('showtips')}</button>`;
}
function translate() {
  document.documentElement.lang = language;
  document.querySelectorAll('[data-i18n]').forEach(element => element.textContent = t(element.dataset.i18n));
  document.querySelectorAll('[data-label]').forEach(element => element.setAttribute('aria-label',t(element.dataset.label)));
  document.querySelectorAll('[data-tip]').forEach(element => element.dataset.tooltip = t(element.dataset.tip));
  $('#language').textContent = language.toUpperCase(); appearance(); updateCaptions(); updateDiagnostics(); observation.render(); workspace?.translate(); batches.render(); lastRender = ''; renderSelected();
}

function showDialog(title, html) {
  modalReturnFocus = document.activeElement;
  $('#dialog-title').textContent = title; $('#dialog-body').innerHTML = html; $('#dialog').showModal();
}
$('#dialog').addEventListener('close', () => modalReturnFocus?.focus());
$('#help').onclick = () => showDialog(t('help'),t('helpbody') + `<p>${esc(t('registerhelp'))}</p><p>${esc(t('watchtip'))}</p><p>${esc(t('watchcountertip'))}</p>`);
$('#diagnostics').onclick = () => showDialog(t('diagnostics'),`<pre class="diagnostic-detail">${esc(lastError || t('nodiagnostics'))}</pre>`);
$('#settings-nav').onclick = event => {
  event.preventDefault();
  showDialog(t('settings'), `<p class="mono">API: ${esc(location.origin)}/api</p><label>${t('accesstoken')}<input id="access-token" type="password" autocomplete="off" value="${esc(token)}"></label><p>${t('tokenhint')}</p><button id="save-token">${t('apply')}</button>`);
  $('#save-token').onclick = () => { token = $('#access-token').value; sessionStorage.setItem('nmfn.webdebug.token',token); $('#dialog').close(); clearError(); poll(); };
};
$('#language').onclick = () => { language = language === 'ru' ? 'en':'ru'; save('language',language); translate(); };
$('#menu').onclick = () => { save('menu',shell.classList.contains('menu-collapsed') ? 'expanded':'collapsed'); appearance(); };
$('#appearance-menu').onclick = event => {
  const button = event.target.closest('button'); if(!button) return;
  if(button.dataset.theme) { theme = button.dataset.theme; save('theme',theme); }
  if(button.dataset.density) { density = button.dataset.density; save('density',density); }
  if(button.dataset.panelScheme) { panelScheme = button.dataset.panelScheme; save('panelScheme',panelScheme); }
  if(button.id === 'toggle-tips') save('tooltips',shell.dataset.tooltips === 'true' ? 'false':'true');
  appearance();
};

async function selectCamera(id) {
  showPage(false); initializeWorkspace();
  if(id === selectedCamera && catalog.length) {
    if(!viewerMode) location.hash=encodeURIComponent(id);
    selectRows(); return;
  }
  selectingCamera=id;
  if(id !== selectedCamera) { selectedRegister = ''; selectedInstruction = null; clearImage(); }
  selectedCamera = id; lastRender = '';
  if(!viewerMode && location.hash !== `#${encodeURIComponent(id)}`) location.hash = encodeURIComponent(id);
  save(`camera.${viewerId}`,id);
  if(viewerMode) document.title = `VM View · ${id}`;
  const watching = observation.setCamera(id);
  try {
    const instructions = await (await api(`${cameraPath()}/instructions`)).json();
    if(selectedCamera === id) { catalog = instructions; renderSelected(); }
  }
  catch(error) { showError(error); }
  await watching;
  if(selectingCamera === id) selectingCamera='';
}
function selectRows() {
  $('#overview-nav').classList.toggle('active',shell.dataset.page === 'overview');
  $('#batches-nav').classList.toggle('active',shell.dataset.page === 'batches');
  document.querySelectorAll('[data-camera]').forEach(element => {
    element.classList.toggle(element.tagName === 'A' ? 'active':'selected',element.dataset.camera === selectedCamera && (element.tagName !== 'A' || shell.dataset.page === 'camera'));
  });
}
function renderCameras(data) {
  const cameras = data.cameras;
  $('#cameras').innerHTML = cameras.length ? cameras.map(camera => {
    const stats = camera.statistics, debug = camera.debug;
    return `<tr data-camera="${esc(camera.id)}" tabindex="0" aria-label="${esc(camera.id)}"><td>${esc(camera.id)}</td><td class="${debug.state === 'failed' ? 'error' : debug.state === 'paused' ? 'warn':''}" data-tooltip="${esc(debug.error ?? stats.lastError ?? t(debug.state))}">${t(debug.state)}</td><td>${esc(camera.decoder)}</td><td class="number">${camera.width}</td><td class="number">${camera.height}</td><td class="number" data-tooltip="${t('totalframes')}">${number(stats.framesRead)}</td><td class="number" data-tooltip="${t('fpshint')}">${number(stats.averageFps,1)}</td><td class="number" data-tooltip="${t('runhint')}">${number(stats.averageRunMilliseconds,1)}</td><td class="number">${number(stats.lastTrackCount)}</td><td class="number ${stats.runsFailed ? 'error':''}">${number(stats.runsFailed)}</td><td data-tooltip="${esc(stats.lastText)}">${esc(stats.lastText || '—')}</td></tr>`;
  }).join('') : `<tr><td colspan="11" class="muted">${t('empty')}</td></tr>`;
  const navKey = cameras.map(camera => camera.id).join('|') + language;
  if($('#camera-nav').dataset.key !== navKey) {
    $('#camera-nav').innerHTML = cameras.map(camera => `<a href="#${encodeURIComponent(camera.id)}" data-camera="${esc(camera.id)}"><i class="camera-state ti" aria-hidden="true"></i><span>${esc(camera.id)}</span></a>`).join('');
    $('#camera-nav').dataset.key = navKey;
  }
  for(const anchor of $('#camera-nav').querySelectorAll('[data-camera]')) {
    const camera=cameras.find(item=>item.id === anchor.dataset.camera), state=camera.debug.state;
    const icon=['running','detached'].includes(state) ? 'player-play' : ['paused','pausing'].includes(state) ? 'player-pause' : state === 'failed' ? 'alert-circle' : ['waiting','starting'].includes(state) ? 'refresh' : 'player-stop';
    const marker=anchor.querySelector('.camera-state'); marker.className=`camera-state ti ti-${icon}`; marker.dataset.state=state;
    anchor.setAttribute('aria-label',`${camera.id} · ${t(state)}`); anchor.dataset.tooltip=`${camera.id} · ${t(state)}${camera.debug.error ? '\n' + camera.debug.error:''}`;
  }
  selectRows();
}

function showPage(overview) {
  const page=overview === 'batches' ? 'batches' : overview ? 'overview':'camera';
  if(shell.dataset.page !== page) {
    shell.dataset.page=page;
    if(page === 'camera' && workspace) workspace.dock.layout(workspace.host.clientWidth,workspace.host.clientHeight,true);
    if(page === 'batches') batches.poll();
  }
  selectRows();
}
async function route() {
  if(viewerMode) return;
  let id; try { id=decodeURIComponent(location.hash.slice(1)) || parameters.get('camera'); } catch { id=''; }
  if(id === 'batches') { showPage('batches'); return; }
  if(!id || !current?.cameras.some(camera=>camera.id === id)) { showPage(true); return; }
  if(id !== selectedCamera || !workspace || (!catalog.length && selectingCamera !== id)) await selectCamera(id);
  else showPage(false);
}
window.addEventListener('hashchange',()=>route());

function renderSelected() {
  const camera = current?.cameras.find(item => item.id === selectedCamera);
  const snapshot = camera?.debug, paused = snapshot?.state === 'paused';
  $('#selected-camera').textContent = selectedCamera || '—';
  const point = paused || snapshot?.state === 'waiting' ? snapshot?.location : null;
  $('#stop-location').textContent = point ? `${t('run')} ${point.runId} · ${point.program} [${point.index}] · ${t(point.phase)} ${point.name}` : snapshot ? t(snapshot.state) : '—';
  $('#stop-location').dataset.tooltip = $('#stop-location').textContent;
  document.querySelectorAll('[data-command]').forEach(button => {
    button.disabled = busy || !camera || (['step','continue','next-frame'].includes(button.dataset.command) && !paused);
  });
  const activeActionKey = document.activeElement?.dataset.focusKey;
  if(activeActionKey) focusedActionKey = activeActionKey;
  document.querySelectorAll('[data-register-action]').forEach(button => button.disabled = busy || !paused);
  document.querySelectorAll('[data-run-to]').forEach(button => button.disabled = busy || !paused || button.dataset.atCurrent === 'true');
  $('#live').disabled = !camera;
  const renderKey = `${selectedCamera}|${snapshot?.revision}|${JSON.stringify(snapshot?.breakpoints)}|${JSON.stringify(snapshot?.runTo)}|${selectedRegister}|${selectedInstruction?.program}|${selectedInstruction?.index}|${language}|${catalog.length}|${observation.renderKey}`;
  if(renderKey === lastRender) { restoreRowFocus(); return; }
  lastRender = renderKey;
  let program = '';
  $('#instructions').innerHTML = catalog.map(instruction => {
    let header = '';
    if(program !== instruction.program) { program = instruction.program; header = `<tr class="program-row"><td colspan="6">${esc(program)}</td></tr>`; }
    const here = point?.program === instruction.program && point.index === instruction.index;
    const selected = selectedInstruction?.program === instruction.program && selectedInstruction.index === instruction.index;
    const breakpoint = snapshot?.breakpoints.some(item => item.program === instruction.program && item.index === instruction.index);
    const timing = snapshot?.timings.find(item => item.program === instruction.program && item.index === instruction.index);
    const atCurrent = Boolean(here && point.phase === 'before');
    const target = snapshot?.runTo?.program === program && snapshot.runTo.index === instruction.index;
    const runLabel = `${t('runto')} ${program} [${instruction.index}] · ${instruction.name}`;
    const runButton = `<button class="row-action" data-run-to data-at-current="${atCurrent}" data-focus-key="${esc(selectedCamera + '|run|' + program + '|' + instruction.index)}" aria-label="${esc(runLabel)}" data-tooltip="${esc(runLabel + '\n' + t('runtotip'))}" ${busy || !paused || atCurrent ? 'disabled':''}><i class="ti ti-player-track-next" aria-hidden="true"></i></button>`;
    return header + `<tr data-program="${esc(program)}" data-index="${instruction.index}" class="${here ? 'current':''} ${selected ? 'selected':''} ${target ? 'run-target':''}"><td><input type="checkbox" aria-label="${esc(t('breakpoint') + ' ' + program + ' ' + instruction.index)}" ${breakpoint ? 'checked':''}></td><td class="row-action-cell">${runButton}</td><td class="instruction-pointer">${here ? `<i class="ti ti-arrow-right" aria-hidden="true"></i>`:''}</td><td class="number">${instruction.index}</td><td class="instruction-name" data-tooltip="${esc(instruction.name + '\n' + instruction.operation)}">${esc(instruction.name)}</td><td class="number">${timing ? number(timing.milliseconds,2):'—'}</td></tr>`;
  }).join('');
  $('#instruction-count').textContent = String(catalog.length);
  const active = selectedInstruction ?? catalog.find(item => item.program === point?.program && item.index === point?.index);
  $('#instruction-io').textContent = active ? `${t('reads')}: ${active.reads.join(', ') || '—'}\n${t('writes')}: ${active.writes.join(', ') || '—'}` : '—';
  const registerVariables = sortRegisters(observation.knownVariables(snapshot?.variables ?? [],paused),registerSort.column,registerSort.direction);
  $('#registers').innerHTML = registerVariables.length ? registerVariables.map(variable => {
    const action = (format, icon, label) => `<button class="row-action" data-register-action="${format}" data-focus-key="${esc(selectedCamera + '|register|' + variable.key + '|' + format)}" aria-label="${esc(t(label) + ' ' + variable.key)}" data-tooltip="${esc(t(label) + ' ' + variable.key)}" ${busy || !paused ? 'disabled':''}><i class="ti ti-${icon}" aria-hidden="true"></i></button>`;
    return `<tr tabindex="0" data-register="${esc(variable.key)}" data-memory-kind="${memoryKind(variable.memory)}" class="${selectedRegister === variable.key ? 'selected':''} ${active?.reads.includes(variable.key) ? 'input-register':''} ${active?.writes.includes(variable.key) ? 'output-register':''}"><td class="register-watch-cell">${observation.checkbox(variable)}</td><td class="register-action-cell">${variable.canInspectValue ? action('json','list-details','valueaction'):''}</td><td class="register-action-cell second-action">${variable.canPreviewImage ? action('png','eye','imageaction'):''}</td><td data-tooltip="${esc(variable.key + '\n' + variable.type + (variable.value != null ? '\n' + variable.value:''))}">${esc(variable.key)}</td><td>${esc(variable.memory)}</td><td class="mono" data-tooltip="${esc(variable.elementType ?? variable.type)}">${esc(variable.shape ?? variable.value ?? '—')}</td></tr>`;
  }).join('') : `<tr><td colspan="6" class="muted">${t('noRegisters')}</td></tr>`;
  $('#register-count').textContent = String(registerVariables.length);
  $('#register-count').dataset.tooltip = t('registermetadatatip');
  for(const header of document.querySelectorAll('th[data-sort]')) header.setAttribute('aria-sort',header.dataset.sort !== registerSort.column ? 'none' : registerSort.direction === 1 ? 'ascending':'descending');
  selectRows(); restoreRowFocus();
}

function restoreRowFocus() {
  if(busy || document.activeElement !== document.body || !focusedActionKey) return;
  const button = [...document.querySelectorAll('[data-focus-key]')].find(item => item.dataset.focusKey === focusedActionKey);
  if(button && !button.disabled) button.focus({preventScroll:true});
}

async function command(name, target = {}) {
  if(viewerMode || shell.dataset.page !== 'camera') return;
  if(busy || !selectedCamera) return;
  busy = true; renderSelected();
  try {
    const snapshot = current.cameras.find(item => item.id === selectedCamera)?.debug;
    await api(`${cameraPath()}/command`,{method:'POST',body:JSON.stringify({command:name,revision:snapshot?.revision,...target})});
    clearError();
  } catch(error) { showError(error); }
  finally { busy = false; await poll(); }
}
document.querySelector('.debug-actions').onclick = event => { const button = event.target.closest('[data-command]'); if(button) command(button.dataset.command); };
for(const selector of ['#cameras','#camera-nav']) $(selector).addEventListener('click',event => { const row = event.target.closest('[data-camera]'); if(row) { event.preventDefault(); selectCamera(row.dataset.camera); } });
$('#cameras').addEventListener('keydown',event => { if(event.key === 'Enter' && event.target.dataset.camera) selectCamera(event.target.dataset.camera); });
$('#instructions').addEventListener('click', async event => {
  if(viewerMode) return;
  const row = event.target.closest('[data-program]'); if(!row) return;
  selectedInstruction = catalog.find(item => item.program === row.dataset.program && item.index === Number(row.dataset.index));
  if(event.target.closest('[data-run-to]')) {
    await command('run-to',{program:row.dataset.program,index:Number(row.dataset.index)});
    return;
  }
  if(event.target.matches('input')) {
    const breakpoints = [...$('#instructions').querySelectorAll('input:checked')].map(input => ({program:input.closest('tr').dataset.program,index:Number(input.closest('tr').dataset.index)}));
    try { await api(`${cameraPath()}/breakpoints`,{method:'PUT',body:JSON.stringify(breakpoints)}); await poll(); }
    catch(error) { showError(error); }
  }
  lastRender = ''; renderSelected();
});
$('#registers').addEventListener('click',event => {
  if(event.target.matches('[data-watch-register]')) return;
  const row = event.target.closest('[data-register]'); if(!row) return;
  selectedRegister = row.dataset.register;
  const action = event.target.closest('[data-register-action]');
  if(action) { inspect(action.dataset.registerAction,row.dataset.register); return; }
  lastRender = ''; renderSelected();
});
$('#registers').addEventListener('keydown',event => { if(event.key === 'Enter' && event.target.dataset.register) { selectedRegister = event.target.dataset.register; lastRender = ''; renderSelected(); } });

function clearImage() {
  if(imageUrl) URL.revokeObjectURL(imageUrl);
  if(pinnedUrl) URL.revokeObjectURL(pinnedUrl);
  imageUrl = pinnedUrl = imageInfo = pinnedInfo = null; $('#preview-figure').hidden = $('#pinned-figure').hidden = $('#value-output').hidden = true;
  $('#image-empty').hidden = false; $('#pin').disabled = true;
}
function captionFor(info) {
  if(!info) return '';
  return info.result ? `${info.camera} · ${t('resultframe')}` : `${info.camera} · ${t('run')} ${info.runId} · ${info.key} · ${info.program} [${info.index}] ${t(info.phase)}`;
}
function updateCaptions() {
  imageCaption = captionFor(imageInfo);
  $('#preview-caption').textContent = imageCaption; $('#preview-image').alt = imageCaption;
  $('#pinned-caption').textContent = pinnedInfo ? `${t('input')}: ${captionFor(pinnedInfo)}` : '';
  $('#pinned-image').alt = captionFor(pinnedInfo);
}
function displayImage(blob, info) {
  workspace?.show('preview');
  if(imageUrl && imageUrl !== pinnedUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = URL.createObjectURL(blob); imageInfo = info;
  $('#preview-image').src = imageUrl; $('#image-link').href = imageUrl;
  updateCaptions(); $('#preview-figure').hidden = false;
  $('#image-empty').hidden = $('#value-output').hidden = true; $('#pin').disabled = false;
}
async function inspect(format, key) {
  const snapshot = current?.cameras.find(item => item.id === selectedCamera)?.debug;
  const variable = snapshot?.variables.find(item => item.key === key);
  if(!snapshot || snapshot.state !== 'paused' || !variable || busy ||
    (format === 'json' ? !variable.canInspectValue : !variable.canPreviewImage)) return;
  busy = true; renderSelected();
  const camera = selectedCamera;
  try {
    const response = await api(`${cameraPath()}/inspect`,{method:'POST',body:JSON.stringify({key,format,revision:snapshot.revision,maxEdge:1000}),signal:AbortSignal.timeout(120000)});
    if(camera !== selectedCamera) return;
    if(format === 'png') displayImage(await response.blob(),{camera,key,...snapshot.location});
    else { $('#value-output').textContent = `${key}\n${await response.text()}`; $('#value-output').hidden = false; $('#image-empty').hidden = true; }
    clearError();
  } catch(error) { showError(error); }
  finally { busy = false; renderSelected(); }
}
$('#pin').onclick = () => {
  if(!imageUrl) return;
  if(pinnedUrl && pinnedUrl !== imageUrl) URL.revokeObjectURL(pinnedUrl);
  pinnedUrl = imageUrl; pinnedInfo = imageInfo; $('#pinned-image').src = pinnedUrl;
  updateCaptions(); $('#pinned-figure').hidden = false;
};
$('#live').onclick = async () => {
  try { const camera = selectedCamera; const response = await api(`${cameraPath()}/preview.png`); if(camera === selectedCamera) displayImage(await response.blob(),{camera,result:true}); clearError(); }
  catch(error) { showError(error); }
};
$('#start').onclick = async () => { try { await api('/api/start',{method:'POST'}); clearError(); await poll(); } catch(error) { showError(error); } };
$('#stop').onclick = async () => { try { $('#stop').disabled = true; await api('/api/stop',{method:'POST',signal:AbortSignal.timeout(120000)}); catalog = []; clearError(); await poll(); } catch(error) { showError(error); } };
document.addEventListener('keydown',event => {
  if($('#dialog').open || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
  if(event.key === 'F6') { event.preventDefault(); command('step'); }
  if(event.key === 'F8') { event.preventDefault(); command('continue'); }
});

async function poll() {
  if(polling) return;
  polling = true;
  try {
    current = await (await api('/api/status')).json();
    $('#connection').textContent = `${t('connected')} · ${t(current.state)}${current.backend ? ' · ' + current.backend:''}`;
    $('#connection').classList.remove('error');
    $('#start').disabled = current.canStart === undefined ? ['starting','running'].includes(current.state) : !current.canStart;
    $('#stop').disabled = current.canStop === undefined ? !['starting','running'].includes(current.state) : !current.canStop;
    renderCameras(current);
    const cameraOptions = current.cameras.map(camera=>`<option value="${esc(camera.id)}">${esc(camera.id)}</option>`).join('');
    if($('#viewer-camera').innerHTML !== cameraOptions) $('#viewer-camera').innerHTML = cameraOptions;
    if(!viewerMode) await route();
    else if(!current.cameras.some(camera => camera.id === selectedCamera)) {
      const preferred = read(`camera.${viewerId}`,'') || parameters.get('camera');
      selectedCamera = current.cameras.find(camera=>camera.id === preferred)?.id ?? current.cameras[0]?.id ?? ''; catalog = []; lastRender = '';
      if(selectedCamera) await selectCamera(selectedCamera); else await observation.setCamera('');
    } else if(catalog.length === 0 && selectedCamera) await selectCamera(selectedCamera);
    renderSelected();
    $('#viewer-camera').value = selectedCamera;
    if(current.error) showError(current.error);
  } catch(error) {
    $('#connection').textContent = t('disconnected'); $('#connection').classList.add('error');
    document.querySelectorAll('[data-command]').forEach(button => button.disabled = true);
    showError(error);
  } finally { polling = false; }
}
function openViewer(key) {
  const id = `view-${Date.now().toString(36)}-${Math.random().toString(36).slice(2,10)}`;
  const configuration = {...observation.configuration,registers:observation.selected.filter(item=>!key || item.key === key)};
  save(`viewer.${id}`,JSON.stringify({camera:selectedCamera,configuration}));
  const url = new URL(location.pathname,location.origin); url.searchParams.set('viewer',id); url.searchParams.set('camera',selectedCamera);
  if(!window.open(url.href,id,'popup=yes,width=1150,height=800,resizable=yes,scrollbars=yes')) showError(new Error(t('popupblocked')));
}
$('#viewer-camera').onchange = () => selectCamera($('#viewer-camera').value);
document.querySelector('.register-table thead').onclick = event => {
  const button = event.target.closest('button[data-sort]'); if(!button) return;
  registerSort = {column:button.dataset.sort,direction:registerSort.column === button.dataset.sort ? -registerSort.direction:1};
  save('registerSort',JSON.stringify(registerSort)); lastRender = ''; renderSelected();
};
function initializeWorkspace() {
  if(workspace) return;
  workspace = new DebugWorkspace({t,read,save,onError:showError,onPopup:openViewer,onCloseTile:key=>{
  const registers=observation.selected.filter(item=>item.key !== key);
  observation.configure({...observation.configuration,registers,enabled:registers.length > 0 && observation.configuration.enabled});
},viewer:viewerMode,id:viewerId,hasWatches:Boolean(viewerBootstrap?.configuration?.registers?.length)});
}
translate();
if(viewerMode) { showPage(false); initializeWorkspace(); }
poll(); setInterval(poll,600);
