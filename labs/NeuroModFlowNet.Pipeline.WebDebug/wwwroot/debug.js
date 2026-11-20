import './shell/ui-tooltip.js';
import { mountDebuggerWorkspace } from './debugger/debugger-workspace.js';
import { createDebuggerSession } from './debugger-session.js';
import { BatchView,batchWords } from './batches.js';

const $ = selector => document.querySelector(selector);
const shell = $('#shell');
const storage = 'nmfn.webdebug.ui.';
const parameters = new URLSearchParams(location.search);
const viewerId = parameters.get('viewer') || 'main';
const viewerMode = viewerId !== 'main';
const read = (key, fallback) => { try { return localStorage.getItem(storage + key) ?? fallback; } catch { return fallback; } };
const save = (key, value) => { try { localStorage.setItem(storage + key, value); } catch { /* Preferences are optional. */ } };
let viewerBootstrap = null;
try { viewerBootstrap = JSON.parse(read('viewer.' + viewerId,'null')); if(viewerBootstrap) viewerBootstrap.targetId ??= viewerBootstrap.camera; } catch { }
shell.classList.toggle('viewer-mode',viewerMode);
if(viewerMode) { $('#viewer-camera').hidden = false; }
let language = read('language', 'ru');
let theme = read('theme', 'cream');
let density = read('density', 'nano');
let panelScheme = read('panelScheme','compact');
if(!['compact','rounded','square'].includes(panelScheme)) panelScheme = 'compact';
let token = sessionStorage.getItem('nmfn.webdebug.token') || '';
let current = null, polling = false, modalReturnFocus = null, lastError = '';
const words = {
  ...batchWords,
  sections:['Разделы','Sections'], overview:['Обзор камер','Camera overview'], settings:['Настройки','Settings'],
  heading:['Виртуальные машины · камеры','Virtual machines · cameras'], start:['Запустить','Start'],stop:['Остановить всё','Stop all'],
  source:['Источник','Source'],file:['Файл','File'],files:['Файлы','Files'],folder:['Папка','Folder'],rtsp:['RTSP','RTSP'],
  labels:['Метки','Labels'], labelstip:['Метки переходов, локальные для программы','Jump labels local to the program'],
  guard:['Условие перед командой','Guard checked before the instruction'],
  unnamedGuard:['Без названия','Unnamed'],
  'ocr.hasCrops':['Есть вырезки треков','Track crops exist'],
  'ocr.hasTextRegions':['Есть области текста','Text regions exist'],
  flowSpan:['Строк в условном диапазоне','Rows in conditional span'],
  flowNext:['Следующая инструкция','Next instruction'],
  flowStop:['Остановка прогона','Stop run'],
  flowEnd:['Конец программы','End of program'],
  stoppipelinestip:['Полностью остановить все отладочные конвейеры и освободить декодеры, VM и нейросети. Повторный запуск — на странице обзора.','Fully stop all debug pipelines and release decoders, VMs and inference services. Start again from the overview.'],
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
  pintip:['Сохранить текущий снимок как «Вход» для сравнения со следующим. Он остаётся в панели при шагах и при запросе нового кадра.','Keep the current snapshot as Input to compare with the next one. It stays visible while stepping and when requesting another frame.'],
  floatpaneltip:['Сделать панель плавающей в этой вкладке. Перетаскивайте её за заголовок и меняйте размер за края. Повторное нажатие возвращает её в раскладку.','Float in this tab. Drag by the title and resize by the edges. Click again to dock.'],
  dockbacktip:['Вернуть плавающую панель в раскладку просмотра.','Return this floating panel to the viewer layout.'],
  newviewertip:['Открыть отдельное окно с уже показанным кадром или значением, в том числе на паузе. Если наблюдение включено, окно также получает новые выборки.','Open a separate window with the currently displayed frame or value, including on pause. When observation is enabled it also receives new samples.'],
  imageempty:['Для динамики отметьте регистры и одну галочку «Съём» у инструкции. Значения читаются после неё. На паузе доступны значки просмотра.','For live data, check registers and one instruction in the Capture column. Values are read after it. Inspection icons are available on pause.'],
  menu:['Свернуть / развернуть меню','Collapse / expand navigation'],help:['Справка об отладке камер','Camera debugger help'],close:['Закрыть окно','Close dialog'],
  pausetip:['Остановиться перед следующей инструкцией выбранной камеры. Другие камеры продолжают работу.','Stop before the next instruction of this camera. Other cameras continue.'],
  continuetip:['Продолжить до точки останова или запроса паузы. F8.','Continue until a breakpoint or pause request. F8.'],
  steptip:['Выполнить ровно одну инструкцию и остановиться после неё. F6.','Execute exactly one instruction and stop after it. F6.'],
  nextframetip:['Закончить текущий кадр и остановиться перед первой инструкцией следующего. Точки останова сохраняются.','Finish this frame and stop before the first instruction of the next. Breakpoints remain active.'],
  detachtip:['Снять паузу и отключить остановки этой камеры. Повторная пауза подключит отладчик.','Release the pause and disable stops for this camera. Pause attaches the debugger again.'],
  previewtip:['Явно уменьшить и выгрузить выбранный GPU-тензор в PNG на остановленной VM. Это добавляет работу только во время просмотра.','Explicitly resize and download the selected GPU tensor to PNG on the paused VM. This adds work only during inspection.'],
  paused:['Пауза','Paused'],pausing:['Запрос паузы','Pause requested'],running:['Работает','Running'],starting:['Запуск','Starting'],stopped:['Остановлено','Stopped'],waiting:['Прогрев','Warmup'],failed:['Ошибка','Failed'],detached:['Без отладки','Detached'],
  before:['до','before'],after:['после','after'],warmup:['прогрев','warmup'],empty:['Нет запущенных камер','No cameras running'],noRegisters:['Каталог заполнится при выполнении инструкций','The catalog fills as instructions execute'],
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
  watchsizetip:['Предел длинной стороны картинки; для батча — каждой ячейки сетки. Одиночные GPU-картинки уменьшаются до выгрузки, батч — при сборке сетки на CPU. «Исходный» сохраняет разрешение. На вход нейросети не влияет. Кнопка 1:1 отдельно меняет масштаб отображения.','Longest image edge; for a batch, each grid cell. Single GPU images are downscaled before download; batch cells during CPU grid assembly. Original preserves resolution. Model inputs are unchanged. The 1:1 button independently changes display scale.'],
  originalsize:['Исходный','Original'],nativeview:['Отображение картинки 1:1','Display image at 1:1'],
  nativeviewtip:['1:1 — один пиксель картинки на CSS-пиксель с прокруткой внутри просмотра. Повторное нажатие вписывает картинку в панель. Передаваемый размер не меняется.','1:1 displays one image pixel per CSS pixel with scrolling inside the viewer. Click again to fit the image in the panel. Transmitted resolution is unchanged.'],
  watchtip:['Отметьте до 8 регистров и одну инструкцию в колонке «Съём». Все выбранные значения читаются после неё на каждом прогоне с ограничением частоты. Если имени в этой точке нет, значение недоступно; другая точка не подставляется. Занятые кадры пропускаются до выгрузки с GPU. Без клиента сбор и наблюдение засыпают через 5 секунд.','Check up to 8 registers and one instruction in the Capture column. All selected values are read after it on each run, subject to rate limits. Missing names are unavailable; another step is never substituted. Busy frames are dropped before GPU download. Collection and observation sleep after 5 seconds without a client.'],
  watchwaiting:['Ожидание выполнения VM','Waiting for VM execution'],watcherror:['Ошибка наблюдения · подробности в подсказке','Observation error · hover for details'],
  watchreceived:['Снимки','Samples'],watchskipped:['пропуски','skipped'],
  watchcountertip:['С момента изменения настроек: подготовленные снимки (включая ошибки) и прохождения точки съёмки, пропущенные до подготовки/GPU-выгрузки из-за лимита частоты или занятого слота.','Since configuration changed: prepared samples (including errors), and capture-point visits skipped before preparation/GPU download due to frequency limits or a busy slot.'],
  registermetadatatip:['Каталог реально встречавшихся регистров между кадрами, а не текущие значения. Удалённые имена остаются. Метаданные точки обновляются не чаще раза в секунду; значения читаются только по выбранному наблюдению. До/после относится к границе инструкции. Создание и удаление внутри одной инструкции не видны. Предел 512 имён; + у счётчика означает переполнение.','Catalog of registers actually observed across frames, not current values. Removed names remain. Point metadata refreshes at most once per second; values are read only by selected observations. Before/after means an instruction boundary. Creation and removal within a single instruction are invisible. Capacity is 512 names; + on the count means overflow.'],
  captureshort:['Съём','Capture'],
  watchmissing:['Нет регистра в точке съёма','Register absent at capture point'],
  watchchoosepoint:['Отметьте «Съём» у инструкции','Check Capture on an instruction'],
  resultpreviewtip:['Один снимок по запросу. При работе — следующий завершённый кадр с разметкой; на паузе — текущее image.rect, если оно уже подготовлено. Без запроса или готового наблюдения превью не выгружается.','One snapshot on demand. While running: the next completed annotated frame; while paused: current image.rect if already prepared. No preview download without a request or a ready observation.'],
  watchpoint:['Точка съёма','Capture point'],watchpointtip:['Отметьте одну инструкцию: после её выполнения читаются все регистры, выбранные галочками. Другая отметка переносит точку съёма, снятие выключает наблюдение. Это не точка останова.','Check one instruction: all checked registers are read after it executes. Checking another moves the capture point; unchecking disables observation. This is independent of breakpoints.'],
  registercataloghelp:['Справка: каталог регистров и точки съёма','Register catalog and capture point help'],
  creatorShort:['Оп.','Op.'],creator:['Операция, создавшая регистр','Operation that created the register'],
  creatorTip:['Номер инструкции, до которой имени не было, а после которой оно появилось. Программа и имя инструкции — в подсказке значения. — означает внешний вход или ещё не подтверждённое создание.','Instruction index with confirmed absence before and presence after execution. The value tooltip identifies the program and instruction. — means an external input or creation not yet confirmed.'],
  registerview:['Просмотр регистра','Inspect register'],registerviewtip:['На паузе открыть изображение или данные в зависимости от содержимого регистра.','On pause, open an image or structured data according to register contents.'],
  additionOrder:['№','№'],additionOrderLabel:['Порядок появления регистров','Register discovery order'],additionOrderTip:['Номер первого обнаружения имени в каталоге камеры. Сохраняется между кадрами, при перезаписи и удалении. Отражает наблюдаемое выполнение с момента подключения: поздно достигнутая ветка добавит имена в конец. Точный порядок изменений внутри одной инструкции не определяется.','First discovery number in this camera catalog. Retained across frames, overwrites and removal. Reflects observed execution since connection: a branch reached later appends names. Mutation order within a single instruction is not determined.'],
  'catalog-idle':['Сбор каталога выключен; сохранённые сведения','Catalog collection is off; retained metadata'],
  'previous-run':['Встречался в предыдущих прогонах','Seen in previous runs'],'run-ended':['Прогон завершён; имя сохранено','Run ended; name retained'],
  seen:['Замечен на последней наблюдаемой границе','Seen at the last observed boundary'],removed:['Исчез после границы инструкции; имя сохранено','Disappeared at an instruction boundary; name retained'],
  'paused-present':['Есть на текущей паузе','Present at the current pause'],'not-at-pause':['На текущей паузе отсутствует; можно выбрать динамическое наблюдение','Absent at the current pause; live observation may still be selected'],
  registerhelp:['«Значение» (значок списка в строке регистра) показывает число, текст, типизированный массив или данные обратного преобразования в JSON. «Изображение» (значок глаза) явно получает PNG из поддерживаемого Mat/GPU-тензора на паузе. Иконки присутствуют только у доступных действий. «Выполнить до» в строке инструкции продолжает камеру до остановки перед этой командой; текущая команда после остановки ещё не выполнена. Если выбранная строка уже пройдена, цель может быть достигнута в следующем кадре. Обычные точки останова продолжают действовать.','Value (the list icon in a register row) shows a scalar, text, typed array or back-transform data as JSON. Image (the eye icon) explicitly obtains a PNG from a supported Mat/GPU tensor while paused. Icons appear only for available actions. Run to in an instruction row continues this camera and stops before that instruction; it has not executed at that stop. A previously passed target may be reached on the next frame. Regular breakpoints remain active.'],
  helpbody:[
    '<p>Отладчик запускает отдельный экземпляр существующего MultiCamera pipeline: NVDEC → GPU NV12 → BGR → преобразования → детектор → трекинг → OCR. На камеру — один VmController и один кадр в работе. Прогрев выполняется без остановок, первый рабочий кадр останавливается перед первой инструкцией.</p><p>Камеры, программа, регистры и данные доступны в одном обзоре. Выберите камеру строкой таблицы или в меню. «Т.О.» задаёт точку останова, «Съём» — момент наблюдения после инструкции. «До» — инструкция ещё не выполнена, «после» — её результаты уже в регистрах. F6 — один шаг, F8 — продолжить.</p><p>Просмотр картинки выполняется явно на паузе. Вход можно закрепить над следующим результатом и сравнить этапы преобразования. Нажатие картинки открывает полный PNG. Поддержаны NV12, BGR U8 NHWC, RGB FP16/FP32 NCHW и нумерованные сетки батчей. ocr.recInput восстанавливается из PaddleOCR −1…1; ocr.roiCount и ocr.crop.count отделяют реальные области от заполнения. Неподдержанные форматы не подменяются.</p><p>Время инструкции исключает ожидание паузы и отладочный просмотр. Время прогона включает паузы. Захват продолжает читать источник, очередь хранит два последних кадра с вытеснением старых; «следующий кадр» — следующий принятый VM кадр. Это не покадровый проигрыватель всех кадров видео.</p><p>Закрытие вкладки сохраняет паузу. Переподключитесь или используйте API, чтобы продолжить. «Отсоединить» снимает паузу выбранной камеры. «Остановить всё» освобождает ресурсы отладочного запуска. Рабочие службы и другие процессы не управляются этим UI.</p>',
    '<p>The debugger starts a separate instance of the existing MultiCamera pipeline: NVDEC → GPU NV12 → BGR → transforms → detection → tracking → OCR. Each camera has one VmController and one in-flight frame. Warmup runs without stops; the first working frame stops before its first instruction.</p><p>Cameras, programs, registers and inspected data share one overview. Select a camera in the table or navigation. The breakpoint checkbox stops execution; Capture chooses observation after the instruction. Before means it has not executed; after means its results are already in the registers. F6 steps one instruction; F8 continues.</p><p>Image inspection is explicit and requires a pause. Pin an input above the next result to compare transform stages. Click an image to open its full PNG. NV12, BGR U8 NHWC, RGB FP16/FP32 NCHW and numbered image-batch grids are supported. ocr.recInput uses explicit PaddleOCR −1…1 normalization; ocr.roiCount and ocr.crop.count distinguish real regions from padding. Unsupported formats are never reinterpreted.</p><p>Instruction timing excludes debugger waits and inspection. Run timing includes pauses. Capture continues reading into a two-frame drop-oldest queue; next frame means the next frame accepted by the VM, not every consecutive frame of a video.</p><p>Closing the tab preserves the pause. Reconnect or use the API to resume. Detach releases this camera. Stop all releases the debug run resources. This UI does not control production services or other processes.</p>'
  ],
  reads:['Читает','Reads'],writes:['Пишет','Writes'],input:['Вход','Input'],frame:['Кадр','Frame'],run:['Прогон','Run'],requesting:['Получение данных…','Loading data…'],
  totalframes:['Количество декодированных кадров с начала этого запуска. Вытесненные кадры также посчитаны.','Decoded frames since this run started, including frames dropped before VM submission.'],
  fpshint:['Сглаженная частота завершения VM-прогонов, включая время пауз. Не частота декодирования.','Smoothed completed VM runs per second, including pauses. This is not decoder FPS.'],
  runhint:['Сглаженная длительность VM-прогона, включая паузы. Времена отдельных команд справа исключают паузы.','Smoothed VM run duration including pauses. Individual instruction times exclude pauses.'],
};
const t = key => words[key]?.[language === 'ru' ? 0 : 1] ?? key;
const esc = value => String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
const number = (value, digits = 0) => Number.isFinite(value) ? value.toFixed(digits) : '—';
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
  debuggerUi.applyAppearance();
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
  $('#language').textContent = language.toUpperCase(); appearance(); updateDiagnostics(); debuggerUi.translate(); batches.render();
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

const debuggerUi = await mountDebuggerWorkspace($('#debugger-workspace'),{
  session:createDebuggerSession(api),
  t,read,save,onError:showError,clearError,onRefresh:poll,onOpenViewer:openViewer,
  onHelp:()=>showDialog(t('watchpoint'),'<p>'+esc(t('watchpointtip'))+'</p><p>'+esc(t('registermetadatatip'))+'</p><p>'+esc(t('watchtip'))+'</p>'),
  isActive:()=>shell.dataset.page === 'camera',
  canUseShortcuts:()=>shell.dataset.page === 'camera' && !$('#dialog').open,
  viewerId,viewerBootstrap
});
$('.debug-actions').prepend($('#debugger-host-actions').content.cloneNode(true));
$('#panels-menu').onclick = event => {
  const button = event.target.closest('[data-panel]'); if(!button) return;
  debuggerUi.showPanel(button.dataset.panel); $('#panels-menu').open = false;
};
$('#reset-layout').onclick = () => { debuggerUi.resetLayout(); $('#panels-menu').open = false; };
$('#new-viewer').onclick = () => { debuggerUi.openViewer(); $('#panels-menu').open = false; };
async function selectCamera(id) {
  showPage(false);
  if(!viewerMode && location.hash !== '#' + encodeURIComponent(id)) location.hash = encodeURIComponent(id);
  save('camera.' + viewerId,id);
  if(viewerMode) document.title = 'VM View · ' + id;
  await debuggerUi.selectTarget(id);
  if(debuggerUi.targetId === id) debuggerUi.update(current?.cameras.find(camera=>camera.id === id)?.debug);
  selectRows();
}
function selectRows() {
  $('#overview-nav').classList.toggle('active',shell.dataset.page === 'overview');
  $('#batches-nav').classList.toggle('active',shell.dataset.page === 'batches');
  document.querySelectorAll('[data-camera]').forEach(element => {
    element.classList.toggle(element.tagName === 'A' ? 'active':'selected',element.dataset.camera === debuggerUi.targetId && (element.tagName !== 'A' || shell.dataset.page === 'camera'));
  });
}
function renderCameras(data) {
  const cameras = data.cameras;
  $('#cameras').innerHTML = cameras.length ? cameras.map(camera => {
    const stats = camera.statistics, debug = camera.debug;
    return `<tr data-camera="${esc(camera.id)}" tabindex="0" aria-label="${esc(camera.id)}"><td>${esc(camera.id)}</td><td>${esc(t(camera.sourceKind))}</td><td class="${debug.state === 'failed' ? 'error' : debug.state === 'paused' ? 'warn':''}" data-tooltip="${esc(debug.error ?? stats.lastError ?? t(debug.state))}">${t(debug.state)}</td><td>${esc(camera.decoder)}</td><td class="number">${camera.width}</td><td class="number">${camera.height}</td><td class="number" data-tooltip="${t('totalframes')}">${number(stats.framesRead)}</td><td class="number" data-tooltip="${t('fpshint')}">${number(stats.averageFps,1)}</td><td class="number" data-tooltip="${t('runhint')}">${number(stats.averageRunMilliseconds,1)}</td><td class="number">${number(stats.lastTrackCount)}</td><td class="number ${stats.runsFailed ? 'error':''}">${number(stats.runsFailed)}</td><td data-tooltip="${esc(stats.lastText)}">${esc(stats.lastText || '—')}</td></tr>`;
  }).join('') : `<tr><td colspan="12" class="muted">${t('empty')}</td></tr>`;
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
    if(page === 'camera') debuggerUi.layout();
    if(page === 'batches') batches.poll();
  }
  selectRows();
}
async function route() {
  if(viewerMode) return;
  let id; try { id=decodeURIComponent(location.hash.slice(1)) || parameters.get('camera'); } catch { id=''; }
  if(id === 'batches') { showPage('batches'); return; }
  if(!id || !current?.cameras.some(camera=>camera.id === id)) { showPage(true); return; }
  if(id !== debuggerUi.targetId || debuggerUi.needsInstructions) await selectCamera(id);
  else showPage(false);
}
window.addEventListener('hashchange',()=>route());
for(const selector of ['#cameras','#camera-nav']) $(selector).addEventListener('click',event => { const row = event.target.closest('[data-camera]'); if(row) { event.preventDefault(); selectCamera(row.dataset.camera); } });
$('#cameras').addEventListener('keydown',event => { if(event.key === 'Enter' && event.target.dataset.camera) selectCamera(event.target.dataset.camera); });
$('#start').onclick = async () => { try { await api('/api/start',{method:'POST'}); clearError(); await poll(); } catch(error) { showError(error); } };
$('#stop').onclick = $('#stop-pipelines').onclick = async () => {
  try {
    $('#stop').disabled = $('#stop-pipelines').disabled = true;
    await api('/api/stop',{method:'POST',signal:AbortSignal.timeout(120000)});
    await debuggerUi.clearTarget(); clearError(); await poll();
  } catch(error) { showError(error); }
};
async function poll() {
  if(polling) return;
  polling = true;
  try {
    current = await (await api('/api/status')).json();
    $('#connection').textContent = `${t('connected')} · ${t(current.state)}${current.backend ? ' · ' + current.backend:''}`;
    $('#connection').classList.remove('error');
    $('#start').disabled = current.canStart === undefined ? ['starting','running'].includes(current.state) : !current.canStart;
    $('#stop').disabled = current.canStop === undefined ? !['starting','running'].includes(current.state) : !current.canStop;
    $('#stop-pipelines').disabled = $('#stop').disabled;
    renderCameras(current);
    const cameraOptions = current.cameras.map(camera=>`<option value="${esc(camera.id)}">${esc(camera.id)}</option>`).join('');
    if($('#viewer-camera').innerHTML !== cameraOptions) $('#viewer-camera').innerHTML = cameraOptions;
    if(!viewerMode) await route();
    else if(!current.cameras.some(camera => camera.id === debuggerUi.targetId)) {
      const preferred = read(`camera.${viewerId}`,'') || parameters.get('camera');
      const id = current.cameras.find(camera=>camera.id === preferred)?.id ?? current.cameras[0]?.id ?? '';
      if(id) await selectCamera(id); else await debuggerUi.clearTarget();
    } else if(debuggerUi.needsInstructions && debuggerUi.targetId) await selectCamera(debuggerUi.targetId);
    debuggerUi.update(current.cameras.find(camera=>camera.id === debuggerUi.targetId)?.debug);
    $('#viewer-camera').value = debuggerUi.targetId;
    if(current.error) showError(current.error);
  } catch(error) {
    $('#connection').textContent = t('disconnected'); $('#connection').classList.add('error');
    document.querySelectorAll('[data-command]').forEach(button => button.disabled = true);
    showError(error);
  } finally { polling = false; }
}
function openViewer({targetId,configuration,sample}) {
  const id = `view-${Date.now().toString(36)}-${Math.random().toString(36).slice(2,10)}`;
  save(`viewer.${id}`,JSON.stringify({targetId,configuration}));
  const url = new URL(location.pathname,location.origin); url.searchParams.set('viewer',id); url.searchParams.set('camera',targetId);
  const popup = window.open(url.href,id,'popup=yes,width=1150,height=800,resizable=yes,scrollbars=yes');
  if(!popup) { showError(new Error(t('popupblocked'))); return; }
  const snapshot = sample;
  const ready = event => {
    if(event.origin !== location.origin || event.source !== popup || event.data?.previewReady !== id) return;
    window.removeEventListener('message',ready);
    if(snapshot && !popup.closed) popup.postMessage({previewSample:id,sample:snapshot},location.origin);
  };
  window.addEventListener('message',ready);
  setTimeout(()=>window.removeEventListener('message',ready),30000);
}
$('#viewer-camera').onchange = () => selectCamera($('#viewer-camera').value);
translate();
if(viewerMode) {
  showPage(false);
  const parent = window.opener;
  if(parent) {
    const receive = event => {
      if(event.origin !== location.origin || event.source !== parent || event.data?.previewSample !== viewerId) return;
      window.removeEventListener('message',receive);
      debuggerUi.showSample(event.data.sample,Boolean(viewerBootstrap?.configuration?.registers?.length));
    };
    window.addEventListener('message',receive);
    setTimeout(()=>window.removeEventListener('message',receive),30000);
    parent.postMessage({previewReady:viewerId},location.origin);
    window.opener = null;
  }
}
poll(); setInterval(poll,600);
