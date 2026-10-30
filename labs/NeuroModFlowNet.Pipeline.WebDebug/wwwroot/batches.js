export const batchWords = {
  batches:['Батчи нейросетей','Inference batches'],batchServices:['Общие сервисы нейросетей','Shared inference services'],
  batchCollect:['Собирать историю','Collect history'],batchCollecting:['Сбор включён','Collecting'],batchIdle:['Сбор выключен','Collection off'],
  batchHelp:['Как считаются батчи','How batches are measured'],batchModel:['Модель','Model'],batchBackend:['Backend','Backend'],
  batchLimit:['Заявок ≤','Requests ≤'],batchWait:['Ожидание, мс','Wait, ms'],batchQueue:['Очередь ≤','Queue ≤'],batchInput:['Вход, N','Input, N'],
  batchPrepared:['Собрано','Prepared'],batchCompleted:['Завершено','Completed'],batchRequests:['Заявки','Requests'],
  batchRows:['Строки тензора','Tensor rows'],batchDuration:['Обработка, мс','Processing, ms'],batchAssembly:['Сборка, мс','Assembly, ms'],
  batchTime:['Время','Time'],batchSplit:['Строк в каждой заявке','Rows per request'],batchHistory:['История выбранного сервиса','Selected service history'],
  batchRecent:['Последние','Recent'],batchWindow:['Число батчей на графике','Batches shown in chart'],
  batchRequestChart:['Заявок на батч · по порядку сборки','Requests per batch · assembly order'],
  batchRowChart:['Строк входного тензора · по порядку сборки','Input tensor rows · assembly order'],
  batchEmpty:['Нет наблюдений. Включите сбор и продолжите камеры.','No observations. Enable collection and resume cameras.'],
  batchNoServices:['Сервисы появятся после запуска камер','Services appear when cameras start'],
  batchPreparedState:['Собран','Prepared'],batchCompletedState:['Завершён','Completed'],batchFailedState:['Ошибка декодирования','Decode failed'],batchInterruptedState:['Не завершён','Incomplete'],
  batchLimitTip:['Максимум VM-заявок, объединяемых сервисом за один запуск. Одна заявка может содержать несколько строк тензора.','Maximum VM requests merged by the service per execution. One request may contain several tensor rows.'],
  batchQueueTip:['Настроенная ёмкость очереди заявок; это не текущая длина очереди.','Configured pending request capacity; this is not the current queue depth.'],
  batchPreparedTip:['Число успешно собранных входных батчей за периоды включённого наблюдения. Завершение модели подтверждается отдельно.','Successfully assembled input batches during observed periods. Model completion is confirmed separately.'],
  batchRowsTip:['Размер первой оси входного тензора. Включает padding, если он уже есть во входе; это не число реальных объектов или вырезок.','First dimension of the input tensor. Includes padding already present in the input; this is not a count of actual objects or crops.'],
  batchDurationTip:['От начала сборки входа до возврата декодера результата, включая вызов модели. Без ожидания в очереди и без первого создания сессии.','From input assembly start through result decoder return, including model execution. Excludes queue waiting and initial session creation.'],
  batchHelpBody:[
    'Сервисы общие для камер. «Лимит заявок», «Ожидание» и «Лимит очереди» — действующая конфигурация, а не текущая заполненность очереди. В последних трёх колонках — последний наблюдаемый батч. «Собрано» и «Завершено» считаются только за периоды включённого наблюдения, до перезапуска камер.\n\nОдна VM-заявка может содержать целый пакет вырезок. Поэтому число заявок и число строк тензора показаны отдельно. Строки включают уже подготовленный padding; число реальных ROI здесь неизвестно. Процент полезной загрузки по ним не вычисляется.\n\n«Собран» означает успешный возврат сборщика входа, до вызова модели. «Завершён» подтверждается возвратом декодера. Если выполнение оборвалось раньше декодера, успешное завершение не приписывается. «Обработка» включает сборку, модель и декодирование; ожидание в очереди и первое создание сессии не измеряются.\n\nНа графиках каждый столбик — один наблюдаемый батч, слева старые, справа новые. Это порядок батчей, не равномерная шкала времени. Полные числа, время и разбивка по заявкам доступны в таблице. Хранятся последние 256 батчей каждого сервиса; более старые вытесняются, общие счётчики сохраняются.\n\nСбор действует при открытой видимой странице и отмеченной галочке. После ухода со страницы или закрытия окна разрешение истекает за 5 секунд; несколько окон продлевают его независимо. Снятие галочки останавливает сбор в этом окне, уже собранная история остаётся. Тензоры не копируются, параметры батчинга не меняются.',
    'Services are shared by cameras. Request limit, Wait and Queue limit are current configuration, not live queue occupancy. The last three columns describe the last observed batch. Prepared and Completed count only observed periods until cameras restart.\n\nOne VM request may contain a whole crop batch. Request count and tensor row count are therefore separate. Rows include padding already prepared upstream; the actual ROI count is unknown here. No useful-capacity percentage is inferred.\n\nPrepared means input assembly returned successfully, before the model call. Completed is confirmed by result decoder return. An interruption before decoding is not reported as success. Processing includes assembly, model execution and decoding; queue waiting and initial session creation are not measured.\n\nEach bar is one observed batch, oldest left, newest right. This is batch order, not uniform time. Exact counts, time and rows per request are in the table. The latest 256 batches per service are retained; older samples are evicted while aggregate counters remain.\n\nCollection runs while this page is visible and the checkbox is selected. Leaving or closing the page expires permission within 5 seconds; multiple windows renew independently. Unchecking stops collection from this window and retains history. Tensors are not copied and batching parameters are unchanged.'
  ]
};

export class BatchView {
  constructor({api,t,esc,read,save,onError,showDialog}) {
    Object.assign(this,{api,t,esc,read,save,onError,showDialog});
    this.collect=document.querySelector('#batch-collect');
    this.collect.checked=read('batchCollect','true') === 'true';
    this.selected=read('batchService','');
    this.window=document.querySelector('#batch-window');
    this.window.value=['64','128','256'].includes(read('batchWindow','64')) ? read('batchWindow','64'):'64';
    this.collect.onchange=()=>{ save('batchCollect',String(this.collect.checked)); this.poll(); };
    this.window.onchange=()=>{ save('batchWindow',this.window.value); this.render(); };
    document.querySelector('#batch-help').onclick=()=>showDialog(t('batchHelp'),`<pre class="batch-help-text">${esc(t('batchHelpBody'))}</pre>`);
    const rows=document.querySelector('#batch-services');
    const select=event=>{ const row=event.target.closest('[data-endpoint]'); if(row) { this.selected=row.dataset.endpoint; save('batchService',this.selected); this.render(); } };
    rows.onclick=select;
    rows.onkeydown=event=>{ if(event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(event); } };
    setInterval(()=>this.poll(),1000);
  }

  async poll() {
    if(this.polling || document.hidden || document.querySelector('#shell').dataset.page !== 'batches') return;
    this.polling=true;
    try {
      this.snapshot=await (await this.api(this.collect.checked ? '/api/batches/poll':'/api/batches',this.collect.checked ? {method:'POST'}:{})).json();
      this.render();
    } catch(error) { this.onError(error); }
    finally { this.polling=false; }
  }

  state(sample) {
    return this.t(({prepared:'batchPreparedState',completed:'batchCompletedState','decode-failed':'batchFailedState',interrupted:'batchInterruptedState'})[sample?.state] ?? 'batchPreparedState');
  }

  render() {
    const {t,esc}=this, endpoints=this.snapshot?.endpoints ?? [];
    const num=(value,digits=0)=>Number.isFinite(value) ? value.toFixed(digits):'—';
    document.querySelector('#batch-state').textContent=t(this.snapshot?.collecting ? 'batchCollecting':'batchIdle');
    if(!endpoints.some(endpoint=>endpoint.name === this.selected)) this.selected=endpoints[0]?.name;
    const rows=document.querySelector('#batch-services');
    const key=JSON.stringify([endpoints.map(endpoint=>endpoint.name),t('batchModel'),this.snapshot?.generation]);
    if(this.rowKey !== key) {
      rows.innerHTML=endpoints.length ? endpoints.map(endpoint=>`<tr tabindex="0" data-endpoint="${esc(endpoint.name)}"><td></td><td></td>${'<td class="number"></td>'.repeat(8)}</tr>`).join('') : `<tr><td colspan="10" class="muted">${t('batchNoServices')}</td></tr>`;
      this.rowKey=key;
    }
    endpoints.forEach((endpoint,index)=>{
      const sample=endpoint.samples.at(-1), row=rows.children[index];
      row.classList.toggle('selected',endpoint.name === this.selected);
      row.setAttribute('aria-label',`${endpoint.model} · ${endpoint.name}`);
      const values=[endpoint.model,endpoint.backend,endpoint.maxRequests,num(endpoint.maxWaitMilliseconds,1),endpoint.pendingCapacity,endpoint.prepared,endpoint.completed,sample?.requestCount ?? '—',sample?.tensorRows ?? '—',num(sample?.processingMilliseconds,2)];
      values.forEach((value,column)=>row.children[column].textContent=value);
      row.children[0].dataset.tooltip=`${endpoint.name}\n${endpoint.model}`;
    });
    const endpoint=endpoints.find(item=>item.name === this.selected), samples=endpoint?.samples ?? [];
    document.querySelector('#batch-selected').textContent=endpoint?.name ?? '—';
    document.querySelector('#batch-selected').dataset.tooltip=endpoint?.name ?? '—';
    document.querySelector('#batch-history-count').textContent=`${samples.length} / ${this.snapshot?.capacity ?? 256}`;
    document.querySelector('#batch-history').innerHTML=samples.length ? samples.toReversed().map(sample=>`<tr><td class="number">${sample.sequence}</td><td class="mono">${this.time(sample.timestamp)}</td><td class="${sample.state === 'decode-failed' ? 'error':sample.state !== 'completed' ? 'warn':''}">${this.state(sample)}</td><td class="number">${sample.requestCount}</td><td class="number">${sample.tensorRows}</td><td class="mono">${sample.requestRows.join(' + ') || '—'}</td><td class="number">${num(sample.assemblyMilliseconds,2)}</td><td class="number">${num(sample.processingMilliseconds,2)}</td></tr>`).join('') : `<tr><td colspan="8" class="muted">${t('batchEmpty')}</td></tr>`;
    const window=samples.slice(-Number(this.window.value));
    document.querySelector('#batch-request-chart').innerHTML=this.chart(window,'requestCount',t('batchRequestChart'),endpoint?.maxRequests ?? 1,'requests');
    document.querySelector('#batch-row-chart').innerHTML=this.chart(window,'tensorRows',t('batchRowChart'),1,'rows');
  }

  time(timestamp) {
    const date=new Date(timestamp);
    return `${date.toLocaleTimeString('en-GB',{hour12:false})}.${String(date.getMilliseconds()).padStart(3,'0')}`;
  }

  chart(samples,field,label,minimum,kind) {
    const {t,esc}=this;
    if(!samples.length) return `<div class="batch-chart-empty muted">${t('batchEmpty')}</div>`;
    const maximum=Math.max(minimum,...samples.map(sample=>sample[field]));
    const width=600,height=150,left=40,right=10,top=8,bottom=24;
    const plotHeight=height-top-bottom, plotWidth=width-left-right;
    const slot=plotWidth/samples.length;
    const bars=samples.map((sample,index)=>{
      const h=sample[field]/maximum*plotHeight, x=left+index*slot;
      const detail=`#${sample.sequence} · ${this.time(sample.timestamp)}\n${t('batchRequests')}: ${sample.requestCount}\n${t('batchRows')}: ${sample.tensorRows}\n${this.state(sample)}`;
      return `<rect class="batch-bar batch-${kind}" x="${x.toFixed(2)}" y="${(top+plotHeight-h).toFixed(2)}" width="${Math.max(.5,slot-1).toFixed(2)}" height="${h.toFixed(2)}" data-tooltip="${esc(detail)}"/>`;
    }).join('');
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(label)}: #${samples[0].sequence}–${samples.at(-1).sequence}; max ${maximum}"><path class="batch-axis" d="M${left} ${top}H${width-right}M${left} ${height-bottom}H${width-right}"/><text class="batch-axis-label" x="${left-6}" y="${top+10}" text-anchor="end">${maximum}</text><text class="batch-axis-label" x="${left-6}" y="${height-bottom}" text-anchor="end">0</text>${bars}<text class="batch-axis-label" x="${left}" y="${height-5}">#${samples[0].sequence}</text><text class="batch-axis-label" x="${width-right}" y="${height-5}" text-anchor="end">#${samples.at(-1).sequence}</text></svg>`;
  }
}
