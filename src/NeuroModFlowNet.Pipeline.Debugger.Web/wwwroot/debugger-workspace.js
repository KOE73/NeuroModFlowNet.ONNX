import { ObservationView } from './observation.js';
import { DebugWorkspace } from './workspace.js';
import { memoryKind,sortRegisters } from './register-table.js';
import { layoutInstructions } from './instruction-flow.js';
import { enablePreviewLayout,sampleImages } from './preview-layout.js';

export async function mountDebuggerWorkspace(root, {
  session, t = key => key, read = (_, fallback) => fallback, save = () => {},
  onError = () => {}, clearError = () => {}, onRefresh = () => {}, onHelp,
  onOpenViewer, isActive = () => !root.hidden,
  canUseShortcuts = () => root.contains(document.activeElement),
  viewerId = 'main', viewerBootstrap = null
}) {
  const template = await fetch(new URL('./workspace.html',import.meta.url));
  if(!template.ok) throw new Error('Debugger workspace markup could not be loaded.');
  root.dataset.debuggerWorkspace = '';
  root.innerHTML = await template.text();
  const $ = selector => root.querySelector(selector);
  const lifetime = new AbortController();
  const signal = lifetime.signal;
  const viewerMode = viewerId !== 'main';
  const esc = value => String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
  const number = (value, digits = 0) => Number.isFinite(value) ? value.toFixed(digits) : '—';
  let selectedTarget = '', selectedRegister = '', selectedInstruction = null, snapshot;
  let catalog = [], instructionLayout = [], lastRender = '', busy = false, selectingTarget = '';
  let imageUrl = null, pinnedUrl = null, imageInfo = null, pinnedInfo = null, imageCaption = '', focusedActionKey = '';
  let workspace;
  let registerSort = {column:'key',direction:1};
  try { registerSort = JSON.parse(read('registerSort','null')) ?? registerSort; } catch { }
  if(!['key','memory','shape','additionOrder'].includes(registerSort.column) || ![1,-1].includes(registerSort.direction))
    registerSort = {column:'key',direction:1};

  const observation = new ObservationView({root,session,signal,isActive,t,esc,read,save,bootstrap:viewerBootstrap,viewerId,
    onTile:(key,tile)=>workspace?.attachTile(key,tile),onRemove:key=>workspace?.removeTile(key),
    onFloat:(key,tile)=>workspace?.floatTile(key,tile),onPopup:onOpenViewer && openViewer,
    onSync:keys=>workspace?.pruneTiles(keys),onError,
    onChange:() => {
      if(viewerMode && observation.snapshot) {
        save(`viewer.${viewerId}`,JSON.stringify({targetId:observation.targetId,configuration:observation.configuration}));
        if(workspace && observation.selected.length === 0 && workspace.dock.panels.length === 0) workspace.show('registers');
      }
      lastRender = ''; renderSelected();
    }
  });

  function showError(error) { if(!signal.aborted) onError(error); }

  function initializeLayout() {
    workspace ??= new DebugWorkspace({root,t,read,save,onError,onPopup:onOpenViewer && openViewer,onCloseTile:key=>{
      const registers=observation.selected.filter(item=>item.key !== key);
      observation.configure({...observation.configuration,registers,enabled:registers.length > 0 && observation.configuration.enabled});
    },viewer:viewerMode,id:viewerId,hasWatches:Boolean(viewerBootstrap?.configuration?.registers?.length)});
  }

  async function selectTarget(id) {
    if(id === selectedTarget && (catalog.length || selectingTarget === id)) return;
    if(id) initializeLayout();
    if(id !== selectedTarget) {
      selectedRegister = ''; selectedInstruction = null; snapshot = undefined; clearImage();
      catalog = []; instructionLayout = [];
    }
    selectingTarget = selectedTarget = id;
    lastRender = '';
    renderSelected();
    const watching = observation.setTarget(id);
    try {
      if(id) {
        const instructions = await session.getInstructions(id,signal);
        if(!signal.aborted && selectedTarget === id) {
          catalog = instructions;
          instructionLayout = layoutInstructions(catalog);
          renderSelected();
        }
      }
    } catch(error) { showError(error); }
    await watching;
    if(selectingTarget === id) selectingTarget = '';
  }

  function openViewer(key) {
    onOpenViewer?.({
      targetId:selectedTarget,
      configuration:{...observation.configuration,registers:observation.selected.filter(item=>!key || item.key === key)},
      sample:currentSample(key)
    });
  }

  function showSample({title,blob,value}, hasWatches) {
    initializeLayout();
    const tile = document.createElement('figure'); tile.id = 'detached-sample';
    const caption = document.createElement('figcaption'); caption.textContent = title; tile.append(caption);
    if(blob instanceof Blob) {
      const link = document.createElement('a'); link.target = '_blank'; link.rel = 'noopener';
      const img = document.createElement('img'); img.alt = title; img.src = link.href = URL.createObjectURL(blob);
      lifetime.signal.addEventListener('abort',()=>URL.revokeObjectURL(img.src),{once:true});
      sampleImages.set(img,blob); link.append(img); tile.append(link);
    } else {
      const pre = document.createElement('pre'); pre.textContent = value; tile.append(pre);
    }
    setupSample(tile); $('#watch-gallery').prepend(tile);
    if(!hasWatches) {
      const registers = workspace.dock.getPanel('registers');
      if(registers) workspace.dock.removePanel(registers);
    }
    const firstWatch = workspace.dock.panels.find(panel=>panel.id.startsWith('watch:'));
    workspace.show('preview',firstWatch ? {referencePanel:firstWatch,direction:'above'} : undefined);
  }

  function translate() {
    root.querySelectorAll('[data-i18n]').forEach(element=>element.textContent=t(element.dataset.i18n));
    root.querySelectorAll('[data-label]').forEach(element=>element.setAttribute('aria-label',t(element.dataset.label)));
    root.querySelectorAll('[data-tip]').forEach(element=>element.dataset.tooltip=t(element.dataset.tip));
    updateCaptions(); observation.render(); workspace?.translate();
    lastRender = ''; renderSelected();
  }
  function flowDetails(instruction, layout) {
    const lines = layout.guards.map(({name,first,last}) => {
      const span = first === last ? `#${first}` : `#${first}–#${last}`;
      return `${t('guard')} · ${t(name ?? 'unnamedGuard')} · ${span} (${last - first + 1})`;
    });
    const flow = instruction.flow;
    if(flow?.jumps.length || flow?.canStop) {
      const targets = flow.jumps.map(jump => `→ ${jump.label} ${jump.index == null ? '' : '#' + jump.index}`.trimEnd());
      if(flow.canStop) targets.push(`■ ${t('flowStop')}`);
      if(flow.fallsThrough) targets.push(t('flowNext'));
      lines.push(targets.join(' · '));
    }
    if(layout.skipped) lines.push(`${t('flowSpan')} · ${layout.skipped}`);
    return lines;
  }

  function renderSelected() {
    if(signal.aborted) return;
    const paused = snapshot?.state === 'paused';
    $('#selected-target').textContent = selectedTarget || '—';
    const point = paused || snapshot?.state === 'waiting' ? snapshot?.location : null;
    $('#stop-location').textContent = point ? `${t('run')} ${point.runId} · ${point.program} [${point.index}] · ${t(point.phase)} ${point.name}` : snapshot ? t(snapshot.state) : '—';
    $('#stop-location').dataset.tooltip = $('#stop-location').textContent;
    root.querySelectorAll('[data-command]').forEach(button => {
      button.disabled = busy || !snapshot || (['step','continue','next-frame'].includes(button.dataset.command) && !paused);
    });
    const activeActionKey = document.activeElement?.dataset.focusKey;
    if(activeActionKey) focusedActionKey = activeActionKey;
    root.querySelectorAll('[data-register-action]').forEach(button => button.disabled = busy || !paused || button.dataset.present !== 'true');
    root.querySelectorAll('[data-run-to]').forEach(button => button.disabled = viewerMode || busy || !paused || button.dataset.atCurrent === 'true');
    $('#live').disabled = !snapshot;
    const renderKey = `${selectedTarget}|${snapshot?.revision}|${JSON.stringify(snapshot?.breakpoints)}|${JSON.stringify(snapshot?.runTo)}|${selectedRegister}|${selectedInstruction?.program}|${selectedInstruction?.index}|${catalog.length}|${observation.renderKey}`;
    if(renderKey === lastRender) { restoreRowFocus(); return; }
    lastRender = renderKey;
    let program = '';
    $('#instructions').innerHTML = catalog.map((instruction, position) => {
      let header = '';
      if(program !== instruction.program) { program = instruction.program; header = `<tr class="program-row"><td colspan="8">${esc(program)}</td></tr>`; }
      const layout = instructionLayout[position];
      const labels = (instruction.labels ?? []).join(', ');
      const indent = '<span class="instruction-indent" aria-hidden="true"></span>'.repeat(Math.min(layout.depth,12));
      const guard = layout.guards.length ? '<span class="instruction-guard" aria-hidden="true">? </span>' : '';
      const destinations = (instruction.flow?.jumps ?? []).map(jump => jump.label);
      if(instruction.flow?.canStop) destinations.push('■');
      const targets = destinations.length ? ` <span class="instruction-target">→ ${esc(destinations.join(', '))}</span>` : '';
      const instructionTip = esc([instruction.name,instruction.operation,...flowDetails(instruction,layout)].join('\n'));
      const endLabels = (instruction.endLabels ?? []).join(', ');
      const end = endLabels ? `<tr><td colspan="3"></td><td class="number">${instruction.index + 1}</td><td class="instruction-label" data-tooltip="${esc(endLabels)}">${esc(endLabels)}</td><td colspan="3">${t('flowEnd')}</td></tr>` : '';
      const here = point?.program === instruction.program && point.index === instruction.index;
      const selected = selectedInstruction?.program === instruction.program && selectedInstruction.index === instruction.index;
      const breakpoint = snapshot?.breakpoints.some(item => item.program === instruction.program && item.index === instruction.index);
      const capture = observation.capturePoint?.program === instruction.program && observation.capturePoint.index === instruction.index;
      const captureLabel = `${t('watchpoint')} ${program} [${instruction.index}] · ${t('after')} ${instruction.name}`;
      const timing = snapshot?.timings.find(item => item.program === instruction.program && item.index === instruction.index);
      const atCurrent = Boolean(here && point.phase === 'before');
      const target = snapshot?.runTo?.program === program && snapshot.runTo.index === instruction.index;
      const runLabel = `${t('runto')} ${program} [${instruction.index}] · ${instruction.name}`;
      const runButton = `<button class="row-action" data-run-to data-at-current="${atCurrent}" data-focus-key="${esc(selectedTarget + '|run|' + program + '|' + instruction.index)}" aria-label="${esc(runLabel)}" data-tooltip="${esc(runLabel + '\n' + t('runtotip'))}" ${busy || !paused || atCurrent ? 'disabled':''}><i class="ti ti-player-track-next" aria-hidden="true"></i></button>`;
      return header + `<tr data-program="${esc(program)}" data-index="${instruction.index}" class="${here ? 'current':''} ${selected ? 'selected':''} ${target ? 'run-target':''} ${capture ? 'capture-point':''}"><td><input type="checkbox" data-breakpoint aria-label="${esc(t('breakpoint') + ' ' + program + ' ' + instruction.index)}" ${breakpoint ? 'checked':''}></td><td class="row-action-cell">${runButton}</td><td class="instruction-pointer">${here ? `<i class="ti ti-arrow-right" aria-hidden="true"></i>`:''}</td><td class="number">${instruction.index}</td><td class="instruction-label" data-tooltip="${esc(labels)}">${esc(labels)}</td><td class="instruction-name" data-tooltip="${instructionTip}">${indent}${guard}${esc(instruction.name)}${targets}</td><td class="capture-cell"><input type="checkbox" data-watch-instruction data-focus-key="${esc(selectedTarget+'|capture|'+program+'|'+instruction.index)}" aria-label="${esc(captureLabel)}" data-tooltip="${esc(captureLabel+'\n'+t('watchpointtip'))}" ${capture ? 'checked':''} ${!observation.snapshot || observation.updating ? 'disabled':''}></td><td class="number">${timing ? number(timing.milliseconds,2):'—'}</td></tr>` + end;
    }).join('');
    $('#instruction-count').textContent = String(catalog.length);
    const active = selectedInstruction ?? catalog.find(item => item.program === point?.program && item.index === point?.index);
    $('#instruction-io').textContent = active ? `${t('reads')}: ${active.reads.join(', ') || '—'}\n${t('writes')}: ${active.writes.join(', ') || '—'}${flowDetails(active,instructionLayout[catalog.indexOf(active)]).map(line => '\n' + line).join('')}` : '—';
    const registerVariables = sortRegisters(observation.knownVariables(snapshot?.variables ?? [],paused),registerSort.column,registerSort.direction);
    $('#registers').innerHTML = registerVariables.length ? registerVariables.map(variable => {
      const action = (format, icon, label) => `<button class="row-action" data-present="${Boolean(variable.availableNow)}" data-register-action="${format}" data-focus-key="${esc(selectedTarget + '|register|' + variable.key + '|' + format)}" aria-label="${esc(t(label) + ' ' + variable.key)}" data-tooltip="${esc(t(label) + ' ' + variable.key)}" ${busy || !variable.availableNow ? 'disabled':''}><i class="ti ti-${icon}" aria-hidden="true"></i></button>`;
      const where=location=>location ? `${location.program} [${location.index}] ${t(location.phase)} · ${t('run')} ${location.runId}`:'';
      const detail=`${variable.key}\n${variable.type}\n${t(variable.presence ?? 'registermetadatatip')}\n${where(variable.lastSeen)}${variable.removedAt ? '\n'+t('removed')+': '+where(variable.removedAt):''}${variable.metadataError ? '\n'+variable.metadataError:''}`;
      const created=variable.createdAt, creatorDetail=created ? `${t('creator')}: ${where(created)} · ${created.name}`:t('creatorTip');
      const inspectAction=variable.canPreviewImage ? action('png','eye','imageaction'):variable.canInspectValue ? action('json','list-details','valueaction'):'';
      return `<tr tabindex="0" data-register="${esc(variable.key)}" data-memory-kind="${memoryKind(variable.memory)}" class="${selectedRegister === variable.key ? 'selected':''} ${active?.reads.includes(variable.key) ? 'input-register':''} ${active?.writes.includes(variable.key) ? 'output-register':''}"><td class="number register-creator" data-tooltip="${esc(creatorDetail)}">${created?.index ?? '—'}</td><td class="number register-order">${variable.additionOrder ?? '—'}</td><td data-register-name data-tooltip="${esc(detail)}">${esc(variable.key)}</td><td class="register-watch-cell">${observation.checkbox(variable)}</td><td class="register-action-cell">${inspectAction}</td><td>${esc(variable.memory)}</td><td class="mono" data-tooltip="${esc(variable.elementType ?? variable.type)}">${esc(variable.shape ?? variable.value ?? '—')}</td></tr>`;
    }).join('') : `<tr><td colspan="7" class="muted">${t('noRegisters')}</td></tr>`;
    $('#register-count').textContent = String(registerVariables.length) + (observation.catalogs.get(selectedTarget)?.truncated ? '+':'');
    $('#register-count').dataset.tooltip = t('registermetadatatip');
    if(viewerMode) root.querySelectorAll('[data-run-to],[data-breakpoint]').forEach(control=>control.disabled=true);
    for(const header of root.querySelectorAll('th[data-sort]')) header.setAttribute('aria-sort',header.dataset.sort !== registerSort.column ? 'none' : registerSort.direction === 1 ? 'ascending':'descending');
    restoreRowFocus();
  }

  function restoreRowFocus() {
    if(busy || document.activeElement !== document.body || !focusedActionKey) return;
    const button = [...root.querySelectorAll('[data-focus-key]')].find(item => item.dataset.focusKey === focusedActionKey);
    if(button && !button.disabled) button.focus({preventScroll:true});
  }

  async function command(name, target = {}) {
    if(viewerMode || !isActive()) return;
    if(busy || !selectedTarget) return;
    busy = true; renderSelected();
    try {
      const revision = ['step','continue','next-frame','run-to'].includes(name) ? snapshot?.revision:undefined;
      await session.sendCommand(selectedTarget,{command:name,revision,...target},signal);
      clearError();
    } catch(error) { showError(error); }
    finally { busy = false; await onRefresh(); }
  }
  $('.debug-actions').onclick = event => { const button = event.target.closest('[data-command]'); if(button) command(button.dataset.command); };
  $('#instructions').addEventListener('click', async event => {
    const row = event.target.closest('[data-program]'); if(!row) return;
    selectedInstruction = catalog.find(item => item.program === row.dataset.program && item.index === Number(row.dataset.index));
    if(event.target.closest('[data-run-to]')) {
      await command('run-to',{program:row.dataset.program,index:Number(row.dataset.index)});
      return;
    }
    if(event.target.matches('[data-watch-instruction]')) {
      await observation.setCapturePoint(selectedInstruction,event.target.checked);
    } else if(!viewerMode && event.target.matches('[data-breakpoint]')) {
      const breakpoints = [...$('#instructions').querySelectorAll('[data-breakpoint]:checked')].map(input => ({program:input.closest('tr').dataset.program,index:Number(input.closest('tr').dataset.index)}));
      try { await session.setBreakpoints(selectedTarget,breakpoints,signal); await onRefresh(); }
      catch(error) { showError(error); }
    }
    lastRender = ''; renderSelected();
  });
  $('#registers').addEventListener('click',event => {
    const row = event.target.closest('[data-register]'); if(!row) return;
    selectedRegister = row.dataset.register;
    if(event.target.matches('[data-watch-register]')) { lastRender = ''; return; }
    const action = event.target.closest('[data-register-action]');
    if(action) { inspect(action.dataset.registerAction,row.dataset.register); return; }
    lastRender = ''; renderSelected();
  });
  $('#registers').addEventListener('keydown',event => { if(event.key === 'Enter' && event.target.dataset.register) { selectedRegister = event.target.dataset.register; lastRender = ''; renderSelected(); } });

  function clearImage() {
    sampleImages.delete($('#preview-image')); sampleImages.delete($('#pinned-image'));
    for(const id of ['pinned-figure','preview-figure','value-figure']) {
      const panel = `watch:@preview:${id}`;
      if(workspace?.dock.getPanel(panel)) workspace.returnTile(panel);
    }
    if(imageUrl) URL.revokeObjectURL(imageUrl);
    if(pinnedUrl) URL.revokeObjectURL(pinnedUrl);
    imageUrl = pinnedUrl = imageInfo = pinnedInfo = null; $('#preview-figure').hidden = $('#pinned-figure').hidden = $('#value-figure').hidden = true;
    $('#image-empty').hidden = false; $('#pin').disabled = true;
  }
  function captionFor(info) {
    if(!info) return '';
    return info.result ? `${info.targetId} · ${t('resultframe')}` : `${info.targetId} · ${t('run')} ${info.runId} · ${info.key} · ${info.program} [${info.index}] ${t(info.phase)}`;
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
    sampleImages.set($('#preview-image'),blob);
    updateCaptions(); $('#preview-figure').hidden = false;
    $('#image-empty').hidden = $('#value-figure').hidden = true; $('#pin').disabled = false;
  }
  async function inspect(format, key) {
    const variable = snapshot?.variables.find(item => item.key === key);
    if(!snapshot || snapshot.state !== 'paused' || !variable || busy ||
      (format === 'json' ? !variable.canInspectValue : !variable.canPreviewImage)) return;
    busy = true; renderSelected();
    const targetId = selectedTarget;
    try {
      const location = snapshot.location;
      const value = await session.inspect(targetId,{key,format,revision:snapshot.revision,maxEdge:Number(observation.size.value)},signal);
      if(signal.aborted || targetId !== selectedTarget) return;
      if(format === 'png') displayImage(value,{targetId,key,...location});
      else { $('#value-figure figcaption span').textContent = key; $('#value-output').textContent = value; $('#value-figure').hidden = false; $('#image-empty').hidden = true; }
      clearError();
    } catch(error) { showError(error); }
    finally { busy = false; renderSelected(); }
  }
  $('#pin').onclick = () => {
    if(!imageUrl) return;
    if(pinnedUrl && pinnedUrl !== imageUrl) URL.revokeObjectURL(pinnedUrl);
    pinnedUrl = imageUrl; pinnedInfo = imageInfo; $('#pinned-image').src = pinnedUrl;
    sampleImages.set($('#pinned-image'),sampleImages.get($('#preview-image')));
    updateCaptions(); $('#pinned-figure').hidden = false;
  };
  $('#live').hidden = typeof session.getResultImage !== 'function';
  $('#live').onclick = async () => {
    try {
      const targetId = selectedTarget;
      const image = await session.getResultImage(targetId,signal);
      if(!signal.aborted && targetId === selectedTarget) displayImage(image,{targetId,result:true});
      clearError();
    }
    catch(error) { showError(error); }
  };
  function setupSample(tile) {
    tile.classList.add('preview-sample');
    const caption = tile.querySelector('figcaption');
    const title = document.createElement('span');
    if(caption.id) { title.id = caption.id; caption.removeAttribute('id'); }
    title.textContent = caption.textContent;
    caption.replaceChildren(title);
    const body = document.createElement('div'); body.className = 'preview-sample-body';
    for(const item of [...tile.children]) if(item !== caption) body.append(item);
    tile.append(body);
    const key = `@preview:${tile.id}`;
    const add = (label,tip,markup,action) => {
      const button = document.createElement('button'); button.className = 'pane-action';
      button.innerHTML = markup; button.dataset.label = label; button.dataset.tip = tip;
      button.setAttribute('aria-label',t(label)); button.dataset.tooltip = t(tip);
      button.onclick = action; caption.append(button); return button;
    };
    if(body.querySelector('img')) {
      const native = add('nativeview','nativeviewtip','1:1',() => {
        tile.dataset.native = String(tile.dataset.native !== 'true');
        native.setAttribute('aria-pressed',tile.dataset.native);
      });
      native.dataset.tileNative = ''; native.setAttribute('aria-pressed','false');
    }
    add('floatpanel','floatpaneltip','<i class="ti ti-maximize" aria-hidden="true"></i>',() => {
      workspace.floatTile(key,tile);
      const panelId = `watch:${key}`;
      const header = workspace.entry(panelId).header;
      header.querySelector('h2').textContent = title.textContent;
      header.querySelector('[data-workspace-action=nativeview]').hidden = !body.querySelector('img');
      workspace.dock.getPanel(panelId)?.api.setTitle(title.textContent);
    });
    if(onOpenViewer) add('newviewer','newviewertip','<i class="ti ti-external-link" aria-hidden="true"></i>',() => openViewer(key));
  }

  function currentSample(key) {
    const tile = key?.startsWith('@preview:') ? root.querySelector('#' + CSS.escape(key.slice(9))) :
      key ? observation.tiles.get(key) :
        ['preview-figure','pinned-figure','value-figure','detached-sample'].map(id=>$('#' + id)).find(item=>item && !item.hidden);
    if(!tile || tile.hidden) return null;
    const image = tile.querySelector('img');
    const value = tile.querySelector('pre')?.textContent;
    const title = [tile.querySelector('figcaption span')?.textContent,tile.querySelector('.watch-location')?.textContent].filter(Boolean).join(' · ');
    const blob = image && sampleImages.get(image);
    if(blob && !image.closest('a')?.hidden) return {title,blob};
    return value ? {title,value} : null;
  }

  $('.register-table thead').onclick = event => {
    const button = event.target.closest('button[data-sort]'); if(!button) return;
    registerSort = {column:button.dataset.sort,direction:registerSort.column === button.dataset.sort ? -registerSort.direction:1};
    save('registerSort',JSON.stringify(registerSort)); lastRender = ''; renderSelected();
  };
  const help = $('#register-help');
  help.hidden = !onHelp;
  help.onclick = () => onHelp?.();
  document.addEventListener('keydown',event => {
    if(!canUseShortcuts() || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
    if(event.key === 'F6') { event.preventDefault(); command('step'); }
    if(event.key === 'F8') { event.preventDefault(); command('continue'); }
  },{signal:lifetime.signal});
  for(const id of ['pinned-figure','preview-figure','value-figure']) setupSample($('#' + id));
  enablePreviewLayout($('#watch-gallery'),root);
  translate();

  return {
    selectTarget,
    update(nextSnapshot) { snapshot = nextSnapshot; renderSelected(); },
    get targetId() { return selectedTarget; },
    get needsInstructions() { return !catalog.length && !selectingTarget; },
    clearTarget() { return selectTarget(''); },
    translate,
    layout() { workspace?.layout(); },
    applyAppearance() { workspace?.applyAppearance(); },
    showPanel(id) { initializeLayout(); workspace.show(id); },
    resetLayout() { initializeLayout(); workspace.defaults(); },
    openViewer,
    showSample,
    dispose() {
      lifetime.abort(); observation.dispose(); clearImage(); workspace?.dispose();
      root.replaceChildren(); delete root.dataset.debuggerWorkspace;
    }
  };
}
