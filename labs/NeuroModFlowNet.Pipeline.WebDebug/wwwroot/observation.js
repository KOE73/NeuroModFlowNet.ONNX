export function sharedCapturePoint(configuration) {
  const point=configuration?.registers?.[0];
  if(!configuration?.enabled || !point?.program || !Number.isInteger(point.index) || point.phase !== 'after' ||
    !configuration.registers.every(item=>item.program === point.program && item.index === point.index && item.phase === point.phase)) return null;
  return {program:point.program,index:point.index,phase:point.phase};
}

export class ObservationView {
  constructor({api,t,esc,onChange,onError,read,save,onTile,onRemove,onFloat,onPopup,onSync,bootstrap,viewerId}) {
    Object.assign(this,{api,t,esc,onChange,onError,read,save,onTile,onRemove,onFloat,onPopup,onSync,bootstrap,viewerId});
    this.camera = ''; this.cameraRevision = 0; this.snapshot = null; this.updating = false; this.polling = false;
    this.after = {}; this.frames = new Map(); this.urls = new Map(); this.variables = new Map(); this.tiles = new Map();
    this.catalogs = new Map(); this.capturePoints = new Map(); this.catalogVersion = 0;
    this.configurationRevision = 0;
    this.gallery = document.querySelector('#watch-gallery');
    this.enabled = document.querySelector('#watch-enabled');
    this.frequency = document.querySelector('#watch-fps');
    this.size = document.querySelector('#watch-size');
    this.enabled.onchange = () => this.configure({...this.configuration,enabled:this.enabled.checked && Boolean(this.capturePoint)});
    this.frequency.onchange = () => {
      if(!this.frequency.reportValidity()) return;
      this.save('watchFps',this.frequency.value);
      this.configure({...this.configuration,framesPerSecond:Number(this.frequency.value)});
    };
    this.size.onchange = () => {
      this.save('watchSize',this.size.value);
      this.configure({...this.configuration,maxEdge:Number(this.size.value)});
    };
    document.querySelector('.register-pane').addEventListener('change',event => {
      const checkbox = event.target.closest('[data-watch-register]');
      if(!checkbox) return;
      const key = checkbox.dataset.watchRegister;
      const registers = this.configuration.registers.filter(item => item.key !== key);
      if(checkbox.checked) {
        const variable=this.catalogs.get(this.camera)?.variables.find(variable=>variable.key === key);
        registers.push(variable ? this.selection(variable):{key,format:checkbox.dataset.format,...this.capturePoint});
      }
      this.configure({...this.configuration,enabled:registers.length > 0 && Boolean(this.capturePoint),registers});
    });
    this.schedule();
    setInterval(()=>this.pollCatalog(),1000);
  }

  get configuration() {
    return this.pendingConfiguration ?? this.snapshot?.configuration ?? {enabled:false,framesPerSecond:Number(this.read('watchFps','3')),maxEdge:Number(this.read('watchSize','640')),registers:[]};
  }
  get renderKey() { return `${this.snapshot?.generation}|${this.updating}|${this.catalogVersion}`; }
  get selected() { return this.configuration.registers; }
  get capturePoint() { return this.capturePoints.get(this.camera) ?? null; }

  async setCapturePoint(instruction,checked) {
    const camera=this.camera;
    const previous=this.capturePoint;
    const point=checked ? {program:instruction.program,index:instruction.index,phase:'after'}:null;
    this.capturePoints.set(this.camera,point);
    this.save(`capturePoint.${this.viewerId}.${this.camera}`,JSON.stringify(point));
    const registers=this.selected.map(({key,format})=>({key,format,...point}));
    if(await this.configure({...this.configuration,enabled:Boolean(point) && registers.length > 0,registers}) === false) {
      this.capturePoints.set(camera,previous);
      this.save(`capturePoint.${this.viewerId}.${camera}`,JSON.stringify(previous));
      this.onChange();
    }
  }

  knownVariables(variables,paused) {
    if(paused) this.variables.set(this.camera,variables);
    const catalog=this.catalogs.get(this.camera)?.variables;
    if(!catalog?.length) return (paused ? variables : this.variables.get(this.camera) ?? []).map(variable=>({...variable,availableNow:paused}));
    const actual=new Map(variables.map(variable=>[variable.key,variable]));
    return catalog.map(variable=>({...variable,...this.point(variable)?.variable,
      ...(paused ? actual.get(variable.key):{}),points:variable.points,
      presence:paused ? actual.has(variable.key) ? 'paused-present':'not-at-pause':variable.presence,
      lastSeen:variable.lastSeen,removedAt:variable.removedAt,additionOrder:variable.additionOrder,createdAt:variable.createdAt,
      availableNow:paused && actual.has(variable.key)}));
  }

  point(variable) {
    const selection=this.capturePoint;
    const points=variable.points ?? [];
    return points.find(point=>point.program === selection?.program && point.index === selection?.index && point.phase === selection?.phase);
  }

  selection(variable) {
    const metadata=this.point(variable)?.variable ?? variable;
    return {key:variable.key,format:metadata.canPreviewImage ? 'png':'json',
      ...this.capturePoint};
  }

  async pollCatalog() {
    if(this.catalogPolling || !this.camera || document.hidden || document.querySelector('#shell').dataset.page !== 'camera') return;
    this.catalogPolling=true;
    const camera=this.camera, cameraRevision=this.cameraRevision;
    try {
      const catalog=await (await this.api(`/api/cameras/${encodeURIComponent(camera)}/registers/poll`,{method:'POST'})).json();
      if(cameraRevision !== this.cameraRevision) return;
      this.catalogs.set(camera,catalog); this.catalogVersion++; this.onChange();
    } catch(error) { if(cameraRevision === this.cameraRevision) this.onError(error); }
    finally { this.catalogPolling=false; }
  }

  async setCamera(camera) {
    const cameraRevision = ++this.cameraRevision;
    this.camera = camera; this.snapshot = null; this.pendingConfiguration = null; this.queuedConfiguration = null; this.updating = false; this.after = {}; this.frames.clear();
    for(const url of this.urls.values()) URL.revokeObjectURL(url);
    this.urls.clear(); for(const key of this.tiles.keys()) this.onRemove(key); this.tiles.clear(); this.gallery.replaceChildren(); this.render();
    if(!camera) return;
    try {
      const snapshot = await (await this.api(`/api/cameras/${encodeURIComponent(camera)}/watch`)).json();
      if(cameraRevision !== this.cameraRevision) return;
      const initial=this.bootstrap?.camera === camera ? this.bootstrap.configuration:snapshot.configuration;
      const savedPoint=this.read(`capturePoint.${this.viewerId}.${camera}`,'');
      let point=null;
      try { point=sharedCapturePoint(initial) ?? (savedPoint ? JSON.parse(savedPoint):initial?.registers.findLast(item=>item.program)); } catch { }
      point=point?.program && Number.isInteger(point.index) ? {program:point.program,index:point.index,phase:'after'}:null;
      this.capturePoints.set(camera,point);
      this.save(`capturePoint.${this.viewerId}.${camera}`,JSON.stringify(point));
      if(snapshot.generation === 0) {
        const frequency = Number(this.read('watchFps','3')), size = Number(this.read('watchSize','640'));
        // A fresh session has no observations yet; the first selection applies the viewer's saved defaults.
        if(Number.isFinite(frequency) && frequency >= 0.2 && frequency <= 15) snapshot.configuration.framesPerSecond = frequency;
        if(size === 0 || Number.isInteger(size) && size >= 100 && size <= 8192) snapshot.configuration.maxEdge = size;
      }
      if(snapshot.generation === 0 && this.bootstrap?.camera === camera && this.bootstrap.configuration?.registers?.length) {
        this.snapshot = snapshot;
        await this.configure({...this.bootstrap.configuration,enabled:this.bootstrap.configuration.enabled && Boolean(point),registers:this.bootstrap.configuration.registers.map(({key,format})=>({key,format,...point}))});
      } else if(snapshot.configuration.registers.length && snapshot.configuration.registers.some(item=>!point || item.program !== point.program || item.index !== point.index || item.phase !== point.phase)) {
        this.snapshot=snapshot;
        await this.configure({...snapshot.configuration,enabled:snapshot.configuration.enabled && Boolean(point),registers:snapshot.configuration.registers.map(({key,format})=>({key,format,...point}))});
      } else { this.accept(snapshot); this.onChange(); }
      this.pollCatalog();
    } catch(error) { if(camera === this.camera) this.onError(error); }
  }

  checkbox(variable) {
    const checked = this.selected.some(item => item.key === variable.key);
    if(!checked && !variable.canPreviewImage && !variable.canInspectValue) return '';
    const label = `${this.t('watchregister')} ${variable.key}`;
    return `<input type="checkbox" data-watch-register="${this.esc(variable.key)}" data-format="${variable.canPreviewImage ? 'png':'json'}" data-focus-key="${this.esc(this.camera + '|watch|' + variable.key)}" aria-label="${this.esc(label)}" data-tooltip="${this.esc(label)}" data-tooltip-passive ${checked ? 'checked':''} ${!this.snapshot || (!checked && this.selected.length >= 8) ? 'disabled':''}>`;
  }

  async configure(configuration) {
    if(!this.camera) return;
    const camera = this.camera, cameraRevision = this.cameraRevision;
    this.configurationRevision++;
    this.pendingConfiguration = configuration;
    if(this.updating) { this.queuedConfiguration = configuration; this.render(); this.onChange(); return; }
    this.updating = true; this.render(); this.onChange();
    try {
      const snapshot = await (await this.api(`/api/cameras/${encodeURIComponent(camera)}/watch`,{method:'PUT',body:JSON.stringify(configuration)})).json();
      if(cameraRevision === this.cameraRevision) this.accept(snapshot);
      return true;
    } catch(error) {
      this.onError(error);
      return false;
    }
    finally {
      if(cameraRevision === this.cameraRevision) {
        this.updating = false;
        const queued = this.queuedConfiguration; this.queuedConfiguration = null;
        if(queued) await this.configure(queued);
        else { this.pendingConfiguration = null; this.render(); this.onChange(); }
      }
    }
  }

  accept(snapshot) {
    if(this.snapshot && snapshot.generation < this.snapshot.generation) return;
    const point=sharedCapturePoint(snapshot.configuration);
    if(point) {
      this.capturePoints.set(this.camera,point);
      this.save(`capturePoint.${this.viewerId}.${this.camera}`,JSON.stringify(point));
    }
    if(snapshot.variables) this.variables.set(this.camera,snapshot.variables);
    if(this.snapshot?.generation !== snapshot.generation) {
      this.after = {}; this.frames.clear();
      for(const url of this.urls.values()) URL.revokeObjectURL(url);
      this.urls.clear();
    }
    this.snapshot = snapshot; this.render();
  }

  render() {
    const configuration = this.configuration;
    this.enabled.checked = configuration.enabled;
    this.enabled.disabled = !this.snapshot || configuration.registers.length === 0 || !this.capturePoint;
    for(const control of [this.frequency,this.size]) control.disabled = !this.snapshot;
    if(document.activeElement !== this.frequency) this.frequency.value = configuration.framesPerSecond;
    if(document.activeElement !== this.size) this.size.value = configuration.maxEdge;
    const keys = new Set(configuration.registers.map(item => item.key));
    if(this.snapshot) this.onSync(keys);
    for(const [key,tile] of this.tiles) if(!keys.has(key)) { this.onRemove(key); tile.remove(); this.tiles.delete(key); }
    for(const [key,url] of this.urls) if(!keys.has(key)) { URL.revokeObjectURL(url); this.urls.delete(key); this.frames.delete(key); }
    for(const selection of configuration.registers) {
      let tile = this.tiles.get(selection.key);
      if(!tile) {
        tile = document.createElement('figure'); tile.className = 'watch-tile'; tile.dataset.watchKey = selection.key; tile.dataset.format = selection.format;
        tile.innerHTML = `<figcaption><span class="watch-title" data-tooltip="${this.esc(selection.key)}">${this.esc(selection.key)}</span><button class="pane-action" data-tile-native aria-pressed="false" aria-label="${this.esc(this.t('nativeview'))}" data-tooltip="${this.esc(this.t('nativeviewtip'))}">1:1</button><button class="pane-action" data-tile-float aria-label="${this.esc(this.t('floatpanel'))}" data-tooltip="${this.esc(this.t('floatpanel'))}"><i class="ti ti-maximize" aria-hidden="true"></i></button><button class="pane-action" data-tile-popup aria-label="${this.esc(this.t('newviewer'))}" data-tooltip="${this.esc(this.t('newviewer'))}"><i class="ti ti-external-link" aria-hidden="true"></i></button></figcaption><div class="watch-location mono"></div><div class="watch-content">${selection.format === 'png' ? '<a target="_blank" rel="noopener" hidden><img alt=""></a><span class="watch-placeholder"></span>':'<pre></pre>'}</div><div class="watch-counts"></div>`;
        const created = tile;
        created.dataset.native=this.read(`nativeView.${this.viewerId}.${selection.key}`,'false');
        tile.querySelector('[data-tile-native]').onclick = () => {
          created.dataset.native=String(created.dataset.native !== 'true');
          this.save(`nativeView.${this.viewerId}.${selection.key}`,created.dataset.native);
          this.render();
        };
        tile.querySelector('[data-tile-float]').onclick = () => this.onFloat(selection.key,created);
        tile.querySelector('[data-tile-popup]').onclick = () => this.onPopup(selection.key);
        this.tiles.set(selection.key,tile);
        this.gallery.append(tile);
        this.onTile(selection.key,tile);
      }
      tile.dataset.watchOrder = configuration.registers.indexOf(selection);
      tile.querySelector('[data-tile-native]').hidden=selection.format !== 'png';
      tile.querySelector('[data-tile-native]').setAttribute('aria-pressed',tile.dataset.native === 'true');
      for(const control of document.querySelectorAll('[data-workspace-action=nativeview]')) if(control.dataset.panelId === `watch:${selection.key}`) {
        control.hidden=selection.format !== 'png'; control.setAttribute('aria-pressed',tile.dataset.native === 'true');
        control.dataset.tooltip=this.t('nativeviewtip');
      }
      if(tile.dataset.generation !== String(this.snapshot?.generation) || tile.dataset.format !== selection.format) {
        tile.dataset.generation=String(this.snapshot?.generation); tile.dataset.format=selection.format;
        tile.querySelector('.watch-content').innerHTML=selection.format === 'png' ? '<a target="_blank" rel="noopener" hidden><img alt=""></a><span class="watch-placeholder"></span>':'<pre></pre>';
      }
      const item = this.snapshot?.items.find(item => item.target.key === selection.key);
      const frame = this.frames.get(selection.key);
      const location = frame?.location ?? item?.target;
      const detail = tile.querySelector('.watch-location');
      const missing=frame?.error?.includes('not present');
      detail.textContent = frame?.error ? this.t(missing ? 'watchmissing':'watcherror') : location ? `${frame ? `${this.t('run')} ${location.runId} · `:''}${location.program} [${location.index}] ${this.t(location.phase)}` : this.t('watchwaiting');
      detail.classList.toggle('error',Boolean(frame?.error));
      detail.dataset.tooltip = frame?.error ?? (location ? `${this.camera} · ${location.program} [${location.index}] ${this.t(location.phase)}${frame ? ` · ${this.t('run')} ${location.runId} · ${new Date(frame.capturedAt).toLocaleTimeString()}`:''}\n${this.t(configuration.enabled ? 'watch':'watchoff')}` : this.t('watchtip'));
      const placeholder = tile.querySelector('.watch-placeholder');
      const imageLink = tile.querySelector('.watch-content a');
      if(imageLink) imageLink.hidden = !frame?.available;
      const value = tile.querySelector('.watch-content pre');
      if(value && frame?.error) value.textContent = '';
      if(placeholder) { placeholder.hidden = Boolean(frame?.available); placeholder.textContent = this.t(!this.capturePoint ? 'watchchoosepoint':missing ? 'watchmissing':configuration.enabled ? 'watchwaiting':'watchoff'); }
      tile.querySelector('.watch-counts').textContent = `${frame?.width ? `${frame.width}×${frame.height} · `:''}${this.t('watchreceived')} ${item?.captured ?? 0} · ${this.t('watchskipped')} ${item?.skipped ?? 0}`;
      tile.querySelector('.watch-counts').dataset.tooltip = `${tile.querySelector('.watch-counts').textContent}\n${this.t('watchcountertip')}`;
    }
    this.gallery.hidden = keys.size === 0;
    if(!document.querySelector('#preview-figure').hidden || !document.querySelector('#value-output').hidden) return;
    document.querySelector('#image-empty').hidden = keys.size > 0;
  }

  async poll() {
    if(this.polling || this.updating || !this.camera || !this.snapshot || !this.configuration.enabled || !this.selected.length || document.hidden || document.querySelector('#shell').dataset.page !== 'camera') return;
    this.polling = true;
    const camera = this.camera, generation = this.snapshot.generation;
    const cameraRevision=this.cameraRevision, configurationRevision=this.configurationRevision;
    const current=()=>cameraRevision === this.cameraRevision && configurationRevision === this.configurationRevision &&
      camera === this.camera && generation === this.snapshot?.generation && !this.updating;
    try {
      const snapshot = await (await this.api(`/api/cameras/${encodeURIComponent(camera)}/watch/poll`,{method:'POST',body:JSON.stringify({generation,after:this.after})})).json();
      if(!current()) return;
      this.accept(snapshot);
      if(!current()) return;
      for(const item of snapshot.items) {
        const frame = item.frame;
        if(!frame || frame.sequence <= (this.after[item.target.key] ?? 0)) continue;
        const tile = this.tiles.get(item.target.key);
        if(!tile) continue;
        if(frame.data && frame.contentType === 'image/png') {
          const bytes = Uint8Array.from(atob(frame.data),character => character.charCodeAt(0));
          const url = URL.createObjectURL(new Blob([bytes],{type:'image/png'}));
          const image = tile.querySelector('img'), link = tile.querySelector('a');
          try {
            image.src = url; await image.decode();
            if(!current()) { URL.revokeObjectURL(url); return; }
            const previous = this.urls.get(item.target.key); if(previous) URL.revokeObjectURL(previous);
            this.urls.set(item.target.key,url); link.href = url; link.hidden = false;
            image.alt = `${camera} · ${item.target.key} · ${this.t('run')} ${frame.location.runId}`;
            frame.width=image.naturalWidth; frame.height=image.naturalHeight;
          } catch(error) { URL.revokeObjectURL(url); if(!current()) return; frame.error = error.message; }
        } else if(frame.data && frame.contentType === 'application/json') {
          tile.querySelector('pre').textContent = new TextDecoder().decode(Uint8Array.from(atob(frame.data),character => character.charCodeAt(0)));
        }
        if(!current()) return;
        this.frames.set(item.target.key,{...frame,data:undefined,available:Boolean(frame.data) && !frame.error}); this.after[item.target.key] = frame.sequence;
      }
      this.render();
    } catch(error) { if(current()) this.onError(error); }
    finally { this.polling = false; }
  }

  schedule() {
    setTimeout(async () => { await this.poll(); this.schedule(); },Math.max(50,Math.min(500,500 / this.configuration.framesPerSecond)));
  }
}
