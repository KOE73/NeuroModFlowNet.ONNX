export class ObservationView {
  constructor({api,t,esc,onChange,onError,read,save,onTile,onRemove,onFloat,onPopup,onSync,bootstrap}) {
    Object.assign(this,{api,t,esc,onChange,onError,read,save,onTile,onRemove,onFloat,onPopup,onSync,bootstrap});
    this.camera = ''; this.cameraRevision = 0; this.snapshot = null; this.updating = false; this.polling = false;
    this.after = {}; this.frames = new Map(); this.urls = new Map(); this.variables = new Map(); this.tiles = new Map();
    this.catalogs = new Map(); this.pointChoices = new Map(); this.catalogVersion = 0;
    this.gallery = document.querySelector('#watch-gallery');
    this.enabled = document.querySelector('#watch-enabled');
    this.frequency = document.querySelector('#watch-fps');
    this.size = document.querySelector('#watch-size');
    this.enabled.onchange = () => this.configure({...this.configuration,enabled:this.enabled.checked});
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
      const pointControl = event.target.closest('[data-watch-point]');
      if(pointControl) {
        const key=pointControl.dataset.watchPoint, point=JSON.parse(pointControl.value);
        this.pointChoices.set(`${this.camera}|${key}`,point);
        const variable=this.catalogs.get(this.camera)?.variables.find(variable=>variable.key === key);
        const selected=variable?.points?.find(item=>item.program === point[0] && item.index === point[1] && item.phase === point[2]);
        if(this.selected.some(item=>item.key === key) && selected) {
          const registers=this.selected.map(item=>item.key === key ? this.selection(variable,selected):item);
          this.configure({...this.configuration,registers});
        }
        this.onChange(); return;
      }
      const checkbox = event.target.closest('[data-watch-register]');
      if(!checkbox) return;
      const key = checkbox.dataset.watchRegister;
      const registers = this.configuration.registers.filter(item => item.key !== key);
      if(checkbox.checked) {
        const variable=this.catalogs.get(this.camera)?.variables.find(variable=>variable.key === key);
        registers.push(variable ? this.selection(variable,this.point(variable)):{key,format:checkbox.dataset.format});
      }
      this.configure({...this.configuration,enabled:registers.length > 0,registers});
    });
    this.schedule();
    setInterval(()=>this.pollCatalog(),1000);
  }

  get configuration() {
    return this.pendingConfiguration ?? this.snapshot?.configuration ?? {enabled:false,framesPerSecond:Number(this.read('watchFps','3')),maxEdge:Number(this.read('watchSize','640')),registers:[]};
  }
  get renderKey() { return `${this.snapshot?.generation}|${this.updating}|${this.catalogVersion}`; }
  get selected() { return this.configuration.registers; }

  knownVariables(variables,paused) {
    if(paused) this.variables.set(this.camera,variables);
    const catalog=this.catalogs.get(this.camera)?.variables;
    if(!catalog?.length) return (paused ? variables : this.variables.get(this.camera) ?? []).map(variable=>({...variable,availableNow:paused}));
    const actual=new Map(variables.map(variable=>[variable.key,variable]));
    return catalog.map(variable=>({...variable,...this.point(variable)?.variable,
      ...(paused ? actual.get(variable.key):{}),points:variable.points,
      presence:paused ? actual.has(variable.key) ? 'paused-present':'not-at-pause':variable.presence,
      lastSeen:variable.lastSeen,removedAt:variable.removedAt,
      availableNow:paused && actual.has(variable.key)}));
  }

  point(variable) {
    const selection=this.selected.find(item=>item.key === variable.key);
    const choice=this.pointChoices.get(`${this.camera}|${variable.key}`);
    const points=variable.points ?? [];
    return points.find(point=>choice ? point.program === choice[0] && point.index === choice[1] && point.phase === choice[2] :
      point.program === selection?.program && point.index === selection?.index && point.phase === selection?.phase)
      ?? points.findLast(point=>point.isWrite) ?? points[0];
  }

  selection(variable,point) {
    const metadata=point?.variable ?? variable;
    return {key:variable.key,format:metadata.canPreviewImage ? 'png':'json',
      ...(point ? {program:point.program,index:point.index,phase:point.phase}:{})};
  }

  pointSelector(variable) {
    const {esc,t}=this, selected=this.point(variable), points=variable.points ?? [];
    const label=`${t('watchpoint')} ${variable.key}`;
    const detail=selected ? `${selected.program} [${selected.index}] ${t(selected.phase)} · ${selected.name}`:t('watchpointtip');
    return `<select class="watch-point" data-watch-point="${esc(variable.key)}" data-focus-key="${esc(this.camera+'|point|'+variable.key)}" aria-label="${esc(label)}" data-tooltip="${esc(variable.key+'\n'+detail)}" ${!points.length || this.updating ? 'disabled':''}>${points.length ? points.map(point=>`<option value="${esc(JSON.stringify([point.program,point.index,point.phase]))}" ${point === selected ? 'selected':''} ${!point.variable.canPreviewImage && !point.variable.canInspectValue ? 'disabled':''}>${esc(`${point.program} [${point.index}] ${t(point.phase)} · ${variable.key} · ${point.name}`)}</option>`).join(''):'<option>—</option>'}</select>`;
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
      if(snapshot.generation === 0) {
        const frequency = Number(this.read('watchFps','3')), size = Number(this.read('watchSize','640'));
        // A fresh session has no observations yet; the first selection applies the viewer's saved defaults.
        if(Number.isFinite(frequency) && frequency >= 0.2 && frequency <= 15) snapshot.configuration.framesPerSecond = frequency;
        if([320,640,1000,1600,2000].includes(size)) snapshot.configuration.maxEdge = size;
      }
      if(snapshot.generation === 0 && this.bootstrap?.camera === camera && this.bootstrap.configuration?.registers?.length) {
        this.snapshot = snapshot;
        await this.configure(this.bootstrap.configuration);
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
    this.pendingConfiguration = configuration;
    if(this.updating) { this.queuedConfiguration = configuration; this.render(); this.onChange(); return; }
    this.updating = true; this.render(); this.onChange();
    try {
      const snapshot = await (await this.api(`/api/cameras/${encodeURIComponent(camera)}/watch`,{method:'PUT',body:JSON.stringify(configuration)})).json();
      if(cameraRevision === this.cameraRevision) this.accept(snapshot);
    } catch(error) {
      for(const selection of configuration.registers) this.pointChoices.delete(`${camera}|${selection.key}`);
      this.onError(error);
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
    this.enabled.disabled = !this.snapshot || configuration.registers.length === 0;
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
        tile.innerHTML = `<figcaption><span class="watch-title" data-tooltip="${this.esc(selection.key)}">${this.esc(selection.key)}</span><button class="pane-action" data-tile-float aria-label="${this.esc(this.t('floatpanel'))}" data-tooltip="${this.esc(this.t('floatpanel'))}"><i class="ti ti-maximize" aria-hidden="true"></i></button><button class="pane-action" data-tile-popup aria-label="${this.esc(this.t('newviewer'))}" data-tooltip="${this.esc(this.t('newviewer'))}"><i class="ti ti-external-link" aria-hidden="true"></i></button></figcaption><div class="watch-location mono"></div><div class="watch-content">${selection.format === 'png' ? '<a target="_blank" rel="noopener" hidden><img alt=""></a><span class="watch-placeholder"></span>':'<pre></pre>'}</div><div class="watch-counts"></div>`;
        const created = tile;
        tile.querySelector('[data-tile-float]').onclick = () => this.onFloat(selection.key,created);
        tile.querySelector('[data-tile-popup]').onclick = () => this.onPopup(selection.key);
        this.tiles.set(selection.key,tile);
        this.gallery.append(tile);
        this.onTile(selection.key,tile);
      }
      tile.dataset.watchOrder = configuration.registers.indexOf(selection);
      if(tile.dataset.generation !== String(this.snapshot?.generation) || tile.dataset.format !== selection.format) {
        tile.dataset.generation=String(this.snapshot?.generation); tile.dataset.format=selection.format;
        tile.querySelector('.watch-content').innerHTML=selection.format === 'png' ? '<a target="_blank" rel="noopener" hidden><img alt=""></a><span class="watch-placeholder"></span>':'<pre></pre>';
      }
      const item = this.snapshot?.items.find(item => item.target.key === selection.key);
      const frame = this.frames.get(selection.key);
      const location = frame?.location ?? item?.target;
      const detail = tile.querySelector('.watch-location');
      detail.textContent = frame?.error ? this.t('watcherror') : location ? `${frame ? `${this.t('run')} ${location.runId} · `:''}${location.program} [${location.index}] ${this.t(location.phase)}` : this.t('watchwaiting');
      detail.classList.toggle('error',Boolean(frame?.error));
      detail.dataset.tooltip = frame?.error ?? (location ? `${this.camera} · ${location.program} [${location.index}] ${this.t(location.phase)}${frame ? ` · ${this.t('run')} ${location.runId} · ${new Date(frame.capturedAt).toLocaleTimeString()}`:''}\n${this.t(configuration.enabled ? 'watch':'watchoff')}` : this.t('watchtip'));
      const placeholder = tile.querySelector('.watch-placeholder');
      if(placeholder) { placeholder.hidden = Boolean(frame?.available); placeholder.textContent = this.t(configuration.enabled ? 'watchwaiting':'watchoff'); }
      tile.querySelector('.watch-counts').textContent = `${this.t('watchreceived')} ${item?.captured ?? 0} · ${this.t('watchskipped')} ${item?.skipped ?? 0}`;
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
    try {
      const snapshot = await (await this.api(`/api/cameras/${encodeURIComponent(camera)}/watch/poll`,{method:'POST',body:JSON.stringify({generation,after:this.after})})).json();
      if(camera !== this.camera || this.updating || generation !== this.snapshot?.generation) return;
      this.accept(snapshot);
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
            if(camera !== this.camera || snapshot.generation !== this.snapshot?.generation) { URL.revokeObjectURL(url); return; }
            const previous = this.urls.get(item.target.key); if(previous) URL.revokeObjectURL(previous);
            this.urls.set(item.target.key,url); link.href = url; link.hidden = false;
            image.alt = `${camera} · ${item.target.key} · ${this.t('run')} ${frame.location.runId}`;
          } catch(error) { URL.revokeObjectURL(url); frame.error = error.message; }
        } else if(frame.data && frame.contentType === 'application/json') {
          tile.querySelector('pre').textContent = new TextDecoder().decode(Uint8Array.from(atob(frame.data),character => character.charCodeAt(0)));
        }
        this.frames.set(item.target.key,{...frame,data:undefined,available:Boolean(frame.data) && !frame.error}); this.after[item.target.key] = frame.sequence;
      }
      this.render();
    } catch(error) { if(camera === this.camera) this.onError(error); }
    finally { this.polling = false; }
  }

  schedule() {
    setTimeout(async () => { await this.poll(); this.schedule(); },Math.max(50,Math.min(500,500 / this.configuration.framesPerSecond)));
  }
}
