export class ObservationView {
  constructor({api,t,esc,onChange,onError,read,save}) {
    Object.assign(this,{api,t,esc,onChange,onError,read,save});
    this.camera = ''; this.cameraRevision = 0; this.snapshot = null; this.updating = false; this.polling = false;
    this.after = {}; this.frames = new Map(); this.urls = new Map(); this.variables = new Map();
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
    document.querySelector('#registers').addEventListener('change',event => {
      const checkbox = event.target.closest('[data-watch-register]');
      if(!checkbox) return;
      const key = checkbox.dataset.watchRegister;
      const registers = this.configuration.registers.filter(item => item.key !== key);
      if(checkbox.checked) registers.push({key,format:checkbox.dataset.format});
      this.configure({...this.configuration,enabled:registers.length > 0,registers});
    });
    this.schedule();
  }

  get configuration() {
    return this.pendingConfiguration ?? this.snapshot?.configuration ?? {enabled:false,framesPerSecond:Number(this.read('watchFps','3')),maxEdge:Number(this.read('watchSize','640')),registers:[]};
  }
  get renderKey() { return `${this.snapshot?.generation}|${this.updating}`; }
  get selected() { return this.configuration.registers; }

  knownVariables(variables,paused) {
    if(paused) this.variables.set(this.camera,variables);
    return paused ? variables : this.variables.get(this.camera) ?? [];
  }

  async setCamera(camera) {
    const cameraRevision = ++this.cameraRevision;
    this.camera = camera; this.snapshot = null; this.pendingConfiguration = null; this.queuedConfiguration = null; this.updating = false; this.after = {}; this.frames.clear();
    for(const url of this.urls.values()) URL.revokeObjectURL(url);
    this.urls.clear(); this.gallery.replaceChildren(); this.render();
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
      this.accept(snapshot); this.onChange();
    } catch(error) { if(camera === this.camera) this.onError(error); }
  }

  checkbox(variable) {
    if(!variable.canPreviewImage && !variable.canInspectValue) return '';
    const checked = this.selected.some(item => item.key === variable.key);
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
    } catch(error) { this.onError(error); }
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
    if(this.snapshot?.generation !== snapshot.generation) this.after = {};
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
    for(const tile of this.gallery.querySelectorAll('[data-watch-key]')) if(!keys.has(tile.dataset.watchKey)) tile.remove();
    for(const [key,url] of this.urls) if(!keys.has(key)) { URL.revokeObjectURL(url); this.urls.delete(key); this.frames.delete(key); }
    for(const selection of configuration.registers) {
      let tile = [...this.gallery.children].find(element => element.dataset.watchKey === selection.key);
      if(!tile) {
        tile = document.createElement('figure'); tile.className = 'watch-tile'; tile.dataset.watchKey = selection.key; tile.dataset.format = selection.format;
        tile.innerHTML = `<figcaption data-tooltip="${this.esc(selection.key)}">${this.esc(selection.key)}</figcaption><div class="watch-location mono"></div><div class="watch-content">${selection.format === 'png' ? '<a target="_blank" rel="noopener" hidden><img alt=""></a><span class="watch-placeholder"></span>':'<pre></pre>'}</div><div class="watch-counts"></div>`;
        this.gallery.append(tile);
      }
      const item = this.snapshot?.items.find(item => item.target.key === selection.key);
      const frame = this.frames.get(selection.key);
      const location = frame?.location ?? item?.target;
      const detail = tile.querySelector('.watch-location');
      detail.textContent = frame?.error ? this.t('watcherror') : location ? `${frame ? `${this.t('run')} ${location.runId} · `:''}[${location.index}] ${this.t(location.phase)}` : this.t('watchwaiting');
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
    if(this.polling || this.updating || !this.camera || !this.snapshot || !this.configuration.enabled || !this.selected.length || document.hidden) return;
    this.polling = true;
    const camera = this.camera, generation = this.snapshot.generation;
    try {
      const snapshot = await (await this.api(`/api/cameras/${encodeURIComponent(camera)}/watch/poll`,{method:'POST',body:JSON.stringify({generation,after:this.after})})).json();
      if(camera !== this.camera || this.updating || generation !== this.snapshot?.generation) return;
      this.accept(snapshot);
      for(const item of snapshot.items) {
        const frame = item.frame;
        if(!frame || frame.sequence <= (this.after[item.target.key] ?? 0)) continue;
        const tile = [...this.gallery.children].find(element => element.dataset.watchKey === item.target.key);
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
