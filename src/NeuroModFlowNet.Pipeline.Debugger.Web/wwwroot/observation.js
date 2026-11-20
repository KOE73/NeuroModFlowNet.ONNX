import { RegisterVideoView } from './register-video.js';
import { sampleImages } from './preview-layout.js';

export function sharedCapturePoint(configuration) {
  const point=configuration?.registers?.[0];
  if(!configuration?.enabled || !point?.program || !Number.isInteger(point.index) || point.phase !== 'after' ||
    !configuration.registers.every(item=>item.program === point.program && item.index === point.index && item.phase === point.phase)) return null;
  return {program:point.program,index:point.index,phase:point.phase};
}

export class ObservationView {
  constructor({root,session,signal,isActive,t,esc,onChange,onError,read,save,onTile,onRemove,onFloat,onPopup,onSync,bootstrap,viewerId}) {
    Object.assign(this,{root,session,signal,isActive,t,esc,onChange,onError,read,save,onTile,onRemove,onFloat,onPopup,onSync,bootstrap,viewerId});
    const query = selector => root.querySelector(selector);
    this.targetId = ''; this.targetRevision = 0; this.snapshot = null; this.updating = false; this.polling = false;
    this.after = {}; this.frames = new Map(); this.urls = new Map(); this.variables = new Map(); this.tiles = new Map();
    this.catalogs = new Map(); this.capturePoints = new Map(); this.catalogVersion = 0;
    this.configurationRevision = 0;
    this.videos = new Map();
    this.gallery = query('#watch-gallery');
    this.enabled = query('#watch-enabled');
    this.frequency = query('#watch-fps');
    this.size = query('#watch-size');
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
    query('.register-pane').addEventListener('change',event => {
      const checkbox = event.target.closest('[data-watch-register]');
      if(!checkbox) return;
      const key = checkbox.dataset.watchRegister;
      const registers = this.configuration.registers.filter(item => item.key !== key);
      if(checkbox.checked) {
        const variable=this.catalogs.get(this.targetId)?.variables.find(variable=>variable.key === key);
        registers.push(variable ? this.selection(variable):{key,format:checkbox.dataset.format,...this.capturePoint});
      }
      this.configure({...this.configuration,enabled:registers.length > 0 && Boolean(this.capturePoint),registers});
    });
    this.schedule();
    this.catalogTimer = setInterval(()=>this.pollCatalog(),1000);
  }

  dispose() {
    this.stopVideos();
    this.disposed = true;
    this.targetRevision++;
    clearInterval(this.catalogTimer);
    clearTimeout(this.watchTimer);
    for(const url of this.urls.values()) URL.revokeObjectURL(url);
    this.urls.clear();
  }

  get configuration() {
    return this.pendingConfiguration ?? this.snapshot?.configuration ?? {enabled:false,framesPerSecond:Number(this.read('watchFps','3')),maxEdge:Number(this.read('watchSize','640')),registers:[]};
  }
  get renderKey() { return `${this.snapshot?.generation}|${this.updating}|${this.catalogVersion}`; }
  get selected() { return this.configuration.registers; }
  get capturePoint() { return this.capturePoints.get(this.targetId) ?? null; }

  async setCapturePoint(instruction,checked) {
    const targetId=this.targetId;
    const previous=this.capturePoint;
    const point=checked ? {program:instruction.program,index:instruction.index,phase:'after'}:null;
    this.capturePoints.set(this.targetId,point);
    this.save(`capturePoint.${this.viewerId}.${this.targetId}`,JSON.stringify(point));
    const registers=this.selected.map(({key,format})=>({key,format,...point}));
    if(await this.configure({...this.configuration,enabled:Boolean(point) && registers.length > 0,registers}) === false) {
      this.capturePoints.set(targetId,previous);
      this.save(`capturePoint.${this.viewerId}.${targetId}`,JSON.stringify(previous));
      this.onChange();
    }
  }

  knownVariables(variables,paused) {
    if(paused) this.variables.set(this.targetId,variables);
    const catalog=this.catalogs.get(this.targetId)?.variables;
    if(!catalog?.length) return (paused ? variables : this.variables.get(this.targetId) ?? []).map(variable=>({...variable,availableNow:paused}));
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
    if(this.catalogPolling || !this.targetId || document.hidden || !this.isActive()) return;
    this.catalogPolling=true;
    const targetId=this.targetId, targetRevision=this.targetRevision;
    try {
      const catalog=await this.session.pollRegisterCatalog(targetId,this.signal);
      if(targetRevision !== this.targetRevision) return;
      this.catalogs.set(targetId,catalog); this.catalogVersion++; this.onChange();
    } catch(error) { if(targetRevision === this.targetRevision) this.onError(error); }
    finally { this.catalogPolling=false; }
  }

  async setTarget(targetId) {
    this.stopVideos();
    const targetRevision = ++this.targetRevision;
    this.targetId = targetId; this.snapshot = null; this.pendingConfiguration = null; this.queuedConfiguration = null; this.updating = false; this.after = {}; this.frames.clear();
    for(const url of this.urls.values()) URL.revokeObjectURL(url);
    this.urls.clear();
    for(const [key,tile] of this.tiles) { this.onRemove(key); tile.remove(); }
    this.tiles.clear(); this.render();
    if(!targetId) return;
    try {
      const snapshot = await this.session.getObservation(targetId,this.signal);
      if(targetRevision !== this.targetRevision) return;
      const initial=(this.bootstrap?.targetId) === targetId ? this.bootstrap.configuration:snapshot.configuration;
      const savedPoint=this.read(`capturePoint.${this.viewerId}.${targetId}`,'');
      let point=null;
      try { point=sharedCapturePoint(initial) ?? (savedPoint ? JSON.parse(savedPoint):initial?.registers.findLast(item=>item.program)); } catch { }
      point=point?.program && Number.isInteger(point.index) ? {program:point.program,index:point.index,phase:'after'}:null;
      this.capturePoints.set(targetId,point);
      this.save(`capturePoint.${this.viewerId}.${targetId}`,JSON.stringify(point));
      if(snapshot.generation === 0) {
        const frequency = Number(this.read('watchFps','3')), size = Number(this.read('watchSize','640'));
        // A fresh session has no observations yet; the first selection applies the viewer's saved defaults.
        if(Number.isFinite(frequency) && frequency >= 0.2 && frequency <= 15) snapshot.configuration.framesPerSecond = frequency;
        if(size === 0 || Number.isInteger(size) && size >= 100 && size <= 8192) snapshot.configuration.maxEdge = size;
      }
      if(snapshot.generation === 0 && (this.bootstrap?.targetId) === targetId && this.bootstrap.configuration?.registers?.length) {
        this.snapshot = snapshot;
        await this.configure({...this.bootstrap.configuration,enabled:this.bootstrap.configuration.enabled && Boolean(point),registers:this.bootstrap.configuration.registers.map(({key,format})=>({key,format,...point}))});
      } else if(snapshot.configuration.registers.length && snapshot.configuration.registers.some(item=>!point || item.program !== point.program || item.index !== point.index || item.phase !== point.phase)) {
        this.snapshot=snapshot;
        await this.configure({...snapshot.configuration,enabled:snapshot.configuration.enabled && Boolean(point),registers:snapshot.configuration.registers.map(({key,format})=>({key,format,...point}))});
      } else { this.accept(snapshot); this.onChange(); }
      this.pollCatalog();
    } catch(error) { if(targetRevision === this.targetRevision) this.onError(error); }
  }

  checkbox(variable) {
    const checked = this.selected.some(item => item.key === variable.key);
    if(!checked && !variable.canPreviewImage && !variable.canInspectValue) return '';
    const label = `${this.t('watchregister')} ${variable.key}`;
    return `<input type="checkbox" data-watch-register="${this.esc(variable.key)}" data-format="${variable.canPreviewImage ? 'png':'json'}" data-focus-key="${this.esc(this.targetId + '|watch|' + variable.key)}" aria-label="${this.esc(label)}" data-tooltip="${this.esc(label)}" data-tooltip-passive ${checked ? 'checked':''} ${!this.snapshot || (!checked && this.selected.length >= 8) ? 'disabled':''}>`;
  }

  async configure(configuration) {
    if(!this.targetId) return;
    const targetId = this.targetId, targetRevision = this.targetRevision;
    this.stopVideos();
    this.configurationRevision++;
    this.pendingConfiguration = configuration;
    if(this.updating) { this.queuedConfiguration = configuration; this.render(); this.onChange(); return; }
    this.updating = true; this.render(); this.onChange();
    try {
      const snapshot = await this.session.setObservation(targetId,configuration,this.signal);
      if(targetRevision === this.targetRevision) this.accept(snapshot);
      return true;
    } catch(error) {
      if(targetRevision === this.targetRevision) this.onError(error);
      return false;
    }
    finally {
      if(targetRevision === this.targetRevision) {
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
      this.capturePoints.set(this.targetId,point);
      this.save(`capturePoint.${this.viewerId}.${this.targetId}`,JSON.stringify(point));
    }
    if(snapshot.variables) this.variables.set(this.targetId,snapshot.variables);
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
    for(const [key,video] of this.videos) if(!configuration.enabled || !configuration.registers.some(item => item.key === key && item.format === 'h264')) {
      video.stop(); this.videos.delete(key);
    }
    if(this.snapshot) this.onSync(keys);
    for(const [key,tile] of this.tiles) if(!keys.has(key)) { this.onRemove(key); tile.remove(); this.tiles.delete(key); }
    for(const [key,url] of this.urls) if(!keys.has(key)) { URL.revokeObjectURL(url); this.urls.delete(key); this.frames.delete(key); }
    for(const selection of configuration.registers) {
      let tile = this.tiles.get(selection.key);
      if(!tile) {
        tile = document.createElement('figure'); tile.className = 'watch-tile'; tile.dataset.watchKey = selection.key; tile.dataset.format = selection.format;
        tile.innerHTML = `<figcaption><span class="watch-title" data-tooltip="${this.esc(selection.key)}">${this.esc(selection.key)}</span><button class="pane-action" data-tile-native aria-pressed="false" aria-label="${this.esc(this.t('nativeview'))}" data-tooltip="${this.esc(this.t('nativeviewtip'))}">1:1</button><button class="pane-action" data-tile-float aria-label="${this.esc(this.t('floatpaneltip'))}" data-tooltip="${this.esc(this.t('floatpaneltip'))}"><i class="ti ti-maximize" aria-hidden="true"></i></button><button class="pane-action" data-tile-popup aria-label="${this.esc(this.t('newviewertip'))}" data-tooltip="${this.esc(this.t('newviewertip'))}"><i class="ti ti-external-link" aria-hidden="true"></i></button></figcaption><div class="watch-location mono"></div><div class="watch-content">${selection.format === 'png' ? '<a target="_blank" rel="noopener" hidden><img alt=""></a><span class="watch-placeholder"></span>':'<pre></pre>'}</div><div class="watch-counts"></div>`;
        const created = tile;
        created.dataset.native=this.read(`nativeView.${this.viewerId}.${selection.key}`,'false');
        tile.querySelector('[data-tile-native]').onclick = () => {
          created.dataset.native=String(created.dataset.native !== 'true');
          this.save(`nativeView.${this.viewerId}.${selection.key}`,created.dataset.native);
          this.render();
        };
        tile.querySelector('[data-tile-float]').onclick = () => this.onFloat(selection.key,created);
        const popup = tile.querySelector('[data-tile-popup]');
        popup.hidden = !this.onPopup;
        popup.onclick = () => this.onPopup?.(selection.key);
        this.tiles.set(selection.key,tile);
        this.gallery.append(tile);
        this.onTile(selection.key,tile);
      }
      if(this.session.openRegisterVideo && selection.format !== 'json' && !tile.querySelector('[data-video-format]')) {
        const format = document.createElement('select');
        format.dataset.videoFormat = '';
        format.setAttribute('aria-label', `${selection.key} PNG / H.264`);
        format.innerHTML = '<option value="png">PNG</option><option value="h264">H.264 · BGR U8 NHWC CUDA</option>';
        format.onchange = () => this.configure({...this.configuration, registers:this.selected.map(item =>
          item.key === selection.key ? {...item,format:format.value}:item)});
        tile.querySelector('.watch-location').before(format);
      }
      const format = tile.querySelector('[data-video-format]');
      if(format) format.value = selection.format;
      tile.dataset.watchOrder = configuration.registers.indexOf(selection);
      tile.querySelector('[data-tile-native]').hidden=selection.format !== 'png';
      tile.querySelector('[data-tile-native]').setAttribute('aria-pressed',tile.dataset.native === 'true');
      for(const control of this.root.querySelectorAll('[data-workspace-action=nativeview]')) if(control.dataset.panelId === `watch:${selection.key}`) {
        control.hidden=selection.format !== 'png'; control.setAttribute('aria-pressed',tile.dataset.native === 'true');
        control.dataset.tooltip=this.t('nativeviewtip');
      }
      if(selection.format === 'h264') {
        this.renderVideo(selection,tile);
        continue;
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
      detail.dataset.tooltip = frame?.error ?? (location ? `${this.targetId} · ${location.program} [${location.index}] ${this.t(location.phase)}${frame ? ` · ${this.t('run')} ${location.runId} · ${new Date(frame.capturedAt).toLocaleTimeString()}`:''}\n${this.t(configuration.enabled ? 'watch':'watchoff')}` : this.t('watchtip'));
      const placeholder = tile.querySelector('.watch-placeholder');
      const imageLink = tile.querySelector('.watch-content a');
      if(imageLink) imageLink.hidden = !frame?.available;
      const value = tile.querySelector('.watch-content pre');
      if(value && frame?.error) value.textContent = '';
      if(placeholder) { placeholder.hidden = Boolean(frame?.available || frame?.error); placeholder.textContent = this.t(!this.capturePoint ? 'watchchoosepoint':configuration.enabled ? 'watchwaiting':'watchoff'); }
      tile.querySelector('.watch-counts').textContent = `${frame?.width ? `${frame.width}×${frame.height} · `:''}${this.t('watchreceived')} ${item?.captured ?? 0} · ${this.t('watchskipped')} ${item?.skipped ?? 0}`;
      tile.querySelector('.watch-counts').dataset.tooltip = `${tile.querySelector('.watch-counts').textContent}\n${this.t('watchcountertip')}`;
    }

    if(!this.root.querySelector('#preview-figure').hidden || !this.root.querySelector('#value-figure').hidden) return;
    this.root.querySelector('#image-empty').hidden = keys.size > 0;
  }

  stopVideos() {
    for(const video of this.videos.values()) video.stop();
    this.videos.clear();
  }

  renderVideo(selection,tile) {
    tile.dataset.format = 'h264';
    const location = tile.querySelector('.watch-location');
    location.textContent = `${selection.program} [${selection.index}] ${this.t(selection.phase)}`;
    if(!tile.querySelector('canvas')) tile.querySelector('.watch-content').innerHTML = '<canvas></canvas>';
    if(!this.configuration.enabled || !this.capturePoint || this.updating || this.videos.has(selection.key)) return;
    const canvas = tile.querySelector('canvas');
    const stats = tile.querySelector('.watch-counts');
    stats.textContent = 'H.264';
    location.classList.remove('error');
    location.dataset.tooltip = location.textContent;
    const video = new RegisterVideoView((target,signal) => this.session.openRegisterVideo(this.targetId,target,signal),canvas,stats);
    this.videos.set(selection.key,video);
    video.show(selection).catch(error => {
      if(video.abort.signal.aborted) return;
      location.classList.add('error');
      location.dataset.tooltip = error.message;
      this.onError(error);
    });
  }

  async poll() {
    if(this.polling || this.updating || !this.targetId || !this.snapshot || !this.configuration.enabled || !this.selected.length || document.hidden || !this.isActive()) return;
    this.polling = true;
    const targetId = this.targetId, generation = this.snapshot.generation;
    const targetRevision=this.targetRevision, configurationRevision=this.configurationRevision;
    const current=()=>targetRevision === this.targetRevision && configurationRevision === this.configurationRevision &&
      targetId === this.targetId && generation === this.snapshot?.generation && !this.updating;
    try {
      const snapshot = await this.session.pollObservation(targetId,{generation,after:this.after},this.signal);
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
          const blob = new Blob([bytes],{type:'image/png'});
          const url = URL.createObjectURL(blob);
          const image = tile.querySelector('img'), link = tile.querySelector('a');
          try {
            image.src = url; await image.decode();
            if(!current()) { URL.revokeObjectURL(url); return; }
            const previous = this.urls.get(item.target.key); if(previous) URL.revokeObjectURL(previous);
            this.urls.set(item.target.key,url); sampleImages.set(image,blob); link.href = url; link.hidden = false;
            image.alt = `${targetId} · ${item.target.key} · ${this.t('run')} ${frame.location.runId}`;
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
    if(this.disposed) return;
    this.watchTimer = setTimeout(async () => { await this.poll(); this.schedule(); },Math.max(50,Math.min(500,500 / this.configuration.framesPerSecond)));
  }
}
