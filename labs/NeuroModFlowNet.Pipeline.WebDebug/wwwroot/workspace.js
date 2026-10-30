import { createDockview } from './vendor/dockview/dockview.mjs';

export class DebugWorkspace {
  constructor({t,read,save,onError,onPopup,onCloseTile,viewer=false,id='main',hasWatches=false}) {
    Object.assign(this,{t,read,save,onError,onPopup,onCloseTile,viewer,id,hasWatches});
    this.entries = new Map(); this.loading = true;
    this.storageKey = `layout.${id}.v1`;
    this.host = document.querySelector('.debug-columns'); this.host.classList.add('dock-workspace');
    this.stash = document.createElement('div'); this.stash.hidden = true; document.querySelector('#shell').append(this.stash);
    for(const [key,selector,title] of [['instructions','.instruction-pane','instructions'],['registers','.register-pane','registers'],['preview','.preview-pane','image']]) {
      const node = document.querySelector(selector), header = node.querySelector('.panel-header');
      this.entries.set(key,{node,header,title}); this.stash.append(node,header);
      this.decorateHeader(header,key);
    }
    this.host.replaceChildren();
    this.dock = createDockview(this.host,{
      theme:{name:'nmfn',className:'dockview-theme-nmfn'},singleTabMode:'fullwidth',defaultRenderer:'onlyWhenVisible',
      floatingGroupDragHandle:'titlebar',
      createComponent: options => this.content(options.id),
      createTabComponent: options => this.tab(options.id)
    });
    this.dock.layout(this.host.clientWidth,this.host.clientHeight,true);
    let restored = false;
    try {
      const state = JSON.parse(read(this.storageKey,'null'));
      if(state?.grid && state?.panels && Object.keys(state.panels).length <= 24 && Object.keys(state.panels).every(key => this.entries.has(key) || key.startsWith('watch:'))) {
        this.dock.fromJSON(state); restored = true;
      }
    } catch(error) { onError(new Error(`${t('layoutinvalid')}: ${error.message}`)); }
    if(!restored) this.defaults();
    this.applyAppearance();
    this.loading = false;
    this.dock.onDidLayoutChange(() => { if(!this.loading) save(this.storageKey,JSON.stringify(this.dock.toJSON())); });
    new ResizeObserver(() => { if(this.host.clientWidth && this.host.clientHeight) this.dock.layout(this.host.clientWidth,this.host.clientHeight); }).observe(this.host);
    document.querySelector('#panels-menu').onclick = event => {
      const button = event.target.closest('[data-panel]'); if(!button) return;
      this.show(button.dataset.panel); document.querySelector('#panels-menu').open = false;
    };
    document.querySelector('#reset-layout').onclick = () => { this.defaults(); document.querySelector('#panels-menu').open = false; };
    document.querySelector('#new-viewer').onclick = () => { onPopup(); document.querySelector('#panels-menu').open = false; };
    if(viewer) {
      const toolbar = document.querySelector('.watch-toolbar'); toolbar.classList.add('viewer-watch-toolbar');
      document.querySelector('.debug-columns').before(toolbar);
    }
  }

  applyAppearance() {
    const gap = Number.parseFloat(getComputedStyle(this.host).getPropertyValue('--vm-panel-gap')) || 0;
    if(this.panelGap === gap) return;
    this.panelGap = gap;
    this.dock.updateOptions({theme:{name:'nmfn',className:'dockview-theme-nmfn',gap}});
    this.dock.layout(this.host.clientWidth,this.host.clientHeight,true);
  }

  entry(id) {
    if(!this.entries.has(id)) {
      const node = document.createElement('div'); node.className = 'watch-panel-body';
      const header = document.createElement('div'); header.className = 'panel-header';
      const title = document.createElement('h2'); title.textContent = id.slice(6); header.append(title);
      this.entries.set(id,{node,header,title:null}); this.decorateHeader(header,id);
    }
    return this.entries.get(id);
  }
  content(id) {
    const entry = this.entry(id);
    const wrapper=document.createElement('div'); wrapper.className='dock-content';
    return {element:wrapper,init:()=>wrapper.append(entry.node),onShow:()=>wrapper.append(entry.node),
      onHide:()=>this.stash.append(entry.node),dispose:()=>this.stash.append(entry.node)};
  }
  tab(id) {
    const entry = this.entry(id);
    return {element:entry.header,init(){},dispose:()=>this.stash.append(entry.header)};
  }
  decorateHeader(header,id) {
    const add = (icon,label,action) => {
      const button = document.createElement('button'); button.className = 'pane-action';
      button.dataset.workspaceAction = label; button.dataset.panelId = id;
      button.innerHTML = `<i class="ti ti-${icon}" aria-hidden="true"></i>`;
      button.setAttribute('aria-label',this.t(label)); button.dataset.tooltip = this.t(label);
      button.addEventListener('pointerdown',event=>event.stopPropagation());
      button.addEventListener('mousedown',event=>event.stopPropagation());
      button.onclick = event=>{event.stopPropagation();action();}; header.append(button); return button;
    };
    add('maximize','floatpanel',()=>this.toggleFloat(id));
    if(id.startsWith('watch:')) {
      const native=add('zoom-in','nativeview',()=>this.entry(id).node.querySelector('[data-tile-native]')?.click());
      native.textContent='1:1';
    }
    if(id === 'preview' || id.startsWith('watch:')) add('external-link','newviewer',()=>this.onPopup(id.startsWith('watch:') ? id.slice(6):undefined));
    if(id.startsWith('watch:')) add('x',this.viewer ? 'closepanel':'returngallery',()=>this.viewer ? this.onCloseTile(id.slice(6)):this.returnTile(id));
  }
  show(id) {
    const existing = this.dock.getPanel(id); if(existing) { existing.api.setActive(); return existing; }
    const panel=this.dock.addPanel({id,component:'panel',tabComponent:'panel-tab',title:this.entries.get(id)?.title ? this.t(this.entries.get(id).title):id.slice(6),minimumWidth:180,minimumHeight:120});
    this.dock.layout(this.host.clientWidth,this.host.clientHeight,true); return panel;
  }
  toggleFloat(id) {
    const panel = this.dock.getPanel(id) ?? this.show(id); if(!panel) return;
    if(panel.group.api.location.type === 'floating') panel.group.api.moveTo({position:'right'});
    else this.dock.addFloatingGroup(panel,{width:Math.min(600,this.host.clientWidth-30),height:Math.min(500,this.host.clientHeight-30),position:{left:30,top:35}});
  }
  attachTile(key,tile) {
    const id = `watch:${key}`;
    const panel = this.dock.getPanel(id);
    if(this.viewer || panel) {
      this.entry(id).node.replaceChildren(tile);
      if(!panel) {
        const previous = this.dock.panels.filter(item=>item.id.startsWith('watch:')).at(-1);
        this.dock.addPanel({id,component:'panel',tabComponent:'panel-tab',title:key,minimumWidth:180,minimumHeight:150,
          ...(previous ? {position:{referencePanel:previous,direction:'right'}}:{})});
      }
      this.dock.layout(this.host.clientWidth,this.host.clientHeight,true);
    }
  }
  floatTile(key,tile) {
    const id = `watch:${key}`; this.entry(id).node.replaceChildren(tile); this.show(id); this.toggleFloat(id);
  }
  removeTile(key) {
    const id = `watch:${key}`, panel = this.dock.getPanel(id);
    if(panel) this.dock.removePanel(panel);
    this.entries.get(id)?.node.replaceChildren(); this.entries.delete(id);
  }
  pruneTiles(keys) {
    for(const panel of [...this.dock.panels]) if(panel.id.startsWith('watch:') && !keys.has(panel.id.slice(6))) this.removeTile(panel.id.slice(6));
  }
  returnTile(id) {
    const entry = this.entry(id), tile = entry.node.querySelector('.watch-tile');
    const panel = this.dock.getPanel(id); if(panel) this.dock.removePanel(panel);
    if(this.viewer) { if(tile) entry.node.append(tile); this.show(id); }
    else if(tile) {
      const gallery=document.querySelector('#watch-gallery');
      const next=[...gallery.children].find(other=>Number(other.dataset.watchOrder) > Number(tile.dataset.watchOrder));
      gallery.insertBefore(tile,next ?? null);
    }
  }
  defaults() {
    this.loading = true; this.dock.clear();
    if(this.viewer) {
      const tiles = [...this.entries].filter(([key,entry])=>key.startsWith('watch:') && entry.node.querySelector('.watch-tile'));
      if(tiles.length) for(const [id] of tiles) {
        const previous=this.dock.panels.at(-1);
        this.dock.addPanel({id,component:'panel',tabComponent:'panel-tab',title:id.slice(6),minimumWidth:180,minimumHeight:150,...(previous ? {position:{referencePanel:previous,direction:'right'}}:{})});
      } else if(!this.hasWatches) this.show('registers');
      this.dock.layout(this.host.clientWidth,this.host.clientHeight,true);
    } else {
      for(const id of this.entries.keys()) if(id.startsWith('watch:')) this.returnTile(id);
      this.dock.addPanel({id:'instructions',component:'panel',tabComponent:'panel-tab',title:this.t('instructions'),minimumWidth:220});
      this.dock.addPanel({id:'registers',component:'panel',tabComponent:'panel-tab',title:this.t('registers'),minimumWidth:220,position:{referencePanel:'instructions',direction:'right'}});
      this.dock.addPanel({id:'preview',component:'panel',tabComponent:'panel-tab',title:this.t('image'),minimumWidth:220,position:{referencePanel:'registers',direction:'right'}});
      this.dock.layout(this.host.clientWidth,this.host.clientHeight);
      let weights = [1.1,1,1];
      try { const stored=JSON.parse(this.read('panelWidths','null')); if(stored?.length === 3 && stored.every(value=>Number.isFinite(value) && value > 0)) weights=stored; } catch { }
      const total=weights.reduce((sum,value)=>sum+value,0), available=this.host.clientWidth-12;
      this.dock.getPanel('instructions').group.api.setSize({width:Math.max(220,available*weights[0]/total)});
      this.dock.getPanel('registers').group.api.setSize({width:Math.max(220,available*weights[1]/total)});
    }
    this.loading = false; this.save(this.storageKey,JSON.stringify(this.dock.toJSON()));
  }
  translate() {
    for(const [id,entry] of this.entries) if(entry.title) this.dock.getPanel(id)?.api.setTitle(this.t(entry.title));
    for(const button of document.querySelectorAll('[data-workspace-action]')) {
      button.setAttribute('aria-label',this.t(button.dataset.workspaceAction)); button.dataset.tooltip=this.t(button.dataset.workspaceAction);
    }
  }
}
