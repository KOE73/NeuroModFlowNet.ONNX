export const sampleImages = new WeakMap();

export function enablePreviewLayout(gallery,root) {
  let drag;
  gallery.addEventListener('pointerdown', event => {
    const caption = event.target.closest('figcaption');
    const tile = caption?.parentElement;
    if(event.button !== 0 || event.target.closest('button') || tile?.parentElement !== gallery) return;
    drag = {tile,id:event.pointerId,x:event.clientX,y:event.clientY,title:caption.querySelector('span')?.textContent};
    gallery.setPointerCapture(event.pointerId);
  });
  gallery.addEventListener('pointermove', event => {
    if(!drag || event.pointerId !== drag.id) return;
    if(!drag.badge) {
      if(Math.hypot(event.clientX-drag.x,event.clientY-drag.y) < 5) return;
      drag.badge = document.createElement('div');
      drag.badge.className = 'preview-drag-label';
      drag.badge.textContent = drag.title;
      root.append(drag.badge);
      drag.tile.classList.add('preview-moving');
    }
    drag.badge.style.transform = `translate(${event.clientX+14}px,${event.clientY+14}px)`;
    const target = document.elementFromPoint(event.clientX,event.clientY)?.closest('figure');
    if(target?.parentElement !== gallery || target === drag.tile) return;
    // Ignore the same pointer location after layout moved the hit target.
    if(drag.last && Math.hypot(event.clientX-drag.last.x,event.clientY-drag.last.y) < 8) return;
    const bounds = target.getBoundingClientRect();
    const after = event.clientX > bounds.left + bounds.width/2;
    const anchor = after ? target.nextElementSibling : target;
    if(anchor === drag.tile || anchor === drag.tile.nextElementSibling) return;
    gallery.insertBefore(drag.tile,anchor);
    drag.last = {x:event.clientX,y:event.clientY};
  });
  gallery.addEventListener('lostpointercapture', () => {
    drag?.badge?.remove();
    drag?.tile.classList.remove('preview-moving');
    drag = null;
  });
}
