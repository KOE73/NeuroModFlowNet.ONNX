import assert from 'node:assert/strict';
import test from 'node:test';
import { DebugWorkspace } from '../../src/NeuroModFlowNet.Pipeline.Debugger.Web/wwwroot/workspace.js';

test('a detached viewer keeps its captured frame visible when live panels arrive', () => {
  const snapshot = {id:'preview'};
  const panels = [snapshot];
  const workspace = Object.assign(Object.create(DebugWorkspace.prototype), {
    viewer:true,
    host:{clientWidth:900,clientHeight:700},
    entry:()=>({node:{replaceChildren(){}}}),
    dock:{
      panels,
      getPanel:id=>panels.find(panel=>panel.id === id),
      addPanel:options=>panels.push(options),
      layout(){}
    }
  });
  workspace.attachTile('image.rect',{});
  workspace.attachTile('ocr.recInput',{});
  assert.deepEqual(panels[1].position,{referencePanel:snapshot,direction:'below'});
  assert.equal(panels[2].position.referencePanel,panels[1]);
  assert.equal(panels[2].position.direction,'right');
});
