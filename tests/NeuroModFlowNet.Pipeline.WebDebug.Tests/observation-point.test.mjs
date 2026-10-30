import {test} from 'node:test';
import assert from 'node:assert/strict';
import {ObservationView} from '../../labs/NeuroModFlowNet.Pipeline.WebDebug/wwwroot/observation.js';

function view() {
  const view=Object.create(ObservationView.prototype);
  Object.assign(view,{camera:'camera',viewerId:'main',capturePoints:new Map(),saved:new Map(),
    snapshot:{configuration:{enabled:true,framesPerSecond:3,maxEdge:640,registers:[
      {key:'image',format:'png',program:'prepare',index:0,phase:'before'},
      {key:'data',format:'json',program:'common',index:2,phase:'after'}]}}});
  view.save=(key,value)=>view.saved.set(key,value);
  view.configure=async configuration=>{view.snapshot={configuration};return true;};
  view.onChange=()=>{};
  return view;
}

test('one checked instruction moves all selected registers to the same after boundary',async()=>{
  const subject=view();
  await subject.setCapturePoint({program:'common',index:5},true);
  assert.deepEqual(subject.selected,[{key:'image',format:'png',program:'common',index:5,phase:'after'},
    {key:'data',format:'json',program:'common',index:5,phase:'after'}]);
  await subject.setCapturePoint({program:'prepare',index:1},true);
  assert.equal(subject.configuration.enabled,true);
  assert.ok(subject.selected.every(item=>item.program === 'prepare' && item.index === 1));
  assert.deepEqual(JSON.parse(subject.saved.get('capturePoint.main.camera')),subject.capturePoint);
});

test('unchecking the point stops observation and retains selected register names',async()=>{
  const subject=view();
  await subject.setCapturePoint({program:'common',index:5},true);
  await subject.setCapturePoint({program:'common',index:5},false);
  assert.equal(subject.capturePoint,null);
  assert.equal(subject.configuration.enabled,false);
  assert.deepEqual(subject.selected.map(item=>item.key),['image','data']);
});

test('failed point change restores the visible and persisted point',async()=>{
  const subject=view(), previous={program:'common',index:2,phase:'after'};
  subject.capturePoints.set('camera',previous);
  subject.configure=async()=>false;
  await subject.setCapturePoint({program:'common',index:5},true);
  assert.deepEqual(subject.capturePoint,previous);
  assert.deepEqual(JSON.parse(subject.saved.get('capturePoint.main.camera')),previous);
});
