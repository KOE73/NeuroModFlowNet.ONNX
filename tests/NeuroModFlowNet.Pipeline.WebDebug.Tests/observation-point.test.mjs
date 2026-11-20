import {test} from 'node:test';
import assert from 'node:assert/strict';
import {ObservationView,sharedCapturePoint} from '../../src/NeuroModFlowNet.Pipeline.Debugger.Web/wwwroot/observation.js';

function view() {
  const view=Object.create(ObservationView.prototype);
  Object.assign(view,{targetId:'targetId',viewerId:'main',capturePoints:new Map(),saved:new Map(),
    videos:new Map(),snapshot:{configuration:{enabled:true,framesPerSecond:3,maxEdge:640,registers:[
      {key:'image',format:'png',program:'prepare',index:0,phase:'before'},
      {key:'data',format:'json',program:'common',index:2,phase:'after'}]}}});
  view.save=(key,value)=>view.saved.set(key,value);
  view.configure=async configuration=>{view.snapshot={configuration};return true;};
  view.onChange=()=>{};
  return view;
}

test('an active server capture point takes precedence over another tab saved point',()=>{
  const configuration={enabled:true,registers:[{key:'image',program:'common',index:12,phase:'after'},{key:'data',program:'common',index:12,phase:'after'}]};
  assert.deepEqual(sharedCapturePoint(configuration),{program:'common',index:12,phase:'after'});
  assert.equal(sharedCapturePoint({...configuration,enabled:false}),null);
  assert.equal(sharedCapturePoint({...configuration,registers:[configuration.registers[0],{...configuration.registers[1],index:0}]}),null);
});

test('one checked instruction moves all selected registers to the same after boundary',async()=>{
  const subject=view();
  await subject.setCapturePoint({program:'common',index:5},true);
  assert.deepEqual(subject.selected,[{key:'image',format:'png',program:'common',index:5,phase:'after'},
    {key:'data',format:'json',program:'common',index:5,phase:'after'}]);
  await subject.setCapturePoint({program:'prepare',index:1},true);
  assert.equal(subject.configuration.enabled,true);
  assert.ok(subject.selected.every(item=>item.program === 'prepare' && item.index === 1));
  assert.deepEqual(JSON.parse(subject.saved.get('capturePoint.main.targetId')),subject.capturePoint);
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
  subject.capturePoints.set('targetId',previous);
  subject.configure=async()=>false;
  await subject.setCapturePoint({program:'common',index:5},true);
  assert.deepEqual(subject.capturePoint,previous);
  assert.deepEqual(JSON.parse(subject.saved.get('capturePoint.main.targetId')),previous);
});

for(const cancelledDecode of [false,true]) test(`size change discards an old PNG ${cancelledDecode ? 'decode rejection':'decoded frame'} without poisoning acknowledgements`,async()=>{
  const previousDocument=globalThis.document;
  globalThis.document={hidden:false,querySelector:()=>({dataset:{page:'targetId'}})};
  let startDecode, resolveDecode, rejectDecode;
  const started=new Promise(resolve=>{startDecode=resolve;});
  const decoded=new Promise((resolve,reject)=>{resolveDecode=resolve;rejectDecode=reject;});
  const image={decode:()=>{startDecode();return decoded;}};
  const link={hidden:true};
  const tile={querySelector:selector=>selector === 'img' ? image:link};
  const subject=view(), errors=[];
  Object.assign(subject,{targetRevision:1,configurationRevision:0,polling:false,updating:false,after:{},frames:new Map(),urls:new Map(),tiles:new Map([['image',tile]])});
  subject.snapshot.generation=1;
  subject.render=()=>{};
  subject.onError=error=>errors.push(error);
  subject.accept=snapshot=>{subject.snapshot=snapshot;};
  subject.isActive=()=>true;
  subject.session={
    setObservation:async(id,configuration)=>({generation:2,configuration,items:[]}),
    pollObservation:async()=>({generation:1,configuration:subject.configuration,
      items:[{target:{key:'image'},frame:{sequence:91,data:'AAAA',contentType:'image/png',location:{runId:1}}}]})
  };
  subject.configure=ObservationView.prototype.configure;
  try {
    const polling=subject.poll();
    await started;
    await subject.configure({...subject.configuration,maxEdge:2000});
    if(cancelledDecode) rejectDecode(new Error('Old image was detached'));
    else resolveDecode();
    await polling;
    assert.equal(subject.configuration.maxEdge,2000);
    assert.deepEqual(subject.after,{});
    assert.equal(subject.frames.size,0);
    assert.equal(errors.length,0);
  } finally {globalThis.document=previousDocument;}
});
