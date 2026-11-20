import {test} from 'node:test';
import assert from 'node:assert/strict';
import {sortRegisters,memoryKind} from '../../src/NeuroModFlowNet.Pipeline.Debugger.Web/wwwroot/register-table.js';
test('register sorting is natural, reversible and does not mutate metadata',()=>{
  const rows=[{key:'image.10',memory:'Managed'},{key:'image.2',memory:'Cuda:0'}];
  assert.deepEqual(sortRegisters(rows,'key',1).map(row=>row.key),['image.2','image.10']);
  assert.deepEqual(sortRegisters(rows,'key',-1).map(row=>row.key),['image.10','image.2']);
  assert.equal(rows[0].key,'image.10');
  assert.equal(sortRegisters(rows,'memory',1)[0].memory,'Cuda:0');
});
test('shape column compares scalar values numerically',()=>{
  const rows=[{key:'a',value:'2.9'},{key:'b',value:'2.10'},{key:'c',value:'-4'}];
  assert.deepEqual(sortRegisters(rows,'shape',1).map(row=>row.key),['c','b','a']);
});

test('addition order sorts numerically in both directions with unknown orders last',()=>{
  const rows=[{key:'earlier-name',additionOrder:10},{key:'later-name',additionOrder:2},{key:'unknown'}];
  assert.deepEqual(sortRegisters(rows,'additionOrder',1).map(row=>row.key),['later-name','earlier-name','unknown']);
  assert.deepEqual(sortRegisters(rows,'additionOrder',-1).map(row=>row.key),['earlier-name','later-name','unknown']);
  assert.equal(rows[0].additionOrder,10);
});
test('memory accents identify the provider and never guess GPU hardware',()=>{
  assert.equal(memoryKind('Cuda:0'),'cuda'); assert.equal(memoryKind('Managed'),'managed');
  assert.equal(memoryKind('DirectML:0'),'directml'); assert.equal(memoryKind('Cpu:0'),'other');
});
