import assert from 'node:assert/strict';
import test from 'node:test';
import { layoutInstructions } from '../../src/NeuroModFlowNet.Pipeline.Debugger.Web/wwwroot/instruction-flow.js';

const row = (index, flow, program = 'common') => ({program,index,flow});
const flow = ({guards = [], jumps = [], fallsThrough = true, canStop = false} = {}) =>
  ({guards,jumps:jumps.map(index => ({index})),fallsThrough,canStop});

test('forward conditional jumps and guards indent independently up to the target', () => {
  const layout = layoutInstructions([
    row(0,flow({jumps:[4]})),
    row(1,flow({guards:['crops']})),
    row(2,flow({guards:['crops','text']})),
    row(3,flow({guards:['crops']})),
    row(4),
    row(0,flow({guards:['crops']}),'next')
  ]);
  assert.deepEqual(layout.map(item => item.depth),[0,2,3,2,0,1]);
  assert.equal(layout[0].skipped,3);
  assert.deepEqual(layout[2].guards,[{name:'crops',first:1,last:3},{name:'text',first:2,last:2}]);
  assert.deepEqual(layout[5].guards,[{name:'crops',first:0,last:0}]);
});

test('an unconditional goto, a backward edge and a missing label create no forward indentation', () => {
  const layout = layoutInstructions([
    row(0,flow({jumps:[3],fallsThrough:false})),
    row(1,flow({jumps:[0]})),
    row(2,flow({jumps:[null]})),
    row(3)
  ]);
  assert.deepEqual(layout.map(item => item.depth),[0,0,0,0]);
});
