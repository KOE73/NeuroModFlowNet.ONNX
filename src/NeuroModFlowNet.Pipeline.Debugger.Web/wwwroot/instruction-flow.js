// Layout is local to each program. Guard names group adjacent checks without changing execution.
export function layoutInstructions(instructions) {
  const layout = instructions.map(() => ({depth:0, guards:[], skipped:0}));
  for(let start = 0; start < instructions.length;) {
    let end = start + 1;
    while(end < instructions.length && instructions[end].program === instructions[start].program) end++;
    const program = instructions.slice(start,end);
    const depths = Array(program.length + 1).fill(0);
    const programEnd = program.length;

    for(const [index,instruction] of program.entries()) {
      const flow = instruction.flow;
      if(!flow) continue;
      const successors = flow.jumps.map(jump => jump.index)
        .filter(target => Number.isInteger(target) && target >= 0 && target <= programEnd);
      if(flow.canStop) successors.push(programEnd);
      if(flow.fallsThrough) successors.push(index + 1);
      // The span between forward alternatives is conditional. A lone goto or a back edge
      // has a destination but does not by itself establish an indented region.
      const forward = successors.filter(target => target > index);
      if(forward.length < 2) continue;
      const first = Math.min(...forward), after = Math.max(...forward);
      depths[first]++;
      depths[after]--;
      layout[start + index].skipped = after - first;
    }

    let depth = 0;
    for(const [index,instruction] of program.entries()) {
      depth += depths[index];
      const names = instruction.flow?.guards ?? [];
      const row = layout[start + index];
      row.depth = depth + names.length;
      for(const [level,name] of names.entries()) {
        const previous = layout[start + index - 1]?.guards[level];
        const samePrefix = index > 0 && name && names.slice(0,level + 1).every((guard,position) =>
          guard && guard === program[index - 1].flow?.guards[position]);
        const scope = samePrefix && previous ? previous : {name, first:instruction.index, last:instruction.index};
        scope.last = instruction.index;
        row.guards.push(scope);
      }
    }
    start = end;
  }
  return layout;
}
