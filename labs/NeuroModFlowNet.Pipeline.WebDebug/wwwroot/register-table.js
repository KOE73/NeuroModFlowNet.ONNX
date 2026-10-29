export function memoryKind(memory) {
  if(/cuda|tensorrt/i.test(memory)) return 'cuda';
  if(/^managed$/i.test(memory)) return 'managed';
  if(/dml|directml/i.test(memory)) return 'directml';
  return 'other';
}

const collator = new Intl.Collator(undefined,{numeric:true,sensitivity:'base'});
export function sortRegisters(variables,column,direction) {
  const value = variable => column === 'key' ? variable.key : column === 'memory' ? variable.memory : variable.shape ?? variable.value ?? '';
  return [...variables].sort((left,right) => {
    const leftValue=value(left),rightValue=value(right);
    const numeric=column === 'shape' && !left.shape && !right.shape && String(leftValue).trim() !== '' && String(rightValue).trim() !== '' && Number.isFinite(Number(leftValue)) && Number.isFinite(Number(rightValue));
    return direction * (numeric ? Number(leftValue)-Number(rightValue):collator.compare(leftValue,rightValue)) || collator.compare(left.key,right.key);
  });
}
