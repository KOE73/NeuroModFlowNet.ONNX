# Format rule for the `Fabric` static class
- Preserve trivial methods in single-line expression-bodied form.
- Do not rewrite methods like `=> new(model)` into multi-line block bodies.

## Architectural Change Restrictions

- Changes in `Runners` and `Core` are allowed only after prior agreement.
- Before making such changes, the agent must explain:
  - what problem the change solves;
  - why the problem cannot be solved locally in a narrower layer;
  - which public APIs, contracts, and usage scenarios are affected;
  - what risks exist for performance, allocations, and backward compatibility.

- Prior agreement is also required for changes to stable code. Code is considered stable if it matches any of these conditions:
  - it is used by `samples`, `labs`, `tests`, or more than one feature/model family;
  - it defines public APIs, interfaces, base classes, factories, runners, converters, extractors, or output/input data contracts;
  - it is part of inference hot paths, memory conversion, tensor binding, pooling, unsafe code, marshal logic, or data layout;
  - it contains `StructLayout`, inline arrays, spans/memory-based APIs, or low-allocation logic;
  - it is not clearly marked as experimental, obsolete, temporary, or lab-only.

- For stable code, the agent must ask before:
  - changing behavior;
  - changing public API or contracts;
  - renaming or moving files/types;
  - changing data layout or allocation patterns;
  - replacing existing abstractions with new ones.

- Without prior agreement, only local fixes for clear compilation errors, failing tests, or obvious bugs are allowed, and only when they do not change architecture or public contracts.
