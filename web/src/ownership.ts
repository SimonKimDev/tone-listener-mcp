// Monotonic ownership token: only work started under the latest generation may apply its result.
export function createGeneration() {
  let generation = 0;
  return {
    next() {
      generation += 1;
      return generation;
    },
    current() {
      return generation;
    },
    owns(token: number) {
      return token === generation;
    },
  };
}

// Wraps a callback so it becomes a no-op once a newer generation has started.
export function guardedBy<A extends unknown[]>(
  owner: ReturnType<typeof createGeneration>,
  token: number,
  callback: (...args: A) => void,
) {
  return (...args: A) => {
    if (owner.owns(token)) callback(...args);
  };
}
