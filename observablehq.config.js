// See https://observablehq.com/framework/config for documentation.
export default {
  title: "SAPS Benchmarks",

  // Path to the root of the project
  root: "docs",

  // Data loaders run under Poetry
  interpreters: {".py": ["poetry", "run", "python3"]}
};
