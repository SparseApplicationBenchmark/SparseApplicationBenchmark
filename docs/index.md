---
title: Test Visualizaiton
---
# Testing!

```js
const profile = FileAttachment("./data/profile.json").json();
```

```js
const allTags = ["dense", "tensor", "test", "trace"];
const keep = view(Inputs.checkbox(allTags, {label: "Include tags"}));
const drop = view(Inputs.checkbox(allTags, {label: "Exclude tags"}));
```

```js
const ok = profile.problems.map((p) =>
  (keep.length === 0 || p.tags.some((t) => keep.includes(t))) &&
  !p.tags.some((t) => drop.includes(t))
);

const nKept = new Map();
profile.problems.forEach((p, i) => {
  if (ok[i]) nKept.set(p.benchmark, (nKept.get(p.benchmark) ?? 0) + 1);
});
const total = nKept.size;

const curves = Object.entries(profile.series).flatMap(([framework, points]) => {
  let cum = 0;
  const pts = points
    .filter(([i]) => ok[i])
    .map(([i, ratio]) => ({
      framework,
      ratio,
      pct: (100 * (cum += 1 / nKept.get(profile.problems[i].benchmark))) / total
    }));
  pts.push({framework, ratio: profile.xMax, pct: pts.at(-1)?.pct ?? 0});
  return pts;
});
```

```js
Plot.plot({
  title: `${ok.filter(Boolean).length} problems`,
  width,
  x: {type: "log", domain: [1, profile.xMax], label: "Ratio (runtime / best runtime)"},
  y: {domain: [0, 100], grid: true, label: "% of suite completed"},
  color: {legend: true},
  marks: [
    Plot.line(curves, {
      x: "ratio",
      y: "pct",
      stroke: "framework",
      curve: "step-after",
      strokeWidth: 3,
      tip: true
    })
  ]
})
```

<!-- ```js
ok
```

```js
curves
``` -->
