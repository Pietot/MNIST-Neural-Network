/* Load the model inside an async function so this also works in a classic script. */
(async () => {
  let layers;
  let model = null;
  const f = await fetch("model.bin").then((r) => r.blob());
  const buf = await f.arrayBuffer(),
    hl = new DataView(buf).getUint32(0, true);
  const shapes = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buf, 4, hl)),
  ).shapes;
  let off = 4 + hl;
  layers = [];
  for (const [n_out, n_in] of shapes) {
    const w = new Float64Array(buf.slice(off, off + n_out * n_in * 8));
    off += n_out * n_in * 8;
    const b = new Float64Array(buf.slice(off, off + n_out * 8));
    off += n_out * 8;
    layers.push({ n_out, n_in, w, b });
  }
  model = layers;

  const N = 28,
    S = 20,
    NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
  const cv = document.getElementById("c"),
    ctx = cv.getContext("2d");
  const grid = new Float32Array(N * N);
  let drawing = null,
    last = null;

  function render() {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, cv.width, cv.height);
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++) {
        const v = grid[y * N + x];
        if (v > 0.01) {
          const g = Math.round(v * 255);
          ctx.fillStyle = `rgb(${g},${g},${g})`;
          ctx.fillRect(x * S, y * S, S, S);
        }
      }
  }

  // Pinceau : centre plein, voisins directs adoucis (proche du trait MNIST)
  function stamp(cx, cy, erase) {
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const x = cx + dx,
          y = cy + dy;
        if (x < 0 || y < 0 || x >= N || y >= N) continue;
        const w = dx === 0 && dy === 0 ? 1 : dx === 0 || dy === 0 ? 0.45 : 0.15;
        const i = y * N + x;
        grid[i] = erase ? 0 : Math.min(1, Math.max(grid[i], w));
      }
  }
  function cell(e) {
    const r = cv.getBoundingClientRect();
    return [
      Math.min(
        N - 1,
        Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * N)),
      ),
      Math.min(
        N - 1,
        Math.max(0, Math.floor(((e.clientY - r.top) / r.height) * N)),
      ),
    ];
  }
  function line(a, b, erase) {
    const n = Math.max(Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1]), 1);
    for (let t = 0; t <= n; t++)
      stamp(
        Math.round(a[0] + ((b[0] - a[0]) * t) / n),
        Math.round(a[1] + ((b[1] - a[1]) * t) / n),
        erase,
      );
  }
  cv.addEventListener("contextmenu", (e) => e.preventDefault());
  cv.addEventListener("pointerdown", (e) => {
    cv.setPointerCapture(e.pointerId);
    drawing = e.button === 2 ? "erase" : "draw";
    last = cell(e);
    line(last, last, drawing === "erase");
    update();
  });
  cv.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const p = cell(e);
    line(last, p, drawing === "erase");
    last = p;
    update();
  });
  ["pointerup", "pointercancel"].forEach((t) =>
    cv.addEventListener(t, () => {
      drawing = null;
    }),
  );
  function clearAll() {
    grid.fill(0);
    update();
  }
  addEventListener("keydown", (e) => {
    if (e.key === "c" || e.key === "C") clearAll();
  });

  // Réseau : ReLU sur les couches cachées, softmax en sortie
  function predict(x) {
    let a = Array.from(x);
    model.forEach((L, k) => {
      const out = new Array(L.n_out);
      for (let i = 0; i < L.n_out; i++) {
        const o = i * L.n_in;
        let s = L.b[i];
        for (let j = 0; j < L.n_in; j++) s += L.w[o + j] * a[j];
        out[i] = s;
      }
      if (k < model.length - 1) a = out.map((v) => (v > 0 ? v : 0));
      else {
        const m = Math.max(...out),
          ex = out.map((v) => Math.exp(v - m)),
          t = ex.reduce((p, q) => p + q, 0);
        a = ex.map((v) => v / t);
      }
    });
    return a;
  }
  const rowsEl = document.getElementById("rows");
  function show(p) {
    const order = NAMES.map((n, i) => [n, p ? p[i] : 0]).sort(
      (a, b) => b[1] - a[1],
    );
    rowsEl.innerHTML = order
      .map(
        (r, i) =>
          `<div class="row${i === 0 && p ? " top" : ""}"><span>${r[0]}</span><span>${(r[1] * 100).toFixed(2)}%</span><i class="bar" style="width:${(r[1] * 100).toFixed(1)}%"></i></div>`,
      )
      .join("");
  }
  let pending = false;
  function update() {
    render();
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const empty = !grid.some((v) => v > 0);
      show(model && !empty ? predict(grid) : null);
    });
  }
  render();
  show(null);
})();
