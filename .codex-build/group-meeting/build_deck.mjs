import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const SKILL_DIR = "C:/Users/Sherlock/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.11113/skills/presentations";
const RUNTIME_PYTHON = "C:/Users/Sherlock/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const RUNTIME_NODE_MODULES = "C:/Users/Sherlock/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
process.env.RUNTIME_NODE_MODULES = RUNTIME_NODE_MODULES;
process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES = RUNTIME_NODE_MODULES;
const workspaceDir = "D:/University/Alpha-SQL";
const buildDir = path.join(workspaceDir, ".codex-build", "group-meeting");
const previewDir = path.join(buildDir, "previews-v4");
const finalPath = path.join(workspaceDir, "output", "presentations", "jev_schema_aware_group_meeting_zh_v4.pptx");

const {
  applyPresentationChartFont,
  finalizePresentation,
  makeNativeBulletParagraphs,
} = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs")).href);

const W = 1280;
const H = 720;
const FONT = "Microsoft YaHei";
const C = {
  bg: "#F6F5F1",
  paper: "#FFFEFA",
  ink: "#102A43",
  muted: "#617385",
  line: "#D8DFE5",
  teal: "#0E7C7B",
  teal2: "#43AAA5",
  tealPale: "#E2F1EF",
  orange: "#E07A3F",
  orangePale: "#F9E8DC",
  red: "#C4473A",
  redPale: "#F7E3E0",
  blue: "#3B6EA8",
  bluePale: "#E4ECF6",
  green: "#2E8B57",
  greenPale: "#E3F1E8",
  grayPale: "#EEF1F4",
  gray: "#AAB5BF",
  white: "#FFFFFF",
};

const p = Presentation.create({ slideSize: { width: W, height: H } });

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addText(slide, text, pos, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: pos,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: style.fontSize ?? 20,
    color: style.color ?? C.ink,
    bold: style.bold ?? false,
    italic: style.italic ?? false,
    alignment: style.alignment ?? "left",
    verticalAlignment: style.verticalAlignment ?? "top",
    autoFit: style.autoFit ?? "shrinkText",
    ...style,
  };
  return shape;
}

function addBox(slide, pos, opts = {}) {
  return slide.shapes.add({
    geometry: opts.geometry ?? "roundRect",
    position: pos,
    fill: opts.fill ?? C.paper,
    line: { style: "solid", fill: opts.line ?? C.line, width: opts.lineWidth ?? 1 },
    borderRadius: opts.borderRadius ?? "rounded-xl",
    shadow: opts.shadow ?? undefined,
  });
}

function addBoxText(slide, text, pos, opts = {}) {
  const box = addBox(slide, pos, opts);
  box.text = text;
  box.text.style = {
    typeface: FONT,
    fontSize: opts.fontSize ?? 20,
    color: opts.color ?? C.ink,
    bold: opts.bold ?? false,
    alignment: opts.alignment ?? "center",
    verticalAlignment: opts.verticalAlignment ?? "middle",
    autoFit: "shrinkText",
  };
  return box;
}

function addBulletList(slide, items, pos, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: pos,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = makeNativeBulletParagraphs(items, {
    marginLeftPoints: opts.marginLeftPoints ?? 18,
    hangingPoints: opts.hangingPoints ?? 9,
    spaceAfterPoints: opts.spaceAfterPoints ?? 7,
  });
  shape.text.style = {
    typeface: FONT,
    fontSize: opts.fontSize ?? 19,
    color: opts.color ?? C.ink,
    autoFit: "shrinkText",
  };
  return shape;
}

function addHeader(slide, title, section = "") {
  addText(slide, title, { left: 64, top: 42, width: 1035, height: 48 }, { fontSize: 32, bold: true });
  if (section) addText(slide, section, { left: 1090, top: 48, width: 126, height: 25 }, { fontSize: 13, bold: true, color: C.teal, alignment: "right" });
  slide.shapes.add({ geometry: "rect", position: { left: 64, top: 102, width: 1152, height: 2 }, fill: C.line, line: { fill: "none", width: 0 } });
}

function addFooter(slide, number, label = "Jev + Schema-aware SQL 选择") {
  addText(slide, label, { left: 64, top: 681, width: 390, height: 18 }, { fontSize: 11, color: C.muted });
  addText(slide, String(number).padStart(2, "0"), { left: 1160, top: 678, width: 56, height: 20 }, { fontSize: 12, bold: true, color: C.muted, alignment: "right" });
}

function addArrow(slide, left, top, width = 50, height = 28, fill = C.gray) {
  return slide.shapes.add({ geometry: "rightArrow", position: { left, top, width, height }, fill, line: { fill: "none", width: 0 } });
}

function addNotes(slide, lines) {
  slide.speakerNotes.text = lines;
  slide.speakerNotes.setVisible(true);
}

function styleTable(table, { headerFill = C.ink, headerColor = C.white, fontSize = 16, rowFills = [] } = {}) {
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  const all = table.cells.block({ row: 0, column: 0, rowCount: table.rows.length, columnCount: table.columns.length });
  all.assign({
    textStyle: { typeface: FONT, fontSize, color: C.ink },
    margins: { left: 10, right: 10, top: 8, bottom: 8 },
    anchor: "middle",
  });
  const header = table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: table.columns.length });
  header.assign({ fill: headerFill, textStyle: { typeface: FONT, fontSize, bold: true, color: headerColor }, anchor: "middle" });
  rowFills.forEach((fill, i) => {
    if (i + 1 < table.rows.length) table.cells.block({ row: i + 1, column: 0, rowCount: 1, columnCount: table.columns.length }).fill = fill;
  });
  for (let r = 0; r < table.rows.length; r++) {
    for (let c = 0; c < table.columns.length; c++) {
      table.getCell(r, c).text.style = {
        typeface: FONT,
        fontSize,
        color: r === 0 ? headerColor : C.ink,
        bold: r === 0,
        verticalAlignment: "middle",
        autoFit: "shrinkText",
      };
    }
  }
}

function addMetricChip(slide, value, label, pos, color = C.teal, pale = C.tealPale) {
  addBox(slide, pos, { fill: pale, line: color, lineWidth: 1.2 });
  addText(slide, value, { left: pos.left + 14, top: pos.top + 12, width: pos.width - 28, height: 48 }, { fontSize: 34, bold: true, color, alignment: "center", verticalAlignment: "middle" });
  addText(slide, label, { left: pos.left + 14, top: pos.top + 62, width: pos.width - 28, height: 36 }, { fontSize: 15, color: C.muted, alignment: "center" });
}

// Slide 1
{
  const s = p.slides.add();
  s.background.fill = C.bg;
  s.shapes.add({ geometry: "rect", position: { left: 0, top: 0, width: 18, height: H }, fill: C.teal, line: { fill: "none", width: 0 } });
  s.shapes.add({ geometry: "rect", position: { left: 18, top: 0, width: 360, height: H }, fill: C.tealPale, line: { fill: "none", width: 0 } });
  addText(s, "Jev + Schema-aware\nSQL 选择实验", { left: 96, top: 115, width: 690, height: 180 }, { fontSize: 54, bold: true, color: C.ink });
  addText(s, "从高召回 Schema Linking，到可验证的候选重排", { left: 100, top: 323, width: 720, height: 52 }, { fontSize: 25, color: C.teal, bold: true });
  addText(s, "组会汇报", { left: 934, top: 104, width: 220, height: 40 }, { fontSize: 20, bold: true, color: C.orange, alignment: "right" });
  addBoxText(s, "12题探路", { left: 884, top: 200, width: 250, height: 78 }, { fill: C.paper, line: C.line, fontSize: 23, bold: true });
  addBoxText(s, "300题扩展", { left: 884, top: 301, width: 250, height: 78 }, { fill: C.paper, line: C.line, fontSize: 23, bold: true });
  addBoxText(s, "100题监督重排", { left: 884, top: 402, width: 250, height: 78 }, { fill: C.paper, line: C.line, fontSize: 23, bold: true });
  addText(s, "Dongx Xia · 50062682", { left: 100, top: 600, width: 430, height: 30 }, { fontSize: 19, bold: true });
  addText(s, "2026-10-13", { left: 100, top: 638, width: 240, height: 24 }, { fontSize: 15, color: C.muted });
  addNotes(s, ["本次汇报以当前仓库中的三组实验和五篇论文为依据。", "重点不是展示一个完成的系统，而是明确下一步应该训练什么、为什么。"]);
}

// Slide 2
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "这次汇报先回答三个问题", "研究问题");
  const cards = [
    ["01", "Jev 能否替换\nSchema Linker？", "高表列召回是否等价于高 SQL coverage"],
    ["02", "Jev reranker 能否\n替换 self-consistency？", "固定同一候选池，比较 selector 本身"],
    ["03", "监督式 selector\n下一步怎样训练？", "从五篇 curriculum / ORM 论文得到方案"],
  ];
  cards.forEach((d, i) => {
    const x = 64 + i * 388;
    addBox(s, { left: x, top: 150, width: 350, height: 420 }, { fill: C.paper, line: i === 1 ? C.orange : C.teal, lineWidth: 1.4 });
    addText(s, d[0], { left: x + 28, top: 176, width: 70, height: 36 }, { fontSize: 18, bold: true, color: i === 1 ? C.orange : C.teal });
    addText(s, d[1], { left: x + 28, top: 245, width: 294, height: 110 }, { fontSize: 28, bold: true });
    addText(s, d[2], { left: x + 28, top: 392, width: 294, height: 80 }, { fontSize: 18, color: C.muted });
  });
  addBoxText(s, "判断标准：分开看 candidate coverage 与 selector accuracy given coverage", { left: 190, top: 610, width: 900, height: 45 }, { fill: C.ink, line: C.ink, color: C.white, fontSize: 19, bold: true });
  addFooter(s, 2);
  addNotes(s, ["Source: D:\\University\\Alpha-SQL\\PHASE2_EXPERIMENT_REPORT.md", "Source: D:\\University\\Alpha-SQL\\LLM_JEV_300_INTERIM_REPORT.md"]);
}

// Slide 3
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "项目最初想法：把生成能力与选择能力拆开", "系统设计");
  const y = 210;
  addBoxText(s, "问题 + 全量 schema", { left: 70, top: y, width: 190, height: 90 }, { fill: C.bluePale, line: C.blue, fontSize: 20, bold: true });
  addArrow(s, 278, y + 31);
  addBoxText(s, "Jev typed\nschema decisions", { left: 344, top: y, width: 196, height: 90 }, { fill: C.tealPale, line: C.teal, fontSize: 20, bold: true });
  addArrow(s, 556, y + 31);
  addBoxText(s, "QueryPlan beam\nPython 编译 SQL", { left: 622, top: y, width: 204, height: 90 }, { fill: C.orangePale, line: C.orange, fontSize: 20, bold: true });
  addArrow(s, 842, y + 31);
  addBoxText(s, "SQLite\n执行验证", { left: 908, top: y, width: 150, height: 90 }, { fill: C.grayPale, line: C.gray, fontSize: 20, bold: true });
  addArrow(s, 1074, y + 31, 40);
  addBoxText(s, "Jev\nrerank", { left: 1128, top: y, width: 96, height: 90 }, { fill: C.redPale, line: C.red, fontSize: 19, bold: true });

  addText(s, "独立对照路径", { left: 70, top: 365, width: 180, height: 28 }, { fontSize: 18, bold: true, color: C.muted });
  addBoxText(s, "Qwen / MCTS 生成候选池", { left: 70, top: 416, width: 300, height: 76 }, { fill: C.paper, line: C.blue, fontSize: 20, bold: true });
  addArrow(s, 390, 440, 56, 28, C.blue);
  addBoxText(s, "同一候选池", { left: 466, top: 416, width: 180, height: 76 }, { fill: C.paper, line: C.line, fontSize: 20, bold: true });
  addArrow(s, 666, 440, 56, 28, C.gray);
  addBoxText(s, "execution\nself-consistency", { left: 742, top: 397, width: 206, height: 114 }, { fill: C.greenPale, line: C.green, fontSize: 20, bold: true });
  addText(s, "VS", { left: 970, top: 436, width: 46, height: 30 }, { fontSize: 20, bold: true, color: C.muted, alignment: "center" });
  addBoxText(s, "Jev schema-aware\nreranker", { left: 1036, top: 397, width: 188, height: 114 }, { fill: C.redPale, line: C.red, fontSize: 20, bold: true });
  addBoxText(s, "关键：Jev 不是生成模型，也不替换现有 Qwen/OpenAI 路径", { left: 272, top: 585, width: 736, height: 50 }, { fill: C.ink, line: C.ink, color: C.white, fontSize: 19, bold: true });
  addFooter(s, 3);
  addNotes(s, ["Source: D:\\University\\Alpha-SQL\\PHASE2_EXPERIMENT_REPORT.md", "Source: D:\\University\\Alpha-SQL\\LLM_JEV_300_INTERIM_REPORT.md"]);
}

// Slide 4
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "三步实验路线：先定位问题，再训练 selector", "实验路线");
  s.shapes.add({ geometry: "rect", position: { left: 140, top: 309, width: 1000, height: 4 }, fill: C.line, line: { fill: "none", width: 0 } });
  const stages = [
    { x: 150, c: C.blue, pale: C.bluePale, n: "12题", t: "受约束探路", d: "验证 pipeline 可跑通\n观察 linker / compiler / reranker" },
    { x: 490, c: C.teal, pale: C.tealPale, n: "300题", t: "扩展与分解", d: "区分能力范围\n固定候选池比较 selector" },
    { x: 830, c: C.orange, pale: C.orangePale, n: "100题", t: "监督式 pairwise", d: "筛 hard cases\n保留 SC 特征并做 confidence gate" },
  ];
  stages.forEach((v) => {
    s.shapes.add({ geometry: "ellipse", position: { left: v.x + 118, top: 279, width: 64, height: 64 }, fill: v.c, line: { fill: C.white, width: 4 } });
    addBox(s, { left: v.x, top: 148, width: 300, height: 110 }, { fill: v.pale, line: v.c, lineWidth: 1.2 });
    addText(s, v.n, { left: v.x + 20, top: 166, width: 260, height: 38 }, { fontSize: 27, bold: true, color: v.c, alignment: "center" });
    addText(s, v.t, { left: v.x + 20, top: 208, width: 260, height: 28 }, { fontSize: 18, bold: true, alignment: "center" });
    addText(s, v.d, { left: v.x + 15, top: 380, width: 270, height: 98 }, { fontSize: 19, color: C.muted, alignment: "center" });
  });
  addBoxText(s, "结论演进：能跑通 → 找到真正瓶颈 → 证明监督信号存在但安全性不足", { left: 160, top: 560, width: 960, height: 60 }, { fill: C.paper, line: C.line, fontSize: 21, bold: true });
  addFooter(s, 4);
  addNotes(s, ["Source: D:\\University\\Alpha-SQL\\PHASE2_EXPERIMENT_REPORT.md", "Source: D:\\University\\Alpha-SQL\\HARD_CASE_PAIRWISE_100_REPORT.md", "Source: D:\\University\\Alpha-SQL\\LLM_JEV_300_INTERIM_REPORT.md"]);
}

// Slide 5
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "12题探路：Jev 有选择信号，但不能直接替换 SC", "阶段一");
  const table = s.tables.add({
    rows: 4, columns: 3, left: 64, top: 150, width: 740, height: 300,
    columnWidths: [360, 190, 190],
    values: [
      ["方法", "Coverage", "Final EX"],
      ["MCTS + self-consistency", "12/12", "12/12"],
      ["Jev constrained pipeline", "8/12", "8/12"],
      ["MCTS pool + Jev reranker", "12/12", "11/12"],
    ],
  });
  styleTable(table, { fontSize: 17, rowFills: [C.greenPale, C.tealPale, C.redPale] });
  addMetricChip(s, "5 → 8", "Jev 自有候选池：top-beam 到 rerank", { left: 850, top: 150, width: 360, height: 118 }, C.teal, C.tealPale);
  addMetricChip(s, "12 → 11", "固定 MCTS 池：一次改选造成回归", { left: 850, top: 292, width: 360, height: 118 }, C.red, C.redPale);
  addBoxText(s, "4 个 Jev coverage miss 均已召回正确表列，错误来自 DISTINCT 与 literal / 阈值组合", { left: 64, top: 492, width: 1146, height: 82 }, { fill: C.paper, line: C.orange, fontSize: 20, bold: true });
  addText(s, "小样本能证明 reranker 有信息，不能证明它比强 SC baseline 更可靠。", { left: 86, top: 604, width: 1080, height: 34 }, { fontSize: 19, color: C.muted, alignment: "center" });
  addFooter(s, 5);
  addNotes(s, ["Source: D:\\University\\Alpha-SQL\\PHASE2_EXPERIMENT_REPORT.md", "MCTS budget: 2 rollouts, max depth 5. The 12 questions were chosen within the constrained compiler slice."]);
}

// Slide 6
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "300题扩展：先把可比边界说清楚", "阶段二");
  addMetricChip(s, "300", "冻结、gold 可本地执行", { left: 70, top: 150, width: 240, height: 120 }, C.blue, C.bluePale);
  addArrow(s, 330, 194, 62, 30, C.gray);
  addMetricChip(s, "300/300", "Jev native 完成", { left: 412, top: 150, width: 240, height: 120 }, C.teal, C.tealPale);
  addArrow(s, 672, 194, 62, 30, C.gray);
  addMetricChip(s, "287/300", "MCTS 完成，共同集", { left: 754, top: 150, width: 240, height: 120 }, C.orange, C.orangePale);
  addArrow(s, 1014, 194, 62, 30, C.gray);
  addMetricChip(s, "13", "欠费/截断待补跑", { left: 1096, top: 150, width: 120, height: 120 }, C.red, C.redPale);

  addBoxText(s, "83 题", { left: 140, top: 374, width: 250, height: 90 }, { fill: C.tealPale, line: C.teal, color: C.teal, fontSize: 34, bold: true });
  addText(s, "当前 compiler 范围内", { left: 140, top: 474, width: 250, height: 34 }, { fontSize: 18, bold: true, alignment: "center" });
  addBoxText(s, "217 题", { left: 510, top: 374, width: 250, height: 90 }, { fill: C.orangePale, line: C.orange, color: C.orange, fontSize: 34, bold: true });
  addText(s, "包含尚未建模的 SQL 结构", { left: 490, top: 474, width: 290, height: 34 }, { fontSize: 18, bold: true, alignment: "center" });
  addBoxText(s, "287 题", { left: 880, top: 374, width: 250, height: 90 }, { fill: C.bluePale, line: C.blue, color: C.blue, fontSize: 34, bold: true });
  addText(s, "三路方法严格可比", { left: 880, top: 474, width: 250, height: 34 }, { fontSize: 18, bold: true, alignment: "center" });
  addBoxText(s, "报告口径：Jev 用完整 300；方法对比用共同 287；不把缺失 13 题伪装成最终结果", { left: 154, top: 574, width: 972, height: 58 }, { fill: C.ink, line: C.ink, color: C.white, fontSize: 19, bold: true });
  addFooter(s, 6);
  addNotes(s, ["Source: D:\\University\\Alpha-SQL\\LLM_JEV_300_INTERIM_REPORT.md", "The 13 missing MCTS questions are due to provider arrearage and five earlier output truncations. Strict 300-way comparison remains incomplete."]);
}

// Slide 7
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "共同287题：MCTS coverage 是优势，Jev 直接重排会下降", "核心结果");
  const chart = s.charts.add("bar", {
    position: { left: 64, top: 150, width: 770, height: 430 },
    categories: ["MCTS + SC", "Jev constrained", "MCTS pool + Jev"],
    series: [
      { name: "Candidate Coverage", values: [79.79, 29.27, 79.79], fill: C.blue },
      { name: "Final EX", values: [75.61, 25.44, 72.47], fill: C.teal },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 64 },
    legend: { position: "bottom", overlay: false },
    xAxis: { textStyle: { fontSize: 15, fill: C.muted }, line: { style: "solid", fill: C.line, width: 1 } },
    yAxis: { minimumScale: 0, maximumScale: 100, majorUnit: 20, numberFormatCode: "0", majorGridlines: { style: "solid", fill: C.line, width: 1 }, textStyle: { fontSize: 13, fill: C.muted } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 14, fill: C.ink, bold: true }, numberFormatCode: "0.0" },
  });
  applyPresentationChartFont(chart, { fontFamily: FONT });
  addMetricChip(s, "-3.14 pp", "Jev-on-MCTS 相对 SC", { left: 870, top: 158, width: 330, height: 120 }, C.red, C.redPale);
  addMetricChip(s, "[-5.92, -0.35]", "paired bootstrap 95% CI", { left: 870, top: 302, width: 330, height: 120 }, C.red, C.redPale);
  addBoxText(s, "当前证据支持：无监督 Jev reranker 不应直接替换 execution self-consistency", { left: 854, top: 474, width: 360, height: 104 }, { fill: C.paper, line: C.red, color: C.red, fontSize: 21, bold: true });
  addFooter(s, 7);
  addNotes(s, ["Source: D:\\University\\Alpha-SQL\\LLM_JEV_300_INTERIM_REPORT.md", "Percent values are reported as percentage points. McNemar/exact binomial p=0.049 on the 17 disagreements."]);
}

// Slide 8
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "Jev Schema Linking 很强，但这不等于 SQL 已经正确", "瓶颈诊断");
  const chart = s.charts.add("bar", {
    position: { left: 64, top: 152, width: 760, height: 420 },
    categories: ["Table Precision", "Table Recall", "Column Precision", "Column Recall"],
    series: [
      { name: "MCTS path evidence", values: [54.24, 53.98, 47.54, 52.86], fill: C.gray },
      { name: "Jev linker", values: [63.35, 99.30, 60.16, 97.35], fill: C.teal },
    ],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 48 },
    legend: { position: "bottom", overlay: false },
    xAxis: { minimumScale: 0, maximumScale: 100, majorUnit: 20, majorGridlines: { style: "solid", fill: C.line, width: 1 }, textStyle: { fontSize: 13, fill: C.muted } },
    yAxis: { textStyle: { fontSize: 14, fill: C.muted }, line: { style: "solid", fill: C.line, width: 1 } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 13, fill: C.ink, bold: true }, numberFormatCode: "0.0" },
  });
  applyPresentationChartFont(chart, { fontFamily: FONT });
  addMetricChip(s, "181 / 203", "coverage miss 仍完整召回 gold 表列", { left: 870, top: 165, width: 330, height: 126 }, C.teal, C.tealPale);
  addMetricChip(s, "22 / 203", "真正 schema recall 不完整", { left: 870, top: 320, width: 330, height: 126 }, C.orange, C.orangePale);
  addBoxText(s, "资源优先级：QueryPlan / compiler / literal / operator  >  再训练表列二分类器", { left: 82, top: 606, width: 1116, height: 48 }, { fill: C.ink, line: C.ink, color: C.white, fontSize: 19, bold: true });
  addFooter(s, 8);
  addNotes(s, ["Source: D:\\University\\Alpha-SQL\\LLM_JEV_300_INTERIM_REPORT.md", "MCTS schema evidence is not a pure linker baseline because some stored paths skip the schema-selection action."]);
}

// Slide 9
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "Jev constrained 的主瓶颈是能力范围，不是语义召回", "能力范围");
  const chart = s.charts.add("bar", {
    position: { left: 64, top: 145, width: 700, height: 440 },
    categories: ["compiler 范围内（83）", "范围外（217）"],
    series: [
      { name: "Coverage", values: [77.11, 11.98], fill: C.blue },
      { name: "Final EX", values: [74.70, 7.37], fill: C.teal },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 62 },
    legend: { position: "bottom", overlay: false },
    xAxis: { textStyle: { fontSize: 15, fill: C.muted }, line: { style: "solid", fill: C.line, width: 1 } },
    yAxis: { minimumScale: 0, maximumScale: 100, majorUnit: 20, majorGridlines: { style: "solid", fill: C.line, width: 1 }, textStyle: { fontSize: 13, fill: C.muted } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 14, fill: C.ink, bold: true }, numberFormatCode: "0.0" },
  });
  applyPresentationChartFont(chart, { fontFamily: FONT });
  addText(s, "当前尚未完整建模", { left: 825, top: 154, width: 350, height: 34 }, { fontSize: 23, bold: true, color: C.orange });
  addBulletList(s, ["多投影、多过滤与复杂布尔条件", "子查询与集合操作", "GROUP BY / HAVING", "ORDER BY / LIMIT", "CASE、日期与字符串表达式", "literal、类型转换与聚合层级"], { left: 830, top: 210, width: 360, height: 305 }, { fontSize: 19, spaceAfterPoints: 9 });
  addBoxText(s, "范围内 EX 74.7%：说明 12 题结果不是偶然，但系统还不是通用生成器", { left: 794, top: 540, width: 410, height: 70 }, { fill: C.orangePale, line: C.orange, color: C.orange, fontSize: 19, bold: true });
  addFooter(s, 9);
  addNotes(s, ["Source: D:\\University\\Alpha-SQL\\LLM_JEV_300_INTERIM_REPORT.md", "The slice definition follows the original 12-question experiment scope."]);
}

// Slide 10
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "Selector 的真实差距：修复 4 题，但回归 13 题", "选择器");
  const chart = s.charts.add("bar", {
    position: { left: 64, top: 152, width: 620, height: 350 },
    categories: ["Self-consistency", "Jev-on-MCTS"],
    series: [{ name: "Final EX", values: [75.61, 72.47], fill: C.teal }],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 90 },
    hasLegend: false,
    xAxis: { textStyle: { fontSize: 16, fill: C.muted }, line: { style: "solid", fill: C.line, width: 1 } },
    yAxis: { minimumScale: 68, maximumScale: 78, majorUnit: 2, majorGridlines: { style: "solid", fill: C.line, width: 1 }, textStyle: { fontSize: 13, fill: C.muted } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 16, fill: C.ink, bold: true }, numberFormatCode: "0.00" },
  });
  applyPresentationChartFont(chart, { fontFamily: FONT });
  addMetricChip(s, "4", "Jev 修复", { left: 750, top: 160, width: 200, height: 118 }, C.green, C.greenPale);
  addMetricChip(s, "13", "Jev 回归", { left: 980, top: 160, width: 200, height: 118 }, C.red, C.redPale);
  addMetricChip(s, "94.76%", "SC 在 coverage 内选对", { left: 750, top: 305, width: 200, height: 118 }, C.blue, C.bluePale);
  addMetricChip(s, "90.83%", "Jev 在 coverage 内选对", { left: 980, top: 305, width: 200, height: 118 }, C.orange, C.orangePale);
  addBoxText(s, "GapClosure = -75%", { left: 166, top: 530, width: 330, height: 70 }, { fill: C.redPale, line: C.red, color: C.red, fontSize: 28, bold: true });
  addText(s, "不是缺少改选能力，而是缺少“何时不要改选”的监督。", { left: 530, top: 541, width: 670, height: 48 }, { fontSize: 20, bold: true, color: C.ink, alignment: "center" });
  addFooter(s, 10);
  addNotes(s, ["Source: D:\\University\\Alpha-SQL\\LLM_JEV_300_INTERIM_REPORT.md", "GapClosure uses the 12-question oracle-to-SC selector gap as denominator: -9/12 = -75%."]);
}

// Slide 11
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "错误不是随机的：通用语义与 benchmark 约定发生冲突", "逐题分析");
  addBox(s, { left: 64, top: 146, width: 542, height: 470 }, { fill: C.greenPale, line: C.green, lineWidth: 1.2 });
  addText(s, "修复的 4 题", { left: 92, top: 171, width: 220, height: 40 }, { fontSize: 27, bold: true, color: C.green });
  addBulletList(s, ["210：COUNT(*) → SUM(qty)，修复计量对象", "3099：>= → >，匹配严格比较", "8963：US$ 字符串先转数值再求最大", "2481：修复年份上界的执行约定"], { left: 92, top: 234, width: 474, height: 250 }, { fontSize: 19, spaceAfterPoints: 13 });
  addText(s, "共同点：类型、比较符、聚合对象等可验证语义", { left: 92, top: 528, width: 474, height: 46 }, { fontSize: 18, bold: true, color: C.green, alignment: "center" });

  addBox(s, { left: 674, top: 146, width: 542, height: 470 }, { fill: C.redPale, line: C.red, lineWidth: 1.2 });
  addText(s, "回归的 13 题", { left: 702, top: 171, width: 240, height: 40 }, { fontSize: 27, bold: true, color: C.red });
  addBulletList(s, ["5 题添加了 gold 不需要的 DISTINCT", "投影从 ID 改成更自然的姓名 / 拼接字段", "增加额外语义条件或换用相近字段", "改变 GROUP BY、计数对象或百分比分母", "从另一张语义相近表读取结果"], { left: 702, top: 234, width: 474, height: 270 }, { fontSize: 19, spaceAfterPoints: 10 });
  addText(s, "共同点：SQL 看起来更自然，但 execution gold 不等价", { left: 702, top: 528, width: 474, height: 46 }, { fontSize: 18, bold: true, color: C.red, alignment: "center" });
  addFooter(s, 11);
  addNotes(s, ["Source: D:\\University\\Alpha-SQL\\LLM_JEV_300_INTERIM_REPORT.md"]);
}

// Slide 12
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "100题监督式 pairwise：证明信号存在，数据仍远远不够", "阶段三");
  addMetricChip(s, "269", "可评估问题", { left: 64, top: 142, width: 205, height: 112 }, C.blue, C.bluePale);
  addMetricChip(s, "903", "候选 SQL", { left: 287, top: 142, width: 205, height: 112 }, C.teal, C.tealPale);
  addMetricChip(s, "17", "全部 hard cases", { left: 510, top: 142, width: 205, height: 112 }, C.orange, C.orangePale);
  addMetricChip(s, "9", "训练 hard cases", { left: 733, top: 142, width: 205, height: 112 }, C.red, C.redPale);
  addMetricChip(s, "129/40/100", "train / val / test", { left: 956, top: 142, width: 260, height: 112 }, C.ink, C.grayPale);

  addBoxText(s, "48维特征\n含 7 个 SC 特征", { left: 100, top: 365, width: 220, height: 100 }, { fill: C.paper, line: C.blue, fontSize: 20, bold: true });
  addArrow(s, 342, 401, 62, 30, C.gray);
  addBoxText(s, "MLP ensemble\nRankNet + pointwise", { left: 426, top: 365, width: 255, height: 100 }, { fill: C.paper, line: C.teal, fontSize: 20, bold: true });
  addArrow(s, 703, 401, 62, 30, C.gray);
  addBoxText(s, "validation 选阈值\nscore margin ≥ 0.20", { left: 787, top: 365, width: 255, height: 100 }, { fill: C.paper, line: C.orange, fontSize: 20, bold: true });
  addArrow(s, 1064, 401, 44, 30, C.gray);
  addBoxText(s, "否则回退 SC", { left: 1120, top: 365, width: 96, height: 100 }, { fill: C.greenPale, line: C.green, fontSize: 18, bold: true });
  addBoxText(s, "训练目标很窄：只解决候选中“正确 SQL 存在但多数簇选错”的问题", { left: 184, top: 551, width: 912, height: 60 }, { fill: C.ink, line: C.ink, color: C.white, fontSize: 20, bold: true });
  addFooter(s, 12);
  addNotes(s, ["Source: D:\\University\\Alpha-SQL\\HARD_CASE_PAIRWISE_100_REPORT.md", "The split is frozen by question ID but is not database-disjoint."]);
}

// Slide 13
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "监督实验：hard 子集修复 4/5，但被普通题回归抵消", "100题结果");
  const chart = s.charts.add("bar", {
    position: { left: 64, top: 150, width: 650, height: 355 },
    categories: ["Self-consistency", "Pairwise + SC", "Oracle"],
    series: [{ name: "Test EX", values: [76, 75, 81], fill: C.teal }],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 78 },
    hasLegend: false,
    xAxis: { textStyle: { fontSize: 15, fill: C.muted }, line: { style: "solid", fill: C.line, width: 1 } },
    yAxis: { minimumScale: 68, maximumScale: 84, majorUnit: 4, majorGridlines: { style: "solid", fill: C.line, width: 1 }, textStyle: { fontSize: 13, fill: C.muted } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 16, fill: C.ink, bold: true }, numberFormatCode: "0" },
  });
  applyPresentationChartFont(chart, { fontFamily: FONT });
  addMetricChip(s, "4", "修复", { left: 785, top: 158, width: 180, height: 112 }, C.green, C.greenPale);
  addMetricChip(s, "5", "回归", { left: 995, top: 158, width: 180, height: 112 }, C.red, C.redPale);
  addMetricChip(s, "19", "coverage miss", { left: 785, top: 296, width: 180, height: 112 }, C.orange, C.orangePale);
  addMetricChip(s, "1", "未修复 selection", { left: 995, top: 296, width: 180, height: 112 }, C.blue, C.bluePale);
  addBoxText(s, "5 个回归全部从较大结果簇跳到 singleton", { left: 770, top: 446, width: 420, height: 66 }, { fill: C.redPale, line: C.red, color: C.red, fontSize: 21, bold: true });
  addBoxText(s, "结论：模型学到了“反多数簇”信号，但还没学会安全边界", { left: 165, top: 559, width: 950, height: 60 }, { fill: C.paper, line: C.line, fontSize: 21, bold: true });
  addFooter(s, 13);
  addNotes(s, ["Source: D:\\University\\Alpha-SQL\\HARD_CASE_PAIRWISE_100_REPORT.md", "Primary model delta vs SC is -1 point; paired bootstrap 95% CI [-7, +5] points."]);
}

// Slide 14
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "五篇论文：从“样本顺序”走向“SQL outcome verifier”", "文献学习");
  const covers = [
    ["curriculum-cover.png", "Bengio et al.\n2009", "顺序改变优化路径"],
    ["beyond-cover.png", "Beyond Random\nSampling, 2026", "warmup 最稳健"],
    ["difficulty-cover2.png", "DUCL\n2026", "难度 + 效用 + 多样性"],
    ["alignment-cover.png", "Curri-DPO\n2024", "多 pair 从易到难"],
    ["gradesql-cover.png", "GradeSQL\n2025", "执行标签训练 ORM"],
  ];
  const dir = path.join(workspaceDir, "tmp", "pdfs", "papers", "renders");
  for (let i = 0; i < covers.length; i++) {
    const x = 70 + i * 236;
    const img = await fs.readFile(path.join(dir, covers[i][0]));
    addBox(s, { left: x - 5, top: 140, width: 200, height: 325 }, { fill: C.paper, line: C.line });
    s.images.add({
      blob: img.buffer.slice(img.byteOffset, img.byteOffset + img.byteLength),
      contentType: "image/png",
      alt: covers[i][1], fit: "contain",
      position: { left: x + 3, top: 150, width: 184, height: 242 },
    });
    addText(s, covers[i][1], { left: x, top: 405, width: 190, height: 46 }, { fontSize: 15, bold: true, alignment: "center" });
    addText(s, covers[i][2], { left: x - 4, top: 500, width: 198, height: 54 }, { fontSize: 17, bold: true, color: i < 3 ? C.teal : C.orange, alignment: "center" });
  }
  addBoxText(s, "共同主线：先定义什么值得学，再决定以什么顺序、什么强度学习", { left: 214, top: 590, width: 852, height: 54 }, { fill: C.ink, line: C.ink, color: C.white, fontSize: 20, bold: true });
  addFooter(s, 14);
  addNotes(s, ["Sources:", "E:\\Text2SQL\\Curriculum Learning.pdf", "E:\\Text2SQL\\Beyond Random Sampling.pdf", "E:\\Text2SQL\\Difficulty Is Not Enough.pdf", "E:\\Text2SQL\\Enhancing Alignment using Curriculum Learning.pdf", "E:\\Text2SQL\\GradeSQL.pdf"]);
}

// Slide 15
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "固定数据下，课程学习改变的是优化路径", "论文启发 I");
  const nodes = [
    [80, C.bluePale, C.blue, "Bengio", "从容易目标逐步过渡\n像 continuation method"],
    [350, C.tealPale, C.teal, "Beyond Random Sampling", "难度指标 × pacing\n课程 warmup 更稳"],
    [650, C.orangePale, C.orange, "DUCL", "difficulty 不够\n还要估计 utility"],
    [950, C.greenPale, C.green, "Window Ordering", "逐步扩大窗口\n保留 batch diversity"],
  ];
  nodes.forEach((n, i) => {
    addBoxText(s, n[3], { left: n[0], top: 180, width: i === 1 ? 250 : 220, height: 70 }, { fill: n[1], line: n[2], color: n[2], fontSize: i === 1 ? 19 : 21, bold: true });
    addText(s, n[4], { left: n[0] - 5, top: 280, width: i === 1 ? 260 : 230, height: 80 }, { fontSize: 18, color: C.muted, alignment: "center" });
    if (i < 3) addArrow(s, n[0] + (i === 1 ? 263 : 233), 202, 42, 26, C.gray);
  });
  addBox(s, { left: 100, top: 430, width: 1080, height: 150 }, { fill: C.paper, line: C.line });
  addText(s, "映射到 SQL selector", { left: 136, top: 455, width: 240, height: 36 }, { fontSize: 25, bold: true, color: C.ink });
  addBulletList(s, ["难度 D：当前模型有多难区分这对 SQL", "效用 U：现在学习它能否减少尚未解决的错误", "噪声 N：持续高损失但无学习进展的样本应降权"], { left: 405, top: 445, width: 720, height: 110 }, { fontSize: 18, spaceAfterPoints: 6 });
  addFooter(s, 15);
  addNotes(s, ["Source: E:\\Text2SQL\\Curriculum Learning.pdf", "Source: E:\\Text2SQL\\Beyond Random Sampling.pdf", "Source: E:\\Text2SQL\\Difficulty Is Not Enough.pdf"]);
}

// Slide 16
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "Curri-DPO 与 GradeSQL：最接近当前 selector 的两条路线", "论文启发 II");
  addBox(s, { left: 64, top: 145, width: 550, height: 430 }, { fill: C.orangePale, line: C.orange, lineWidth: 1.2 });
  addText(s, "Curri-DPO", { left: 92, top: 170, width: 210, height: 40 }, { fontSize: 28, bold: true, color: C.orange });
  addText(s, "同一 prompt 的多个响应形成多个 preference pairs", { left: 92, top: 220, width: 470, height: 42 }, { fontSize: 18, color: C.muted });
  addBoxText(s, "大评分差\n容易 pair", { left: 92, top: 300, width: 130, height: 78 }, { fill: C.paper, line: C.orange, fontSize: 18, bold: true });
  addArrow(s, 238, 325, 42, 26, C.orange);
  addBoxText(s, "中等差异", { left: 296, top: 300, width: 120, height: 78 }, { fill: C.paper, line: C.orange, fontSize: 18, bold: true });
  addArrow(s, 432, 325, 42, 26, C.orange);
  addBoxText(s, "小评分差\n困难 pair", { left: 490, top: 300, width: 100, height: 78 }, { fill: C.paper, line: C.orange, fontSize: 17, bold: true });
  addText(s, "启发：不需要穷举所有 pair；迭代次序与 reference 更新很重要。", { left: 92, top: 442, width: 470, height: 78 }, { fontSize: 19, bold: true, color: C.orange, alignment: "center" });

  addBox(s, { left: 666, top: 145, width: 550, height: 430 }, { fill: C.tealPale, line: C.teal, lineWidth: 1.2 });
  addText(s, "GradeSQL", { left: 694, top: 170, width: 210, height: 40 }, { fontSize: 28, bold: true, color: C.teal });
  addText(s, "候选 SQL 用执行等价打标签，训练 ORM 输出 P(Yes)", { left: 694, top: 220, width: 470, height: 42 }, { fontSize: 18, color: C.muted });
  addBoxText(s, "N 个候选", { left: 694, top: 300, width: 112, height: 78 }, { fill: C.paper, line: C.teal, fontSize: 18, bold: true });
  addArrow(s, 820, 325, 40, 26, C.teal);
  addBoxText(s, "执行标签", { left: 874, top: 300, width: 112, height: 78 }, { fill: C.paper, line: C.teal, fontSize: 18, bold: true });
  addArrow(s, 1000, 325, 40, 26, C.teal);
  addBoxText(s, "ORM 排序", { left: 1054, top: 300, width: 112, height: 78 }, { fill: C.paper, line: C.teal, fontSize: 18, bold: true });
  addText(s, "BIRD dev：68.90%，高于 Majority Voting 66.95%；SQL-only prompt 最好。", { left: 694, top: 442, width: 470, height: 78 }, { fontSize: 19, bold: true, color: C.teal, alignment: "center" });
  addBoxText(s, "本阶段不增加样本：GradeSQL 给执行标签，Curri-DPO 给 pair 顺序", { left: 197, top: 603, width: 886, height: 48 }, { fill: C.ink, line: C.ink, color: C.white, fontSize: 20, bold: true });
  addFooter(s, 16);
  addNotes(s, ["Source: E:\\Text2SQL\\Enhancing Alignment using Curriculum Learning.pdf", "Source: E:\\Text2SQL\\GradeSQL.pdf", "GradeSQL reports BIRD dev ORM-BoN 68.90, Majority Voting 66.95, ex-BoN 66.04, Pass@32 80.57."]);
}

// Slide 17
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "下一阶段：固定 training set 的错误感知课程学习", "方案");
  const phases = [
    [70, C.bluePale, C.blue, "阶段 A", "冻结实验条件", "同一数据、模型、初始化、预算与种子"],
    [365, C.tealPale, C.teal, "阶段 B", "诊断 warmup", "记录 loss / margin\n与错误类型"],
    [660, C.orangePale, C.orange, "阶段 C", "动态错误课程", "扩大难度窗口；优先尚未解决且可学习的错误"],
    [955, C.greenPale, C.green, "阶段 D", "随机混合巩固", "最后 10%–20% 恢复近似均匀采样"],
  ];
  phases.forEach((v, i) => {
    addBox(s, { left: v[0], top: 155, width: 255, height: 190 }, { fill: v[1], line: v[2], lineWidth: 1.2 });
    addText(s, v[3], { left: v[0] + 20, top: 176, width: 215, height: 28 }, { fontSize: 16, bold: true, color: v[2] });
    addText(s, v[4], { left: v[0] + 20, top: 220, width: 215, height: 42 }, { fontSize: 23, bold: true, alignment: "center" });
    addText(s, v[5], { left: v[0] + 22, top: 280, width: 211, height: 48 }, { fontSize: 16, color: C.muted, alignment: "center" });
    if (i < 3) addArrow(s, v[0] + 263, 238, 24, 22, C.gray);
  });
  addBoxText(s, "priorityᵢ(t) = windowₜ(Dᵢ) · exp(βUᵢ(t) − γNᵢ(t))", { left: 112, top: 398, width: 650, height: 72 }, { fill: C.paper, line: C.teal, color: C.teal, fontSize: 23, bold: true });
  addBoxText(s, "唯一主变量\n样本顺序与采样概率", { left: 812, top: 398, width: 340, height: 72 }, { fill: C.redPale, line: C.red, color: C.red, fontSize: 19, bold: true });
  addBulletList(s, ["错误标签：projection / filter / join / aggregation / DISTINCT / ORDER-LIMIT / literal-type-date", "动态量：当前 loss 与 margin、近期学习进展、validation 未解决错误频率", "约束：相同 optimizer steps、tokens、样本曝光次数；test 标签不参与 utility"], { left: 128, top: 505, width: 1024, height: 128 }, { fontSize: 17.5, spaceAfterPoints: 5 });
  addFooter(s, 17);
  addNotes(s, ["Sources: D:\\University\\Alpha-SQL\\HARD_CASE_PAIRWISE_100_REPORT.md; E:\\Text2SQL\\Difficulty Is Not Enough.pdf; E:\\Text2SQL\\Enhancing Alignment using Curriculum Learning.pdf; E:\\Text2SQL\\GradeSQL.pdf"]);
}

// Slide 18
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "实验矩阵：同一数据、同一预算，只比较训练策略", "评估设计");
  const table = s.tables.add({
    rows: 6, columns: 4, left: 52, top: 132, width: 1176, height: 398,
    columnWidths: [210, 370, 155, 441],
    values: [
      ["方法", "调度方式", "动态更新", "检验的问题"],
      ["Random Shuffle", "全程均匀随机", "否", "标准训练基线"],
      ["Difficulty-only", "按 loss / margin / AST 距离由易到难", "可选", "只看难度是否足够"],
      ["Static Error-aware", "错误类型 + 初始效用固定排序", "否", "错误标签本身是否有效"],
      ["DUCL-style", "difficulty + utility + expanding window", "部分", "效用与窗口带来多少收益"],
      ["Dynamic Error-aware", "周期重评分 + interleaving + anchor replay", "是", "本文完整方案"],
    ],
  });
  styleTable(table, { fontSize: 14.5, rowFills: [C.bluePale, C.tealPale, C.orangePale, C.greenPale, C.redPale] });
  addBoxText(s, "训练效率", { left: 72, top: 566, width: 160, height: 54 }, { fill: C.ink, line: C.ink, color: C.white, fontSize: 20, bold: true });
  addText(s, "steps / GPU-hours 到目标分数 · learning-curve AUC · 每千步增益", { left: 250, top: 569, width: 440, height: 48 }, { fontSize: 17.5, bold: true, color: C.ink, verticalAlignment: "middle" });
  addBoxText(s, "成功标准", { left: 720, top: 566, width: 160, height: 54 }, { fill: C.teal, line: C.teal, color: C.white, fontSize: 20, bold: true });
  addText(s, "少用 ≥20% 步数达到基线；或同预算最终 EX 稳定提高", { left: 898, top: 569, width: 310, height: 48 }, { fontSize: 17, bold: true, color: C.ink, verticalAlignment: "middle" });
  addFooter(s, 18);
  addNotes(s, ["Sources: D:\\University\\Alpha-SQL\\HARD_CASE_PAIRWISE_100_REPORT.md; E:\\Text2SQL\\Difficulty Is Not Enough.pdf; E:\\Text2SQL\\GradeSQL.pdf"]);
}

// Slide 19
{
  const s = p.slides.add(); s.background.fill = C.bg; addHeader(s, "研究问题与预期贡献", "下一阶段结论");
  const conclusions = [
    [C.teal, C.tealPale, "RQ1 · 更快吗", "相同数据下，错误感知课程能否更早达到随机训练的最终分数？"],
    [C.orange, C.orangePale, "RQ2 · 为何有效", "utility 是否比 difficulty 更能区分可学习 hard case 与噪声？"],
    [C.red, C.redPale, "RQ3 · 更稳吗", "动态重评分、窗口与 anchor replay 能否减少回归和遗忘？"],
    [C.blue, C.bluePale, "贡献", "错误类型体系 + 动态调度器 + 固定数据/预算的可复现实验协议。"],
  ];
  conclusions.forEach((v, i) => {
    const y = 142 + i * 118;
    addBox(s, { left: 96, top: y, width: 1088, height: 92 }, { fill: v[1], line: v[0], lineWidth: 1.2 });
    addText(s, v[2], { left: 124, top: y + 24, width: 245, height: 40 }, { fontSize: 23, bold: true, color: v[0] });
    addText(s, v[3], { left: 390, top: y + 20, width: 755, height: 52 }, { fontSize: 20, bold: true, verticalAlignment: "middle" });
  });
  addBoxText(s, "执行顺序：冻结 split 与预算 → 跑基线曲线 → 标注错误类型 → 比较课程与消融", { left: 130, top: 620, width: 1020, height: 52 }, { fill: C.ink, line: C.ink, color: C.white, fontSize: 20, bold: true });
  addFooter(s, 19);
  addNotes(s, ["Sources: D:\\University\\Alpha-SQL\\LLM_JEV_300_INTERIM_REPORT.md; D:\\University\\Alpha-SQL\\HARD_CASE_PAIRWISE_100_REPORT.md; E:\\Text2SQL\\GradeSQL.pdf"]);
}

await fs.mkdir(previewDir, { recursive: true });
for (const [i, slide] of p.slides.items.entries()) {
  const png = await p.export({ slide, format: "png", scale: 1 });
  await writeBlob(path.join(previewDir, `slide-${String(i + 1).padStart(2, "0")}.png`), png);
}
const montage = await p.export({ format: "webp", montage: true });
await writeBlob(path.join(buildDir, "montage-v4.webp"), montage);

const requirements = {
  explicitTotalSlideCount: 19,
  requiredNativeTableOwnerSlides: [5, 18],
  requiredNativeChartOwnerSlides: [7, 8, 9, 10, 13],
  materializeLiteralChartWorkbooks: true,
  nativeChartTargetApplication: "powerpoint",
};
const fontPolicy = { basis: "design", families: [FONT, "Aptos"] };
const stagingDir = path.join(workspaceDir, ".codex-finalizer-group-meeting-v4");
await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });
const candidatePath = path.join(stagingDir, "candidate-v4.pptx");
await (await PresentationFile.exportPptx(p)).save(candidatePath);

const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    "--require-native-table-slide", "5",
    "--require-native-table-slide", "18",
  ],
  requiredNativeTableOwnerSlides: requirements.requiredNativeTableOwnerSlides,
  requiredNativeChartOwnerSlides: requirements.requiredNativeChartOwnerSlides,
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "jev_schema_aware_group_meeting_zh_v4.validation.json"),
});

console.log(JSON.stringify({ finalPath, slideCount: p.slides.items.length, result }, null, 2));

