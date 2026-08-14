import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const workspace = "E:/mumps/supergpu";
const sourcePath = path.join(workspace, "data", "result.md");
const zhPath = path.join(workspace, "data", "results_zh.csv");
const enPath = path.join(workspace, "data", "results_en.csv");
const previewDir = path.join(workspace, ".codex-tmp", "results-csv");

const columns = [
  ["matrixDimension", "Matrix dimension N", "矩阵维度 N"],
  ["matchingMode", "Maximum matching", "最大匹配"],
  ["inputNnz", "Input nonzeros", "输入非零元数"],
  ["metisFillIn", "METIS symbolic fill-in", "METIS 符号填充数"],
  ["metisFillRatio", "METIS fill-in ratio", "METIS 填充比"],
  ["selectedOrdering", "Selected ordering method", "选用排序方法"],
  ["selectedLNonzeros", "Selected symbolic L nonzeros", "选定符号因子 L 非零元数"],
  ["symbolicCache", "Symbolic cache", "符号缓存"],
  ["symOrderingAnalysisMs", "CHOLMOD ordering and symbolic analysis (ms)", "CHOLMOD 排序与符号分析（毫秒）"],
  ["symPermutationMs", "Symmetric CSC permutation (ms)", "对称 CSC 置换（毫秒）"],
  ["symPipelineMs", "Symbolic pipeline incl. permutation (ms)", "含置换的符号流水线（毫秒）"],
  ["eliminationTreeColumns", "Elimination tree columns", "消去树列数"],
  ["basicSupernodes", "Basic supernodes", "基本超节点数"],
  ["gpuSingleColumnNodes", "GPU single-column nodes", "GPU 单列节点数"],
  ["gpu2To64Nodes", "GPU 2-64 column nodes", "GPU 2 至 64 列节点数"],
  ["gpuOver64Nodes", "GPU >64 column nodes", "GPU 大于 64 列节点数"],
  ["accepted1x1Pivots", "Accepted 1x1 pivots", "接受的 1×1 主元数"],
  ["accepted2x2Blocks", "Accepted 2x2 pivot blocks", "接受的 2×2 主元块数"],
  ["delayedPivotEvents", "Delayed pivot events", "延迟主元事件数"],
  ["symTreeWaves", "Symmetric tree waves", "对称树波次数"],
  ["inputPreprocessUploadMs", "Input preprocessing and upload (ms)", "输入预处理与上传（毫秒）"],
  ["symFrontAssemblyMs", "Symmetric front assembly enqueue (ms)", "对称前沿装配入队（毫秒）"],
  ["contributionReleaseMs", "Contribution release/barrier (ms)", "贡献释放与屏障（毫秒）"],
  ["smallMediumFactorMs", "Single/small/medium factor (ms)", "单列/小型/中型分解（毫秒）"],
  ["largePanelFactorMs", "Large panel factor (ms)", "大型面板分解（毫秒）"],
  ["factorSaveMs", "Factor save/contribution extraction (ms)", "因子保存与贡献提取（毫秒）"],
  ["timedStageTotalMs", "Timed stage total (ms)", "计时阶段总计（毫秒）"],
  ["unattributedSchedulingMs", "Unattributed scheduling (ms)", "未归因调度时间（毫秒）"],
  ["ldltFactorMs", "GPU LDLT factorization (ms)", "GPU LDLT 分解（毫秒）"],
  ["ldltSolveMs", "GPU LDLT solve (ms)", "GPU LDLT 求解（毫秒）"],
  ["ldltAbsResidual", "LDLT absolute residual", "LDLT 绝对残差"],
  ["ldltRelResidual", "LDLT relative residual", "LDLT 相对残差"],
  ["ldltBackwardError", "LDLT scaled backward error", "LDLT 缩放后向误差"],
  ["ldltSolutionError", "LDLT relative solution error", "LDLT 相对解误差"],
  ["ldltTolerance", "FP32 verification tolerance", "FP32 验证容差"],
  ["ldltVerification", "LDLT residual verification", "LDLT 残差验证"],
  ["luPreprocessing", "LU preprocessing", "LU 预处理"],
  ["luMatchingRank", "LU structural matching rank", "LU 结构匹配秩"],
  ["luOrderingMs", "LU matching/scaling/ordering (ms)", "LU 匹配/缩放/排序（毫秒）"],
  ["luPermutationMs", "LU row/column permutation (ms)", "LU 行列置换（毫秒）"],
  ["luSymbolicAnalysisMs", "In-project LU symbolic analysis (ms)", "项目内 LU 符号分析（毫秒）"],
  ["luEnvelopeOffDiagonal", "LU envelope off-diagonal entries", "LU 包络非对角元素数"],
  ["luSymbolicNonzeros", "LU symbolic factor nonzeros", "LU 符号因子非零元数"],
  ["luSupernodes", "LU supernodes", "LU 超节点数"],
  ["luSingleColumnFronts", "GPU LU single-column fronts", "GPU LU 单列前沿数"],
  ["luBatchedFronts", "GPU LU batched small/medium fronts", "GPU LU 批处理小型/中型前沿数"],
  ["luPanelFronts", "GPU LU panel fronts", "GPU LU 面板前沿数"],
  ["luAcceptedPivots", "Accepted LU pivots", "接受的 LU 主元数"],
  ["luDelayedColumns", "Delayed LU columns", "延迟 LU 列数"],
  ["luTreeWaves", "LU tree waves", "LU 树波次数"],
  ["luFrontAssemblyMs", "LU front assembly (ms)", "LU 前沿装配（毫秒）"],
  ["luBatchedWallMs", "Batched small/medium LU wall time (ms)", "批处理小型/中型 LU 墙钟时间（毫秒）"],
  ["luLargeWorkMs", "Large-front GPU work sum (ms)", "大型前沿 GPU 工作总计（毫秒）"],
  ["luFactorWallMs", "GPU LU factorization wall time (ms)", "GPU LU 分解墙钟时间（毫秒）"],
  ["luSolveMs", "GPU LU solve (ms)", "GPU LU 求解（毫秒）"],
  ["luRefinementSteps", "LU iterative refinement steps", "LU 迭代改进步数"],
  ["luInitialResidual", "LU initial residual before refinement", "LU 改进前初始残差"],
  ["luAbsResidual", "LU absolute residual", "LU 绝对残差"],
  ["luBackwardError", "LU scaled backward error", "LU 缩放后向误差"],
  ["luSolutionError", "LU relative solution error", "LU 相对解误差"],
  ["luVerification", "LU residual verification", "LU 残差验证"],
];

function escapeRegex(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function rawValue(text, label) {
  const match = text.match(new RegExp(`^${escapeRegex(label)} = (.+)$`, "m"));
  return match ? match[1].trim() : null;
}

function numericValue(text, label) {
  const raw = rawValue(text, label);
  if (raw === null) return null;
  const match = raw.match(/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/);
  return match ? Number(match[0]) : null;
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(headers, rows) {
  return [headers, ...rows]
    .map((row) => row.map(csvCell).join(","))
    .join("\r\n") + "\r\n";
}

function localizedValue(key, value, language) {
  if (value === null || value === undefined) return "";
  if (language === "en") return value;
  const translations = {
    matchingMode: { Enabled: "启用", Disabled: "禁用" },
    selectedOrdering: { "CHOLMOD-METIS fast path": "CHOLMOD-METIS 快速路径" },
    symbolicCache: { hit: "命中", "miss; cache updated": "未命中（已更新缓存）" },
    ldltVerification: { PASS: "通过", FAIL: "失败" },
    luVerification: { PASS: "通过", FAIL: "失败" },
    luPreprocessing: {
      "Duff-Koster matching -> AMD": "Duff-Koster 匹配 → AMD",
      "AMD only (maximum matching disabled)": "仅 AMD（已禁用最大匹配）",
    },
  };
  return translations[key]?.[value] ?? value;
}

const markdown = await fs.readFile(sourcePath, "utf8");
const blocks = [...markdown.matchAll(/```cpp\s*([\s\S]*?)```/g)].map((match) => match[1]);

if (blocks.length !== 10) {
  throw new Error(`Expected 10 result blocks, found ${blocks.length}`);
}

const records = blocks.map((block) => {
  const [symmetric, lu = ""] = block.split(/\r?\n=== General\/unsymmetric sparse LU ===\r?\n/);
  const matrixMatch = block.match(/^Matrix = (\d+)x(\d+), nonzero = (\d+)$/m);
  if (!matrixMatch) throw new Error("Could not parse matrix dimensions from a result block");
  const matchingEnabled = !block.includes("--disable-matching");
  const rankRaw = rawValue(lu, "Structural matching rank");
  const rank = rankRaw?.match(/^\d+/)?.[0];

  return {
    matrixDimension: Number(matrixMatch[1]),
    matchingMode: matchingEnabled ? "Enabled" : "Disabled",
    inputNnz: Number(matrixMatch[3]),
    metisFillIn: numericValue(symmetric, "METIS symbolic fill-in"),
    metisFillRatio: numericValue(symmetric, "METIS symbolic fill-in") === null
      ? null
      : Number(rawValue(symmetric, "METIS symbolic fill-in").match(/ratio = ([^,\s]+)/)?.[1]),
    selectedOrdering: rawValue(symmetric, "Selected ordering method"),
    selectedLNonzeros: numericValue(symmetric, "Selected symbolic factor L nonzeros"),
    symbolicCache: rawValue(symmetric, "Symbolic cache"),
    symOrderingAnalysisMs: numericValue(symmetric, "CHOLMOD ordering and symbolic analysis time (ms)"),
    symPermutationMs: numericValue(symmetric, "Symmetric CSC permutation time (ms)"),
    symPipelineMs: numericValue(symmetric, "Symbolic pipeline including permutation time (ms)"),
    eliminationTreeColumns: numericValue(symmetric, "Elimination tree columns"),
    basicSupernodes: numericValue(symmetric, "Basic supernodes"),
    gpuSingleColumnNodes: numericValue(symmetric, "GPU single-column nodes"),
    gpu2To64Nodes: numericValue(symmetric, "GPU 2--64 column nodes"),
    gpuOver64Nodes: numericValue(symmetric, "GPU >64 column nodes"),
    accepted1x1Pivots: numericValue(symmetric, "Accepted 1x1 pivots"),
    accepted2x2Blocks: numericValue(symmetric, "Accepted 2x2 pivot blocks"),
    delayedPivotEvents: numericValue(symmetric, "Delayed pivot events"),
    symTreeWaves: numericValue(symmetric, "Supernode tree waves"),
    inputPreprocessUploadMs: numericValue(symmetric, "Input preprocessing and upload time (ms)"),
    symFrontAssemblyMs: numericValue(symmetric, "Front assembly enqueue time (ms)"),
    contributionReleaseMs: numericValue(symmetric, "Contribution release/barrier time (ms)"),
    smallMediumFactorMs: numericValue(symmetric, "Single/small/medium factor time (ms)"),
    largePanelFactorMs: numericValue(symmetric, "Large panel factor time (ms)"),
    factorSaveMs: numericValue(symmetric, "Factor save/contribution extraction time (ms)"),
    timedStageTotalMs: numericValue(symmetric, "Timed stage total (ms)"),
    unattributedSchedulingMs: numericValue(symmetric, "Unattributed scheduling time (ms)"),
    ldltFactorMs: numericValue(symmetric, "GPU factorization time (ms)"),
    ldltSolveMs: numericValue(symmetric, "GPU solve time (ms)"),
    ldltAbsResidual: numericValue(symmetric, "Absolute residual ||A*x-b||_inf"),
    ldltRelResidual: numericValue(symmetric, "Relative residual ||A*x-b||_inf/||b||_inf"),
    ldltBackwardError: numericValue(symmetric, "Scaled backward error"),
    ldltSolutionError: numericValue(symmetric, "Relative solution error"),
    ldltTolerance: numericValue(symmetric, "FP32 verification tolerance"),
    ldltVerification: rawValue(symmetric, "Residual verification"),
    luPreprocessing: matchingEnabled
      ? "Duff-Koster matching -> AMD"
      : "AMD only (maximum matching disabled)",
    luMatchingRank: rank ? Number(rank) : null,
    luOrderingMs: matchingEnabled
      ? numericValue(lu, "Maximum matching + scaling + AMD time (ms)")
      : numericValue(lu, "AMD-only ordering time (ms)"),
    luPermutationMs: numericValue(lu, "Unsymmetric row/column permutation time (ms)"),
    luSymbolicAnalysisMs: numericValue(lu, "In-project symbolic analysis time (ms)"),
    luEnvelopeOffDiagonal: numericValue(lu, "Structural envelope off-diagonal entries"),
    luSymbolicNonzeros: numericValue(lu, "Symbolic factor nonzeros"),
    luSupernodes: numericValue(lu, "Unsymmetric supernodes"),
    luSingleColumnFronts: numericValue(lu, "GPU LU single-column fronts"),
    luBatchedFronts: numericValue(lu, "GPU LU batched small/medium fronts"),
    luPanelFronts: numericValue(lu, "GPU LU panel fronts"),
    luAcceptedPivots: numericValue(lu, "Accepted unsymmetric pivots"),
    luDelayedColumns: numericValue(lu, "Delayed unsymmetric columns"),
    luTreeWaves: numericValue(lu, "Unsymmetric tree waves"),
    luFrontAssemblyMs: numericValue(lu, "Unsymmetric front assembly time (ms)"),
    luBatchedWallMs: numericValue(lu, "Batched small/medium LU wall time (ms)"),
    luLargeWorkMs: numericValue(lu, "Large-front right-looking GPU work sum (ms)"),
    luFactorWallMs: numericValue(lu, "Unsymmetric GPU factorization wall time (ms)"),
    luSolveMs: numericValue(lu, "Unsymmetric solve time (ms)"),
    luRefinementSteps: numericValue(lu, "Unsymmetric iterative refinement steps"),
    luInitialResidual: numericValue(lu, "Initial residual before refinement"),
    luAbsResidual: numericValue(lu, "Absolute residual ||A*x-b||_inf"),
    luBackwardError: numericValue(lu, "Scaled backward error"),
    luSolutionError: numericValue(lu, "Relative solution error"),
    luVerification: rawValue(lu, "Unsymmetric residual verification"),
  };
});

records.sort((a, b) => a.matrixDimension - b.matrixDimension || (a.matchingMode === "Enabled" ? -1 : 1));

const enHeaders = columns.map(([, en]) => en);
const zhHeaders = columns.map(([, , zh]) => zh);
const enRows = records.map((record) => columns.map(([key]) => localizedValue(key, record[key], "en")));
const zhRows = records.map((record) => columns.map(([key]) => localizedValue(key, record[key], "zh")));
const enCsv = toCsv(enHeaders, enRows);
const zhCsv = toCsv(zhHeaders, zhRows);

await fs.writeFile(enPath, `\uFEFF${enCsv}`, "utf8");
await fs.writeFile(zhPath, `\uFEFF${zhCsv}`, "utf8");

async function verifyAndRender(csvText, sheetName, previewName) {
  const workbook = await Workbook.fromCSV(csvText, { sheetName });
  const sheet = workbook.worksheets.getItem(sheetName);
  const used = sheet.getUsedRange();
  used.format.columnWidth = 14;
  used.format.rowHeight = 21;
  const header = sheet.getRangeByIndexes(0, 0, 1, columns.length);
  header.format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
  };
  header.format.rowHeight = 48;
  sheet.getRangeByIndexes(0, 0, records.length + 1, 2).format.columnWidth = 18;
  sheet.getRangeByIndexes(0, 5, records.length + 1, 1).format.columnWidth = 28;
  sheet.getRangeByIndexes(0, 6, records.length + 1, 1).format.columnWidth = 20;
  sheet.getRangeByIndexes(0, 7, records.length + 1, 1).format.columnWidth = 26;
  sheet.getRangeByIndexes(0, 8, records.length + 1, 4).format.columnWidth = 19;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);
  sheet.showGridLines = false;

  const inspected = await workbook.inspect({
    kind: "table",
    range: `A1:BI${records.length + 1}`,
    include: "values,formulas",
    tableMaxRows: records.length + 1,
    tableMaxCols: columns.length,
    maxChars: 10000,
  });
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 50 },
    summary: "final formula error scan",
  });
  const preview = await workbook.render({
    sheetName,
    range: `A1:L${records.length + 1}`,
    scale: 1,
    format: "png",
  });
  await fs.writeFile(path.join(previewDir, previewName), new Uint8Array(await preview.arrayBuffer()));
  return { inspected: inspected.ndjson, errors: errors.ndjson };
}

const enVerification = await verifyAndRender(enCsv, "English Results", "results_en_preview.png");
const zhVerification = await verifyAndRender(zhCsv, "中文结果", "results_zh_preview.png");

console.log(JSON.stringify({
  sourcePath,
  outputs: [zhPath, enPath],
  rows: records.length,
  columns: columns.length,
  sizes: [...new Set(records.map((record) => record.matrixDimension))],
  verification: {
    en: enVerification,
    zh: zhVerification,
  },
}, null, 2));
