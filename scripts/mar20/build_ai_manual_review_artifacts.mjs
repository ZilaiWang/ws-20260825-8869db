import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        cell += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n") {
      row.push(cell.replace(/\r$/, ""));
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  if (cell.length || row.length) {
    row.push(cell.replace(/\r$/, ""));
    rows.push(row);
  }
  const headers = rows.shift();
  return rows.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(headers, rows) {
  return `${headers.join(",")}\n${rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")).join("\n")}\n`;
}

function must(condition, message) {
  if (!condition) throw new Error(message);
}

function styleDataSheet(sheet, headerRange, bodyRange, widths) {
  sheet.getRange(headerRange).format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
    wrapText: true,
    rowHeight: 38,
  };
  sheet.getRange(bodyRange).format = { verticalAlignment: "top", wrapText: true };
  sheet.freezePanes.freezeRows(1);
  widths.forEach(([column, width]) => {
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  });
}

const repo = process.cwd();
const root = path.join(repo, "outputs", "MAR20-GROUPING-TASK-00A-blind-review-safe");
const viewTemplatePath = path.join(root, "view-review", "manual_view_review_v2.csv");
const calibrationTemplatePath = path.join(root, "calibration-review", "manual_calibration_decisions.csv");
const viewWorklogPath = path.join(root, "ai_view_review_worklog.csv");
const calibrationWorklogPath = path.join(root, "ai_calibration_review_worklog.csv");

const [viewTemplate, calibrationTemplate, viewWorklog, calibrationWorklog] = await Promise.all(
  [viewTemplatePath, calibrationTemplatePath, viewWorklogPath, calibrationWorklogPath].map(async (file) =>
    parseCsv(await fs.readFile(file, "utf8")),
  ),
);

must(viewTemplate.length === 120, `view template rows: ${viewTemplate.length}`);
must(viewWorklog.length === 120, `view worklog rows: ${viewWorklog.length}`);
must(calibrationTemplate.length === 389, `calibration template rows: ${calibrationTemplate.length}`);
must(calibrationWorklog.length === 389, `calibration worklog rows: ${calibrationWorklog.length}`);

const viewById = new Map(viewWorklog.map((row) => [row.node_uid, row]));
const calibrationById = new Map(calibrationWorklog.map((row) => [row.card_id, row]));
must(viewById.size === 120, "view worklog has duplicate node_uid");
must(calibrationById.size === 389, "calibration worklog has duplicate card_id");

const viewHeaders = [
  "node_uid",
  "valid",
  "blur_aircraft_remnant",
  "blur_inpaint_artifact",
  "local_mean_aircraft_remnant",
  "local_mean_inpaint_artifact",
  "telea_aircraft_remnant",
  "telea_inpaint_artifact",
  "background_tile_available",
  "background_tile_aircraft",
  "notes",
];
const calibrationHeaders = ["card_id", "label", "confidence", "supporting_evidence", "counter_evidence", "notes"];

const viewFinal = viewTemplate.map((base) => {
  const review = viewById.get(base.node_uid);
  must(review, `missing view review: ${base.node_uid}`);
  return Object.fromEntries(viewHeaders.map((header) => [header, header === "background_tile_available" ? base[header] : (review[header] ?? base[header] ?? "")]));
});
const calibrationFinal = calibrationTemplate.map((base) => {
  const review = calibrationById.get(base.card_id);
  must(review, `missing calibration review: ${base.card_id}`);
  return Object.fromEntries(calibrationHeaders.map((header) => [header, review[header] ?? base[header] ?? ""]));
});

await Promise.all([
  fs.writeFile(viewTemplatePath, toCsv(viewHeaders, viewFinal), "utf8"),
  fs.writeFile(calibrationTemplatePath, toCsv(calibrationHeaders, calibrationFinal), "utf8"),
]);

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Review Summary");
summary.getRange("A1").values = [["MAR20 grouping visual-review summary"]];
summary.getRange("A1:B1").merge();
summary.getRange("A1:B1").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF", size: 16 }, rowHeight: 28 };
summary.getRange("A3:B13").values = [
  ["Item", "Value"],
  ["Reviewer type", "AI visual review (Codex), not an independent human review"],
  ["Review date", "2026-07-21"],
  ["View nodes", viewFinal.length],
  ["Calibration pairs", calibrationFinal.length],
  ["Valid view nodes", viewFinal.filter((row) => row.valid === "1").length],
  ["Blur aircraft remnant", viewFinal.filter((row) => row.blur_aircraft_remnant === "1").length],
  ["Local-mean aircraft remnant", viewFinal.filter((row) => row.local_mean_aircraft_remnant === "1").length],
  ["Telea aircraft remnant", viewFinal.filter((row) => row.telea_aircraft_remnant === "1").length],
  ["Background tile contains aircraft", viewFinal.filter((row) => row.background_tile_aircraft === "1").length],
  ["Strict positive calibration labels", calibrationFinal.filter((row) => ["same_frame", "geometric_overlap", "same_local_site"].includes(row.label)).length],
];
summary.getRange("A3:B3").format = { fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" } };
summary.getRange("A4:A13").format.font = { bold: true, color: "#17365D" };
summary.getRange("A:A").format.columnWidth = 34;
summary.getRange("B:B").format.columnWidth = 78;
summary.getRange("A15:D20").values = [
  ["Protocol note", "Meaning"],
  ["same_frame", "Same source frame / effectively identical geometry."],
  ["geometric_overlap", "Directly alignable overlapping ground layout."],
  ["same_local_site", "Clearly the same local facility without exposed overlap."],
  ["likely_same_airport", "Weak airport-level resemblance only; excluded from strict positives."],
  ["not_same_local_site", "No shared local geometry; does not prove different airports."],
];
summary.getRange("A15:D15").format = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" } };
summary.getRange("A15:B20").format.wrapText = true;
summary.freezePanes.freezeRows(3);

const viewSheet = workbook.worksheets.add("View Review");
viewSheet.getRange(`A1:K${viewFinal.length + 1}`).values = [viewHeaders, ...viewFinal.map((row) => viewHeaders.map((header) => row[header]))];
styleDataSheet(viewSheet, "A1:K1", `A2:K${viewFinal.length + 1}`, [["A", 18], ["B", 9], ["C", 16], ["D", 16], ["E", 18], ["F", 18], ["G", 16], ["H", 16], ["I", 17], ["J", 17], ["K", 55]]);

const calibrationSheet = workbook.worksheets.add("Calibration Review");
calibrationSheet.getRange(`A1:F${calibrationFinal.length + 1}`).values = [calibrationHeaders, ...calibrationFinal.map((row) => calibrationHeaders.map((header) => row[header]))];
styleDataSheet(calibrationSheet, "A1:F1", `A2:F${calibrationFinal.length + 1}`, [["A", 14], ["B", 24], ["C", 12], ["D", 58], ["E", 52], ["F", 22]]);

const protocol = workbook.worksheets.add("Protocol");
protocol.getRange("A1:B8").values = [
  ["Field", "Value"],
  ["Reviewer", "Codex AI visual review"],
  ["Human-review claim", "No. These decisions must not be represented as independent human annotations."],
  ["Blinding", "The card-to-source mapping was not opened while reviewing."],
  ["View review", "120/120 nodes inspected from 15 contact sheets."],
  ["Calibration review", "389/389 pairs inspected from 98 contact sheets."],
  ["Evidence standard", "Only alignable fixed ground geometry supports strict positive labels."],
  ["Recommended use", "Machine-assisted preliminary gate; human adjudication remains appropriate for disputed/high-impact pairs."],
];
protocol.getRange("A1:B1").format = { fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" } };
protocol.getRange("A:A").format.columnWidth = 24;
protocol.getRange("B:B").format.columnWidth = 92;
protocol.getRange("A1:B8").format.wrapText = true;

const xlsxPath = path.join(root, "MAR20_GROUPING_TASK_00A_AI_VISUAL_REVIEW.xlsx");
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(xlsxPath);

const previewDir = path.join(root, "xlsx-preview");
await fs.mkdir(previewDir, { recursive: true });
for (const [sheetName, range] of [["Review Summary", "A1:B20"], ["View Review", "A1:K26"], ["Calibration Review", "A1:F26"], ["Protocol", "A1:B8"]]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, `${sheetName.replaceAll(" ", "_")}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const labelCounts = Object.fromEntries([...new Set(calibrationFinal.map((row) => row.label))].sort().map((label) => [label, calibrationFinal.filter((row) => row.label === label).length]));
const metadata = {
  reviewer_type: "AI visual review",
  reviewer: "Codex",
  review_date: "2026-07-21",
  independent_human_review: false,
  blinded_mapping_not_opened: true,
  view_rows: viewFinal.length,
  calibration_rows: calibrationFinal.length,
  label_counts: labelCounts,
};
await fs.writeFile(path.join(root, "ai_review_metadata.json"), `${JSON.stringify(metadata, null, 2)}\n`, "utf8");
console.log(JSON.stringify({ xlsxPath, viewRows: viewFinal.length, calibrationRows: calibrationFinal.length, labelCounts }, null, 2));
