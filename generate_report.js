/**
 * generate_report.js
 * ------------------
 * Generates a professional DOCX appraisal report from:
 *   - validation_report_<doc_id>.json   (compliance check results)
 *   - engine_results_<doc_id>.json      (consistency + anomaly engine results)
 *
 * Usage:
 *   node generate_report.js <doc_id> [output_dir]
 *   node generate_report.js 7d887599
 *   node generate_report.js 7d887599 ./output
 *
 * Output:
 *   output/DPR_Appraisal_Report_<doc_id>.docx
 *
 * Report structure:
 *   Cover Page
 *   Table of Contents
 *   1. Executive Summary
 *   2. Knowledge Graph Statistics
 *   3. Compliance Check (Compliant / Non-Compliant rows, score, verdict)
 *   4. Consistency Analysis
 *   5. Anomaly Detection
 *   6. Traceability Matrix
 *   Appendix A – Full Anomaly Flag List
 */

"use strict";

const fs   = require("fs");
const path = require("path");

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, TableOfContents,
  LevelFormat, InternalHyperlink, ExternalHyperlink,
  TabStopType, TabStopPosition,
} = require("docx");

// ─── CLI args ──────────────────────────────────────────────────────────────────

const docId     = process.argv[2];
const outputDir = process.argv[3] || path.join(__dirname, "output");

if (!docId) {
  console.error("Usage: node generate_report.js <doc_id> [output_dir]");
  process.exit(1);
}

const valPath = path.join(outputDir, `validation_report_${docId}.json`);
const engPath = path.join(outputDir, `engine_results_${docId}.json`);

if (!fs.existsSync(valPath)) {
  console.error(`validation_report not found: ${valPath}`);
  process.exit(1);
}
if (!fs.existsSync(engPath)) {
  console.error(`engine_results not found: ${engPath}`);
  process.exit(1);
}

const valReport = JSON.parse(fs.readFileSync(valPath, "utf-8"));
const engReport = JSON.parse(fs.readFileSync(engPath, "utf-8"));

// ─── Colour palette ────────────────────────────────────────────────────────────

const C = {
  RITES_BLUE:     "1F4E79",  // header / cover background
  RITES_BLUE_MID: "2E75B6",  // section headers, table headers
  RITES_BLUE_LT:  "D5E8F0",  // table header fill (light)
  GREEN:          "375623",   // Compliant text
  GREEN_BG:       "E2EFDA",   // Compliant row fill
  RED:            "9C0006",   // Non-Compliant text
  RED_BG:         "FFCCCC",   // Non-Compliant row fill
  ORANGE:         "974706",   // HIGH severity text
  ORANGE_BG:      "FCE4D6",   // HIGH severity row fill
  YELLOW_BG:      "FFEB9C",   // MEDIUM severity row fill
  CRITICAL_BG:    "FFC7CE",   // CRITICAL row fill
  GREY_LT:        "F2F2F2",   // alternating row fill
  GREY_BDR:       "CCCCCC",   // table border colour
  WHITE:          "FFFFFF",
  BLACK:          "000000",
  DARK_GREY:      "404040",
  MED_GREY:       "767676",
};

// ─── Page geometry ─────────────────────────────────────────────────────────────
// A4 portrait with 1" margins → content width = 11906 − 2880 = 9026 DXA

const PAGE_W    = 11906;
const PAGE_H    = 16838;
const MARGIN    = 1440;        // 1 inch
const CONTENT_W = PAGE_W - MARGIN * 2;  // 9026 DXA

// ─── Style helpers ─────────────────────────────────────────────────────────────

function cell(text, opts = {}) {
  const {
    bold = false, color = C.BLACK, fill = C.WHITE, width = null,
    align = AlignmentType.LEFT, fontSize = 18, italic = false,
    colSpan = 1, vAlign = VerticalAlign.CENTER,
  } = opts;

  const border = { style: BorderStyle.SINGLE, size: 1, color: C.GREY_BDR };
  const borders = { top: border, bottom: border, left: border, right: border };

  return new TableCell({
    columnSpan: colSpan,
    verticalAlign: vAlign,
    borders,
    width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: { fill, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text: String(text ?? "—"), bold, color, size: fontSize, italics: italic })],
    })],
  });
}

function hdrRow(cells_def) {
  return new TableRow({
    tableHeader: true,
    children: cells_def.map(([text, width, align]) =>
      cell(text, { bold: true, fill: C.RITES_BLUE_LT, color: C.RITES_BLUE, width, fontSize: 18, align: align || AlignmentType.LEFT })
    ),
  });
}

function para(text, opts = {}) {
  const {
    bold = false, size = 20, color = C.DARK_GREY, italic = false,
    spacing = { after: 120 }, align = AlignmentType.LEFT,
    heading = null, indent = null,
  } = opts;
  const p = {
    alignment: align,
    spacing,
    children: [new TextRun({ text: String(text), bold, size, color, italics: italic })],
  };
  if (heading) p.heading = heading;
  if (indent)  p.indent  = indent;
  return new Paragraph(p);
}

function sectionDivider(label) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 180 },
    children: [new TextRun({ text: label, bold: true, size: 28, color: C.RITES_BLUE })],
  });
}

function subHeading(label) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text: label, bold: true, size: 24, color: C.RITES_BLUE_MID })],
  });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function dividerLine() {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.RITES_BLUE_MID, space: 1 } },
    children: [new TextRun("")],
  });
}

function bulletPara(text, numbering) {
  return new Paragraph({
    numbering: { reference: numbering, level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text: String(text), size: 18, color: C.DARK_GREY })],
  });
}

// ─── Severity helpers ──────────────────────────────────────────────────────────

function severityColor(sev) {
  return { CRITICAL: C.RED, HIGH: C.ORANGE, MEDIUM: "7D6608", LOW: "375623" }[sev] || C.DARK_GREY;
}
function severityBg(sev) {
  return { CRITICAL: C.CRITICAL_BG, HIGH: C.ORANGE_BG, MEDIUM: C.YELLOW_BG, LOW: C.GREY_LT }[sev] || C.WHITE;
}
function verdictColor(v) {
  return { GOOD: C.GREEN, SATISFACTORY: "375623", "NEEDS IMPROVEMENT": C.ORANGE, POOR: C.RED }[v] || C.DARK_GREY;
}
function verdictBg(v) {
  return { GOOD: C.GREEN_BG, SATISFACTORY: C.GREEN_BG, "NEEDS IMPROVEMENT": C.YELLOW_BG, POOR: C.RED_BG }[v] || C.GREY_LT;
}

// ─── Data helpers ──────────────────────────────────────────────────────────────

// Support BOTH old format (failures/missing_parameters) and new format (results + score)
function getValidationRows(vr) {
  // New format: flat results list with classification field
  if (vr.results && Array.isArray(vr.results)) {
    return vr.results;
  }
  // Old format: convert failures + missing_parameters into row objects
  const rows = [];
  for (const f of (vr.findings?.failures || [])) {
    rows.push({
      classification: "Non-Compliant",
      check_area:     f.attribute,
      category:       f.standard,
      dpr_value:      f.dpr_value,
      rule_expected:  f.rule,
      standard:       f.standard,
      severity:       f.severity,
      weight:         { CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1 }[f.severity] || 2,
      reason:         f.explanation || "Does not meet rule requirement.",
      source_page:    f.source_page,
    });
  }
  for (const m of (vr.findings?.missing_parameters || [])) {
    rows.push({
      classification: "Non-Compliant",
      check_area:     m.parameter,
      category:       "DPR Completeness",
      dpr_value:      "Not found",
      rule_expected:  "Must be present",
      standard:       "DPR Completeness",
      severity:       "HIGH",
      weight:         3,
      reason:         m.rule_text || "Mandatory parameter missing from DPR.",
      source_page:    0,
    });
  }
  return rows;
}

function getScore(vr) {
  if (vr.score) return vr.score;
  // Derive score from old format
  const rows   = getValidationRows(vr);
  const total  = rows.length;
  const nc     = rows.filter(r => r.classification === "Non-Compliant").length;
  const c      = total - nc;
  const ws     = total ? Math.round(c / total * 100) : 0;
  let verdict  = "POOR";
  if (ws >= 90) verdict = "GOOD";
  else if (ws >= 75) verdict = "SATISFACTORY";
  else if (ws >= 50) verdict = "NEEDS IMPROVEMENT";
  return {
    weighted_score: ws, verdict,
    total_checks: total, compliant_count: c, non_compliant_count: nc,
    by_severity: {}, by_category: {},
  };
}

// ─── Section builders ──────────────────────────────────────────────────────────

function buildCoverPage(vr, score) {
  const genDate = new Date(vr.generated_at || new Date()).toLocaleDateString("en-IN", {
    day: "2-digit", month: "long", year: "numeric",
  });

  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 1440, after: 0 },
      shading: { fill: C.RITES_BLUE, type: ShadingType.CLEAR },
      children: [new TextRun({ text: "RITES LIMITED", bold: true, size: 48, color: C.WHITE, font: "Arial" })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 0 },
      shading: { fill: C.RITES_BLUE, type: ShadingType.CLEAR },
      children: [new TextRun({ text: "DPR Appraisal — Automated Validation Report", size: 28, color: C.RITES_BLUE_LT, font: "Arial" })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 720 },
      shading: { fill: C.RITES_BLUE, type: ShadingType.CLEAR },
      children: [new TextRun({ text: " ", size: 24, color: C.WHITE })],
    }),
    dividerLine(),
    new Paragraph({ spacing: { after: 240 }, children: [new TextRun("")] }),
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA },
      columnWidths: [3000, 6026],
      rows: [
        new TableRow({ children: [
          cell("Document ID",  { bold: true, fill: C.RITES_BLUE_LT, color: C.RITES_BLUE, width: 3000, fontSize: 20 }),
          cell(vr.doc_id,      { width: 6026, fontSize: 20 }),
        ]}),
        new TableRow({ children: [
          cell("Sector",       { bold: true, fill: C.RITES_BLUE_LT, color: C.RITES_BLUE, width: 3000, fontSize: 20 }),
          cell(vr.sector,      { width: 6026, fontSize: 20 }),
        ]}),
        new TableRow({ children: [
          cell("Report Date",  { bold: true, fill: C.RITES_BLUE_LT, color: C.RITES_BLUE, width: 3000, fontSize: 20 }),
          cell(genDate,        { width: 6026, fontSize: 20 }),
        ]}),
        new TableRow({ children: [
          cell("Verdict",      { bold: true, fill: C.RITES_BLUE_LT, color: C.RITES_BLUE, width: 3000, fontSize: 20 }),
          cell(score.verdict,  { bold: true, fill: verdictBg(score.verdict), color: verdictColor(score.verdict), width: 6026, fontSize: 20 }),
        ]}),
        new TableRow({ children: [
          cell("Weighted Compliance Score", { bold: true, fill: C.RITES_BLUE_LT, color: C.RITES_BLUE, width: 3000, fontSize: 20 }),
          cell(`${score.weighted_score}%`,  { bold: true, fill: verdictBg(score.verdict), color: verdictColor(score.verdict), width: 6026, fontSize: 20 }),
        ]}),
      ],
    }),
    new Paragraph({ spacing: { before: 480, after: 120 }, children: [
      new TextRun({ text: "CONFIDENTIAL — For Internal Appraisal Use Only", size: 16, color: C.MED_GREY, italics: true }),
    ]}),
    pageBreak(),
  ];
}

function buildTOC() {
  return [
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      spacing: { before: 0, after: 240 },
      children: [new TextRun({ text: "Table of Contents", bold: true, size: 28, color: C.RITES_BLUE })],
    }),
    new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
    pageBreak(),
  ];
}

function buildExecutiveSummary(vr, engReport, score, valRows) {
  const consistency = engReport.consistency || {};
  const anomaly     = engReport.anomaly     || {};

  const consTotal  = consistency.total_issues || 0;
  const consCrit   = (consistency.by_severity || {}).CRITICAL || 0;
  const consHigh   = (consistency.by_severity || {}).HIGH     || 0;
  const anomTotal  = anomaly.total_flags || 0;
  const anomHigh   = (anomaly.by_severity || {}).HIGH    || 0;
  const anomCrit   = (anomaly.by_severity || {}).CRITICAL || 0;
  const ncCount    = score.non_compliant_count || 0;

  const critCompliance = valRows.filter(r => r.classification === "Non-Compliant" && r.severity === "CRITICAL").length;

  // Determine overall health text
  let healthText = "";
  if      (score.verdict === "GOOD")             healthText = "The DPR is in good shape overall and meets most applicable standards.";
  else if (score.verdict === "SATISFACTORY")      healthText = "The DPR is broadly satisfactory with a small number of issues that should be addressed.";
  else if (score.verdict === "NEEDS IMPROVEMENT") healthText = "The DPR has several gaps against applicable standards and requires targeted revisions before appraisal sign-off.";
  else                                            healthText = "The DPR has significant compliance deficiencies. Substantial revision is required before it can be appraised favourably.";

  const bullets = [];
  if (critCompliance > 0)
    bullets.push(`${critCompliance} CRITICAL non-compliances identified — these are safety or regulatory showstoppers requiring mandatory correction.`);
  if (ncCount > 0)
    bullets.push(`${ncCount} total Non-Compliant check(s) found across the rulebook comparison.`);
  if (consCrit + consHigh > 0)
    bullets.push(`${consCrit + consHigh} HIGH/CRITICAL internal consistency issues detected (e.g., conflicting values across DPR sections).`);
  if (anomHigh + anomCrit > 0)
    bullets.push(`${anomHigh + anomCrit} HIGH/CRITICAL anomalies flagged in extracted data (potential data entry or decimal errors).`);

  return [
    sectionDivider("1. Executive Summary"),
    para(healthText, { size: 20, spacing: { after: 180 } }),
    para("Key Findings:", { bold: true, size: 20, spacing: { after: 80 } }),
    ...bullets.map(b => bulletPara(b, "bullets")),
    new Paragraph({ spacing: { after: 120 }, children: [new TextRun("")] }),

    // Summary scorecard table
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA },
      columnWidths: [4513, 4513],
      rows: [
        hdrRow([["Engine / Check", 4513], ["Result", 4513]]),
        new TableRow({ children: [
          cell("Weighted Compliance Score", { bold: true, width: 4513, fill: C.GREY_LT }),
          cell(`${score.weighted_score}% — ${score.verdict}`, { bold: true, width: 4513, fill: verdictBg(score.verdict), color: verdictColor(score.verdict) }),
        ]}),
        new TableRow({ children: [
          cell("Total Rulebook Checks",  { width: 4513 }),
          cell(`${score.total_checks} checks (${score.compliant_count} Compliant, ${ncCount} Non-Compliant)`, { width: 4513 }),
        ]}),
        new TableRow({ children: [
          cell("Consistency Engine",  { width: 4513 }),
          cell(`${consTotal} issues (CRITICAL: ${consCrit}, HIGH: ${consHigh})`, {
            width: 4513, fill: consCrit + consHigh > 0 ? C.ORANGE_BG : C.GREEN_BG,
            color: consCrit + consHigh > 0 ? C.ORANGE : C.GREEN,
          }),
        ]}),
        new TableRow({ children: [
          cell("Anomaly Detection Engine", { width: 4513 }),
          cell(`${anomTotal} flags (CRITICAL: ${anomCrit}, HIGH: ${anomHigh})`, {
            width: 4513, fill: anomCrit + anomHigh > 0 ? C.ORANGE_BG : C.GREEN_BG,
            color: anomCrit + anomHigh > 0 ? C.ORANGE : C.GREEN,
          }),
        ]}),
      ],
    }),
    pageBreak(),
  ];
}

function buildKGStats(vr) {
  const kg = vr.kg_stats || {};
  return [
    sectionDivider("2. Knowledge Graph Statistics"),
    para("The DPR was processed through the PARAKH extraction pipeline. The following knowledge graph (KG) statistics reflect the structured information extracted from the document and stored in Neo4j.", { spacing: { after: 180 } }),
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA },
      columnWidths: [4513, 4513],
      rows: [
        hdrRow([["KG Metric", 4513], ["Value", 4513]]),
        new TableRow({ children: [cell("Entity Nodes",         { width: 4513 }), cell(String(kg.entities   || "—"), { width: 4513, bold: true })] }),
        new TableRow({ children: [cell("Relationship Triples", { width: 4513 }), cell(String(kg.triples    || "—"), { width: 4513, bold: true })] }),
        new TableRow({ children: [cell("Ontology Concepts",    { width: 4513 }), cell(String(kg.concepts   || "—"), { width: 4513, bold: true })] }),
      ],
    }),
    pageBreak(),
  ];
}

function buildComplianceSection(valRows, score) {
  const bySev = score.by_severity || {};
  const byCat = score.by_category || {};

  // Severity breakdown table
  const sevRows = ["CRITICAL", "HIGH", "MEDIUM", "LOW"].filter(s => bySev[s]).map(sev => {
    const d = bySev[sev];
    return new TableRow({ children: [
      cell(sev, { bold: true, color: severityColor(sev), fill: severityBg(sev), width: 2000 }),
      cell(String(d.total   || 0), { width: 1800, align: AlignmentType.CENTER }),
      cell(String(d.compliant || 0), { width: 1800, color: C.GREEN, bold: true, align: AlignmentType.CENTER }),
      cell(String(d.non_compliant || 0), { width: 1800, color: d.non_compliant > 0 ? C.RED : C.GREEN, bold: d.non_compliant > 0, align: AlignmentType.CENTER }),
      cell(`${Math.round((d.compliant || 0) / Math.max(d.total || 1, 1) * 100)}%`, { width: 1626, align: AlignmentType.CENTER }),
    ]});
  });

  // Category score table
  const catRows = Object.entries(byCat)
    .sort((a, b) => a[1].category_score - b[1].category_score)
    .map(([cat, d]) => {
      const cs  = d.category_score || 0;
      const col = cs >= 75 ? C.GREEN : cs >= 50 ? C.ORANGE : C.RED;
      const bg  = cs >= 75 ? C.GREEN_BG : cs >= 50 ? C.ORANGE_BG : C.RED_BG;
      return new TableRow({ children: [
        cell(cat,              { width: 3800 }),
        cell(String(d.total),  { width: 1406, align: AlignmentType.CENTER }),
        cell(String(d.compliant),     { width: 1406, color: C.GREEN, bold: true, align: AlignmentType.CENTER }),
        cell(String(d.non_compliant), { width: 1406, color: d.non_compliant > 0 ? C.RED : C.GREEN, bold: d.non_compliant > 0, align: AlignmentType.CENTER }),
        cell(`${cs}%`, { width: 1008, bold: true, color: col, fill: bg, align: AlignmentType.CENTER }),
      ]});
    });

  // Full results table — Non-Compliant first (sorted CRITICAL→LOW), then Compliant
  const sortedRows = [
    ...valRows.filter(r => r.classification === "Non-Compliant")
              .sort((a, b) => ({ CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 }[a.severity] - ({ CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 }[b.severity]))),
    ...valRows.filter(r => r.classification === "Compliant"),
  ];

  const resultRows = sortedRows.map((r, i) => {
    const isNC  = r.classification === "Non-Compliant";
    const rowBg = isNC ? (i % 2 === 0 ? C.RED_BG : "FFD9D9") : (i % 2 === 0 ? C.WHITE : C.GREY_LT);
    const pg    = r.source_page && r.source_page > 0 ? `p.${r.source_page}` : "—";
    return new TableRow({ children: [
      cell(r.classification, {
        bold: true, width: 1400,
        color: isNC ? C.RED : C.GREEN,
        fill:  isNC ? C.RED_BG : C.GREEN_BG,
        align: AlignmentType.CENTER,
      }),
      cell(r.check_area     || "—", { width: 1600, fill: rowBg }),
      cell(r.category       || "—", { width: 1300, fill: rowBg }),
      cell(r.dpr_value      || "—", { width: 1100, fill: rowBg }),
      cell(r.rule_expected  || "—", { width: 1100, fill: rowBg }),
      cell(r.severity       || "—", { width: 700,  bold: true, color: severityColor(r.severity), fill: severityBg(r.severity), align: AlignmentType.CENTER }),
      cell(String(r.weight  || ""), { width: 500,  align: AlignmentType.CENTER, fill: rowBg }),
      cell(pg,                       { width: 500,  align: AlignmentType.CENTER, fill: rowBg }),
      cell(r.reason         || "—", { width: 1326, fill: rowBg, fontSize: 16 }),
    ]});
  });

  return [
    sectionDivider("3. Compliance Check — Rulebook vs DPR"),
    para(
      `The following table records the comparison of DPR-extracted facts against rules loaded from applicable standards. ` +
      `Each check is classified as Compliant or Non-Compliant, with a plain-English reason. ` +
      `The weighted score accounts for the severity of each rule — CRITICAL rules carry 4× the weight of LOW rules.`,
      { spacing: { after: 180 } }
    ),

    // Score banner
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA },
      columnWidths: [4513, 4513],
      rows: [
        new TableRow({ children: [
          cell("Weighted Compliance Score", { bold: true, fill: C.RITES_BLUE_LT, color: C.RITES_BLUE, width: 4513, fontSize: 22 }),
          cell(`${score.weighted_score}% — ${score.verdict}`, { bold: true, fill: verdictBg(score.verdict), color: verdictColor(score.verdict), width: 4513, fontSize: 22 }),
        ]}),
        new TableRow({ children: [
          cell("Total Checks",    { width: 4513, fill: C.GREY_LT }),
          cell(String(score.total_checks || 0), { width: 4513 }),
        ]}),
        new TableRow({ children: [
          cell("Compliant",      { width: 4513, fill: C.GREEN_BG, color: C.GREEN, bold: true }),
          cell(String(score.compliant_count || 0), { width: 4513, color: C.GREEN, bold: true }),
        ]}),
        new TableRow({ children: [
          cell("Non-Compliant",  { width: 4513, fill: C.RED_BG, color: C.RED, bold: true }),
          cell(String(score.non_compliant_count || 0), { width: 4513, color: C.RED, bold: true }),
        ]}),
      ],
    }),

    new Paragraph({ spacing: { after: 240 }, children: [new TextRun("")] }),

    // Severity breakdown
    ...(sevRows.length > 0 ? [
      subHeading("3.1 Score Breakdown by Severity"),
      new Table({
        width: { size: CONTENT_W, type: WidthType.DXA },
        columnWidths: [2000, 1800, 1800, 1800, 1626],
        rows: [
          hdrRow([["Severity", 2000], ["Total", 1800], ["Compliant", 1800], ["Non-Compliant", 1800], ["Pass Rate", 1626]]),
          ...sevRows,
        ],
      }),
      new Paragraph({ spacing: { after: 240 }, children: [new TextRun("")] }),
    ] : []),

    // Category breakdown
    ...(catRows.length > 0 ? [
      subHeading("3.2 Score by Rulebook Category"),
      new Table({
        width: { size: CONTENT_W, type: WidthType.DXA },
        columnWidths: [3800, 1406, 1406, 1406, 1008],
        rows: [
          hdrRow([["Category", 3800], ["Total", 1406], ["Compliant", 1406], ["Non-Compliant", 1406], ["Score", 1008]]),
          ...catRows,
        ],
      }),
      new Paragraph({ spacing: { after: 240 }, children: [new TextRun("")] }),
    ] : []),

    // Full results table
    subHeading("3.3 Detailed Check Results"),
    para("Non-Compliant checks are listed first (sorted by severity). Each row includes the DPR value, the rule requirement, the score weight, and a plain-English reason.", { spacing: { after: 120 } }),
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA },
      columnWidths: [1400, 1600, 1300, 1100, 1100, 700, 500, 500, 1326],
      rows: [
        hdrRow([
          ["Classification", 1400],
          ["Check Area",     1600],
          ["Category",       1300],
          ["DPR Value",      1100],
          ["Rule Requires",  1100],
          ["Severity",        700],
          ["Weight",          500],
          ["Page",            500],
          ["Reason",         1326],
        ]),
        ...resultRows,
      ],
    }),
    pageBreak(),
  ];
}

function buildConsistencySection(consistency) {
  const issues   = consistency.issues || [];
  const byType   = consistency.by_type || {};
  const bySev    = consistency.by_severity || {};

  const typeLabels = {
    numeric_mismatch:   "Numeric Cross-Section Mismatch",
    missing_dependency: "Missing Ontology Dependency",
    llm_flagged:        "LLM-Flagged Inconsistency",
  };

  // Summary table
  const typeRows = Object.entries(byType).map(([t, cnt]) =>
    new TableRow({ children: [
      cell(typeLabels[t] || t, { width: 5000 }),
      cell(String(cnt),        { width: 4026, bold: true, align: AlignmentType.CENTER }),
    ]})
  );
  const sevRows = Object.entries(bySev)
    .filter(([, v]) => v > 0)
    .map(([sev, cnt]) =>
      new TableRow({ children: [
        cell(sev, { width: 5000, bold: true, color: severityColor(sev), fill: severityBg(sev) }),
        cell(String(cnt), { width: 4026, bold: true, align: AlignmentType.CENTER }),
      ]})
    );

  // Issues table — sorted CRITICAL→LOW
  const sortOrder = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
  const sorted = [...issues].sort((a, b) => (sortOrder[a.severity] || 4) - (sortOrder[b.severity] || 4));

  const issueRows = sorted.map((iss, i) => {
    const sev  = iss.severity || "MEDIUM";
    const bg   = i % 2 === 0 ? C.WHITE : C.GREY_LT;
    const pages = (iss.pages || []).length > 0 ? iss.pages.map(p => `p.${p}`).join(", ") : "—";
    const itype = typeLabels[iss.type] || iss.type || "—";
    return new TableRow({ children: [
      cell(sev,              { width: 800, bold: true, color: severityColor(sev), fill: severityBg(sev), align: AlignmentType.CENTER }),
      cell(itype,            { width: 1600, fill: bg }),
      cell(iss.description,  { width: 3200, fill: bg, fontSize: 16 }),
      cell(iss.evidence || "—", { width: 2126, fill: bg, italic: true, fontSize: 16 }),
      cell(pages,            { width: 1300, fill: bg, align: AlignmentType.CENTER }),
    ]});
  });

  return [
    sectionDivider("4. Consistency Analysis"),
    para(
      "The Consistency Engine checks for internal contradictions within the DPR — values that conflict with each other across different sections of the same document. " +
      "These are distinct from rulebook non-compliances; a DPR can be internally inconsistent while still meeting external standards, or vice versa.",
      { spacing: { after: 180 } }
    ),

    // Summary
    subHeading("4.1 Summary"),
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA },
      columnWidths: [5000, 4026],
      rows: [
        hdrRow([["Category", 5000], ["Count", 4026]]),
        new TableRow({ children: [
          cell("Total Consistency Issues", { bold: true, width: 5000, fill: C.GREY_LT }),
          cell(String(consistency.total_issues || 0), { width: 4026, bold: true, align: AlignmentType.CENTER }),
        ]}),
        ...typeRows,
        ...sevRows,
      ],
    }),

    new Paragraph({ spacing: { after: 240 }, children: [new TextRun("")] }),

    // Full issues table
    subHeading("4.2 Consistency Issues — Full List"),
    ...(issueRows.length === 0
      ? [para("No consistency issues detected.", { italic: true })]
      : [
          new Table({
            width: { size: CONTENT_W, type: WidthType.DXA },
            columnWidths: [800, 1600, 3200, 2126, 1300],
            rows: [
              hdrRow([
                ["Severity", 800],
                ["Issue Type", 1600],
                ["Description", 3200],
                ["Evidence", 2126],
                ["Pages", 1300],
              ]),
              ...issueRows,
            ],
          }),
        ]
    ),
    pageBreak(),
  ];
}

function buildAnomalySection(anomaly) {
  const flags  = anomaly.flags || [];
  const byType = anomaly.by_type || {};
  const bySev  = anomaly.by_severity || {};

  const typeLabels = {
    statistical_outlier: "Statistical Outlier",
    oom_error:           "Order-of-Magnitude Error",
    duplicate_values:    "Duplicate Values",
    unit_mismatch:       "Unit Mismatch",
    llm_flagged:         "LLM-Flagged Anomaly",
  };

  const typeRows = Object.entries(byType).map(([t, cnt]) =>
    new TableRow({ children: [
      cell(typeLabels[t] || t, { width: 5000 }),
      cell(String(cnt),        { width: 4026, bold: true, align: AlignmentType.CENTER }),
    ]})
  );
  const sevRows = Object.entries(bySev)
    .filter(([, v]) => v > 0)
    .map(([sev, cnt]) =>
      new TableRow({ children: [
        cell(sev, { width: 5000, bold: true, color: severityColor(sev), fill: severityBg(sev) }),
        cell(String(cnt), { width: 4026, bold: true, align: AlignmentType.CENTER }),
      ]})
    );

  // Show HIGH+ flags inline; rest in appendix
  const highFlags = flags
    .filter(f => ["CRITICAL", "HIGH"].includes(f.severity))
    .sort((a, b) => (({ CRITICAL: 0, HIGH: 1 }[a.severity] || 2) - ({ CRITICAL: 0, HIGH: 1 }[b.severity] || 2)));

  const flagRows = highFlags.map((f, i) => {
    const sev = f.severity || "MEDIUM";
    const bg  = i % 2 === 0 ? C.WHITE : C.GREY_LT;
    return new TableRow({ children: [
      cell(sev,                    { width: 800, bold: true, color: severityColor(sev), fill: severityBg(sev), align: AlignmentType.CENTER }),
      cell(typeLabels[f.type] || f.type || "—", { width: 1600, fill: bg }),
      cell(f.attribute || "—",     { width: 1400, fill: bg }),
      cell(String(f.flagged_value || "—"), { width: 1200, fill: bg, bold: true }),
      cell(f.expected_range || "—",{ width: 1500, fill: bg, italic: true, fontSize: 16 }),
      cell(f.description || "—",   { width: 1526, fill: bg, fontSize: 16 }),
    ]});
  });

  return [
    sectionDivider("5. Anomaly Detection"),
    para(
      "The Anomaly Detection Engine scans the extracted DPR data for statistically improbable values, decimal placement errors, repeated copy-paste entries, and domain-aware anomalies identified by the LLM. " +
      "These are not necessarily rulebook violations — they are data quality flags that warrant human review.",
      { spacing: { after: 180 } }
    ),

    subHeading("5.1 Summary"),
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA },
      columnWidths: [5000, 4026],
      rows: [
        hdrRow([["Category", 5000], ["Count", 4026]]),
        new TableRow({ children: [
          cell("Total Anomaly Flags", { bold: true, width: 5000, fill: C.GREY_LT }),
          cell(String(anomaly.total_flags || 0), { width: 4026, bold: true, align: AlignmentType.CENTER }),
        ]}),
        ...typeRows,
        ...sevRows,
      ],
    }),

    new Paragraph({ spacing: { after: 240 }, children: [new TextRun("")] }),

    subHeading("5.2 HIGH / CRITICAL Anomalies — Requiring Review"),
    ...(flagRows.length === 0
      ? [para("No HIGH or CRITICAL anomaly flags detected.", { italic: true })]
      : [
          para("The following anomalies are HIGH or CRITICAL severity. Full list of all flags is in Appendix A.", { spacing: { after: 120 } }),
          new Table({
            width: { size: CONTENT_W, type: WidthType.DXA },
            columnWidths: [800, 1600, 1400, 1200, 1500, 1526],
            rows: [
              hdrRow([
                ["Severity",       800],
                ["Anomaly Type",  1600],
                ["Attribute",     1400],
                ["Flagged Value", 1200],
                ["Expected Range",1500],
                ["Description",   1526],
              ]),
              ...flagRows,
            ],
          }),
        ]
    ),
    pageBreak(),
  ];
}

function buildTraceabilityMatrix(valRows, consistency, anomaly) {
  // Group compliance rows by category
  const byCategory = {};
  for (const r of valRows) {
    const cat = r.category || "Uncategorised";
    if (!byCategory[cat]) byCategory[cat] = { compliant: [], non_compliant: [] };
    if (r.classification === "Compliant") byCategory[cat].compliant.push(r);
    else byCategory[cat].non_compliant.push(r);
  }

  // Build a flat traceability table: one row per check, anchored to source
  const sortOrder = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
  const traceRows = [
    ...valRows.filter(r => r.classification === "Non-Compliant")
              .sort((a, b) => (sortOrder[a.severity] || 4) - (sortOrder[b.severity] || 4)),
    ...valRows.filter(r => r.classification === "Compliant"),
  ].map((r, i) => {
    const isNC = r.classification === "Non-Compliant";
    const bg   = isNC ? (i % 2 === 0 ? C.RED_BG : "FFD9D9") : (i % 2 === 0 ? C.WHITE : C.GREY_LT);
    const pg   = r.source_page && r.source_page > 0 ? `p.${r.source_page}` : "—";
    return new TableRow({ children: [
      cell(String(i + 1), { width: 400, align: AlignmentType.CENTER, fill: bg }),
      cell(r.standard || "—",        { width: 1600, fill: bg, fontSize: 16 }),
      cell(r.check_area || "—",      { width: 1600, fill: bg, fontSize: 16 }),
      cell(r.dpr_value  || "—",      { width: 1000, fill: bg, fontSize: 16 }),
      cell(r.rule_expected || "—",   { width: 1000, fill: bg, fontSize: 16 }),
      cell(pg,                        { width: 500,  fill: bg, align: AlignmentType.CENTER }),
      cell(r.classification, {
        width: 1200, bold: true,
        color: isNC ? C.RED : C.GREEN,
        fill:  isNC ? C.RED_BG : C.GREEN_BG,
        align: AlignmentType.CENTER,
      }),
      cell(r.severity || "—", {
        width: 700, bold: true,
        color: severityColor(r.severity || ""),
        fill:  severityBg(r.severity  || ""),
        align: AlignmentType.CENTER,
      }),
      cell(r.reason || "—", { width: 1026, fill: bg, fontSize: 16 }),
    ]});
  });

  // Consistency traceability
  const consRows = (consistency.issues || [])
    .sort((a, b) => (sortOrder[a.severity] || 4) - (sortOrder[b.severity] || 4))
    .map((iss, i) => {
      const sev  = iss.severity || "MEDIUM";
      const bg   = i % 2 === 0 ? C.WHITE : C.GREY_LT;
      const pages = (iss.pages || []).map(p => `p.${p}`).join(", ") || "—";
      return new TableRow({ children: [
        cell(String(i + 1),      { width: 400,  align: AlignmentType.CENTER, fill: bg }),
        cell(iss.type || "—",    { width: 1800, fill: bg, fontSize: 16 }),
        cell(iss.description || "—", { width: 3400, fill: bg, fontSize: 16 }),
        cell(iss.evidence || "—",    { width: 2000, fill: bg, italic: true, fontSize: 16 }),
        cell(pages,               { width: 500,  align: AlignmentType.CENTER, fill: bg }),
        cell(sev,                 { width: 926,  bold: true, color: severityColor(sev), fill: severityBg(sev), align: AlignmentType.CENTER }),
      ]});
    });

  // Anomaly traceability (HIGH+ only)
  const anomRows = (anomaly.flags || [])
    .filter(f => ["CRITICAL", "HIGH"].includes(f.severity))
    .sort((a, b) => (sortOrder[a.severity] || 4) - (sortOrder[b.severity] || 4))
    .map((f, i) => {
      const sev = f.severity || "HIGH";
      const bg  = i % 2 === 0 ? C.WHITE : C.GREY_LT;
      return new TableRow({ children: [
        cell(String(i + 1),          { width: 400,  align: AlignmentType.CENTER, fill: bg }),
        cell(f.type  || "—",         { width: 1600, fill: bg, fontSize: 16 }),
        cell(f.attribute || "—",     { width: 1400, fill: bg, fontSize: 16 }),
        cell(String(f.flagged_value || "—"), { width: 1000, fill: bg, bold: true }),
        cell(f.expected_range || "—",{ width: 1500, fill: bg, italic: true, fontSize: 16 }),
        cell(f.description || "—",   { width: 2000, fill: bg, fontSize: 16 }),
        cell(sev, { width: 626, bold: true, color: severityColor(sev), fill: severityBg(sev), align: AlignmentType.CENTER }),
      ]});
    });

  return [
    sectionDivider("6. Traceability Matrix"),
    para(
      "This section provides end-to-end traceability from each engineering rule (standard + clause) to the DPR source page and the final classification. " +
      "This matrix is the primary audit trail for the appraisal team and enables a reviewer to locate the exact DPR page that was the basis for each finding.",
      { spacing: { after: 180 } }
    ),

    subHeading("6.1 Rulebook Compliance Traceability"),
    para("Each row traces: Standard → Check → DPR-extracted value → DPR source page → Classification + Reason.", { spacing: { after: 120 } }),
    ...(traceRows.length === 0
      ? [para("No compliance checks were run (no rules loaded for this sector).", { italic: true })]
      : [
          new Table({
            width: { size: CONTENT_W, type: WidthType.DXA },
            columnWidths: [400, 1600, 1600, 1000, 1000, 500, 1200, 700, 1026],
            rows: [
              hdrRow([
                ["#",             400],
                ["Standard",     1600],
                ["Check Area",   1600],
                ["DPR Value",    1000],
                ["Rule Requires",1000],
                ["Page",          500],
                ["Classification",1200],
                ["Severity",      700],
                ["Reason",       1026],
              ]),
              ...traceRows,
            ],
          }),
        ]
    ),
    new Paragraph({ spacing: { after: 240 }, children: [new TextRun("")] }),

    subHeading("6.2 Consistency Issue Traceability"),
    ...(consRows.length === 0
      ? [para("No consistency issues detected.", { italic: true })]
      : [
          new Table({
            width: { size: CONTENT_W, type: WidthType.DXA },
            columnWidths: [400, 1800, 3400, 2000, 500, 926],
            rows: [
              hdrRow([["#", 400], ["Issue Type", 1800], ["Description", 3400], ["Evidence", 2000], ["Pages", 500], ["Severity", 926]]),
              ...consRows,
            ],
          }),
        ]
    ),
    new Paragraph({ spacing: { after: 240 }, children: [new TextRun("")] }),

    subHeading("6.3 Anomaly Traceability (HIGH / CRITICAL)"),
    ...(anomRows.length === 0
      ? [para("No HIGH or CRITICAL anomaly flags detected.", { italic: true })]
      : [
          new Table({
            width: { size: CONTENT_W, type: WidthType.DXA },
            columnWidths: [400, 1600, 1400, 1000, 1500, 2000, 626],
            rows: [
              hdrRow([["#", 400], ["Anomaly Type", 1600], ["Attribute", 1400], ["Flagged Value", 1000], ["Expected Range", 1500], ["Description", 2000], ["Severity", 626]]),
              ...anomRows,
            ],
          }),
        ]
    ),
    pageBreak(),
  ];
}

function buildAppendixA(anomaly) {
  const flags    = anomaly.flags || [];
  const typeLabels = {
    statistical_outlier: "Statistical Outlier",
    oom_error:           "Order-of-Magnitude Error",
    duplicate_values:    "Duplicate Values",
    unit_mismatch:       "Unit Mismatch",
    llm_flagged:         "LLM-Flagged Anomaly",
  };
  const sortOrder = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
  const sorted = [...flags].sort((a, b) => (sortOrder[a.severity] || 4) - (sortOrder[b.severity] || 4));

  const allRows = sorted.map((f, i) => {
    const sev = f.severity || "MEDIUM";
    const bg  = i % 2 === 0 ? C.WHITE : C.GREY_LT;
    return new TableRow({ children: [
      cell(String(i + 1),             { width: 400,  align: AlignmentType.CENTER, fill: bg }),
      cell(typeLabels[f.type] || f.type || "—", { width: 1600, fill: bg, fontSize: 16 }),
      cell(f.attribute || "—",        { width: 1400, fill: bg, fontSize: 16 }),
      cell(String(f.flagged_value || "—"), { width: 1200, fill: bg, bold: true }),
      cell(f.expected_range || "—",   { width: 1500, fill: bg, italic: true, fontSize: 16 }),
      cell(f.description || "—",      { width: 2300, fill: bg, fontSize: 16 }),
      cell(sev, { width: 626, bold: true, color: severityColor(sev), fill: severityBg(sev), align: AlignmentType.CENTER }),
    ]});
  });

  return [
    sectionDivider("Appendix A – Full Anomaly Flag List"),
    para(`Complete list of all ${flags.length} anomaly flags detected by the Anomaly Detection Engine, sorted by severity.`, { spacing: { after: 180 } }),
    ...(allRows.length === 0
      ? [para("No anomaly flags detected.", { italic: true })]
      : [
          new Table({
            width: { size: CONTENT_W, type: WidthType.DXA },
            columnWidths: [400, 1600, 1400, 1200, 1500, 2300, 626],
            rows: [
              hdrRow([["#", 400], ["Anomaly Type", 1600], ["Attribute", 1400], ["Flagged Value", 1200], ["Expected Range", 1500], ["Description", 2300], ["Severity", 626]]),
              ...allRows,
            ],
          }),
        ]
    ),
  ];
}

// ─── Assemble document ────────────────────────────────────────────────────────

const valRows    = getValidationRows(valReport);
const score      = getScore(valReport);
const consistency = engReport.consistency || {};
const anomaly     = engReport.anomaly     || {};

const docChildren = [
  ...buildCoverPage(valReport, score),
  ...buildTOC(),
  ...buildExecutiveSummary(valReport, engReport, score, valRows),
  ...buildKGStats(valReport),
  ...buildComplianceSection(valRows, score),
  ...buildConsistencySection(consistency),
  ...buildAnomalySection(anomaly),
  ...buildTraceabilityMatrix(valRows, consistency, anomaly),
  ...buildAppendixA(anomaly),
];

const doc = new Document({
  creator: "RITES PARAKH DPR Validation System",
  title:   `DPR Appraisal Report — ${valReport.doc_id}`,
  description: `Automated validation report for ${valReport.sector} DPR`,
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
    ],
  },
  styles: {
    default: {
      document: { run: { font: "Arial", size: 20 } },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run:       { size: 28, bold: true, font: "Arial", color: C.RITES_BLUE },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0,
                     border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.RITES_BLUE_MID, space: 4 } } },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run:       { size: 24, bold: true, font: "Arial", color: C.RITES_BLUE_MID },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 },
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          spacing: { after: 0 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: C.RITES_BLUE_MID, space: 4 } },
          children: [
            new TextRun({ text: "RITES LIMITED — DPR Appraisal Report", bold: true, size: 18, color: C.RITES_BLUE }),
            new TextRun({ text: `\t${valReport.sector} | Doc: ${valReport.doc_id}`, size: 16, color: C.MED_GREY }),
          ],
          tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          spacing: { before: 0 },
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: C.RITES_BLUE_MID, space: 4 } },
          children: [
            new TextRun({ text: "CONFIDENTIAL — PARAKH Automated Validation", size: 16, color: C.MED_GREY, italics: true }),
            new TextRun({ text: "\tPage ", size: 16, color: C.DARK_GREY }),
            new TextRun({ children: [PageNumber.CURRENT], size: 16, color: C.DARK_GREY }),
            new TextRun({ text: " of ", size: 16, color: C.DARK_GREY }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16, color: C.DARK_GREY }),
          ],
          tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
        })],
      }),
    },
    children: docChildren,
  }],
});

// ─── Write file ────────────────────────────────────────────────────────────────

const outFile = path.join(outputDir, `DPR_Appraisal_Report_${docId}.docx`);

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outFile, buffer);
  console.log(`Report written: ${outFile}`);
}).catch(err => {
  console.error("Failed to generate report:", err);
  process.exit(1);
});