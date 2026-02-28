"""
pdf_report.py — PD-PREVeNT PDF report generator
Uses reportlab to produce a single A4 page academic-style risk report.
Call generate_report(data) → returns bytes of the PDF.
"""

from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ── Colour palette ────────────────────────────────────────────────────────────
TEAL        = colors.HexColor("#0b4f58")
TEAL_LIGHT  = colors.HexColor("#e6f2f4")
RED         = colors.HexColor("#c0392b")
RED_LIGHT   = colors.HexColor("#fdecea")
GREEN       = colors.HexColor("#27ae60")
GREEN_LIGHT = colors.HexColor("#eafaf1")
ORANGE      = colors.HexColor("#e67e22")
GREY        = colors.HexColor("#95a5a6")
GREY_LIGHT  = colors.HexColor("#f8f9fa")
WHITE       = colors.white
BLACK       = colors.HexColor("#2c3e50")

# ── Style helpers ─────────────────────────────────────────────────────────────

def _style(name, **kwargs):
    defaults = dict(fontName="Helvetica", fontSize=10, textColor=BLACK, leading=14)
    defaults.update(kwargs)
    return ParagraphStyle(name, **defaults)

H1    = _style("H1", fontName="Helvetica-Bold", fontSize=16, textColor=TEAL,    alignment=TA_CENTER, leading=20)
H2    = _style("H2", fontName="Helvetica-Bold", fontSize=11, textColor=TEAL,    leading=16)
BODY  = _style("BODY", fontSize=9,  leading=13)
SMALL = _style("SMALL", fontSize=8, textColor=GREY, leading=11)
DISC  = _style("DISC", fontSize=7.5, textColor=GREY, leading=10, alignment=TA_CENTER)
BOLD  = _style("BOLD", fontName="Helvetica-Bold", fontSize=9, leading=13)
CENTER= _style("CENTER", fontSize=9, leading=13, alignment=TA_CENTER)
RIGHT = _style("RIGHT", fontSize=9, leading=13, alignment=TA_RIGHT)

RISK_HIGH = _style("RISK_HIGH", fontName="Helvetica-Bold", fontSize=18,
                   textColor=RED, alignment=TA_CENTER, leading=24)
RISK_LOW  = _style("RISK_LOW",  fontName="Helvetica-Bold", fontSize=18,
                   textColor=GREEN, alignment=TA_CENTER, leading=24)

# ── Probability bar (table-based, no drawing canvas needed) ──────────────────

def _prob_bar_table(label, prob, bar_color, width_mm=60):
    """Return a 1-row Table that looks like a progress bar."""
    pct     = round(prob * 100)
    filled  = max(1, round(width_mm * prob))
    empty   = width_mm - filled

    bar_data = [[""]]
    bar_style = [
        ("BACKGROUND", (0,0), (0,0), bar_color),
        ("ROWHEIGHT",  (0,0), (0,0), 5),
        ("LEFTPADDING",(0,0), (0,0), 0),("RIGHTPADDING",(0,0),(0,0),0),
        ("TOPPADDING", (0,0), (0,0), 0),("BOTTOMPADDING",(0,0),(0,0),0),
    ]
    bar_inner = Table(bar_data, colWidths=[filled*mm], rowHeights=[5*mm])
    bar_inner.setStyle(TableStyle(bar_style))

    # outer track
    outer_data = [[bar_inner, ""]]
    outer = Table(outer_data, colWidths=[filled*mm, empty*mm], rowHeights=[5*mm])
    outer.setStyle(TableStyle([
        ("BACKGROUND", (1,0),(1,0), colors.HexColor("#e8e8e8")),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
        ("TOPPADDING", (0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
    ]))

    row = Table(
        [[Paragraph(label, BODY), Paragraph(str(pct)+"%", BOLD), outer]],
        colWidths=[55*mm, 12*mm, width_mm*mm]
    )
    row.setStyle(TableStyle([
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),2),
        ("TOPPADDING", (0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),
    ]))
    return row

# ── Main generator ────────────────────────────────────────────────────────────

def generate_report(data: dict) -> bytes:
    """
    data keys:
        patientName  str
        age          str/int
        gender       str  ("Male"/"Female")
        features     list of {label, score}
        result       dict from /submit JSON
    Returns bytes of a single-page A4 PDF.
    """
    buf = BytesIO()

    W, H = A4
    margin = 18*mm

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=16*mm,   bottomMargin=14*mm,
    )

    r       = data.get("result", {})
    isPD    = r.get("userStatus") == "PROFILE_1"
    n_provided  = r.get("numProvided", len(data.get("features", [])))
    n_stacker   = len(r.get("usedFeatures", [])) + len(r.get("defaultedFeatures", []))
    n_defaulted = max(0, n_stacker - n_provided)
    risk_lbl = "HIGH RISK — Parkinson's Disease" if isPD else "LOW RISK — Healthy / No PD"
    risk_sty = RISK_HIGH if isPD else RISK_LOW
    risk_bg  = RED_LIGHT  if isPD else GREEN_LIGHT
    risk_col = RED        if isPD else GREEN

    band      = r.get("confidenceBand", "")
    band_col  = RED if "Low" in band else (ORANGE if "Moderate" in band else GREEN)
    conf_pct  = r.get("confidenceScore", 0)
    final_prob = r.get("finalProb", 0)

    story = []

    # ── HEADER ────────────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("PD-PREVeNT", _style("HH", fontName="Helvetica-Bold", fontSize=20,
                                        textColor=WHITE, alignment=TA_LEFT, leading=24)),
        Paragraph("Parkinson's Disease Prognostic Risk Evaluation<br/>"
                  "using Non-Motor Tests &amp; Three-Model ML Ensemble",
                  _style("HS", fontSize=8, textColor=colors.HexColor("#d0e8ec"),
                         alignment=TA_RIGHT, leading=11)),
    ]]
    hdr = Table(header_data, colWidths=[80*mm, W - 2*margin - 80*mm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), TEAL),
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0),(-1,-1), 8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(0,0),10),("RIGHTPADDING",(-1,-1),(-1,-1),10),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 5*mm))

    # ── DATE + PATIENT INFO ───────────────────────────────────────────────────
    name   = data.get("patientName") or "User"
    age    = data.get("age", "—")
    gender = data.get("gender", "—")
    date   = datetime.now().strftime("%d %B %Y")

    info_data = [
        [Paragraph("<b>Patient Name</b>", BODY), Paragraph(str(name),   BODY),
         Paragraph("<b>Date</b>",         BODY), Paragraph(date,         BODY)],
        [Paragraph("<b>Age</b>",          BODY), Paragraph(str(age)+" yrs", BODY),
         Paragraph("<b>Sex</b>",          BODY), Paragraph(str(gender),  BODY)],
    ]
    col_w = (W - 2*margin) / 4
    info_tbl = Table(info_data, colWidths=[col_w*0.55, col_w*1.15, col_w*0.55, col_w*1.15])
    info_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), GREY_LIGHT),
        ("TOPPADDING", (0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("GRID", (0,0),(-1,-1), 0.3, colors.HexColor("#dce1e3")),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 4*mm))

    # ── RISK OUTCOME BOX ─────────────────────────────────────────────────────
    pd_pct  = final_prob * 100
    hl_pct  = 100 - pd_pct
    primary_pct = pd_pct if isPD else hl_pct
    risk_tbl = Table(
        [[Paragraph(risk_lbl, risk_sty),
          Paragraph(
            "<b>%.1f%%</b>" % primary_pct,
            _style("PP", fontName="Helvetica-Bold", fontSize=18,
                   textColor=risk_col, alignment=TA_CENTER, leading=22))]],
        colWidths=[(W-2*margin)*0.62, (W-2*margin)*0.38]
    )
    risk_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), risk_bg),
        ("TOPPADDING", (0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("VALIGN",     (0,0),(-1,-1),"MIDDLE"),
        ("LINEBELOW",  (0,0),(-1,-1), 1.5, risk_col),
    ]))
    story.append(risk_tbl)
    story.append(Spacer(1, 4*mm))

    # ── TWO-COLUMN SECTION: Features | Confidence + Models ───────────────────
    left_story  = []
    right_story = []

    # LEFT — Features entered
    left_story.append(Paragraph("Clinical Features Provided", H2))
    left_story.append(Spacer(1, 1.5*mm))
    feats = data.get("features", [])
    if feats:
        feat_rows = [
            [Paragraph("<b>Feature</b>",  _style("FH", fontSize=8, fontName="Helvetica-Bold")),
             Paragraph("<b>Score</b>",    _style("FH", fontSize=8, fontName="Helvetica-Bold", alignment=TA_CENTER))]
        ]
        for f in feats:
            feat_rows.append([
                Paragraph(f.get("label", f.get("name", "")), _style("FV", fontSize=8.5, leading=12)),
                Paragraph(str(f.get("score", "")),           _style("FV", fontSize=8.5, leading=12, alignment=TA_CENTER)),
            ])
        lw = (W - 2*margin) * 0.44
        ft = Table(feat_rows, colWidths=[lw*0.78, lw*0.22])
        ft.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,0),  TEAL),
            ("TEXTCOLOR",    (0,0),(-1,0),  WHITE),
            ("BACKGROUND",   (0,1),(-1,-1), GREY_LIGHT),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, GREY_LIGHT]),
            ("GRID",         (0,0),(-1,-1), 0.3, colors.HexColor("#dce1e3")),
            ("TOPPADDING",   (0,0),(-1,-1), 3),("BOTTOMPADDING",(0,0),(-1,-1),3),
            ("LEFTPADDING",  (0,0),(-1,-1), 5),("RIGHTPADDING",(0,0),(-1,-1),5),
            ("VALIGN",       (0,0),(-1,-1),"MIDDLE"),
        ]))
        left_story.append(ft)
        feat_note = str(n_provided) + " feature(s) measured; " + str(n_defaulted) + " used population-average defaults."
        left_story.append(Spacer(1, 2*mm))
        left_story.append(Paragraph(feat_note, SMALL))
    else:
        left_story.append(Paragraph("No features recorded.", BODY))

    # RIGHT — Confidence + sub-model votes + probabilities
    # Confidence
    right_story.append(Paragraph("Assessment Confidence", H2))
    right_story.append(Spacer(1, 1.5*mm))

    conf_data = [
        [Paragraph("Confidence Band", BODY),
         Paragraph("<b>%s</b>" % band,
                   _style("CB", fontName="Helvetica-Bold", fontSize=9, textColor=band_col))],
        [Paragraph("Confidence Score", BODY),
         Paragraph("<b>%d%%</b>" % conf_pct,
                   _style("CS", fontName="Helvetica-Bold", fontSize=9))],
        [Paragraph("Probability Margin", BODY),
         Paragraph("%.1f%%" % (r.get("probabilityMargin",0)*100), BODY)],
        [Paragraph("Model Consensus", BODY),
         Paragraph("%.0f%%" % (r.get("modelConsensus",0)*100), BODY)],
    ]
    rw = (W - 2*margin) * 0.52
    ct = Table(conf_data, colWidths=[rw*0.56, rw*0.44])
    ct.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,-1), GREY_LIGHT),
        ("GRID",        (0,0),(-1,-1), 0.3, colors.HexColor("#dce1e3")),
        ("TOPPADDING",  (0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING", (0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ]))
    right_story.append(ct)
    right_story.append(Spacer(1, 3*mm))

    # Sub-model votes
    right_story.append(Paragraph("Sub-Model Votes", H2))
    right_story.append(Spacer(1, 1.5*mm))
    votes = r.get("subModelVotes", {})
    vote_rows = []
    for model, vote in votes.items():
        match = (vote == "PD") == isPD
        v_col = RED if vote == "PD" else GREEN
        icon  = "v" if match else "x"
        short = model.replace("M1 (Feature-Stacker)","M1 Feature-Stacker")\
                     .replace("M2 (XGB-Only)","M2 XGB-Only")\
                     .replace("M3 (CB-Collab-Filter)","M3 CB-Collab")
        vote_rows.append([
            Paragraph(icon + "  " + short, _style("VR", fontSize=8.5, leading=12)),
            Paragraph("<b>" + vote + "</b>",
                      _style("VV", fontName="Helvetica-Bold", fontSize=8.5,
                             textColor=v_col, alignment=TA_CENTER, leading=12)),
        ])
    vt = Table(vote_rows, colWidths=[rw*0.72, rw*0.28])
    vt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), GREY_LIGHT),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#dce1e3")),
        ("TOPPADDING",    (0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING",   (0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
    ]))
    right_story.append(vt)

    # Assemble two-column layout
    left_cell  = [item for item in left_story]
    right_cell = [item for item in right_story]

    two_col = Table(
        [[left_cell, right_cell]],
        colWidths=[(W-2*margin)*0.46, (W-2*margin)*0.54]
    )
    two_col.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",  (0,0),(0,0),0),
        ("RIGHTPADDING", (0,0),(0,0),4*mm),
        ("LEFTPADDING",  (1,0),(1,0),0),
        ("RIGHTPADDING", (1,0),(1,0),0),
        ("TOPPADDING",   (0,0),(-1,-1),0),
        ("BOTTOMPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(two_col)
    story.append(Spacer(1, 4*mm))

    # ── PER-MODEL PROBABILITY BARS ────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=TEAL_LIGHT))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph("Per-Model Probability Breakdown", H2))
    story.append(Spacer(1, 2*mm))

    bar_defs = [
        ("M1  Feature-Stacker",  r.get("probM1",0),  colors.HexColor("#3498db")),
        ("M2  XGB-Only",         r.get("probM2",0),  colors.HexColor("#9b59b6")),
        ("M3  CB-Collab Filter", r.get("probM3",0),  colors.HexColor("#1abc9c")),
        ("Final Ensemble", r.get("finalProb",0), risk_col),
    ]

    bar_rows = []
    for label, prob, color in bar_defs:
        bar_rows.append([_prob_bar_table(label, prob, color, width_mm=62)])

    bt = Table(bar_rows, colWidths=[W - 2*margin])
    bt.setStyle(TableStyle([
        ("TOPPADDING",   (0,0),(-1,-1),1),("BOTTOMPADDING",(0,0),(-1,-1),1),
        ("LEFTPADDING",  (0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(bt)
    story.append(Spacer(1, 3*mm))

    # ── METHODOLOGY NOTE ─────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=TEAL_LIGHT))
    story.append(Spacer(1, 2*mm))
    method = ("Confidence = (Probability Margin + Model Consensus) / 2.  "
              "M1: LR+XGB feature stacker per biomarker.  "
              "M2: XGBoost global model (d=6, n=600).  "
              "M3: Collaborative Filter using cosine, Pearson, and Euclidean RBF similarity (K=7 each class).  "
              "Final: Logistic-regression meta-stacker on M1/M2/M3 out-of-fold predictions.")
    story.append(Paragraph(method, SMALL))
    story.append(Spacer(1, 2*mm))

    # ── DISCLAIMER ───────────────────────────────────────────────────────────
    disc_tbl = Table(
        [[Paragraph(
            "DISCLAIMER: PD-PREVeNT is an academic research tool developed at the University of Hyderabad. "
            "It does NOT constitute a clinical diagnosis and must not replace professional medical advice. "
            "Consult a qualified neurologist for evaluation. Results subject to inherent model limitations.",
            DISC
        )]],
        colWidths=[W - 2*margin]
    )
    disc_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,-1), TEAL_LIGHT),
        ("TOPPADDING",  (0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING", (0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("LINEABOVE",   (0,0),(-1,-1), 1, TEAL),
    ]))
    story.append(disc_tbl)

    doc.build(story)
    return buf.getvalue()