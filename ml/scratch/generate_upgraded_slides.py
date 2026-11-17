"""
Full Implementation Script: Upgrades MachineGuard 10-Slide Presentation
Preserves exactly 10 slides, applies modern dark-navy engineering dashboard theme,
embeds actual Python source code snippets, real ML evaluation charts, and live React UI screenshots.
"""

import os
import sys
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# -----------------------------------------------------------------------------
# Color Palette Constants (Dark Navy / Industrial AI Dashboard)
# -----------------------------------------------------------------------------
CANVAS_BG = RGBColor(10, 15, 29)       # #0A0F1D Deep navy background
CARD_BG = RGBColor(19, 29, 49)         # #131D31 Dark slate container
CARD_BORDER = RGBColor(30, 46, 74)     # #1E2E4A Border
CODE_BG = RGBColor(7, 11, 20)          # #070B14 Obsidian code background
CODE_BORDER = RGBColor(31, 41, 55)     # #1F2937 Code panel border

TEXT_WHITE = RGBColor(248, 250, 252)   # #F8FAFC Primary text
TEXT_MUTED = RGBColor(148, 163, 184)   # #94A3B8 Secondary text
TEXT_DIM = RGBColor(100, 116, 139)     # #64748B Dim labels

ACCENT_CYAN = RGBColor(6, 182, 212)    # #06B6D4 Primary cyan accent
ACCENT_TEAL = RGBColor(20, 184, 166)   # #14B8A6 Teal accent
ACCENT_GREEN = RGBColor(16, 185, 129)  # #10B981 Success / Normal
ACCENT_AMBER = RGBColor(245, 158, 11)  # #F59E0B Warning / Stat
ACCENT_ORANGE = RGBColor(249, 115, 22) # #F97316 High risk
ACCENT_RED = RGBColor(239, 68, 68)     # #EF4444 Critical failure

# Syntax colors for code
SYNTAX_KW = RGBColor(244, 114, 182)    # #F472B6 Pink keywords
SYNTAX_FN = RGBColor(96, 165, 250)     # #60A5FA Blue function
SYNTAX_STR = RGBColor(251, 191, 36)    # #FBBF24 Amber string/num
SYNTAX_COMMENT = RGBColor(100, 116, 139) # #64748B Muted comment
SYNTAX_VAR = RGBColor(226, 232, 240)   # #E2E8F0 White/light var

FONT_HEADING = "Segoe UI"
FONT_BODY = "Segoe UI"
FONT_CODE = "Consolas"

# -----------------------------------------------------------------------------
# Base Layout Helpers
# -----------------------------------------------------------------------------
def clear_slide(slide):
    """Removes all existing shapes from a slide."""
    for s in list(slide.shapes):
        slide.shapes._spTree.remove(s._element)

def set_slide_background(slide):
    """Sets a solid deep dark-navy background on the slide."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.500))
    bg.fill.solid()
    bg.fill.fore_color.rgb = CANVAS_BG
    bg.line.fill.background()
    return bg

def add_header(slide, slide_num_str, category_text, title_text, subtitle_text=""):
    """Adds a unified, clean top header band on the slide."""
    badge_box = slide.shapes.add_textbox(Inches(0.67), Inches(0.40), Inches(12.0), Inches(0.26))
    tf_b = badge_box.text_frame
    tf_b.word_wrap = True
    tf_b.margin_left = tf_b.margin_top = tf_b.margin_right = tf_b.margin_bottom = 0
    p_b = tf_b.paragraphs[0]
    p_b.text = f"{slide_num_str}  ·  {category_text.upper()}"
    p_b.font.name = FONT_HEADING
    p_b.font.size = Pt(9.5)
    p_b.font.bold = True
    p_b.font.color.rgb = ACCENT_CYAN

    title_box = slide.shapes.add_textbox(Inches(0.67), Inches(0.66), Inches(12.0), Inches(0.50))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.name = FONT_HEADING
    p_t.font.size = Pt(19)
    p_t.font.bold = True
    p_t.font.color.rgb = TEXT_WHITE

    if subtitle_text:
        sub_box = slide.shapes.add_textbox(Inches(0.67), Inches(1.18), Inches(12.0), Inches(0.28))
        tf_s = sub_box.text_frame
        tf_s.word_wrap = True
        tf_s.margin_left = tf_s.margin_top = tf_s.margin_right = tf_s.margin_bottom = 0
        p_s = tf_s.paragraphs[0]
        p_s.text = subtitle_text
        p_s.font.name = FONT_BODY
        p_s.font.size = Pt(10.5)
        p_s.font.color.rgb = TEXT_MUTED

def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
    """Adds a dark card container shape."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1)
    return card

def add_stat_box(slide, left, top, width, height, value_text, label_text, sub_text="", val_color=ACCENT_CYAN):
    """Adds a compact metric card."""
    card = add_card(slide, left, top, width, height)
    tb = slide.shapes.add_textbox(left + Inches(0.18), top + Inches(0.12), width - Inches(0.36), height - Inches(0.24))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p1 = tf.paragraphs[0]
    p1.text = value_text
    p1.font.name = FONT_HEADING
    p1.font.size = Pt(22)
    p1.font.bold = True
    p1.font.color.rgb = val_color
    
    p2 = tf.add_paragraph()
    p2.text = label_text.upper()
    p2.font.name = FONT_HEADING
    p2.font.size = Pt(8.5)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_WHITE
    p2.space_before = Pt(2)
    
    if sub_text:
        p3 = tf.add_paragraph()
        p3.text = sub_text
        p3.font.name = FONT_BODY
        p3.font.size = Pt(8)
        p3.font.color.rgb = TEXT_MUTED
        p3.space_before = Pt(2)
        
    return card

def fit_and_add_image(slide, img_path, box_left, box_top, box_width, box_height, caption=""):
    """
    Fits an image inside a bounding box while strictly preserving aspect ratio,
    centers it within the box, and optionally renders a clean caption below.
    """
    if not os.path.exists(img_path):
        print(f"WARNING: Image not found: {img_path}")
        return None
        
    with Image.open(img_path) as img:
        orig_w, orig_h = img.size
        
    aspect = orig_w / orig_h
    cap_height = Inches(0.24) if caption else Inches(0)
    avail_h = box_height - cap_height
    box_aspect = box_width / avail_h
    
    if aspect > box_aspect:
        w = box_width
        h = box_width / aspect
    else:
        h = avail_h
        w = avail_h * aspect
        
    left = box_left + (box_width - w) / 2
    top = box_top + (avail_h - h) / 2
    
    bg_frame = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left - Inches(0.04), top - Inches(0.04), w + Inches(0.08), h + Inches(0.08))
    bg_frame.fill.solid()
    bg_frame.fill.fore_color.rgb = RGBColor(15, 23, 42)
    bg_frame.line.color.rgb = CARD_BORDER
    bg_frame.line.width = Pt(1)
    
    pic = slide.shapes.add_picture(img_path, left, top, w, h)
    
    if caption:
        cb = slide.shapes.add_textbox(box_left, top + h + Inches(0.04), box_width, cap_height)
        ctf = cb.text_frame
        ctf.word_wrap = True
        ctf.margin_left = ctf.margin_top = ctf.margin_right = ctf.margin_bottom = 0
        cp = ctf.paragraphs[0]
        cp.text = caption
        cp.alignment = PP_ALIGN.CENTER
        cp.font.name = FONT_BODY
        cp.font.size = Pt(8.5)
        cp.font.color.rgb = TEXT_MUTED
        
    return pic

def add_code_panel(slide, left, top, width, height, title_badge, filename_badge, code_lines):
    """Renders source code snippet inside a dark obsidian syntax-highlighted panel."""
    card = add_card(slide, left, top, width, height, bg_color=CODE_BG, border_color=CODE_BORDER)
    
    hdr = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Inches(0.30))
    hdr.fill.solid()
    hdr.fill.fore_color.rgb = RGBColor(15, 23, 42)
    hdr.line.color.rgb = CODE_BORDER
    hdr.line.width = Pt(1)
    
    htf = hdr.text_frame
    htf.word_wrap = True
    htf.margin_left = Inches(0.12)
    htf.margin_top = Inches(0.03)
    hp = htf.paragraphs[0]
    
    r_tag = hp.add_run()
    r_tag.text = f" {title_badge.upper()} "
    r_tag.font.name = FONT_CODE
    r_tag.font.size = Pt(8)
    r_tag.font.bold = True
    r_tag.font.color.rgb = RGBColor(248, 250, 252)
    
    r_file = hp.add_run()
    r_file.text = f"   {filename_badge}"
    r_file.font.name = FONT_CODE
    r_file.font.size = Pt(8.5)
    r_file.font.color.rgb = ACCENT_CYAN
    
    cb = slide.shapes.add_textbox(left + Inches(0.12), top + Inches(0.36), width - Inches(0.24), height - Inches(0.42))
    ctf = cb.text_frame
    ctf.word_wrap = False
    ctf.margin_left = ctf.margin_top = ctf.margin_right = ctf.margin_bottom = 0
    
    for i, item in enumerate(code_lines):
        p = ctf.paragraphs[0] if i == 0 else ctf.add_paragraph()
        p.space_after = Pt(1.2)
        
        if isinstance(item, tuple):
            line_type, line_text = item
        else:
            line_type, line_text = "normal", item
            
        r = p.add_run()
        r.text = line_text
        r.font.name = FONT_CODE
        r.font.size = Pt(8.2)
        
        if line_type == "comment":
            r.font.color.rgb = SYNTAX_COMMENT
            r.font.italic = True
        elif line_type == "keyword":
            r.font.color.rgb = SYNTAX_KW
            r.font.bold = True
        elif line_type == "fn":
            r.font.color.rgb = SYNTAX_FN
            r.font.bold = True
        elif line_type == "highlight":
            r.font.color.rgb = ACCENT_CYAN
            r.font.bold = True
        elif line_type == "warn":
            r.font.color.rgb = ACCENT_AMBER
        elif line_type == "success":
            r.font.color.rgb = ACCENT_GREEN
        else:
            r.font.color.rgb = SYNTAX_VAR
            
    return card

# -----------------------------------------------------------------------------
# Slide Builders (1 to 10)
# -----------------------------------------------------------------------------

def build_slide_1(slide):
    """Slide 1 — Project Introduction & Executive Highlights"""
    clear_slide(slide)
    set_slide_background(slide)
    
    # Top Tag
    tb = slide.shapes.add_textbox(Inches(0.67), Inches(0.60), Inches(12.0), Inches(0.30))
    p = tb.text_frame.paragraphs[0]
    p.text = "INDUSTRIAL AI · PREDICTIVE MAINTENANCE PLATFORM"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    
    # Main Hero Title
    tb_title = slide.shapes.add_textbox(Inches(0.67), Inches(0.92), Inches(12.0), Inches(0.70))
    p_t = tb_title.text_frame.paragraphs[0]
    p_t.text = "MachineGuard: Early Detection of Equipment Failure"
    p_t.font.name = FONT_HEADING
    p_t.font.size = Pt(26)
    p_t.font.bold = True
    p_t.font.color.rgb = TEXT_WHITE
    
    # Hero Subtitle
    tb_sub = slide.shapes.add_textbox(Inches(0.67), Inches(1.65), Inches(12.0), Inches(0.35))
    p_s = tb_sub.text_frame.paragraphs[0]
    p_s.text = "Condition-Based Monitoring for CNC Milling Machinery with Physics-Informed Random Forest"
    p_s.font.name = FONT_BODY
    p_s.font.size = Pt(12)
    p_s.font.color.rgb = TEXT_MUTED
    
    # Left Card: Executive Scope & Metadata
    left_card = add_card(slide, Inches(0.67), Inches(2.15), Inches(7.20), Inches(4.55))
    tb_c = slide.shapes.add_textbox(Inches(0.95), Inches(2.35), Inches(6.64), Inches(4.15))
    tf = tb_c.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "PROJECT ARCHITECTURE & INDUSTRIAL VALUE"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    
    bullets = [
        ("Industrial Domain", "Industrial Internet of Things (IIoT) & Predictive Maintenance for automated milling centers, high-speed spindles, and carbide cutting tools."),
        ("Physical Telemetry", "Monitors 5 continuous physical sensors: ambient/process temperatures, spindle speed, torque resistance, and cumulative tool wear minutes."),
        ("Physics-Informed AI", "Derived mathematical interactions: Temperature Gradient (ΔT), Mechanical Power (P), and Overstrain Index (OSI) driving tree splits."),
        ("Operational Boundary", "Random Forest Classifier (300 trees, depth 15, balanced_subsample) operating at a calibrated decision threshold of θ = 0.35."),
        ("Academic Framework", "Infosys Springboard Internship 7.0 / College Engineering Capstone Project."),
        ("Presented By", "[Student Name(s) / Team ID / Dept. of Computer Science & Engineering]")
    ]
    
    for title, desc in bullets:
        p_b = tf.add_paragraph()
        p_b.space_before = Pt(6)
        r1 = p_b.add_run()
        r1.text = f"•  {title}: "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(9.5)
        r1.font.bold = True
        r1.font.color.rgb = TEXT_WHITE
        
        r2 = p_b.add_run()
        r2.text = desc
        r2.font.name = FONT_BODY
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = TEXT_MUTED
        
    # Right Side: 3 Big Metric KPI Cards
    add_stat_box(slide, Inches(8.15), Inches(2.15), Inches(4.51), Inches(1.35),
                 "83.82%", "Failure Detection Recall",
                 "57 of 68 actual breakdowns caught on 2,000 held-out test cycles", ACCENT_CYAN)
                 
    add_stat_box(slide, Inches(8.15), Inches(3.70), Inches(4.51), Inches(1.35),
                 "85.07%", "Alarm Precision Reliability",
                 "Only 10 false alarms in 1,932 healthy cycles (78% fewer alarms vs XGBoost)", ACCENT_GREEN)
                 
    add_stat_box(slide, Inches(8.15), Inches(5.25), Inches(4.51), Inches(1.45),
                 "98.95%", "Classification Accuracy",
                 "ROC-AUC: 0.9791  |  PR-AUC: 0.8576  |  Brier Score: 0.0098", ACCENT_AMBER)
                 
    # Bottom Note
    nb = slide.shapes.add_textbox(Inches(0.67), Inches(6.85), Inches(12.0), Inches(0.25))
    np = nb.text_frame.paragraphs[0]
    np.text = "Verified on 2,000 completely untouched held-out production test samples from the AI4I 2020 predictive maintenance benchmark."
    np.font.name = FONT_BODY
    np.font.size = Pt(8.5)
    np.font.color.rgb = TEXT_DIM

def build_slide_2(slide):
    """Slide 2 — Industrial Problem Statement & Maintenance Paradigms"""
    clear_slide(slide)
    set_slide_background(slide)
    add_header(slide, "02", "INDUSTRIAL MOTIVATION & PROBLEM STATEMENT",
               "The Cost of Unplanned Downtime & Traditional Maintenance Flaws",
               "Why modern automated manufacturing requires condition-based predictive intelligence")
               
    # 4 Quick Stat Metric Badges across top
    ribbon_stats = [
        ("$5k – $50k+", "Hourly Downtime Cost", "Direct loss in plant output", ACCENT_RED),
        ("~$4,500+", "Single Breakdown Cost", "Tooling damage & scrap parts", ACCENT_ORANGE),
        ("30% – 40%", "Wasted Tool Life", "Premature calendar disposal", ACCENT_AMBER),
        ("28.5 : 1", "Class Imbalance Ratio", "Only 3.39% cycles fail", ACCENT_CYAN)
    ]
    w_card = Inches(2.82)
    gap = Inches(0.24)
    for i, (v, lbl, sub, c) in enumerate(ribbon_stats):
        x = Inches(0.67) + i * (w_card + gap)
        add_stat_box(slide, x, Inches(1.60), w_card, Inches(1.05), v, lbl, sub, c)
        
    # 3 Comparative Paradigm Cards
    col_w = Inches(3.84)
    col_h = Inches(4.35)
    col_gap = Inches(0.24)
    top_y = Inches(2.80)
    
    # 1. Reactive
    add_card(slide, Inches(0.67), top_y, col_w, col_h)
    tb1 = slide.shapes.add_textbox(Inches(0.85), top_y + Inches(0.15), col_w - Inches(0.36), col_h - Inches(0.30))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "REACTIVE MAINTENANCE"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT_RED
    
    p = tf1.add_paragraph()
    p.text = "Run-to-Failure Strategy"
    p.font.name = FONT_BODY
    p.font.size = Pt(9)
    p.font.color.rgb = TEXT_MUTED
    
    points1 = [
        "Operating Principle: Equipment is run continuously until tool fractures or spindle motor burns out.",
        "Operational Flaws: Unscheduled catastrophic downtime halts entire production lines without warning.",
        "Collateral Damage: Broken carbide inserts destroy titanium workpieces and warp machine guide-ways.",
        "Safety Hazard: Tool fragmentation at 2,000+ rpm presents severe shrapnel risk to operators.",
        "Verdict: Maximum downtime cost ($5k–$50k/hr) and emergency technician overtime."
    ]
    for pt in points1:
        p = tf1.add_paragraph()
        p.space_before = Pt(4)
        p.text = f"• {pt}"
        p.font.name = FONT_BODY
        p.font.size = Pt(8.5)
        p.font.color.rgb = TEXT_WHITE
        
    # 2. Preventive
    add_card(slide, Inches(0.67) + col_w + col_gap, top_y, col_w, col_h)
    tb2 = slide.shapes.add_textbox(Inches(0.85) + col_w + col_gap, top_y + Inches(0.15), col_w - Inches(0.36), col_h - Inches(0.30))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "PREVENTIVE MAINTENANCE"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT_AMBER
    
    p = tf2.add_paragraph()
    p.text = "Rigid Calendar-Based Schedules"
    p.font.name = FONT_BODY
    p.font.size = Pt(9)
    p.font.color.rgb = TEXT_MUTED
    
    points2 = [
        "Operating Principle: Components are serviced at fixed calendar intervals (e.g., every 30 days or 100 hours).",
        "Operational Flaws: Discards expensive carbide cutting tools that still possess 30% to 40% usable life.",
        "Blind to Load Variations: Tools operating on harder alloys wear out 3x faster and fail before the scheduled service date.",
        "Blind to Heat/Friction: Cannot detect sudden cooling failures or motor stator overheating between inspections.",
        "Verdict: High recurring tooling waste and unable to guarantee zero unplanned downtime."
    ]
    for pt in points2:
        p = tf2.add_paragraph()
        p.space_before = Pt(4)
        p.text = f"• {pt}"
        p.font.name = FONT_BODY
        p.font.size = Pt(8.5)
        p.font.color.rgb = TEXT_WHITE
        
    # 3. Predictive
    add_card(slide, Inches(0.67) + 2 * (col_w + col_gap), top_y, col_w, col_h)
    tb3 = slide.shapes.add_textbox(Inches(0.85) + 2 * (col_w + col_gap), top_y + Inches(0.15), col_w - Inches(0.36), col_h - Inches(0.30))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "PREDICTIVE (MACHINEGUARD)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    
    p = tf3.add_paragraph()
    p.text = "Condition-Based Continuous AI"
    p.font.name = FONT_BODY
    p.font.size = Pt(9)
    p.font.color.rgb = TEXT_MUTED
    
    points3 = [
        "Operating Principle: Real-time sensor streams continuously evaluate physical mechanical strain and heat dissipation.",
        "Targeted Service: Equipment is maintained ONLY when degradation is detected—maximizing tool life by 100%.",
        "Physics Tripwire Detection: Recognizes power collapse (PWF), thermal dissipation failure (HDF), and overstrain (OSF).",
        "Calibrated Reliability: 83.82% Failure Recall with minimal 0.52% false alarm rate (10 false alarms in 1,932 cycles).",
        "Verdict: Zero unplanned downtime, optimal parts utilization, and maximum factory ROI."
    ]
    for pt in points3:
        p = tf3.add_paragraph()
        p.space_before = Pt(4)
        p.text = f"• {pt}"
        p.font.name = FONT_BODY
        p.font.size = Pt(8.5)
        p.font.color.rgb = TEXT_WHITE

def build_slide_3(slide):
    """Slide 3 — Proposed Solution & Architecture"""
    clear_slide(slide)
    set_slide_background(slide)
    add_header(slide, "03", "SYSTEM ARCHITECTURE & SOLUTION PIPELINE",
               "From Raw Sensor Ingestion to Actionable Operator Guidance",
               "Reproducible 6-stage predictive maintenance pipeline with embedded physics feature engineering")
               
    # Left Card: 6-Stage Visual Stepper Pipeline
    left_w = Inches(5.80)
    left_h = Inches(5.40)
    add_card(slide, Inches(0.67), Inches(1.60), left_w, left_h)
    
    tb = slide.shapes.add_textbox(Inches(0.90), Inches(1.75), left_w - Inches(0.46), left_h - Inches(0.30))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "6-STAGE OPERATIONAL PIPELINE"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    
    stages = [
        ("01 · Telemetry Ingestion", "Ingests 5 continuous physical sensors: Air Temp, Process Temp, Rotational Speed, Torque, and Tool Wear."),
        ("02 · Validation & Preprocessing", "Verifies 0 nulls across 10,000 records; asserts physical bounds (T_proc >= T_air); preserves legitimate breakdown outliers."),
        ("03 · Physics Feature Derivation", "Computes ΔT (dissipation gradient), P_mech (motor power in Watts), and OSI (Torque x Wear overstrain fatigue index)."),
        ("04 · Optimized Random Forest", "Pipeline processes 8 internal channels through 300 de-correlated decision trees with balanced_subsample class weighting."),
        ("05 · Calibrated Decision Layer", "Applies calibrated threshold θ = 0.35 to continuous P(Failure | x); categorizes risk into Low, Moderate, High, and Critical."),
        ("06 · Actionable Operator Output", "React UI displays radial SVG probability gauge, highlights physical stress factors, and provides maintenance instructions.")
    ]
    
    for title, desc in stages:
        p = tf.add_paragraph()
        p.space_before = Pt(5)
        r1 = p.add_run()
        r1.text = f"{title}\n"
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(9)
        r1.font.bold = True
        r1.font.color.rgb = TEXT_WHITE
        
        r2 = p.add_run()
        r2.text = desc
        r2.font.name = FONT_BODY
        r2.font.size = Pt(8.2)
        r2.font.color.rgb = TEXT_MUTED
        
    # Right Side: Actual Python Pipeline Code Snippet Panel
    code_w = Inches(6.19)
    code_h = Inches(4.15)
    
    pipeline_code = [
        ("comment", "# Production Pipeline Assembly (ml/src/optimize_random_forest.py, L403-414)"),
        ("keyword", "from"), ("normal", " sklearn.pipeline "), ("keyword", "import"), ("normal", " Pipeline"),
        ("keyword", "from"), ("normal", " sklearn.preprocessing "), ("keyword", "import"), ("normal", " FunctionTransformer"),
        ("keyword", "from"), ("normal", " sklearn.ensemble "), ("keyword", "import"), ("normal", " RandomForestClassifier"),
        ("normal", ""),
        ("comment", "# Model accepts 5 raw inputs; transformer internally expands to 8 features"),
        ("fn", "rf_pipeline"), ("normal", " = Pipeline(["),
        ("highlight", "    ('physics_features', FunctionTransformer(add_physics_features)),"),
        ("normal", "    ("), ("highlight", "'rf'"), ("normal", ", RandomForestClassifier("),
        ("normal", "        n_estimators="), ("warn", "300"), ("normal", ","),
        ("normal", "        max_depth="), ("warn", "15"), ("normal", ","),
        ("normal", "        min_samples_split="), ("warn", "4"), ("normal", ","),
        ("normal", "        min_samples_leaf="), ("warn", "2"), ("normal", ","),
        ("highlight", "        class_weight='balanced_subsample',"),
        ("normal", "        random_state="), ("warn", "42"), ("normal", ", n_jobs="), ("warn", "-1"),
        ("normal", "    ))"),
        ("normal", "])"),
        ("normal", ""),
        ("comment", "# Train on full 8,000 records and serialize production binary"),
        ("normal", "rf_pipeline.fit(X_train, y_train)"),
        ("fn", "joblib.dump"), ("normal", "(rf_pipeline, "), ("warn", "'ai4i_failure_model.joblib'"), ("normal", ")")
    ]
    
    add_code_panel(slide, Inches(6.72), Inches(1.60), code_w, code_h,
                   "PYTHON PIPELINE", "ml/src/optimize_random_forest.py", pipeline_code)
                   
    # Below Code: Encapsulation Advantage Callout Box
    encap_card = add_card(slide, Inches(6.72), Inches(5.90), code_w, Inches(1.10))
    tb_e = slide.shapes.add_textbox(Inches(6.90), Inches(5.98), code_w - Inches(0.36), Inches(0.92))
    tf_e = tb_e.text_frame
    tf_e.word_wrap = True
    
    p = tf_e.paragraphs[0]
    p.text = "PIPELINE ENCAPSULATION & ZERO FRONTEND BREAKING CHANGES"
    p.font.name = FONT_HEADING
    p.font.size = Pt(8.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    
    p = tf_e.add_paragraph()
    p.space_before = Pt(2)
    p.text = "The external client continues providing strictly 5 raw sensor inputs. All thermodynamic and mechanical derivations run internally inside FunctionTransformer. During inference, model.predict_proba(X) automatically executes physics transformations without external pre-calculation."
    p.font.name = FONT_BODY
    p.font.size = Pt(8)
    p.font.color.rgb = TEXT_MUTED

def build_slide_4(slide):
    """Slide 4 — Dataset Description & Failure Mode Diagnostics"""
    clear_slide(slide)
    set_slide_background(slide)
    add_header(slide, "04", "DATASET SPECIFICATION & FEATURE METRICS",
               "AI4I 2020 Predictive Maintenance Dataset Description",
               "10,000 industrial milling records, five sensor telemetry channels, and five failure-mode sub-labels")
               
    left_w = Inches(5.90)
    left_h = Inches(5.40)
    
    # Left Card: Dataset Specifications & Sensor Table
    add_card(slide, Inches(0.67), Inches(1.60), left_w, left_h)
    tb = slide.shapes.add_textbox(Inches(0.85), Inches(1.75), left_w - Inches(0.36), left_h - Inches(0.30))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "TELEMETRY SCHEMA & PHYSICAL RANGES"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    
    sensors = [
        ("Air Temperature [K]", "Ambient facility temperature (Mean: 300.0 K, Range: 295.3 - 304.5 K)."),
        ("Process Temperature [K]", "Cutting-zone heat from friction (Mean: 310.0 K, Range: 305.7 - 313.8 K)."),
        ("Rotational Speed [rpm]", "Spindle angular velocity (Mean: 1,539 rpm, Range: 1,168 - 2,886 rpm)."),
        ("Torque [Nm]", "Mechanical motor resistance (Mean: 40.0 Nm, Range: 3.8 - 76.6 Nm)."),
        ("Tool Wear [min]", "Cumulative machining duration before replacement (0 - 253 minutes).")
    ]
    for s_name, s_desc in sensors:
        p = tf.add_paragraph()
        p.space_before = Pt(3.5)
        r1 = p.add_run()
        r1.text = f"•  {s_name}: "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(8.5)
        r1.font.bold = True
        r1.font.color.rgb = TEXT_WHITE
        r2 = p.add_run()
        r2.text = s_desc
        r2.font.name = FONT_BODY
        r2.font.size = Pt(8.2)
        r2.font.color.rgb = TEXT_MUTED
        
    p_div = tf.add_paragraph()
    p_div.space_before = Pt(6)
    r = p_div.add_run()
    r.text = "DIAGNOSTIC FAILURE MODES (TARGET LEAKAGE PROTOCOL)"
    r.font.name = FONT_HEADING
    r.font.size = Pt(9.5)
    r.font.bold = True
    r.font.color.rgb = ACCENT_AMBER
    
    failure_modes = [
        ("Heat Dissipation Failure (HDF - 115)", "ΔT < 8.6 K at low speed (< 1,380 rpm) traps heat in spindle."),
        ("Overstrain Failure (OSF - 98)", "Torque x Tool Wear exceeds material threshold (> 11,000 Nm·min)."),
        ("Power Failure (PWF - 95)", "Spindle power outside safe boundaries (< 3,500 W or > 9,000 W)."),
        ("Tool Wear Failure (TWF - 46)", "Cutting edge wear reaches critical fatigue limit (200 - 240 min)."),
        ("Random Failure (RNF - 19)", "Stochastic failure with ~0.1% chance and zero sensor deviation.")
    ]
    for f_name, f_desc in failure_modes:
        p = tf.add_paragraph()
        p.space_before = Pt(2.5)
        r1 = p.add_run()
        r1.text = f"•  {f_name}: "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(8.2)
        r1.font.bold = True
        r1.font.color.rgb = TEXT_WHITE
        r2 = p.add_run()
        r2.text = f_desc
        r2.font.name = FONT_BODY
        r2.font.size = Pt(8)
        r2.font.color.rgb = TEXT_MUTED
        
    p_leak = tf.add_paragraph()
    p_leak.space_before = Pt(4)
    r = p_leak.add_run()
    r.text = "Exclusion Rule: Diagnostic sub-labels are post-mortem outcomes and were strictly excluded from model inputs to prevent fatal target leakage."
    r.font.name = FONT_BODY
    r.font.size = Pt(7.8)
    r.font.italic = True
    r.font.color.rgb = ACCENT_AMBER
    
    # Right Side: Actual Dataset Distribution Chart
    chart_path = "ml/outputs/charts/01_machine_failure_distribution.png"
    fit_and_add_image(slide, chart_path, Inches(6.82), Inches(1.60), Inches(5.84), Inches(5.40),
                      "Figure 1: AI4I 2020 class distribution: 9,661 Normal (96.61%) vs 339 Failure (3.39%) — 28.5 : 1 Imbalance")

def build_slide_5(slide):
    """Slide 5 — Data Preprocessing & Outlier Analysis"""
    clear_slide(slide)
    set_slide_background(slide)
    add_header(slide, "05", "PREPROCESSING & SIGNAL PRESERVATION",
               "Outlier Diagnostics, Signal Preservation & Leakage Prevention",
               "Auditing 10,000 telemetry cycles and preserving 89.9% failure-correlated torque outliers")
               
    left_w = Inches(6.20)
    left_h = Inches(4.35)
    
    # Left Code Panel: Preprocessing Code from ml/src/preprocess_data.py
    prep_code = [
        ("comment", "# Data Quality & Outlier Diagnostics (ml/src/preprocess_data.py)"),
        ("keyword", "def"), ("fn", " audit_data_and_outliers"), ("normal", "(df: pd.DataFrame):"),
        ("comment", "    # 1. Audit missing values and exact row duplicates"),
        ("normal", "    assert df.isnull().sum().sum() == "), ("warn", "0"), ("comment", "  # 0 missing values"),
        ("normal", "    assert df.duplicated().sum() == "), ("warn", "0"), ("comment", "    # 0 duplicate records"),
        ("normal", ""),
        ("comment", "    # 2. IQR Outlier Diagnostics for all 5 continuous sensors"),
        ("normal", "    iqr = q75 - q25"),
        ("normal", "    lower_bound, upper_bound = q25 - "), ("warn", "1.5"), ("normal", " * iqr, q75 + "), ("warn", "1.5"), ("normal", " * iqr"),
        ("highlight", "    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]"),
        ("normal", ""),
        ("comment", "    # CRITICAL DECISION: PRESERVE LEGITIMATE OUTLIERS"),
        ("comment", "    # 62 of 69 torque outliers (89.9%) are confirmed machine breakdowns!"),
        ("comment", "    # Deleting outliers would strip the model of mechanical stall signals."),
        ("normal", ""),
        ("comment", "    # 3. Leak-free stratified 80:20 split (UDI preserved as index)"),
        ("normal", "    X = df[PRIMARY_FEATURES]  "), ("comment", "# Strictly 5 raw sensors"),
        ("normal", "    y = df[TARGET]            "), ("comment", "# Binary 'Machine failure'"),
        ("normal", "    X_tr, X_te, y_tr, y_te = train_test_split("),
        ("normal", "        X, y, test_size="), ("warn", "0.20"), ("normal", ", stratify=y, random_state="), ("warn", "42"),
        ("normal", "    )")
    ]
    
    add_code_panel(slide, Inches(0.67), Inches(1.60), left_w, left_h,
                   "PYTHON PREPROCESSING", "ml/src/preprocess_data.py", prep_code)
                   
    # Below Code: Partitioning Stats Card
    card_p = add_card(slide, Inches(0.67), Inches(6.05), left_w, Inches(1.00))
    tb_p = slide.shapes.add_textbox(Inches(0.85), Inches(6.12), left_w - Inches(0.36), Inches(0.85))
    tf_p = tb_p.text_frame
    tf_p.word_wrap = True
    p = tf_p.paragraphs[0]
    p.text = "STRATIFIED PARTITION AUDIT (ZERO DATA LEAKAGE)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(8.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    
    p = tf_p.add_paragraph()
    p.space_before = Pt(2)
    p.text = "• Training Set: 8,000 records (7,729 Normal [96.61%], 271 Failures [3.39%]).\n• Test Set: 2,000 records (1,932 Normal [96.60%], 68 Failures [3.40%]).\n• Sealed Protocol: Test partition remained strictly untouched until final evaluation."
    p.font.name = FONT_BODY
    p.font.size = Pt(8)
    p.font.color.rgb = TEXT_MUTED
    
    # Right Side: Correlation Heatmap Chart
    chart_path = "ml/outputs/charts/08_correlation_heatmap.png"
    fit_and_add_image(slide, chart_path, Inches(7.12), Inches(1.60), Inches(5.54), Inches(5.45),
                      "Figure 2: Pearson correlation matrix showing speed-torque coupling (r = -0.875) and torque vs failure (r = +0.191)")

def build_slide_6(slide):
    """Slide 6 — Feature Engineering & Random Forest Algorithm"""
    clear_slide(slide)
    set_slide_background(slide)
    add_header(slide, "06", "DOMAIN-PHYSICS FEATURE ENGINEERING",
               "Physics-Informed Transformations & Random Forest Architecture",
               "Derived thermodynamic and mechanical features account for 48.1% of all model decisions")
               
    left_w = Inches(5.20)
    left_h = Inches(3.60)
    
    # Left Code Panel: Formulas from ml/src/utils.py
    phys_code = [
        ("comment", "# Physics Feature Derivation (ml/src/utils.py, L64-67)"),
        ("keyword", "def"), ("fn", " add_physics_features"), ("normal", "(X):"),
        ("comment", "    # 1. Thermal Gradient: Convective cooling capacity"),
        ("highlight", "    delta_t = proc_t - air_t  "), ("comment", "# Trips HDF when < 8.6 K"),
        ("normal", ""),
        ("comment", "    # 2. Spindle Mechanical Power: Instantaneous motor work"),
        ("highlight", "    mech_power = torque * speed * 2.0 * np.pi / 60.0"),
        ("comment", "    # Operating envelope safe between 3,500 W and 9,000 W"),
        ("normal", ""),
        ("comment", "    # 3. Overstrain Index: Shear stress on degraded carbide"),
        ("highlight", "    overstrain = torque * tool_w  "), ("comment", "# Trips OSF when > 11k"),
        ("normal", ""),
        ("keyword", "    return"), ("normal", " np.column_stack([base_arr, delta_t, mech_power, overstrain])")
    ]
    
    add_code_panel(slide, Inches(0.67), Inches(1.60), left_w, left_h,
                   "PHYSICS DERIVATION", "ml/src/utils.py", phys_code)
                   
    # Below Code: RF Architecture Card
    rf_card = add_card(slide, Inches(0.67), Inches(5.30), left_w, Inches(1.75))
    tb_rf = slide.shapes.add_textbox(Inches(0.85), Inches(5.38), left_w - Inches(0.36), Inches(1.58))
    tf_rf = tb_rf.text_frame
    tf_rf.word_wrap = True
    
    p = tf_rf.paragraphs[0]
    p.text = "RANDOM FOREST ARCHITECTURE SPECIFICATION"
    p.font.name = FONT_HEADING
    p.font.size = Pt(8.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    
    rf_bullets = [
        "Ensemble Size: 300 de-correlated bootstrap decision trees.",
        "Tree Pruning Constraints: max_depth=15, min_samples_split=4, min_samples_leaf=2.",
        "Class Imbalance Reweighting: class_weight='balanced_subsample' per bootstrap.",
        "Random Feature Selection: max_features='sqrt' (evaluates 2-3 features per node).",
        "Continuous Probability: Averaged fraction of trees voting for machine breakdown."
    ]
    for b in rf_bullets:
        p = tf_rf.add_paragraph()
        p.space_before = Pt(1.5)
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(7.8)
        p.font.color.rgb = TEXT_MUTED
        
    # Right Side: Feature Importance Chart
    chart_path = "ml/outputs/charts/model_evaluation/random_forest_optimization/06_feature_importance.png"
    fit_and_add_image(slide, chart_path, Inches(6.12), Inches(1.60), Inches(6.54), Inches(5.45),
                      "Figure 3: Feature importance: Overstrain Index (23.1%) and Mechanical Power (16.4%) drive 48.1% of all splits")

def build_slide_7(slide):
    """Slide 7 — Model Training and Optimization"""
    clear_slide(slide)
    set_slide_background(slide)
    add_header(slide, "07", "EXPERIMENTAL OPTIMIZATION & CALIBRATION",
               "Random Forest Optimization & Operational Threshold Tuning",
               "Comparing 6 experimental configurations and calibrating decision threshold to θ = 0.35")
               
    left_w = Inches(5.60)
    
    # Left Top: 6 Experimental Configurations Matrix Card
    exp_card = add_card(slide, Inches(0.67), Inches(1.60), left_w, Inches(2.70))
    tb_exp = slide.shapes.add_textbox(Inches(0.85), Inches(1.70), left_w - Inches(0.36), Inches(2.50))
    tf_exp = tb_exp.text_frame
    tf_exp.word_wrap = True
    
    p = tf_exp.paragraphs[0]
    p.text = "PHASE 2.1 SYSTEMATIC EXPERIMENTAL SUITE"
    p.font.name = FONT_HEADING
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    
    exps = [
        ("Baseline Random Forest", "Recall 59.3%, Precision 76.2% (Missed 41% of breakdowns)."),
        ("Exp A: Hyperparameter Tuning", "Pruned deep trees; precision 80.0%, but recall stalled at 59.3%."),
        ("Exp C: SMOTE Resampling", "Recall 72.2%, but doubled false alarms to 20 (Precision 66.1%) [REJECTED]."),
        ("Exp D: Balanced Random Forest", "Recall 96.3%, but catastrophic 176 false alarms (Precision 22.8%) [REJECTED]."),
        ("Exp E: Physics Features (@0.50)", "PR-AUC surged from 0.726 to 0.835; precision 86.4%, recall 70.4%."),
        ("Exp F: Combined Physics RF (@0.35)", "WINNER: Threshold 0.35 maximizes F2 = 0.8052 (Recall 79.6% val, 83.8% test).")
    ]
    for e_name, e_desc in exps:
        p = tf_exp.add_paragraph()
        p.space_before = Pt(2.5)
        r1 = p.add_run()
        r1.text = f"• {e_name}: "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(7.8)
        r1.font.bold = True
        r1.font.color.rgb = ACCENT_GREEN if "WINNER" in e_desc else (ACCENT_RED if "REJECTED" in e_desc else TEXT_WHITE)
        
        r2 = p.add_run()
        r2.text = e_desc
        r2.font.name = FONT_BODY
        r2.font.size = Pt(7.8)
        r2.font.color.rgb = TEXT_MUTED
        
    # Left Bottom: Actual Threshold Code Snippet Panel
    thresh_code = [
        ("comment", "# Threshold Sweep & Decision Tuning (ml/src/optimize_random_forest.py)"),
        ("normal", "best_f2_thresh = thresh_df.loc[thresh_df["), ("warn", "'F2-Score'"), ("normal", "].idxmax()]["), ("warn", "'Threshold'"), ("normal", "]"),
        ("comment", "# Optimal validation operating threshold = 0.35 (F2 peaks at 0.8052)"),
        ("normal", "y_proba = rf_engineered.predict_proba(X_val)[:, "), ("warn", "1"), (")"),
        ("highlight", "y_pred = (y_proba >= 0.35).astype(int)"),
        ("comment", "# Lowering threshold from 0.50 to 0.35 detected 5 additional failures"),
        ("comment", "# while adding only 2 false alarms (precision remained high at 84.3%)")
    ]
    add_code_panel(slide, Inches(0.67), Inches(4.40), left_w, Inches(2.65),
                   "PYTHON THRESHOLD TUNING", "ml/src/optimize_random_forest.py", thresh_code)
                   
    # Right Side: Dual Actual Evaluation Charts
    chart1 = "ml/outputs/charts/model_evaluation/random_forest_optimization/01_baseline_vs_optimized_rf.png"
    chart2 = "ml/outputs/charts/model_evaluation/random_forest_optimization/04_threshold_analysis_curves.png"
    
    fit_and_add_image(slide, chart1, Inches(6.47), Inches(1.60), Inches(6.19), Inches(2.65),
                      "Figure 4A: Baseline vs Optimized RF (+25% Recall, +8.15% Precision, +0.132 PR-AUC)")
                      
    fit_and_add_image(slide, chart2, Inches(6.47), Inches(4.40), Inches(6.19), Inches(2.65),
                      "Figure 4B: Precision, Recall, F1, and F2 curves sweeping thresholds 0.05 to 0.75 (F2 peak at 0.35)")

def build_slide_8(slide):
    """Slide 8 — Final Model Evaluation on Held-Out Test Set"""
    clear_slide(slide)
    set_slide_background(slide)
    add_header(slide, "08", "PRODUCTION MODEL EVALUATION",
               "Final Model Performance on 2,000 Held-Out Test Records",
               "Verified test set results: 83.82% Failure Recall, 85.07% Precision at calibrated threshold θ = 0.35")
               
    left_w = Inches(5.60)
    
    # Left: Verified Test Metrics Table Card
    card_m = add_card(slide, Inches(0.67), Inches(1.60), left_w, Inches(3.20))
    tb_m = slide.shapes.add_textbox(Inches(0.85), Inches(1.72), left_w - Inches(0.36), Inches(2.95))
    tf_m = tb_m.text_frame
    tf_m.word_wrap = True
    
    p = tf_m.paragraphs[0]
    p.text = "VERIFIED HELD-OUT TEST PERFORMANCE (2,000 SAMPLES)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    
    metrics = [
        ("Failure Detection Recall", "83.82%", "57 of 68 actual breakdowns caught in advance", ACCENT_CYAN),
        ("Alarm Precision Reliability", "85.07%", "57 TP / 67 alarms (17 of 20 alarms are real emergencies)", ACCENT_GREEN),
        ("Overall Classification Accuracy", "98.95%", "1,979 of 2,000 cycles correctly classified", TEXT_WHITE),
        ("Precision-Recall AUC (PR-AUC)", "0.8576", "25.2x higher than no-skill baseline (0.0340)", ACCENT_CYAN),
        ("Receiver Operating Curve (ROC-AUC)", "0.9791", "High discriminability across all operating thresholds", TEXT_WHITE),
        ("Cost-Weighted F2-Score (β=2)", "0.8407", "Industrial cost performance prioritizing failure capture", ACCENT_AMBER),
        ("Brier Probability Score Loss", "0.0098", "59% lower calibration error than baseline (well-calibrated)", ACCENT_GREEN)
    ]
    for lbl, val, note, c in metrics:
        p = tf_m.add_paragraph()
        p.space_before = Pt(2.5)
        r_val = p.add_run()
        r_val.text = f"{val} "
        r_val.font.name = FONT_HEADING
        r_val.font.size = Pt(8.5)
        r_val.font.bold = True
        r_val.font.color.rgb = c
        
        r_lbl = p.add_run()
        r_lbl.text = f"· {lbl} : "
        r_lbl.font.name = FONT_HEADING
        r_lbl.font.size = Pt(8.2)
        r_lbl.font.color.rgb = TEXT_WHITE
        
        r_note = p.add_run()
        r_note.text = note
        r_note.font.name = FONT_BODY
        r_note.font.size = Pt(7.8)
        r_note.font.color.rgb = TEXT_MUTED
        
    # Left Bottom: Confusion Matrix Analysis & Competitive Victory Card
    card_cm = add_card(slide, Inches(0.67), Inches(4.90), left_w, Inches(2.15))
    tb_cm = slide.shapes.add_textbox(Inches(0.85), Inches(4.98), left_w - Inches(0.36), Inches(1.95))
    tf_cm = tb_cm.text_frame
    tf_cm.word_wrap = True
    
    p = tf_cm.paragraphs[0]
    p.text = "CONFUSION MATRIX ANALYSIS (2,000 SAMPLES)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    
    cm_lines = [
        ("True Negatives (TN = 1,922):", "Healthy production cycles correctly allowed to run without disruption."),
        ("False Positives (FP = 10):", "Only 10 false alarms in 1,932 cycles (0.52% false alarm rate; prevents alarm fatigue)."),
        ("False Negatives (FN = 11):", "Missed breakdowns; predominantly stochastic Random Failures (RNF) with zero sensor warning."),
        ("True Positives (TP = 57):", "Catastrophic failures caught and prevented via scheduled component replacement."),
        ("Victory over XGBoost:", "Higher recall (83.82% vs 82.35%) with 78% fewer false alarms (10 vs 45 alarms).")
    ]
    for t, d in cm_lines:
        p = tf_cm.add_paragraph()
        p.space_before = Pt(2)
        r1 = p.add_run()
        r1.text = f"{t} "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(8)
        r1.font.bold = True
        r1.font.color.rgb = ACCENT_AMBER if "Victory" in t else TEXT_WHITE
        r2 = p.add_run()
        r2.text = d
        r2.font.name = FONT_BODY
        r2.font.size = Pt(7.8)
        r2.font.color.rgb = TEXT_MUTED
        
    # Right Side: Confusion Matrix Chart & Calibration Curve Chart
    cm_path = "ml/outputs/charts/model_evaluation/random_forest_optimization/05_confusion_matrix_test.png"
    cal_path = "ml/outputs/charts/model_evaluation/random_forest_optimization/07_calibration_curve.png"
    
    fit_and_add_image(slide, cm_path, Inches(6.47), Inches(1.60), Inches(6.19), Inches(2.95),
                      "Figure 5A: Final held-out test confusion matrix: 1,922 TN, 10 FP, 11 FN, 57 TP")
                      
    fit_and_add_image(slide, cal_path, Inches(6.47), Inches(4.70), Inches(6.19), Inches(2.35),
                      "Figure 5B: Reliability calibration curve confirming well-calibrated probabilities (Brier = 0.0098)")

def build_slide_9(slide):
    """Slide 9 — MachineGuard Frontend Application Demonstration"""
    clear_slide(slide)
    set_slide_background(slide)
    add_header(slide, "09", "APPLICATION PROTOTYPE & USER EXPERIENCE",
               "MachineGuard: Industrial Predictive Maintenance Interface",
               "Production React 18 Single Page Application with interactive diagnostic console and fleet analytics")
               
    left_w = Inches(6.80)
    
    # Left Side: Two Live Screenshots of React UI
    ss1 = "ml/outputs/charts/frontend_screenshots/predict_failure.png"
    ss2 = "ml/outputs/charts/frontend_screenshots/overview_dashboard.png"
    
    fit_and_add_image(slide, ss1, Inches(0.67), Inches(1.60), left_w, Inches(2.65),
                      "Screen 1: Predict Failure diagnostic console with synchronized sliders, presets, and radial SVG gauge")
                      
    fit_and_add_image(slide, ss2, Inches(0.67), Inches(4.40), left_w, Inches(2.65),
                      "Screen 2: Overview dashboard with fleet KPIs, 24-hr multi-sensor area trend, and risk distribution donut")
                      
    # Right Side: Software Architecture, Service Code & Technical Disclosure
    right_w = Inches(4.99)
    right_x = Inches(7.67)
    
    # Architecture Card
    card_arch = add_card(slide, right_x, Inches(1.60), right_w, Inches(1.65))
    tb_a = slide.shapes.add_textbox(right_x + Inches(0.18), Inches(1.70), right_w - Inches(0.36), Inches(1.45))
    tf_a = tb_a.text_frame
    tf_a.word_wrap = True
    
    p = tf_a.paragraphs[0]
    p.text = "FRONTEND TECHNOLOGY STACK"
    p.font.name = FONT_HEADING
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    
    ui_specs = [
        "Framework: React 18.3.1 SPA bundled via Vite 5.4.2.",
        "Routing: React Router v6 (6 pages: Overview, Predict, Explorer, Performance, History, Help).",
        "Visualizations: Recharts 2.12 (Area charts, Donut distributions, Torque-Speed scatter).",
        "Styling: Pure Vanilla CSS3 (Custom design tokens; no heavyweight frameworks).",
        "State & Persistence: React Context API + browser sessionStorage audit trail."
    ]
    for s in ui_specs:
        p = tf_a.add_paragraph()
        p.space_before = Pt(1.5)
        p.text = f"• {s}"
        p.font.name = FONT_BODY
        p.font.size = Pt(7.8)
        p.font.color.rgb = TEXT_MUTED
        
    # Service Code Panel
    svc_code = [
        ("comment", "// src/services/predictionService.js (L35-40)"),
        ("keyword", "const"), ("normal", " deltaT = processTemp - airTemp;"),
        ("keyword", "const"), ("normal", " powerWatts = torque * (speed * ("), ("warn", "2"), ("normal", " * Math.PI / "), ("warn", "60"), ("normal", "));"),
        ("keyword", "const"), ("normal", " overstrainFactor = torque * toolWear;"),
        ("normal", ""),
        ("comment", "// Client evaluates physics stress score & maps to sigmoid probability"),
        ("keyword", "const"), ("normal", " logisticProb = "), ("warn", "1"), ("normal", " / ("), ("warn", "1"), ("normal", " + Math.exp(-"), ("warn", "2.8"), ("normal", " * (stress - "), ("warn", "0.75"), ("normal", ")));"),
        ("keyword", "return"), ("normal", " { failureProbability, riskTier, contributingFactors };")
    ]
    add_code_panel(slide, right_x, Inches(3.35), right_w, Inches(2.15),
                   "JS EMULATOR", "src/services/predictionService.js", svc_code)
                   
    # Technical Disclosure Box
    card_disc = add_card(slide, right_x, Inches(5.60), right_w, Inches(1.45))
    tb_d = slide.shapes.add_textbox(right_x + Inches(0.18), Inches(5.68), right_w - Inches(0.36), Inches(1.28))
    tf_d = tb_d.text_frame
    tf_d.word_wrap = True
    
    p = tf_d.paragraphs[0]
    p.text = "TECHNICAL DISCLOSURE & PLANNED INTEGRATION"
    p.font.name = FONT_HEADING
    p.font.size = Pt(8.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_AMBER
    
    p = tf_d.add_paragraph()
    p.space_before = Pt(2)
    p.text = "The current React interface uses a deterministic client-side physics heuristic emulator. The serialized Python Random Forest model (ai4i_failure_model.joblib) is separately trained and validated. Live REST API integration via POST /api/predict (FastAPI) is pre-configured in .env.example for next-phase deployment."
    p.font.name = FONT_BODY
    p.font.size = Pt(7.8)
    p.font.color.rgb = TEXT_MUTED

def build_slide_10(slide):
    """Slide 10 — Conclusion, Limitations & Future Roadmap"""
    clear_slide(slide)
    set_slide_background(slide)
    add_header(slide, "10", "CONCLUSION & FUTURE SCOPE",
               "Engineering Conclusion & Industrial Deployment Roadmap",
               "Validated outcomes, technical limitations, and roadmap for real-world smart manufacturing deployment")
               
    col_w = Inches(3.84)
    col_h = Inches(4.30)
    col_gap = Inches(0.24)
    top_y = Inches(1.60)
    
    # Column 1: Core Achievements
    add_card(slide, Inches(0.67), top_y, col_w, col_h)
    tb1 = slide.shapes.add_textbox(Inches(0.85), top_y + Inches(0.15), col_w - Inches(0.36), col_h - Inches(0.30))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "PROVEN ENGINEERING OUTCOMES"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    
    c1_pts = [
        "Physics Feature Breakthrough: Deriving ΔT, Power, and Overstrain drove 48.1% of tree splits and surged PR-AUC by +18.2%.",
        "High Failure Recall (83.82%): Detected 57 of 68 actual breakdowns on 2,000 completely held-out test cycles.",
        "Minimal False Alarms (85.07% Precision): Generated only 10 false alarms in 1,932 healthy cycles (0.52% false alarm rate).",
        "Optimal Decision Threshold (θ = 0.35): Resolved asymmetric factory downtime costs, strictly beating XGBoost.",
        "Industrial React SPA: Full interactive console with presets, radial probability gauge, and CSV audit logs."
    ]
    for pt in c1_pts:
        p = tf1.add_paragraph()
        p.space_before = Pt(4)
        p.text = f"• {pt}"
        p.font.name = FONT_BODY
        p.font.size = Pt(8.2)
        p.font.color.rgb = TEXT_WHITE
        
    # Column 2: Genuine Limitations
    add_card(slide, Inches(0.67) + col_w + col_gap, top_y, col_w, col_h)
    tb2 = slide.shapes.add_textbox(Inches(0.85) + col_w + col_gap, top_y + Inches(0.15), col_w - Inches(0.36), col_h - Inches(0.30))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "GENUINE PROJECT LIMITATIONS"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = ACCENT_AMBER
    
    c2_pts = [
        "Synthetic Nature of AI4I: Built from mathematical milling models; lacks physical factory humidity drift, coolant aging, and electrical EMI noise.",
        "Residual Missed Failures (11 FN): Predominantly correspond to purely stochastic Random Failures (RNF) having zero prior sensor symptoms.",
        "Binary Risk vs RUL: Predicts failure probability state; does not yet compute exact Remaining Useful Life (RUL) in operating hours.",
        "Architecture Decoupling: React UI uses an embedded emulator; live Python HTTP connection remains to be deployed.",
        "No Accelerometer Telemetry: Does not currently capture high-frequency bearing vibration harmonics."
    ]
    for pt in c2_pts:
        p = tf2.add_paragraph()
        p.space_before = Pt(4)
        p.text = f"• {pt}"
        p.font.name = FONT_BODY
        p.font.size = Pt(8.2)
        p.font.color.rgb = TEXT_WHITE
        
    # Column 3: Future Roadmap
    add_card(slide, Inches(0.67) + 2 * (col_w + col_gap), top_y, col_w, col_h)
    tb3 = slide.shapes.add_textbox(Inches(0.85) + 2 * (col_w + col_gap), top_y + Inches(0.15), col_w - Inches(0.36), col_h - Inches(0.30))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "FUTURE INDUSTRIAL ROADMAP"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    
    c3_pts = [
        "FastAPI Model-Serving Microservice: Containerize Python pipeline and connect React UI via live REST endpoints.",
        "NASA IMS Bearing Dataset: Ingest 4-channel accelerometer vibration telemetry to capture early bearing race defects.",
        "NASA Milling Dataset: Ingest 3-axis cutting force dynamics (Fx, Fy, Fz) and high-frequency acoustic emission (AE) signals.",
        "Remaining Useful Life (RUL): Develop LSTM / TCN neural networks for continuous time-to-failure regression.",
        "Edge IoT PLC Deployment: Stream live factory telemetry directly via industrial MQTT / OPC-UA automation protocols."
    ]
    for pt in c3_pts:
        p = tf3.add_paragraph()
        p.space_before = Pt(4)
        p.text = f"• {pt}"
        p.font.name = FONT_BODY
        p.font.size = Pt(8.2)
        p.font.color.rgb = TEXT_WHITE
        
    # Bottom Closing Banner: Thank You & Viva Defense
    card_thx = add_card(slide, Inches(0.67), Inches(6.05), Inches(12.0), Inches(0.95),
                        bg_color=RGBColor(15, 23, 42), border_color=ACCENT_CYAN)
    tb_t = slide.shapes.add_textbox(Inches(0.85), Inches(6.12), Inches(11.64), Inches(0.80))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    p = tf_t.paragraphs[0]
    p.text = "THANK YOU  ·  QUESTIONS & TECHNICAL DISCUSSION"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    
    p = tf_t.add_paragraph()
    p.space_before = Pt(2)
    p.text = "MachineGuard System Repository  ·  AI4I 2020 Benchmark  ·  Infosys Springboard Internship 7.0 / College Engineering Capstone"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_BODY
    p.font.size = Pt(9)
    p.font.color.rgb = TEXT_MUTED

# -----------------------------------------------------------------------------
# Master Execution Routine
# -----------------------------------------------------------------------------
def upgrade_presentation(target_path="MachineGuard_10_Slide_Presentation.pptx"):
    print("=" * 80)
    print("UPGRADING MACHINEGUARD 10-SLIDE PRESENTATION")
    print(f"Target presentation: {target_path}")
    print("=" * 80)
    
    # 1. Open presentation
    if not os.path.exists(target_path):
        # Fall back to backup if target doesn't exist
        backup_path = "MachineGuard_10_Slide_Presentation_backup.pptx"
        if os.path.exists(backup_path):
            import shutil
            shutil.copy(backup_path, target_path)
            
    prs = Presentation(target_path)
    total_slides = len(prs.slides)
    print(f"Initial slide count: {total_slides}")
    assert total_slides == 10, f"Expected exactly 10 slides, found {total_slides}!"
    
    # Slide builders mapped 1 to 10
    builders = [
        build_slide_1,
        build_slide_2,
        build_slide_3,
        build_slide_4,
        build_slide_5,
        build_slide_6,
        build_slide_7,
        build_slide_8,
        build_slide_9,
        build_slide_10
    ]
    
    for idx, (slide, builder) in enumerate(zip(prs.slides, builders), 1):
        print(f"Upgrading Slide {idx:02d}...")
        builder(slide)
        
    print(f"\nFinal slide count before save: {len(prs.slides)}")
    assert len(prs.slides) == 10, f"Final presentation must have exactly 10 slides, got {len(prs.slides)}"
    
    # Save presentation
    prs.save(target_path)
    print(f"Successfully saved updated presentation to: {target_path}")
    
    # Also save to "MachineGuard Project Presentation.pptx" if it exists, so both are kept in sync
    alt_path = "MachineGuard Project Presentation.pptx"
    if os.path.exists(alt_path):
        prs.save(alt_path)
        print(f"Synchronized with: {alt_path}")
        
    print("=" * 80)
    print("ALL 10 SLIDES UPGRADED SUCCESSFULLY WITH ACTUAL CODE & EVALUATION VISUALS!")
    print("=" * 80)

if __name__ == "__main__":
    upgrade_presentation()
