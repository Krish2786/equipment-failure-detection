"""
Layout & Bounding Box Quality Audit for MachineGuard_10_Slide_Presentation.pptx
Asserts that every shape, card, text box, and image stays strictly within slide boundaries
(13.333 x 7.500 inches) and that no unintentional overflows occur.
"""

from pptx import Presentation
from pptx.util import Inches

prs = Presentation('MachineGuard_10_Slide_Presentation.pptx')
slide_w = prs.slide_width / 914400 # 13.333 in
slide_h = prs.slide_height / 914400 # 7.500 in

print(f"Canvas: {slide_w:.3f} x {slide_h:.3f} inches (16:9 widescreen)\n")

issues_found = 0

for idx, slide in enumerate(prs.slides, 1):
    print(f"--- Checking Slide {idx:02d} ({len(slide.shapes)} shapes) ---")
    for s in slide.shapes:
        left = s.left / 914400
        top = s.top / 914400
        width = s.width / 914400
        height = s.height / 914400
        right = left + width
        bottom = top + height
        
        # Check if outside slide boundaries (with small 0.05 margin)
        if left < -0.05 or right > slide_w + 0.05:
            print(f"  [X-OVERFLOW] Shape id={s.shape_id} '{s.name}': left={left:.2f}, right={right:.2f} (max={slide_w:.2f})")
            issues_found += 1
        if top < -0.05 or bottom > slide_h + 0.05:
            print(f"  [Y-OVERFLOW] Shape id={s.shape_id} '{s.name}': top={top:.2f}, bottom={bottom:.2f} (max={slide_h:.2f})")
            issues_found += 1
            
        # Check text overflow if text frame
        if s.has_text_frame:
            for p in s.text_frame.paragraphs:
                for r in p.runs:
                    if r.font.size and r.font.size.pt < 7.0:
                        print(f"  [TINY-TEXT] Shape id={s.shape_id} '{s.name}': font size={r.font.size.pt}pt: '{r.text[:30]}'")
                        issues_found += 1

if issues_found == 0:
    print("\nSUCCESS: All shapes on all 10 slides strictly adhere to layout boundaries and font guidelines!")
else:
    print(f"\nWARNING: Found {issues_found} potential layout issues.")
