import sys
from pptx import Presentation

sys.stdout.reconfigure(encoding='utf-8')
prs = Presentation('MachineGuard_10_Slide_Presentation_backup.pptx')

print(f"Presentation has {len(prs.slides)} slides.")
print(f"Dimensions: {prs.slide_width / 914400:.2f} x {prs.slide_height / 914400:.2f} inches (16:9 widescreen)\n")

for idx, slide in enumerate(prs.slides, 1):
    print(f"============================== SLIDE {idx} ==============================")
    for shape in slide.shapes:
        pos = f"pos=({shape.left/914400:.2f}, {shape.top/914400:.2f}), size=({shape.width/914400:.2f} x {shape.height/914400:.2f})"
        if shape.has_text_frame:
            text = ' | '.join(p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip())
            if text:
                print(f"  [Text] id={shape.shape_id} '{shape.name}' {pos} :: {text[:140]}")
        elif shape.has_table:
            print(f"  [Table] id={shape.shape_id} '{shape.name}' {pos}")
        elif shape.shape_type == 13: # picture
            print(f"  [Picture] id={shape.shape_id} '{shape.name}' {pos}")
        else:
            print(f"  [Shape {shape.shape_type}] id={shape.shape_id} '{shape.name}' {pos}")
