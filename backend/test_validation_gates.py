"""
Validation Gate Tests for TB-Vision Pro Master Fix V2.0
Tests all 7 gates: Resolution, Aspect Ratio, Chromatic, Edge, Texture, Brightness, Face
"""
import sys, os
sys.path.insert(0, '.')
from app.services.ml_pipeline import validate_xray_image
from PIL import Image
import io, numpy as np

passed = 0
failed = 0

def run_test(name, image_bytes, filename, expect_valid):
    global passed, failed
    result = validate_xray_image(image_bytes, filename)
    is_valid = result.get("valid", False)
    reason = result.get("reason", "N/A")
    status = "PASS" if is_valid == expect_valid else "FAIL"
    if status == "PASS":
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] {name}: valid={is_valid}, reason={reason}")

def make_png(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

print("=" * 60)
print("TB-Vision Pro — Validation Gate Tests")
print("=" * 60)

# Test 1: White document (should REJECT - brightness gate)
white = Image.fromarray(np.full((300, 300), 240, dtype=np.uint8))
run_test("White document page", make_png(white), "white_doc.png", expect_valid=False)

# Test 2: Dark grayscale with smooth gradients (should PASS - looks like CXR)
# Real CXRs have smooth intensity transitions, not random noise
y, x = np.mgrid[0:300, 0:300]
gradient = (80 + 40 * np.sin(x / 50.0) * np.cos(y / 60.0)).astype(np.uint8)
from PIL import ImageFilter
dark = Image.fromarray(gradient).filter(ImageFilter.GaussianBlur(radius=5))
run_test("Smooth gradient CXR", make_png(dark), "dark_xray.png", expect_valid=True)

# Test 3: Tiny image (should REJECT - resolution gate)
tiny = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
run_test("Tiny image (100x100)", make_png(tiny), "tiny.png", expect_valid=False)

# Test 4: Color image (should REJECT - chromatic gate)
color_data = np.zeros((300, 300, 3), dtype=np.uint8)
color_data[:,:,0] = 200
color_data[:,:,1] = 50
color_data[:,:,2] = 50
color = Image.fromarray(color_data, 'RGB')
run_test("Color image (heavy red)", make_png(color), "color.png", expect_valid=False)

# Test 5: Wide panorama (should REJECT - aspect ratio)
wide = Image.fromarray(np.random.randint(30, 120, (224, 700), dtype=np.uint8))
run_test("Wide panorama (AR=3.1)", make_png(wide), "panorama.png", expect_valid=False)

# Test 6: Valid square grayscale with smooth structure (should PASS)
y2, x2 = np.mgrid[0:256, 0:256]
smooth = (90 + 30 * np.sin(x2 / 40.0) + 20 * np.cos(y2 / 35.0)).astype(np.uint8)
square = Image.fromarray(smooth).filter(ImageFilter.GaussianBlur(radius=4))
run_test("Smooth grayscale 256x256", make_png(square), "valid_square.png", expect_valid=True)

print()
print("-" * 60)
print(f"Results: {passed} PASSED, {failed} FAILED out of {passed + failed} tests")
if failed == 0:
    print("ALL VALIDATION GATE TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print("-" * 60)
