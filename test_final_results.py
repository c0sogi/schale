"""Final test of scanner with all improvements."""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from schale.scanner import scan_inventory

print("="*60)
print("FINAL SCANNER TEST - All Improvements Applied")
print("="*60)

result = scan_inventory(
    r"C:\Users\cosogi\Downloads\제목 없음.png",
    confidence_threshold=0.05  # Lower threshold to see all matches
)

print(f"\nRESULTS:")
print(f"   Grid: {result.grid_dimensions}")
print(f"   Total cells: {result.grid_dimensions[0] * result.grid_dimensions[1]}")
print(f"   Recognized: {len(result.items)}")
print(f"   Unrecognized: {len(result.unrecognized_cells)}")
print(f"   Recognition rate: {len(result.items) / (result.grid_dimensions[0] * result.grid_dimensions[1]) * 100:.1f}%")

print(f"\nITEMS BY CATEGORY:")
categories = {}
for item in result.items:
    cat = item.category
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(f"T{item.tier} x{item.quantity} ({item.confidence:.3f})")

for cat in sorted(categories.keys()):
    print(f"\n   {cat}:")
    for item_str in categories[cat]:
        print(f"      {item_str}")

if result.unrecognized_cells:
    print(f"\nUNRECOGNIZED CELLS: {result.unrecognized_cells}")

print("\n" + "="*60)
