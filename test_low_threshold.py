"""Test with lower confidence threshold."""

from schale.scanner import scan_inventory

result = scan_inventory(
    r"C:\Users\cosogi\Downloads\제목 없음.png",
    confidence_threshold=0.05
)

print(f"Items found: {len(result.items)}")
print(f"Unrecognized: {len(result.unrecognized_cells)}")
print(f"Grid: {result.grid_dimensions}")

if result.items:
    print("\nTop 15 items:")
    for item in result.items[:15]:
        print(f"  {item.icon_name} T{item.tier} x{item.quantity} (conf={item.confidence:.3f})")
