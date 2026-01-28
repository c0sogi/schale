"""Debug Exp sphere template matching."""
import logging

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from schale.scanner import scan_inventory

result = scan_inventory(
    r"C:\Users\cosogi\Downloads\제목 없음.png",
    confidence_threshold=0.03
)

print(f"\nResults: {len(result.items)} items found")
print("\nExp sphere items:")
for item in result.items:
    if 'exp' in item.icon_name.lower():
        print(f"  {item.icon_name} T{item.tier} x{item.quantity} at ({item.grid_position[0]}, {item.grid_position[1]}) - confidence: {item.confidence:.3f}")

print(f"\nUnrecognized cells: {len(result.unrecognized_cells)}")
print(f"Positions: {result.unrecognized_cells}")
