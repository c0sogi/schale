import logging

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from schale.scanner import scan_inventory

# Note: Using lower confidence threshold for smaller resolution images
result = scan_inventory(
    r"C:\Users\cosogi\Downloads\제목 없음.png",
    confidence_threshold=0.03  # Lowered from default 0.45 for small images
)

print(f"\nScan results: {len(result.items)} items found, {len(result.unrecognized_cells)} unrecognized")
print(f"Grid: {result.grid_dimensions}")
print(f"Source resolution: {result.source_resolution}\n")

for item in result.items:
    print(f"{item.icon_name} T{item.tier} x{item.quantity} (confidence: {item.confidence:.3f})")
