import logging
from schale.scanner import scan_inventory

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Note: Using lower confidence threshold for smaller resolution images
result = scan_inventory(
    r"C:\Users\cosogi\Downloads\제목 없음.png",
    confidence_threshold=0.45  # Back to default threshold
)

print(f"Scan results: {len(result.items)} items found, {len(result.unrecognized_cells)} unrecognized")
print(f"Grid: {result.grid_dimensions}\n")

# Group by icon_name to see duplicates
from collections import Counter
icon_counts = Counter(item.icon_name for item in result.items)

print("Item counts:")
for icon_name, count in icon_counts.most_common():
    if count > 1:
        print(f"  {icon_name}: {count} times")

print("\nAll items:")
for item in result.items:
    print(f"{item.icon_name} T{item.tier} x{item.quantity} at ({item.grid_position[0]}, {item.grid_position[1]}) (confidence: {item.confidence:.3f})")
