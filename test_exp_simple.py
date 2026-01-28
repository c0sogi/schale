"""Simple test to check if color detection is being called."""
import logging

# Set DEBUG level before importing scanner
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(message)s')

from schale.scanner import scan_inventory

result = scan_inventory(
    r"C:\Users\cosogi\Downloads\제목 없음.png",
    confidence_threshold=0.03
)

print(f"\nResults: {len(result.items)} items, {len(result.unrecognized_cells)} unrecognized")
print(f"Unrecognized cell positions: {result.unrecognized_cells}")
