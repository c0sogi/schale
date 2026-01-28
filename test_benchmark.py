import logging
import time

# Disable debug logging for clean output
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from schale.scanner import scan_inventory

# Run scan and measure time
start = time.time()
result = scan_inventory(
    r"C:\Users\cosogi\Downloads\제목 없음.png",
    confidence_threshold=0.03
)
elapsed = time.time() - start

print(f"\n{'='*60}")
print(f"BENCHMARK RESULTS")
print(f"{'='*60}")
print(f"Scan time: {elapsed:.2f} seconds")
print(f"Items found: {len(result.items)}")
print(f"Unrecognized: {len(result.unrecognized_cells)}")
print(f"Grid: {result.grid_dimensions}")
print(f"Source resolution: {result.source_resolution}")
print(f"{'='*60}")
