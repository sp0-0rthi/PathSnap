# Data-quality record (partitions analysed: train + student_test)

* GPS rows inspected: 227577
* Rows with missing/unparseable cells: 0 (dropped)
* Out-of-range coordinates: 0 (dropped)
* Non-monotonic timestamps: 0 (sorted; duplicates collapsed)
* Trips containing >5 s sampling gaps: 47 of 90
* Largest single gap: 153064.0 s
* Arc rows referencing missing nodes: 0
* Route entries referencing unknown arcs: 0

Assumptions recorded:
1. The vehicle traverses .route arcs in chronological order; per-observation
   reference arcs are assigned by monotonic DP alignment (see src/labeling.py).
2. Reference routes are treated as reference information, not independently
   verified ground truth (dataset brief, section 6).
3. Arcs are straight segments between their endpoint nodes.
4. Distances use haversine / a local equirectangular projection in meters.
