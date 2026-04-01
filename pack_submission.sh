#!/bin/bash
# Pack Stage 2 submission zip
set -e

ZIPNAME="stage2_submission.zip"
rm -f "$ZIPNAME"

echo "Packing submission..."

zip -r "$ZIPNAME" \
    README.md \
    requirements.txt \
    "Report Stage 2 Final.docx" \
    "Report Stage 2 Final.pdf" \
    configs/ \
    src/ \
    notebooks/stage2_experiments.ipynb \
    notebooks/stage2_final_summary.ipynb \
    figures/stage2/ \
    results/all_metrics_consolidated.json \
    -x "src/__pycache__/*" "src/**/__pycache__/*"

# Add individual metrics.json files preserving directory structure
find results -name "metrics.json" -not -path "*_archived*" | while read f; do
    zip "$ZIPNAME" "$f"
done

# Add meta.json files too (they have run config info)
find results -name "meta.json" -not -path "*_archived*" | while read f; do
    zip "$ZIPNAME" "$f"
done

echo ""
echo "=== Submission contents ==="
zipinfo -1 "$ZIPNAME" | head -80
echo "..."
echo ""
echo "Total files: $(zipinfo -1 "$ZIPNAME" | wc -l)"
echo "Zip size: $(du -h "$ZIPNAME" | cut -f1)"
echo ""
echo "Done: $ZIPNAME"
