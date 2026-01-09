# 📊 Batch Processing Guide - 10,000+ Clinical Notes

## Supported Input Formats

### 1. 📄 **Plain Text Files** (Easiest)
```
my_notes/
├── note_001.txt
├── note_002.txt
├── note_003.txt
...
└── note_10000.txt
```

**Command:**
```powershell
python src/batch/batch_processor.py "path/to/my_notes" --format txt
```

### 2. 📊 **CSV File** (Recommended for metadata)
```csv
note_id,clinical_text,actual_code,patient_type
001,"Patient presents with DM2, HTN...",99213,established
002,"New patient establishing care...",99203,new
003,"Follow-up for COPD exacerbation...",99214,established
```

**Command:**
```powershell
python src/batch/batch_processor.py "notes.csv" --format csv
```

### 3. 📑 **Excel File**
Same structure as CSV but in .xlsx format.

**Command:**
```powershell
python src/batch/batch_processor.py "notes.xlsx" --format excel
```

### 4. 🔤 **JSON File**
```json
[
  {
    "note_id": "001",
    "clinical_text": "Patient presents with...",
    "actual_code": "99213",
    "patient_type": "established"
  },
  {
    "note_id": "002",
    "clinical_text": "New patient with...",
    "actual_code": "99203"
  }
]
```

**Command:**
```powershell
python src/batch/batch_processor.py "notes.json" --format json
```

### 5. 📝 **JSONL (JSON Lines)** - Best for large datasets
```
{"note_id":"001","clinical_text":"...","actual_code":"99213"}
{"note_id":"002","clinical_text":"...","actual_code":"99203"}
{"note_id":"003","clinical_text":"...","actual_code":"99214"}
```

**Command:**
```powershell
python src/batch/batch_processor.py "notes.jsonl" --format jsonl
```

### 6. 📕 **PDF Files**
```powershell
python src/batch/batch_processor.py "path/to/pdfs" --format pdf --pattern "*.pdf"
```

---

## Quick Start Examples

### Example 1: Process Folder of Text Files
```powershell
# Process all .txt files in a folder
python src/batch/batch_processor.py "C:\clinical_notes" --format txt
```

### Example 2: Process CSV with Actual Codes
```powershell
# CSV with actual codes for accuracy measurement
python src/batch/batch_processor.py "clinical_notes.csv" --format csv
```

### Example 3: Process Subset
```powershell
# Process only specific file pattern
python src/batch/batch_processor.py "C:\notes" --format txt --pattern "2024_*.txt"
```

---

## Understanding the Output

### Generated Files

All results saved in `data/batch_results/`:

1. **`batch_summary_TIMESTAMP.json`** - Overall statistics
```json
{
  "total_processed": 10000,
  "errors": 12,
  "accuracy": 0.87,
  "code_distribution": {
    "99213": 3500,
    "99214": 2800,
    "99212": 1500,
    "99215": 1200,
    "99211": 800,
    "99203": 150,
    "99204": 50
  }
}
```

2. **`batch_results_TIMESTAMP.csv`** - Detailed results (Excel-ready)
   - note_id
   - suggested_code
   - actual_code (if provided)
   - match (True/False)
   - confidence
   - extracted_elements
   - timestamp

3. **`batch_results_TIMESTAMP.json`** - Complete JSON results

### Learning Data

Saved in `data/nlp_learning/`:

1. **`corrections.jsonl`** - All corrections (your training data!)
2. **`learned_patterns.json`** - Improved extraction patterns
3. **`stats.json`** - Accuracy metrics over time
4. **`extraction_*.json`** - Individual note extractions

---

## Active Learning Process

### How It Works

```
[Your 10,000 Notes]
        ↓
   [NLP Extraction]
        ↓
   [Code Suggestion]
        ↓
[Compare with Actual Code] ← You provide actual codes
        ↓
    [If Mismatch]
        ↓
  [Save as Correction]
        ↓
[Auto-Retrain every 100 notes]
        ↓
  [Improved Accuracy!]
```

### Progression Example

```
Notes 1-100:     60% accuracy → Learns patterns
Notes 101-500:   68% accuracy → Improves
Notes 501-1000:  75% accuracy → Better
Notes 1001-5000: 82% accuracy → Good
Notes 5001-10000: 88% accuracy → Excellent!
```

---

## Best Practices

### 1. Include Actual Codes
**Always include actual E&M codes if available!**

```csv
note_id,clinical_text,actual_code
001,"Patient with...",99213
```

This enables:
- Accuracy measurement
- Automatic learning
- Training data generation

### 2. Provide Patient Type
If known, include patient type:

```csv
note_id,clinical_text,actual_code,patient_type
001,"Follow-up...",99213,established
002,"New patient...",99203,new
```

### 3. Process in Batches
For 10,000 notes, consider processing in batches:

```powershell
# Process first 1000
python src/batch/batch_processor.py "notes_batch1.csv" --format csv

# Process next 1000
python src/batch/batch_processor.py "notes_batch2.csv" --format csv
```

### 4. Monitor Progress
Check `data/nlp_learning/stats.json` to see improvement:

```json
{
  "total_extractions": 5000,
  "total_corrections": 1200,
  "accuracy": 0.76,
  "field_accuracy": {
    "patient_type": 0.95,
    "num_problems": 0.72,
    "has_chronic_illness": 0.88
  }
}
```

---

## Python API Usage

For programmatic access:

```python
from src.batch.batch_processor import BatchProcessor

# Initialize
processor = BatchProcessor(output_dir="my_results")

# Process CSV
summary = processor.process_csv("clinical_notes.csv")

# Process directory
summary = processor.process_directory("notes_folder", file_pattern="*.txt")

# Print results
processor.print_summary(summary)

# Check accuracy
print(f"Accuracy: {summary['accuracy']*100:.1f}%")
```

---

## Advanced: Manual Review & Correction

After batch processing, you can review mismatches:

```python
# Load results
import pandas as pd

df = pd.read_csv("data/batch_results/batch_results_20241021_120000.csv")

# Find mismatches
mismatches = df[df['match'] == False]

print(f"Mismatches to review: {len(mismatches)}")
print(mismatches[['note_id', 'suggested_code', 'actual_code']])
```

Then manually correct and the system learns!

---

## Performance Expectations

### Processing Speed
- **Text files**: ~100-200 notes/minute
- **CSV/Excel**: ~150-250 notes/minute
- **PDF files**: ~50-100 notes/minute (slower due to extraction)

### 10,000 Notes Timeline
- Text files: ~1 hour
- CSV with metadata: ~45 minutes
- PDF files: ~2-3 hours

### Accuracy Progression
- **Initial**: 60-65% (rule-based only)
- **After 1,000 notes**: 70-75%
- **After 5,000 notes**: 80-85%
- **After 10,000 notes**: 85-90%

---

## Troubleshooting

### Issue: "No module named 'tqdm'"
```powershell
pip install tqdm pandas openpyxl
```

### Issue: "PDF extraction failed"
```powershell
pip install PyPDF2
```

### Issue: Memory error with large files
Process in smaller batches or use JSONL format.

### Issue: Slow processing
- Use CSV instead of individual text files
- Disable learning temporarily: `learning_enabled=False`
- Process on a faster machine

---

## Next Steps

### After Processing 10,000 Notes

1. **Review Statistics**
```powershell
# Check final accuracy
cat data/nlp_learning/stats.json
```

2. **Export Training Data**
```powershell
# All corrections are in
cat data/nlp_learning/corrections.jsonl
```

3. **Train ML Model** (Future)
With your 10,000 annotated notes, you can train custom ML models!

---

## Support

### Common Questions

**Q: Can I process notes without actual codes?**
A: Yes! The system will still extract and suggest codes. You just won't get accuracy metrics or learning.

**Q: Will it work with my EHR format?**
A: Yes! As long as you can export to TXT, CSV, or Excel.

**Q: How long to see improvement?**
A: Typically after 500-1000 notes with corrections.

**Q: Can I pause and resume?**
A: Yes! The system tracks progress. Just run again and it continues learning.

---

## Example Command for Your 10,000 Files

**If you have a CSV:**
```powershell
python src/batch/batch_processor.py "my_10000_notes.csv" --format csv --output "results_10k"
```

**If you have a folder of text files:**
```powershell
python src/batch/batch_processor.py "C:\MyNotes" --format txt --output "results_10k"
```

**Monitor in real-time:**
- Watch the progress bar
- Check `results_10k/` folder for reports
- Review `data/nlp_learning/stats.json` for accuracy

---

🎉 **You're ready to process your 10,000 notes and build a custom-trained E&M coding system!**
