"""
Batch Processor for Clinical Notes

Processes thousands of clinical notes for E&M code extraction.
Supports multiple input formats: TXT, CSV, JSON, Excel, PDF
Tracks progress, learns from data, generates reports.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from loguru import logger
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nlp.clinical_extractor import ClinicalNLPExtractor
from src.core.coder import MedicalCoder
from src.core.em_codes import PatientType
from src.core.em_mdm_calculator import MDMElements, ProblemComplexity, DataComplexity, RiskLevel


class BatchProcessor:
    """
    Process large batches of clinical notes for E&M coding.
    
    Features:
    - Multiple input formats (TXT, CSV, JSON, Excel, PDF)
    - Progress tracking with resume capability
    - Automatic learning from actual codes
    - Detailed reporting and analytics
    - Error handling and logging
    """
    
    def __init__(self, output_dir: str = "data/batch_results"):
        self.nlp_extractor = ClinicalNLPExtractor(learning_enabled=True)
        self.coder = MedicalCoder()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.processed_count = 0
        self.error_count = 0
        self.results = []
        
        logger.info("Batch Processor initialized")
    
    # ==================== MAIN PROCESSING ====================
    
    def process_directory(self, input_dir: str, file_pattern: str = "*.txt") -> Dict:
        """
        Process all files in a directory.
        
        Args:
            input_dir: Path to directory containing clinical notes
            file_pattern: File pattern to match (e.g., "*.txt", "*.pdf")
            
        Returns:
            Summary statistics
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        # Find all matching files
        files = list(input_path.glob(file_pattern))
        logger.info(f"Found {len(files)} files matching {file_pattern}")
        
        # Process with progress bar
        for file_path in tqdm(files, desc="Processing notes"):
            try:
                self._process_single_file(file_path)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                self.error_count += 1
        
        return self._generate_summary()
    
    def process_csv(self, csv_path: str, text_column: str = "clinical_text", 
                   id_column: str = "note_id", actual_code_column: Optional[str] = "actual_code") -> Dict:
        """
        Process clinical notes from CSV file.
        
        CSV Format:
        note_id,clinical_text,actual_code,patient_type
        001,"Patient presents with...",99213,established
        
        Args:
            csv_path: Path to CSV file
            text_column: Column name containing clinical text
            id_column: Column name for note ID
            actual_code_column: Column name for actual E&M code (optional)
        """
        logger.info(f"Processing CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        
        for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Processing CSV"):
            try:
                note_id = str(row[id_column]) if id_column in df.columns else f"note_{idx}"
                clinical_text = str(row[text_column])
                actual_code = str(row[actual_code_column]) if actual_code_column and actual_code_column in df.columns else None
                
                self._process_note(note_id, clinical_text, actual_code)
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                self.error_count += 1
        
        return self._generate_summary()
    
    def process_excel(self, excel_path: str, sheet_name: str = 0, 
                     text_column: str = "clinical_text", id_column: str = "note_id") -> Dict:
        """
        Process clinical notes from Excel file.
        
        Args:
            excel_path: Path to Excel file
            sheet_name: Sheet name or index
            text_column: Column name containing clinical text
            id_column: Column name for note ID
        """
        logger.info(f"Processing Excel: {excel_path}")
        
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        total_rows = len(df)
        
        for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Processing Excel"):
            try:
                note_id = str(row[id_column]) if id_column in df.columns else f"note_{idx}"
                clinical_text = str(row[text_column])
                actual_code = str(row['actual_code']) if 'actual_code' in df.columns else None
                
                self._process_note(note_id, clinical_text, actual_code)
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                self.error_count += 1
        
        return self._generate_summary()
    
    def process_json(self, json_path: str, is_jsonl: bool = False) -> Dict:
        """
        Process clinical notes from JSON/JSONL file.
        
        JSON Format:
        [
            {"note_id": "001", "clinical_text": "...", "actual_code": "99213"},
            {"note_id": "002", "clinical_text": "...", "actual_code": "99214"}
        ]
        
        JSONL Format (one JSON object per line):
        {"note_id": "001", "clinical_text": "...", "actual_code": "99213"}
        {"note_id": "002", "clinical_text": "...", "actual_code": "99214"}
        """
        logger.info(f"Processing JSON{'L' if is_jsonl else ''}: {json_path}")
        
        if is_jsonl:
            records = []
            with open(json_path, 'r', encoding='utf-8') as f:
                for line in f:
                    records.append(json.loads(line))
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                records = data if isinstance(data, list) else [data]
        
        for record in tqdm(records, desc="Processing JSON"):
            try:
                note_id = record.get('note_id', f"note_{self.processed_count}")
                clinical_text = record.get('clinical_text', '')
                actual_code = record.get('actual_code')
                
                self._process_note(note_id, clinical_text, actual_code)
                
            except Exception as e:
                logger.error(f"Error processing record: {e}")
                self.error_count += 1
        
        return self._generate_summary()
    
    # ==================== CORE PROCESSING ====================
    
    def _process_single_file(self, file_path: Path):
        """Process a single file (TXT or PDF)"""
        note_id = file_path.stem
        
        # Read file content
        if file_path.suffix.lower() == '.pdf':
            clinical_text = self._extract_pdf_text(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                clinical_text = f.read()
        
        self._process_note(note_id, clinical_text)
    
    def _process_note(self, note_id: str, clinical_text: str, actual_code: Optional[str] = None):
        """
        Process a single clinical note.
        
        Steps:
        1. Extract MDM elements using NLP
        2. Get E&M code suggestion
        3. Compare with actual code if provided
        4. Save for learning
        """
        # Extract using NLP
        extracted = self.nlp_extractor.extract_from_note(clinical_text, note_id)
        
        # Convert to patient type enum
        patient_type = PatientType.NEW if extracted['patient_type'] == 'new' else PatientType.ESTABLISHED
        
        # Build MDM elements
        from src.core.em_mdm_calculator import (
            determine_problem_complexity,
            determine_data_complexity,
            determine_risk_level
        )
        
        problem = determine_problem_complexity(
            num_problems=extracted['num_problems'],
            has_chronic_illness=extracted['has_chronic_illness'],
            chronic_illness_exacerbation=extracted['chronic_illness_exacerbation'],
            chronic_illness_severe=extracted['chronic_illness_severe'],
            life_threatening_condition=extracted['life_threatening_condition']
        )
        
        data = determine_data_complexity(
            reviewed_external_records=extracted['reviewed_external_records'],
            independent_interpretation=extracted['independent_interpretation'],
            discussed_with_external_physician=extracted['discussed_with_external_physician']
        )
        
        risk = determine_risk_level(
            prescription_drug_management=extracted['prescription_drug_management'],
            decision_for_surgery=extracted['decision_for_surgery'],
            drug_therapy_requiring_monitoring=extracted['drug_therapy_requiring_monitoring'],
            acute_threat_to_life=extracted['acute_threat_to_life']
        )
        
        mdm_elements = MDMElements(
            problem_complexity=problem,
            data_complexity=data,
            risk_level=risk
        )
        
        # Get E&M code suggestion
        try:
            result = self.coder.suggest_em_code(
                patient_type=patient_type,
                mdm_elements=mdm_elements,
                time_minutes=extracted['time_minutes']
            )
            
            suggested_code = result.selected_code.code
            confidence = result.confidence
            
        except Exception as e:
            logger.error(f"Error getting code for {note_id}: {e}")
            suggested_code = None
            confidence = "error"
        
        # Store result
        result_record = {
            "note_id": note_id,
            "extracted_elements": extracted,
            "suggested_code": suggested_code,
            "actual_code": actual_code,
            "match": suggested_code == actual_code if actual_code else None,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results.append(result_record)
        self.processed_count += 1
        
        # If actual code provided, save as correction for learning
        if actual_code and suggested_code != actual_code:
            # This is a mismatch - save for learning
            corrected = self._infer_corrections_from_code(extracted, actual_code)
            self.nlp_extractor.save_correction(
                note_id=note_id,
                clinical_text=clinical_text,
                extracted=extracted,
                corrected=corrected,
                actual_code=actual_code
            )
        
        # Auto-retrain every 100 notes
        if self.processed_count % 100 == 0:
            logger.info(f"Processed {self.processed_count} notes. Running auto-retrain...")
            self.nlp_extractor.retrain_from_corrections()
    
    def _infer_corrections_from_code(self, extracted: Dict, actual_code: str) -> Dict:
        """Infer what the corrected MDM elements should be based on actual code"""
        # This is a simple heuristic - in real use, you'd have the actual corrections
        # For now, just return the extracted values (will be improved with manual corrections)
        return extracted.copy()
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except ImportError:
            logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
            return ""
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    # ==================== REPORTING ====================
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics"""
        total = len(self.results)
        
        if total == 0:
            return {"error": "No notes processed"}
        
        # Calculate accuracy (if actual codes provided)
        results_with_actual = [r for r in self.results if r['actual_code']]
        matches = sum(1 for r in results_with_actual if r['match'])
        accuracy = matches / len(results_with_actual) if results_with_actual else None
        
        # Code distribution
        suggested_codes = [r['suggested_code'] for r in self.results if r['suggested_code']]
        code_counts = pd.Series(suggested_codes).value_counts().to_dict()
        
        summary = {
            "total_processed": total,
            "errors": self.error_count,
            "accuracy": accuracy,
            "code_distribution": code_counts,
            "processing_date": datetime.now().isoformat()
        }
        
        # Save summary
        summary_file = self.output_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        results_file = self.output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV for easy viewing
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = self.output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_file, index=False)
        
        logger.success(f"Summary saved to {summary_file}")
        
        return summary
    
    def print_summary(self, summary: Dict):
        """Print summary to console"""
        print("\n" + "="*70)
        print("BATCH PROCESSING SUMMARY")
        print("="*70)
        print(f"Total Processed: {summary['total_processed']}")
        print(f"Errors: {summary['errors']}")
        
        if summary.get('accuracy'):
            print(f"Accuracy: {summary['accuracy']*100:.1f}%")
        
        print("\nCode Distribution:")
        for code, count in sorted(summary.get('code_distribution', {}).items()):
            print(f"  {code}: {count} ({count/summary['total_processed']*100:.1f}%)")
        
        print("="*70)


def main():
    """CLI for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process clinical notes for E&M coding")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("--format", choices=['txt', 'csv', 'json', 'jsonl', 'excel', 'pdf'], 
                       default='txt', help="Input format")
    parser.add_argument("--pattern", default="*.txt", help="File pattern for directory processing")
    parser.add_argument("--output", default="data/batch_results", help="Output directory")
    
    args = parser.parse_args()
    
    processor = BatchProcessor(output_dir=args.output)
    
    print(f"\n🏥 Starting batch processing...")
    print(f"Input: {args.input}")
    print(f"Format: {args.format}\n")
    
    try:
        if args.format == 'txt' or args.format == 'pdf':
            summary = processor.process_directory(args.input, args.pattern)
        elif args.format == 'csv':
            summary = processor.process_csv(args.input)
        elif args.format == 'json':
            summary = processor.process_json(args.input, is_jsonl=False)
        elif args.format == 'jsonl':
            summary = processor.process_json(args.input, is_jsonl=True)
        elif args.format == 'excel':
            summary = processor.process_excel(args.input)
        
        processor.print_summary(summary)
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
