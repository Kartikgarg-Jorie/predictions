"""
Clinical Text Extractor - Rule-Based NLP with Active Learning

Extracts MDM elements from clinical notes using pattern matching.
Learns from user corrections to improve over time.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
from loguru import logger
from transformers import BertTokenizerFast, BertForSequenceClassification
import numpy as np
import torch
import os
import pandas as pd
import PyPDF2
import docx

DEVICE = torch.device("cpu")

def load_text_from_file(filepath):
    ext = filepath.lower().split(".")[-1]

    if ext == "txt":
        return open(filepath, "r", encoding="utf-8", errors="ignore").read()

    elif ext == "csv":
        df = pd.read_csv(filepath)
        return " ".join(df.astype(str).fillna("").values.flatten())

    elif ext == "pdf":
        
        text = ""
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + " "
        return text

    elif ext == "docx":
       
        doc = docx.Document(filepath)
        return " ".join([p.text for p in doc.paragraphs])
    
    else:
        raise ValueError("Unsupported file format")


def load_tokenizer(model_dir):
    try:
        
        return BertTokenizerFast.from_pretrained(model_dir)
    except:
        return BertTokenizerFast.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    



def chunk_text(text, tokenizer, max_length=512, stride=128):
    enc = tokenizer(
        text,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride
    )
    for ids in enc["input_ids"]:
        yield tokenizer.decode(ids, skip_special_tokens=True)


# Collapse ICD families (e.g., K21.*, R10.*, K59.* → 1 code of icd)
def collapse_families(codes):
    fam = defaultdict(list)
    for c in codes:
        fam[c[:3]].append(c)
    return [max(v, key=len) for v in fam.values()]

# Suppress background / low-information ICDs unless confident
BACKGROUND_PREFIXES = ("R79", "Z71", "R63", "Z80", "Z87", "Z78")

def suppress_background_codes(codes, probs, classes, min_prob=0.65):
    filtered = []
    for c in codes:
        idx = list(classes).index(c)
        if c.startswith(BACKGROUND_PREFIXES) and probs[idx] <= min_prob:
            continue
        filtered.append(c)
    return filtered

def predict_codes_from_text(model_dir, input_text, TOP_K=5):
    tokenizer = load_tokenizer(model_dir)
    classes = np.load(os.path.join(model_dir, "mlb_classes.npy"), allow_pickle=True)

    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()

    all_probs = []

    #probability of icd
    for chunk in chunk_text(input_text, tokenizer):
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        
        with torch.no_grad():
            logits = model(**enc).logits

        probs = torch.sigmoid(logits).cpu().numpy()[0]
        all_probs.append(probs)

    # Aggregate (MAX pooling)
    stacked = np.vstack(all_probs)
    probs = 0.7 * stacked.max(axis=0) + 0.3 * stacked.mean(axis=0)

    MIN_PROB = 0.35
    DELTA = 0.15
    max_prob = probs.max()

    top_idx = np.argsort(probs)[-TOP_K:]
    predicted_codes = [
        classes[i]
        for i in top_idx
        if probs[i] >= max(max_prob - DELTA, MIN_PROB)
    ]

    predicted_codes = collapse_families(predicted_codes)

    
    predicted_codes = suppress_background_codes(predicted_codes, probs, classes)

    return predicted_codes

class ClinicalNLPExtractor:
    """
    Rule-based clinical text extractor with active learning.

    Features:
    - Extracts patient type, problems, data reviewed, risk factors
    - Learns new patterns from user corrections
    - Builds training dataset automatically
    - Tracks accuracy metrics
    """

    def __init__(self, learning_enabled: bool = True):
        self.learning_enabled = learning_enabled
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "nlp_learning"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load learned patterns
        self.learned_patterns = self._load_learned_patterns()

        # Statistics tracking
        self.stats = self._load_stats()

        logger.info(
            f"Clinical NLP Extractor initialized (Learning: {learning_enabled})"
        )


    def extract_from_note(
        self, clinical_text: str, note_id: Optional[str] = None
    ) -> Dict:
        """
        Extract MDM elements from clinical note.

        Args:
            clinical_text: Raw clinical note text
            note_id: Optional unique identifier for this note

        Returns:
            Dictionary with extracted elements and confidence scores
        """
        if not note_id:
            note_id = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.debug(f"Extracting from note {note_id} ({len(clinical_text)} chars)")

        # Extract each component
        patient_type = self._extract_patient_type(clinical_text)
        num_problems = self._count_problems(clinical_text)
        chronic_illness = self._detect_chronic_illness(clinical_text)
        exacerbation = self._detect_exacerbation(clinical_text)
        severe_exacerbation = self._detect_severe_exacerbation(clinical_text)
        life_threatening = self._detect_life_threatening(clinical_text)
        external_records = self._detect_external_records(clinical_text)
        interpretation = self._detect_independent_interpretation(clinical_text)
        discussion = self._detect_physician_discussion(clinical_text)
        prescriptions = self._detect_prescriptions(clinical_text)
        surgery = self._detect_surgery_decision(clinical_text)
        monitoring = self._detect_intensive_monitoring(clinical_text)
        acute_threat = self._detect_acute_threat(clinical_text)
        time_minutes = self._extract_time(clinical_text)
        codes_and_meds = self._extract_codes_and_medications(clinical_text)

        # Calculate confidence
        confidence = self._calculate_confidence(
            clinical_text,
            {
                "patient_type": patient_type,
                "num_problems": num_problems,
                "chronic_illness": chronic_illness,
            },
        )

        result = {
            "note_id": note_id,
            "patient_type": patient_type,
            "num_problems": num_problems,
            "has_chronic_illness": chronic_illness,
            "chronic_illness_exacerbation": exacerbation,
            "chronic_illness_severe": severe_exacerbation,
            "life_threatening_condition": life_threatening,
            "reviewed_external_records": external_records,
            "independent_interpretation": interpretation,
            "discussed_with_external_physician": discussion,
            "prescription_drug_management": prescriptions,
            "decision_for_surgery": surgery,
            "drug_therapy_requiring_monitoring": monitoring,
            "acute_threat_to_life": acute_threat,
            "time_minutes": time_minutes,
            "icd_codes": codes_and_meds["icd_codes"],
            "cpt_codes": codes_and_meds["cpt_codes"],
            "medications": codes_and_meds["medications"],
            "confidence_score": confidence,
            "extraction_timestamp": datetime.now().isoformat(),
        }

        if self.learning_enabled:
            self._save_extraction(note_id, clinical_text, result)

        return result


    def _extract_patient_type(self, text: str) -> str:
        """Determine if new or established patient"""
        text_lower = text.lower()

        # Check learned patterns first
        for pattern in self.learned_patterns.get("new_patient", []):
            if re.search(pattern, text_lower):
                return "new"

        for pattern in self.learned_patterns.get("established_patient", []):
            if re.search(pattern, text_lower):
                return "established"

        # Default patterns
        new_patterns = [
            r"(?=(?:.*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})){1}$)\b",
            r"\bnew\s+patient\b",
            r"\bfirst\s+visit\b",
            r"\binitial\s+visit\b",
            r"\bnew\s+consult",
            r"\bestablishing\s+care\b",
        ]

        established_patterns = [
            
            r"\bestablished\s+patient\b",
            r"\bfollow[\s-]?up\b",
            r"\breturn\s+visit\b",
            r"\bseen\s+previously\b",
            r"\bcontinuing\s+care\b",
        ]


        for pattern in new_patterns:
            if re.search(pattern, text_lower):
                return "new"

        for pattern in established_patterns:
            if re.search(pattern, text_lower):
                return "established"
            
        return "established"


    def _count_problems(self, text: str) -> int:
        """Count medical problems addressed"""
        problems = set()

        # Method 1: Numbered list in assessment/plan
        numbered_problems = re.findall(
            r"^\s*\d+[\.\)]\s*([A-Z][^\n]+)", text, re.MULTILINE
        )
        problems.update([p.strip() for p in numbered_problems if len(p.strip()) > 3])

        # Method 2: Look for assessment section
        assessment_match = re.search(
            r"(?:assessment|impression|diagnos[ie]s?)[:\s]+(.*?)(?=plan|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if assessment_match:
            assessment_text = assessment_match.group(1)
            # Split by newlines or semicolons
            problem_lines = [
                line.strip()
                for line in re.split(r"[;\n]", assessment_text)
                if line.strip()
            ]
            problems.update([p for p in problem_lines if len(p) > 5 and len(p) < 100])

        # Method 3: ICD codes mentioned
        icd_codes = re.findall(r"\b[A-Z]\d{2}(?:\.\d{1,2})?\b", text)
        problems.update(icd_codes)

        count = len(problems)

        # Sanity check
        if count >= 10:
            count = 10  # Cap at 10 to avoid over-counting
        elif count == 0:
            count = 1  # At least one problem if it's a visit

        return count

    # ==================== PROBLEM CHARACTERISTICS ====================

    def _detect_chronic_illness(self, text: str) -> bool:
        """Detect chronic conditions"""
        chronic_conditions = {
            # Endocrine
            "diabetes",
            "dm",
            "dm2",
            "type 2 diabetes",
            "hypothyroid",
            "hyperthyroid",
            # Cardiovascular
            "hypertension",
            "htn",
            "cad",
            "coronary artery disease",
            "chf",
            "heart failure",
            "atrial fibrillation",
            "afib",
            # Respiratory
            "copd",
            "asthma",
            "emphysema",
            "chronic bronchitis",
            # Renal
            "ckd",
            "chronic kidney disease",
            "esrd",
            # Neurological
            "dementia",
            "alzheimer",
            "parkinson",
            "epilepsy",
            "seizure disorder",
            # Rheumatologic
            "arthritis",
            "lupus",
            "rheumatoid",
            # Oncologic
            "cancer",
            "malignancy",
            "carcinoma",
            # Other
            "cirrhosis",
            "hepatitis c",
            "hiv",
        }

        text_lower = text.lower()
        return any(condition in text_lower for condition in chronic_conditions)

    def _detect_exacerbation(self, text: str) -> bool:
        """Detect exacerbation of chronic illness"""
        exacerbation_keywords = {
            "exacerbation",
            "acute on chronic",
            "worsening",
            "flare",
            "uncontrolled",
            "poorly controlled",
            "decompensated",
            "unstable",
            "deteriorating",
        }

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in exacerbation_keywords)

    def _detect_severe_exacerbation(self, text: str) -> bool:
        """Detect severe exacerbation"""
        severe_keywords = {
            "severe exacerbation",
            "critical",
            "decompensated",
            "acute respiratory failure",
            "acute kidney injury",
            "diabetic ketoacidosis",
            "dka",
            "hypertensive emergency",
        }

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in severe_keywords)

    def _detect_life_threatening(self, text: str) -> bool:
        """Detect life-threatening conditions"""
        critical_keywords = {
            "life threatening",
            "critical",
            "emergent",
            "acute mi",
            "myocardial infarction",
            "stroke",
            "cva",
            "sepsis",
            "shock",
            "respiratory failure",
            "cardiac arrest",
            "unstable angina",
            "pulmonary embolism",
            "pe",
        }

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in critical_keywords)

    # ==================== DATA REVIEWED ====================

    def _detect_external_records(self, text: str) -> bool:
        """Detect review of external records"""
        patterns = [
            r"reviewed?\s+(?:external|outside|prior)\s+records?",
            r"reviewed?\s+(?:labs?|imaging|tests?)\s+from",
            r"obtained\s+records?\s+from",
            r"reviewed?\s+(?:previous|old)\s+(?:labs?|records?|studies)",
            r"reviewed?\s+(?:ekg|ecg|chest x-?ray|ct|mri)\s+from",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in patterns)

    def _detect_independent_interpretation(self, text: str) -> bool:
        """Detect independent interpretation of tests"""
        patterns = [
            r"(?:personally\s+)?reviewed?\s+(?:and\s+)?interpreted?",
            r"(?:my|personal)\s+interpretation",
            r"independently\s+reviewed?",
            r"i\s+(?:reviewed?|interpreted)",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in patterns)

    def _detect_physician_discussion(self, text: str) -> bool:
        """Detect discussion with external physician"""
        patterns = [
            r"discussed?\s+with\s+(?:dr|cardiolog|nephrolog|endocrinolog)",
            r"spoke\s+with\s+(?:specialist|consultant)",
            r"consultation\s+with",
            r"discussed?\s+case\s+with",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in patterns)

    # ==================== RISK ASSESSMENT ====================

    def _detect_prescriptions(self, text: str) -> bool:
        """Detect prescription drug management"""
        patterns = [
            r"(?:prescribed|started|initiated|added)\s+\w+",
            r"(?:adjusted|changed|increased|decreased)\s+(?:dose|dosage)",
            r"medication\s+(?:management|adjustment)",
            r"rx:",
            r"continue\s+(?:current\s+)?medications?",      
        ]
    
        # Common drug suffixes  
        drug_suffixes = ["pril", "olol", "pine", "statin", "formin", "ide"]

        text_lower = text.lower()
        has_patterns = any(re.search(pattern, text_lower) for pattern in patterns)
        has_drugs = any(suffix in text_lower for suffix in drug_suffixes)

        return has_patterns or has_drugs

    def _detect_surgery_decision(self, text: str) -> bool:
        """Detect surgical decision making"""
        patterns = [
            r"(?:discussed|considering|planned)\s+(?:surgery|surgical|operation)",
            r"surgical\s+(?:consult|referral)",
            r"(?:elective|scheduled)\s+(?:surgery|procedure)",
            r"decision\s+(?:for|regarding)\s+surgery",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in patterns)

    def _detect_intensive_monitoring(self, text: str) -> bool:
        """Detect drug therapy requiring intensive monitoring"""
        high_risk_drugs = {
            "warfarin",
            "coumadin",
            "chemotherapy",
            "immunosuppressant",
            "tacrolimus",
            "cyclosporine",
            "methotrexate",
            "lithium",
        }

        monitoring_keywords = {
            "intensive monitoring",
            "close monitoring",
            "frequent monitoring",
            "monitor closely",
            "inr monitoring",
            "therapeutic drug monitoring",
        }

        text_lower = text.lower()
        has_high_risk_drugs = any(drug in text_lower for drug in high_risk_drugs)
        has_monitoring = any(keyword in text_lower for keyword in monitoring_keywords)

        return has_high_risk_drugs or has_monitoring

    def _detect_acute_threat(self, text: str) -> bool:
        """Detect acute threat to life or bodily function"""
        # Similar to life_threatening but for risk context
        return self._detect_life_threatening(text)

    def _extract_time(self, text: str) -> Optional[int]:
        """Extract total time spent"""
        patterns = [
            r"total\s+time[:\s]+(\d+)\s*(?:min|minutes?)",
            r"time\s+spent[:\s]+(\d+)\s*(?:min|minutes?)",
            r"(\d+)\s*(?:min|minutes?)\s+(?:spent|total)",
            r"visit\s+time[:\s]+(\d+)",
            r"face[- ]to[- ]face\s+time[:\s]+(\d+)",
        ]

        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                minutes = int(match.group(1))
                if 1 <= minutes <= 300:  # Sanity check
                    return minutes

        return None

    #  ICD, CPT, MEDICATION EXTRACTION

    def _extract_codes_and_medications(self, text: str) -> Dict:
        # ICD from ML model
        icd_codes = predict_codes_from_text("ICD_MODEL", text)

        if icd_codes:
            np.random.shuffle(icd_codes) 
            icd_codes = icd_codes[:5]  

        # CPT from ML model
        cpt_codes = predict_codes_from_text("CPT_MODEL", text)

        # Medications with dose (regex stays)
        meds = re.findall(r"\b([A-Za-z]+)\s+(\d+(?:mg|mcg|g|ml))\b", text)
        medications_list = [{"drug": m[0], "dose": m[1]} for m in meds]

        return {
            "icd_codes": ",".join(icd_codes),
            "cpt_codes": ",".join(cpt_codes),
            "medications": medications_list,
        }

    def _calculate_confidence(self, text: str, extracted: Dict) -> float:
        """Calculate confidence score for extraction"""
        confidence = 0.0
        max_score = 0.0

        if "patient" in text.lower():
            confidence += 0.2
        max_score += 0.2

        if extracted["num_problems"] > 0:
            confidence += 0.3
        max_score += 0.3

        if extracted["chronic_illness"]:
            confidence += 0.2
        max_score += 0.2

        # Structure confidence (has sections)
        if any(
            section in text.lower() for section in ["assessment", "plan", "diagnosis"]
        ):
            confidence += 0.3
        max_score += 0.3

        return round(confidence / max_score if max_score > 0 else 0.5, 2)


    def save_correction(
        self,
        note_id: str,
        clinical_text: str,
        extracted: Dict,
        corrected: Dict,
        actual_code: str,
    ):
        """
        Save user correction for learning.

        This is called when user corrects the AI suggestion.
        Builds training dataset automatically.
        """
        if not self.learning_enabled:
            return

        correction_record = {
            "note_id": note_id,
            "timestamp": datetime.now().isoformat(),
            "extracted": extracted,
            "corrected": corrected,
            "actual_code": actual_code,
            "clinical_text": clinical_text[:500],  # Preview only
        }

        # Save to corrections file
        corrections_file = self.data_dir / "corrections.jsonl"
        with open(corrections_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(correction_record) + "\n")

        # Update statistics
        self.stats["total_corrections"] += 1
        self._analyze_correction(extracted, corrected)
        self._save_stats()

        logger.info(
            f"Saved correction for {note_id}. Total corrections: {self.stats['total_corrections']}"
        )

    def _analyze_correction(self, extracted: Dict, corrected: Dict):
        """Analyze what was corrected to improve patterns"""
        for key in corrected:
            if key in extracted and extracted[key] != corrected[key]:
                self.stats["field_corrections"][key] = (
                    self.stats["field_corrections"].get(key, 0) + 1
                )

    def get_accuracy_stats(self) -> Dict:
        """Get current accuracy statistics"""
        return {
            "total_extractions": self.stats["total_extractions"],
            "total_corrections": self.stats["total_corrections"],
            "accuracy": 1.0
            - (
                self.stats["total_corrections"]
                / max(self.stats["total_extractions"], 1)
            ),
            "field_accuracy": {
                field: 1.0 - (corrections / max(self.stats["total_extractions"], 1))
                for field, corrections in self.stats["field_corrections"].items()
            },
        }

    def retrain_from_corrections(self):
        """
        Retrain patterns from accumulated corrections.
        Called periodically (e.g., after 100 corrections).
        """
        corrections_file = self.data_dir / "corrections.jsonl"
        if not corrections_file.exists():
            logger.warning("No corrections file found for retraining")
            return

        logger.info("Retraining from corrections...")

        # Load all corrections
        corrections = []
        with open(corrections_file, "r", encoding="utf-8") as f:
            for line in f:
                corrections.append(json.loads(line))

        # Extract new patterns
        new_patterns = self._extract_patterns_from_corrections(corrections)

        # Update learned patterns
        self.learned_patterns.update(new_patterns)
        self._save_learned_patterns()

        logger.success(
            f"Retrained with {len(corrections)} corrections. Learned {len(new_patterns)} new patterns."
        )

    def _extract_patterns_from_corrections(self, corrections: List[Dict]) -> Dict:
        """Extract new patterns from corrections"""
        new_patterns = defaultdict(list)

        # Analyze patient type corrections
        for corr in corrections:
            if corr["extracted"]["patient_type"] != corr["corrected"]["patient_type"]:
                # Extract phrases that indicate patient type
                text = corr["clinical_text"].lower()
                # Simple pattern extraction (can be made more sophisticated)
                if corr["corrected"]["patient_type"] == "new":
                    new_patterns["new_patient"].append(text[:50])  # Store snippet

        return dict(new_patterns)

    # static content
    def _save_extraction(self, note_id: str, text: str, result: Dict):
        """Save extraction for potential training"""
        extraction_file = self.data_dir / f"extraction_{note_id}.json"
        with open(extraction_file, "w", encoding="utf-8") as f:
            json.dump(
                {"note_id": note_id, "text_preview": text[:500], "extracted": result},
                f,
                indent=2,
            )

        self.stats["total_extractions"] += 1

    def _load_learned_patterns(self) -> Dict:
        """Load previously learned patterns"""
        patterns_file = self.data_dir / "learned_patterns.json"
        if patterns_file.exists():
            with open(patterns_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_learned_patterns(self):
        """Save learned patterns"""
        patterns_file = self.data_dir / "learned_patterns.json"
        with open(patterns_file, "w", encoding="utf-8") as f:
            json.dump(self.learned_patterns, f, indent=2)

    def _load_stats(self) -> Dict:
        """Load statistics"""
        stats_file = self.data_dir / "stats.json"
        if stats_file.exists():
            with open(stats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"total_extractions": 0, "total_corrections": 0, "field_corrections": {}}

    def _save_stats(self):
        """Save statistics"""
        stats_file = self.data_dir / "stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)
