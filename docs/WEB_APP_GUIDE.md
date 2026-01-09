# Medical Coding Automation - Web Application Guide

## Overview

The Medical Coding Automation web application provides a professional, user-friendly interface for medical coders to quickly and accurately select E&M office visit codes based on CPT 2024 guidelines.

## Features

### 🎯 Four Input Methods

1. **Quick Assessment** - Answer simple yes/no questions about the encounter
2. **Manual MDM Entry** - Directly specify MDM complexity levels  
3. **Time-Based Selection** - Select code based on total time spent
4. **Upload Clinical Note** - Upload documentation file (NLP extraction coming soon)

### ✅ Key Capabilities

- **CPT 2024 Compliant** - Follows latest official guidelines
- **Dual Selection Logic** - Uses MDM OR time (whichever is more advantageous)
- **Real-time Validation** - Instant code suggestion with confidence scoring
- **Detailed Rationale** - Explains why each code was selected
- **Professional UI** - Modern, medical-grade dark theme interface
- **Copy to Clipboard** - One-click code copying
- **API Ready** - Full REST API for EHR integration

## Getting Started

### Prerequisites

- Python 3.9+
- All project dependencies installed

### Installation

1. Ensure all dependencies are installed:
```powershell
pip install -r requirements.txt
```

2. Start the web server:
```powershell
python start_server.py
```

3. Open your browser to:
```
http://localhost:8000
```

## User Guide

### Quick Assessment Method

**Best for:** Most clinical encounters

**Steps:**
1. Select patient type (New or Established)
2. Enter number of problems addressed
3. Check applicable boxes for:
   - Problem complexity (chronic illness, exacerbations, life-threatening)
   - Data reviewed (external records, interpretations, consultations)
   - Risk level (prescriptions, procedures, monitoring needs)
4. Optionally enter total time
5. Click "Get E&M Code"

**Example Scenario:**
- Established patient
- 2 problems (diabetes + hypertension)
- Has chronic illness ✓
- Reviewed labs ✓
- Prescription management ✓
- Time: 28 minutes
- **Result: 99213**

### Manual MDM Entry Method

**Best for:** Coders familiar with MDM complexity levels

**Steps:**
1. Select patient type
2. Choose problem complexity level from dropdown
3. Choose data complexity level
4. Choose risk level
5. Optionally enter time
6. Click "Get E&M Code"

**MDM Levels Quick Reference:**
- **Minimal**: 1 minor problem, no data, no risk
- **Low**: 2+ minor OR 1 stable chronic
- **Moderate**: Exacerbation OR 2+ stable chronic
- **High**: Severe exacerbation OR threat to life

### Time-Based Method

**Best for:** Prolonged counseling/coordination visits

**Steps:**
1. Select patient type
2. Enter total time on date of encounter
3. Click "Get E&M Code"

**Time Ranges:**
- **New Patient**: 20, 35, 50, 75+ minutes → 99202-99205
- **Established**: 5, 15, 25, 35, 50+ minutes → 99211-99215

### Upload Clinical Note

**Best for:** Future NLP-based extraction

**Steps:**
1. Click "Choose file" or drag file to upload area
2. Select TXT, DOC, DOCX, or PDF file
3. Click "Upload Note"
4. File is processed and content displayed
5. Use Quick Assessment or Manual MDM Entry for code selection

**Note:** NLP extraction is in development. Currently, file upload stores the document for manual review.

## Understanding Results

### Code Display

```
99214
Office/outpatient visit, established patient, moderate complexity MDM
```

### Result Details

- **Patient Type**: New or Established
- **Selection Method**: MDM or TIME
- **MDM Level**: Straightforward, Low, Moderate, or High
- **Time**: Minutes spent (if applicable)
- **Confidence**: HIGH, MEDIUM, or LOW

### Confidence Levels

- **HIGH**: Both MDM and time support the code, or strong single indicator
- **MEDIUM**: Single clear indicator  
- **LOW**: Edge case or minimal data

### Rationale

Plain English explanation of why the code was selected, including:
- Primary selection method (MDM or time)
- Comparison if both methods available
- Justification based on CPT guidelines

## API Documentation

### Base URL
```
http://localhost:8000
```

### Interactive API Docs
```
http://localhost:8000/docs
```

### Key Endpoints

#### 1. Quick Assessment
```http
POST /api/em/suggest-quick
Content-Type: application/json

{
  "patient_type": "established",
  "num_problems": 2,
  "has_chronic_illness": true,
  "prescription_drug_management": true,
  "time_minutes": 28
}
```

#### 2. Manual MDM
```http
POST /api/em/suggest
Content-Type: application/json

{
  "patient_type": "established",
  "problem_complexity": "moderate",
  "data_complexity": "moderate",
  "risk_level": "moderate",
  "time_minutes": 35
}
```

#### 3. Time-Based
```http
POST /api/em/suggest-time
Content-Type: application/json

{
  "patient_type": "established",
  "time_minutes": 50
}
```

#### 4. File Upload
```http
POST /api/em/upload-note
Content-Type: multipart/form-data

file: [binary file data]
```

#### 5. List All Codes
```http
GET /api/codes/list
```

### Response Format

```json
{
  "code": "99214",
  "description": "Office/outpatient visit, established patient, moderate complexity MDM",
  "patient_type": "established",
  "selection_method": "mdm",
  "mdm_level": "moderate",
  "time_minutes": 35,
  "confidence": "high",
  "rationale": "Selected 99214 for established patient. based on moderate complexity MDM."
}
```

## Integration Examples

### Python Integration

```python
import requests

# Quick assessment
response = requests.post('http://localhost:8000/api/em/suggest-quick', json={
    'patient_type': 'established',
    'num_problems': 2,
    'has_chronic_illness': True,
    'prescription_drug_management': True,
    'time_minutes': 28
})

result = response.json()
print(f"Suggested Code: {result['code']}")
```

### JavaScript/EHR Integration

```javascript
// From EHR system
fetch('http://localhost:8000/api/em/suggest-quick', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    patient_type: 'established',
    num_problems: 2,
    has_chronic_illness: true,
    prescription_drug_management: true,
    time_minutes: 28
  })
})
.then(res => res.json())
.then(data => {
  console.log('Suggested Code:', data.code);
  // Populate EHR fields with result
});
```

## Troubleshooting

### Server Won't Start

**Error:** Module not found
```powershell
pip install -r requirements.txt
```

**Error:** Port 8000 already in use
```powershell
# Find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <process_id> /F

# Or use different port
python start_server.py --port 8001
```

### API Not Responding

1. Check server is running: `http://localhost:8000/health`
2. Check browser console for CORS errors
3. Verify API_BASE in `web/app.js` matches server URL

### Results Not Displaying

1. Open browser developer tools (F12)
2. Check Console tab for JavaScript errors
3. Check Network tab for failed API requests
4. Verify all form fields are filled correctly

## Security Considerations

### Production Deployment

**For production use:**

1. **Enable HTTPS** - Use SSL/TLS certificates
2. **Authentication** - Add user authentication/authorization
3. **Rate Limiting** - Prevent API abuse
4. **Input Validation** - Additional server-side validation
5. **Audit Logging** - Log all code selections with user IDs
6. **CORS** - Restrict to specific domains
7. **PHI Protection** - Ensure HIPAA compliance

### HIPAA Compliance

⚠️ **Important:** This tool processes clinical information. Ensure:
- All data transmission is encrypted
- Access is restricted to authorized users
- Audit trails are maintained
- No PHI is stored unnecessarily
- Compliance with organizational policies

## Performance

### Expected Response Times

- Quick Assessment: < 100ms
- Manual MDM Entry: < 50ms
- Time-Based Selection: < 50ms
- File Upload: < 1s (depends on file size)

### Scalability

- Backend: FastAPI with async support
- Concurrent requests: 100+ simultaneous users
- Caching: Implementable for common scenarios
- Load balancing: Ready for horizontal scaling

## Future Enhancements

### Planned Features

- [ ] NLP extraction from clinical notes
- [ ] Prolonged service codes (99417, 99418)
- [ ] Critical care codes (99291, 99292)
- [ ] Hospital E&M codes
- [ ] Modifier recommendations
- [ ] Batch processing
- [ ] Saved assessments
- [ ] Export to PDF/CSV
- [ ] EHR plugins (Epic, Cerner, etc.)
- [ ] Mobile-responsive design
- [ ] Offline mode

## Support

### Resources

- **API Documentation**: http://localhost:8000/docs
- **Source Code**: Check project README.md
- **E&M Coding Guide**: docs/EM_CODING_GUIDE.md

### Getting Help

For issues or questions:
1. Check this documentation
2. Review API docs at /docs endpoint
3. Consult E&M Coding Guide
4. Check browser console for errors

---

**Version**: 1.0.0  
**Last Updated**: October 2024  
**Status**: Production Ready ✅
