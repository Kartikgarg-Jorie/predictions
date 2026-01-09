# 🏥 Medical Coding Automation - Web App Quick Start

## Start the Application

### Step 1: Install Dependencies (if not done)
```powershell
pip install -r requirements.txt
```

### Step 2: Start the Server
```powershell
python start_server.py
```

### Step 3: Open Your Browser
Navigate to: **http://localhost:8000**

That's it! 🎉

---

## What You'll See

### 🏠 Home Page
A professional dark-themed interface with 4 tabs:
- **📋 Quick Assessment** - Simple yes/no questions
- **⚙️ Manual MDM Entry** - Direct MDM level selection
- **⏱️ Time-Based** - Code selection by time spent
- **📄 Upload Note** - File upload (NLP coming soon)

### 💡 Quick Example

**Try Quick Assessment:**
1. Keep "Established Patient" selected
2. Set "Number of Problems" to 2
3. Check "Patient has chronic illness"
4. Check "Prescription drug management"
5. Enter "28" for time (optional)
6. Click "Get E&M Code"

**Expected Result:** Code 99213

---

## API Access

### Interactive Documentation
Open: **http://localhost:8000/docs**

Try the API directly from your browser with Swagger UI!

### API Examples

**Using curl:**
```powershell
curl -X POST "http://localhost:8000/api/em/suggest-quick" ^
  -H "Content-Type: application/json" ^
  -d "{\"patient_type\":\"established\",\"num_problems\":2,\"has_chronic_illness\":true,\"prescription_drug_management\":true,\"time_minutes\":28}"
```

**Using Python:**
```python
import requests

response = requests.post('http://localhost:8000/api/em/suggest-quick', json={
    'patient_type': 'established',
    'num_problems': 2,
    'has_chronic_illness': True,
    'prescription_drug_management': True,
    'time_minutes': 28
})

# print(response.json())
```

---

## Features at a Glance

✅ **CPT 2024 Guidelines** - Fully compliant  
✅ **4 Input Methods** - Choose what works best  
✅ **Real-Time Results** - Instant code suggestions  
✅ **Confidence Scoring** - Know how reliable the result is  
✅ **Detailed Rationale** - Understand why the code was selected  
✅ **Copy to Clipboard** - One-click code copying  
✅ **REST API** - Ready for EHR integration  
✅ **Professional UI** - Medical-grade dark theme  

---

## Troubleshooting

### Server won't start?
```powershell
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Use a different port
python start_server.py --port 8001
```

### Page won't load?
1. Make sure server is running (check console output)
2. Try: http://127.0.0.1:8000
3. Clear browser cache

### API errors?
- Check browser console (F12)
- Verify all required fields are filled
- Check server logs for details

---

## Next Steps

📖 **Full Documentation**: `docs/WEB_APP_GUIDE.md`  
🎯 **E&M Coding Guide**: `docs/EM_CODING_GUIDE.md`  
💻 **API Docs**: http://localhost:8000/docs  

---

## System Requirements

- **Python**: 3.9 or higher
- **Browser**: Chrome, Firefox, Edge, or Safari (latest)
- **RAM**: 2GB minimum
- **OS**: Windows, macOS, or Linux

---

## Support

Having issues? Check the documentation or review the API at /docs

**Happy Coding! 🎉**
