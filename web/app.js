// Medical Coding Automation - Frontend JavaScript

const API_BASE = 'http://localhost:8000';


document.addEventListener('DOMContentLoaded', () => {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    const resultsSection = document.getElementById('results-section');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;

            // Hide all tab contents
            tabContents.forEach(tab => tab.classList.remove('active'));

            // Deactivate all buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));

            // Show selected tab
            document.getElementById(`${tabName}-tab`).classList.add('active');

            // Activate clicked button
            button.classList.add('active');

            // Hide results section
            if (resultsSection) resultsSection.style.display = 'none';
        });
    });
});

// Show/Hide Loading
function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'grid' : 'none';
}

// Display Results
function displayResults(data) {
    document.getElementById('result-code').textContent = data.code;
    document.getElementById('result-description').textContent = data.description;
    document.getElementById('result-patient-type').textContent =
        data.patient_type.charAt(0).toUpperCase() + data.patient_type.slice(1);
    document.getElementById('result-method').textContent = data.selection_method.toUpperCase();

    // MDM Level
    if (data.mdm_level) {
        document.getElementById('result-mdm-level').textContent =
            data.mdm_level.charAt(0).toUpperCase() + data.mdm_level.slice(1);
    } else {
        document.getElementById('result-mdm-level').textContent = 'N/A';
    }

    // Time
    if (data.time_minutes) {
        document.getElementById('result-time').textContent = `${data.time_minutes} minutes`;
        document.getElementById('result-time-row').style.display = 'flex';
    } else {
        document.getElementById('result-time-row').style.display = 'none';
    }

    // Confidence
    const confidenceEl = document.getElementById('result-confidence');
    confidenceEl.textContent = data.confidence.toUpperCase();
    confidenceEl.style.color = data.confidence.toLowerCase() === 'high' ? 'var(--primary)' :
        data.confidence.toLowerCase() === 'medium' ? 'var(--warning)' :
            'var(--muted)';

    // Rationale
    document.getElementById('result-rationale').textContent = data.rationale;

    // Show results section
    document.getElementById('results-section').style.display = 'block';

    // Scroll to results
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
}

// Error Handling
function showError(message) {
    alert(`Error: ${message}`);
}

// Quick Assessment Form Handler
document.getElementById('quick-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    showLoading(true);

    try {
        const patientType = document.querySelector('input[name="quick-patient-type"]:checked').value;
        const timeInput = document.getElementById('quick-time').value;

        const payload = {
            patient_type: patientType,
            num_problems: parseInt(document.getElementById('quick-num-problems').value),
            has_chronic_illness: document.getElementById('quick-chronic-illness').checked,
            chronic_illness_exacerbation: document.getElementById('quick-exacerbation').checked,
            chronic_illness_severe: document.getElementById('quick-severe-exacerbation').checked,
            life_threatening_condition: document.getElementById('quick-life-threatening').checked,
            reviewed_external_records: document.getElementById('quick-external-records').checked,
            independent_interpretation: document.getElementById('quick-interpretation').checked,
            discussed_with_external_physician: document.getElementById('quick-discussion').checked,
            prescription_drug_management: document.getElementById('quick-prescriptions').checked,
            decision_for_surgery: document.getElementById('quick-surgery').checked,
            drug_therapy_requiring_monitoring: document.getElementById('quick-monitoring').checked,
            acute_threat_to_life: document.getElementById('quick-acute-threat').checked,
            time_minutes: timeInput ? parseInt(timeInput) : null
        };

        const response = await fetch(`${API_BASE}/api/em/suggest-quick`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Request failed');
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
});

// Manual MDM Form Handler
document.getElementById('manual-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    showLoading(true);

    try {
        const patientType = document.querySelector('input[name="manual-patient-type"]:checked').value;
        const timeInput = document.getElementById('manual-time').value;

        const payload = {
            patient_type: patientType,
            problem_complexity: document.getElementById('manual-problem').value,
            data_complexity: document.getElementById('manual-data').value,
            risk_level: document.getElementById('manual-risk').value,
            time_minutes: timeInput ? parseInt(timeInput) : null
        };

        const response = await fetch(`${API_BASE}/api/em/suggest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Request failed');
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
});

// Time-Based Form Handler
document.getElementById('time-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    showLoading(true);

    try {
        const patientType = document.querySelector('input[name="time-patient-type"]:checked').value;

        const payload = {
            patient_type: patientType,
            time_minutes: parseInt(document.getElementById('time-minutes').value)
        };

        const response = await fetch(`${API_BASE}/api/em/suggest-time`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Request failed');
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
});

// File Upload Handler
document.addEventListener("submit", async (e) => {
    if (e.target.id !== "upload-form") return;

    e.preventDefault();
    showLoading(true);

    const fileInput = document.getElementById("note-file");
    if (!fileInput.files.length) {
        alert("Please upload a file first.");
        showLoading(false);
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {

        const token = localStorage.getItem('access_token')
        const res = await fetch(`${API_BASE}/api/em/upload-note`, {
            method: "POST",
            headers: {'Authorization':`Bearer ${token}`},
            body: formData
        });

        if (!res.ok) {
            const error = await res.json();
            throw new Error(error.detail || "Upload failed");
        }

        const data = await res.json();
        const extracted = data.extracted_elements;

        document.getElementById("upload-result").style.display = "block";
        document.getElementById("upload-details").innerHTML = `
            <p><strong>Filename:</strong> ${data.filename}</p>
            <p><strong>Note ID:</strong> ${data.note_id}</p>
            <p><strong>Confidence Score:</strong> ${data.confidence_score}</p>
            <h4>Extracted Elements:</h4>
            <ul>
                <li>Patient Type: ${extracted.patient_type}</li>
                <li>Number of Problems: ${extracted.num_problems}</li>
                <li>Has Chronic Illness: ${extracted.has_chronic_illness}</li>
                <li>Chronic Illness Exacerbation: ${extracted.chronic_illness_exacerbation}</li>
                <li>Chronic Illness Severe: ${extracted.chronic_illness_severe}</li>
                <li>Life Threatening Condition: ${extracted.life_threatening_condition}</li>
                <li>Reviewed External Records: ${extracted.reviewed_external_records}</li>
                <li>Independent Interpretation: ${extracted.independent_interpretation}</li>
                <li>Discussed With External Physician: ${extracted.discussed_with_external_physician}</li>
                <li>Prescription Drug Management: ${extracted.prescription_drug_management}</li>
                <li>Decision for Surgery: ${extracted.decision_for_surgery}</li>
                <li>Drug Therapy Requiring Monitoring: ${extracted.drug_therapy_requiring_monitoring}</li>
                <li>Acute Threat to Life: ${extracted.acute_threat_to_life}</li>
            </ul>
            <h4>ICD Codes:</h4>
            <pre>${extracted.icd_codes || []}</pre>
            <h4>CPT Codes:</h4>
            <pre>${extracted.cpt_codes || []}</pre>
        `;

        document.getElementById("upload-result")
            .scrollIntoView({ behavior: "smooth" });

    } catch (err) {
        showError(err.message);
    } finally {
        showLoading(false);
    }
});

document.addEventListener("change", (e) => {
    if (e.target.id !== "note-file") return;

    const fileName = e.target.files[0]?.name || "Choose file or drag here";
    document.getElementById("file-name").textContent = fileName;
});


document.addEventListener("DOMContentLoaded", () => {
    const logoutBtn = document.getElementById("logout-btn");

    const token = localStorage.getItem("access_token");
    if (!token) {
        window.location.href = "/";
        return;
    }

    // Logout handler
    if (logoutBtn) {
        logoutBtn.addEventListener("click", async () => {
            try {
                await fetch("/logout", {
                    method: "POST",
                    headers: { 'Authorization':`Bearer ${token}`},
                });
            } catch (err) {
                console.warn("Logout request failed:", err);
            } finally {
                // Always clear client-side auth
                localStorage.removeItem("access_token");
                window.location.href = "/";
            }
        });
    }
});


