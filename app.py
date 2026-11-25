import io
import json
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction, molecular_weight

# --- NATIVE HARMONIZED DARK CLINICAL THEME ---
st._config.set_option("theme.base", "dark")
st._config.set_option("theme.primaryColor", "#38bdf8")
st._config.set_option("theme.backgroundColor", "#0b1120")
st._config.set_option("theme.secondaryBackgroundColor", "#131c31")
st._config.set_option("theme.textColor", "#f8fafc")

st.set_page_config(
    page_title="BioTrial Intelligence | RBQM Command Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- EXECUTIVE CLINICAL DARK SAAS CSS & GOOGLE FONTS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    /* Global Font & Deep Obsidian Canvas */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0b1120 !important;
        color: #f8fafc !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Crisp High-Contrast Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    p, label, span, li, div {
        color: #e2e8f0;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        color: #38bdf8 !important;
        background-color: #0f172a !important;
    }

    /* Executive Glass-Gradient Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0369a1 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        margin-bottom: 1.6rem;
        border: 1px solid rgba(56, 189, 248, 0.35);
        box-shadow: 0 10px 30px rgba(2, 132, 199, 0.22);
    }
    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 2rem !important;
        margin: 0 !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    .hero-banner p {
        color: #bae6fd !important;
        margin-top: 0.45rem !important;
        font-size: 1rem !important;
        font-weight: 500;
    }

    /* Sleek Elevated KPI Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #162038 0%, #111827 100%) !important;
        border: 1px solid #263554 !important;
        border-top: 4px solid #38bdf8 !important;
        padding: 1.15rem 1.3rem !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35) !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="stMetricValue"] div {
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 1.85rem !important;
    }

    /* Elevated Cards for Forms, Expanders & Containers */
    [data-testid="stForm"], [data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #131c31 !important;
        border: 1px solid #263554 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
    }

    /* Clean High-Contrast Form Inputs */
    input, textarea, select, div[data-baseweb="select"] > div {
        background-color: #0b1120 !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    input:focus, textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
    }

    /* Sidebar Navigation Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090d16 0%, #0f172a 100%) !important;
        border-right: 1px solid #1e293b !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #f1f5f9 !important;
    }

    /* Glowing Cyan-Blue Primary Action Buttons */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(90deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.2rem !important;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #38bdf8 0%, #3b82f6 100%) !important;
        color: #0b1120 !important;
        transform: translateY(-1px);
    }
    .stButton > button p, .stDownloadButton > button p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* High-Visibility Neon Risk Badges */
    .badge-high {
        background-color: rgba(239, 68, 68, 0.18);
        color: #fca5a5 !important;
        padding: 5px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid #ef4444;
    }
    .badge-med {
        background-color: rgba(245, 158, 11, 0.18);
        color: #fcd34d !important;
        padding: 5px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid #f59e0b;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.18);
        color: #6ee7b7 !important;
        padding: 5px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid #10b981;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PASSWORD = "Python"
DATA_FILE = "biotrial_data.json"
SEQ_FILE = "biotrial_sequences.json"
AUDIT_FILE = "biotrial_audit.json"
CAPA_FILE = "biotrial_capa.json"


# --- FILE & AUDIT HELPERS ---
def load(filename):
    try:
        with open(filename, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def log_audit(action, record_id, details):
    audit_log = load(AUDIT_FILE)
    audit_log.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user": st.session_state.get("user", "Unknown"),
        "action": action,
        "record_id": record_id,
        "details": details,
    })
    save(AUDIT_FILE, audit_log)


# --- CORE RBQM RISK ENGINE (ALL ORIGINAL + RBQM CHECKS) ---
def risk(p):
    score, reasons = 0, []
    base, latest = p["baseline"], p["latest"]
    change = (latest - base) / abs(base) * 100 if base else None

    checks = [
        (not p.get("consent_signed", True), 5, "CtQ BREACH: Informed consent missing"),
        (not p.get("eligible", True), 4, "CtQ BREACH: Eligibility criteria failed"),
        (p.get("status", "active") == "dropped", 2, "Participant discontinued/dropped out"),
        (change is not None and abs(change) >= 30, 3, "Biomarker changed >=30%"),
        (not p["bio_min"] <= latest <= p["bio_max"], 2, "Biomarker outside range"),
        (not p["lab_min"] <= p["lab"] <= p["lab_max"], 2, "Lab result outside range"),
        (len(p["events"]) >= 3, 2, "Three or more adverse events"),
        (sum(e["serious"] for e in p["events"]) > 0, 5, "Serious adverse event (SAE)"),
        (any(e.get("sae_delay_hours", 0) > 24 for e in p["events"]), 3, "SAE reported >24h late"),
        (len(p["deviations"]) >= 2, 2, "Two or more protocol deviations"),
        (len(p["deviations"]) == 1, 1, "One protocol deviation"),
        (len(p["missing"]) >= 2, 2, "Two or more missing fields"),
        (len(p["missing"]) == 1, 1, "One missing field"),
        (p.get("open_queries", 0) >= 3, 2, "3+ unresolved data queries"),
        (p["sample"] in ("pending", "damaged"), 2, "Sample pending/damaged"),
        (p["sample_delay"] >= 2, 2, "Sample delay >=2 days"),
        (p["sample_delay"] == 1, 1, "Sample delayed one day"),
        (p["entry_delay"] > 7, 2, "Data entry delay >7 days"),
        (3 < p["entry_delay"] <= 7, 1, "Data entry delay >3 days"),
    ]

    for condition, points, reason in checks:
        if condition:
            score += points
            reasons.append(reason)

    serious = any(e["serious"] for e in p["events"])
    ctq_breach = (not p.get("consent_signed", True)) or (not p.get("eligible", True))

    category = (
        "HIGH" if (serious or ctq_breach or score >= 7)
        else "MEDIUM" if score >= 3
        else "LOW"
    )
    needs_tsdv = (category == "HIGH") or (not p.get("sdv_verified", False))

    return {
        "score": score,
        "category": category,
        "change": change,
        "serious": serious,
        "needs_tsdv": needs_tsdv,
        "reasons": reasons or ["No configured risk flags"],
    }


# --- LOGIN GATE ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🧬 BioTrial Intelligence — RBQM & RBM Clinical Suite</h1>
            <p>ICH E6 (R2/R3) Risk-Based Quality Management • Centralized Statistical Monitoring • Molecular Lab</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_l, col_c, col_r = st.columns([1, 1.4, 1])
    with col_c:
        with st.form("login_form"):
            st.subheader("Authorized Monitor Login")
            user_name = st.text_input("Monitor / Researcher Name (for 21 CFR Part 11 Audit Trail)", placeholder="Enter your full name")
            pwd = st.text_input("System Access Password", type="password", placeholder="Enter authorized access password")
            submitted = st.form_submit_button("Access RBQM Command Center", use_container_width=True)
            if submitted:
                if pwd == PASSWORD and user_name.strip():
                    st.session_state.authenticated = True
                    st.session_state.user = user_name.strip()
                    log_audit("LOGIN", st.session_state.user, "Authenticated into Web Portal")
                    st.rerun()
                else:
                    st.error("Invalid password or missing researcher name.")
    st.stop()


# --- LOAD DATA ---
participants = load(DATA_FILE)
sequences = load(SEQ_FILE)
capas = load(CAPA_FILE)
audit_log = load(AUDIT_FILE)


# --- TOP HERO BANNER ---
st.markdown(
    f"""
    <div class="hero-banner">
        <h1>🧬 BioTrial Intelligence — RBQM / RBM Command Center</h1>
        <p>Active Monitor: <b style="color:#ffffff;">{st.session_state.user}</b>  |  Real-Time KRI, QTL, Fraud Detection & Biopython Analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("## 🧭 RBQM Modules")
page = st.sidebar.radio(
    "Select Workflow",
    [
        "📊 1. Executive Dashboard",
        "➕ 2. Enroll Subject (CtQ)",
        "📋 3. Subject Risk Report",
        "🏥 4. Site KRI & QTL Simulator",
        "🔬 5. CSM Stats & Fraud Check",
        "✏️ 6. Follow-up & tSDV",
        "🛠️ 7. CAPA Management",
        "📜 8. 21 CFR Part 11 Audit",
        "🧬 9. Biopython Sequence Lab",
    ],
)

st.sidebar.markdown("---")
if st.sidebar.button("Logout", use_container_width=True):
    log_audit("LOGOUT", st.session_state.user, "Logged out of Web Portal")
    st.session_state.authenticated = False
    st.rerun()


# --- PAGE 1: EXECUTIVE DASHBOARD ---
if page == "📊 1. Executive Dashboard":
    counts = Counter(risk(p)["category"] for p in participants)
    open_capas = sum(1 for c in capas if c["status"] == "open")
    tsdv_needed = sum(1 for p in participants if risk(p)["needs_tsdv"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Enrolled Subjects", len(participants))
    c2.metric("High / Med / Low Risk", f"{counts['HIGH']} / {counts['MEDIUM']} / {counts['LOW']}")
    c3.metric("tSDV Verification Queue", tsdv_needed)
    c4.metric("Open CAPAs", open_capas)

    st.markdown("")
    m1, m2, m3 = st.columns(3)
    m1.metric("Serious Adverse Events (SAEs)", sum(risk(p)["serious"] for p in participants))
    m2.metric("Pending / Damaged Samples", sum(p["sample"] in ("pending", "damaged") for p in participants))
    m3.metric("Unresolved Follow-Ups", sum(p["followup"] != "resolved" for p in participants))

    if participants:
        rows = []
        for p in participants:
            a = risk(p)
            rows.append({
                "id": p["id"],
                "site": p["site"],
                "status": p.get("status", "active"),
                "consent_signed": p.get("consent_signed", True),
                "eligible": p.get("eligible", True),
                "bio_name": p["bio_name"],
                "baseline": p["baseline"],
                "latest": p["latest"],
                "unit": p["unit"],
                "lab_name": p["lab_name"],
                "lab": p["lab"],
                "sample": p["sample"],
                "sample_delay": p["sample_delay"],
                "entry_delay": p["entry_delay"],
                "open_queries": p.get("open_queries", 0),
                "sdv_verified": p.get("sdv_verified", False),
                "followup": p["followup"],
                "risk_score": a["score"],
                "risk_category": a["category"],
                "needs_tsdv": a["needs_tsdv"],
                "percentage_change": round(a["change"], 2) if a["change"] is not None else "N/A",
                "risk_reasons": "; ".join(a["reasons"]),
            })
        df = pd.DataFrame(rows)

        st.markdown("### 📈 Visual Risk Telemetry")
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**Individual Subject Risk Scores**")
            st.bar_chart(df.set_index("id")["risk_score"], color="#38bdf8")
        with ch2:
            st.markdown("**Average Risk Score Across Clinical Sites**")
            st.bar_chart(df.groupby("site")["risk_score"].mean(), color="#10b981")

        st.markdown("### 🗂️ Live Subject Risk Register")
        display_df = df[["id", "site", "status", "risk_score", "risk_category", "needs_tsdv", "followup", "risk_reasons"]]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export Full RBQM Registry (CSV)",
            data=csv_buffer.getvalue(),
            file_name="biotrial_report.csv",
            mime="text/csv",
        )
    else:
        st.info("No subjects enrolled yet. Go to **'Enroll Subject (CtQ)'** in the left sidebar to register a participant.")


# --- PAGE 2: ENROLL SUBJECT (WITH MULTIPLE ADVERSE EVENTS) ---
elif page == "➕ 2. Enroll Subject (CtQ)":
    st.subheader("Enroll Participant & Capture Critical-to-Quality (CtQ) Data")

    num_aes = st.number_input("How many Adverse Events (AEs) to record for this subject?", min_value=0, max_value=10, value=0, step=1)

    with st.form("add_participant_form"):
        tab1, tab2, tab3 = st.tabs(["1️⃣ Identity & CtQ Gates", "2️⃣ Biomarkers & Labs", "3️⃣ Safety, Deviations & Logistics"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                pid = st.text_input("Participant ID *", placeholder="e.g., SUBJ-401")
                site = st.text_input("Hospital Site Code *", placeholder="e.g., Chennai-Med-01")
                status = st.selectbox("Enrollment Status", ["active", "completed", "dropped"])
            with col2:
                st.markdown("**Critical-to-Quality (CtQ) Protocol Checks**")
                consent = st.checkbox("Informed Consent Signed Prior to Procedures", value=True)
                eligible = st.checkbox("Meets All Inclusion / Exclusion Criteria", value=True)
                sdv_verified = st.checkbox("Critical Source Data Verified (tSDV)")

        with tab2:
            b1, b2 = st.columns(2)
            with b1:
                bio_name = st.text_input("Primary Biomarker Name *", placeholder="e.g., CRP")
                bio_min = st.number_input("Biomarker Normal Min", value=0.0)
                bio_max = st.number_input("Biomarker Normal Max", value=10.0)
                baseline = st.number_input("Baseline Value", value=5.0)
                latest = st.number_input("Latest Value", value=5.0)
                unit = st.text_input("Unit (optional)", placeholder="e.g., mg/L")
            with b2:
                lab_name = st.text_input("Safety Lab Test Name *", placeholder="e.g., ALT")
                lab_min = st.number_input("Lab Normal Min", value=7.0)
                lab_max = st.number_input("Lab Normal Max", value=56.0)
                lab_val = st.number_input("Latest Lab Result", value=25.0)

        with tab3:
            ae_inputs = []
            if num_aes > 0:
                st.markdown(f"**Adverse Event Details ({num_aes} event(s))**")
                for i in range(int(num_aes)):
                    st.markdown(f"*Event #{i + 1}*")
                    ec1, ec2, ec3, ec4 = st.columns([2, 2, 1, 1])
                    e_name = ec1.text_input(f"Event #{i + 1} Name", key=f"ae_n_{i}")
                    e_desc = ec2.text_input("Description", key=f"ae_d_{i}")
                    e_ser = ec3.checkbox("Serious (SAE)?", key=f"ae_s_{i}")
                    e_hrs = ec4.number_input("Report Delay (hrs)", min_value=0, step=1, key=f"ae_h_{i}")
                    e_date = st.text_input(f"Event #{i + 1} Date (optional)", placeholder="YYYY-MM-DD", key=f"ae_dt_{i}")
                    ae_inputs.append((e_name, e_desc, e_ser, e_hrs, e_date))
                st.markdown("---")

            s1, s2 = st.columns(2)
            with s1:
                sample = st.selectbox("Biological Sample Status", ["received", "pending", "damaged"])
                sample_delay = st.number_input("Sample Transit Delay (days)", min_value=0, step=1)
                entry_delay = st.number_input("CRF Data Entry Delay (days)", min_value=0, step=1)
                open_queries = st.number_input("Open Data Queries", min_value=0, step=1)
            with s2:
                devs_text = st.text_input("Protocol Deviations (comma-separated)")
                miss_text = st.text_input("Missing CRF Fields (comma-separated)")
                followup = st.selectbox("Follow-up Status", ["open", "under_review", "resolved"])
                notes = st.text_area("Follow-up Notes (optional)")

        if st.form_submit_button("💾 Save Subject & Calculate RBQM Score", use_container_width=True):
            if not pid or not site or not bio_name or not lab_name:
                st.error("Please complete all required (*) fields across tabs.")
            elif any(p["id"] == pid for p in participants):
                st.error("Participant ID already exists.")
            else:
                events = []
                for e_name, e_desc, e_ser, e_hrs, e_date in ae_inputs:
                    if e_name.strip():
                        events.append({
                            "name": e_name.strip(),
                            "description": e_desc.strip(),
                            "serious": e_ser,
                            "sae_delay_hours": e_hrs if e_ser else 0,
                            "date": e_date.strip(),
                        })

                new_p = {
                    "id": pid.strip(),
                    "site": site.strip(),
                    "consent_signed": consent,
                    "eligible": eligible,
                    "status": status,
                    "bio_name": bio_name.strip(),
                    "bio_min": bio_min,
                    "bio_max": max(bio_min, bio_max),
                    "baseline": baseline,
                    "latest": latest,
                    "unit": unit.strip(),
                    "lab_name": lab_name.strip(),
                    "lab_min": lab_min,
                    "lab_max": max(lab_min, lab_max),
                    "lab": lab_val,
                    "events": events,
                    "deviations": [x.strip() for x in devs_text.split(",") if x.strip()],
                    "missing": [x.strip() for x in miss_text.split(",") if x.strip()],
                    "open_queries": open_queries,
                    "sample": sample,
                    "sample_delay": sample_delay,
                    "entry_delay": entry_delay,
                    "sdv_verified": sdv_verified,
                    "followup": followup,
                    "notes": notes.strip(),
                    "created": datetime.now().isoformat(timespec="seconds"),
                }
                participants.append(new_p)
                save(DATA_FILE, participants)
                log_audit("ADD_PARTICIPANT", new_p["id"], f"Enrolled at site {new_p['site']}")
                st.success(f"Subject {new_p['id']} saved and recorded in the 21 CFR Part 11 audit trail!")


# --- PAGE 3: SUBJECT REPORT ---
elif page == "📋 3. Subject Risk Report":
    st.subheader("Individual Subject Risk & CtQ Inspection")
    if not participants:
        st.info("No subjects available.")
    else:
        pid = st.selectbox("Select Subject ID", [p["id"] for p in participants])
        p = next(x for x in participants if x["id"] == pid)
        a = risk(p)

        badge_class = "badge-high" if a["category"] == "HIGH" else "badge-med" if a["category"] == "MEDIUM" else "badge-low"
        st.markdown(
            f"### Subject `{p['id']}` ({p['site']})   <span class='{badge_class}'>{a['category']} RISK • SCORE {a['score']}</span>",
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Biomarker Shift", "N/A" if a["change"] is None else f"{round(a['change'], 2)}%")
        m2.metric("Latest Lab", f"{p['lab']} ({p['lab_name']})")
        m3.metric("tSDV Status", "Action Needed" if a["needs_tsdv"] else "Verified")
        m4.metric("Follow-Up", p["followup"].upper())

        with st.container(border=True):
            st.markdown("**Clinical & Operational Summary**")
            st.write(f"- **Biomarker:** {p['bio_name']} ({p['baseline']} → {p['latest']} {p['unit']})")
            st.write(f"- **Sample Status:** {p['sample']} (Delay: {p['sample_delay']} days) | **Data Entry Delay:** {p['entry_delay']} days")
            st.write(f"- **Adverse Events:** {p['events'] or 'None'}")
            st.write(f"- **Protocol Deviations:** {p['deviations'] or 'None'}")
            st.write(f"- **Missing Fields:** {p['missing'] or 'None'}")
            st.write(f"- **Notes:** {p['notes'] or 'None'}")

        st.markdown("**Triggered RBQM Risk Indicators:**")
        for r in a["reasons"]:
            st.write(f"- {r}")


# --- PAGE 4: SITE KRI & QTL SIMULATOR ---
elif page == "🏥 4. Site KRI & QTL Simulator":
    st.subheader("Site KRIs, White-Box QTL Simulator & Monitoring Memo Generator")

    with st.container(border=True):
        st.markdown("**🎛️ Interactive Quality Tolerance Limit (QTL) Threshold Simulator**")
        s1, s2, s3 = st.columns(3)
        qtl_dev = s1.slider("Max Site Deviation Rate (%)", 5, 100, 25) / 100.0
        qtl_drop = s2.slider("Max Site Dropout Rate (%)", 5, 100, 20) / 100.0
        qtl_high = s3.slider("Max High-Risk Subject Rate (%)", 5, 100, 30) / 100.0

    if not participants:
        st.info("No site records available.")
    else:
        sites = {}
        for p in participants:
            sites.setdefault(p["site"], []).append(p)

        for site, plist in sites.items():
            n = len(plist)
            results = [risk(p) for p in plist]
            scores = np.array([x["score"] for x in results])

            high_risk_count = sum(x["category"] == "HIGH" for x in results)
            dev_count = sum(len(p["deviations"]) > 0 for p in plist)
            dropout_count = sum(p.get("status") == "dropped" for p in plist)
            sae_count = sum(x["serious"] for x in results)
            tsdv_pending = sum(x["needs_tsdv"] for x in results)

            dev_rate = dev_count / n
            dropout_rate = dropout_count / n
            high_risk_rate = high_risk_count / n

            qtl_breaches = []
            if dev_rate > qtl_dev:
                qtl_breaches.append(f"Deviation Rate {dev_rate:.0%} > {qtl_dev:.0%}")
            if dropout_rate > qtl_drop:
                qtl_breaches.append(f"Dropout Rate {dropout_rate:.0%} > {qtl_drop:.0%}")
            if high_risk_rate > qtl_high:
                qtl_breaches.append(f"High-Risk Rate {high_risk_rate:.0%} > {qtl_high:.0%}")

            if qtl_breaches or sae_count > 0 or scores.mean() >= 6:
                visit_plan = "🚨 ON-SITE MONITORING VISIT REQUIRED"
            elif scores.mean() >= 3 or tsdv_pending > 0:
                visit_plan = "💻 REMOTE MONITORING + TARGETED SDV"
            else:
                visit_plan = "✅ CENTRALIZED MONITORING ONLY"

            with st.expander(f"🏥 {site}  |  {visit_plan}", expanded=True):
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Site Subjects", n)
                k2.metric("Mean Risk Score", round(float(scores.mean()), 2))
                k3.metric("SAEs Reported", sae_count)
                k4.metric("Pending tSDV", tsdv_pending)

                if qtl_breaches:
                    st.error("QTL BREACH DETECTED: " + " | ".join(qtl_breaches))
                else:
                    st.success("All Site KRIs are within Quality Tolerance Limits.")

                memo = (
                    f"OFFICIAL RBQM SITE MONITORING ACTION MEMO\n"
                    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                    f"Prepared By: {st.session_state.user}\n"
                    f"Clinical Site: {site}\n"
                    f"--------------------------------------------------\n"
                    f"Recommended Action: {visit_plan}\n"
                    f"Total Enrolled Subjects: {n}\n"
                    f"Mean Site Risk Score: {round(float(scores.mean()), 2)}\n"
                    f"High-Risk Subjects: {high_risk_count} ({high_risk_rate:.0%})\n"
                    f"Serious Adverse Events (SAEs): {sae_count}\n"
                    f"Subjects Requiring Targeted SDV: {tsdv_pending}\n"
                    f"QTL Breach Status: {', '.join(qtl_breaches) if qtl_breaches else 'None (Within Limits)'}\n"
                    f"--------------------------------------------------\n"
                    f"Generated by BioTrial Intelligence RBQM System.\n"
                )
                st.download_button(
                    label=f"📄 Download Official Monitoring Memo ({site})",
                    data=memo,
                    file_name=f"RBQM_Memo_{site}.txt",
                    mime="text/plain",
                    key=f"memo_{site}",
                )


# --- PAGE 5: CENTRALIZED STATS & FRAUD DETECTOR ---
elif page == "🔬 5. CSM Stats & Fraud Check":
    st.subheader("Centralized Statistical Monitoring (CSM) & Fraud Detection")

    st.markdown("#### 🕵️ 1. Terminal Digit Preference Fraud Detector")
    st.caption("Natural clinical measurements end in 0 or 5 roughly 20% of the time. Rates >45% indicate potential manual rounding or fabricated entries.")
    if participants:
        sites = {}
        for p in participants:
            sites.setdefault(p["site"], []).extend([p["baseline"], p["latest"], p["lab"]])

        fraud_rows = []
        for site, nums in sites.items():
            last_digits = [int(abs(round(val))) % 10 for val in nums]
            zero_five_count = sum(1 for d in last_digits if d in (0, 5))
            ratio = zero_five_count / len(last_digits)
            flag = "⚠️ SUSPICIOUS (Excess 0/5 Rounding)" if (len(last_digits) >= 6 and ratio > 0.45) else "✅ Normal Distribution"
            fraud_rows.append({
                "Hospital Site": site,
                "Values Checked": len(last_digits),
                "0 or 5 Ending Rate": f"{ratio:.0%}",
                "Data Integrity Verdict": flag,
            })
        st.dataframe(pd.DataFrame(fraud_rows), use_container_width=True, hide_index=True)

    st.markdown("#### 📐 2. NumPy Biomarker Cohort Statistics (IQR, Z-Scores & Correlation)")
    groups = {}
    for p in participants:
        groups.setdefault((p["bio_name"].lower(), p["unit"]), []).append(p)

    if not groups:
        st.info("No biomarker data available.")
    else:
        for (name, unit), group in groups.items():
            v = np.array([p["latest"] for p in group], dtype=float)
            with st.container(border=True):
                st.markdown(f"**Biomarker: {name.upper()} ({unit or 'unit not specified'})** — Sample Size `n={len(v)}`")
                st.write(f"**Mean:** `{round(float(np.mean(v)), 2)}` | **Median:** `{round(float(np.median(v)), 2)}` | **SD:** `{round(float(np.std(v)), 2)}`")
                q1, q3 = np.percentile(v, [25, 75])
                out = np.where((v < q1 - 1.5 * (q3 - q1)) | (v > q3 + 1.5 * (q3 - q1)))[0]
                st.write("**Potential IQR Outliers:**", [group[int(i)]["id"] for i in out] or "None")
                if np.std(v) > 0:
                    z_scores = {group[i]["id"]: round(float((x - v.mean()) / v.std()), 2) for i, x in enumerate(v)}
                    st.write("**Z-Scores:**", z_scores)
                b = np.array([p["baseline"] for p in group], dtype=float)
                if len(v) > 1 and np.std(v) and np.std(b):
                    st.write("**Baseline/Latest Correlation:**", round(float(np.corrcoef(b, v)[0, 1]), 3))


# --- PAGE 6: UPDATE FOLLOW-UP & tSDV ---
elif page == "✏️ 6. Follow-up & tSDV":
    st.subheader("Update Subject Follow-up & Targeted SDV Status")
    if not participants:
        st.info("No subjects available.")
    else:
        pid = st.selectbox("Select Subject ID", [p["id"] for p in participants])
        p = next(x for x in participants if x["id"] == pid)

        with st.form("update_form"):
            new_status = st.selectbox("Follow-up Resolution Status", ["open", "under_review", "resolved"], index=["open", "under_review", "resolved"].index(p["followup"]))
            new_sdv = st.checkbox("Critical Source Data Verified via tSDV", value=p.get("sdv_verified", False))
            new_queries = st.number_input("Remaining Open Queries", min_value=0, value=p.get("open_queries", 0))
            new_notes = st.text_area("Follow-up Note (optional)", value=p.get("notes", ""))

            if st.form_submit_button("Update & Log to Audit Trail"):
                old_status = p["followup"]
                p["followup"] = new_status
                p["sdv_verified"] = new_sdv
                p["open_queries"] = new_queries
                p["notes"] = new_notes
                save(DATA_FILE, participants)
                log_audit("UPDATE_FOLLOWUP", pid, f"Status {old_status} -> {new_status}, tSDV={new_sdv}")
                st.success("Updated successfully!")


# --- PAGE 7: CAPA MANAGEMENT ---
elif page == "🛠️ 7. CAPA Management":
    st.subheader("Corrective & Preventive Action (CAPA) Tracker")
    with st.form("capa_form"):
        c1, c2 = st.columns(2)
        site = c1.text_input("Hospital Site Code *")
        owner = c2.text_input("Responsible Investigator / CRA *")
        issue = st.text_input("Identified Issue / QTL Breach *")
        root_cause = st.text_input("Root Cause Analysis *")
        action_plan = st.text_input("Corrective & Preventive Action Plan *")
        if st.form_submit_button("Log New CAPA"):
            if site and issue and root_cause and action_plan and owner:
                cid = f"CAPA-{len(capas) + 1:03d}"
                capas.append({
                    "capa_id": cid,
                    "site": site,
                    "issue": issue,
                    "root_cause": root_cause,
                    "action_plan": action_plan,
                    "owner": owner,
                    "status": "open",
                    "created": datetime.now().isoformat(timespec="seconds"),
                })
                save(CAPA_FILE, capas)
                log_audit("CREATE_CAPA", cid, f"Site {site}: {issue}")
                st.success(f"{cid} created and logged!")
            else:
                st.error("All CAPA fields are required.")

    if capas:
        st.dataframe(pd.DataFrame(capas), use_container_width=True, hide_index=True)
        open_ids = [c["capa_id"] for c in capas if c["status"] == "open"]
        if open_ids:
            close_id = st.selectbox("Select Open CAPA to Close", open_ids)
            if st.button("Close Selected CAPA"):
                target = next(c for c in capas if c["capa_id"] == close_id)
                target["status"] = "closed"
                target["closed_date"] = datetime.now().isoformat(timespec="seconds")
                save(CAPA_FILE, capas)
                log_audit("CLOSE_CAPA", close_id, "CAPA verified and closed")
                st.rerun()


# --- PAGE 8: AUDIT TRAIL ---
elif page == "📜 8. 21 CFR Part 11 Audit":
    st.subheader("Immutable Electronic Audit Trail (21 CFR Part 11)")
    if audit_log:
        st.dataframe(pd.DataFrame(audit_log[::-1]), use_container_width=True, hide_index=True)
    else:
        st.info("No audit records yet.")


# --- PAGE 9: BIOPYTHON SEQUENCE LAB ---
elif page == "🧬 9. Biopython Sequence Lab":
    st.subheader("Biopython DNA & Protein Sequence Analyzer")
    with st.form("seq_form"):
        c1, c2 = st.columns(2)
        kind = c1.selectbox("Sequence Type", ["dna", "protein"])
        rid = c2.text_input("Unique Sequence ID *", placeholder="e.g., BRCA1-EXON2")
        raw_seq = st.text_area("Paste Sequence *", placeholder="ATGCGATCGATCG...")
        if st.form_submit_button("Analyze & Save Sequence"):
            seq = "".join(raw_seq.split()).upper()
            if not rid or not seq:
                st.error("Sequence ID and sequence are required.")
            elif any(s["id"] == rid for s in sequences):
                st.error("Sequence ID already exists.")
            else:
                record = {"id": rid, "type": kind, "sequence": seq}
                if kind == "dna":
                    if set(seq) - set("ACGTN"):
                        st.error("DNA can contain only A, C, G, T, N.")
                        st.stop()
                    obj = Seq(seq)
                    record["length"] = len(seq)
                    record["gc_percent"] = round(float(gc_fraction(obj) * 100), 2)
                    record["base_counts"] = {b: seq.count(b) for b in "ACGTN"}
                    if "N" not in seq:
                        record["molecular_weight_Da"] = round(float(molecular_weight(obj, seq_type="DNA")), 2)
                        if len(seq) % 3 == 0:
                            record["translation"] = str(obj.translate())
                else:
                    if set(seq) - set("ACDEFGHIKLMNPQRSTVWY"):
                        st.error("Use standard one-letter amino acid codes.")
                        st.stop()
                    obj = Seq(seq)
                    record["length"] = len(seq)
                    record["amino_acid_counts"] = dict(Counter(seq))
                    record["molecular_weight_Da"] = round(float(molecular_weight(obj, seq_type="protein")), 2)

                record["created"] = datetime.now().isoformat(timespec="seconds")
                sequences.append(record)
                save(SEQ_FILE, sequences)
                log_audit("ADD_SEQUENCE", rid, f"Analyzed {kind} sequence")
                st.success("Sequence analyzed and saved!")

                if kind == "dna":
                    st.write(f"**Length:** {record['length']} | **GC%:** {record['gc_percent']}% | **Bases:** {record['base_counts']}")
                    if "molecular_weight_Da" in record:
                        st.write(f"**Approx. molecular weight:** {record['molecular_weight_Da']} Da")
                    if "translation" in record:
                        st.write(f"**Translation:** `{record['translation']}` *(check reading frame)*")
                else:
                    st.write(f"**Length:** {record['length']} | **Composition:** {record['amino_acid_counts']}")
                    st.write(f"**Approx. molecular weight:** {record['molecular_weight_Da']} Da")

    if sequences:
        st.subheader("Saved Sequences")
        st.dataframe(pd.DataFrame(sequences), use_container_width=True, hide_index=True)