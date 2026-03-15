/**
 * JobTrust - Chrome Extension Popup
 * Handles UI interaction, API calls, and result rendering
 */

// ===================================================================
// Configuration
// ===================================================================
const DEFAULT_API_URL = "http://localhost:8000";
let API_URL = DEFAULT_API_URL;

// Load saved API URL
chrome.storage.sync.get(["apiUrl"], (result) => {
  if (result.apiUrl) API_URL = result.apiUrl;
});

// ===================================================================
// DOM Elements
// ===================================================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const elements = {
  // Status
  apiStatus: $("#apiStatus"),
  statusDot: $(".status-dot"),
  statusText: $(".status-text"),

  // Tabs
  tabs: $$(".tab"),
  tabContents: $$(".tab-content"),

  // Scan
  scanBtn: $("#scanBtn"),
  pageDetected: $("#pageDetected"),
  pageInfo: $("#pageInfo"),

  // Manual form
  manualForm: $("#manualForm"),

  // Loading
  loadingOverlay: $("#loadingOverlay"),
  loadingText: $("#loadingText"),

  // Results
  resultsPanel: $("#resultsPanel"),
  closeResults: $("#closeResults"),
  gaugeArc: $("#gaugeArc"),
  gaugePercent: $("#gaugePercent"),
  gaugeLabel: $("#gaugeLabel"),
  verdict: $("#verdict"),
  verdictIcon: $("#verdictIcon"),
  verdictText: $("#verdictText"),
  rawProb: $("#rawProb"),
  methodUsed: $("#methodUsed"),
  procTime: $("#procTime"),
  domainTrust: $("#domainTrust"),
  domainAge: $("#domainAge"),
  domainRegistrar: $("#domainRegistrar"),
  freeEmail: $("#freeEmail"),
  linkedinFound: $("#linkedinFound"),
  suspDomain: $("#suspDomain"),
  llmSection: $("#llmSection"),
  llmBadge: $("#llmBadge"),
  llmProb: $("#llmProb"),
  llmProvider: $("#llmProvider"),
  llmCached: $("#llmCached"),
  redFlagsList: $("#redFlagsList"),
  llmReasoning: $("#llmReasoning"),
  redFlagsSection: $("#redFlagsSection"),
  reasoningSection: $("#reasoningSection"),
  explanationText: $("#explanationText"),

  // History
  historyList: $("#historyList"),
  clearHistory: $("#clearHistory"),

  // Settings
  settingsBtn: $("#settingsBtn"),
};

// ===================================================================
// Initialization
// ===================================================================
document.addEventListener("DOMContentLoaded", () => {
  checkApiStatus();
  setupTabs();
  setupScanTab();
  setupManualForm();
  setupResults();
  setupHistory();
  setupSettings();
});

// ===================================================================
// API Health Check
// ===================================================================
async function checkApiStatus() {
  try {
    const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();

    elements.statusDot.classList.add("connected");
    elements.statusDot.classList.remove("error");

    let statusParts = ["API connected"];
    if (data.llm_available) {
      statusParts.push(`LLM: ${data.llm_provider}`);
    }
    statusParts.push(`Device: ${data.device}`);
    elements.statusText.textContent = statusParts.join(" | ");
  } catch (e) {
    elements.statusDot.classList.add("error");
    elements.statusDot.classList.remove("connected");
    elements.statusText.textContent = `API offline - start server at ${API_URL}`;
  }
}

// ===================================================================
// Tab Navigation
// ===================================================================
function setupTabs() {
  elements.tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const targetId = tab.dataset.tab;

      // Deactivate all
      elements.tabs.forEach((t) => t.classList.remove("active"));
      elements.tabContents.forEach((c) => c.classList.remove("active"));

      // Activate clicked
      tab.classList.add("active");
      $(`#tab-${targetId}`).classList.add("active");

      // Hide results when switching tabs
      elements.resultsPanel.classList.add("hidden");
    });
  });
}

// ===================================================================
// Scan Tab - Extract job data from current page
// ===================================================================
function setupScanTab() {
  // Detect current page
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tab = tabs[0];
    if (!tab || !tab.url) return;

    const url = tab.url;
    const supported = detectSupportedSite(url);

    if (supported) {
      elements.pageDetected.innerHTML = `
        <span class="site-name">${supported.name}</span>
        <p>Job posting detected. Click Scan to analyze.</p>
      `;
      elements.pageInfo.querySelector(".page-info-icon").style.color = "var(--success)";
      elements.scanBtn.disabled = false;
    } else {
      elements.pageDetected.innerHTML = `<p>Navigate to a job posting on LinkedIn, Indeed, Glassdoor, or other supported job sites to auto-detect job data.</p>`;
      elements.scanBtn.disabled = true;
    }
  });

  elements.scanBtn.addEventListener("click", handleScan);
}

function detectSupportedSite(url) {
  const sites = [
    { pattern: /linkedin\.com\/jobs/i, name: "LinkedIn Jobs" },
    { pattern: /indeed\.com/i, name: "Indeed" },
    { pattern: /glassdoor\.com\/job-listing/i, name: "Glassdoor" },
    { pattern: /monster\.com/i, name: "Monster" },
    { pattern: /ziprecruiter\.com\/jobs/i, name: "ZipRecruiter" },
    { pattern: /naukri\.com/i, name: "Naukri" },
  ];
  return sites.find((s) => s.pattern.test(url));
}

async function handleScan() {
  elements.scanBtn.disabled = true;
  showLoading("Extracting job data from page...");

  try {
    // Inject content script and extract data
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractJobDataFromPage,
    });

    const jobData = results[0]?.result;
    if (!jobData || !jobData.title) {
      throw new Error("Could not extract job data from this page. Try manual entry.");
    }

    // Send to API
    elements.loadingText.textContent = "Analyzing with AI model + LLM...";
    const prediction = await callPredictApi(jobData);
    showResults(prediction, jobData.title);
    saveToHistory(jobData.title, prediction);
  } catch (e) {
    hideLoading();
    alert(e.message || "Failed to scan page. Try manual entry.");
  } finally {
    elements.scanBtn.disabled = false;
  }
}

/**
 * Injected into the page to extract job posting data.
 * Supports LinkedIn, Indeed, Glassdoor, etc.
 */
function extractJobDataFromPage() {
  const url = window.location.href;
  let data = {
    title: "",
    description: "",
    company_profile: "",
    requirements: "",
    benefits: "",
    salary_range: "",
    location: "",
    employment_type: "",
    contact_email: "",
    source_url: url,
  };

  // ---- LinkedIn ----
  if (url.includes("linkedin.com")) {
    data.title =
      document.querySelector(".job-details-jobs-unified-top-card__job-title")?.textContent?.trim() ||
      document.querySelector(".jobs-unified-top-card__job-title")?.textContent?.trim() ||
      document.querySelector("h1")?.textContent?.trim() ||
      "";

    data.company_profile =
      document.querySelector(".job-details-jobs-unified-top-card__company-name")?.textContent?.trim() ||
      document.querySelector(".jobs-unified-top-card__company-name")?.textContent?.trim() ||
      "";

    data.location =
      document.querySelector(".job-details-jobs-unified-top-card__bullet")?.textContent?.trim() ||
      document.querySelector(".jobs-unified-top-card__bullet")?.textContent?.trim() ||
      "";

    const descEl =
      document.querySelector(".jobs-description__content") ||
      document.querySelector(".jobs-description-content__text") ||
      document.querySelector("#job-details");
    data.description = descEl?.textContent?.trim() || "";

    // Employment type from job insight
    const insightEls = document.querySelectorAll(".job-details-jobs-unified-top-card__job-insight span");
    insightEls.forEach((el) => {
      const text = el.textContent.trim();
      if (/full.time|part.time|contract|temporary|internship/i.test(text)) {
        data.employment_type = text;
      }
    });
  }

  // ---- Indeed ----
  else if (url.includes("indeed.com")) {
    data.title =
      document.querySelector(".jobsearch-JobInfoHeader-title")?.textContent?.trim() ||
      document.querySelector("h1[data-testid='jobsearch-JobInfoHeader-title']")?.textContent?.trim() ||
      document.querySelector("h1")?.textContent?.trim() ||
      "";

    data.company_profile =
      document.querySelector("[data-testid='inlineHeader-companyName']")?.textContent?.trim() ||
      document.querySelector(".jobsearch-InlineCompanyRating-companyHeader")?.textContent?.trim() ||
      "";

    data.location =
      document.querySelector("[data-testid='inlineHeader-companyLocation']")?.textContent?.trim() ||
      document.querySelector(".jobsearch-JobInfoHeader-subtitle div:last-child")?.textContent?.trim() ||
      "";

    const descEl =
      document.querySelector("#jobDescriptionText") ||
      document.querySelector(".jobsearch-jobDescriptionText");
    data.description = descEl?.textContent?.trim() || "";

    data.salary_range =
      document.querySelector("#salaryInfoAndJobType span")?.textContent?.trim() || "";

    data.employment_type =
      document.querySelector("#salaryInfoAndJobType .jobsearch-JobMetadataHeader-item:last-child")?.textContent?.trim() || "";
  }

  // ---- Glassdoor ----
  else if (url.includes("glassdoor.com")) {
    data.title = document.querySelector("[data-test='job-title']")?.textContent?.trim() || "";
    data.company_profile = document.querySelector("[data-test='employer-name']")?.textContent?.trim() || "";
    data.location = document.querySelector("[data-test='location']")?.textContent?.trim() || "";
    const descEl = document.querySelector(".jobDescriptionContent") || document.querySelector("[data-test='description']");
    data.description = descEl?.textContent?.trim() || "";
  }

  // ---- Generic fallback ----
  else {
    data.title =
      document.querySelector("h1")?.textContent?.trim() ||
      document.title.replace(/ - .+$/, "").trim();

    // Try to grab the largest text block on the page
    const allText = document.querySelector("main") || document.querySelector("article") || document.body;
    data.description = allText?.textContent?.substring(0, 3000)?.trim() || "";
  }

  // Clean up
  Object.keys(data).forEach((k) => {
    if (typeof data[k] === "string") {
      data[k] = data[k].replace(/\s+/g, " ").trim().substring(0, 5000);
    }
  });

  return data;
}

// ===================================================================
// Manual Form
// ===================================================================
function setupManualForm() {
  elements.manualForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const jobData = {
      title: $("#jobTitle").value.trim(),
      description: $("#jobDescription").value.trim() || null,
      company_profile: $("#jobCompany").value.trim() || null,
      requirements: $("#jobRequirements").value.trim() || null,
      salary_range: $("#jobSalary").value.trim() || null,
      location: $("#jobLocation").value.trim() || null,
      contact_email: $("#jobEmail").value.trim() || null,
      employment_type: $("#jobType").value || null,
      use_llm: $("#useLlm").checked,
      source_url: null,
    };

    if (!jobData.title) {
      alert("Please enter a job title.");
      return;
    }

    showLoading("Analyzing job posting...");
    try {
      const prediction = await callPredictApi(jobData);
      showResults(prediction, jobData.title);
      saveToHistory(jobData.title, prediction);
    } catch (e) {
      hideLoading();
      alert(e.message || "Failed to analyze. Is the API running?");
    }
  });
}

// ===================================================================
// API Call
// ===================================================================
async function callPredictApi(jobData) {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(jobData),
    signal: AbortSignal.timeout(60000), // 60s for LLM calls
  });

  if (!res.ok) {
    const errBody = await res.json().catch(() => null);
    throw new Error(errBody?.detail || `API error: ${res.status}`);
  }

  return await res.json();
}

// ===================================================================
// Results Display
// ===================================================================
function setupResults() {
  elements.closeResults.addEventListener("click", () => {
    elements.resultsPanel.classList.add("hidden");
  });
}

function showResults(data, title) {
  hideLoading();

  const prob = data.fraud_probability;
  const pct = Math.round(prob * 100);
  const risk = data.risk_level.toLowerCase();

  // -- Gauge animation --
  const circumference = 2 * Math.PI * 52; // r=52
  const offset = circumference * (1 - prob);
  elements.gaugeArc.style.transition = "stroke-dashoffset 1s ease, stroke 0.5s ease";
  elements.gaugeArc.setAttribute("stroke-dasharray", circumference);
  elements.gaugeArc.setAttribute("stroke-dashoffset", offset);

  // Gauge color
  const gaugeColors = { low: "#059669", medium: "#D97706", high: "#DC2626" };
  elements.gaugeArc.setAttribute("stroke", gaugeColors[risk] || "#0D9488");

  elements.gaugePercent.textContent = `${pct}%`;
  elements.gaugePercent.style.color = gaugeColors[risk];
  elements.gaugeLabel.textContent = "Fraud Risk";

  // -- Verdict --
  elements.verdict.className = `verdict ${risk}`;
  const verdicts = {
    low: { icon: "\u2705", text: "Likely Legitimate" },
    medium: { icon: "\u26A0\uFE0F", text: "Suspicious - Review Carefully" },
    high: { icon: "\uD83D\uDEA8", text: "High Risk - Likely Fake" },
  };
  const v = verdicts[risk] || verdicts.medium;
  elements.verdictIcon.textContent = v.icon;
  elements.verdictText.textContent = v.text;

  // -- Model details --
  elements.rawProb.textContent = `${(data.model_probability * 100).toFixed(1)}%`;
  elements.methodUsed.textContent = data.method;
  elements.procTime.textContent = `${data.processing_time_ms}ms`;

  // -- Graph features --
  const gf = data.graph_features;
  elements.domainTrust.textContent = `${(gf.domain_trust * 100).toFixed(0)}%`;

  // Domain age (from WHOIS)
  if (gf.domain_age_days >= 0) {
    const years = Math.floor(gf.domain_age_days / 365);
    const months = Math.floor((gf.domain_age_days % 365) / 30);
    if (years > 0) {
      elements.domainAge.textContent = `${years}y ${months}m`;
    } else if (months > 0) {
      elements.domainAge.textContent = `${months} months`;
    } else {
      elements.domainAge.textContent = `${Math.floor(gf.domain_age_days)}d`;
    }
    elements.domainAge.className = `stat-value ${gf.domain_age_days > 365 ? "text-success" : gf.domain_age_days > 90 ? "" : "text-danger"}`;
  } else {
    elements.domainAge.textContent = "N/A";
    elements.domainAge.className = "stat-value";
  }

  // Registrar (from WHOIS)
  const registrar = gf.registrar || "Unknown";
  elements.domainRegistrar.textContent = registrar;
  elements.domainRegistrar.title = registrar;

  elements.freeEmail.textContent = gf.email_free ? "Yes" : "No";
  elements.freeEmail.className = `stat-value ${gf.email_free ? "text-warning" : "text-success"}`;
  elements.linkedinFound.textContent = gf.has_linkedin ? "Yes" : "No";
  elements.linkedinFound.className = `stat-value ${gf.has_linkedin ? "text-success" : "text-warning"}`;
  elements.suspDomain.textContent = gf.suspicious_domain ? "Yes" : "No";
  elements.suspDomain.className = `stat-value ${gf.suspicious_domain ? "text-danger" : "text-success"}`;

  // -- LLM section --
  if (data.llm_result) {
    elements.llmSection.classList.remove("hidden");

    const llm = data.llm_result;
    elements.llmProb.textContent = `${(llm.probability * 100).toFixed(1)}%`;
    elements.llmProvider.textContent = llm.provider;
    elements.llmCached.textContent = llm.cached ? "Yes (instant)" : "No (live call)";

    // Confidence badge
    elements.llmBadge.textContent = llm.confidence.toUpperCase();
    elements.llmBadge.className = `badge ${llm.confidence}`;

    // Red flags
    if (llm.red_flags && llm.red_flags.length > 0) {
      elements.redFlagsSection.classList.remove("hidden");
      elements.redFlagsList.innerHTML = llm.red_flags
        .map((f) => `<li>${escapeHtml(f)}</li>`)
        .join("");
    } else {
      elements.redFlagsSection.classList.add("hidden");
    }

    // Reasoning
    if (llm.reasoning) {
      elements.reasoningSection.classList.remove("hidden");
      elements.llmReasoning.textContent = llm.reasoning;
    } else {
      elements.reasoningSection.classList.add("hidden");
    }
  } else {
    elements.llmSection.classList.add("hidden");
  }

  // -- Explanation --
  elements.explanationText.textContent = data.explanation;

  // Show panel, hide other content
  elements.tabContents.forEach((c) => c.classList.remove("active"));
  elements.tabs.forEach((t) => t.classList.remove("active"));
  elements.resultsPanel.classList.remove("hidden");
}

// ===================================================================
// Loading
// ===================================================================
function showLoading(text = "Analyzing...") {
  elements.loadingText.textContent = text;
  elements.loadingOverlay.classList.remove("hidden");
}

function hideLoading() {
  elements.loadingOverlay.classList.add("hidden");
}

// ===================================================================
// History
// ===================================================================
function setupHistory() {
  loadHistory();
  elements.clearHistory.addEventListener("click", () => {
    chrome.storage.local.set({ scanHistory: [] }, loadHistory);
  });
}

function saveToHistory(title, prediction) {
  chrome.storage.local.get(["scanHistory"], (result) => {
    const history = result.scanHistory || [];
    history.unshift({
      title: title.substring(0, 60),
      probability: prediction.fraud_probability,
      risk: prediction.risk_level,
      method: prediction.method,
      timestamp: Date.now(),
    });
    // Keep last 50
    chrome.storage.local.set({ scanHistory: history.slice(0, 50) }, loadHistory);
  });
}

function loadHistory() {
  chrome.storage.local.get(["scanHistory"], (result) => {
    const history = result.scanHistory || [];

    if (history.length === 0) {
      elements.historyList.innerHTML = `
        <div class="empty-state">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.4">
            <path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          <p>No analyses yet</p>
          <span>Your scan history will appear here</span>
        </div>`;
      elements.clearHistory.style.display = "none";
      return;
    }

    elements.clearHistory.style.display = "block";
    elements.historyList.innerHTML = history
      .map((item) => {
        const risk = item.risk.toLowerCase();
        const pct = Math.round(item.probability * 100);
        const time = formatRelativeTime(item.timestamp);
        const probColor =
          risk === "low" ? "text-success" : risk === "high" ? "text-danger" : "text-warning";

        return `
        <div class="history-item">
          <span class="history-risk ${risk}"></span>
          <div class="history-info">
            <div class="history-title">${escapeHtml(item.title)}</div>
            <div class="history-meta">${time} &bull; ${item.method}</div>
          </div>
          <span class="history-prob ${probColor}">${pct}%</span>
        </div>`;
      })
      .join("");
  });
}

// ===================================================================
// Settings
// ===================================================================
function setupSettings() {
  elements.settingsBtn.addEventListener("click", () => {
    if (chrome.runtime.openOptionsPage) {
      chrome.runtime.openOptionsPage();
    }
  });
}

// ===================================================================
// Utilities
// ===================================================================
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function formatRelativeTime(ts) {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}
