/**
 * JobTrust - Background Service Worker
 * Handles message passing between content scripts and popup,
 * manages API calls, and coordinates analysis.
 */

const DEFAULT_API_URL = "http://localhost:8000";

// ===================================================================
// Message Handler
// ===================================================================
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "analyzeCurrentPage") {
    handleAnalyzeCurrentPage(sender.tab).then(sendResponse);
    return true; // Keep message channel open for async response
  }

  if (message.action === "analyzeJobData") {
    handleAnalyzeJobData(message.data).then(sendResponse);
    return true;
  }

  if (message.action === "checkApiHealth") {
    handleHealthCheck().then(sendResponse);
    return true;
  }
});

// ===================================================================
// Analyze current page by extracting data via content script
// ===================================================================
async function handleAnalyzeCurrentPage(tab) {
  try {
    if (!tab || !tab.id) {
      return { success: false, error: "No active tab found" };
    }

    // Extract job data from the page
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractJobData,
    });

    const jobData = results[0]?.result;
    if (!jobData || !jobData.title) {
      return { success: false, error: "Could not extract job data from this page" };
    }

    // Call prediction API
    const apiUrl = await getApiUrl();
    const response = await fetch(`${apiUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...jobData, use_llm: true }),
    });

    if (!response.ok) {
      return { success: false, error: `API error: ${response.status}` };
    }

    const prediction = await response.json();

    // Update badge icon based on risk
    updateBadgeIcon(tab.id, prediction.risk_level);

    return { success: true, data: prediction };
  } catch (e) {
    return { success: false, error: e.message };
  }
}

// ===================================================================
// Analyze provided job data
// ===================================================================
async function handleAnalyzeJobData(jobData) {
  try {
    const apiUrl = await getApiUrl();
    const response = await fetch(`${apiUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(jobData),
    });

    if (!response.ok) {
      return { success: false, error: `API error: ${response.status}` };
    }

    const prediction = await response.json();
    return { success: true, data: prediction };
  } catch (e) {
    return { success: false, error: e.message };
  }
}

// ===================================================================
// Health check
// ===================================================================
async function handleHealthCheck() {
  try {
    const apiUrl = await getApiUrl();
    const response = await fetch(`${apiUrl}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    const data = await response.json();
    return { success: true, data };
  } catch (e) {
    return { success: false, error: e.message };
  }
}

// ===================================================================
// Update extension badge based on risk level
// ===================================================================
function updateBadgeIcon(tabId, riskLevel) {
  const colors = {
    LOW: "#059669",
    MEDIUM: "#D97706",
    HIGH: "#DC2626",
  };

  const labels = {
    LOW: "OK",
    MEDIUM: "!",
    HIGH: "!!",
  };

  chrome.action.setBadgeBackgroundColor({
    tabId,
    color: colors[riskLevel] || "#0D9488",
  });

  chrome.action.setBadgeText({
    tabId,
    text: labels[riskLevel] || "",
  });
}

// ===================================================================
// Extract job data (injected into pages)
// ===================================================================
function extractJobData() {
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
  };

  // LinkedIn
  if (url.includes("linkedin.com")) {
    data.title =
      document.querySelector(".job-details-jobs-unified-top-card__job-title")?.textContent?.trim() ||
      document.querySelector(".jobs-unified-top-card__job-title")?.textContent?.trim() ||
      document.querySelector("h1")?.textContent?.trim() || "";

    data.company_profile =
      document.querySelector(".job-details-jobs-unified-top-card__company-name")?.textContent?.trim() ||
      document.querySelector(".jobs-unified-top-card__company-name")?.textContent?.trim() || "";

    data.location =
      document.querySelector(".job-details-jobs-unified-top-card__bullet")?.textContent?.trim() ||
      document.querySelector(".jobs-unified-top-card__bullet")?.textContent?.trim() || "";

    const descEl =
      document.querySelector(".jobs-description__content") ||
      document.querySelector(".jobs-description-content__text") ||
      document.querySelector("#job-details");
    data.description = descEl?.textContent?.trim() || "";
  }

  // Indeed
  else if (url.includes("indeed.com")) {
    data.title =
      document.querySelector(".jobsearch-JobInfoHeader-title")?.textContent?.trim() ||
      document.querySelector("h1")?.textContent?.trim() || "";

    data.company_profile =
      document.querySelector("[data-testid='inlineHeader-companyName']")?.textContent?.trim() || "";

    data.location =
      document.querySelector("[data-testid='inlineHeader-companyLocation']")?.textContent?.trim() || "";

    const descEl =
      document.querySelector("#jobDescriptionText") ||
      document.querySelector(".jobsearch-jobDescriptionText");
    data.description = descEl?.textContent?.trim() || "";
  }

  // Generic
  else {
    data.title = document.querySelector("h1")?.textContent?.trim() || document.title;
    const mainContent = document.querySelector("main") || document.querySelector("article") || document.body;
    data.description = mainContent?.textContent?.substring(0, 3000)?.trim() || "";
  }

  // Cleanup
  Object.keys(data).forEach((k) => {
    if (typeof data[k] === "string") {
      data[k] = data[k].replace(/\s+/g, " ").trim().substring(0, 5000);
    }
  });

  return data;
}

// ===================================================================
// Get API URL from storage
// ===================================================================
async function getApiUrl() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(["apiUrl"], (result) => {
      resolve(result.apiUrl || DEFAULT_API_URL);
    });
  });
}

// ===================================================================
// Extension install handler
// ===================================================================
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === "install") {
    // Set default settings
    chrome.storage.sync.set({
      apiUrl: DEFAULT_API_URL,
      autoScan: false,
      showBadge: true,
    });
  }
});
