/**
 * JobTrust - Content Script
 * Injects a floating badge on supported job sites showing scan status.
 * Communicates with the background service worker.
 */

(() => {
  // Avoid double-injection
  if (window.__fakeJobDetectorInjected) return;
  window.__fakeJobDetectorInjected = true;

  // ===================================================================
  // Create floating badge
  // ===================================================================
  const badge = document.createElement("div");
  badge.id = "fjd-badge";
  badge.innerHTML = `
    <div id="fjd-badge-inner">
      <svg width="20" height="20" viewBox="0 0 28 28" fill="none">
        <rect width="28" height="28" rx="6" fill="#0D9488"/>
        <path d="M14 4C14 4 7 7 7 13C7 19 14 24 14 24C14 24 21 19 21 13C21 7 14 4 14 4Z" fill="white" fill-opacity="0.25" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M11 14L13 16L17 12" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <span id="fjd-badge-text">JobTrust</span>
    </div>
    <button id="fjd-badge-scan" title="Scan this job posting">Scan</button>
    <div id="fjd-badge-result" class="fjd-hidden"></div>
  `;
  document.body.appendChild(badge);

  // ===================================================================
  // Badge behavior
  // ===================================================================
  const scanBtn = badge.querySelector("#fjd-badge-scan");
  const badgeText = badge.querySelector("#fjd-badge-text");
  const badgeResult = badge.querySelector("#fjd-badge-result");

  scanBtn.addEventListener("click", async () => {
    scanBtn.disabled = true;
    scanBtn.textContent = "Scanning...";
    badgeText.textContent = "Analyzing...";

    try {
      const response = await chrome.runtime.sendMessage({
        action: "analyzeCurrentPage",
      });

      if (response && response.success) {
        const data = response.data;
        const pct = Math.round(data.fraud_probability * 100);
        const risk = data.risk_level.toLowerCase();

        const colors = { low: "#059669", medium: "#D97706", high: "#DC2626" };
        const labels = {
          low: "Likely Safe",
          medium: "Suspicious",
          high: "Likely Fake",
        };

        badgeText.textContent = `${labels[risk]} (${pct}%)`;
        badgeText.style.color = colors[risk];

        badge.style.borderColor = colors[risk];
        badge.querySelector("rect").setAttribute("fill", colors[risk]);

        // Show result details
        badgeResult.classList.remove("fjd-hidden");
        badgeResult.innerHTML = `
          <div class="fjd-result-row">
            <span>Fraud Risk:</span>
            <strong style="color:${colors[risk]}">${pct}%</strong>
          </div>
          <div class="fjd-result-row">
            <span>Method:</span>
            <strong>${data.method}</strong>
          </div>
          ${data.llm_result
            ? `<div class="fjd-result-row">
                   <span>LLM:</span>
                   <strong>${(data.llm_result.probability * 100).toFixed(0)}% (${data.llm_result.confidence})</strong>
                 </div>`
            : ""
          }
          <div class="fjd-explanation">${data.explanation}</div>
        `;
      } else {
        badgeText.textContent = response?.error || "Analysis failed";
        badgeText.style.color = "#ef4444";
      }
    } catch (e) {
      badgeText.textContent = "Error - Is API running?";
      badgeText.style.color = "#ef4444";
    } finally {
      scanBtn.disabled = false;
      scanBtn.textContent = "Re-scan";
    }
  });

  // Make badge draggable
  let isDragging = false;
  let dragOffset = { x: 0, y: 0 };

  badge.querySelector("#fjd-badge-inner").addEventListener("mousedown", (e) => {
    isDragging = true;
    dragOffset.x = e.clientX - badge.getBoundingClientRect().left;
    dragOffset.y = e.clientY - badge.getBoundingClientRect().top;
    badge.style.transition = "none";
  });

  document.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    const x = e.clientX - dragOffset.x;
    const y = e.clientY - dragOffset.y;
    badge.style.left = `${x}px`;
    badge.style.top = `${y}px`;
    badge.style.right = "auto";
    badge.style.bottom = "auto";
  });

  document.addEventListener("mouseup", () => {
    isDragging = false;
    badge.style.transition = "";
  });
})();
