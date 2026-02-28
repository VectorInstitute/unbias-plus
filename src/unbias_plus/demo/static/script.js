/* ============================================================
   UnBias — Frontend Logic
   Handles: API call, highlighting, tooltip, breakdown cards
   ============================================================ */

   const inputEl      = document.getElementById("input-text");
   const charCountEl  = document.getElementById("char-count");
   const analyzeBtn   = document.getElementById("analyze-btn");
   const loadingEl    = document.getElementById("loading");
   const resultsEl    = document.getElementById("results");
   const highlightEl  = document.getElementById("highlighted-text");
   const unbiasedEl   = document.getElementById("unbiased-text");
   const segCountEl   = document.getElementById("segment-count");
   const pillsEl      = document.getElementById("severity-pills");
   const segListEl    = document.getElementById("segment-list");
   const noBiasEl     = document.getElementById("no-bias");
   const copyBtn      = document.getElementById("copy-btn");
   const tooltip      = document.getElementById("tooltip");

   const MAX_CHARS = 5000;

   // ============================================================
   // CHAR COUNTER
   // ============================================================

   inputEl.addEventListener("input", () => {
     const len = inputEl.value.length;
     charCountEl.textContent = `${len} / ${MAX_CHARS}`;
     charCountEl.className = "char-count";
     if (len > MAX_CHARS * 0.9) charCountEl.classList.add("warn");
     if (len >= MAX_CHARS)      charCountEl.classList.add("error");
   });

   // ============================================================
   // ANALYZE
   // ============================================================

   analyzeBtn.addEventListener("click", runAnalysis);

   inputEl.addEventListener("keydown", (e) => {
     if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runAnalysis();
   });

   async function runAnalysis() {
     const text = inputEl.value.trim();
     if (!text) return;
     if (text.length > MAX_CHARS) {
       alert("Text too long. Please keep it under 5000 characters.");
       return;
     }

     analyzeBtn.disabled = true;
     resultsEl.classList.add("hidden");
     loadingEl.classList.remove("hidden");

     try {
       const res = await fetch("/analyze", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ text }),
       });

       const data = await res.json();

       if (!res.ok || data.detail) {
         throw new Error(data.detail || "Server error");
       }

       renderResults(data);

     } catch (err) {
       alert(`Analysis failed: ${err.message}`);
     } finally {
       analyzeBtn.disabled = false;
       loadingEl.classList.add("hidden");
     }
   }

   // ============================================================
   // RENDER RESULTS
   // ============================================================

   function renderResults(data) {
     // API returns: binary_label, severity, bias_found, biased_segments,
     // unbiased_text, original_text
     const { original_text, unbiased_text, bias_found, biased_segments } = data;

     resultsEl.classList.remove("hidden");
     resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });

     if (!bias_found || biased_segments.length === 0) {
       document.querySelector(".summary-bar").classList.add("hidden");
       document.querySelector(".panels").classList.add("hidden");
       document.querySelector(".breakdown-section").classList.add("hidden");
       noBiasEl.classList.remove("hidden");
       return;
     }

     document.querySelector(".summary-bar").classList.remove("hidden");
     document.querySelector(".panels").classList.remove("hidden");
     document.querySelector(".breakdown-section").classList.remove("hidden");
     noBiasEl.classList.add("hidden");

     renderSummary(biased_segments);
     highlightEl.innerHTML = buildHighlightedHTML(original_text, biased_segments);
     attachMarkTooltips(biased_segments);
     unbiasedEl.textContent = unbiased_text;
     renderSegmentCards(biased_segments);
   }

   // ============================================================
   // SUMMARY BAR
   // ============================================================

   function renderSummary(segments) {
     segCountEl.textContent = `${segments.length} segment${segments.length !== 1 ? "s" : ""}`;

     const counts = { high: 0, medium: 0, low: 0 };
     segments.forEach(s => { if (counts[s.severity] !== undefined) counts[s.severity]++; });

     pillsEl.innerHTML = "";
     ["high", "medium", "low"].forEach(sev => {
       if (counts[sev] > 0) {
         const pill = document.createElement("span");
         pill.className = `pill pill-${sev}`;
         pill.textContent = `${counts[sev]} ${sev}`;
         pillsEl.appendChild(pill);
       }
     });
   }

   // ============================================================
   // HIGHLIGHTED HTML BUILDER
   // Uses start/end offsets computed server-side by pipeline.py
   // ============================================================

   function buildHighlightedHTML(text, segments) {
     // Filter segments that have valid offsets and sort by start
     const sorted = segments
       .filter(s => s.start != null && s.end != null)
       .sort((a, b) => a.start - b.start);

     let html = "";
     let cursor = 0;

     sorted.forEach((seg, idx) => {
       const { start, end, severity } = seg;
       if (start < cursor) return; // skip overlapping

       if (start > cursor) {
         html += escapeHtml(text.slice(cursor, start));
       }

       html += `<mark class="severity-${severity}" data-seg-idx="${idx}" tabindex="0">${escapeHtml(text.slice(start, end))}</mark>`;
       cursor = end;
     });

     if (cursor < text.length) {
       html += escapeHtml(text.slice(cursor));
     }

     return html;
   }

   function escapeHtml(str) {
     return str
       .replace(/&/g, "&amp;")
       .replace(/</g, "&lt;")
       .replace(/>/g, "&gt;")
       .replace(/"/g, "&quot;")
       .replace(/\n/g, "<br/>");
   }

   // ============================================================
   // TOOLTIP
   // ============================================================

   function attachMarkTooltips(segments) {
     const marks = highlightEl.querySelectorAll("mark[data-seg-idx]");
     marks.forEach(mark => {
       mark.addEventListener("mouseenter", (e) => showTooltip(e, mark, segments));
       mark.addEventListener("mouseleave", hideTooltip);
       mark.addEventListener("mousemove", repositionTooltip);
       mark.addEventListener("focus", (e) => showTooltip(e, mark, segments));
       mark.addEventListener("blur", hideTooltip);
     });
   }

   function showTooltip(e, mark, segments) {
     const idx = parseInt(mark.dataset.segIdx, 10);
     const seg = segments.filter(s => s.start != null).sort((a, b) => a.start - b.start)[idx];
     if (!seg) return;

     const sevEl = document.getElementById("tooltip-severity");
     sevEl.textContent = seg.severity.toUpperCase();
     sevEl.className = `tooltip-severity sev-${seg.severity}`;

     document.getElementById("tooltip-type").textContent = seg.bias_type || "";
     document.getElementById("tooltip-reasoning").textContent = seg.reasoning || "";
     document.getElementById("tooltip-replacement").textContent = seg.replacement || "";

     tooltip.classList.remove("hidden");
     repositionTooltip(e);
   }

   function hideTooltip() {
     tooltip.classList.add("hidden");
   }

   function repositionTooltip(e) {
     const pad = 16;
     const tw = tooltip.offsetWidth;
     const th = tooltip.offsetHeight;

     let x = e.clientX + pad;
     let y = e.clientY - th / 2;

     if (x + tw > window.innerWidth - pad) x = e.clientX - tw - pad;
     if (y < pad) y = pad;
     if (y + th > window.innerHeight - pad) y = window.innerHeight - th - pad;

     tooltip.style.left = `${x}px`;
     tooltip.style.top  = `${y}px`;
   }

   // ============================================================
   // SEGMENT BREAKDOWN CARDS
   // ============================================================

   function renderSegmentCards(segments) {
     segListEl.innerHTML = "";

     segments.forEach((seg) => {
       const card = document.createElement("div");
       card.className = `segment-card sev-${seg.severity}`;
       card.innerHTML = `
         <span class="seg-badge sev-${seg.severity}">${seg.severity}</span>
         <div class="seg-content">
           <span class="seg-original">"${escapeHtmlText(seg.original)}"</span>
           <span class="seg-type">${escapeHtmlText(seg.bias_type || "")}</span>
           <span class="seg-reasoning">${escapeHtmlText(seg.reasoning || "")}</span>
         </div>
         <div class="seg-replacement">
           <span class="seg-rep-label">Replace with</span>
           <span class="seg-rep-value">${escapeHtmlText(seg.replacement || "")}</span>
         </div>
       `;
       segListEl.appendChild(card);
     });
   }

   function escapeHtmlText(str) {
     return String(str)
       .replace(/&/g, "&amp;")
       .replace(/</g, "&lt;")
       .replace(/>/g, "&gt;")
       .replace(/"/g, "&quot;");
   }

   // ============================================================
   // COPY BUTTON
   // ============================================================

   copyBtn.addEventListener("click", () => {
     const text = unbiasedEl.textContent;
     if (!text) return;

     navigator.clipboard.writeText(text).then(() => {
       const original = copyBtn.innerHTML;
       copyBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
       setTimeout(() => { copyBtn.innerHTML = original; }, 1800);
     });
   });
