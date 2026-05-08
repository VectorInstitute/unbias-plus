/* ============================================================
   UnBias — Frontend Logic
   Handles: streaming API, highlighting, tooltip, breakdown cards,
            example chips, file upload (.txt / .pdf)
   ============================================================ */
   const inputEl       = document.getElementById("input-text");
   const charCountEl   = document.getElementById("char-count");
   const analyzeBtn    = document.getElementById("analyze-btn");
   const loadingEl     = document.getElementById("loading");
   const resultsEl     = document.getElementById("results");
   const highlightEl   = document.getElementById("highlighted-text");
   const unbiasedEl    = document.getElementById("unbiased-text");
   const segCountEl    = document.getElementById("segment-count");
   const pillsEl       = document.getElementById("severity-pills");
   const segListEl     = document.getElementById("segment-list");
   const noBiasEl      = document.getElementById("no-bias");
   const copyBtn       = document.getElementById("copy-btn");
   const tooltip       = document.getElementById("tooltip");
   const errorBannerEl = document.getElementById("error-banner");
   const MAX_CHARS = 5000;
   const MAX_COLD_START_RETRIES = 10; // 10 × 5 s = 50 s window for uvicorn to start
   let coldStartRetries = 0;
   let _activeAnalysisController = null; // AbortController for the in-flight stream
   // ============================================================
   // INLINE ERROR BANNER
   // ============================================================
   function showInlineError(msg) {
     if (!errorBannerEl) return;
     errorBannerEl.textContent = msg;
     errorBannerEl.classList.remove("hidden");
     setTimeout(() => errorBannerEl.classList.add("hidden"), 8000);
   }
   // ============================================================
   // EXAMPLE CHIPS
   // ============================================================
   document.querySelectorAll(".example-chip").forEach(chip => {
     chip.addEventListener("click", () => {
       inputEl.value = chip.dataset.text;
       inputEl.dispatchEvent(new Event("input"));
       inputEl.focus();
     });
   });
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
   // FILE UPLOAD — .txt and text-based .pdf, fully client-side
   // ============================================================
   const uploadBtn      = document.getElementById("upload-btn");
   const fileInput      = document.getElementById("file-input");
   const uploadFilename = document.getElementById("upload-filename");
   uploadBtn.addEventListener("click", () => fileInput.click());
   fileInput.addEventListener("change", async () => {
     const file = fileInput.files[0];
     if (!file) return;
     const ext = file.name.split(".").pop().toLowerCase();
     let text = "";
     try {
       if (ext === "txt") {
         text = await _readAsText(file);
       } else if (ext === "pdf") {
         text = await _extractPdfText(file);
       } else {
         showInlineError("Unsupported file type. Please upload a .txt or .pdf file.");
         return;
       }
     } catch (err) {
       showInlineError("Could not read file: " + err.message);
       fileInput.value = "";
       return;
     }
     text = text.trim();
     if (!text) {
       showInlineError("This PDF appears to be scanned or image-based. Please paste the text manually.");
       fileInput.value = "";
       return;
     }
     if (text.length > MAX_CHARS) {
       text = text.slice(0, MAX_CHARS);
       showInlineError(`File truncated to ${MAX_CHARS.toLocaleString()} characters.`);
     }
     inputEl.value = text;
     inputEl.dispatchEvent(new Event("input"));
     inputEl.focus();
     uploadFilename.textContent = file.name;
     uploadFilename.classList.remove("hidden");
     fileInput.value = "";
   });
   function _readAsText(file) {
     return new Promise((resolve, reject) => {
       const reader = new FileReader();
       reader.onload  = e => resolve(e.target.result);
       reader.onerror = () => reject(new Error("Failed to read file"));
       reader.readAsText(file, "utf-8");
     });
   }
   async function _extractPdfText(file) {
     if (typeof pdfjsLib === "undefined") {
       throw new Error("PDF library not loaded. Please refresh and try again.");
     }
     pdfjsLib.GlobalWorkerOptions.workerSrc =
       "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
     const arrayBuffer = await file.arrayBuffer();
     const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
     const pages = [];
     for (let i = 1; i <= pdf.numPages; i++) {
       const page    = await pdf.getPage(i);
       const content = await page.getTextContent();
       pages.push(content.items.map(item => item.str).join(" "));
     }
     return pages.join("\n\n");
   }
   // ============================================================
   // ANALYZE — /analyze/stream (SSE)
   // ============================================================
   analyzeBtn.addEventListener("click", runAnalysis);
   inputEl.addEventListener("keydown", (e) => {
     if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runAnalysis();
   });
   async function runAnalysis() {
     const text = inputEl.value.trim();
     if (!text) return;
     if (text.length > MAX_CHARS) {
       showInlineError(`Text too long. Please keep it under ${MAX_CHARS.toLocaleString()} characters.`);
       return;
     }
     // Cancel any in-flight stream so its DOM writes don't race with this one.
     if (_activeAnalysisController) {
       _activeAnalysisController.abort();
     }
     _activeAnalysisController = new AbortController();
     const signal = _activeAnalysisController.signal;
     analyzeBtn.disabled = true;
     resultsEl.classList.add("hidden");
     loadingEl.classList.remove("hidden");
     unbiasedEl.innerHTML = "";
     const labelEl = document.querySelector(".loading-label");
     let tokenCount = 0;
     let firstTokenReceived = false;
     let serverConnected = false; // true once HTTP 200 received — model is up, may just be busy
     let accumulated = "";
     let streamingSegments = [];
     let lastStreamedUnbiased = "";
     let resultRendered = false;
     let streamLoadingDismissed = false;
     // Elapsed timer — updates label every second until first token arrives.
     // Cold-start warning only fires before HTTP 200 arrives; once connected,
     // slow tokens mean the GPU is under load, not that the model is loading.
     const startTime = Date.now();
     const timerInterval = setInterval(() => {
       if (firstTokenReceived) return;
       const elapsed = Math.floor((Date.now() - startTime) / 1000);
       if (!labelEl) return;
       if (serverConnected || elapsed < 8) {
         labelEl.textContent = "Analyzing bias patterns...";
       } else {
         labelEl.innerHTML =
           "Loading model from GCS\u00a0\u00a0"
           + "<span style='font-variant-numeric:tabular-nums;'>" + elapsed + "s</span>"
           + "<br><span style='font-size:0.85em;opacity:0.6;'>New GPU instance is warming up (~7 min). Hang tight.</span>";
       }
     }, 1000);
     try {
      const res = await fetch("/analyze/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal,
      });
       if (!res.ok) {
         let detail = `Server error (${res.status})`;
         try { detail = (await res.json()).detail || detail; } catch {}
         const httpErr = new Error(detail);
         httpErr.status = res.status;
         throw httpErr;
       }
       serverConnected = true; // model is loaded; remaining wait is GPU queue time
       const reader  = res.body.getReader();
       const decoder = new TextDecoder();
       let buffer = "";
       //Parse one SSE `data: ...` line (chunk boundaries and EOF flush).
       function consumeDataLine(line) {
         const trimmed = line.replace(/\r$/, "").trim();
         if (!trimmed.startsWith("data: ")) return;
         let payload;
         try {
           payload = JSON.parse(trimmed.slice(6));
         } catch {
           return;
         }
         if (payload.t !== undefined) {
           if (!firstTokenReceived) {
             firstTokenReceived = true;
             clearInterval(timerInterval);
           }
           tokenCount++;
           accumulated += payload.t;
           if (labelEl) labelEl.textContent = `Analyzing... (${tokenCount} tokens)`;
           const newSegs = parseNewSegments(accumulated, streamingSegments.length, text);
           if (newSegs.length > 0) {
             streamingSegments = streamingSegments.concat(newSegs);
             renderStreamingPartial(text, streamingSegments);
           }
           // unbiased_text is last in the schema; the server only sends {"result":...}
           // after the full stream ends — extract the string from partial JSON so the
           // UnBiased panel fills as it streams instead of staying empty until max_tokens.
           const ub = parseUnbiasedTextField(accumulated);
           const unbiasedGrew = ub && ub.text.length > 0 && ub.text !== lastStreamedUnbiased;
           const segmentsJustAdded = newSegs.length > 0;
           if (ub && ub.text.length > 0 && (unbiasedGrew || segmentsJustAdded)) {
             lastStreamedUnbiased = ub.text;
             resultsEl.classList.remove("hidden");
             document.querySelector(".panels")?.classList.remove("hidden");
             // Same green replacement marks as final renderResults — not plain escapeHtml,
             // or highlights would only appear after the stream ends.
             unbiasedEl.innerHTML =
               streamingSegments.length > 0
                 ? buildUnbiasedHTML(text, ub.text, streamingSegments)
                 : escapeHtml(ub.text);
           }
          // Neutral rewrite JSON string is closed — hide spinner and re-enable
          // the button immediately. The stream may still drain the final
          // ``result`` event + EOF in the background, but the UI is done.
          if (ub && ub.closed && !streamLoadingDismissed) {
            streamLoadingDismissed = true;
            clearInterval(timerInterval);
            loadingEl.classList.add("hidden");
            analyzeBtn.disabled = false;
          }
         } else if (payload.result !== undefined) {
           resultRendered = true;
           renderResults(payload.result);
         } else if (payload.error !== undefined) {
           throw new Error(payload.error);
         }
       }
       while (true) {
         const { done, value } = await reader.read();
         if (done) {
           // Last event may have no trailing newline and remain in `buffer`; parse it
           // so the final `result` (unbiased_text) is not dropped.
           if (buffer.trim()) {
             for (const line of buffer.split("\n")) {
               consumeDataLine(line);
             }
           }
           buffer = "";
           // Fallback: if stream ended without a result event, parse accumulated
           // tokens directly. Handles <think> blocks, markdown fences, and
           // truncated JSON — mirrors the same logic as backend parser.py.
           if (accumulated && !resultRendered) {
             try {
               let raw = accumulated;
               // Step 1: strip <think>...</think>
               if (raw.includes("</think>")) {
                 raw = raw.split("</think>").pop().trim();
               } else if (raw.includes("<think>")) {
                 raw = "";
               }
               // Step 2: strip markdown fences
               const fenced = raw.match(/```(?:json)?\s*(\{[\s\S]*\})\s*```/);
               if (fenced) {
                 raw = fenced[1].trim();
               } else {
                 // Step 3: extract outermost { ... }
                 const start = raw.indexOf("{");
                 const end = raw.lastIndexOf("}");
                 if (start !== -1 && end !== -1) {
                   raw = raw.slice(start, end + 1);
                 }
               }
               if (raw) {
                 const parsed = JSON.parse(raw);
                 renderResults({ ...parsed, original_text: text });
               }
             } catch {}
           }
           break;
         }
         buffer += decoder.decode(value, { stream: true });
         const lines = buffer.split("\n");
         buffer = lines.pop() ?? "";
         for (const line of lines) {
           consumeDataLine(line);
         }
       }
     } catch (err) {
       // Silently discard — this stream was intentionally cancelled by a new analysis.
       if (err.name === "AbortError") return;
       clearInterval(timerInterval);
       const elapsed = Math.floor((Date.now() - startTime) / 1000);
       // 502/503 with no tokens = uvicorn still starting behind nginx (CPU cold start ~30s).
       // Auto-retry with a live countdown instead of showing an error popup.
       const isCpuColdStart = !firstTokenReceived && (err.status === 502 || err.status === 503);
       if (isCpuColdStart && coldStartRetries < MAX_COLD_START_RETRIES) {
         coldStartRetries++;
         let retryIn = 5;
         if (labelEl) labelEl.innerHTML =
           "Server is starting up \u2014 retrying in <b id='retry-cd'>" + retryIn + "</b>s"
           + "\u00a0\u00a0<span style='font-size:0.85em;opacity:0.6;'>(attempt " + coldStartRetries + "/" + MAX_COLD_START_RETRIES + ")</span>";
         const tick = setInterval(() => {
           retryIn--;
           const cd = document.getElementById("retry-cd");
           if (cd) cd.textContent = retryIn;
           if (retryIn <= 0) { clearInterval(tick); runAnalysis(); }
         }, 1000);
         return; // keep button disabled, retry automatically
       }
       // GPU cold start: long wait with no tokens received.
       const isGpuColdStart = elapsed > 8 && !firstTokenReceived && !serverConnected;
       analyzeBtn.disabled = false;
       coldStartRetries = 0;
       if (isGpuColdStart) {
         if (labelEl) labelEl.innerHTML =
           "No response after " + elapsed + "s — service may be starting up or temporarily unavailable.<br>"
           + "<span style='font-size:0.85em;opacity:0.6;'>If a new GPU instance is loading, it takes ~7 min. Click Analyze again to retry.</span>";
       } else {
         loadingEl.classList.add("hidden");
         if (labelEl) labelEl.textContent = "Analyzing bias patterns...";
         showInlineError(err.message);
       }
       return;
     }
     clearInterval(timerInterval);
     coldStartRetries = 0;
     analyzeBtn.disabled = false;
     if (!streamLoadingDismissed) loadingEl.classList.add("hidden");
     if (labelEl) labelEl.textContent = "Analyzing bias patterns...";
   }
   // ============================================================
   // PARTIAL unbiased_text (streaming JSON string value)
   // ============================================================
   /**
    * Read the JSON string value for the top-level "unbiased_text" key from partial
    * model output. Supports \\n, \\r, \\t, \\", \\\\, and \\uXXXX inside the string.
    *
    * @param {string} raw
    * @returns {{ text: string, closed: boolean } | null}
    */
   function parseUnbiasedTextField(raw) {
     const key = '"unbiased_text"';
     const k = raw.indexOf(key);
     if (k === -1) return null;
     let i = k + key.length;
     while (i < raw.length && /\s/.test(raw[i])) i++;
     if (i >= raw.length || raw[i] !== ":") return null;
     i++;
     while (i < raw.length && /\s/.test(raw[i])) i++;
     if (i >= raw.length || raw[i] !== '"') return null;
     i++;
     let out = "";
     while (i < raw.length) {
       const c = raw[i];
       if (c === '"') return { text: out, closed: true };
       if (c === "\\") {
         if (i + 1 >= raw.length) return { text: out, closed: false };
         const n = raw[i + 1];
         if (n === "u") {
           if (i + 6 <= raw.length) {
             const hex = raw.slice(i + 2, i + 6);
             if (/^[0-9a-fA-F]{4}$/.test(hex)) {
               out += String.fromCharCode(parseInt(hex, 16));
               i += 6;
               continue;
             }
           }
           return { text: out, closed: false };
         }
         i += 2;
         switch (n) {
           case "n": out += "\n"; break;
           case "r": out += "\r"; break;
           case "t": out += "\t"; break;
           case '"': out += '"'; break;
           case "\\": out += "\\"; break;
           default: out += n;
         }
         continue;
       }
       out += c;
       i++;
     }
     return { text: out, closed: false };
   }
   // ============================================================
   // PROGRESSIVE SEGMENT PARSER
   // Walks accumulated raw JSON text, extracts complete segment
   // objects from the biased_segments array using brace counting,
   // skips already-rendered ones, computes offsets client-side.
   // ============================================================
   function parseNewSegments(raw, alreadyParsed, inputText) {
     const marker = '"biased_segments"';
     const markerIdx = raw.indexOf(marker);
     if (markerIdx === -1) return [];
     const arrStart = raw.indexOf("[", markerIdx);
     if (arrStart === -1) return [];
     const segments = [];
     let i = arrStart + 1;
     let segsParsed = 0;
     while (i < raw.length) {
       // Skip whitespace
       while (i < raw.length && /\s/.test(raw[i])) i++;
       if (i >= raw.length || raw[i] === "]") break;
       if (raw[i] !== "{") { i++; continue; }
       // Find the matching closing brace
       let depth = 0;
       let j = i;
       while (j < raw.length) {
         if (raw[j] === "{") depth++;
         else if (raw[j] === "}") { depth--; if (depth === 0) break; }
         j++;
       }
       if (depth !== 0) break; // incomplete object, wait for more tokens
       const objStr = raw.slice(i, j + 1);
       try {
         const seg = JSON.parse(objStr);
         if (segsParsed >= alreadyParsed) {
           // Compute client-side offsets
           const start = inputText.indexOf(seg.original);
           seg.start = start >= 0 ? start : null;
           seg.end   = start >= 0 ? start + seg.original.length : null;
           segments.push(seg);
         }
         segsParsed++;
       } catch { /* incomplete JSON, stop */ break; }
       i = j + 1;
       // Skip comma between objects
       while (i < raw.length && /[\s,]/.test(raw[i])) i++;
     }
     return segments;
   }
   // ============================================================
   // PARTIAL RENDER (called as each streaming segment arrives)
   // ============================================================
   function renderStreamingPartial(inputText, segments) {
     resultsEl.classList.remove("hidden");
     document.querySelector(".summary-bar").classList.remove("hidden");
     document.querySelector(".panels").classList.remove("hidden");
     document.querySelector(".breakdown-section").classList.remove("hidden");
     noBiasEl.classList.add("hidden");
     renderSummary(segments);
     highlightEl.innerHTML = buildHighlightedHTML(inputText, segments);
     attachMarkTooltips(segments);
     renderSegmentCards(segments);
   }
   // ============================================================
   // RENDER RESULTS (final — called on result event)
   // ============================================================
   function renderResults(data) {
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
    unbiasedEl.innerHTML  = buildUnbiasedHTML(original_text, unbiased_text, biased_segments);
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
   // ============================================================
   function buildHighlightedHTML(text, segments) {
     const sorted = segments
       .filter(s => s.start != null && s.end != null)
       .sort((a, b) => a.start - b.start);
     let html   = "";
     let cursor = 0;
     sorted.forEach((seg, idx) => {
       const { start, end, severity } = seg;
       if (start < cursor) return;
       if (start > cursor) html += escapeHtml(text.slice(cursor, start));
       html += `<mark class="severity-${severity}" data-seg-idx="${idx}" tabindex="0">${escapeHtml(text.slice(start, end))}</mark>`;
       cursor = end;
     });
     if (cursor < text.length) html += escapeHtml(text.slice(cursor));
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
       mark.addEventListener("mousemove",  repositionTooltip);
       mark.addEventListener("focus",      (e) => showTooltip(e, mark, segments));
       mark.addEventListener("blur",       hideTooltip);
     });
   }
   function showTooltip(e, mark, segments) {
     const idx = parseInt(mark.dataset.segIdx, 10);
     const seg = segments.filter(s => s.start != null).sort((a, b) => a.start - b.start)[idx];
     if (!seg) return;
     const sevEl = document.getElementById("tooltip-severity");
     sevEl.textContent = seg.severity.toUpperCase();
     sevEl.className   = `tooltip-severity sev-${seg.severity}`;
     document.getElementById("tooltip-type").textContent        = seg.bias_type   || "";
     document.getElementById("tooltip-reasoning").textContent   = seg.reasoning   || "";
     document.getElementById("tooltip-replacement").textContent = seg.replacement || "";
     tooltip.classList.remove("hidden");
     repositionTooltip(e);
   }
   function hideTooltip() { tooltip.classList.add("hidden"); }
   function repositionTooltip(e) {
     const pad = 16;
     const tw  = tooltip.offsetWidth;
     const th  = tooltip.offsetHeight;
     let x = e.clientX + pad;
     let y = e.clientY - th / 2;
     if (x + tw > window.innerWidth  - pad) x = e.clientX - tw - pad;
     if (y < pad)                           y = pad;
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
   // ============================================================
   // UNBIASED HTML BUILDER
   // ============================================================
  function buildUnbiasedHTML(original, unbiased, segments) {
    const replacements = segments.map(s => s.replacement).filter(Boolean);
    let html = escapeHtmlText(unbiased);
    replacements.forEach((rep) => {
      if (!rep) return;
      const escaped = escapeHtmlText(rep);
      html = html.replace(escaped, `<mark class="replaced-green">${escaped}</mark>`);
    });
    return html;
  }
