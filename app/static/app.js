"use strict";
/* ==========================================================================
   Property Assistant Client
   Streaming SSE · Real Estate Property Cards · Bilingual (EN/AR) · Fast UX
   ========================================================================== */

(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const thread = $("#thread");
  const form = $("#composer");
  const input = $("#composer-input");
  const sendBtn = $("#send");
  const banner = $("#banner");
  const welcome = $("#welcome");
  const chat = $("#chat");
  const statsDialog = $("#stats-dialog");
  const statusToggle = $("#status-toggle");
  const statsClose = $("#stats-close");
  const langToggle = $("#lang-toggle");
  const langLabel = $("#lang-label");

  const history = [];
  let lastSourceIds = [];
  let controller = null;
  let msgCounter = 0;
  const shownNotices = new Set();
  let currentLang = "en"; // "en" | "ar"

  const KIND = {
    listing: "Listing",
    project: "Development",
    article: "Article",
    faq: "FAQ",
    stat: "Market Data",
    guide: "Guide",
    page: "Page",
  };

  const KIND_AR = {
    listing: "عقار معروض",
    project: "مشروع تطويري",
    article: "مقال",
    faq: "سؤال شائع",
    stat: "بيانات السوق",
    guide: "دليل",
    page: "صفحة",
  };

  // ------------------------------------------------------------ Helpers
  const isArabic = (t) => {
    const letters = t.match(/[A-Za-z؀-ۿ]/g) || [];
    if (!letters.length) return false;
    return letters.filter((c) => /[؀-ۿ]/.test(c)).length / letters.length >= 0.3;
  };

  const escapeHtml = (s) =>
    String(s || "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    })[c]);

  const tpl = (id) => document.getElementById(id).content.firstElementChild.cloneNode(true);

  const prettyModel = (m) => {
    if (!m) return "AI Model";
    const name = m.split("/").pop();
    return name.endsWith(":free") ? `${name.slice(0, -5)} (free)` : name;
  };

  const scrollToEnd = () => {
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  };

  // ------------------------------------------------------------ Safe Markdown
  function inline(text, msgId, maxN) {
    let s = escapeHtml(text);
    // Links [text](url)
    s = s.replace(/\[([^\]\n]{1,200})\]\((https?:\/\/[^\s)]+)\)/g, (_m, label, url) =>
      `<a href="${url}" target="_blank" rel="noopener">${label}</a>`
    );
    // Bold
    s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    // Italics
    s = s.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");
    s = s.replace(/(^|[\s(])_([^_\n]+)_(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");
    // Code
    s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>");
    // Numbered citations [1] or [1, 2]
    s = s.replace(/\[(\d{1,2}(?:\s*[,،]\s*\d{1,2})*)\]/g, (_m, nums) =>
      nums
        .split(/\s*[,،]\s*/)
        .map((n) =>
          +n >= 1 && +n <= maxN
            ? `<a class="cite" href="#src-${msgId}-${+n}" data-n="${+n}" aria-label="Source ${+n}">${+n}</a>`
            : `[${n}]`
        )
        .join("")
    );
    return s;
  }

  function renderTable(rows, msgId, maxN) {
    const cells = (r) => r.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map((c) => c.trim());
    const head = cells(rows[0]);
    const body = rows.slice(2).map(cells);
    return (
      "<table><thead><tr>" +
      head.map((c) => `<th>${inline(c, msgId, maxN)}</th>`).join("") +
      "</tr></thead><tbody>" +
      body.map((r) => "<tr>" + r.map((c) => `<td>${inline(c, msgId, maxN)}</td>`).join("") + "</tr>").join("") +
      "</tbody></table>"
    );
  }

  function renderMarkdown(md, msgId, maxN) {
    const lines = md.replace(/\r/g, "").split("\n");
    let html = "";
    let list = null;
    let para = [];

    const flush = () => {
      if (para.length) html += `<p>${para.map((l) => inline(l, msgId, maxN)).join("<br>")}</p>`;
      para = [];
    };

    const closeList = () => {
      if (list) html += `</${list}>`;
      list = null;
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trimEnd();
      let m;
      if (!line.trim()) { flush(); closeList(); continue; }

      // Table check
      if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
        flush(); closeList();
        const rows = [];
        while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
        i--;
        html += renderTable(rows, msgId, maxN);
        continue;
      }

      // Headers # to ####
      if ((m = line.match(/^\s{0,3}(#{1,4})\s+(.*)$/))) {
        flush(); closeList();
        const lvl = Math.min(4, m[1].length + 2);
        html += `<h${lvl}>${inline(m[2], msgId, maxN)}</h${lvl}>`;
        continue;
      }

      // Unordered list
      if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) {
        flush();
        if (list !== "ul") { closeList(); html += "<ul>"; list = "ul"; }
        html += `<li>${inline(m[1], msgId, maxN)}</li>`;
        continue;
      }

      // Ordered list
      if ((m = line.match(/^\s*(\d{1,2})[.)]\s+(.*)$/))) {
        flush();
        if (list !== "ol") { closeList(); html += "<ol>"; list = "ol"; }
        html += `<li>${inline(m[2], msgId, maxN)}</li>`;
        continue;
      }

      // Horizontal rule
      if (/^\s*(---|\*\*\*)\s*$/.test(line)) { flush(); closeList(); html += "<hr>"; continue; }

      closeList();
      para.push(line);
    }
    flush();
    closeList();
    return html;
  }

  // ------------------------------------------------------------ Property Cards
  function renderSources(el, sources, msgId) {
    const box = $(".sources", el);
    const grid = $(".sources__grid", el);
    const countEl = $(".sources__count", el);
    grid.textContent = "";

    if (!sources || !sources.length) {
      box.hidden = true;
      return;
    }

    if (countEl) {
      countEl.textContent = `(${sources.length})`;
    }

    const isAr = currentLang === "ar";

    for (const s of sources) {
      const card = document.createElement("div");
      card.className = "prop-card";
      card.id = `src-${msgId}-${s.n}`;

      const origin = s.source === "darglobal" ? "darglobal" : "wasalt";
      const originName = origin === "darglobal" ? "DarGlobal" : "Wasalt.sa";
      const kindLabel = (isAr ? KIND_AR[s.kind] : KIND[s.kind]) || (isAr ? "عقار" : "Property");

      // Build specs row
      const specPills = [];
      if (s.bedrooms != null) {
        specPills.push(`<span class="prop-card__spec-pill">🛏️ ${s.bedrooms} ${isAr ? "غرف" : "Beds"}</span>`);
      }
      if (s.bathrooms != null) {
        specPills.push(`<span class="prop-card__spec-pill">🛁 ${s.bathrooms} ${isAr ? "حمامات" : "Baths"}</span>`);
      }
      if (s.area_sqm) {
        specPills.push(`<span class="prop-card__spec-pill">📐 ${Math.round(s.area_sqm).toLocaleString()} m²</span>`);
      }

      let snippetClean = "";
      if (s.snippet) {
        snippetClean = s.snippet.split("\n").slice(1).join(" ").replace(/^#+\s*/, "").replace(/\b[QA]:\s*/g, "").trim();
        if (snippetClean.length > 140) snippetClean = snippetClean.slice(0, 137) + "…";
      }

      const ctaLabel = origin === "darglobal"
        ? (isAr ? "استكشف المشروع ↗" : "Explore Project ↗")
        : (isAr ? "عرض تفاصيل العقار ↗" : "View on Wasalt ↗");

      card.innerHTML = `
        <div class="prop-card__head">
          <span class="prop-card__badge prop-card__badge--${origin}">
            ${origin === "darglobal" ? "🏛️" : "🏠"} ${originName} · ${kindLabel}
          </span>
          <span class="prop-card__cite">#${s.n}</span>
        </div>

        ${s.price ? `<div class="prop-card__price">${escapeHtml(s.price)}</div>` : ""}

        <h4 class="prop-card__title">
          <a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.title)}</a>
        </h4>

        ${specPills.length ? `<div class="prop-card__specs">${specPills.join("")}</div>` : ""}

        ${s.location ? `<div class="prop-card__loc">📍 ${escapeHtml(s.location)}</div>` : ""}

        ${s.completion ? `<div class="prop-card__completion" style="font-size:0.8rem;color:var(--text-secondary)">⏳ ${isAr ? "التسليم" : "Completion"}: ${escapeHtml(s.completion)}</div>` : ""}

        ${snippetClean ? `<p class="prop-card__snippet">${escapeHtml(snippetClean)}</p>` : ""}

        <a class="prop-card__btn" href="${escapeHtml(s.url)}" target="_blank" rel="noopener">
          ${ctaLabel}
        </a>
      `;

      grid.append(card);
    }

    box.hidden = false;
  }

  // ------------------------------------------------------------ Search Query Feedback
  function renderSearch(el, meta) {
    const box = $(".msg__search", el);
    if (!meta || meta.total_matches == null || !meta.filters_text) {
      box.hidden = true;
      return;
    }
    const isAr = currentLang === "ar";
    let text = isAr
      ? `🔍 تم العثور على ${meta.total_matches.toLocaleString()} نتائج مطابقة: ${meta.filters_text}.`
      : `🔍 Found ${meta.total_matches.toLocaleString()} matching properties for: ${meta.filters_text}.`;
    if (meta.relaxed && meta.relaxed.length) {
      text += isAr
        ? ` (تم تخفيف فلاتر: ${meta.relaxed.join(", ")})`
        : ` (relaxed: ${meta.relaxed.join(", ")})`;
    }
    box.textContent = text;
    box.hidden = false;
  }

  // ------------------------------------------------------------ Suggested Follow-ups
  function renderSuggestions(el, question, sources) {
    const wrap = $(".msg__suggestions", el);
    wrap.textContent = "";
    if (!sources || !sources.length) {
      wrap.hidden = true;
      return;
    }

    const isAr = isArabic(question) || currentLang === "ar";
    const suggestions = [];

    const hasDarGlobal = sources.some((s) => s.source === "darglobal");
    const hasWasalt = sources.some((s) => s.source === "wasalt");

    if (isAr) {
      if (hasWasalt) {
        suggestions.push("هل تتوفر خيارات بأسعار أقل؟", "ما هي مساحات الوحدات المتوفرة؟", "شقق للإيجار في أحياء مجاورة");
      }
      if (hasDarGlobal) {
        suggestions.push("ما هي خطة الدفع وتفاصيل الاستثمار؟", "ما هي مميزات الإقامة الذهبية والتملك الحر؟");
      }
    } else {
      if (hasWasalt) {
        suggestions.push("Are there cheaper options available?", "Show 3-bedroom villas nearby", "What amenities are included?");
      }
      if (hasDarGlobal) {
        suggestions.push("What is the payment plan and ROI?", "Tell me about Golden Visa & freehold perks");
      }
    }

    if (!suggestions.length) {
      wrap.hidden = true;
      return;
    }

    for (const text of suggestions.slice(0, 3)) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "suggestion-chip";
      btn.innerHTML = `<span>💬</span><span>${escapeHtml(text)}</span>`;
      btn.onclick = () => ask(text);
      wrap.append(btn);
    }

    wrap.hidden = false;
  }

  function statusText(done) {
    const secs = done.latency_ms ? `${(done.latency_ms / 1000).toFixed(1)}s` : "";
    switch (done.mode) {
      case "llm": {
        const first = done.ttft_ms != null ? `, initial words in ${(done.ttft_ms / 1000).toFixed(1)}s` : "";
        return `⚡ Answered by ${prettyModel(done.model)}${secs ? ` in ${secs}` : ""}${first}`;
      }
      case "cache":
        return `⚡ Instant cache match (${prettyModel(done.model)})`;
      case "retrieval-only":
        return "Direct data match (AI model in standby)";
      case "partial":
        return "Model response completed early; sources are full.";
      default:
        return "";
    }
  }

  function showNotice(notice) {
    if (!notice || shownNotices.has(notice)) return;
    shownNotices.add(notice);
    const text = {
      quota: "Today's free AI quota has reached its daily limit. Answering with verified listings straight from the data.",
      no_key: "The AI model is not connected yet. Answers list verified matches straight from DarGlobal and Wasalt data.",
      llm_error: "AI models are experiencing high demand right now. Answering with verified listings straight from the database.",
    }[notice];
    if (!text) return;
    banner.textContent = text;
    banner.className = "banner";
    banner.hidden = false;
  }

  // ------------------------------------------------------------ Network & Streaming
  async function streamChat(payload, handlers, signal) {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(payload),
      signal,
    });
    if (!res.ok) {
      let detail = `The server responded with status ${res.status}.`;
      try {
        const j = await res.json();
        detail = Array.isArray(j.detail) ? j.detail.map((d) => d.msg).join("; ") : j.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        let event = "message";
        let data = "";
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        let parsed;
        try { parsed = JSON.parse(data); } catch (_) { continue; }
        if (event === "error") throw new Error(parsed.message || "The answer could not be completed.");
        if (handlers[event]) handlers[event](parsed);
      }
    }
  }

  function setBusy(busy) {
    chat.setAttribute("aria-busy", String(busy));
    sendBtn.classList.toggle("is-stop", busy);
    sendBtn.title = busy ? "Stop generating" : "Send question";
    sendBtn.type = busy ? "button" : "submit";
  }

  async function ask(question) {
    question = question.trim();
    if (!question || controller) return;
    welcome.hidden = true;
    const rtl = isArabic(question) || currentLang === "ar";

    // 1. Append user message
    const userEl = tpl("tpl-user");
    $(".msg__body", userEl).textContent = question;
    if (rtl) userEl.dir = "rtl";
    thread.append(userEl);

    // 2. Append assistant placeholder
    const id = ++msgCounter;
    const el = tpl("tpl-assistant");
    el.id = `msg-${id}`;
    if (rtl) el.dir = "rtl";
    thread.append(el);
    scrollToEnd();

    const body = $(".msg__body", el);
    const status = $(".msg__status", el);
    let text = "";
    let sources = [];
    let maxN = 0;
    let pending = false;

    const render = () => {
      pending = false;
      body.innerHTML = renderMarkdown(text, id, maxN);
    };

    controller = new AbortController();
    setBusy(true);

    const payload = {
      message: question,
      history: history.slice(-8),
      previous_source_ids: lastSourceIds,
      stream: true,
    };

    try {
      await streamChat(
        payload,
        {
          meta: (m) => {
            sources = m.sources || [];
            maxN = sources.length;
            renderSources(el, sources, id);
            renderSearch(el, m);
          },
          token: (t) => {
            text += t.t;
            if (!pending) {
              pending = true;
              requestAnimationFrame(render);
            }
          },
          done: (d) => {
            render();
            status.textContent = statusText(d);
            showNotice(d.notice);

            // Copy button
            const copyBtn = $(".copy-btn", el);
            if (copyBtn) {
              copyBtn.hidden = !text;
              copyBtn.onclick = () => {
                navigator.clipboard?.writeText(text).then(() => {
                  const span = $("span", copyBtn);
                  const orig = span.textContent;
                  span.textContent = "Copied!";
                  setTimeout(() => { span.textContent = orig; }, 2000);
                });
              };
            }

            // Render suggested follow-ups
            renderSuggestions(el, question, sources);
            scrollToEnd();
          },
        },
        controller.signal
      );

      history.push({ role: "user", content: question }, { role: "assistant", content: text.slice(0, 4000) });
      lastSourceIds = sources.map((s) => s.id);
    } catch (err) {
      if (err.name === "AbortError") {
        text += text ? "\n\n_Answer stopped by user._" : "_Answer stopped by user._";
        render();
      } else {
        body.classList.add("is-error");
        body.textContent = err.message || "The request could not be completed. Please check your connection.";
        const retry = $(".retry-btn", el);
        if (retry) {
          retry.hidden = false;
          retry.onclick = () => {
            el.remove();
            userEl.remove();
            ask(question);
          };
        }
      }
    } finally {
      controller = null;
      setBusy(false);
      input.focus();
    }
  }

  // ------------------------------------------------------------ Stats & Status
  async function loadStats() {
    const fresh = $("#freshness");
    const dl = $("#about-stats");
    try {
      const res = await fetch("/api/stats");
      const s = await res.json();
      const c = (s.index && s.index.counts) || {};
      const listings = c["wasalt:listing"] || 0;
      const dgProjects = c["darglobal:project"] || 0;
      const wProjects = c["wasalt:project"] || 0;
      const totalProps = listings + dgProjects + wProjects;
      const collected = s.snapshot && s.snapshot.collected_at ? new Date(s.snapshot.collected_at) : null;
      const when = collected ? collected.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) : "Live";

      if (s.ready) {
        fresh.textContent = `${totalProps.toLocaleString()} properties & projects (${when})`;
      } else {
        fresh.textContent = "Syncing market data…";
        banner.textContent = "The assistant is refreshing property data from Wasalt and DarGlobal. Queries remain active.";
        banner.hidden = false;
        setTimeout(loadStats, 15000);
      }

      const rows = [
        ["Status", s.ready ? "Live & Ready" : "Syncing"],
        ["Wasalt Listings", listings.toLocaleString()],
        ["Wasalt Projects", wProjects.toLocaleString()],
        ["DarGlobal Projects", dgProjects.toLocaleString()],
        ["Search Engine", s.index && s.index.dense_retrieval ? "Hybrid (BM25 + Dense Semantic)" : "Keyword Search"],
        ["AI Provider", s.llm && s.llm.configured ? prettyModel(s.llm.last_model || "OpenRouter") : "Offline / Cache Only"],
        ["Last Synchronized", collected ? collected.toLocaleString() : "Unknown"],
      ];

      dl.innerHTML = rows.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`).join("");
    } catch (_) {
      fresh.textContent = "Market Data Online";
    }
  }

  // ------------------------------------------------------------ Bilingual Toggle
  function toggleLanguage() {
    currentLang = currentLang === "en" ? "ar" : "en";
    const isAr = currentLang === "ar";
    document.documentElement.dir = isAr ? "rtl" : "ltr";
    langLabel.textContent = isAr ? "English" : "عربي";
    input.placeholder = isAr
      ? "اسأل عن العقارات، الأسعار، المشاريع الفاخرة، أو الأحياء…"
      : "Ask about properties, rental prices, luxury projects, or locations…";
    input.dir = isAr ? "rtl" : "auto";
  }

  // ------------------------------------------------------------ Wiring & Event Listeners
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value;
    input.value = "";
    autosize();
    ask(q);
  });

  sendBtn.addEventListener("click", (e) => {
    if (controller) {
      e.preventDefault();
      controller.abort();
    }
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (!controller) form.requestSubmit();
    }
  });

  const autosize = () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  };

  input.addEventListener("input", () => {
    autosize();
    if (currentLang !== "ar") {
      input.dir = isArabic(input.value) ? "rtl" : "auto";
    }
  });

  // Starter buttons & quick filter chips
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".starter-card, .chip");
    if (btn && btn.dataset.q) {
      ask(btn.dataset.q);
    }
  });

  // New Chat Button
  $("#new-chat").addEventListener("click", () => {
    if (controller) controller.abort();
    thread.textContent = "";
    history.length = 0;
    lastSourceIds = [];
    welcome.hidden = false;
    input.focus();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // Language Toggle
  langToggle.addEventListener("click", toggleLanguage);

  // Status Dialog Toggle
  statusToggle.addEventListener("click", () => {
    statsDialog.showModal();
  });
  statsClose.addEventListener("click", () => {
    statsDialog.close();
  });
  statsDialog.addEventListener("click", (e) => {
    if (e.target === statsDialog) statsDialog.close();
  });

  // Citation click scrolls and highlights the property card
  thread.addEventListener("click", (e) => {
    const cite = e.target.closest(".cite");
    if (!cite) return;
    e.preventDefault();
    const target = document.querySelector(cite.getAttribute("href"));
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      target.classList.remove("is-highlighted");
      void target.offsetWidth; // trigger reflow
      target.classList.add("is-highlighted");
    }
  });

  // Initialize
  loadStats();
})();
