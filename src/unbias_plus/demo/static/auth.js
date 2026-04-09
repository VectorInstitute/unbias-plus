/* ============================================================
   UnBias+ — Supabase Auth
   Handles: session management, sign in, sign up, sign out,
            JWT retrieval for API calls, cloud-mode detection
   ============================================================ */

// Cloud mode detection — injected by server via window.__UNBIAS_CONFIG__
const SUPABASE_URL      = window.__UNBIAS_CONFIG__?.supabaseUrl || "";
const SUPABASE_ANON_KEY = window.__UNBIAS_CONFIG__?.supabaseAnonKey || "";
const IS_CLOUD = Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);

let _supabase = null;

function getSupabase() {
  if (!IS_CLOUD) return null;
  if (!_supabase) {
    _supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
  }
  return _supabase;
}

// ── Session helpers ──────────────────────────────────────────

async function getSession() {
  const client = getSupabase();
  if (!client) return null;
  const { data } = await client.auth.getSession();
  return data?.session ?? null;
}

async function getJwt() {
  const session = await getSession();
  return session?.access_token ?? null;
}

async function getCurrentUser() {
  const session = await getSession();
  return session?.user ?? null;
}

// ── Auth guard for index.html ────────────────────────────────
// Call on page load. Redirects to /login if not authenticated (cloud only).

async function requireAuth() {
  if (!IS_CLOUD) return; // local mode — no auth needed
  const session = await getSession();
  if (!session) {
    window.location.href = "/login";
  }
}

// ── Sign out ─────────────────────────────────────────────────

async function signOut() {
  const client = getSupabase();
  if (!client) return;
  await client.auth.signOut();
  window.location.href = "/login";
}

// ── Feedback submission ──────────────────────────────────────

async function submitFeedback({ reaction, message, inputText, rating, speed, accuracy }) {
  const jwt = await getJwt();
  if (!jwt) return { ok: false, error: "Not authenticated" };

  try {
    const res = await fetch("/feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${jwt}`,
      },
      body: JSON.stringify({
        reaction,
        message: message || "",
        input_text: inputText || "",
        rating:   rating   ?? null,
        speed:    speed    || null,
        accuracy: accuracy || null,
      }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      return { ok: false, error: data.detail || "Server error" };
    }
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

// ── Patch fetch for /analyze/stream to include JWT ───────────
// Wraps the existing runAnalysis so the JWT is sent automatically.

async function getAuthHeaders() {
  if (!IS_CLOUD) return {};
  const jwt = await getJwt();
  if (!jwt) return {};
  return { "Authorization": `Bearer ${jwt}` };
}

/* ============================================================
   INIT — runs only on /demo (the protected app shell)
   ============================================================ */

document.addEventListener("DOMContentLoaded", async () => {
  if (!IS_CLOUD) return;                             // local mode — nothing to do
  if (window.location.pathname !== "/demo") return;  // only guard the app page

  // 1. Auth guard — redirect to /login if not signed in
  await requireAuth();

  // 2. Show header auth controls
  const user = await getCurrentUser();
  if (user) {
    const headerAuth = document.getElementById("header-auth");
    const emailEl    = document.getElementById("header-user-email");
    if (headerAuth) headerAuth.classList.remove("hidden");
    if (emailEl) emailEl.textContent = user.email.split("@")[0];
  }

  // 3. Sign out button
  document.getElementById("signout-btn")?.addEventListener("click", signOut);
});
