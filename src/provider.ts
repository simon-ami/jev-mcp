// Jev transport: TypeSafe direct (default), OpenRouter Decisions, Cloudflare
// Workers AI, Vercel AI Gateway, Google Gemini Decisions, or a caller-supplied
// Jev-compatible System One endpoint. All speak the noul/choice/score judgment
// contract; URL, auth, model slugs, and minor wire fields differ. Proxies add
// hops, so direct TypeSafe remains the recommended default.

import { ask, openrouterJevModel, resolveTransport, type JevTransport, type JevTransportReply } from "@jkudish/jev-agent-tools";
import { isRecord } from "./lib.js";

export type JevProvider = "typesafe" | "openrouter" | "cloudflare" | "vercel" | "gemini" | "compatible";

export interface AskResult {
  answers: Record<string, any>;
  usage: { input_tokens: number; output_tokens: number };
  provider: JevProvider;
  model: string;
}

const DEFAULT_GEMINI_MODEL = "gemini-decision-1";

const X_TITLE = "jev-mcp";
const REFERER = "https://github.com/jkudish/jev-mcp";

// Replace every occurrence of the secret so a reflecting endpoint cannot leak
// it into MCP-visible error text; covering the bare form also covers the
// "Bearer <secret>" form.
function redactSecret(text: string, secret: string): string {
  return secret ? text.split(secret).join("[redacted]") : text;
}

// ── Transport resilience ─────────────────────────────────────────────────────
// Research and the original report: GitHub issue #23 by oppih. Applies to the
// fetch-based transports (openrouter, cloudflare, gemini, compatible); the
// typesafe and vercel branches go through SDK-owned transports whose retry and
// timeout semantics are theirs, so no uniformity is claimed for those.

const positiveIntFromEnv = (name: string, fallback: number): number => {
  const raw = Number(process.env[name]);
  return Number.isInteger(raw) && raw > 0 ? raw : fallback;
};

/** Whole-request deadline (all attempts), overridable for tests and tight hosts. */
const REQUEST_TIMEOUT_MS = positiveIntFromEnv("JEV_MCP_REQUEST_TIMEOUT_MS", 60_000);
/** Total attempts per request, including the first; clamped to 1..6. */
const MAX_ATTEMPTS = Math.min(6, Math.max(1, positiveIntFromEnv("JEV_MCP_MAX_ATTEMPTS", 3)));
const BASE_RETRY_DELAY_MS = 500;
const MAX_RETRY_DELAY_MS = 4_000;
/** Stream-checked ceiling for success and error bodies alike. */
const MAX_RESPONSE_BYTES = 1_000_000;

const SAFE_OPENROUTER_ERROR_TYPES = new Set(["max_tokens_exceeded"]);

const SAFE_GEMINI_ERROR_CODES = new Set([
  "safety",
  "prohibited_content",
  "blocklist",
  "image_safety",
  "content_blocked",
]);

const SAFE_GEMINI_BLOCK_REASONS = new Set([
  "safety",
  "other",
  "blocklist",
  "prohibited_content",
  "image_safety",
]);

/**
 * Gemini's Decisions API reports safety blocks either as an HTTP 400
 * `{ error: { code, message } }` or as `prompt_feedback.block_reason` on 200.
 * Expose only allow-listed constants; never return upstream text.
 */
function safeGeminiErrorCode(bodyText: string): string | undefined {
  try {
    const body = JSON.parse(bodyText);
    const code = body?.error?.code;
    return typeof code === "string" && SAFE_GEMINI_ERROR_CODES.has(code) ? code : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Map a Jev model name to Gemini Decisions API model IDs.
 * The global default `jev-latest` maps to `gemini-decision-1`; any explicit
 * model (e.g. `gemini-decision-1` or future versions) passes through verbatim.
 */
export function geminiModel(model: string): string {
  return !model || model === "jev-latest" ? DEFAULT_GEMINI_MODEL : model;
}

/**
 * Map a model name to OpenRouter's Decisions API model ID.
 * Bare Jev versions and `jev-latest` delegate to `openrouterJevModel`, while
 * explicit provider-prefixed slugs (e.g. `google/gemini-decision-1`) pass
 * through verbatim instead of getting `typesafe/` prepended.
 */
export function openrouterModel(model: string): string {
  if (model === "jev-latest" || model === "typesafe/jev-latest" || model === "~typesafe/jev-latest") {
    return openrouterJevModel(model);
  }
  return model.includes("/") ? model : openrouterJevModel(model);
}

/**
 * Serialize Jev's arbitrary JSON `state` to Gemini's `input` string.
 * Per Google's Decisions API docs, structured JSON payloads should be
 * serialized as text so payload keys stay data rather than content blocks,
 * and `input` cannot be empty.
 */
export function toGeminiInput(state: unknown): string {
  if (typeof state === "string") return state.length > 0 ? state : JSON.stringify(state);
  return JSON.stringify(state) ?? "null";
}

/**
 * TypeSafe Choice confidence: scales p_max linearly from 0 (even split 1/n)
 * to 1 (certainty). Matches https://docs.typesafe.ai/confidence.md.
 */
export function choiceConfidence(probabilities: Record<string, unknown>): number | null {
  const values = Object.values(probabilities);
  const n = values.length;
  if (n < 2) return null;
  let pMax = -1;
  for (const v of values) {
    if (typeof v !== "number" || !Number.isFinite(v) || v < 0 || v > 1) return null;
    if (v > pMax) pMax = v;
  }
  const raw = (pMax - 1 / n) / (1 - 1 / n);
  return Math.min(1, Math.max(0, raw));
}

/**
 * TypeSafe Score confidence: compares probability-weighted distance from the
 * peak level against the uniform distribution's mean absolute deviation from
 * the middle level. Matches https://docs.typesafe.ai/confidence.md.
 */
export function scoreConfidence(probabilities: Record<string, unknown>): number | null {
  const n = Object.keys(probabilities).length;
  if (n < 2) return null;
  const probs: number[] = [];
  let peakIndex = 0;
  let peakProb = -1;
  for (let i = 0; i < n; i++) {
    if (!Object.hasOwn(probabilities, String(i))) return null;
    const p = probabilities[String(i)];
    if (typeof p !== "number" || !Number.isFinite(p) || p < 0 || p > 1) return null;
    probs.push(p);
    if (p > peakProb) {
      peakProb = p;
      peakIndex = i;
    }
  }
  let spread = 0;
  let evenSpread = 0;
  const mid = (n - 1) / 2;
  for (let i = 0; i < n; i++) {
    spread += probs[i] * Math.abs(i - peakIndex);
    evenSpread += Math.abs(i - mid);
  }
  evenSpread /= n;
  if (evenSpread <= 0) return null;
  return Math.min(1, Math.max(0, 1 - spread / evenSpread));
}

/**
 * Recompute Choice and Score `confidence` on Gemini answers using TypeSafe's
 * formulas (0 = uniform/split, 1 = certain) instead of Gemini's raw `p_max`,
 * so tool default thresholds (`auto_accept`, `review_at`) behave consistently.
 */
export function adaptGeminiAnswers(answers: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [id, raw] of Object.entries(answers)) {
    if (!isRecord(raw) || !isRecord(raw.probabilities)) {
      out[id] = raw;
      continue;
    }
    if (raw.type === "choice") {
      const confidence = choiceConfidence(raw.probabilities);
      out[id] = confidence !== null ? { ...raw, confidence } : raw;
    } else if (raw.type === "score") {
      const confidence = scoreConfidence(raw.probabilities);
      out[id] = confidence !== null ? { ...raw, confidence } : raw;
    } else {
      out[id] = raw;
    }
  }
  return out;
}

/**
 * Translate Jev question objects to the Gemini Decisions API wire shape:
 * - `instructions`: must be text (serialize structured object instructions such
 *   as `jev_classify`'s `{ task, item }` to JSON).
 * - `noul.criteria`: Jev uses `{ true, false }`; Gemini uses
 *   `{ true_description, false_description }` (optional).
 * - `choice.criteria`: Jev allows `null` for self-explanatory option IDs;
 *   Gemini requires string descriptions (`""` when self-explanatory).
 * - `score.criteria`: both use a 0-indexed string array.
 */
export function toGeminiQuestions(questions: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [id, raw] of Object.entries(questions)) {
    if (!isRecord(raw)) {
      out[id] = raw;
      continue;
    }
    const instructions = typeof raw.instructions === "string" ? raw.instructions : JSON.stringify(raw.instructions ?? "");
    const type = raw.type;
    if (type === "noul") {
      const q: Record<string, unknown> = { type: "noul", instructions };
      if (isRecord(raw.criteria)) {
        const criteria: Record<string, string> = {};
        const trueDesc = raw.criteria.true_description ?? raw.criteria.true;
        const falseDesc = raw.criteria.false_description ?? raw.criteria.false;
        if (typeof trueDesc === "string") criteria.true_description = trueDesc;
        if (typeof falseDesc === "string") criteria.false_description = falseDesc;
        if (Object.keys(criteria).length > 0) q.criteria = criteria;
      }
      out[id] = q;
    } else if (type === "choice") {
      const criteria: Record<string, string> = {};
      if (isRecord(raw.criteria)) {
        for (const [key, desc] of Object.entries(raw.criteria)) {
          criteria[key] = typeof desc === "string" ? desc : "";
        }
      }
      out[id] = { type: "choice", instructions, criteria };
    } else if (type === "score") {
      const criteria = Array.isArray(raw.criteria)
        ? raw.criteria.map((entry) => (typeof entry === "string" ? entry : String(entry ?? "")))
        : raw.criteria;
      out[id] = { type: "score", instructions, criteria };
    } else {
      out[id] = { ...raw, instructions };
    }
  }
  return out;
}

/**
 * OpenRouter's Decisions envelope is { error: { code, message } }. For an
 * oversized Jev request, live responses wrap TypeSafe's typed body inside the
 * message as `HTTP 400: {"detail":{"error_type":"max_tokens_exceeded"}}`.
 * Expose only allow-listed constants; never return upstream text.
 */
function safeOpenRouterErrorType(bodyText: string): string | undefined {
  try {
    const body = JSON.parse(bodyText);
    const message = body?.error?.message;
    if (typeof message !== "string") return undefined;
    const wrapped = message.match(/^HTTP 4\d\d: (\{.*\})$/s);
    if (!wrapped) return undefined;
    const code = JSON.parse(wrapped[1])?.detail?.error_type;
    return typeof code === "string" && SAFE_OPENROUTER_ERROR_TYPES.has(code) ? code : undefined;
  } catch {
    return undefined;
  }
}

/** Only the not-processed status allowlist is retried, and 5xx means 500-599: out-of-range statuses are protocol noise, not retry signals. */
const isRetryableStatus = (status: number) => status === 408 || status === 409 || status === 429 || (status >= 500 && status <= 599);

interface Deadline {
  signal: AbortSignal;
  timedOut: () => boolean;
  dispose: () => void;
}

/** One deadline covering every attempt; expiry and caller aborts never retry. */
function deadlineSignal(signal: AbortSignal | undefined, timeoutMs: number): Deadline {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const relay = () => controller.abort(signal?.reason);
  if (signal) {
    if (signal.aborted) relay();
    else signal.addEventListener("abort", relay, { once: true });
  }
  return {
    signal: controller.signal,
    timedOut: () => timedOut,
    dispose: () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", relay);
    },
  };
}

/** Jittered exponential backoff: 50-100% of the doubling delay, capped. */
function retryDelayMs(attempt: number): number {
  const exp = Math.min(BASE_RETRY_DELAY_MS * 2 ** (attempt - 1), MAX_RETRY_DELAY_MS);
  return exp * (0.5 + Math.random() * 0.5);
}

/**
 * Backoff that ends early, without another attempt, when the deadline expires
 * or the caller aborts: the whole-request deadline must not be overrun by a
 * sleep, and a cancelled call must surface promptly.
 */
function backoffDelay(attempt: number, deadline: Deadline): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup();
      resolve();
    }, retryDelayMs(attempt));
    const onAbort = () => {
      cleanup();
      reject(
        deadline.timedOut()
          ? new Error(`Jev request exceeded the ${REQUEST_TIMEOUT_MS}ms deadline.`)
          : deadline.signal.reason,
      );
    };
    const cleanup = () => {
      clearTimeout(timer);
      deadline.signal.removeEventListener("abort", onAbort);
    };
    if (deadline.signal.aborted) onAbort();
    else deadline.signal.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * Read a response body with the byte ceiling enforced while streaming.
 * Content-Length is advisory (and absent for chunked responses), so the
 * limit is enforced on the bytes actually read, on success and error paths
 * alike. Never retried: an oversized body is a protocol violation.
 */
async function readBodyBounded(response: Response, deadline: Deadline): Promise<string> {
  const reader = response.body?.getReader();
  if (!reader) return "";
  const chunks: Uint8Array[] = [];
  let total = 0;
  // One abort watcher for the whole read, not one per chunk: a fragmented body
  // must not accumulate listeners and closures while it streams.
  let onAbort!: () => void;
  const aborted = new Promise<never>((_, reject) => {
    onAbort = () =>
      reject(
        deadline.timedOut()
          ? new Error(`Jev request exceeded the ${REQUEST_TIMEOUT_MS}ms deadline while reading the response.`)
          : deadline.signal.reason,
      );
  });
  aborted.catch(() => {}); // stays handled when a read wins every race
  deadline.signal.addEventListener("abort", onAbort, { once: true });
  try {
    if (deadline.signal.aborted) onAbort();
    for (;;) {
      const { done, value } = await Promise.race([reader.read(), aborted]);
      if (done) break;
      total += value.byteLength;
      if (total > MAX_RESPONSE_BYTES) {
        throw new Error(`Response exceeded ${MAX_RESPONSE_BYTES} bytes after reading ${total}; aborting the read.`);
      }
      chunks.push(value);
    }
  } catch (error) {
    // The body reader races our watcher to the same abort and may reject
    // first with its raw AbortError; translate, exactly as the fetch catch
    // does, so deadline expiry reads as a deadline, not "operation aborted".
    if (deadline.signal.aborted) {
      throw deadline.timedOut()
        ? new Error(`Jev request exceeded the ${REQUEST_TIMEOUT_MS}ms deadline while reading the response.`)
        : deadline.signal.reason;
    }
    throw error;
  } finally {
    deadline.signal.removeEventListener("abort", onAbort);
    // On the success path the reader is already done; on throw this releases
    // the connection instead of leaking it.
    await reader.cancel().catch(() => {});
    try {
      reader.releaseLock();
    } catch {
      // Already released by cancel on some runtimes.
    }
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder().decode(bytes);
}

/**
 * Appends a fixed path to a configured API root. Both defaults carry path
 * prefixes, not just origins, and a conventional trailing slash must not
 * produce a "//path" request.
 */
function apiUrl(root: string, path: string): string {
  return `${root.replace(/\/+$/, "")}${path}`;
}

/**
 * Fetch with bounded, jittered retries on the 408/409/429/5xx allowlist only.
 * A status cannot prove the request was not processed, but that allowlist is
 * the standard not-processed signal set and the only retry trigger. Ambiguous
 * network-level failures (connection reset mid-response, TLS errors) never
 * retry: without an idempotency key, re-sending after an ambiguous failure can
 * double-process a paid call. Caller aborts and deadline expiry surface
 * immediately, cutting any backoff short, and never retry.
 */
async function fetchWithResilience(url: string, init: RequestInit, deadline: Deadline): Promise<Response> {
  for (let attempt = 1; ; attempt++) {
    let response: Response;
    try {
      response = await fetch(url, { ...init, signal: deadline.signal });
    } catch (error) {
      if (deadline.timedOut()) {
        throw new Error(`Jev request exceeded the ${REQUEST_TIMEOUT_MS}ms deadline.`);
      }
      throw error; // caller cancellation or network failure: never re-sent
    }
    if (isRetryableStatus(response.status) && attempt < MAX_ATTEMPTS) {
      // Drain and release the connection before backing off.
      await response.body?.cancel().catch(() => {});
      await backoffDelay(attempt, deadline);
      continue;
    }
    return response;
  }
}

function resolve(env: NodeJS.ProcessEnv): JevProvider {
  const explicit = (env.JEV_PROVIDER ?? "auto").toLowerCase();
  const hasGemini = Boolean(env.GEMINI_API_KEY || env.GOOGLE_API_KEY);
  const hasCompatible = Boolean(env.JEV_API_KEY && env.JEV_API_BASE_URL);
  if (explicit === "gemini") {
    if (!hasGemini) {
      throw new Error(
        "JEV_PROVIDER=gemini but GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. " +
          `JEV_MCP_MODEL is optional and defaults to ${DEFAULT_GEMINI_MODEL}.`,
      );
    }
    return "gemini";
  }
  if (explicit === "compatible") {
    const missing = ["JEV_API_KEY", "JEV_API_BASE_URL"].filter((name) => !env[name]);
    if (missing.length > 0) {
      throw new Error(
        `JEV_PROVIDER=compatible but ${missing.join(" and ")} ${missing.length > 1 ? "are" : "is"} not set. ` +
          "JEV_MCP_MODEL is optional and defaults to jev-latest.",
      );
    }
    return "compatible";
  }
  // The published package owns the four built-in TypeSafe-Jev credential rules
  // and order. Gemini and compatible follow only when none of those four are set.
  const hasBuiltin = Boolean(
    env.TYPESAFE_API_KEY ||
      /^sk-or-/.test(env.OPENROUTER_API_KEY ?? "") ||
      ((env.JEV_CLOUDFLARE_API_TOKEN || env.CLOUDFLARE_API_TOKEN) && env.CLOUDFLARE_ACCOUNT_ID) ||
      env.AI_GATEWAY_API_KEY,
  );
  if ((explicit === "auto" || explicit === "") && !hasBuiltin) {
    if (hasGemini) return "gemini";
    if (hasCompatible) return "compatible";
  }
  return resolveTransport({ ...env, JEV_PROVIDER: explicit || "auto" }).name as JevProvider;
}

export async function askJev(
  state: unknown,
  questions: Record<string, unknown>,
  model: string,
  signal?: AbortSignal,
): Promise<AskResult> {
  const provider = resolve(process.env);

  if (provider === "typesafe" || provider === "vercel") {
    // Keep the raw envelope for MCP's per-judgment invalid_response behavior:
    // the shared package rejects a whole batch if even one answer is invalid.
    const builtin = resolveTransport({ ...process.env, JEV_PROVIDER: process.env.JEV_PROVIDER || "auto" });
    let reply: JevTransportReply | undefined;
    const transport: JevTransport = {
      name: builtin.name,
      async ask(input) {
        try {
          return reply = await builtin.ask(input);
        } catch (error) {
          // Both published drivers throw on an invalid usage *container* before
          // the Result validator sees it. Preserve the validation path, not a
          // misleading request_failed network error. Never reinterpret other
          // transport exceptions (including cancellation).
          if (error instanceof Error && (error.message === "TypeSafe API invalid usage (response omitted)" ||
              error.message === "Vercel AI Gateway invalid usage (response omitted)")) {
            return reply = { answers: {}, usage: { input_tokens: 0, output_tokens: 0 }, model: input.model };
          }
          throw error;
        }
      },
    };
    const result = await ask({ state, questions, model, signal: signal ?? new AbortController().signal }, { transport });
    if (!result.ok) {
      if (result.code === "request_failed" || result.code === "rate_limited" || result.code === "unavailable") throw new Error(result.message);
      // Envelope validation cannot be projected to individual judgments.
      // A bad envelope invalidates the call's judgments, not the MCP call.
      if (!isRecord(reply?.answers) || result.code === "invalid_usage" || result.code === "invalid_model" ||
          !Number.isSafeInteger(reply.usage?.input_tokens) || reply.usage.input_tokens < 0 ||
          !Number.isSafeInteger(reply.usage?.output_tokens) || reply.usage.output_tokens < 0 ||
          typeof reply.model !== "string" || !reply.model.trim() ||
          (result.code === "answer_id_mismatch" && Object.keys(reply.answers).some((id) => !Object.hasOwn(questions, id)))) {
        return { answers: {}, usage: { input_tokens: 0, output_tokens: 0 }, provider, model };
      }
      const answers = Object.fromEntries(Object.entries(reply.answers).filter(([id, answer]) =>
        !isRecord(answer) || !Object.hasOwn(answer, "type") || answer.type === (questions[id] as { type?: unknown } | undefined)?.type,
      ));
      // Package rejection is batch-wide; the existing per-tool guards are
      // authoritative for answer validity, including optional score
      // distributions, fractional scores, and valid sibling judgments.
      return { answers, usage: reply.usage, provider, model: reply.model };
    }
    return { answers: result.answer, usage: result.usage, provider, model: result.model };
  }

  if (provider === "openrouter") {
    // The shared package owns Jev model names; provider-prefixed slugs (e.g.
    // google/gemini-decision-1) pass through. This local path keeps MCP's
    // deadline, retry, body-size, and base-URL behavior.
    const slug = openrouterModel(model);
    const deadline = deadlineSignal(signal, REQUEST_TIMEOUT_MS);
    try {
      const response = await fetchWithResilience(apiUrl(process.env.JEV_OPENROUTER_BASE_URL || "https://openrouter.ai/api", "/alpha/decisions"), {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
          "Content-Type": "application/json",
          "HTTP-Referer": REFERER,
          "X-Title": X_TITLE,
          "X-OpenRouter-Title": X_TITLE,
        },
        body: JSON.stringify({ model: slug, state, questions }),
      }, deadline);
      const bodyText = await readBodyBounded(response, deadline);
      if (!response.ok) {
        const code = safeOpenRouterErrorType(bodyText);
        throw new Error(`OpenRouter decisions API ${response.status}${code ? ` (${code})` : ""}`);
      }
      let body: any;
      try {
        body = JSON.parse(bodyText);
      } catch {
        // Node's parse errors quote the malformed input; a reflecting endpoint
        // must not leak even a snippet through a 200 body.
        throw new Error(`OpenRouter decisions API ${response.status} returned an unparseable response`);
      }
      if (isRecord(body) && Object.hasOwn(body, "model") && (typeof body.model !== "string" || !body.model.trim())) {
        throw new Error(`OpenRouter decisions API ${response.status} returned an invalid model`);
      }
      return {
        answers: body.answers ?? {},
        // The decisions endpoint does not document a usage block; tolerate absence.
        usage: { input_tokens: body.usage?.input_tokens ?? 0, output_tokens: body.usage?.output_tokens ?? 0 },
        provider,
        model: isRecord(body) && Object.hasOwn(body, "model") ? body.model as string : slug,
      };
    } finally {
      deadline.dispose();
    }
  }

  if (provider === "gemini") {
    const slug = geminiModel(model);
    const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || "";
    const baseUrl = process.env.JEV_GEMINI_BASE_URL || "https://generativelanguage.googleapis.com";
    const deadline = deadlineSignal(signal, REQUEST_TIMEOUT_MS);
    try {
      const response = await fetchWithResilience(
        apiUrl(baseUrl, "/v1beta/decisions"),
        {
          method: "POST",
          headers: {
            "x-goog-api-key": apiKey,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: slug,
            input: toGeminiInput(state),
            questions: toGeminiQuestions(questions),
          }),
        },
        deadline,
      );
      const bodyText = await readBodyBounded(response, deadline);
      if (!response.ok) {
        const code = safeGeminiErrorCode(bodyText);
        throw new Error(`Gemini decisions API ${response.status}${code ? ` (${code})` : ""}`);
      }
      let body: unknown;
      try {
        body = JSON.parse(bodyText);
      } catch {
        throw new Error(`Gemini decisions API ${response.status} returned an unparseable response`);
      }
      if (!isRecord(body)) {
        throw new Error(`Gemini decisions API ${response.status} returned an invalid response`);
      }
      if (isRecord(body.prompt_feedback) && typeof body.prompt_feedback.block_reason === "string" && body.prompt_feedback.block_reason) {
        const reason = SAFE_GEMINI_BLOCK_REASONS.has(body.prompt_feedback.block_reason)
          ? body.prompt_feedback.block_reason
          : undefined;
        throw new Error(`Gemini decisions API blocked input${reason ? ` (${reason})` : ""}`);
      }
      const rawUsage = isRecord(body.usage) ? body.usage : null;
      const tokenCount = (v: unknown): number =>
        typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : 0;
      const returnedModel =
        typeof body.model === "string" && body.model.trim() ? body.model : slug;
      return {
        answers: isRecord(body.answers) ? adaptGeminiAnswers(body.answers) : {},
        usage: {
          input_tokens: tokenCount(rawUsage?.total_input_tokens ?? rawUsage?.input_tokens),
          output_tokens: tokenCount(rawUsage?.total_output_tokens ?? rawUsage?.output_tokens),
        },
        provider,
        model: returnedModel,
      };
    } finally {
      deadline.dispose();
    }
  }

  if (provider === "compatible") {
    const deadline = deadlineSignal(signal, REQUEST_TIMEOUT_MS);
    try {
      const response = await fetchWithResilience(process.env.JEV_API_BASE_URL!, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.JEV_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model, state, questions }),
      }, deadline);
      const bodyText = await readBodyBounded(response, deadline);
      if (!response.ok) {
        // Client-visible errors stay fixed-string: provider name and numeric
        // status only — a reflecting proxy can echo nothing back through them.
        throw new Error(`Jev-compatible endpoint ${response.status}`);
      }
      let body: unknown;
      try {
        body = JSON.parse(bodyText);
      } catch {
        body = null; // parse failures never retry; surface as invalid response
      }
        const invalid = (why: string) => new Error(`Jev-compatible endpoint returned an invalid response: ${why}`);
        if (!isRecord(body)) throw invalid("expected a JSON object.");
        if (!isRecord(body.answers)) throw invalid("expected an answers object.");
        // Envelope shape is validated here; per-question answer validity is the
        // tools' job. Each tool fails closed under its invalid_response contract,
        // so a missing or malformed answer can never reach tool-level defaults.
        let inputTokens = 0;
        let outputTokens = 0;
        if (body.usage !== undefined && body.usage !== null) {
      const tokenCount = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0;
      if (!isRecord(body.usage) || !tokenCount(body.usage.input_tokens) || !tokenCount(body.usage.output_tokens)) {
        throw invalid("usage must report finite non-negative input_tokens and output_tokens.");
      }
      inputTokens = body.usage.input_tokens;
      outputTokens = body.usage.output_tokens;
        }
        if (body.model !== undefined && body.model !== null && typeof body.model !== "string") {
      throw invalid("model must be absent or a string.");
        }
      return {
        answers: body.answers,
        usage: { input_tokens: inputTokens, output_tokens: outputTokens },
        provider,
        model: typeof body.model === "string" ? body.model : model,
      };
    } finally {
      deadline.dispose();
    }
  }

  // Cloudflare Workers AI wraps the same contract in {model, input} and the
  // v4 {result, success} envelope. Single alias; no version pinning.
  const cfSlug = model.startsWith("typesafe/") ? model : `typesafe/${model === "jev-latest" ? "jev" : model}`;
  const cfToken = process.env.JEV_CLOUDFLARE_API_TOKEN || process.env.CLOUDFLARE_API_TOKEN || "";
  const cfBase = process.env.JEV_CLOUDFLARE_BASE_URL || "https://api.cloudflare.com/client/v4";
  const cfDeadline = deadlineSignal(signal, REQUEST_TIMEOUT_MS);
  let cfBody: Record<string, any>;
  let cfStatus = 0;
  try {
    const cfResponse = await fetchWithResilience(apiUrl(cfBase, `/accounts/${process.env.CLOUDFLARE_ACCOUNT_ID}/ai/run`), {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.JEV_CLOUDFLARE_API_TOKEN || process.env.CLOUDFLARE_API_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model: cfSlug, input: { state, questions } }),
    }, cfDeadline);
    cfStatus = cfResponse.status;
    const bodyText = await readBodyBounded(cfResponse, cfDeadline);
    cfBody = JSON.parse(cfStatus >= 400 ? redactSecret(bodyText, cfToken) : bodyText);
  } catch (error) {
    if (error instanceof SyntaxError) {
      cfBody = {} as Record<string, any>; // parse failures never retry; surface as invalid response
    } else {
      throw error;
    }
  } finally {
    cfDeadline.dispose();
  }
  // Fixed-string formatter for every Cloudflare error path (HTTP status,
  // success:false, non-Completed state): upstream body and state text never
  // reach MCP-visible error messages, so a reflecting endpoint can echo
  // nothing — not even unredacted — back through them.
  const cfError = () => new Error(`Cloudflare AI run ${cfStatus}${cfStatus < 400 ? " did not complete" : ""}`);
  if (cfStatus >= 400 || cfBody.success === false) {
    throw cfError();
  }
  // The v4 envelope double-nests: body.result.result holds the model output.
  const cfOuter = cfBody.result;
  if (cfOuter && typeof cfOuter.state === "string" && cfOuter.state !== "Completed") {
    throw cfError();
  }
  const cfPayload = cfOuter?.result ?? cfOuter ?? cfBody;
  return {
    answers: cfPayload.answers ?? {},
    usage: { input_tokens: cfPayload.usage?.input_tokens ?? 0, output_tokens: cfPayload.usage?.output_tokens ?? 0 },
    provider,
    model: cfPayload.model ?? cfSlug,
  };
}
