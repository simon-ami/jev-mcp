import assert from "node:assert/strict";
import { test } from "node:test";
import { ask } from "@jkudish/jev-agent-tools";
import {
  adaptGeminiAnswers,
  choiceConfidence,
  geminiModel,
  openrouterModel,
  scoreConfidence,
  toGeminiInput,
  toGeminiQuestions,
} from "../dist/provider.js";

test("shared wire validator distinguishes a malformed judgment from a failed request", async () => {
  const input = {
    state: "example",
    questions: { decision: { type: "choice", criteria: { yes: "yes", no: "no" } } },
    model: "jev-latest",
    signal: new AbortController().signal,
  };
  const invalid = await ask(input, { transport: {
    name: "fixture",
    async ask() {
      return { answers: { decision: { type: "choice", choice: "yes", probabilities: { yes: 0.1, no: 0.9 } } },
        usage: { input_tokens: 1, output_tokens: 1 }, model: "jev-latest" };
    },
  } });
  assert.equal(invalid.ok, false);
  assert.equal(invalid.code, "invalid_choice");

  const failed = await ask(input, { transport: { name: "fixture", async ask() { throw new Error("offline"); } } });
  assert.equal(failed.ok, false);
  assert.equal(failed.code, "request_failed");
});

test("geminiModel maps the default jev-latest alias to gemini-decision-1 and preserves explicit models", () => {
  assert.equal(geminiModel("jev-latest"), "gemini-decision-1");
  assert.equal(geminiModel(""), "gemini-decision-1");
  assert.equal(geminiModel("gemini-decision-1"), "gemini-decision-1");
  assert.equal(geminiModel("gemini-decision-2"), "gemini-decision-2");
});

test("openrouterModel preserves non-typesafe provider-prefixed models such as google/gemini-decision-1", () => {
  assert.equal(openrouterModel("jev-latest"), "~typesafe/jev-latest");
  assert.equal(openrouterModel("jev-1.13"), "typesafe/jev-1.13");
  assert.equal(openrouterModel("typesafe/jev-1.13"), "typesafe/jev-1.13");
  assert.equal(openrouterModel("google/gemini-decision-1"), "google/gemini-decision-1");
});

test("toGeminiInput serializes structured state objects and keeps non-empty strings", () => {
  assert.equal(toGeminiInput("plain text"), "plain text");
  assert.equal(toGeminiInput(""), '""');
  assert.equal(toGeminiInput({ claims: ["a"], evidence: "b" }), '{"claims":["a"],"evidence":"b"}');
  assert.equal(toGeminiInput(undefined), "null");
});

test("toGeminiQuestions adapts Jev noul, choice, score, and object instructions for Gemini Decisions API", () => {
  const translated = toGeminiQuestions({
    yn: {
      type: "noul",
      instructions: "Is it damaged?",
      criteria: { true: "Yes, broken", false: "No, intact" },
    },
    yn_already_gemini: {
      type: "noul",
      instructions: "Already Gemini?",
      criteria: { true_description: "T", false_description: "F" },
    },
    yn_bare: {
      type: "noul",
      instructions: "Bare noul",
    },
    cls: {
      type: "choice",
      instructions: { task: "Which class?", item: { id: "i0", text: "hello" } },
      criteria: { c0: null, c1: "explicit description" },
    },
    rubric: {
      type: "score",
      instructions: "Rate quality",
      criteria: ["Bad", "Okay", "Good"],
    },
  });

  assert.deepEqual(translated, {
    yn: {
      type: "noul",
      instructions: "Is it damaged?",
      criteria: { true_description: "Yes, broken", false_description: "No, intact" },
    },
    yn_already_gemini: {
      type: "noul",
      instructions: "Already Gemini?",
      criteria: { true_description: "T", false_description: "F" },
    },
    yn_bare: {
      type: "noul",
      instructions: "Bare noul",
    },
    cls: {
      type: "choice",
      instructions: '{"task":"Which class?","item":{"id":"i0","text":"hello"}}',
      criteria: { c0: "", c1: "explicit description" },
    },
    rubric: {
      type: "score",
      instructions: "Rate quality",
      criteria: ["Bad", "Okay", "Good"],
    },
  });
});

test("choiceConfidence and scoreConfidence match TypeSafe's documented formulas", () => {
  // Examples directly from https://docs.typesafe.ai/confidence.md
  assert.ok(Math.abs(choiceConfidence({ a: 0.6, b: 0.3, c: 0.1 }) - 0.4) < 1e-12);
  assert.ok(Math.abs(choiceConfidence({ a: 0.75, b: 0.25 }) - 0.5) < 1e-12);
  assert.equal(choiceConfidence({ a: 1, b: 0, c: 0 }), 1);
  assert.equal(choiceConfidence({ a: 1 / 3, b: 1 / 3, c: 1 / 3 }), 0);

  assert.ok(Math.abs(scoreConfidence({ "0": 0, "1": 0.5, "2": 0.5 }) - 0.25) < 1e-12);
  assert.equal(scoreConfidence({ "0": 0.5, "1": 0, "2": 0.5 }), 0);
  assert.ok(Math.abs(scoreConfidence({ "0": 0, "1": 0.52, "2": 0.48 }) - 0.28) < 1e-12);
  assert.equal(scoreConfidence({ "0": 0, "1": 0, "2": 1 }), 1);

  const adapted = adaptGeminiAnswers({
    yn: { type: "noul", noul: 0.9, confidence: 0.9 },
    ch: { type: "choice", choice: "a", probabilities: { a: 0.6, b: 0.3, c: 0.1 }, confidence: 0.6 },
    sc: { type: "score", score: 1.48, probabilities: { "0": 0, "1": 0.52, "2": 0.48 }, confidence: 0.52 },
  });
  assert.equal(adapted.yn.confidence, 0.9);
  assert.ok(Math.abs(adapted.ch.confidence - 0.4) < 1e-12);
  assert.ok(Math.abs(adapted.sc.confidence - 0.28) < 1e-12);
});
