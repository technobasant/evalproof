import type { Server } from 'node:http';

/** The request evalproof sends for one case: on stdin to a `command:` target, as the POST body
 * to an `http(s)://` target. `id` and `input` are always there; the rest only when set. */
export interface EvalRequest {
  /** The case id. */
  id: string;
  /** The case input: a string, or an object such as `{ question: "..." }`. */
  input: string | Record<string, unknown>;
  /** The user turns that follow `input`, for a multi-turn case. */
  turns?: Array<string | Record<string, unknown>>;
  /** The case's starting state. */
  state?: Record<string, unknown>;
  /** The variant being run, when the suite compares variants. */
  variant?: string;
  /** The variant's vars. */
  vars?: Record<string, unknown>;
  /** The 0-based repeat, when each case runs more than once. */
  trial?: number;
}

/** One tool call an agent made. */
export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
  result: unknown;
  ok: boolean;
  error: string | null;
  id: string | null;
  /** The 0-based user turn the call belongs to. */
  turn: number;
  latency_ms: number | null;
}

/** Tokens (and cost, when known) of the model calls of one role. */
export interface ModelUsage {
  role: string;
  model: string | null;
  tokens_in: number;
  tokens_out: number;
  cached_tokens: number;
  cost_usd: number | null;
}

/** What an agent did, not only what it said. The same shape the Python adapters produce. */
export interface Trace {
  /** The reply per user turn; the last is the output. */
  responses: string[];
  tool_calls: ToolCall[];
  /** Model calls or generation rounds, when known. */
  rounds: number | null;
  usage: ModelUsage[];
  meta: Record<string, unknown>;
}

/** A trace a handler writes by hand: every field is optional. Same names as `Trace`. */
export interface TraceInput {
  responses?: string[];
  tool_calls?: Array<{
    name: string;
    args?: Record<string, unknown>;
    result?: unknown;
    ok?: boolean;
    error?: string | null;
    id?: string | null;
    turn?: number;
    latency_ms?: number | null;
  }>;
  rounds?: number | null;
  usage?: Array<{
    role?: string;
    model?: string | null;
    tokens_in?: number;
    tokens_out?: number;
    cached_tokens?: number;
    cost_usd?: number | null;
  }>;
  meta?: Record<string, unknown>;
}

/** What a handler returns besides a plain string. */
export interface HandlerResult {
  /** The answer. Required, and a string. */
  output: string;
  /** The retrieved passages, for retrieval scorers. */
  retrieved?: string[];
  trace?: TraceInput;
  /** Defaults to the sum over `trace.usage`. */
  tokens_in?: number;
  tokens_out?: number;
  cost_usd?: number;
}

/** A system under test: gets the request, returns a string or a `HandlerResult`. */
export type Handler = (
  request: EvalRequest,
) => string | HandlerResult | Promise<string | HandlerResult>;

/**
 * Run a `command:` target. Reads the JSON request from stdin, awaits `handler`, writes one JSON
 * result to stdout and exits 0. A thrown error, bad stdin or an invalid result prints its message
 * and stack to stderr and exits 1. `console.log`, `console.info` and `console.debug` write to
 * stderr while the handler runs, so they cannot corrupt the result. Never resolves: the process
 * exits once the result is written.
 */
export function runCommand(handler: Handler): Promise<never>;

export interface ServeOptions {
  /** Default 8787. `0` picks a free port. */
  port?: number;
  /** Default `"127.0.0.1"` (localhost only). */
  host?: string;
  /** The path that takes the POST. Default `"/"`. */
  path?: string;
}

/**
 * Serve the `http(s)://` target protocol with `node:http`: POST the request JSON, get the result
 * JSON. 400 for a body that is not a JSON object, 500 with `{"error"}` when the handler throws.
 * Returns the server, already listening. EXPERIMENTAL.
 */
export function serve(handler: Handler, options?: ServeOptions): Server;

/** What `fromVercelAI` and `fromLangChain` return: a `HandlerResult` with a full trace. */
export interface ConvertedResult {
  output: string;
  trace: Trace;
  tokens_in: number;
  tokens_out: number;
}

/** The parts of an AI SDK `generateText` result that `fromVercelAI` reads. Every field is
 * optional, so a result from any SDK version is accepted. */
export interface VercelAIResultLike {
  text?: string;
  content?: readonly unknown[];
  toolCalls?: readonly unknown[];
  toolResults?: readonly unknown[];
  /** Every step; calls, results and usage are read from here when it is set. */
  steps?: readonly unknown[];
  usage?: unknown;
  totalUsage?: unknown;
  response?: unknown;
}

/**
 * A Vercel AI SDK `generateText` result as `{ output, trace, tokens_in, tokens_out }`. Pass an
 * array of results, one per user turn, for a multi-turn case. Reads `steps[].toolCalls`,
 * `toolResults`, `content` parts of type `tool-error`, and `usage` (`inputTokens`/`outputTokens`,
 * or the older `promptTokens`/`completionTokens`). EXPERIMENTAL: prints one line to stderr the
 * first time it is called.
 */
export function fromVercelAI(
  result: VercelAIResultLike | readonly VercelAIResultLike[],
): ConvertedResult;

/** The state an agent run returns. */
export interface LangChainStateLike {
  messages: readonly unknown[];
}

/**
 * LangChain.js messages (objects, or `{ type, content, ... }` dicts), or the `{ messages }` an
 * agent run returns, as `{ output, trace, tokens_in, tokens_out }`. Reads `AIMessage.tool_calls`,
 * `ToolMessage.tool_call_id` and `status`, and `usage_metadata`. A human message starts the
 * next turn. EXPERIMENTAL: prints one line to stderr the first time it is called.
 */
export function fromLangChain(messages: readonly unknown[] | LangChainStateLike): ConvertedResult;
