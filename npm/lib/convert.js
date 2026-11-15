'use strict';

// Experimental converters from agent-SDK results to evalproof's {output, trace, tokens_in,
// tokens_out}. They mirror the Python adapters (src/evalproof/adapters): the same duck-typed
// reads, the same pairing of tool calls with their results, and the same trace shape, so a
// trace from JavaScript and one from Python for the same run read the same.

const warned = new Set();

function warnOnce(name) {
  if (warned.has(name)) return;
  warned.add(name);
  process.stderr.write(
    `evalproof: ${name} is experimental: its input shapes and trace mapping may change in a minor release\n`,
  );
}

// ---- duck-typed reads --------------------------------------------------------------------

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

// The first of `names` that is set (not null or undefined) on `obj`.
function get(obj, ...names) {
  if (obj === null || obj === undefined) return undefined;
  for (const name of names) {
    const value = obj[name];
    if (value !== null && value !== undefined) return value;
  }
  return undefined;
}

// A token count as an integer; anything that is not a finite number is 0.
function count(value) {
  return typeof value === 'number' && Number.isFinite(value) ? Math.trunc(value) : 0;
}

// The text of a message content: a string, or the text blocks of a list of blocks.
function textOf(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  let out = '';
  for (const block of content) {
    if (typeof block === 'string') {
      out += block;
      continue;
    }
    const text = get(block, 'text');
    const type = get(block, 'type') ?? 'text';
    if (typeof text === 'string' && (type === 'text' || type === 'output_text')) out += text;
  }
  return out;
}

// `value` as something JSON can hold: JSON types, or a string for anything else.
function jsonable(value, seen = new WeakSet()) {
  if (value === null || value === undefined) return null;
  switch (typeof value) {
    case 'string':
    case 'boolean':
      return value;
    case 'number':
      return Number.isFinite(value) ? value : String(value);
    case 'bigint':
    case 'symbol':
    case 'function':
      return String(value);
    default:
  }
  if (value instanceof Error) return value.message;
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? null : value.toISOString();
  if (seen.has(value)) return '[Circular]';
  seen.add(value);
  try {
    if (Array.isArray(value) || value instanceof Set) {
      return [...value].map((item) => jsonable(item, seen));
    }
    if (value instanceof Map) {
      return Object.fromEntries([...value].map(([k, v]) => [String(k), jsonable(v, seen)]));
    }
    if (typeof value.toJSON === 'function') {
      try {
        return jsonable(value.toJSON(), seen);
      } catch {
        // an object that cannot dump itself falls through to its own properties
      }
    }
    return Object.fromEntries(
      Object.entries(value).map(([k, v]) => [k, jsonable(v, seen)]),
    );
  } finally {
    seen.delete(value);
  }
}

// Tool arguments as an object: an object, a JSON object in a string, or {input: value}.
function argsOf(value) {
  if (value === null || value === undefined) return {};
  if (typeof value === 'string') {
    if (!value.trim()) return {};
    let parsed;
    try {
      parsed = JSON.parse(value);
    } catch {
      return { input: value };
    }
    return isObject(parsed) ? jsonable(parsed) : { input: parsed };
  }
  if (isObject(value)) return jsonable(value);
  return { input: jsonable(value) };
}

// A tool failure as text: a string, an Error's message, or the JSON of anything else.
function failureText(error) {
  if (typeof error === 'string' && error) return error;
  if (error instanceof Error && error.message) return error.message;
  if (error !== null && error !== undefined && typeof error !== 'string') {
    const text = JSON.stringify(jsonable(error));
    if (text && text !== '{}' && text !== 'null') return text;
  }
  return 'tool error';
}

// ---- trace pieces ------------------------------------------------------------------------

// Tool calls in the order they were made, each result matched to its call. A result carrying an
// id goes to the unanswered call with that id. A result with no id, or whose id matches no call
// that has one, goes to the first unanswered call with no id and the same name (any name, when
// the result has none). A result that matches nothing is kept as a call of its own.
class Ledger {
  constructor() {
    this.calls = [];
    this.answered = new Set();
  }

  call(name, args, id, turn) {
    const entry = {
      name: name !== null && name !== undefined ? String(name) : 'unknown',
      args: argsOf(args),
      result: null,
      ok: true,
      error: null,
      id: id !== null && id !== undefined ? String(id) : null,
      turn,
      latency_ms: null,
    };
    this.calls.push(entry);
    return entry;
  }

  find(id, name) {
    if (id !== null) {
      for (let i = 0; i < this.calls.length; i += 1) {
        if (!this.answered.has(i) && this.calls[i].id === id) return i;
      }
    }
    for (let i = 0; i < this.calls.length; i += 1) {
      if (this.answered.has(i) || this.calls[i].id !== null) continue;
      if (name === null || this.calls[i].name === name) return i;
    }
    return null;
  }

  result(id, name, result, { ok = true, error = null, turn = 0 } = {}) {
    const key = id !== null && id !== undefined ? String(id) : null;
    const label = name !== null && name !== undefined ? String(name) : null;
    let index = this.find(key, label);
    if (index === null) {
      this.calls.push({
        name: label || 'unknown',
        args: {},
        result: null,
        ok: true,
        error: null,
        id: key,
        turn,
        latency_ms: null,
      });
      index = this.calls.length - 1;
    }
    this.answered.add(index);
    const entry = this.calls[index];
    entry.result = jsonable(result);
    entry.ok = ok;
    entry.error = ok ? null : error;
    return entry;
  }
}

// Usage per model, in the order the models first appear.
class Usage {
  constructor() {
    this.byModel = new Map();
  }

  add(model, tokensIn, tokensOut, cached) {
    const key = typeof model === 'string' ? model : null;
    if (!this.byModel.has(key)) {
      this.byModel.set(key, {
        role: 'agent',
        model: key,
        tokens_in: 0,
        tokens_out: 0,
        cached_tokens: 0,
        cost_usd: null,
      });
    }
    const entry = this.byModel.get(key);
    entry.tokens_in += count(tokensIn);
    entry.tokens_out += count(tokensOut);
    entry.cached_tokens += count(cached);
  }

  list() {
    return [...this.byModel.values()];
  }
}

// The reply per turn, as the Python adapters do: up to the last turn that has a reply.
function responsesOf(replies) {
  const turns = [...replies.keys()];
  if (turns.length === 0) return [];
  const last = Math.max(...turns);
  return Array.from({ length: last + 1 }, (_, i) => replies.get(i) ?? '');
}

function finishResult(replies, ledger, usage, rounds) {
  const responses = responsesOf(replies);
  const list = usage.list();
  return {
    output: responses.length ? responses[responses.length - 1] : '',
    trace: {
      responses,
      tool_calls: ledger.calls,
      rounds: rounds || null,
      usage: list,
      meta: {},
    },
    tokens_in: list.reduce((sum, u) => sum + u.tokens_in, 0),
    tokens_out: list.reduce((sum, u) => sum + u.tokens_out, 0),
  };
}

// ---- Vercel AI SDK -----------------------------------------------------------------------

function vercelUsage(usage, model, into) {
  if (!usage) return;
  into.add(
    model,
    get(usage, 'inputTokens', 'promptTokens'),
    get(usage, 'outputTokens', 'completionTokens'),
    get(get(usage, 'inputTokenDetails'), 'cacheReadTokens') ?? get(usage, 'cachedInputTokens'),
  );
}

// One generateText-style result is one user turn. Calls, results and usage come from its steps
// when it has them (every step, so a multi-step run is whole whichever SDK version made it),
// else from the result itself. The reply is the text of its last step that called no tool.
function vercelTurn(result, turn, ledger, usage, replies) {
  const hasSteps = Array.isArray(result.steps) && result.steps.length > 0;
  const steps = hasSteps ? result.steps : [result];
  let reply;
  let rounds = 0;
  let stepUsage = false;
  for (const step of steps) {
    rounds += 1;
    const content = Array.isArray(step.content) ? step.content : [];
    let calls = Array.isArray(step.toolCalls) ? step.toolCalls : null;
    if (calls === null) calls = content.filter((part) => get(part, 'type') === 'tool-call');
    for (const call of calls) {
      ledger.call(
        get(call, 'toolName', 'name'),
        get(call, 'input', 'args'),
        get(call, 'toolCallId', 'id'),
        turn,
      );
    }
    let results = Array.isArray(step.toolResults) ? step.toolResults : null;
    if (results === null) results = content.filter((part) => get(part, 'type') === 'tool-result');
    for (const item of results) {
      ledger.result(
        get(item, 'toolCallId', 'id') ?? null,
        get(item, 'toolName', 'name') ?? null,
        get(item, 'output', 'result'),
        { turn },
      );
    }
    for (const part of content) {
      if (get(part, 'type') !== 'tool-error') continue;
      ledger.result(
        get(part, 'toolCallId') ?? null,
        get(part, 'toolName') ?? null,
        null,
        { ok: false, error: failureText(part.error), turn },
      );
    }
    const modelId = get(get(step, 'response'), 'modelId') ?? get(get(result, 'response'), 'modelId');
    if (hasSteps && get(step, 'usage')) {
      vercelUsage(step.usage, modelId, usage);
      stepUsage = true;
    }
    const text = typeof step.text === 'string' ? step.text : textOf(content);
    if (text && calls.length === 0) reply = text;
  }
  if (!stepUsage) {
    // No per-step usage: the result's own total. `totalUsage` is the sum over steps in
    // SDK versions where `usage` is the last step alone; `usage` is the total where it is not.
    vercelUsage(get(result, 'totalUsage', 'usage'), get(get(result, 'response'), 'modelId'), usage);
  }
  if (reply === undefined && typeof result.text === 'string' && result.text) reply = result.text;
  if (reply !== undefined) replies.set(turn, reply);
  return rounds;
}

// fromVercelAI(result): a generateText result (or an array of them, one per user turn) as
// {output, trace, tokens_in, tokens_out}. EXPERIMENTAL.
function fromVercelAI(result) {
  warnOnce('fromVercelAI');
  const results = Array.isArray(result) ? result : [result];
  const ledger = new Ledger();
  const usage = new Usage();
  const replies = new Map();
  let rounds = 0;
  results.forEach((item, turn) => {
    if (!isObject(item)) {
      throw new TypeError('fromVercelAI needs a generateText result (or an array of them)');
    }
    rounds += vercelTurn(item, turn, ledger, usage, replies);
  });
  return finishResult(replies, ledger, usage, rounds);
}

// ---- LangChain.js ------------------------------------------------------------------------

function messageKind(message) {
  let name = get(message, 'type', 'role');
  if (typeof name !== 'string') {
    for (const method of ['getType', '_getType']) {
      if (message && typeof message[method] === 'function') {
        try {
          name = message[method]();
          break;
        } catch {
          // try the next way of asking
        }
      }
    }
  }
  if (typeof name !== 'string') name = message?.constructor?.name ?? '';
  name = name.toLowerCase();
  if (name === 'user' || name.startsWith('human')) return 'human';
  if (name === 'assistant' || name.startsWith('ai')) return 'ai';
  if (name.startsWith('tool')) return 'tool';
  return name;
}

// A tool message's content as a result: the text when it is text, else the blocks.
function toolValue(content) {
  if (typeof content === 'string' || content === null || content === undefined) return content;
  const text = textOf(content);
  return text || jsonable(content);
}

// fromLangChain(messages): LangChain.js messages (or the {messages} state an agent returns) as
// {output, trace, tokens_in, tokens_out}. A human message starts the next turn. EXPERIMENTAL.
function fromLangChain(messages) {
  warnOnce('fromLangChain');
  const list = Array.isArray(messages) ? messages : get(messages, 'messages');
  if (!Array.isArray(list)) {
    throw new TypeError('fromLangChain needs an array of messages, or an object with a messages array');
  }
  const ledger = new Ledger();
  const usage = new Usage();
  const replies = new Map();
  let turn = 0;
  let active = false;
  let rounds = 0;
  for (const message of list) {
    const kind = messageKind(message);
    if (kind === 'human') {
      if (active) {
        turn += 1;
        active = false;
      }
    } else if (kind === 'ai') {
      active = true;
      rounds += 1;
      const calls = get(message, 'tool_calls') ?? [];
      for (const call of calls) {
        ledger.call(get(call, 'name'), get(call, 'args'), get(call, 'id'), turn);
      }
      const meta = get(message, 'usage_metadata');
      if (meta) {
        const metadata = get(message, 'response_metadata');
        usage.add(
          get(metadata, 'model_name', 'model'),
          get(meta, 'input_tokens'),
          get(meta, 'output_tokens'),
          get(get(meta, 'input_token_details'), 'cache_read'),
        );
      }
      const text = textOf(get(message, 'content'));
      if (text && calls.length === 0) replies.set(turn, text);
    } else if (kind === 'tool') {
      active = true;
      const failed = get(message, 'status') === 'error';
      const content = get(message, 'content');
      ledger.result(get(message, 'tool_call_id') ?? null, get(message, 'name') ?? null, toolValue(content), {
        ok: !failed,
        error: failed ? textOf(content) || 'tool error' : null,
        turn,
      });
    }
  }
  return finishResult(replies, ledger, usage, rounds);
}

module.exports = { fromVercelAI, fromLangChain };
