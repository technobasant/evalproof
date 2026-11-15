'use strict';

// The converters read fake objects shaped like the SDK results. No SDK is installed or called.

const assert = require('node:assert/strict');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { before, test } = require('node:test');

const { fromVercelAI, fromLangChain } = require('..');

// Each converter warns once per process on stderr; keep that out of the test output.
before(() => {
  const original = process.stderr.write;
  process.stderr.write = (chunk, ...rest) =>
    String(chunk).startsWith('evalproof: from') ? true : original.call(process.stderr, chunk, ...rest);
});

const CALL_KEYS = ['args', 'error', 'id', 'latency_ms', 'name', 'ok', 'result', 'turn'];
const TRACE_KEYS = ['meta', 'responses', 'rounds', 'tool_calls', 'usage'];
const USAGE_KEYS = ['cached_tokens', 'cost_usd', 'model', 'role', 'tokens_in', 'tokens_out'];

// ---- Vercel AI SDK -----------------------------------------------------------------------

function weatherRun() {
  const call = { type: 'tool-call', toolCallId: 'c1', toolName: 'getWeather', input: { city: 'Paris' } };
  const output = { tempC: 20 };
  const step1 = {
    text: '',
    content: [call, { type: 'tool-result', toolCallId: 'c1', toolName: 'getWeather', input: call.input, output }],
    toolCalls: [call],
    toolResults: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'getWeather', input: call.input, output }],
    usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15, inputTokenDetails: { cacheReadTokens: 2 } },
    response: { modelId: 'model-a' },
  };
  const step2 = {
    text: 'It is 20 degrees in Paris.',
    content: [{ type: 'text', text: 'It is 20 degrees in Paris.' }],
    toolCalls: [],
    toolResults: [],
    usage: { inputTokens: 20, outputTokens: 7, totalTokens: 27 },
    response: { modelId: 'model-a' },
  };
  return { text: step2.text, steps: [step1, step2], usage: step2.usage, totalUsage: {} };
}

test('fromVercelAI: a two-step tool run becomes one reply, one call and summed usage', () => {
  const got = fromVercelAI(weatherRun());
  assert.equal(got.output, 'It is 20 degrees in Paris.');
  assert.deepEqual(got.trace.responses, ['It is 20 degrees in Paris.']);
  assert.equal(got.trace.rounds, 2);
  assert.equal(got.trace.tool_calls.length, 1);
  assert.deepEqual(got.trace.tool_calls[0], {
    name: 'getWeather',
    args: { city: 'Paris' },
    result: { tempC: 20 },
    ok: true,
    error: null,
    id: 'c1',
    turn: 0,
    latency_ms: null,
  });
  assert.deepEqual(got.trace.usage, [
    { role: 'agent', model: 'model-a', tokens_in: 30, tokens_out: 12, cached_tokens: 2, cost_usd: null },
  ]);
  assert.equal(got.tokens_in, 30);
  assert.equal(got.tokens_out, 12);
});

test('fromVercelAI: the shape is the Python trace shape, every field present', () => {
  const got = fromVercelAI(weatherRun());
  assert.deepEqual(Object.keys(got).sort(), ['output', 'tokens_in', 'tokens_out', 'trace']);
  assert.deepEqual(Object.keys(got.trace).sort(), TRACE_KEYS);
  assert.deepEqual(Object.keys(got.trace.tool_calls[0]).sort(), CALL_KEYS);
  assert.deepEqual(Object.keys(got.trace.usage[0]).sort(), USAGE_KEYS);
  assert.deepEqual(got.trace.meta, {});
  JSON.stringify(got); // plain JSON all through
});

test('fromVercelAI: a tool-error part fails its call, with an Error, a string or an object', () => {
  const failing = (error) => ({
    text: 'sorry',
    steps: [
      {
        text: '',
        content: [{ type: 'tool-error', toolCallId: 'e1', toolName: 'charge', input: { amount: 5 }, error }],
        toolCalls: [{ toolCallId: 'e1', toolName: 'charge', input: { amount: 5 } }],
        toolResults: [],
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      { text: 'sorry', content: [], toolCalls: [], toolResults: [], usage: { inputTokens: 1, outputTokens: 1 } },
    ],
  });
  for (const [error, expected] of [
    [new Error('card declined'), 'card declined'],
    ['card declined', 'card declined'],
    [{ code: 402 }, '{"code":402}'],
  ]) {
    const [call] = fromVercelAI(failing(error)).trace.tool_calls;
    assert.equal(call.ok, false);
    assert.equal(call.error, expected);
    assert.equal(call.result, null);
    assert.deepEqual(call.args, { amount: 5 });
  }
});

test('fromVercelAI: the older field names (args, result, promptTokens, completionTokens) work', () => {
  const got = fromVercelAI({
    text: 'done',
    steps: [
      {
        text: '',
        toolCalls: [{ toolCallId: 't', toolName: 'add', args: { a: 1, b: 2 } }],
        toolResults: [{ toolCallId: 't', toolName: 'add', args: { a: 1, b: 2 }, result: 3 }],
        usage: { promptTokens: 7, completionTokens: 3 },
      },
      { text: 'done', toolCalls: [], toolResults: [], usage: { promptTokens: 9, completionTokens: 1 } },
    ],
  });
  assert.deepEqual(got.trace.tool_calls[0].args, { a: 1, b: 2 });
  assert.equal(got.trace.tool_calls[0].result, 3);
  assert.equal(got.tokens_in, 16);
  assert.equal(got.tokens_out, 4);
  assert.equal(got.trace.usage[0].model, null);
});

test('fromVercelAI: a result with no steps is read as one step, using totalUsage over usage', () => {
  const got = fromVercelAI({
    text: 'hello',
    toolCalls: [{ toolCallId: 'x', toolName: 'ping', input: {} }],
    toolResults: [{ toolCallId: 'x', toolName: 'ping', output: 'pong' }],
    usage: { inputTokens: 1, outputTokens: 1 },
    totalUsage: { inputTokens: 4, outputTokens: 2 },
  });
  assert.equal(got.output, 'hello');
  assert.equal(got.trace.rounds, 1);
  assert.equal(got.tokens_in, 4);
  assert.equal(got.tokens_out, 2);
  assert.equal(got.trace.tool_calls[0].result, 'pong');
});

test('fromVercelAI: steps are summed, never the result usage on top of them', () => {
  const run = weatherRun();
  run.totalUsage = { inputTokens: 30, outputTokens: 12 };
  assert.equal(fromVercelAI(run).tokens_in, 30);
});

test('fromVercelAI: no usage at all gives zero tokens and no usage entries', () => {
  const got = fromVercelAI({ text: 'hi', steps: [{ text: 'hi' }] });
  assert.equal(got.tokens_in, 0);
  assert.equal(got.tokens_out, 0);
  assert.deepEqual(got.trace.usage, []);
  assert.deepEqual(got.trace.responses, ['hi']);
});

test('fromVercelAI: an array of results is one user turn each', () => {
  const turn = (text, name) => ({
    text,
    steps: [
      {
        text: '',
        toolCalls: name ? [{ toolCallId: name, toolName: name, input: {} }] : [],
        toolResults: name ? [{ toolCallId: name, toolName: name, output: 'ok' }] : [],
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      { text, toolCalls: [], toolResults: [], usage: { inputTokens: 1, outputTokens: 1 } },
    ],
  });
  const got = fromVercelAI([turn('first', 'a'), turn('second', null), turn('third', 'c')]);
  assert.deepEqual(got.trace.responses, ['first', 'second', 'third']);
  assert.equal(got.output, 'third');
  assert.deepEqual(
    got.trace.tool_calls.map((c) => [c.name, c.turn]),
    [['a', 0], ['c', 2]],
  );
  assert.equal(got.trace.rounds, 6);
});

test('fromVercelAI: a run that ends on a tool call has no reply', () => {
  const got = fromVercelAI({
    text: '',
    steps: [{ text: '', toolCalls: [{ toolCallId: 'z', toolName: 'loop', input: {} }], toolResults: [] }],
  });
  assert.equal(got.output, '');
  assert.deepEqual(got.trace.responses, []);
  assert.equal(got.trace.tool_calls[0].ok, true);
  assert.equal(got.trace.tool_calls[0].result, null);
});

test('fromVercelAI: tool args and results that are not plain JSON are made JSON-safe', () => {
  const cyclic = { name: 'c' };
  cyclic.self = cyclic;
  const got = fromVercelAI({
    text: 'x',
    steps: [
      {
        text: 'x',
        toolCalls: [{ toolCallId: 'j', toolName: 'odd', input: '{"a":1}' }],
        toolResults: [{ toolCallId: 'j', toolName: 'odd', output: { when: new Date(0), big: 10n, cyclic } }],
      },
    ],
  });
  assert.deepEqual(got.trace.tool_calls[0].args, { a: 1 });
  assert.deepEqual(got.trace.tool_calls[0].result, {
    when: '1970-01-01T00:00:00.000Z',
    big: '10',
    cyclic: { name: 'c', self: '[Circular]' },
  });
  JSON.stringify(got);
});

test('fromVercelAI: something that is not a result throws a TypeError', () => {
  assert.throws(() => fromVercelAI(null), TypeError);
  assert.throws(() => fromVercelAI('text'), TypeError);
  assert.throws(() => fromVercelAI([1]), TypeError);
});

// ---- LangChain.js ------------------------------------------------------------------------

const human = (content) => ({ type: 'human', content });
const ai = (content, extra = {}) => ({ type: 'ai', content, ...extra });
const tool = (tool_call_id, name, content, extra = {}) => ({
  type: 'tool',
  tool_call_id,
  name,
  content,
  ...extra,
});

function searchRun() {
  return [
    human('Find the capital of France'),
    ai('', {
      tool_calls: [{ name: 'search', args: { q: 'capital of France' }, id: 'call_1', type: 'tool_call' }],
      usage_metadata: {
        input_tokens: 12,
        output_tokens: 4,
        total_tokens: 16,
        input_token_details: { cache_read: 3 },
      },
      response_metadata: { model_name: 'model-a' },
    }),
    tool('call_1', 'search', 'Paris', { status: 'success' }),
    ai('The capital is Paris.', {
      usage_metadata: { input_tokens: 20, output_tokens: 6, total_tokens: 26 },
      response_metadata: { model_name: 'model-a' },
    }),
  ];
}

test('fromLangChain: a tool run becomes one reply, one paired call and summed usage', () => {
  const got = fromLangChain(searchRun());
  assert.equal(got.output, 'The capital is Paris.');
  assert.deepEqual(got.trace.responses, ['The capital is Paris.']);
  assert.equal(got.trace.rounds, 2);
  assert.deepEqual(got.trace.tool_calls, [
    {
      name: 'search',
      args: { q: 'capital of France' },
      result: 'Paris',
      ok: true,
      error: null,
      id: 'call_1',
      turn: 0,
      latency_ms: null,
    },
  ]);
  assert.deepEqual(got.trace.usage, [
    { role: 'agent', model: 'model-a', tokens_in: 32, tokens_out: 10, cached_tokens: 3, cost_usd: null },
  ]);
  assert.equal(got.tokens_in, 32);
  assert.equal(got.tokens_out, 10);
});

test('fromLangChain: the shape is the Python trace shape, every field present', () => {
  const got = fromLangChain(searchRun());
  assert.deepEqual(Object.keys(got).sort(), ['output', 'tokens_in', 'tokens_out', 'trace']);
  assert.deepEqual(Object.keys(got.trace).sort(), TRACE_KEYS);
  assert.deepEqual(Object.keys(got.trace.tool_calls[0]).sort(), CALL_KEYS);
  assert.deepEqual(Object.keys(got.trace.usage[0]).sort(), USAGE_KEYS);
});

test('fromLangChain: the { messages } state an agent returns is accepted', () => {
  const got = fromLangChain({ messages: searchRun() });
  assert.equal(got.output, 'The capital is Paris.');
});

test('fromLangChain: a ToolMessage with status "error" fails its call', () => {
  const got = fromLangChain([
    human('charge me'),
    ai('', { tool_calls: [{ name: 'charge', args: { amount: 5 }, id: 'k1' }] }),
    tool('k1', 'charge', 'card declined', { status: 'error' }),
    ai('That failed.'),
  ]);
  const [call] = got.trace.tool_calls;
  assert.equal(call.ok, false);
  assert.equal(call.error, 'card declined');
  assert.equal(got.output, 'That failed.');
});

test('fromLangChain: a failed ToolMessage with empty content still has an error text', () => {
  const [call] = fromLangChain([
    ai('', { tool_calls: [{ name: 'x', args: {}, id: 'k' }] }),
    tool('k', 'x', '', { status: 'error' }),
  ]).trace.tool_calls;
  assert.equal(call.ok, false);
  assert.equal(call.error, 'tool error');
});

test('fromLangChain: a human message after a reply starts the next turn', () => {
  const got = fromLangChain([
    human('one'),
    ai('', { tool_calls: [{ name: 'a', args: {}, id: '1' }] }),
    tool('1', 'a', 'r1'),
    ai('first answer'),
    human('two'),
    ai('', { tool_calls: [{ name: 'b', args: {}, id: '2' }] }),
    tool('2', 'b', 'r2'),
    ai('second answer'),
  ]);
  assert.deepEqual(got.trace.responses, ['first answer', 'second answer']);
  assert.deepEqual(
    got.trace.tool_calls.map((c) => [c.name, c.turn]),
    [['a', 0], ['b', 1]],
  );
  assert.equal(got.output, 'second answer');
});

test('fromLangChain: a list with no human message is turn 0 throughout', () => {
  const got = fromLangChain([ai('', { tool_calls: [{ name: 'a', args: {}, id: '1' }] }), tool('1', 'a', 'r'), ai('end')]);
  assert.deepEqual(got.trace.tool_calls.map((c) => c.turn), [0]);
  assert.deepEqual(got.trace.responses, ['end']);
});

test('fromLangChain: content blocks give their text blocks only', () => {
  const got = fromLangChain([
    ai([
      { type: 'text', text: 'Part one. ' },
      { type: 'text', text: 'Part two.' },
      { type: 'image_url', image_url: 'x' },
    ]),
  ]);
  assert.equal(got.output, 'Part one. Part two.');
  const [call] = fromLangChain([
    ai('', { tool_calls: [{ name: 'a', args: {}, id: '1' }] }),
    tool('1', 'a', [{ type: 'text', text: 'block result' }]),
  ]).trace.tool_calls;
  assert.equal(call.result, 'block result');
});

test('fromLangChain: an AI message with text and a tool call has no reply of its own', () => {
  const got = fromLangChain([
    ai('Let me check.', { tool_calls: [{ name: 'a', args: {}, id: '1' }] }),
    tool('1', 'a', 'r'),
  ]);
  assert.equal(got.output, '');
  assert.deepEqual(got.trace.responses, []);
});

test('fromLangChain: results pair by id, then by order and name, and a stray result is kept', () => {
  const got = fromLangChain([
    ai('', {
      tool_calls: [
        { name: 'a', args: {}, id: 'x' },
        { name: 'b', args: {}, id: 'y' },
      ],
    }),
    tool('y', 'b', 'for b'),
    tool('x', 'a', 'for a'),
    tool('zzz', 'ghost', 'nobody asked'),
  ]);
  assert.deepEqual(
    got.trace.tool_calls.map((c) => [c.name, c.result]),
    [['a', 'for a'], ['b', 'for b'], ['ghost', 'nobody asked']],
  );
});

test('fromLangChain: tool call args given as a JSON string are parsed', () => {
  const [call] = fromLangChain([ai('', { tool_calls: [{ name: 'a', args: '{"n":3}', id: '1' }] })]).trace.tool_calls;
  assert.deepEqual(call.args, { n: 3 });
});

test('fromLangChain: usage is grouped by model, in order of first use', () => {
  const got = fromLangChain([
    ai('a', { usage_metadata: { input_tokens: 1, output_tokens: 1 }, response_metadata: { model_name: 'm1' } }),
    ai('b', { usage_metadata: { input_tokens: 2, output_tokens: 2 }, response_metadata: { model_name: 'm2' } }),
    ai('c', { usage_metadata: { input_tokens: 3, output_tokens: 3 }, response_metadata: { model_name: 'm1' } }),
  ]);
  assert.deepEqual(
    got.trace.usage.map((u) => [u.model, u.tokens_in, u.tokens_out]),
    [['m1', 4, 4], ['m2', 2, 2]],
  );
  assert.equal(got.tokens_in, 6);
});

test('fromLangChain: class instances are read by getType() or by class name', () => {
  class HumanMessage { constructor(content) { this.content = content; } getType() { return 'human'; } }
  class AIMessage { constructor(content, tool_calls = []) { this.content = content; this.tool_calls = tool_calls; } }
  class ToolMessage { constructor(content, tool_call_id) { this.content = content; this.tool_call_id = tool_call_id; } }
  const got = fromLangChain([
    new HumanMessage('hi'),
    new AIMessage('', [{ name: 'a', args: {}, id: '1' }]),
    new ToolMessage('r', '1'),
    new AIMessage('bye'),
  ]);
  assert.equal(got.output, 'bye');
  assert.equal(got.trace.tool_calls[0].result, 'r');
});

test('fromLangChain: OpenAI-style roles (user, assistant) are read as human and ai', () => {
  const got = fromLangChain([{ role: 'user', content: 'q' }, { role: 'assistant', content: 'a' }]);
  assert.equal(got.output, 'a');
  assert.equal(got.trace.rounds, 1);
});

test('fromLangChain: an empty list is an empty result', () => {
  const got = fromLangChain([]);
  assert.equal(got.output, '');
  assert.equal(got.trace.rounds, null);
  assert.deepEqual(got.trace.tool_calls, []);
});

test('fromLangChain: something that is not messages throws a TypeError', () => {
  assert.throws(() => fromLangChain(null), TypeError);
  assert.throws(() => fromLangChain('hi'), TypeError);
  assert.throws(() => fromLangChain({ nope: [] }), TypeError);
});

// ---- both ----------------------------------------------------------------------------------

test('each converter warns once on stderr, on the first call only', () => {
  const lib = path.join(__dirname, '..', 'lib', 'index.js');
  const script = `
    const { fromVercelAI, fromLangChain } = require(${JSON.stringify(lib)});
    fromVercelAI({ text: 'a' }); fromVercelAI({ text: 'b' });
    fromLangChain([]); fromLangChain([]);
  `;
  const done = spawnSync(process.execPath, ['-e', script], { encoding: 'utf8' });
  assert.equal(done.status, 0, done.stderr);
  assert.equal(done.stdout, '');
  const lines = done.stderr.trim().split('\n');
  assert.equal(lines.length, 2);
  assert.match(lines[0], /^evalproof: fromVercelAI is experimental/);
  assert.match(lines[1], /^evalproof: fromLangChain is experimental/);
});
