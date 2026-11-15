'use strict';

// runCommand is run for real: each test writes a small script that calls it, starts that
// script as a child process with a request on stdin, and checks stdout, stderr and the exit code.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { after, test } = require('node:test');

const LIB = path.join(__dirname, '..', 'lib', 'index.js');
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'evalproof-run-command-'));
after(() => fs.rmSync(dir, { recursive: true, force: true }));

let counter = 0;
function run(handlerSource, stdin, options = {}) {
  counter += 1;
  const script = path.join(dir, `system-${counter}.js`);
  fs.writeFileSync(
    script,
    `const { runCommand } = require(${JSON.stringify(LIB)});\nrunCommand(${handlerSource});\n`,
  );
  const done = spawnSync(process.execPath, [script], {
    input: typeof stdin === 'string' ? stdin : JSON.stringify(stdin),
    encoding: 'utf8',
    timeout: 15000,
    maxBuffer: 64 * 1024 * 1024,
    ...options,
  });
  return done;
}

const REQUEST = { id: 'c1', input: { question: 'What is 2 + 2?' } };

test('a returned string is sent as {output} and the process exits 0', () => {
  const done = run('(req) => `echo: ${req.input.question}`', REQUEST);
  assert.equal(done.status, 0, done.stderr);
  assert.deepEqual(JSON.parse(done.stdout), { output: 'echo: What is 2 + 2?' });
  assert.equal(done.stderr, '');
});

test('the handler gets the whole request: id, input, turns, state, variant, vars, trial', () => {
  const full = {
    id: 'c9',
    input: 'hi',
    turns: ['and then?'],
    state: { cart: [1] },
    variant: 'b',
    vars: { tone: 'dry' },
    trial: 2,
  };
  const done = run('async (req) => ({ output: JSON.stringify(req) })', full);
  assert.equal(done.status, 0, done.stderr);
  assert.deepEqual(JSON.parse(JSON.parse(done.stdout).output), full);
});

test('an object result keeps output, retrieved, trace, tokens and cost', () => {
  const source = `async () => ({
    output: 'done',
    retrieved: ['doc-1', 2],
    trace: { responses: ['done'], tool_calls: [{ name: 'lookup', args: { id: 1 } }], rounds: 2 },
    tokens_in: 120,
    tokens_out: 30,
    cost_usd: 0.0021,
    ignored: 'not sent',
  })`;
  const done = run(source, REQUEST);
  assert.equal(done.status, 0, done.stderr);
  assert.deepEqual(JSON.parse(done.stdout), {
    output: 'done',
    retrieved: ['doc-1', '2'],
    trace: { responses: ['done'], tool_calls: [{ name: 'lookup', args: { id: 1 } }], rounds: 2 },
    tokens_in: 120,
    tokens_out: 30,
    cost_usd: 0.0021,
  });
});

test('an answer that looks like a JSON object is still a string output', () => {
  const done = run(`() => '{"output": "other", "x": 1}'`, REQUEST);
  assert.equal(done.status, 0, done.stderr);
  assert.deepEqual(JSON.parse(done.stdout), { output: '{"output": "other", "x": 1}' });
});

test('console.log, console.info and console.debug go to stderr while the handler runs', () => {
  const source = `async () => {
    console.log('from log');
    console.info('from info');
    console.debug('from debug');
    console.error('from error');
    return 'ok';
  }`;
  const done = run(source, REQUEST);
  assert.equal(done.status, 0, done.stderr);
  assert.deepEqual(JSON.parse(done.stdout), { output: 'ok' });
  for (const line of ['from log', 'from info', 'from debug', 'from error']) {
    assert.match(done.stderr, new RegExp(line));
  }
});

test('stdout holds exactly one JSON value, even from a chatty handler', () => {
  const done = run(`() => { console.log('a'); console.log({ b: 1 }); return 'x'; }`, REQUEST);
  assert.equal(done.status, 0, done.stderr);
  assert.equal(done.stdout, JSON.stringify({ output: 'x' }));
});

test('a thrown error prints its message and stack to stderr and exits 1', () => {
  const done = run(`() => { throw new Error('model unavailable'); }`, REQUEST);
  assert.equal(done.status, 1);
  assert.equal(done.stdout, '');
  assert.match(done.stderr, /model unavailable/);
  assert.match(done.stderr, /\n\s+at /, 'the stack');
  // evalproof shows the end of stderr on the case, so the message is the last line too
  assert.match(done.stderr.trim().split('\n').pop(), /^evalproof: model unavailable$/);
});

test('a rejected promise and a thrown non-error exit 1', () => {
  const rejected = run(`async () => { throw new Error('async boom'); }`, REQUEST);
  assert.equal(rejected.status, 1);
  assert.match(rejected.stderr, /async boom/);
  const text = run(`() => { throw 'just a string'; }`, REQUEST);
  assert.equal(text.status, 1);
  assert.match(text.stderr, /just a string/);
});

test('stdin that is not JSON exits 1 and the handler is not called', () => {
  const done = run(`() => { console.error('HANDLER RAN'); return 'x'; }`, 'not json {');
  assert.equal(done.status, 1);
  assert.equal(done.stdout, '');
  assert.match(done.stderr, /not valid JSON/);
  assert.doesNotMatch(done.stderr, /HANDLER RAN/);
});

test('stdin that is JSON but not an object exits 1', () => {
  for (const stdin of ['[1, 2]', '"text"', 'null', '42']) {
    const done = run(`() => 'x'`, stdin);
    assert.equal(done.status, 1, stdin);
    assert.match(done.stderr, /must be a JSON object/, stdin);
  }
});

test('output must be a string', () => {
  for (const value of ['42', 'null', '["a"]', '{ text: "a" }']) {
    const done = run(`() => ({ output: ${value} })`, REQUEST);
    assert.equal(done.status, 1, value);
    assert.equal(done.stdout, '');
    assert.match(done.stderr, /'output' must be a string/, value);
  }
});

test('a result without output, or that is not a string or object, exits 1', () => {
  for (const value of ['undefined', '42', '({ text: "a" })', '["a"]']) {
    const done = run(`() => ${value}`, REQUEST);
    assert.equal(done.status, 1, value);
    assert.match(done.stderr, /a handler must return a string or an object/, value);
  }
});

test('tokens and cost must be numbers', () => {
  for (const key of ['tokens_in', 'tokens_out', 'cost_usd']) {
    for (const value of ['"12"', 'true', '[]', 'NaN']) {
      const done = run(`() => ({ output: 'x', ${key}: ${value} })`, REQUEST);
      assert.equal(done.status, 1, `${key}=${value}`);
      assert.match(done.stderr, new RegExp(`'${key}' must be a number`), `${key}=${value}`);
    }
  }
});

test('retrieved must be an array and trace an object', () => {
  const retrieved = run(`() => ({ output: 'x', retrieved: 'doc' })`, REQUEST);
  assert.equal(retrieved.status, 1);
  assert.match(retrieved.stderr, /'retrieved' must be an array/);
  const trace = run(`() => ({ output: 'x', trace: 'steps' })`, REQUEST);
  assert.equal(trace.status, 1);
  assert.match(trace.stderr, /'trace' must be an object/);
});

test('null token fields are left out, as if unset', () => {
  const done = run(`() => ({ output: 'x', tokens_in: null, cost_usd: undefined })`, REQUEST);
  assert.equal(done.status, 0, done.stderr);
  assert.deepEqual(JSON.parse(done.stdout), { output: 'x' });
});

test('a handle the handler leaves open does not keep the process alive', () => {
  const done = run(`async () => { setInterval(() => {}, 1000); return 'ok'; }`, REQUEST, {
    timeout: 10000,
  });
  assert.equal(done.status, 0, done.stderr);
  assert.deepEqual(JSON.parse(done.stdout), { output: 'ok' });
});

test('a large output is written whole before the process exits', () => {
  const done = run(`() => 'x'.repeat(3 * 1024 * 1024)`, REQUEST);
  assert.equal(done.status, 0, done.stderr);
  assert.equal(JSON.parse(done.stdout).output.length, 3 * 1024 * 1024);
});

test('non-ASCII text survives in both directions', () => {
  const done = run(`(req) => req.input + ' ' + 'caf\\u00e9 \\u2713'`, { id: 'u', input: 'naïve' });
  assert.equal(done.status, 0, done.stderr);
  assert.equal(JSON.parse(done.stdout).output, 'naïve café ✓');
});

test('runCommand needs a function', async () => {
  const { runCommand } = require(LIB);
  await assert.rejects(() => runCommand(42), TypeError);
  await assert.rejects(() => runCommand(), /needs a handler function/);
});
