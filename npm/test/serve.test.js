'use strict';

// serve is run for real on a free port of localhost, and called with fetch (Node 18 has it).

const assert = require('node:assert/strict');
const { after, test } = require('node:test');

const { serve } = require('..');

const servers = [];
async function start(handler, options = {}) {
  const server = serve(handler, { port: 0, ...options });
  servers.push(server);
  if (!server.listening) await new Promise((resolve) => server.once('listening', resolve));
  const { address, port } = server.address();
  return { server, address, url: `http://127.0.0.1:${port}` };
}
after(async () => {
  await Promise.all(
    servers.map(
      (server) =>
        new Promise((resolve) => {
          server.close(resolve);
          // fetch keeps connections alive; Node 18's close() would wait for them to time out
          if (typeof server.closeAllConnections === 'function') server.closeAllConnections();
        }),
    ),
  );
});

function post(url, body, headers = { 'content-type': 'application/json' }) {
  return fetch(url, {
    method: 'POST',
    headers,
    body: typeof body === 'string' ? body : JSON.stringify(body),
  });
}

// Run `fn` with process.stderr.write captured, so a logged stack does not clutter the test run.
async function captureStderr(fn) {
  const original = process.stderr.write;
  const lines = [];
  process.stderr.write = (chunk) => {
    lines.push(String(chunk));
    return true;
  };
  try {
    await fn();
  } finally {
    process.stderr.write = original;
  }
  return lines.join('');
}

test('serve returns a listening server bound to localhost by default', async () => {
  const { server, address } = await start(() => 'x');
  assert.equal(typeof server.close, 'function');
  assert.equal(address, '127.0.0.1');
});

test('POST / with a request returns the result as JSON', async () => {
  const { url } = await start(async (req) => ({
    output: `got ${req.id}`,
    retrieved: ['d1'],
    trace: { responses: ['got c1'] },
    tokens_in: 5,
    tokens_out: 2,
  }));
  const res = await post(url, { id: 'c1', input: 'hi' });
  assert.equal(res.status, 200);
  assert.match(res.headers.get('content-type'), /json/);
  assert.deepEqual(await res.json(), {
    output: 'got c1',
    retrieved: ['d1'],
    trace: { responses: ['got c1'] },
    tokens_in: 5,
    tokens_out: 2,
  });
});

test('a returned string is sent as {output}', async () => {
  const { url } = await start(() => 'plain answer');
  const res = await post(url, { id: 'c1', input: 'hi' });
  assert.equal(res.status, 200);
  assert.deepEqual(await res.json(), { output: 'plain answer' });
});

test('the handler gets the whole request body', async () => {
  const full = {
    id: 'c9',
    input: { question: 'q' },
    turns: ['next'],
    state: { a: 1 },
    variant: 'v',
    vars: { k: 'v' },
    trial: 1,
  };
  const { url } = await start((req) => ({ output: JSON.stringify(req) }));
  const res = await post(url, full);
  assert.deepEqual(JSON.parse((await res.json()).output), full);
});

test('a body that is not JSON is a 400 with an error message', async () => {
  let called = false;
  const { url } = await start(() => {
    called = true;
    return 'x';
  });
  const res = await post(url, '{not json');
  assert.equal(res.status, 400);
  assert.match((await res.json()).error, /not valid JSON/);
  assert.equal(called, false);
});

test('a body that is JSON but not an object is a 400', async () => {
  const { url } = await start(() => 'x');
  for (const body of ['[1]', '"text"', 'null']) {
    const res = await post(url, body);
    assert.equal(res.status, 400, body);
    assert.match((await res.json()).error, /must be a JSON object/, body);
  }
});

test('a handler that throws is a 500 with the message, and the stack goes to stderr', async () => {
  const { url } = await start(() => {
    throw new Error('model unavailable');
  });
  let res;
  const logged = await captureStderr(async () => {
    res = await post(url, { id: 'c1', input: 'hi' });
  });
  assert.equal(res.status, 500);
  assert.deepEqual(await res.json(), { error: 'model unavailable' });
  assert.match(logged, /model unavailable/);
  assert.match(logged, /\n\s+at /);
});

test('a rejected promise is a 500 too', async () => {
  const { url } = await start(async () => {
    throw new Error('async boom');
  });
  let res;
  await captureStderr(async () => {
    res = await post(url, { id: 'c1', input: 'hi' });
  });
  assert.equal(res.status, 500);
  assert.equal((await res.json()).error, 'async boom');
});

test('an invalid result (output not a string) is a 500, not a malformed 200', async () => {
  const { url } = await start(() => ({ output: 42 }));
  let res;
  await captureStderr(async () => {
    res = await post(url, { id: 'c1', input: 'hi' });
  });
  assert.equal(res.status, 500);
  assert.match((await res.json()).error, /'output' must be a string/);
});

test('only POST is accepted: GET is a 405 with Allow: POST', async () => {
  const { url } = await start(() => 'x');
  const res = await fetch(url);
  assert.equal(res.status, 405);
  assert.equal(res.headers.get('allow'), 'POST');
});

test('another path is a 404, and `path` moves the endpoint', async () => {
  const { url } = await start(() => 'at run', { path: '/run' });
  assert.equal((await post(url, { id: 'c1', input: 'hi' })).status, 404);
  const res = await post(`${url}/run`, { id: 'c1', input: 'hi' });
  assert.equal(res.status, 200);
  assert.deepEqual(await res.json(), { output: 'at run' });
});

test('a path without a leading slash is accepted, and a query string is ignored', async () => {
  const { url } = await start(() => 'ok', { path: 'agent' });
  const res = await post(`${url}/agent?x=1`, { id: 'c1', input: 'hi' });
  assert.equal(res.status, 200);
});

test('serve needs a function', () => {
  assert.throws(() => serve(), /needs a handler function/);
});

test('a body just over 10 MiB gets a 413 with an error body, not a reset connection', async () => {
  let called = false;
  const { url } = await start(() => {
    called = true;
    return 'x';
  });
  const body = JSON.stringify({ id: 'c1', input: 'x'.repeat(10 * 1024 * 1024) });
  const res = await post(url, body);
  assert.equal(res.status, 413);
  const payload = await res.json();
  assert.match(payload.error, /too large/);
  assert.equal(called, false);
  // the server still answers the next request
  const next = await post(url, { id: 'c2', input: 'hi' });
  assert.equal(next.status, 200);
});
