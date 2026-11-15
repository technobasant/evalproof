'use strict';

const http = require('node:http');
const { isObject, normalize, errorText, errorMessage } = require('./protocol');

const MAX_BODY_BYTES = 10 * 1024 * 1024;

function send(res, status, body, headers = {}) {
  const data = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(data),
    ...headers,
  });
  res.end(data);
}

// Reads the whole request body, or rejects with {status: 413} once it passes the limit.
function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(Object.assign(new Error('request body is too large'), { status: 413 }));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

// An http server for the `http(s)://` target: POST the request JSON, get the result JSON.
// 400 for a body that is not a JSON object, 500 with {"error"} when the handler throws or
// returns something invalid (the stack goes to stderr), 405 for a method other than POST and
// 404 for another path. Listens on localhost unless `host` says otherwise. Returns the server;
// it is already listening (use server.on('listening') or server.address() for the port).
function serve(handler, options = {}) {
  if (typeof handler !== 'function') {
    throw new TypeError('serve needs a handler function: serve(async (request) => ...)');
  }
  const { port = 8787, host = '127.0.0.1' } = options;
  let path = options.path === undefined ? '/' : options.path;
  if (!path.startsWith('/')) path = `/${path}`;

  const server = http.createServer(async (req, res) => {
    let pathname;
    try {
      pathname = new URL(req.url || '/', 'http://localhost').pathname;
    } catch {
      return send(res, 400, { error: 'bad request URL' });
    }
    if (pathname !== path) return send(res, 404, { error: `not found: ${pathname}` });
    if (req.method !== 'POST') {
      return send(res, 405, { error: 'use POST' }, { allow: 'POST' });
    }
    let request;
    try {
      const text = await readBody(req);
      try {
        request = JSON.parse(text);
      } catch (error) {
        return send(res, 400, { error: `request body is not valid JSON: ${error.message}` });
      }
      if (!isObject(request)) {
        return send(res, 400, { error: 'the request body must be a JSON object' });
      }
    } catch (error) {
      return send(res, error.status || 400, { error: errorMessage(error) });
    }
    try {
      return send(res, 200, normalize(await handler(request)));
    } catch (error) {
      process.stderr.write(`${errorText(error)}\n`);
      return send(res, 500, { error: errorMessage(error) });
    }
  });
  server.listen(port, host);
  return server;
}

module.exports = { serve };
