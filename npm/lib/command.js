'use strict';

const { isObject, normalize, errorReport } = require('./protocol');

function readStdin(stream) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    stream.on('data', (chunk) => chunks.push(chunk));
    stream.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    stream.on('error', reject);
  });
}

// console.log, console.info and console.debug write to stdout, which belongs to the result.
// While the handler runs they write to stderr instead (console.error and console.warn already
// do). Returns the function that puts them back.
function redirectConsole() {
  const saved = { log: console.log, info: console.info, debug: console.debug };
  const toStderr = (...args) => console.error(...args);
  console.log = toStderr;
  console.info = toStderr;
  console.debug = toStderr;
  return () => {
    console.log = saved.log;
    console.info = saved.info;
    console.debug = saved.debug;
  };
}

function finish(stream, text, code) {
  stream.write(text, () => process.exit(code));
}

// Read the JSON request from stdin, call the handler with it, write one JSON result to stdout
// and exit 0. An error (bad JSON, a thrown handler, an invalid result) goes to stderr with its
// stack and exits 1, which evalproof records as an error on that case. Resolves never: the
// process exits once the result is flushed, so a handle the handler left open cannot hold it.
async function runCommand(handler) {
  if (typeof handler !== 'function') {
    throw new TypeError('runCommand needs a handler function: runCommand(async (request) => ...)');
  }
  let result;
  try {
    if (process.stdin.isTTY) {
      throw new Error('expected a JSON request on stdin, not a terminal');
    }
    const text = await readStdin(process.stdin);
    let request;
    try {
      request = JSON.parse(text);
    } catch (error) {
      throw new Error(`stdin is not valid JSON: ${error.message}`);
    }
    if (!isObject(request)) throw new Error('the request on stdin must be a JSON object');
    const restore = redirectConsole();
    try {
      result = normalize(await handler(request));
    } finally {
      restore();
    }
  } catch (error) {
    return finish(process.stderr, `${errorReport(error)}\n`, 1);
  }
  return finish(process.stdout, JSON.stringify(result), 0);
}

module.exports = { runCommand };
