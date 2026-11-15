'use strict';

// The result protocol shared by runCommand and serve: what a handler may return, checked, and
// the JSON object that is written for evalproof to read. Nothing here touches stdin or stdout.

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function describe(value) {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

// A handler returns a string, or an object with a required string `output` and optional
// `retrieved`, `trace`, `tokens_in`, `tokens_out` and `cost_usd`. Returns the object to send, or
// throws a TypeError that says what is wrong. A string is sent as {output}, never as bare text,
// so an answer that happens to look like a JSON object is not mistaken for a result.
function normalize(result) {
  if (typeof result === 'string') return { output: result };
  if (!isObject(result) || !('output' in result)) {
    throw new TypeError(
      "a handler must return a string or an object with an 'output' string, got " +
        (isObject(result) ? 'an object without an output key' : describe(result)),
    );
  }
  if (typeof result.output !== 'string') {
    throw new TypeError(`'output' must be a string, got ${describe(result.output)}`);
  }
  const out = { output: result.output };
  if (result.retrieved !== undefined && result.retrieved !== null) {
    if (!Array.isArray(result.retrieved)) {
      throw new TypeError(`'retrieved' must be an array, got ${describe(result.retrieved)}`);
    }
    out.retrieved = result.retrieved.map(String);
  }
  if (result.trace !== undefined && result.trace !== null) {
    if (!isObject(result.trace)) {
      throw new TypeError(`'trace' must be an object, got ${describe(result.trace)}`);
    }
    out.trace = result.trace;
  }
  for (const key of ['tokens_in', 'tokens_out', 'cost_usd']) {
    const value = result[key];
    if (value === undefined || value === null) continue;
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new TypeError(`'${key}' must be a number, got ${describe(value)}`);
    }
    out[key] = value;
  }
  return out;
}

// What went wrong, for a log line: the stack when there is one, else the value as text.
function errorText(error) {
  if (error instanceof Error) {
    const stack = typeof error.stack === 'string' ? error.stack : '';
    return stack.includes(error.message) ? stack : `${error.message}\n${stack}`.trim();
  }
  return String(error);
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

// What runCommand writes to stderr: the stack, then the message again on the last line.
// evalproof shows the end of a failed command's stderr on the case, and a stack ends in frames,
// so without that line the case error would not say what went wrong.
function errorReport(error) {
  const text = errorText(error);
  return text.includes('\n') ? `${text}\nevalproof: ${errorMessage(error)}` : text;
}

module.exports = { isObject, describe, normalize, errorText, errorMessage, errorReport };
