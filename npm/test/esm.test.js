'use strict';

// The library is CommonJS. These tests check that it is also usable the ESM way, with named
// imports, and through the package's `main`, as `import { runCommand } from "evalproof"` is.

const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { test } = require('node:test');

const NAMES = ['fromLangChain', 'fromVercelAI', 'runCommand', 'serve'];

test('the package main exports the four functions', () => {
  const lib = require(path.join(__dirname, '..'));
  assert.deepEqual(Object.keys(lib).sort(), NAMES);
  for (const name of NAMES) assert.equal(typeof lib[name], 'function', name);
});

test('package.json main and types point at files that exist', () => {
  const fs = require('node:fs');
  const dir = path.join(__dirname, '..');
  const pkg = JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8'));
  assert.ok(fs.existsSync(path.join(dir, pkg.main)), pkg.main);
  assert.ok(fs.existsSync(path.join(dir, pkg.types)), pkg.types);
  assert.ok(pkg.files.includes('lib'));
});

test('named ESM imports work: import { runCommand, serve, fromVercelAI, fromLangChain }', async () => {
  const url = pathToFileURL(path.join(__dirname, '..', 'lib', 'index.js')).href;
  const esm = await import(url);
  for (const name of NAMES) assert.equal(typeof esm[name], 'function', name);
  const { runCommand, serve, fromVercelAI, fromLangChain } = esm;
  assert.equal(runCommand, require('..').runCommand);
  assert.equal(serve, require('..').serve);
  assert.equal(fromVercelAI, require('..').fromVercelAI);
  assert.equal(fromLangChain, require('..').fromLangChain);
});

test('every .js file in lib/ has no dependency outside Node', () => {
  const fs = require('node:fs');
  const lib = path.join(__dirname, '..', 'lib');
  for (const file of fs.readdirSync(lib).filter((name) => name.endsWith('.js'))) {
    const source = fs.readFileSync(path.join(lib, file), 'utf8');
    for (const match of source.matchAll(/require\(['"]([^'"]+)['"]\)/g)) {
      const spec = match[1];
      assert.ok(spec.startsWith('./') || spec.startsWith('node:'), `${file} requires ${spec}`);
    }
  }
});
