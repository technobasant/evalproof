#!/usr/bin/env node
'use strict';

// A launcher, not a port. evalproof is a Python program on PyPI; this file runs the PyPI
// release that has the same version as this npm package, with the first of these that works:
//
//   1. uvx   (uvx --from evalproof==VERSION evalproof ...)
//   2. pipx  (pipx run --spec evalproof==VERSION evalproof ...)
//   3. an `evalproof` on PATH that reports exactly VERSION
//
// EVALPROOF_LAUNCHER=uvx|pipx|path forces one of them. It has no dependencies and starts no
// shell on POSIX. On Windows it starts `.exe` files directly and `.cmd`/`.bat` files through
// cmd.exe only when every argument can be quoted for it safely.

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');

const VERSION = require('../package.json').version;
const METHODS = ['uvx', 'pipx', 'path'];
const UV_URL = 'https://docs.astral.sh/uv/getting-started/installation/';
// Set for the `path` method's child: an `evalproof` on PATH that is this launcher again
// (npm installs one) must not start another copy of itself.
const ACTIVE_ENV = 'EVALPROOF_LAUNCHER_ACTIVE';

class LauncherError extends Error {
  constructor(message, exitCode) {
    super(message);
    this.name = 'LauncherError';
    this.exitCode = exitCode;
  }
}

// --- finding executables ---------------------------------------------------------------

function getEnv(env, name, platform) {
  if (platform !== 'win32') return env[name];
  // Windows names are case-insensitive, but a copied environment object keeps one spelling.
  const key = Object.keys(env).find((k) => k.toUpperCase() === name.toUpperCase());
  return key === undefined ? undefined : env[key];
}

// The extensions a spawn can run directly or through cmd.exe, from PATHEXT, with the real
// programs (.exe, .com) first: uv and pipx install `.exe` files, and a `.cmd` needs cmd.exe.
function windowsExtensions(env) {
  const all = ['.exe', '.com', '.bat', '.cmd'];
  const raw = getEnv(env, 'PATHEXT', 'win32') || '.COM;.EXE;.BAT;.CMD';
  let listed = raw
    .split(';')
    .map((e) => e.trim().toLowerCase())
    .filter((e) => all.includes(e));
  if (listed.length === 0) listed = all;
  const binaries = listed.filter((e) => e === '.exe' || e === '.com');
  return [...binaries, ...listed.filter((e) => !binaries.includes(e))];
}

function isRunnableFile(candidate, platform) {
  try {
    if (!fs.statSync(candidate).isFile()) return false;
    if (platform !== 'win32') fs.accessSync(candidate, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

// The first match for `name` on PATH, or null. On Windows a bare `name` is never a match (an
// extensionless file there is a shell script that cannot be spawned); `name.exe` and the
// other PATHEXT forms are. An empty PATH entry (the current directory) is skipped.
function resolveCommand(name, options = {}) {
  const env = options.env || process.env;
  const platform = options.platform || process.platform;
  const isFile = options.isFile || isRunnableFile;
  const skip = options.skip || (() => false);
  const win = platform === 'win32';
  const join = win ? path.win32.join : path.posix.join;
  const dirs = (getEnv(env, 'PATH', platform) || '')
    .split(win ? ';' : ':')
    .map((d) => (win ? d.trim().replace(/^"(.*)"$/, '$1') : d))
    .filter(Boolean);
  const names = win ? windowsExtensions(env).map((ext) => name + ext) : [name];
  for (const dir of dirs) {
    for (const candidate of names) {
      const full = join(dir, candidate);
      if (isFile(full, platform) && !skip(full)) return full;
    }
  }
  return null;
}

function isThisScript(candidate) {
  try {
    return fs.realpathSync(candidate) === fs.realpathSync(__filename);
  } catch {
    return false;
  }
}

// --- starting a program ----------------------------------------------------------------

// What cmd.exe would act on inside double quotes: a quote ends them, `%` and `!` expand
// variables, `^ & | < >` act when a batch file re-expands an argument without its quotes,
// and a line break ends the command. Such an argument is refused, not escaped: cmd.exe has
// no escape that works in every case, and a `.cmd` shim is rare (uv and pipx ship `.exe`).
const CMD_UNSAFE = /[\0\r\n"%^!&|<>]/;

function quoteForCmd(arg) {
  const text = String(arg);
  if (CMD_UNSAFE.test(text)) {
    throw new LauncherError(
      `evalproof: cannot pass the argument ${JSON.stringify(text)} through a .cmd or .bat ` +
        'file: it holds a character that cmd.exe would act on. Install uv ' +
        `(${UV_URL}) or pipx, which start without cmd.exe, or run the Python evalproof directly.`,
      126,
    );
  }
  // Inside quotes a backslash matters only before the closing quote, so double those.
  let end = text.length;
  while (end > 0 && text[end - 1] === '\\') end -= 1;
  return `"${text}${text.slice(end)}"`;
}

// How to start `file` with `args`: a `.cmd`/`.bat` on Windows has to go through cmd.exe
// (Node refuses to start one otherwise) as one command line with every part quoted; anything
// else is started directly, with no shell.
function planSpawn(file, args, platform = process.platform) {
  if (platform === 'win32' && /\.(cmd|bat)$/i.test(file)) {
    return { shell: true, command: [file, ...args].map(quoteForCmd).join(' '), args: [] };
  }
  return { shell: false, command: file, args };
}

function spawnWith(fn, plan, options) {
  return plan.shell
    ? fn(plan.command, { ...options, shell: true })
    : fn(plan.command, plan.args, options);
}

function exitCodeFor(code, signal) {
  if (typeof code === 'number') return code;
  const number = signal ? os.constants.signals[signal] : undefined;
  return number ? 128 + number : 1;
}

// --- choosing a method -----------------------------------------------------------------

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// True when `output` names exactly `version`: "evalproof 1.0.0" does, "evalproof 1.0.01",
// "11.0.0" and "1.0.0rc1" do not.
function reportsVersion(output, version) {
  const edge = '[0-9A-Za-z.+_-]';
  return new RegExp(`(?:^|[^0-9A-Za-z.+_-])${escapeRegExp(version)}(?!${edge})`).test(output);
}

function notFoundMessage(version) {
  return (
    `evalproof: found no way to run evalproof ${version} (this npm package only launches the ` +
    'Python program).\n' +
    `Install uv (${UV_URL}) and run this again, or run: pip install evalproof==${version}\n`
  );
}

function defaultProbe(file, { env, platform }) {
  try {
    const plan = planSpawn(file, ['--version'], platform);
    const done = spawnWith(spawnSync, plan, {
      encoding: 'utf8',
      timeout: 20000,
      env,
      stdio: ['ignore', 'pipe', 'ignore'],
      windowsHide: true,
    });
    return done.status === 0 ? String(done.stdout) : '';
  } catch {
    return '';
  }
}

// The program and arguments to run, as { label, file, argv, env }; throws a LauncherError when
// there is none. `deps.which(name)` and `deps.probe(file)` default to the real ones.
function selectLauncher(args, context = {}) {
  const env = context.env || process.env;
  const platform = context.platform || process.platform;
  const version = context.version || VERSION;
  const spec = `evalproof==${version}`;
  const which = (context.deps && context.deps.which) || ((name, skip) =>
    resolveCommand(name, { env, platform, skip }));
  const childEnv = { ...env, [ACTIVE_ENV]: '1' };
  const probe =
    (context.deps && context.deps.probe) ||
    ((file) => defaultProbe(file, { env: childEnv, platform }));

  const forced = (getEnv(env, 'EVALPROOF_LAUNCHER', platform) || '').trim().toLowerCase();
  if (forced && !METHODS.includes(forced)) {
    throw new LauncherError(
      `evalproof: EVALPROOF_LAUNCHER must be one of ${METHODS.join(', ')} (got "${forced}")\n`,
      2,
    );
  }

  const attempt = {
    uvx() {
      const file = which('uvx');
      if (!file) return { reason: 'uvx is not on PATH' };
      return { label: 'uvx', file, argv: ['--from', spec, 'evalproof', ...args], env };
    },
    pipx() {
      const file = which('pipx');
      if (!file) return { reason: 'pipx is not on PATH' };
      return { label: 'pipx', file, argv: ['run', '--spec', spec, 'evalproof', ...args], env };
    },
    path() {
      if (getEnv(env, ACTIVE_ENV, platform)) {
        return { reason: 'the evalproof on PATH is this launcher, not the Python program' };
      }
      const file = which('evalproof', isThisScript);
      if (!file) return { reason: 'no evalproof is on PATH' };
      if (!reportsVersion(probe(file), version)) {
        return { reason: `the evalproof at ${file} does not report version ${version}`, mismatch: true };
      }
      return { label: 'path', file, argv: [...args], env: childEnv };
    },
  };

  const reasons = [];
  for (const method of forced ? [forced] : METHODS) {
    const result = attempt[method]();
    if (result.file) return result;
    reasons.push(result);
  }
  if (forced) {
    throw new LauncherError(
      `evalproof: EVALPROOF_LAUNCHER=${forced} was asked for, but ${reasons[0].reason}\n`,
      127,
    );
  }
  const mismatch = reasons.find((r) => r.mismatch);
  throw new LauncherError(
    notFoundMessage(version) + (mismatch ? `note: ${mismatch.reason}\n` : ''),
    127,
  );
}

// --- running ---------------------------------------------------------------------------

function run(chosen, platform = process.platform) {
  const plan = planSpawn(chosen.file, chosen.argv, platform);
  const child = spawnWith(spawn, plan, { stdio: 'inherit', env: chosen.env });

  if (platform === 'win32') {
    // No POSIX signals here. Ctrl+C reaches every process on the console, the child
    // included; this process only has to outlive it so that its exit code is passed on.
    process.on('SIGINT', () => {});
    process.on('SIGBREAK', () => {});
  } else {
    // At a terminal the driver already sent Ctrl+C to the child (the same process group),
    // so forwarding it would deliver it twice. A SIGINT sent to this process alone, as
    // with `kill -INT`, comes with no terminal, and is forwarded.
    const interactive = [process.stdin, process.stdout, process.stderr].some((s) => s.isTTY);
    for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
      process.on(signal, () => {
        if (signal === 'SIGINT' && interactive) return;
        child.kill(signal);
      });
    }
  }

  child.on('error', (err) => {
    process.stderr.write(`evalproof: could not start ${chosen.file}: ${err.message}\n`);
    process.exit(err.code === 'ENOENT' ? 127 : 126);
  });
  child.on('exit', (code, signal) => process.exit(exitCodeFor(code, signal)));
}

function main(args) {
  let chosen;
  try {
    chosen = selectLauncher(args);
    planSpawn(chosen.file, chosen.argv); // refuse an unsafe argument before starting anything
  } catch (err) {
    if (err instanceof LauncherError) {
      process.stderr.write(err.message.endsWith('\n') ? err.message : `${err.message}\n`);
      process.exit(err.exitCode);
    }
    throw err;
  }
  run(chosen);
}

if (require.main === module) {
  main(process.argv.slice(2));
}

module.exports = {
  ACTIVE_ENV,
  LauncherError,
  METHODS,
  VERSION,
  exitCodeFor,
  getEnv,
  notFoundMessage,
  planSpawn,
  quoteForCmd,
  reportsVersion,
  resolveCommand,
  selectLauncher,
  windowsExtensions,
};
