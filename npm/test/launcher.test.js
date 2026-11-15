'use strict';

// Run with `npm test` from npm/. The launcher is started for real with fake `uvx`,
// `pipx` and `evalproof` programs on a temporary PATH (POSIX only); the choice of method, the
// Windows path and quoting rules and the exit codes are also tested on any OS with injected
// PATH, PATHEXT and file lookups.

const assert = require('node:assert/strict');
const { once } = require('node:events');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');
const { after, test } = require('node:test');

const launcher = require('../bin/evalproof.js');

const NPM_DIR = path.join(__dirname, '..');
const LAUNCHER = path.join(NPM_DIR, 'bin', 'evalproof.js');
const VERSION = launcher.VERSION;
const posixOnly = { skip: process.platform === 'win32' && 'fake executables are shell scripts' };

const scratch = [];
function tempDir() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'evalproof-launcher-'));
  scratch.push(dir);
  return dir;
}
after(() => {
  for (const dir of scratch) fs.rmSync(dir, { recursive: true, force: true });
});

// --- the package -----------------------------------------------------------------------

test('package.json pins one version, one bin, no dependencies and a minimal file list', () => {
  const pkg = JSON.parse(fs.readFileSync(path.join(NPM_DIR, 'package.json'), 'utf8'));
  assert.equal(pkg.name, 'evalproof');
  assert.match(pkg.version, /^\d+\.\d+\.\d+$/);
  assert.equal(pkg.version, VERSION);
  assert.deepEqual(pkg.bin, { evalproof: 'bin/evalproof.js' });
  assert.deepEqual(pkg.files, ['bin', 'lib', 'README.md', 'LICENSE']);
  assert.equal(pkg.main, 'lib/index.js');
  assert.equal(pkg.types, 'lib/index.d.ts');
  assert.equal(pkg.exports, undefined, 'no exports map: deep requires keep working');
  assert.equal(pkg.engines.node, '>=18');
  assert.equal(pkg.license, 'Apache-2.0');
  assert.equal(pkg.repository.directory, 'npm');
  assert.equal(pkg.homepage, 'https://basantbhattarai.com.np/evalproof/');
  for (const field of ['dependencies', 'devDependencies', 'optionalDependencies', 'scripts']) {
    const value = pkg[field] || {};
    if (field === 'scripts') {
      assert.deepEqual(Object.keys(value).sort(), ['test']);
    } else {
      assert.deepEqual(value, {}, field);
    }
  }
});

test('npm/LICENSE is a regular file with the same bytes as the repository LICENSE', () => {
  const copy = path.join(NPM_DIR, 'LICENSE');
  assert.equal(fs.lstatSync(copy).isSymbolicLink(), false, 'a copy, not a symlink');
  const original = path.join(NPM_DIR, '..', 'LICENSE');
  assert.ok(fs.readFileSync(copy).equals(fs.readFileSync(original)));
});

test('the launcher script starts with a node shebang', () => {
  assert.match(fs.readFileSync(LAUNCHER, 'utf8'), /^#!\/usr\/bin\/env node\n/);
});

// --- fake programs, run for real (POSIX) -----------------------------------------------

const FAKE_TOOL = `
const fs = require('fs');
const [tool, ...argv] = process.argv.slice(2);
if (process.env.FAKE_LOG) {
  fs.appendFileSync(process.env.FAKE_LOG, JSON.stringify({ tool, argv }) + '\\n');
}
if (tool === 'evalproof' && argv[0] === '--version') {
  process.stdout.write('evalproof ' + (process.env.FAKE_VERSION || '0.0.0') + '\\n');
  process.exit(0);
}
if (process.env.FAKE_WAIT) {
  if (process.env.FAKE_WAIT === 'handle') {
    for (const sig of ['SIGTERM', 'SIGINT']) {
      process.on(sig, () => { process.stdout.write('got ' + sig + '\\n'); process.exit(7); });
    }
  }
  process.stdout.write('ready\\n');
  setInterval(() => {}, 1000);
} else {
  process.stdout.write(JSON.stringify({ tool, argv }) + '\\n');
  process.exit(Number(process.env.FAKE_EXIT || 0));
}
`;

// A PATH directory holding fake executables. Each one logs its call and prints its argv as
// JSON, or, for `evalproof --version`, the version in FAKE_VERSION.
function fakeBin(names) {
  const dir = tempDir();
  const script = path.join(dir, 'fake-tool.js');
  fs.writeFileSync(script, FAKE_TOOL);
  const bin = path.join(dir, 'bin');
  fs.mkdirSync(bin);
  for (const name of names) {
    const file = path.join(bin, name);
    fs.writeFileSync(
      file,
      `#!/bin/sh\nexec "${process.execPath}" "${script}" ${name} "$@"\n`,
    );
    fs.chmodSync(file, 0o755);
  }
  return { bin, log: path.join(dir, 'calls.log') };
}

function launchEnv(fake, extra = {}) {
  return { PATH: fake.bin, FAKE_LOG: fake.log, FAKE_VERSION: VERSION, ...extra };
}

function launch(fake, args, extra = {}) {
  fs.rmSync(fake.log, { force: true }); // each launch logs only its own calls
  const done = spawnSync(process.execPath, [LAUNCHER, ...args], {
    env: launchEnv(fake, extra),
    encoding: 'utf8',
    timeout: 30000,
  });
  const calls = fs.existsSync(fake.log)
    ? fs.readFileSync(fake.log, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse)
    : [];
  return { status: done.status, stdout: done.stdout, stderr: done.stderr, calls };
}

const ARGS = ['compare', 'base run.json', "it's", '--gate', ''];

test('uvx is used first, pinned to this version', posixOnly, () => {
  const fake = fakeBin(['uvx', 'pipx', 'evalproof']);
  const done = launch(fake, ARGS);
  const argv = ['--from', `evalproof==${VERSION}`, 'evalproof', ...ARGS];
  assert.equal(done.status, 0);
  assert.deepEqual(done.calls, [{ tool: 'uvx', argv }]);
  assert.deepEqual(JSON.parse(done.stdout), { tool: 'uvx', argv });
});

test('pipx is used when there is no uvx', posixOnly, () => {
  const fake = fakeBin(['pipx', 'evalproof']);
  const done = launch(fake, ARGS);
  const argv = ['run', '--spec', `evalproof==${VERSION}`, 'evalproof', ...ARGS];
  assert.equal(done.status, 0);
  assert.deepEqual(done.calls, [{ tool: 'pipx', argv }]);
});

test('an evalproof on PATH is used when it reports exactly this version', posixOnly, () => {
  const fake = fakeBin(['evalproof']);
  const done = launch(fake, ARGS);
  assert.equal(done.status, 0);
  assert.deepEqual(done.calls, [
    { tool: 'evalproof', argv: ['--version'] },
    { tool: 'evalproof', argv: ARGS },
  ]);
  assert.deepEqual(JSON.parse(done.stdout), { tool: 'evalproof', argv: ARGS });
});

test('an evalproof of another version on PATH is not used', posixOnly, () => {
  const fake = fakeBin(['evalproof']);
  for (const other of ['0.9.0', `${VERSION}1`, `${VERSION}rc1`, `1${VERSION}`]) {
    const done = launch(fake, ARGS, { FAKE_VERSION: other });
    assert.equal(done.status, 127, other);
    assert.match(done.stderr, new RegExp(`pip install evalproof==${VERSION.replace(/\./g, '\\.')}`));
    assert.match(done.stderr, /does not report version/);
    assert.ok(done.calls.every((c) => c.argv[0] === '--version'), 'only probed');
  }
});

test('with nothing available it says how to install uv and exits 127', posixOnly, () => {
  const fake = fakeBin([]);
  const done = launch(fake, ARGS);
  assert.equal(done.status, 127);
  assert.equal(done.stdout, '');
  assert.ok(done.stderr.includes('https://docs.astral.sh/uv/getting-started/installation/'));
  assert.ok(done.stderr.includes(`pip install evalproof==${VERSION}`));
  assert.equal(done.stderr.trim().split('\n').filter((l) => l.includes('astral.sh')).length, 1);
  assert.deepEqual(done.calls, []);
});

test('EVALPROOF_LAUNCHER forces one method', posixOnly, () => {
  const fake = fakeBin(['uvx', 'pipx', 'evalproof']);
  const viaPipx = launch(fake, ['init'], { EVALPROOF_LAUNCHER: 'pipx' });
  assert.deepEqual(viaPipx.calls, [
    { tool: 'pipx', argv: ['run', '--spec', `evalproof==${VERSION}`, 'evalproof', 'init'] },
  ]);
  const viaPath = launch(fake, ['init'], { EVALPROOF_LAUNCHER: 'path' });
  assert.deepEqual(viaPath.calls.at(-1), { tool: 'evalproof', argv: ['init'] });
  assert.ok(viaPath.calls.every((c) => c.tool === 'evalproof'));
  const viaUvx = launch(fake, ['init'], { EVALPROOF_LAUNCHER: ' UVX ' });
  assert.equal(viaUvx.calls.at(-1).tool, 'uvx');
});

test('a forced method that is unavailable is an error, not a fallback', posixOnly, () => {
  const fake = fakeBin(['pipx', 'evalproof']);
  const done = launch(fake, ['init'], { EVALPROOF_LAUNCHER: 'uvx' });
  assert.equal(done.status, 127);
  assert.match(done.stderr, /EVALPROOF_LAUNCHER=uvx was asked for, but uvx is not on PATH/);
  assert.deepEqual(done.calls, []);
  const wrong = launch(fake, ['init'], { EVALPROOF_LAUNCHER: 'path', FAKE_VERSION: '0.0.1' });
  assert.equal(wrong.status, 127);
  assert.match(wrong.stderr, /EVALPROOF_LAUNCHER=path was asked for, but .*does not report/);
});

test('an unknown EVALPROOF_LAUNCHER value is a usage error', posixOnly, () => {
  const fake = fakeBin(['uvx']);
  const done = launch(fake, ['init'], { EVALPROOF_LAUNCHER: 'npm' });
  assert.equal(done.status, 2);
  assert.match(done.stderr, /must be one of uvx, pipx, path \(got "npm"\)/);
  assert.deepEqual(done.calls, []);
});

for (const method of ['uvx', 'pipx', 'path']) {
  test(`the exit code of the ${method} program is the launcher's`, posixOnly, () => {
    const names = method === 'path' ? ['evalproof'] : [method];
    const fake = fakeBin(names);
    for (const code of [0, 1, 3, 4, 42]) {
      assert.equal(launch(fake, ['x'], { FAKE_EXIT: String(code) }).status, code, `${method} ${code}`);
    }
  });
}

test('an evalproof that is this launcher is never run again', posixOnly, () => {
  const dir = tempDir();
  fs.symlinkSync(LAUNCHER, path.join(dir, 'evalproof')); // what `npm i -g evalproof` puts on PATH
  const done = spawnSync(process.execPath, [LAUNCHER, '--version'], {
    env: { PATH: dir },
    encoding: 'utf8',
    timeout: 30000,
  });
  assert.equal(done.status, 127);
  assert.match(done.stderr, /pip install evalproof==/);
});

test('inside the launcher the path method refuses an evalproof on PATH', posixOnly, () => {
  const fake = fakeBin(['evalproof']);
  const done = launch(fake, ['init'], { [launcher.ACTIVE_ENV]: '1' });
  assert.equal(done.status, 127);
  assert.deepEqual(done.calls, []);
});

test('the path method marks its child so a nested launcher cannot loop', posixOnly, () => {
  const fake = fakeBin(['evalproof']);
  fs.writeFileSync(
    path.join(fake.bin, 'evalproof'),
    `#!/bin/sh\n[ "$1" = --version ] && echo "evalproof ${VERSION}" && exit 0\n` +
      `echo "active=$${launcher.ACTIVE_ENV}"\n`,
  );
  const done = launch(fake, ['x']);
  assert.equal(done.status, 0);
  assert.equal(done.stdout.trim(), 'active=1');
});

test('SIGTERM and SIGINT sent to the launcher reach the program', posixOnly, async () => {
  const fake = fakeBin(['uvx']);
  for (const signal of ['SIGTERM', 'SIGINT']) {
    const child = spawn(process.execPath, [LAUNCHER, 'x'], {
      env: launchEnv(fake, { FAKE_WAIT: 'handle' }),
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let out = '';
    await new Promise((resolve) => {
      child.stdout.on('data', (chunk) => {
        out += chunk;
        if (out.includes('ready')) resolve();
      });
    });
    child.kill(signal);
    const [code] = await once(child, 'exit');
    assert.match(out, new RegExp(`got ${signal}`));
    assert.equal(code, 7, 'the exit code of the program, not of the signal');
  }
});

test('a program killed by a signal gives 128 plus the signal number', posixOnly, async () => {
  const fake = fakeBin(['pipx']);
  const child = spawn(process.execPath, [LAUNCHER, 'x'], {
    env: launchEnv(fake, { FAKE_WAIT: 'die' }),
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  await new Promise((resolve) => child.stdout.once('data', resolve));
  child.kill('SIGTERM');
  const [code, signal] = await once(child, 'exit');
  assert.equal(signal, null);
  assert.equal(code, 128 + os.constants.signals.SIGTERM);
});

// --- the same rules with injected lookups, on any OS -----------------------------------

const WIN_ENV = { Path: 'C:\\tools;"C:\\Program Files (x86)\\uv";C:\\later', PATHEXT: '.COM;.EXE;.BAT;.CMD;.VBS' };

function winResolve(name, files, env = WIN_ENV) {
  const present = new Set(files);
  return launcher.resolveCommand(name, { env, platform: 'win32', isFile: (p) => present.has(p) });
}

test('Windows: .exe wins over .cmd, and an extensionless file is never a match', () => {
  assert.equal(
    winResolve('uvx', ['C:\\tools\\uvx.cmd', 'C:\\tools\\uvx.exe']),
    'C:\\tools\\uvx.exe',
  );
  assert.equal(
    winResolve('uvx', ['C:\\tools\\uvx', 'C:\\later\\uvx.exe']),
    'C:\\later\\uvx.exe',
  );
  assert.equal(winResolve('uvx', ['C:\\tools\\uvx', 'C:\\tools\\uvx.ps1', 'C:\\tools\\uvx.vbs']), null);
});

test('Windows: a .cmd or .bat is found when it is all there is; PATH is read in order', () => {
  assert.equal(winResolve('pipx', ['C:\\later\\pipx.bat']), 'C:\\later\\pipx.bat');
  assert.equal(winResolve('pipx', ['C:\\later\\pipx.cmd', 'C:\\tools\\pipx.cmd']), 'C:\\tools\\pipx.cmd');
  assert.equal(winResolve('uvx', []), null);
});

test('Windows: quoted PATH entries lose their quotes, PATH and PATHEXT are case-insensitive', () => {
  assert.equal(
    winResolve('uvx', ['C:\\Program Files (x86)\\uv\\uvx.exe']),
    'C:\\Program Files (x86)\\uv\\uvx.exe',
  );
  const lower = { path: 'C:\\a', pathext: '.cmd' };
  assert.equal(winResolve('x', ['C:\\a\\x.cmd', 'C:\\a\\x.exe'], lower), 'C:\\a\\x.cmd');
  assert.equal(winResolve('x', ['C:\\a\\x.exe'], { PATH: 'C:\\a' }), 'C:\\a\\x.exe'); // default PATHEXT
});

test('POSIX: PATH is split on colons, an empty entry (the current directory) is skipped', () => {
  const present = new Set(['/b/uvx', 'uvx']);
  const seen = [];
  const found = launcher.resolveCommand('uvx', {
    env: { PATH: ':/a::/b' },
    platform: 'linux',
    isFile: (p) => {
      seen.push(p);
      return present.has(p);
    },
  });
  assert.equal(found, '/b/uvx');
  assert.deepEqual(seen, ['/a/uvx', '/b/uvx']);
  assert.equal(launcher.resolveCommand('uvx', { env: {}, platform: 'linux', isFile: () => true }), null);
});

test('resolveCommand skips what `skip` rejects and goes on', () => {
  const found = launcher.resolveCommand('evalproof', {
    env: { PATH: '/a:/b' },
    platform: 'linux',
    isFile: () => true,
    skip: (p) => p === '/a/evalproof',
  });
  assert.equal(found, '/b/evalproof');
});

test('quoteForCmd quotes every argument and doubles a trailing backslash', () => {
  assert.equal(launcher.quoteForCmd('compare'), '"compare"');
  assert.equal(launcher.quoteForCmd('a b.json'), '"a b.json"');
  assert.equal(launcher.quoteForCmd(''), '""');
  assert.equal(launcher.quoteForCmd('C:\\my dir\\'), '"C:\\my dir\\\\"');
  assert.equal(launcher.quoteForCmd('a\\\\'), '"a\\\\\\\\"');
  assert.equal(launcher.quoteForCmd('a\\b'), '"a\\b"');
  assert.equal(launcher.quoteForCmd('--judge=ollama:qwen3:8b'), '"--judge=ollama:qwen3:8b"');
  assert.equal(launcher.quoteForCmd('(x86) [a],b;c=d'), '"(x86) [a],b;c=d"');
  assert.equal(launcher.quoteForCmd('naïve 日本'), '"naïve 日本"');
});

test('quoteForCmd refuses what cmd.exe would act on, with exit 126', () => {
  for (const bad of ['a"b', '%PATH%', '100%', 'a^b', 'a&b', 'a|b', 'a<b', 'a>b', '!x!', 'a\nb', 'a\rb', 'a\0b']) {
    assert.throws(
      () => launcher.quoteForCmd(bad),
      (err) => err instanceof launcher.LauncherError && err.exitCode === 126 && /cmd\.exe/.test(err.message),
      JSON.stringify(bad),
    );
  }
});

test('planSpawn: .exe directly with no shell, .cmd and .bat through cmd.exe as one quoted line', () => {
  assert.deepEqual(launcher.planSpawn('C:\\uv\\uvx.exe', ['a & b'], 'win32'), {
    shell: false,
    command: 'C:\\uv\\uvx.exe',
    args: ['a & b'],
  });
  assert.deepEqual(launcher.planSpawn('C:\\Program Files (x86)\\x\\pipx.CMD', ['run', 'a b'], 'win32'), {
    shell: true,
    command: '"C:\\Program Files (x86)\\x\\pipx.CMD" "run" "a b"',
    args: [],
  });
  assert.equal(launcher.planSpawn('C:\\s\\evalproof.bat', [], 'win32').command, '"C:\\s\\evalproof.bat"');
  assert.throws(() => launcher.planSpawn('C:\\s\\evalproof.cmd', ['--x=%TEMP%'], 'win32'), /cmd\.exe/);
  assert.throws(() => launcher.planSpawn('C:\\a&b\\evalproof.cmd', [], 'win32'), /cmd\.exe/);
  assert.deepEqual(launcher.planSpawn('/bin/x.cmd', ['a&b'], 'linux'), {
    shell: false,
    command: '/bin/x.cmd',
    args: ['a&b'],
  });
});

test('exitCodeFor passes the code on and maps a signal to 128 plus its number', () => {
  assert.equal(launcher.exitCodeFor(0, null), 0);
  assert.equal(launcher.exitCodeFor(3, null), 3);
  assert.equal(launcher.exitCodeFor(null, 'SIGTERM'), 143);
  assert.equal(launcher.exitCodeFor(null, 'SIGINT'), 130);
  assert.equal(launcher.exitCodeFor(null, null), 1);
  assert.equal(launcher.exitCodeFor(null, 'SIGNOPE'), 1);
});

function choose(files, extra = {}) {
  const present = new Set(files);
  return launcher.selectLauncher(['init', 'a b'], {
    env: { EVALPROOF_LAUNCHER: undefined, ...(extra.env || {}) },
    platform: 'win32',
    version: '1.2.3',
    deps: {
      which: (name) => (present.has(name) ? `C:\\bin\\${name}.exe` : null),
      probe: () => extra.probe ?? 'evalproof 1.2.3\n',
    },
  });
}

test('selectLauncher: uvx, then pipx, then a matching evalproof (Windows paths)', () => {
  assert.deepEqual(choose(['uvx', 'pipx', 'evalproof']).argv, [
    '--from', 'evalproof==1.2.3', 'evalproof', 'init', 'a b',
  ]);
  const pipx = choose(['pipx', 'evalproof']);
  assert.equal(pipx.file, 'C:\\bin\\pipx.exe');
  assert.deepEqual(pipx.argv, ['run', '--spec', 'evalproof==1.2.3', 'evalproof', 'init', 'a b']);
  const direct = choose(['evalproof']);
  assert.equal(direct.label, 'path');
  assert.deepEqual(direct.argv, ['init', 'a b']);
  assert.equal(direct.env[launcher.ACTIVE_ENV], '1');
  assert.throws(() => choose(['evalproof'], { probe: 'evalproof 1.2.4\n' }), { exitCode: 127 });
  assert.throws(() => choose([]), (err) => err.exitCode === 127 && err.message.includes('pip install evalproof==1.2.3'));
});

test('selectLauncher: a forced method is the only one tried', () => {
  assert.equal(choose(['uvx', 'pipx'], { env: { EVALPROOF_LAUNCHER: 'pipx' } }).label, 'pipx');
  assert.throws(() => choose(['pipx'], { env: { EVALPROOF_LAUNCHER: 'uvx' } }), { exitCode: 127 });
  assert.throws(() => choose(['uvx'], { env: { EVALPROOF_LAUNCHER: 'bogus' } }), { exitCode: 2 });
  assert.equal(choose(['uvx'], { env: { evalproof_launcher: 'UVX' } }).label, 'uvx'); // Windows spelling
});

test('reportsVersion matches the version as a whole token only', () => {
  for (const ok of ['evalproof 1.0.0', 'evalproof 1.0.0\n', '1.0.0', 'x\nevalproof 1.0.0\r\n']) {
    assert.equal(launcher.reportsVersion(ok, '1.0.0'), true, JSON.stringify(ok));
  }
  for (const bad of ['', 'evalproof 1.0.01', 'evalproof 11.0.0', 'evalproof 1.0.0rc1', 'evalproof 1.0.0.post1', 'evalproof 1.0.0-beta', 'evalproof 0.1.0', 'evalproof 1x0y0']) {
    assert.equal(launcher.reportsVersion(bad, '1.0.0'), false, JSON.stringify(bad));
  }
});
