'use strict';

// The evalproof helper for JavaScript and TypeScript systems. `exports.name = ...` (not a
// spread or a computed assignment) so that Node can find the names for `import { runCommand }`.

const { runCommand } = require('./command');
const { serve } = require('./serve');
const { fromVercelAI, fromLangChain } = require('./convert');

exports.runCommand = runCommand;
exports.serve = serve;
exports.fromVercelAI = fromVercelAI;
exports.fromLangChain = fromLangChain;
