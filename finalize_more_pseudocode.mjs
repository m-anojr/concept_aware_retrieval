import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
process.env.RUNTIME_NODE_MODULES = 'C:/Users/ManojR/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES = process.env.RUNTIME_NODE_MODULES;
const root = 'C:/Users/ManojR/.codex/visualizations/2026/10/15/01a1405a-ac64-7ac1-81da-4c4d35ef5324';
const skill = 'C:/Users/ManojR/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const candidatePath = `${root}/build/Phase-3-Review-2-More-Pseudocode-draft.pptx`;
const finalPath = `${root}/output/Phase-3-Review-2-Professional-with-More-Pseudocode.pptx`;
const stagingDir = `${root}/build/finalizer-more-pseudocode`;
await fs.mkdir(stagingDir, { recursive: true });
const { finalizePresentation } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const result = await finalizePresentation({
  workspaceDir: root,
  candidatePath,
  finalPath,
  pythonExecutable: 'C:/Users/ManojR/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath: path.join(skill, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skill, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-bullet-geometry', '--validate-heading-fit', '--require-native-table-slide', '17'],
  requiredNativeTableOwnerSlides: [17],
  explicitTotalSlideCount: 20,
  fontPolicy: { basis: 'design', families: ['Arial', 'Calibri', 'Trebuchet MS', 'Courier New'] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, 'Phase-3-Review-2-Professional-with-More-Pseudocode.validation.json'),
});
console.log(JSON.stringify(result, null, 2));
