import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation,PresentationFile} from 'file:///C:/Users/Sherlock/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/Sherlock/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations/container_tools/artifact_tool_utils.mjs';

const root='D:/University/Alpha-SQL';
const build=root+'/.codex-build/curriculum-p0a-ppt';
const skill='C:/Users/Sherlock/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
const final=root+'/output/presentations/SQL选择器错误感知课程学习_组会汇报_20261020.pptx';
await fs.mkdir(path.dirname(final),{recursive:true});
const p=await Presentation.load(JSON.parse(await fs.readFile(build+'/deck.proto.json','utf8')));
await (await PresentationFile.exportPptx(p)).save(build+'/draft.pptx');
const result=await finalizePresentation({
  workspaceDir:root,candidatePath:build+'/draft.pptx',finalPath:final,
  pythonExecutable:'C:/Users/Sherlock/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
  layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[3,4,6,8,9,10,12].flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:12,
  requiredNativeTableOwnerSlides:[3,4,6,8,9,10,12],
  requiredNativeChartOwnerSlides:[7],
  materializeLiteralChartWorkbooks:true,
  nativeChartTargetApplication:'powerpoint',
  fontPolicy:{basis:'design',families:['Microsoft YaHei']},
  verifyArtifactToolImport:true,
  receiptPath:build+'/validation.json',
});
console.log(JSON.stringify(result));
