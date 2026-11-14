import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { FileBlob, PresentationFile } from 'file:///C:/Users/ManojR/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';

const root = 'C:/Users/ManojR/.codex/visualizations/2026/10/15/01a1405a-ac64-7ac1-81da-4c4d35ef5324';
const skill = 'C:/Users/ManojR/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const input = `${root}/output/Phase-3-Review-2-Professional.pptx`;
const candidate = `${root}/build/Phase-3-Review-2-Pseudocode-draft.pptx`;
const final = `${root}/output/Phase-3-Review-2-Pseudocode.pptx`;
const p = await PresentationFile.importPptx(await FileBlob.load(input));
const beforeCount = p.slides.items.length;
const info = await p.inspect({ kind: 'slide,textbox', maxChars: 40000 });
await fs.writeFile(`${root}/build/pseudocode-before-inspect.ndjson`, info.ndjson);
console.log(info.ndjson);

const architectureSlide = p.slides.items.find((slide, i) => {
  return i === 5;
});
if (!architectureSlide) throw new Error('Could not locate the architecture slide');
const { slide } = p.slides.insert({ after: architectureSlide, layout: 'BLANK' });
const addText = (name, text, position, style, fill = 'none', lineFill = 'none') => {
  const shape = slide.shapes.add({
    geometry: 'rect', name, position, fill,
    line: { style: 'solid', fill: lineFill, width: lineFill === 'none' ? 0 : 1 },
  });
  shape.text = text;
  shape.text.style = style;
  shape.text.insets = 0;
  return shape;
};
addText('Slide title', 'Main Pipeline Pseudocode',
  { left: 236, top: 76, width: 850, height: 48 },
  { fontSize: 30, typeface: 'Trebuchet MS', color: '#171717', alignment: 'right' });
addText('Main pipeline pseudocode',
`OFFLINE INDEXING
function INDEX_LECTURE(video):
    features  <- extract_features(video, sample_every = 3 seconds)
    segments  <- boundary_detector(features)
    for each segment in segments:
        modalities <- pool(OCR, transcript, CLIP, segment)
        vector     <- segment_fusion_encoder(modalities)
        FAISS.add(vector, video_id, start_time, end_time)
    return FAISS index

ONLINE SEARCH
function SEARCH(query, k):
    query_vector <- query_projector(SentenceBERT(query))
    matches      <- FAISS.search(query_vector, top_k = k)
    return ranked timestamp links for matches`,
  { left: 192, top: 164, width: 952, height: 438 },
  { fontSize: 18, typeface: 'Courier New', color: '#171717', alignment: 'left' },
  '#F8F8F8', '#D7D7D7');
addText('Project label', 'Concept-Aware Lecture Retrieval System',
  { left: 8, top: 6.63, width: 432, height: 38.33 },
  { fontSize: 15, typeface: 'Arial', color: '#777777', alignment: 'center' });
addText('Team footer', 'Darshan Gowda BM  |  Chirag BL  |  Manoj R  |  Kanniyappa Ranjith M',
  { left: 387, top: 678.19, width: 506.33, height: 38.33 },
  { fontSize: 12, typeface: 'Calibri', color: '#FFFFFF', alignment: 'center' });
const logoPath = `${root}/build/pes-logo.png`;
try {
  const logo = new Uint8Array(await fs.readFile(logoPath));
  slide.images.add({ blob: logo, contentType: 'image/png', alt: 'PES University logo', fit: 'contain',
    position: { left: 1144, top: 0, width: 136, height: 107.62 } });
} catch {
  // Some imported deck variants do not include the original extracted logo asset.
}
if (p.slides.items.length !== beforeCount + 1) throw new Error('Expected exactly one new slide');
const out = await PresentationFile.exportPptx(p);
await out.save(candidate);
const preview = await p.export({ slide, format: 'png', scale: 1 });
await fs.writeFile(`${root}/build/pseudocode-preview.png`, new Uint8Array(await preview.arrayBuffer()));
console.log(JSON.stringify({ slides: p.slides.items.length, candidate, final }, null, 2));
