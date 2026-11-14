import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from 'file:///C:/Users/ManojR/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';

const root = 'C:/Users/ManojR/.codex/visualizations/2026/10/15/01a1405a-ac64-7ac1-81da-4c4d35ef5324';
const input = `${root}/output/Phase-3-Review-2-Professional-with-Pseudocode.pptx`;
const candidate = `${root}/build/Phase-3-Review-2-More-Pseudocode-draft.pptx`;
const p = await PresentationFile.importPptx(await FileBlob.load(input));
const pseudoIndex = await p.inspect({ kind: 'slide', search: 'Main Pipeline Pseudocode', maxChars: 4000 });
const match = pseudoIndex.ndjson.match(/"id":"(sl\/[^"]+)"[^\n]*"title":"Main Pipeline Pseudocode"/);
if (!match) throw new Error('Could not locate Main Pipeline Pseudocode slide');
const baseSlide = p.resolve(match[1]);
const { slide } = p.slides.insert({ after: baseSlide, layout: 'BLANK' });
const addText = (name, text, position, style, fill = 'none', lineFill = 'none') => {
  const shape = slide.shapes.add({ geometry: 'rect', name, position, fill,
    line: { style: 'solid', fill: lineFill, width: lineFill === 'none' ? 0 : 1 } });
  shape.text = text;
  shape.text.style = style;
  shape.text.insets = 0;
  return shape;
};
addText('Slide title', 'Model Training & Evaluation Pseudocode',
  { left: 205, top: 76, width: 881, height: 48 },
  { fontSize: 29, typeface: 'Trebuchet MS', color: '#171717', alignment: 'right' });
addText('Training and evaluation pseudocode',
`STAGE 1  |  BOUNDARY DETECTOR
features <- load_temporal_features(video)
labels   <- merge(pseudo_boundaries(features), manual_boundaries)
for epoch in training_epochs:
    scores <- boundary_detector(features)
    loss   <- margin_boundary_loss(scores, labels)
             + segment_InfoNCE(features, labels)
    update(boundary_detector, loss)
segments <- decode_boundaries(boundary_detector(features))

STAGE 2  |  SEGMENT RETRIEVAL
segment_data <- pool_modalities(OCR, transcript, CLIP, segments)
pairs        <- load_query_segment_pairs(segment_data)
for epoch in training_epochs:
    segment_vectors <- fusion_encoder(segment_data, mode = cross_attention)
    query_vectors   <- query_projector(SentenceBERT(pairs.queries))
    loss            <- symmetric_InfoNCE(query_vectors, segment_vectors)
    update(fusion_encoder, query_projector, loss)
save_best_checkpoints(boundary_detector, fusion_encoder, query_projector)

EVALUATION
compare boundary F1 / IoU with fixed-window and shot-detection baselines
compare Recall@k / mean IoU with sliding-window retrieval on held-out queries`,
  { left: 192, top: 154, width: 952, height: 458 },
  { fontSize: 15.5, typeface: 'Courier New', color: '#171717', alignment: 'left' },
  '#F8F8F8', '#D7D7D7');
addText('Project label', 'Concept-Aware Lecture Retrieval System',
  { left: 8, top: 6.63, width: 432, height: 38.33 },
  { fontSize: 15, typeface: 'Arial', color: '#777777', alignment: 'center' });
addText('Team footer', 'Darshan Gowda BM  |  Chirag BL  |  Manoj R  |  Kanniyappa Ranjith M',
  { left: 387, top: 678.19, width: 506.33, height: 38.33 },
  { fontSize: 12, typeface: 'Calibri', color: '#FFFFFF', alignment: 'center' });
const logo = new Uint8Array(await fs.readFile(`${root}/build/pes-logo.png`));
slide.images.add({ blob: logo, contentType: 'image/png', alt: 'PES University logo', fit: 'contain',
  position: { left: 1144, top: 0, width: 136, height: 107.62 } });
const pptx = await PresentationFile.exportPptx(p);
await pptx.save(candidate);
const preview = await p.export({ slide, format: 'png', scale: 1 });
await fs.writeFile(`${root}/build/training-pseudocode-preview.png`, new Uint8Array(await preview.arrayBuffer()));
console.log(JSON.stringify({ slideCount: p.slides.items.length, candidate }, null, 2));
