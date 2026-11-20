import {readFile, writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {poseFeatures, sampleInterval, summarizeMovement, trainMovement, windowSeconds} from '../dist/movement.js';

const [inputDirectory, outputFile] = process.argv.slice(2);
if (!inputDirectory || !outputFile) {
  console.error('Usage: node training/build-baseline.mjs <pose output directory> <baseline module>');
  process.exit(1);
}
const sources = JSON.parse(await readFile(new URL('./footage.json', import.meta.url), 'utf8'));
const joints = [11, 12, 23, 24];
const cocoJoints = [5, 6, 11, 12];

function convertPose(points) {
  const landmarks = [];
  points.forEach(([x, y, visibility], index) => { landmarks[joints[index]] = {x, y, visibility}; });
  return landmarks;
}

function labeledExample(label, observations) {
  const vector = summarizeMovement(observations);
  return vector ? {label, features: vector.map(value => Number(value.toFixed(5)))} : null;
}

async function readResearchExamples() {
  const annotations = JSON.parse(await readFile(join(inputDirectory, 'vicos-annotations.json'), 'utf8'));
  const sequences = new Map();
  for (const row of annotations) {
    const videoId = row.image.slice(0, 2);
    if (!sequences.has(videoId)) sequences.set(videoId, new Map());
    const label = row.position === 'standing' ? 'Standing movement' :
      row.position.startsWith('takedown') ? null : 'Ground movement';
    const poses = [row.pose1, row.pose2].filter(Boolean).map(pose => convertPose(cocoJoints.map(i => pose[i])));
    // COCO coordinates are in image pixels, so both axes already share units.
    sequences.get(videoId).set(row.frame, {label, features: poseFeatures(poses, 1)});
  }
  const rows = [];
  for (const sequence of sequences.values()) {
    const lastFrame = Math.max(...sequence.keys());
    for (let frame = 1; frame + 60 <= lastFrame; frame += 120) {
      const observations = [];
      const labels = new Set();
      for (let time = 0; time < windowSeconds; time += sampleInterval) {
        const sample = sequence.get(frame + Math.round(time * 30));
        if (!sample) continue;
        labels.add(sample.label);
        if (sample.features) observations.push({time, features: sample.features});
      }
      const [label] = labels;
      if (labels.size !== 1 || !label) continue;
      const row = labeledExample(label, observations);
      if (row) rows.push(row);
    }
  }
  return rows;
}

const clips = await Promise.all(sources.map(async ({id, segments, ...attribution}) => {
  const {aspect, sampleRate, frames, start: clipStart = 0} = JSON.parse(await readFile(join(inputDirectory, id + '-poses.json'), 'utf8'));
  if (sampleRate !== 1 / sampleInterval) throw new Error('Unexpected sampling rate in ' + id);
  const track = frames.map(poses => poseFeatures(poses.map(convertPose), aspect));
  const rows = [];
  let labeled = 0;
  for (const [start, end, label] of segments) {
    for (let time = start; time + windowSeconds <= end; time += windowSeconds) {
      labeled++;
      const observations = [];
      for (let offset = 0; offset < windowSeconds; offset += sampleInterval) {
        const features = track[Math.round((time + offset - clipStart) * sampleRate)];
        if (features) observations.push({time: offset, features});
      }
      const row = labeledExample(label, observations);
      if (row) rows.push(row);
    }
  }
  return {...attribution, rows, labeled};
}));

const examples = await readResearchExamples();
const model = trainMovement(examples);
if (!model) throw new Error('The research dataset must contain both movement classes.');

const total = {labeled: 0, evaluated: 0, predicted: 0, correct: 0};
const results = [];
for (const clip of clips) {
  const predictions = clip.rows.map(row => model(row.features));
  const result = {
    title: clip.title,
    labeled: clip.labeled,
    evaluated: clip.rows.length,
    predicted: predictions.filter(Boolean).length,
    correct: predictions.filter((prediction, index) => prediction === clip.rows[index].label).length
  };
  results.push(result);
  for (const key of Object.keys(total)) total[key] += result[key];
}

if (!total.evaluated) throw new Error('No usable windows in the independent evaluation footage.');
const baseline = {
  training: {
    title: 'Brazilian Jiu-Jitsu Positions Dataset',
    credit: 'Valter Hudovernik and Danijel Skočaj, ViCoS Lab, University of Ljubljana',
    url: 'https://vicos.si/resources/jiujitsu/',
    publication: 'https://doi.org/10.1145/3552437.3555707',
    license: 'CC BY-NC-SA 4.0',
    licenseUrl: 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
    changes: 'Sequential COCO keypoints remapped to shoulders and hips and summarized into two-second features. Position labels grouped into standing and ground movement.'
  },
  validation: {...total, clips: results},
  sources: clips.map(({rows, labeled, ...metadata}) => metadata),
  examples
};
const lines = examples.map(row => '    ' + JSON.stringify(row)).join(',\n');
const {examples: _, ...metadata} = baseline;
const header = JSON.stringify(metadata, null, 2).slice(0, -2);
await writeFile(outputFile, '// Generated by training/build-baseline.mjs.\nexport default ' + header + ',\n  "examples": [\n' + lines + '\n  ]\n};\n');
console.log(JSON.stringify({trainingWindows: examples.length, ...total}, null, 2));
console.table(results);
