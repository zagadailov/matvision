import {drawCameraFrame, frameGeometry, getPoseModel} from './vision.js';
import {movementLabels, poseFeatures, readExamples, researchModelEnabled, sampleInterval, saveExamples, setResearchModelEnabled, summarizeMovement, windowSeconds} from './movement.js';
import baseline from './movement-baseline.js';

const $ = id => document.getElementById(id);
const video = $('training-video');
const frame = document.createElement('canvas');
const context = frame.getContext('2d', {alpha: false});
let file;
let fileUrl;
let examples = readExamples();

function youtubeId(value) {
  try {
    const url = new URL(value);
    if (!['https:', 'http:'].includes(url.protocol)) return null;
    const host = url.hostname.replace(/^www\./, '');
    const [route, pathId] = url.pathname.split('/').filter(Boolean);
    const id = host === 'youtu.be' ? route :
      ['youtube.com', 'm.youtube.com'].includes(host) ? url.searchParams.get('v') || pathId : null;
    return /^[\w-]{11}$/.test(id || '') ? id : null;
  } catch {
    return null;
  }
}

function renderExamples() {
  const counts = movementLabels.map(label => examples.filter(example => example.label === label).length);
  const total = counts.reduce((sum, count) => sum + count, 0);
  $('model-state').textContent = counts.filter(Boolean).length >= 2
    ? total + ' saved examples update estimates in your next round.'
    : 'Add examples of two movement types to train a model on your footage.';
  $('label-counts').replaceChildren(...movementLabels.map((label, i) => {
    const row = document.createElement('li');
    const count = document.createElement('strong');
    count.textContent = counts[i];
    row.append(label, count);
    return row;
  }));
}

function renderBaseline() {
  const {labeled, evaluated, predicted, correct} = baseline.validation;
  $('baseline-validation').textContent = 'The noncommercial research model estimates standing and ground movement. ' +
    'In independent footage, tracking yielded ' + evaluated + ' usable windows out of ' + labeled +
    ' labeled windows. It estimated movement for ' + predicted + ' and was correct on ' + correct + '.';
  const {title, credit, url, publication, license, licenseUrl, changes} = baseline.training;
  const paragraph = $('baseline-training');
  const dataset = document.createElement('a');
  dataset.href = url;
  dataset.textContent = title;
  const paper = document.createElement('a');
  paper.href = publication;
  paper.textContent = 'paper';
  const terms = document.createElement('a');
  terms.href = licenseUrl;
  terms.textContent = license;
  paragraph.replaceChildren('Training used ' + baseline.examples.length + ' pose windows from the ', dataset,
    ' by ' + credit + ' (', paper, ', ', terms, '). ' + changes);
  $('baseline-sources').replaceChildren(...baseline.sources.map(({title, credit, url, license, licenseUrl}) => {
    const item = document.createElement('li');
    const footage = document.createElement('a');
    footage.href = url;
    footage.textContent = title;
    const terms = document.createElement('a');
    terms.href = licenseUrl;
    terms.textContent = license;
    item.append(footage, ' · ' + credit + ' · ', terms);
    return item;
  }));
}

async function checkSource(event) {
  event.preventDefault();
  const input = $('source-url');
  const id = youtubeId(input.value);
  const status = $('source-status');
  const link = $('source-link');
  link.hidden = !id;
  if (!id) {
    status.textContent = 'Enter a YouTube watch or share link.';
    return;
  }
  link.href = 'https://www.youtube.com/watch?v=' + id;
  $('check-source').disabled = true;
  status.textContent = 'Checking with YouTube…';
  try {
    const response = await fetch('https://youtube.googleapis.com/youtube/v3/videoTrainability?id=' + id,
      {signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error('Permission check failed');
    const {permitted} = await response.json();
    const permissions = permitted.map(value => value.toLowerCase());
    if (youtubeId(input.value) !== id) return;
    status.textContent = permissions.includes('all') ? 'YouTube lists this video as eligible for third-party training. Use an authorized video file for the clip.' :
      permissions.includes('none') ? 'YouTube reports that this video is unavailable for third-party training.' :
      'YouTube lists permission for specific parties. Verify permission before using a copy.';
  } catch {
    if (youtubeId(input.value) === id) status.textContent = 'YouTube’s training permission is currently unavailable.';
  } finally {
    $('check-source').disabled = false;
  }
}

function updateFile() {
  video.pause();
  if (fileUrl) URL.revokeObjectURL(fileUrl);
  file = $('training-file').files[0];
  $('learn').disabled = $('mark-start').disabled = true;
  $('training-status').textContent = '';
  $('clip-start').value = '0';
  video.hidden = !file;
  if (!file) {
    fileUrl = undefined;
    video.removeAttribute('src');
    video.load();
    return;
  }
  fileUrl = URL.createObjectURL(file);
  video.src = fileUrl;
}

function syncClip() {
  const available = Number.isFinite(video.duration) && video.duration >= windowSeconds;
  $('learn').disabled = $('mark-start').disabled = !available;
  if (available) {
    $('clip-start').max = Math.floor((video.duration - windowSeconds) * 100) / 100;
    $('training-status').textContent = '';
  } else {
    $('training-status').textContent = 'Choose a clip with at least two seconds of video.';
  }
}

function seek(time) {
  if (Math.abs(video.currentTime - time) < 0.01 && video.readyState >= 2) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => complete(new Error('Could not read this part of the clip.')), 5000);
    const ready = () => complete();
    const failed = () => complete(new Error('Could not decode this clip.'));
    function complete(error) {
      clearTimeout(timer);
      video.removeEventListener('seeked', ready);
      video.removeEventListener('error', failed);
      if (error) reject(error);
      else resolve();
    }
    video.addEventListener('seeked', ready, {once: true});
    video.addEventListener('error', failed, {once: true});
    try { video.currentTime = time; } catch { failed(); }
  });
}

async function extractFeatures(start) {
  const model = await getPoseModel('IMAGE');
  const geometry = frameGeometry(video.videoWidth, video.videoHeight);
  frame.width = geometry.width;
  frame.height = geometry.height;
  const observations = [];
  for (let time = 0; time < windowSeconds; time += sampleInterval) {
    await seek(start + time);
    drawCameraFrame(context, video, geometry);
    const result = model.detect(frame);
    const features = poseFeatures(result.landmarks, frame.width / frame.height);
    if (features) observations.push({time, features});
  }
  return summarizeMovement(observations);
}

async function addExample(event) {
  event.preventDefault();
  if (!file || $('learn').disabled) return;
  const start = Number($('clip-start').value);
  const label = $('label').value;
  const source = youtubeId($('source-url').value);
  const fileKey = [file.name, file.size, file.lastModified].join(':');
  const status = $('training-status');
  if (examples.some(example => example.fileKey === fileKey && example.start === start)) {
    status.textContent = 'An example already starts at that time. Choose a different exchange.';
    return;
  }

  const controls = ['training-file', 'clip-start', 'label', 'learn', 'mark-start'].map($);
  controls.forEach(control => { control.disabled = true; });
  video.pause();
  video.controls = false;
  status.textContent = 'Extracting movement from the clip…';
  try {
    const features = await extractFeatures(start);
    if (!features) {
      status.textContent = 'Both athletes need to be visible for more of the clip. Try another timestamp.';
      return;
    }
    const updated = [...examples, {label, start, source, fileKey, features}];
    saveExamples(updated);
    examples = updated;
    renderExamples();
    status.textContent = 'Saved ' + label.toLowerCase() + ' at ' + start.toFixed(2) + ' seconds.';
  } catch {
    status.textContent = 'The example could not be saved. Check the video, model connection, and browser storage.';
  } finally {
    video.controls = true;
    controls.forEach(control => { control.disabled = false; });
  }
}

$('source-form').onsubmit = checkSource;
$('source-url').oninput = () => {
  $('source-link').hidden = true;
  $('source-status').textContent = '';
};
$('training-file').onchange = updateFile;
video.onloadeddata = syncClip;
video.onerror = () => {
  $('learn').disabled = $('mark-start').disabled = true;
  $('training-status').textContent = 'This video could not be opened. Try an MP4 your browser can play.';
};
$('mark-start').onclick = () => {
  $('clip-start').value = Math.min(Number($('clip-start').max), Math.round(video.currentTime * 100) / 100);
};
$('example-form').onsubmit = addExample;
$('label').replaceChildren(...movementLabels.map(label => new Option(label, label)));
$('research-model').checked = researchModelEnabled();
$('research-model').onchange = () => {
  const enabled = $('research-model').checked;
  try {
    setResearchModelEnabled(enabled);
    $('research-status').textContent = enabled ? 'Enabled for your next live round.' : 'Research estimates are disabled.';
  } catch {
    $('research-model').checked = researchModelEnabled();
    $('research-status').textContent = 'Browser storage is unavailable.';
  }
};
renderExamples();
renderBaseline();
