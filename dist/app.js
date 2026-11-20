import {drawCameraFrame, frameGeometry, getPoseModel} from './vision.js';
import {poseFeatures, readExamples, researchModelEnabled, sampleInterval, summarizeMovement, trainMovement, windowSeconds} from './movement.js';
import baseline from './movement-baseline.js';

const $ = id => document.getElementById(id);
const video = $('camera');
const cameraFrame = $('camera-frame');
const frameContext = cameraFrame.getContext('2d', {alpha: false});
const overlay = $('overlay');
const overlayContext = overlay.getContext('2d');
const storageKey = 'matvision-rounds-v1';
const colors = ['#8bb1ff', '#63e4c8'];
const edges = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]];
let rounds = [];
try { rounds = JSON.parse(localStorage.getItem(storageKey) || '[]'); } catch {}
let session;
let selected;

const icon = (name) => '<svg><use href="#' + name + '"/></svg>';
const formatTime = (seconds) => {
  const whole = Math.max(0, Math.floor(seconds));
  return String(Math.floor(whole / 60)).padStart(2, '0') + ':' + String(whole % 60).padStart(2, '0');
};
const describeMotion = (value) => value == null ? '—' : value < 0.025 ? 'Low' : value < 0.09 ? 'Moderate' : 'High';
const coverage = samples => samples.length ? Math.round(samples.filter(s => s.count === 2).length / samples.length * 100) : null;

function feedback(title, body) {
  if ($('cue-title').textContent === title) return;
  $('cue-title').textContent = title;
  $('cue-body').textContent = body;
}

function error(message) {
  $('error').textContent = message;
  $('error').hidden = !message;
}

function switchView(name) {
  $('live').hidden = name !== 'live';
  $('review').hidden = name !== 'review';
  document.querySelectorAll('.tab[data-view]').forEach(tab => {
    const active = tab.dataset.view === name;
    tab.classList.toggle('active', active);
    if (active) tab.setAttribute('aria-current', 'page');
    else tab.removeAttribute('aria-current');
  });
  if (name === 'review') renderReview();
}

function renderLists() {
  for (const [node, items] of [[$('recent-list'), rounds.slice(0, 3)], [$('history-list'), rounds]]) {
    node.replaceChildren();
    if (!items.length) {
      const p = document.createElement('p');
      p.textContent = 'No saved rounds yet.';
      node.append(p);
    }
    items.forEach(round => {
      const button = document.createElement('button');
      button.className = 'round';
      button.setAttribute('aria-current', String(round.id === selected));
      const label = document.createElement('span');
      label.textContent = round.demo ? 'Demo round' : 'Live roll';
      const details = document.createElement('small');
      details.textContent = new Date(round.date).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) + ' · ' + formatTime(round.duration);
      button.append(label, details);
      button.onclick = () => { selected = round.id; switchView('review'); };
      node.append(button);
    });
  }
}

function resizeCanvas(canvas) {
  const scale = Math.min(devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(canvas.clientWidth * scale));
  const height = Math.max(1, Math.round(canvas.clientHeight * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function drawPose(poses) {
  resizeCanvas(overlay);
  const {width, height} = overlay;
  overlayContext.clearRect(0, 0, width, height);
  if (!$('show-overlay').checked) return;
  const scale = Math.min(width / cameraFrame.width, height / cameraFrame.height);
  const w = cameraFrame.width * scale;
  const h = cameraFrame.height * scale;
  const x = point => (width - w) / 2 + point.x * w;
  const y = point => (height - h) / 2 + point.y * h;
  poses.forEach((pose, index) => {
    overlayContext.strokeStyle = colors[index];
    overlayContext.fillStyle = colors[index];
    overlayContext.lineWidth = Math.max(2, width / 370);
    const visible = point => point && (point.visibility ?? 1) >= 0.45;
    for (const [a, b] of edges) {
      if (!visible(pose[a]) || !visible(pose[b])) continue;
      overlayContext.beginPath();
      overlayContext.moveTo(x(pose[a]), y(pose[a]));
      overlayContext.lineTo(x(pose[b]), y(pose[b]));
      overlayContext.stroke();
    }
    for (const point of pose.slice(11, 29)) {
      if (!visible(point)) continue;
      overlayContext.beginPath();
      overlayContext.arc(x(point), y(point), Math.max(2, width / 235), 0, Math.PI * 2);
      overlayContext.fill();
    }
  });
}

function setState(label, active) {
  const state = $('feed-state');
  state.replaceChildren(document.createElement('i'), document.createTextNode(' ' + label));
  state.classList.toggle('live', active);
}

function createSession(demo) {
  return { demo, samples: [], poses: [], lastSample: -1, lastDetect: -Infinity };
}

function previewReady(current) {
  current.loading = false;
  $('start').hidden = false;
  $('start').innerHTML = icon('play') + ' Start analysis';
  renderCameraState(current);
}

function renderCameraState(current) {
  const analyzing = current.started !== undefined;
  $('start').disabled = current.loading;
  $('athlete-a').hidden = $('athlete-b').hidden = !analyzing;
  $('source').textContent = analyzing ? 'ANALYZING LIVE' : 'LIVE PREVIEW';

  if (analyzing) {
    setState('Tracking', true);
    $('tracking-status').textContent = 'Looking for both athletes';
    showFraming('Finding both athletes…');
    feedback('Finding both athletes', 'Keep both athletes in view while the model picks up their poses.');
  } else if (current.loading) {
    setState('Preparing', true);
    $('tracking-status').textContent = 'Loading pose model';
    showFraming('Frame both athletes');
    feedback('Loading pose model', 'The live preview is ready while the model loads.');
  } else {
    setState('Preview', true);
    $('tracking-status').textContent = 'Landscape preview ready';
    showFraming('Frame both athletes');
    feedback('Frame both athletes', 'Use the widest available lens and leave open mat around both athletes. Select Start analysis when you’re ready.');
  }
}

function updateCameraFrame(current) {
  const {videoWidth, videoHeight, readyState, currentTime} = video;
  if (readyState < 2 || !videoWidth || !videoHeight) return;

  const sizeChanged = videoWidth !== current.geometry?.sourceWidth || videoHeight !== current.geometry?.sourceHeight;
  if (sizeChanged) {
    current.geometry = frameGeometry(videoWidth, videoHeight);
    cameraFrame.width = current.geometry.width;
    cameraFrame.height = current.geometry.height;
    current.analyzedTime = undefined;
    current.previous = null;
    current.poses = [];
    current.movementWindow = [];
    drawPose([]);
    $('crop-label').hidden = !current.geometry.cropped;
  }
  if (!sizeChanged && current.videoTime === currentTime) return;

  drawCameraFrame(frameContext, video, current.geometry);
  current.videoTime = currentTime;
  cameraFrame.hidden = false;
  $('poster').hidden = true;
}

function showFraming(message) {
  const cue = $('framing-cue');
  cue.hidden = !message;
  if (cue.textContent !== message) cue.textContent = message;
}

function startDemo() {
  if (session) return;
  error('');
  session = createSession(true);
  $('start').hidden = false;
  $('start').innerHTML = icon('video') + ' Finish demo';
  $('demo').disabled = true;
  $('preview').hidden = true;
  $('movement-label').textContent = 'Movement';
  $('mode-note').textContent = 'Demo metrics are simulated and will be labeled in your round history.';
  run(session);
}

async function openCamera(deviceId) {
  if (session) return;
  error('');
  const current = session = createSession(false);
  $('demo').disabled = true;
  $('preview').classList.replace('primary', 'secondary');
  $('preview').innerHTML = icon('video') + ' Cancel camera setup';
  $('source').textContent = 'OPENING CAMERA';
  setState('Opening', false);
  feedback('Opening camera', 'Allow camera access when prompted to see the live preview.');
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires a secure browser page.');
    const camera = deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: 'environment' } };
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        ...camera, width: { ideal: 1280 }, height: { ideal: 720 },
        aspectRatio: { ideal: 16 / 9 }, resizeMode: { ideal: 'none' }
      },
      audio: false
    });
    if (session !== current) { stream.getTracks().forEach(track => track.stop()); return; }
    current.stream = stream;
    const track = stream.getVideoTracks()[0];
    track.addEventListener('ended', () => {
      if (session === current) {
        finish();
        error('The camera disconnected. Open the camera preview to reconnect.');
      }
    });
    video.srcObject = stream;
    video.hidden = false;
    await video.play();
    if (session !== current) return;
    $('live').classList.add('camera-open');
    $('preview').innerHTML = icon('video') + ' Close camera';
    updateCameraFrame(current);
    previewReady(current);
    current.frame = requestAnimationFrame(now => tick(current, now));
    configureCameraControls(current, track);
  } catch (err) {
    if (session !== current) return;
    finish(false);
    error(err.name === 'NotAllowedError' ? 'Camera access was denied. Allow camera access in your browser or try the demo.' :
      err.name === 'NotFoundError' || err.name === 'OverconstrainedError' ? 'That camera is unavailable. Open the preview to try the default camera.' :
      err.message === 'Camera access requires a secure browser page.' ? err.message :
      'The camera could not start. Close any other app using it and try again.');
  }
}

async function configureCameraControls(current, track) {
  const zoom = track.getCapabilities?.().zoom;
  if (zoom && zoom.max > zoom.min) {
    $('zoom').min = zoom.min;
    $('zoom').max = zoom.max;
    $('zoom').step = zoom.step || 'any';
    updateZoom(track);
    $('zoom-control').hidden = false;
    $('camera-settings').hidden = false;
  }
  if (!navigator.mediaDevices.enumerateDevices) return;
  let devices;
  try { devices = await navigator.mediaDevices.enumerateDevices(); }
  catch { return; }
  if (session !== current) return;
  const cameras = devices.filter(device => device.kind === 'videoinput' && device.deviceId);
  const active = track.getSettings().deviceId;
  if (cameras.length < 2 || !cameras.some(camera => camera.deviceId === active)) return;
  $('lens').replaceChildren(...cameras.map((camera, index) =>
    new Option(camera.label || 'Camera ' + (index + 1), camera.deviceId)));
  $('lens').value = active;
  $('lens-control').hidden = false;
  $('camera-settings').hidden = false;
}

function updateZoom(track) {
  const zoom = track.getSettings().zoom ?? Number($('zoom').min);
  $('zoom').value = zoom;
  $('zoom-level').value = zoom.toLocaleString(undefined, {maximumFractionDigits: 2}) + '×';
}

async function changeZoom() {
  const current = session;
  const track = current?.stream?.getVideoTracks()[0];
  if (!track) return;
  $('zoom').disabled = true;
  try {
    await track.applyConstraints({ advanced: [{ zoom: Number($('zoom').value) }] });
    if (session === current) {
      current.previous = null;
      current.movementWindow = [];
    }
  } catch {
    if (session === current) error('The camera could not adjust zoom.');
  } finally {
    if (session === current) {
      updateZoom(track);
      $('zoom').disabled = false;
    }
  }
}

async function startAnalysis(current) {
  if (current.loading) return;
  error('');
  current.loading = true;
  $('start').innerHTML = icon('play') + ' Loading model…';
  $('lens').disabled = true;
  renderCameraState(current);
  try { current.model = await getPoseModel('VIDEO'); }
  catch {
    if (session !== current) return;
    $('lens').disabled = false;
    previewReady(current);
    error('The pose model could not load. Check your connection and retry Start analysis.');
    return;
  }
  if (session !== current) return;
  current.loading = false;
  current.movementModel = trainMovement([...(researchModelEnabled() ? baseline.examples : []), ...readExamples()]);
  current.movementWindow = [];
  current.lastMovementSample = -Infinity;
  $('classifier-metric').hidden = !current.movementModel;
  $('preview').hidden = true;
  $('start').innerHTML = icon('video') + ' Finish round';
  run(current);
}

function run(current) {
  current.started = performance.now();
  current.date = new Date().toISOString();
  if (current.demo) {
    $('source').textContent = 'DEMO SIGNAL';
    setState('Demo', true);
    current.frame = requestAnimationFrame(now => tick(current, now));
  } else renderCameraState(current);
}

function tick(current, now) {
  if (session !== current) return;
  try {
    if (!current.demo) updateCameraFrame(current);
    if (current.started !== undefined) analyzeFrame(current, now);
  } catch {
    finish();
    error(current.started === undefined ? 'The camera preview stopped. Open the camera to try again.' :
      'Tracking stopped. Any collected samples are available in Round Review.');
    return;
  }
  current.frame = requestAnimationFrame(next => tick(current, next));
}

function analyzeFrame(current, now) {
  const elapsed = (now - current.started) / 1000;
  $('timer').textContent = $('duration').textContent = formatTime(elapsed);
  if (!current.demo) {
    if (current.videoTime === current.analyzedTime || now - current.lastDetect < 100) return;
    current.analyzedTime = current.videoTime;
    current.lastDetect = now;
    const result = current.model.detectForVideo(cameraFrame, now);
    current.poses = result.landmarks.filter(pose => [11,12,23,24].filter(i => pose[i]?.visibility >= 0.45).length >= 2);
    current.poses.sort((a,b) => torso(a).x - torso(b).x);
    drawPose(current.poses);
    updateFraming(current);
    if (current.movementModel && elapsed - current.lastMovementSample >= sampleInterval) {
      current.lastMovementSample = elapsed;
      const features = poseFeatures(current.poses, cameraFrame.width / cameraFrame.height);
      current.movementWindow = current.movementWindow.filter(observation => elapsed - observation.time < windowSeconds);
      if (features) current.movementWindow.push({time: elapsed, features});
    }
  }
  if (elapsed - current.lastSample >= 1) {
    current.lastSample = elapsed;
    sample(current, elapsed);
  }
}

function updateFraming(current) {
  let cue = 'Find both athletes';
  if (current.poses.length === 2) {
    const points = current.poses.flat().filter(point => point.visibility >= 0.45);
    const left = Math.min(...points.map(point => point.x));
    const right = 1 - Math.max(...points.map(point => point.x));
    const top = Math.min(...points.map(point => point.y));
    const bottom = 1 - Math.max(...points.map(point => point.y));
    const edges = [[left, 'Pan left'], [right, 'Pan right'], [top, 'Tilt up'], [bottom, 'Tilt down']];
    const [space, direction] = edges.reduce((nearest, edge) => edge[0] < nearest[0] ? edge : nearest);
    cue = left + right < 0.16 || top + bottom < 0.16 ? 'Widen the shot' :
      space < 0.09 ? direction : 'Both athletes in frame';
  }
  if (current.framingCandidate !== cue) {
    current.framingCandidate = cue;
    return;
  }
  showFraming(cue);
  if (cue === 'Both athletes in frame') {
    feedback(cue, 'Keep open mat around the roll. Overlay colors may swap as athletes cross.');
  } else if (cue === 'Find both athletes') {
    feedback(cue, 'Follow the roll with a wider view. Overlap can temporarily hide a pose.');
  } else if (cue === 'Widen the shot') {
    feedback(cue, 'Zoom out, choose a wider lens, or step back to give both athletes more room in the frame.');
  } else {
    feedback(cue, 'Follow the athletes and leave more room between them and the edge of the frame.');
  }
}

function torso(pose) {
  const visible = [11,12,23,24].map(i => pose[i]).filter(point => point && point.visibility >= 0.45);
  return {x:visible.reduce((sum,p) => sum + p.x,0) / visible.length, y:visible.reduce((sum,p) => sum + p.y,0) / visible.length};
}

function sample(current, elapsed) {
  const centers = current.poses.map(torso);
  let count = centers.length;
  let motion = null;
  const separation = count === 2 ? { x: centers[1].x - centers[0].x, y: centers[1].y - centers[0].y } : null;
  if (separation && current.previous) {
    // A shared shift in the image cancels out of the vector between athletes.
    const {x, y} = current.previous;
    const change = Math.min(Math.hypot(separation.x - x, separation.y - y), Math.hypot(separation.x + x, separation.y + y));
    motion = change / (2 * (elapsed - current.previousTime));
  }
  current.previous = separation;
  current.previousTime = elapsed;
  if (current.demo) {
    count = Math.floor(elapsed) % 17 < 13 ? 2 : 1;
    motion = count === 2 ? 0.012 + Math.abs(Math.sin(elapsed * 0.73)) * 0.095 : null;
  }
  const row = {t: +elapsed.toFixed(2), count, motion: motion == null ? null : +motion.toFixed(4)};
  if (current.movementModel) {
    row.movementLabel = count === 2 ? current.movementModel(summarizeMovement(current.movementWindow)) : null;
    $('recognized-movement').textContent = row.movementLabel || 'Uncertain';
  }
  current.samples.push(row);
  $('athletes').textContent = count + ' / 2';
  $('coverage').textContent = coverage(current.samples) + '%';
  $('movement').textContent = describeMotion(motion);
  $('tracking-status').textContent = current.demo ? 'Simulated tracking data' : count + ' of 2 poses detected';
  if (current.demo) feedback('Demo in progress', 'The demo simulates changes in visibility and movement. Finish it to explore Round Review.');
}

function finish(save = true) {
  const current = session;
  if (!current) return;
  session = undefined;
  cancelAnimationFrame(current.frame);
  current.stream?.getTracks().forEach(track => track.stop());
  video.srcObject = null;
  video.hidden = true;
  cameraFrame.hidden = true;
  $('crop-label').hidden = true;
  $('poster').hidden = false;
  $('live').classList.remove('camera-open');
  drawPose([]);
  $('start').hidden = true;
  $('start').disabled = false;
  $('preview').hidden = false;
  $('preview').classList.replace('secondary', 'primary');
  $('preview').innerHTML = icon('video') + ' Preview camera';
  $('demo').disabled = false;
  $('camera-settings').hidden = $('lens-control').hidden = $('zoom-control').hidden = true;
  $('lens').disabled = $('zoom').disabled = false;
  $('athlete-a').hidden = $('athlete-b').hidden = true;
  $('classifier-metric').hidden = true;
  $('recognized-movement').textContent = '—';
  showFraming('');
  $('source').textContent = 'CAMERA OFF';
  $('movement-label').textContent = 'Relative movement';
  $('tracking-status').textContent = 'Pose tracking inactive';
  $('timer').textContent = $('duration').textContent = '00:00';
  for (const id of ['athletes','coverage','movement']) $(id).textContent = '—';
  $('mode-note').textContent = 'Round metrics are saved on this device when you finish. Camera footage stays in the live view.';
  setState('Standby', false);
  feedback('Set up your camera', 'Use a wide sideline view so both athletes stay in frame throughout the roll.');
  if (!save || !current.samples.length) return;
  const round = { id: crypto.randomUUID(), date: current.date, demo: current.demo,
    duration: (performance.now() - current.started) / 1000, samples: current.samples,
    motionBasis: current.demo ? 'simulated' : 'athlete-relative' };
  rounds = [round, ...rounds];
  try { localStorage.setItem(storageKey, JSON.stringify(rounds)); }
  catch { error('Browser storage is full or unavailable. Export this round before closing the page.'); }
  selected = round.id;
  switchView('review');
}

function renderReview() {
  const round = rounds.find(item => item.id === selected) || rounds[0];
  selected = round?.id;
  renderLists();
  $('report').hidden = !round;
  $('review-empty').hidden = !!round;
  $('export').disabled = !round;
  if (!round) return;
  $('report-title').textContent = round.demo ? 'Demo round' : 'Live roll';
  $('report-meta').textContent = formatTime(round.duration) + ' duration · ' + coverage(round.samples) + '% two-athlete coverage' + (round.demo ? ' · Simulated data' : '');
  const relative = round.motionBasis === 'athlete-relative';
  $('sample-motion-label').textContent = relative ? 'Relative movement' : 'Movement';
  $('sample-classifier').hidden = !round.samples.some(sample => 'movementLabel' in sample);
  const gaps = round.samples.filter(s => s.count < 2);
  const peak = round.samples.reduce((best,s) => s.motion != null && (best == null || s.motion > best.motion) ? s : best, null);
  $('summary-title').textContent = 'Round summary';
  const gapText = gaps.length ? gaps.length + ' sampled moments tracked fewer than two athletes, beginning at ' + formatTime(gaps[0].t) + '.' : 'Both athletes were detected at every sampled moment.';
  $('summary-text').textContent = gapText + (peak ? ' Peak ' + (relative ? 'relative' : 'tracked') + ' movement occurred at ' + formatTime(peak.t) + '.' : '');
  $('scrubber').max = round.samples.length - 1;
  $('scrubber').value = 0;
  showSample();
}

function showSample() {
  const round = rounds.find(item => item.id === selected);
  if (!round || $('review').hidden) return;
  const sample = round.samples[Number($('scrubber').value)] || round.samples[0];
  $('sample-time').textContent = formatTime(sample.t);
  $('sample-count').textContent = sample.count + ' / 2';
  $('sample-motion').textContent = describeMotion(sample.motion);
  $('sample-movement-label').textContent = sample.movementLabel || 'Uncertain';
  const chart = $('chart');
  resizeCanvas(chart);
  const ctx = chart.getContext('2d');
  const ratio = chart.width / Math.max(1, chart.clientWidth);
  ctx.setTransform(ratio,0,0,ratio,0,0);
  const w = chart.width / ratio;
  const h = chart.height / ratio;
  const y1 = h - 28;
  const x = t => 12 + t / Math.max(1, round.duration) * (w - 24);
  ctx.clearRect(0,0,w,h);
  ctx.font = '12px system-ui';
  const motionLabel = $('sample-motion-label').textContent;
  ctx.fillStyle = colors[0]; ctx.fillText(motionLabel, 12, 16);
  ctx.fillStyle = colors[1]; ctx.fillText('Both athletes tracked', 26 + ctx.measureText(motionLabel).width, 16);
  ctx.strokeStyle = '#334358';
  for (let i=0;i<4;i++) {
    const y = 32 + (y1-32) * i/3;
    ctx.beginPath(); ctx.moveTo(12,y); ctx.lineTo(w-12,y); ctx.stroke();
  }
  for (const point of round.samples) if (point.count === 2) {
    ctx.fillStyle = '#63e4c847';
    ctx.fillRect(x(point.t), y1-6, Math.max(2,(w-24)/Math.max(round.duration,1)), 6);
  }
  const max = round.samples.reduce((high,s) => Math.max(high,s.motion ?? 0),0.12);
  ctx.strokeStyle = ctx.fillStyle = colors[0];
  ctx.lineWidth = 2;
  let previous;
  for (const point of round.samples) {
    if (point.motion == null) { previous = undefined; continue; }
    const y = y1 - point.motion / max * (y1-35);
    ctx.beginPath();
    if (previous) { ctx.moveTo(previous.x,previous.y); ctx.lineTo(x(point.t),y); ctx.stroke(); }
    else { ctx.arc(x(point.t), y, 2, 0, Math.PI*2); ctx.fill(); }
    previous = {x:x(point.t),y};
  }
  ctx.strokeStyle = '#f0f4fc';
  ctx.beginPath(); ctx.moveTo(x(sample.t),28); ctx.lineTo(x(sample.t),y1+5); ctx.stroke();
  ctx.fillStyle = '#a2afc2';
  ctx.fillText('00:00', 12, h-7);
  const end = formatTime(round.duration);
  ctx.fillText(end,w-12-ctx.measureText(end).width,h-7);
}

document.querySelectorAll('.tab[data-view]').forEach(tab => tab.onclick = () => switchView(tab.dataset.view));
$('back-live').onclick = () => switchView('live');
$('all-rounds').onclick = () => switchView('review');
$('preview').onclick = () => session ? finish(false) : openCamera();
$('start').onclick = () => {
  if (!session) return;
  if (session.started !== undefined) finish();
  else startAnalysis(session);
};
$('demo').onclick = startDemo;
$('lens').onchange = () => {
  const deviceId = $('lens').value;
  finish(false);
  openCamera(deviceId);
};
$('zoom').onchange = changeZoom;
$('wide-view').onclick = () => {
  const wide = $('live-workspace').classList.toggle('wide');
  $('wide-view').setAttribute('aria-pressed', String(wide));
  $('wide-view').textContent = wide ? 'Standard view' : 'Wide view';
};
$('show-overlay').onchange = () => drawPose(session?.poses || []);
$('scrubber').oninput = showSample;
$('export').onclick = () => {
  const round = rounds.find(item => item.id === selected);
  if (!round) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(round,null,2)],{type:'application/json'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'matvision-' + round.date.slice(0,10) + '-' + round.id.slice(0,8) + '.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
};
new ResizeObserver(showSample).observe($('chart'));
renderLists();
