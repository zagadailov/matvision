const trainingKey = 'matvision-movement-examples-v1';
const researchKey = 'matvision-research-baseline-enabled';

export const movementLabels = ['Standing movement', 'Ground movement', 'Position change', 'Reset or pause'];
export const windowSeconds = 2;
export const sampleInterval = 0.25;

export function readExamples() {
  try {
    const examples = JSON.parse(localStorage.getItem(trainingKey) || '[]');
    return Array.isArray(examples) ? examples : [];
  } catch {
    return [];
  }
}

export function saveExamples(examples) {
  localStorage.setItem(trainingKey, JSON.stringify(examples));
}

export function researchModelEnabled() {
  try { return localStorage.getItem(researchKey) === 'true'; }
  catch { return false; }
}

export function setResearchModelEnabled(enabled) {
  localStorage.setItem(researchKey, String(enabled));
}

function midpoint(pose, indices, aspect) {
  const points = indices.map(index => pose[index]).filter(point => point?.visibility >= 0.45);
  if (!points.length) return null;
  return {
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length / aspect
  };
}

function torso(pose, aspect) {
  const shoulders = midpoint(pose, [11, 12], aspect);
  const hips = midpoint(pose, [23, 24], aspect);
  if (!shoulders || !hips) return null;
  const width = Math.abs(shoulders.x - hips.x);
  const height = Math.abs(shoulders.y - hips.y);
  const length = Math.hypot(width, height);
  return {
    x: (shoulders.x + hips.x) / 2,
    y: (shoulders.y + hips.y) / 2,
    length,
    upright: height / Math.max(0.005, length)
  };
}

export function poseFeatures(poses, aspect) {
  if (poses.length !== 2) return null;
  const [a, b] = poses.map(pose => torso(pose, aspect));
  if (!a || !b) return null;
  const scale = Math.max(0.025, (a.length + b.length) / 2);
  return [
    Math.min(5, Math.abs(a.x - b.x) / scale),
    Math.min(5, Math.abs(a.y - b.y) / scale),
    Math.min(a.upright, b.upright),
    Math.max(a.upright, b.upright)
  ];
}

export function summarizeMovement(samples) {
  if (samples.length < 5 || samples.at(-1).time - samples[0].time < 1) return null;
  const means = [];
  const ranges = [];
  for (let i = 0; i < 4; i++) {
    const values = samples.map(sample => sample.features[i]);
    means.push(values.reduce((sum, value) => sum + value, 0) / values.length);
    ranges.push(Math.max(...values) - Math.min(...values));
  }
  return [...means, ...ranges];
}

export function trainMovement(examples) {
  const groups = new Map();
  for (const {label, features} of examples.filter(Boolean)) {
    if (!movementLabels.includes(label) || !Array.isArray(features) || features.length !== 8 || !features.every(Number.isFinite)) continue;
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(features);
  }
  if (groups.size < 2) return null;

  const rows = [...groups.values()].flat();
  const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
  const spreads = rows[0].map((_, index) => {
    const values = rows.map(row => row[index]);
    const center = mean(values);
    return Math.max(0.1, Math.sqrt(mean(values.map(value => (value - center) ** 2))));
  });
  const neighbors = [...groups].flatMap(([label, vectors]) => vectors.map(vector => ({label, vector})));

  return features => {
    if (!features) return null;
    const ranked = neighbors.map(({label, vector}) => ({
      label,
      distance: mean(features.map((value, index) => ((value - vector[index]) / spreads[index]) ** 2))
    })).sort((a, b) => a.distance - b.distance);
    if (ranked[0].distance >= 2.25) return null;
    const votes = new Map();
    let total = 0;
    for (const {label, distance} of ranked.slice(0, 5)) {
      const weight = 1 / (0.05 + distance);
      votes.set(label, (votes.get(label) || 0) + weight);
      total += weight;
    }
    const [label, weight] = [...votes].sort((a, b) => b[1] - a[1])[0];
    return weight / total >= 0.70 ? label : null;
  };
}
