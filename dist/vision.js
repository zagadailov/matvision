const models = new Map();

export function getPoseModel(mode) {
  if (!models.has(mode)) {
    const model = (async () => {
      const {FilesetResolver, PoseLandmarker} = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/vision_bundle.mjs');
      const files = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm');
      return PoseLandmarker.createFromOptions(files, {
        baseOptions: {modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'},
        runningMode: mode,
        numPoses: 2,
        minPoseDetectionConfidence: 0.45,
        minPosePresenceConfidence: 0.45
      });
    })().catch(err => { models.delete(mode); throw err; });
    models.set(mode, model);
  }
  return models.get(mode);
}

export function frameGeometry(width, height) {
  const cropped = width <= height;
  const cropHeight = cropped ? width * 9 / 16 : height;
  const scale = Math.min(1, 1280 / width, 720 / cropHeight);
  return {
    cropped, sourceWidth: width, sourceHeight: height,
    top: (height - cropHeight) / 2, cropHeight,
    width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(cropHeight * scale))
  };
}

export function drawCameraFrame(context, source, geometry) {
  context.drawImage(source, 0, geometry.top, geometry.sourceWidth, geometry.cropHeight,
    0, 0, geometry.width, geometry.height);
}
