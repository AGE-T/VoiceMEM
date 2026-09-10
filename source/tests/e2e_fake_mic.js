// E2E helper: fake microphone for headless Chromium.
// Patches navigator.mediaDevices.getUserMedia to return a MediaStream fed by
// a looping noise generator (amplitude-modulated white noise, "speechish"),
// so the VoiceMem "ASR test" recorder can capture real audio headlessly.
(() => {
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  const dest = ctx.createMediaStreamDestination();
  const buf = ctx.createBuffer(1, Math.floor(ctx.sampleRate * 2), ctx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < data.length; i++) {
    const env = 0.5 + 0.5 * Math.sin(2 * Math.PI * 3 * i / ctx.sampleRate);
    data[i] = 0.14 * env * (Math.random() * 2 - 1);
  }
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.loop = true;
  src.connect(dest);
  src.start();
  const base = dest.stream;
  const orig = navigator.mediaDevices && navigator.mediaDevices.getUserMedia
    ? navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices)
    : null;
  navigator.mediaDevices.getUserMedia = async (constraints) => {
    if (constraints && constraints.audio) {
      // The click that started the test is the user gesture — resume now.
      try { await ctx.resume(); } catch (e) { /* suspended: silent path */ }
      return base.clone();
    }
    if (orig) return orig(constraints);
    throw new DOMException('no fake video track', 'NotFoundError');
  };
  window.__VM_FAKE_MIC__ = true;
})();
