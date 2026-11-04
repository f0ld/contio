// Speech to text through contiod (`/v1/audio/transcriptions`), and microphone
// recording as WAV (whisper.cpp decodes WAV, MP3, FLAC and OGG, but not the
// WebM/Opus a browser records).

import { DAEMON_URL } from "./daemon";

export type ResponseFormat = "text" | "json" | "verbose_json" | "srt" | "vtt";

export interface TranscribeOptions {
  language?: string;
  translate?: boolean;
  format: ResponseFormat;
}

export interface Transcript {
  /** Plain text, or the raw body for srt/vtt/verbose_json. */
  text: string;
  language?: string;
  seconds: number;
}

export async function transcribe(model: string, audio: Blob, fileName: string, opts: TranscribeOptions): Promise<Transcript> {
  const form = new FormData();
  form.append("model", model);
  form.append("file", audio, fileName);
  form.append("response_format", opts.format);
  if (opts.language) form.append("language", opts.language);
  const started = performance.now();
  const res = await fetch(`${DAEMON_URL}/v1/audio/${opts.translate ? "translations" : "transcriptions"}`, {
    method: "POST",
    body: form,
  });
  const body = await res.text();
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      message = JSON.parse(body).error?.message ?? message;
    } catch {
      // not JSON
    }
    throw new Error(message);
  }
  const seconds = (performance.now() - started) / 1000;
  if (opts.format === "json") return { text: JSON.parse(body).text?.trim() ?? "", seconds };
  if (opts.format === "verbose_json") {
    const v = JSON.parse(body);
    return { text: JSON.stringify(v, null, 2), language: v.language, seconds };
  }
  return { text: body.trim(), seconds };
}

const SAMPLE_RATE = 16000;

/** Records the microphone until `stop()` is called; resolves to 16 kHz mono WAV. */
export async function startRecording(): Promise<{ stop: () => Promise<Blob> }> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  const chunks: Blob[] = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.start();
  return {
    stop: () =>
      new Promise((resolve, reject) => {
        recorder.onstop = async () => {
          stream.getTracks().forEach((t) => t.stop());
          try {
            resolve(await toWav(new Blob(chunks, { type: recorder.mimeType })));
          } catch (e) {
            reject(e);
          }
        };
        recorder.stop();
      }),
  };
}

async function toWav(blob: Blob): Promise<Blob> {
  const decoded = await new AudioContext().decodeAudioData(await blob.arrayBuffer());
  const frames = Math.ceil(decoded.duration * SAMPLE_RATE);
  const offline = new OfflineAudioContext(1, frames, SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const pcm = (await offline.startRendering()).getChannelData(0);

  const out = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const text = (at: number, s: string) => [...s].forEach((c, i) => out.setUint8(at + i, c.charCodeAt(0)));
  text(0, "RIFF");
  out.setUint32(4, 36 + pcm.length * 2, true);
  text(8, "WAVEfmt ");
  out.setUint32(16, 16, true); // fmt chunk size
  out.setUint16(20, 1, true); // PCM
  out.setUint16(22, 1, true); // mono
  out.setUint32(24, SAMPLE_RATE, true);
  out.setUint32(28, SAMPLE_RATE * 2, true); // byte rate
  out.setUint16(32, 2, true); // block align
  out.setUint16(34, 16, true); // bits per sample
  text(36, "data");
  out.setUint32(40, pcm.length * 2, true);
  pcm.forEach((v, i) => out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, v)) * 0x7fff, true));
  return new Blob([out], { type: "audio/wav" });
}
