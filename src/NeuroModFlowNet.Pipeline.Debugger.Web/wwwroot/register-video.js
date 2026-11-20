const headerSize = 61;

export class RegisterVideoView {
  constructor(open, canvas, stats) { Object.assign(this, {open,canvas,stats}); }
  stop() { this.abort?.abort(); }
  async show(selection) {
    this.stop();
    const abort = this.abort = new AbortController();
    await this.stream(selection, abort.signal);
  }

  async stream(selection, signal) {
    if(!('VideoDecoder' in globalThis)) throw new Error('WebCodecs is unavailable in this browser context.');
    let shown = 0, received = 0, decoderError, firstAt;
    const metadata = new Map();
    const context = this.canvas.getContext('2d', {alpha:false});
    const decoder = new VideoDecoder({
      output: frame => {
        try {
          const sample = metadata.get(frame.timestamp);
          metadata.delete(frame.timestamp);
          if(signal.aborted) return;
          if(this.canvas.width !== frame.displayWidth) this.canvas.width = frame.displayWidth;
          if(this.canvas.height !== frame.displayHeight) this.canvas.height = frame.displayHeight;
          context.drawImage(frame, 0, 0);
          shown++;
          const seconds = Math.max(.001, (performance.now() - firstAt) / 1000);
          const {run, seen, skipped, encoded, capture, encode} = sample;
          this.stats.textContent = `${shown}/${received} · ${(shown / seconds).toFixed(1)} fps · #${run} · ${encoded}/${seen} (−${skipped})`;
          this.stats.dataset.tooltip = this.stats.textContent;
          Object.assign(this.stats.dataset, {shown, received, seen, skipped, encoded, capture, encode, run});
        } finally { frame.close(); }
      },
      error: error => { decoderError = error; }
    });
    let reader;
    try {
      const response = await this.open(selection, signal);
      reader = response.body.getReader();
      let rest = new Uint8Array(0);
      while(!signal.aborted) {
        const {done, value} = await reader.read();
        if(done) throw decoderError ?? new Error('H.264 stream ended.');
        const buffer = new Uint8Array(rest.length + value.length);
        buffer.set(rest); buffer.set(value, rest.length);
        let offset = 0;
        while(buffer.length - offset >= headerSize) {
          const view = new DataView(buffer.buffer, offset);
          const length = view.getInt32(0, true);
          if(length <= 0 || length > 4_194_304) throw new Error('Invalid H.264 packet length.');
          if(buffer.length - offset < headerSize + length) break;
          const data = buffer.subarray(offset + headerSize, offset + headerSize + length);
          offset += headerSize + length;
          const number = index => Number(view.getBigInt64(index, true));
          const key = view.getUint8(4) === 1;
          if(decoderError) throw decoderError;
          if(decoder.state === 'unconfigured') {
            const codec = annexBCodec(data);
            if(!key || !codec) continue;
            decoder.configure({codec, optimizeForLatency:true});
          }
          while(decoder.decodeQueueSize >= 4 && !signal.aborted && !decoderError)
            await new Promise(resolve => setTimeout(resolve, 8));
          if(signal.aborted) return;
          if(decoderError) throw decoderError;
          firstAt ??= performance.now();
          received++;
          const timestamp = number(5);
          if(metadata.size >= 16) throw new Error('The browser decoder is not returning frames.');
          metadata.set(timestamp, {run:number(13), seen:number(21), skipped:number(29), encoded:number(37), capture:number(45), encode:number(53)});
          decoder.decode(new EncodedVideoChunk({type:key ? 'key' : 'delta', timestamp, data}));
        }
        rest = buffer.slice(offset);
      }
    } finally {
      if(reader) await reader.cancel().catch(() => {});
      if(decoder.state !== 'closed') decoder.close();
    }
  }
}

function annexBCodec(data) {
  for(let index = 2; index + 4 < data.length; index++) {
    if(data[index] !== 1 || data[index - 1] !== 0 || data[index - 2] !== 0 || (data[index + 1] & 31) !== 7) continue;
    return 'avc1.' + Array.from(data.subarray(index + 2, index + 5), value => value.toString(16).padStart(2, '0')).join('');
  }
  return null;
}
