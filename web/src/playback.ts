export function padChannelForPlayback(channel: Float32Array, commonFrames: number): Float32Array {
  const padded = new Float32Array(commonFrames);
  padded.set(channel.subarray(0, commonFrames));
  return padded;
}

export function startTogether(
  dry: Pick<AudioBufferSourceNode, "start" | "onended">,
  wet: Pick<AudioBufferSourceNode, "start" | "onended">,
  currentTime: number,
  onWetEnded: () => void,
): void {
  dry.onended = null;
  wet.onended = onWetEnded;
  const startAt = currentTime + 0.02;
  dry.start(startAt);
  wet.start(startAt);
}

export function stopSources(
  sources: Pick<AudioBufferSourceNode, "stop" | "disconnect">[],
): void {
  for (const source of sources) {
    try {
      source.stop();
    } catch (reason) {
      if (!(reason instanceof DOMException && reason.name === "InvalidStateError")) throw reason;
    }
    source.disconnect();
  }
}
