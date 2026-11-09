import { describe, expect, it, vi } from "vitest";
import { padChannelForPlayback, startTogether, stopSources } from "./playback";

describe("longer wet tails", () => {
  it("pads dry to the identical wet period without changing dry samples", () => {
    const dry = new Float32Array([0.1, -0.2, 0.3]);
    const padded = padChannelForPlayback(dry, 7);
    expect(padded).toHaveLength(7);
    expect([...padded.subarray(0, 3)]).toEqual([...dry]);
    expect([...padded.subarray(3)]).toEqual([0, 0, 0, 0]);
  });

  it("starts together and retains playback ownership until wet ends", () => {
    const dry: Pick<AudioBufferSourceNode, "start" | "onended"> = { start: vi.fn(), onended: null };
    const wet: Pick<AudioBufferSourceNode, "start" | "onended"> = { start: vi.fn(), onended: null };
    const onComplete = vi.fn();
    startTogether(dry, wet, 12, onComplete);
    expect(dry.start).toHaveBeenCalledWith(12.02);
    expect(wet.start).toHaveBeenCalledWith(12.02);
    expect(dry.onended).toBeNull();
    expect(onComplete).not.toHaveBeenCalled();
    expect(wet.onended).toBe(onComplete);
    onComplete();
    expect(onComplete).toHaveBeenCalledOnce();
  });

  it("stop disconnects both sources, including an already ended dry node", () => {
    const dry = { stop: vi.fn(() => { throw new DOMException("Ended", "InvalidStateError"); }), disconnect: vi.fn() };
    const wet = { stop: vi.fn(), disconnect: vi.fn() };
    stopSources([dry, wet]);
    expect(dry.stop).toHaveBeenCalledOnce();
    expect(wet.stop).toHaveBeenCalledOnce();
    expect(dry.disconnect).toHaveBeenCalledOnce();
    expect(wet.disconnect).toHaveBeenCalledOnce();
  });
});
