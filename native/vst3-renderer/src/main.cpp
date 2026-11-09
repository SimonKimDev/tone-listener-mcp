#include "public.sdk/source/common/memorystream.h"
#include "public.sdk/source/vst/hosting/hostclasses.h"
#include "public.sdk/source/vst/hosting/module.h"
#include "public.sdk/source/vst/hosting/plugprovider.h"
#include "pluginterfaces/vst/ivstaudioprocessor.h"
#include "pluginterfaces/vst/ivsteditcontroller.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>

bool validSinglePresetState(std::string_view json);

namespace {

using namespace Steinberg;
using namespace Steinberg::Vst;

constexpr auto kPluginPath = "/Library/Audio/Plug-Ins/VST3/Soundshed Guitar.vst3";
constexpr int32 kSampleRate = 44100;
constexpr int32 kBlockSize = 512;
constexpr size_t kMaxStateBytes = 8 * 1024 * 1024;
constexpr size_t kMaxFrames = static_cast<size_t>(kSampleRate) * 40;
constexpr std::string_view kJuceMarker = "JUCEPrivateData";

class HostStateError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void setLimit(int resource, rlim_t value) {
    rlimit current{};
    if (::getrlimit(resource, &current) != 0) {
        throw HostStateError("cannot-read-render-limits");
    }
    const rlim_t bounded = current.rlim_max == RLIM_INFINITY
                               ? value
                               : std::min(value, current.rlim_max);
    const rlimit limit{bounded, bounded};
    if (::setrlimit(resource, &limit) != 0) {
        throw HostStateError("cannot-set-render-limits");
    }
}

void setRenderLimits() {
    setLimit(RLIMIT_CPU, 90);
    setLimit(RLIMIT_FSIZE, 16ULL * 1024 * 1024);
}

std::string readTextFile(const std::filesystem::path& path, size_t limit) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec || size == 0 || size > limit) {
        throw HostStateError("invalid-state-file");
    }

    std::ifstream input(path, std::ios::binary);
    std::string contents(static_cast<size_t>(size), '\0');
    input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!input || contents.front() != '{') {
        throw HostStateError("invalid-state-file");
    }
    return contents;
}

size_t trailerOffset(std::string_view bytes) {
    const auto size = bytes.size();
    // The wrapper includes an eight-byte separator BEFORE the private payload.
    // Omitting it makes JUCE remove the last eight bytes of the processor JSON.
    if (size < kJuceMarker.size() + sizeof(uint64_t) ||
        bytes.substr(size - kJuceMarker.size()) != kJuceMarker) {
        throw HostStateError("unsupported-juce-state-trailer");
    }
    const auto lengthOffset = size - kJuceMarker.size() - sizeof(uint64_t);
    uint64_t privateSize = 0;
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
        privateSize |= static_cast<uint64_t>(
            static_cast<unsigned char>(bytes[lengthOffset + i])) << (8 * i);
    }
    const auto trailerSize = privateSize + 2 * sizeof(uint64_t) + kJuceMarker.size();
    if (trailerSize < 32 || trailerSize > 512 || trailerSize >= size) {
        throw HostStateError("unsupported-juce-state-trailer");
    }
    const auto start = size - static_cast<size_t>(trailerSize);
    if (bytes.substr(start, sizeof(uint64_t)) != std::string(sizeof(uint64_t), '\0') ||
        bytes.substr(start + sizeof(uint64_t), kJuceMarker.size()) != kJuceMarker) {
        throw HostStateError("unsupported-juce-state-trailer");
    }
    return start;
}

void writePrivateTrailer(const std::filesystem::path& path, MemoryStream& state) {
    const std::string_view bytes(state.getData(), static_cast<size_t>(state.getSize()));
    const auto start = trailerOffset(bytes);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data() + start, static_cast<std::streamsize>(bytes.size() - start));
    if (!output) {
        throw HostStateError("cannot-write-state-trailer");
    }
}

std::vector<float> readMonoInput(const std::filesystem::path& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec || size == 0 || size % sizeof(float) != 0 || size / sizeof(float) > kMaxFrames) {
        throw HostStateError("invalid-audio-input");
    }

    std::vector<float> samples(static_cast<size_t>(size / sizeof(float)));
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(samples.data()), static_cast<std::streamsize>(size));
    if (!input || !std::all_of(samples.begin(), samples.end(), [](float sample) {
            return std::isfinite(sample) && std::abs(sample) <= 4.0f;
        })) {
        throw HostStateError("invalid-audio-input");
    }
    return samples;
}

void writeFloatFile(const std::filesystem::path& path, const std::vector<float>& samples) {
    if (samples.empty() || samples.size() > kMaxFrames * 2) {
        throw HostStateError("invalid-render-output");
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(samples.data()),
                 static_cast<std::streamsize>(samples.size() * sizeof(float)));
    if (!output) {
        throw HostStateError("cannot-write-render-output");
    }
}

bool successful(tresult result) {
    return result == kResultOk || result == kResultTrue;
}

class PluginInstance {
public:
    explicit PluginInstance(const std::string& stateJson) {
        initialize(&stateJson, true);
    }

    PluginInstance() {
        initialize(nullptr, false);
    }

    void initialize(const std::string* stateJson, bool prepareAudio) {
        std::string error;
        module = VST3::Hosting::Module::create(kPluginPath, error);
        if (!module) {
            throw HostStateError("soundshed-plugin-load-failed");
        }

        static HostApplication hostApplication;
        auto factory = module->getFactory();
        factory.setHostContext(&hostApplication);
        PluginContextFactory::instance().setPluginContext(&hostApplication);

        const auto classes = factory.classInfos();
        const auto found = std::find_if(classes.begin(), classes.end(), [](const auto& info) {
            return info.category() == kVstAudioEffectClass && info.name() == "Soundshed Guitar";
        });
        if (found == classes.end()) {
            throw HostStateError("soundshed-effect-class-not-found");
        }
        engineVersion = found->version();

        provider = owned(new PlugProvider(factory, *found, true));
        if (!provider->initialize()) {
            throw HostStateError("soundshed-plugin-initialize-failed");
        }
        component = owned(provider->getComponent());
        controller = owned(provider->getController());
        processor = U::cast<IAudioProcessor>(component);
        if (!component || !processor) {
            throw HostStateError("soundshed-audio-processor-unavailable");
        }

        if (stateJson) {
            restoreState(*stateJson);
        }
        if (prepareAudio) {
            prepare();
        }
    }

    ~PluginInstance() {
        if (processing) {
            processor->setProcessing(false);
        }
        if (active) {
            component->setActive(false);
        }
    }

    const std::string& version() const noexcept {
        return engineVersion;
    }

    int32 inputChannels() const noexcept {
        return inputChannelCount;
    }

    int32 outputChannels() const noexcept {
        return outputChannelCount;
    }

    std::vector<float> render(const std::vector<float>& monoInput) {
        const auto frames = monoInput.size();
        std::vector<std::vector<float>> inputChannels(
            static_cast<size_t>(inputChannelCount), std::vector<float>(kBlockSize));
        std::vector<std::vector<float>> outputChannels(
            static_cast<size_t>(outputChannelCount), std::vector<float>(kBlockSize));
        std::vector<Sample32*> inputPointers(static_cast<size_t>(inputChannelCount));
        std::vector<Sample32*> outputPointers(static_cast<size_t>(outputChannelCount));
        for (size_t channel = 0; channel < inputPointers.size(); ++channel) {
            inputPointers[channel] = inputChannels[channel].data();
        }
        for (size_t channel = 0; channel < outputPointers.size(); ++channel) {
            outputPointers[channel] = outputChannels[channel].data();
        }

        AudioBusBuffers inputBus;
        inputBus.numChannels = inputChannelCount;
        inputBus.channelBuffers32 = inputPointers.data();
        AudioBusBuffers outputBus;
        outputBus.numChannels = outputChannelCount;
        outputBus.channelBuffers32 = outputPointers.data();

        ProcessData data;
        data.processMode = kOffline;
        data.symbolicSampleSize = kSample32;
        data.numInputs = 1;
        data.numOutputs = 1;
        data.inputs = &inputBus;
        data.outputs = &outputBus;

        std::vector<float> result(frames * static_cast<size_t>(outputChannelCount));
        for (size_t offset = 0; offset < frames; offset += kBlockSize) {
            const auto count = static_cast<int32>(std::min(
                static_cast<size_t>(kBlockSize), frames - offset));
            for (int32 channel = 0; channel < inputChannelCount; ++channel) {
                auto& buffer = inputChannels[static_cast<size_t>(channel)];
                std::copy_n(monoInput.data() + offset, count, buffer.data());
            }
            for (auto& buffer : outputChannels) {
                std::fill_n(buffer.data(), count, 0.0f);
            }
            data.numSamples = count;
            if (!successful(processor->process(data))) {
                throw HostStateError("soundshed-process-failed");
            }
            for (int32 frame = 0; frame < count; ++frame) {
                for (int32 channel = 0; channel < outputChannelCount; ++channel) {
                    const float sample = outputChannels[static_cast<size_t>(channel)]
                                                       [static_cast<size_t>(frame)];
                    if (!std::isfinite(sample)) {
                        throw HostStateError("nonfinite-render-output");
                    }
                    result[(offset + static_cast<size_t>(frame)) *
                               static_cast<size_t>(outputChannelCount) +
                           static_cast<size_t>(channel)] = sample;
                }
            }
        }

        const double energy = std::inner_product(
            result.begin(), result.end(), result.begin(), 0.0);
        const double rms = std::sqrt(energy / static_cast<double>(result.size()));
        if (!std::isfinite(rms) || rms < 1.0e-8) {
            throw HostStateError("silent-render-output");
        }
        return result;
    }

    std::unique_ptr<MemoryStream> captureState() {
        auto state = std::make_unique<MemoryStream>();
        if (!successful(component->getState(state.get()))) {
            throw HostStateError("soundshed-state-capture-failed");
        }
        return state;
    }

private:
    void restoreState(const std::string& json) {
        auto state = owned(new MemoryStream());
        auto stateBuffer = json;
        if (!successful(state->write(stateBuffer.data(), static_cast<int32>(stateBuffer.size()), nullptr))) {
            throw HostStateError("cannot-prepare-render-state");
        }
        state->seek(0, IBStream::kIBSeekSet, nullptr);
        if (!successful(component->setState(state.get()))) {
            throw HostStateError("soundshed-state-restore-failed");
        }
        if (controller) {
            state->seek(0, IBStream::kIBSeekSet, nullptr);
            if (!successful(controller->setComponentState(state.get()))) {
                throw HostStateError("soundshed-controller-state-restore-failed");
            }
        }
    }

    void prepare() {
        if (component->getBusCount(kAudio, kInput) != 1 ||
            component->getBusCount(kAudio, kOutput) != 1) {
            throw HostStateError("unsupported-audio-bus-layout");
        }

        BusInfo inputInfo{};
        BusInfo outputInfo{};
        if (!successful(component->getBusInfo(kAudio, kInput, 0, inputInfo)) ||
            !successful(component->getBusInfo(kAudio, kOutput, 0, outputInfo)) ||
            inputInfo.channelCount < 1 || inputInfo.channelCount > 2 ||
            outputInfo.channelCount < 1 || outputInfo.channelCount > 2) {
            throw HostStateError("unsupported-audio-channel-layout");
        }
        inputChannelCount = inputInfo.channelCount;
        outputChannelCount = outputInfo.channelCount;

        if (!successful(component->activateBus(kAudio, kInput, 0, true)) ||
            !successful(component->activateBus(kAudio, kOutput, 0, true))) {
            throw HostStateError("soundshed-bus-activation-failed");
        }

        ProcessSetup setup{};
        setup.processMode = kOffline;
        setup.symbolicSampleSize = kSample32;
        setup.maxSamplesPerBlock = kBlockSize;
        setup.sampleRate = kSampleRate;
        if (!successful(processor->setupProcessing(setup))) {
            throw HostStateError("soundshed-processing-setup-failed");
        }
        if (!successful(component->setActive(true))) {
            throw HostStateError("soundshed-activation-failed");
        }
        active = true;
        if (!successful(processor->setProcessing(true))) {
            throw HostStateError("soundshed-processing-start-failed");
        }
        processing = true;
    }

    std::shared_ptr<VST3::Hosting::Module> module;
    IPtr<PlugProvider> provider;
    IPtr<IComponent> component;
    IPtr<IEditController> controller;
    IPtr<IAudioProcessor> processor;
    std::string engineVersion;
    int32 inputChannelCount = 0;
    int32 outputChannelCount = 0;
    bool active = false;
    bool processing = false;
};

void ensureScratchPaths(const std::vector<std::filesystem::path>& paths) {
    std::filesystem::path scratchDirectory;
    for (const auto& path : paths) {
        std::error_code ec;
        const auto canonicalParent = std::filesystem::canonical(path.parent_path(), ec);
        if (ec) {
            throw HostStateError("render-path-outside-private-temporary-directory");
        }
        if (scratchDirectory.empty()) {
            scratchDirectory = canonicalParent;
        } else if (scratchDirectory != canonicalParent) {
            throw HostStateError("render-path-outside-private-temporary-directory");
        }
        const auto filename = path.filename();
        const bool allowedFilename = filename == "state.json" || filename == "input.f32" ||
                                     filename == "output.f32" || filename == "trailer.bin";
        if (canonicalParent.parent_path() != "/private/tmp" ||
            canonicalParent.filename().string().rfind("tone-workbench-render-", 0) != 0 ||
            !allowedFilename) {
            throw HostStateError("render-path-outside-private-temporary-directory");
        }
        struct stat directoryInfo {};
        if (::lstat(canonicalParent.c_str(), &directoryInfo) != 0 ||
            !S_ISDIR(directoryInfo.st_mode) || directoryInfo.st_uid != ::getuid() ||
            (directoryInfo.st_mode & 0077) != 0) {
            throw HostStateError("unsafe-render-directory");
        }
        struct stat fileInfo {};
        if (::lstat(path.c_str(), &fileInfo) == 0 &&
            (!S_ISREG(fileInfo.st_mode) || fileInfo.st_uid != ::getuid())) {
            throw HostStateError("unsafe-render-file");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        setRenderLimits();
        if (argc == 3 && std::string(argv[1]) == "--capture-trailer") {
            const std::filesystem::path trailerPath(argv[2]);
            ensureScratchPaths({trailerPath});
            PlugProvider::setErrorStream(nullptr);
            PluginInstance instance;
            auto state = instance.captureState();
            writePrivateTrailer(trailerPath, *state);
            std::cout << "OK engine=" << instance.version()
                      << " trailer_bytes=" << state->getSize() << '\n';
            return 0;
        }
        if (argc != 4) {
            std::cout << "ERROR invalid-arguments\n";
            return 2;
        }

        const std::filesystem::path statePath(argv[1]);
        const std::filesystem::path inputPath(argv[2]);
        const std::filesystem::path outputPath(argv[3]);
        ensureScratchPaths({statePath, inputPath, outputPath});

        const auto stateJson = readTextFile(statePath, kMaxStateBytes);
        const auto stateEnd = trailerOffset(stateJson);
        if (!validSinglePresetState(std::string_view(stateJson).substr(0, stateEnd))) {
            throw HostStateError("invalid-single-preset-state");
        }
        const auto input = readMonoInput(inputPath);
        PlugProvider::setErrorStream(nullptr);
        PluginInstance instance(stateJson);
        auto output = instance.render(input);

        const double energy =
            std::inner_product(output.begin(), output.end(), output.begin(), 0.0);
        const double rms = std::sqrt(energy / static_cast<double>(output.size()));
        const auto peakIt = std::max_element(output.begin(), output.end(), [](float left, float right) {
            return std::abs(left) < std::abs(right);
        });
        const double peak = std::abs(static_cast<double>(*peakIt));
        writeFloatFile(outputPath, output);
        std::cout << "OK engine=" << instance.version() << " sample_rate=" << kSampleRate
                  << " frames=" << input.size() << " input_channels=" << instance.inputChannels()
                  << " output_channels=" << instance.outputChannels() << " rms=" << rms
                  << " peak=" << peak << '\n';
        return 0;
    } catch (const HostStateError& error) {
        std::cout << "ERROR " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cout << "ERROR renderer-failed\n";
        return 1;
    }
}
