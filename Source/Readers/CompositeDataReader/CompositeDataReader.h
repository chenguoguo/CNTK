//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <map>
#include <string>
#include <future>
#include "DataReader.h"
#include <Reader.h>

namespace Microsoft { namespace MSR { namespace CNTK {

class IDataDeserializer;
typedef std::shared_ptr<IDataDeserializer> IDataDeserializerPtr;

class Transformer;
typedef std::shared_ptr<Transformer> TransformerPtr;

class Packer;
typedef std::shared_ptr<Packer> PackerPtr;

class MemoryProvider;
typedef std::shared_ptr<MemoryProvider> MemoryProviderPtr;

class CorpusDescriptor;
typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;

struct StreamDescription;
typedef std::shared_ptr<StreamDescription> StreamDescriptionPtr;

struct EpochConfiguration;
struct Minibatch;

// The whole CompositeDataReader is meant as a stopgap to allow deserializers/transformers composition until SGD talkes 
// directly to the new Reader API. The example of the cntk configuration that this reader supports can be found at
//     Tests/EndToEndTests/Speech/ExperimentalHtkmlfReader/LSTM/FullUtterance/cntk.cntk
// CompositeDataReader is a factory for the new readers. Its main responsibility is to read the configuration and create the
// corresponding set of deserializers, the corpus descriptor, transformers, randomizer and packer, providing the following functionality:
//     - all input sequences are defined by the corpus descriptor
//     - deserializers provide sequences according to the corpus descriptor
//     - sequences can be transformed by the transformers applied on top of deserializer (TODO: not yet in place)
//     - deserializers are bound together using the bundler - it bundles sequences with the same sequence id retrieved from different deserializers
//     - packer is used to pack randomized sequences into the minibatch
// The composite data reader is currently also responsible for asynchronous prefetching of the minibatch data.

// In order not to break existing configs and allow deserializers composition it exposes the same interface as the old readers, but it is not exposed
// to external developers. The actual "reader developer" now has to provide deserializer(s) only.
// TODO: Implement proper corpus descriptor.
// TODO: Add transformers as the next step.
// TODO: Same code as in ReaderLib shim, the one in the ReaderLib will be deleted as the next step.
// TODO: Change this interface when SGD is changed.
class CompositeDataReader : public Reader, protected Plugin
{
public:
    CompositeDataReader(const ConfigParameters& parameters, MemoryProviderPtr provider);

    // Describes the streams this reader produces.
    std::vector<StreamDescriptionPtr> GetStreamDescriptions() override;

    // Starts a new epoch with the provided configuration
    void StartEpoch(const EpochConfiguration& config) override;

    // Reads a minibatch that contains data across all streams.
    Minibatch ReadMinibatch() override;

private:
    void CreateDeserializers(const ConfigParameters& readerConfig);
    IDataDeserializerPtr CreateDeserializer(const ConfigParameters& readerConfig, bool primary);


    enum class PackingMode
    {
        sample,
        sequence,
        truncated
    };

    // Packing mode.
    PackingMode m_packingMode;

    // Pre-fetch task.
    std::future<Minibatch> m_prefetchTask;

    // Launch type of prefetch - async or sync.
    launch m_launchType;

    // Flag indicating end of the epoch.
    bool m_endOfEpoch;

    // MBLayout of the reader. 
    // TODO: Will be taken from the StreamMinibatchInputs.
    MBLayoutPtr m_layout;

    // Stream name to id mapping.
    std::map<std::wstring, size_t> m_nameToStreamId;

    // All streams this reader provides.
    std::vector<StreamDescriptionPtr> m_streams;

    // A list of deserializers.
    std::vector<IDataDeserializerPtr> m_deserializers;

    // Randomizer.
    // TODO: remove Transformer interface from randomizer.
    TransformerPtr m_randomizer;

    // TODO: Should be removed. We already have matrices on this level.
    // Should just get the corresponding pinned memory.
    MemoryProviderPtr m_provider;

    // Corpus descriptor that is shared between deserializers.
    CorpusDescriptorPtr m_corpus;

    // Packer.
    PackerPtr m_packer;

    // Precision - "float" or "double".
    std::string m_precision;

    // Truncation length for BPTT mode.
    size_t m_truncationLength;
};

}}}
