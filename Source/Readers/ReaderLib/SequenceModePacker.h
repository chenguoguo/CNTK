//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Reader.h"
#include "MemoryProvider.h"
#include "Transformer.h"
#include "Packer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Represents a sequence
struct SequenceWrapper;
typedef std::shared_ptr<SequenceWrapper> SequenceWrapperPtr;

// A sample packer that densely packs samples in parallel for GPU consumptions.
class SequenceModePacker : public Packer
{
public:
    SequenceModePacker(
        MemoryProviderPtr memoryProvider,
        TransformerPtr transformer,
        size_t minibatchSize,
        size_t parallelNumberOfSequences,
        const std::vector<StreamDescriptionPtr>& streams);

    virtual Minibatch ReadMinibatch() override;

private:
    std::shared_ptr<char> AllocateBuffer(size_t numElements, size_t elementSize);
    size_t GetSampleSize(StreamDescriptionPtr stream);
    bool GetNextSequence(SequenceWrapperPtr& sequence);
    MBLayoutPtr CreateStreamMBLayout(size_t streamId, const std::vector<std::vector<SequenceWrapperPtr>>& preparedSequences);
    Minibatch PackMinibatch(const std::vector<std::vector<SequenceWrapperPtr>>& preparedSequences);
    void PackStreamMinibatch(size_t streamId, const std::vector<std::vector<SequenceWrapperPtr>>& preparedSequences);
    void PackSparseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t elementSize, size_t sampleSize);
    void PackDenseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t elementSize, size_t sampleSize);

    MemoryProviderPtr m_memoryProvider;
    TransformerPtr m_transformer;
    std::vector<StreamDescriptionPtr> m_outputStreams;
    std::vector<StreamDescriptionPtr> m_inputStreams;

    size_t m_minibatchSize;
    size_t m_parallelNumberOfSequences;

    SequenceWrapperPtr m_currentSequence;

    // A matrix of prepared sequences. The number of rows(RN) = m_parallelNumberOfSequences,
    // in each row we try to fit as many sequences as possible, not exceeding
    // the m_maxLength of samples. m_maxLength is defined as the max length between the first m_parallelNumberOfSequences sequences.
    // It looks something like that:
    //  /***s11***/ /***s12**/
    //  ....
    //  /**********sM1*********/   <-- MaxSize equals its length, because sM1 was the longest among [s11 ... sRN1]
    //  ....
    // /*sRN1*//*sRN2*//*sRN2*/
    // This logic of packing is implemented to support current tests.
    std::vector<std::vector<SequenceWrapperPtr>> m_preparedSequences;
    size_t m_maxLength;

    std::vector<std::shared_ptr<char>> m_streamBuffers;
    std::vector<size_t> m_streamBufferSizes;
};

typedef std::shared_ptr<SequenceModePacker> SequenceModePackerPtr;

}}}
