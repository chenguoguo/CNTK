//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include "SequenceModePacker.h"
#include "ElementTypeUtils.h"

namespace Microsoft { namespace MSR { namespace CNTK {

SequenceModePacker::SequenceModePacker(
    MemoryProviderPtr memoryProvider,
    TransformerPtr transformer,
    size_t minibatchSize,
    size_t parallelNumberOfSequences,
    const std::vector<StreamDescriptionPtr>& streams) : m_transformer(transformer),
    m_minibatchSize(minibatchSize),
    m_outputStreams(streams),
    m_memoryProvider(memoryProvider),
    m_parallelNumberOfSequences(parallelNumberOfSequences)
{
    m_inputStreams = m_transformer->GetStreamDescriptions();
    assert(m_inputStreams.size() == m_outputStreams.size());
    assert(
        std::find_if(
        m_outputStreams.begin(),
        m_outputStreams.end(),
        [](const StreamDescriptionPtr& s)
    {
        return s->m_storageType == StorageType::sparse_csc;
    }) == m_outputStreams.end());

    assert(m_minibatchSize > 0);
    for (int i = 0; i < m_outputStreams.size(); ++i)
    {
        const auto& stream = m_outputStreams[i];

        // Input and output should match in everything except for sparse/dense.
        assert(stream->m_elementType == ElementType::tfloat || stream->m_elementType == ElementType::tdouble);
        assert(stream->m_name == m_inputStreams[i]->m_name);
        assert(stream->m_id == m_inputStreams[i]->m_id);
        assert(GetSampleSize(m_inputStreams[i]) == GetSampleSize(stream));

        m_streamBufferSizes.push_back(0);
        m_streamBuffers.push_back(nullptr);
    }
}

bool SequenceModePacker::GetNextSequence(SequenceWrapperPtr& sequence)
{
    // We always need only a single sequence
    auto s = m_transformer->GetNextSequences(1);
    while (s.m_data.empty() && !s.m_endOfEpoch)
    {
        s = m_transformer->GetNextSequences(1);
    }

    // End of epoch, simply return.
    if (s.m_data.empty())
    {
        return true;
    }

    std::vector<SequenceDataPtr> data;
    for (size_t i = 0; i < s.m_data.size(); ++i)
    {
        // expect only single sequence
        data.push_back(s.m_data[i][0]);
    }

    sequence = std::make_shared<SequenceWrapper>(data);
    return false;
}

Minibatch SequenceModePacker::ReadMinibatch()
{
    size_t sequenceCount = 0;
    m_preparedSequences.clear();

    // Check if there is some data left from the previous read.
    if (m_currentSequence != nullptr)
    {
        m_preparedSequences.push_back(std::vector<SequenceWrapperPtr> { m_currentSequence });
        m_currentSequence = nullptr;
        sequenceCount++;
    }

    // Filling in initial set of m_parallelNumberOfSequences sequences.
    bool endOfEpoch = false;
    while (sequenceCount < m_parallelNumberOfSequences && !endOfEpoch)
    {
        endOfEpoch = GetNextSequence(m_currentSequence);
        m_preparedSequences.push_back(std::vector<SequenceWrapperPtr> { m_currentSequence });
        sequenceCount++;
        m_currentSequence = nullptr;
    }

    assert(m_preparedSequences.size() > m_parallelNumberOfSequences);

    // Ok we have got our m_parallelNumberOfSequences.
    // Let's find the longest.
    m_maxLength = 0;
    for (int i = 0; i < m_preparedSequences.size(); ++i)
    {
        if (m_preparedSequences[i].front()->GetMaxNumberOfSamples() > m_maxLength)
        {
            m_maxLength = m_preparedSequences[i].front()->GetMaxNumberOfSamples();
        }
    }

    // Now maxLength defines the longest possible sequence
    // Let's see whether other sequences can be packed as well.
    std::vector<int> freeSlots(m_preparedSequences.size(), (int)m_maxLength);
    for (int i = 0; i < m_preparedSequences.size(); ++i)
    {
        freeSlots[i] -= (int)m_preparedSequences[i].front()->GetMaxNumberOfSamples();
    }

    // Ok, we identified how many free slots exists per row.
    // Let's fill them in with other sequences.
    endOfEpoch = GetNextSequence(m_currentSequence);
    bool freeSpaceExists = true;
    while (freeSpaceExists && !endOfEpoch)
    {
        freeSpaceExists = false;
        for (int i = 0; i < freeSlots.size(); ++i)
        {
            if (freeSlots[i] - m_currentSequence->GetMaxNumberOfSamples() >= 0)
            {
                freeSpaceExists = true;
                m_preparedSequences[i].push_back(m_currentSequence);
                freeSlots[i] -= (int)m_currentSequence->GetMaxNumberOfSamples();
                break;
            }
        }

        endOfEpoch = GetNextSequence(m_currentSequence);
    }

    // Finished. Now the matrix of prepared sequences has been build.
    // Lets pack it in the format which can be consumed by GPU and create the corresponding MBLaoyuts.
    return PackMinibatch(m_preparedSequences);
}

Minibatch SequenceModePacker::PackMinibatch(const std::vector<std::vector<SequenceWrapperPtr>>& m_preparedSequences)
{
    // Ok, we have all our sequences in order how they have to be packed:
    // Vector of m_parallelNumberOfSequences, in each element is another
    // vector that actually contains sequences not exceeding the max width.
    // Now let's pack them into contigues memory per stream.
    Minibatch result;
    for (size_t i = 0; i < m_outputStreams.size(); ++i)
    {
        PackStreamMinibatch(i, m_preparedSequences);
        MBLayoutPtr layout = CreateStreamMBLayout(i, m_preparedSequences);

        StreamMinibatchPtr m = std::make_shared<StreamMinibatch>();
        m->m_data = m_streamBuffers[i].get();
        m->m_dataSize = m_streamBufferSizes[i] * GetSampleSize(m_outputStreams[i]);
        m->m_layout = layout;
        result.m_data.push_back(m);
    }
    return result;
}

void SequenceModePacker::PackStreamMinibatch(size_t streamId, const std::vector<std::vector<SequenceWrapperPtr>>& preparedSequences)
{
    size_t sampleSize = GetSampleSize(m_outputStreams[streamId]);
    size_t elementSize = GetSizeByType(m_outputStreams[streamId]->m_elementType);
    StorageType storageType = m_inputStreams[streamId]->m_storageType;

    // Get total size of sequences in the stream. Maybe bigger than required.
    size_t totalNumberOfSamplesInStream = preparedSequences.size() * m_maxLength;
    if (totalNumberOfSamplesInStream > m_streamBufferSizes[streamId])
    {
        m_streamBuffers[streamId] = AllocateBuffer(totalNumberOfSamplesInStream, sampleSize);
    }

    // Fill everything with zeros.
    memset(m_streamBuffers[streamId].get(), 0, totalNumberOfSamplesInStream * sampleSize);

    // Identify a stride for a single column.
    size_t stride = sampleSize * preparedSequences.size();

    // Filling up the rows.
    for (int i = 0; i < preparedSequences.size(); ++i)
    {
        size_t columnIndex = 0;
        for (int j = 0; j < preparedSequences[i].size(); ++j)
        {
            auto sequence = preparedSequences[i][j]->m_dataPerStream[streamId];
            for (int k = 0; k < sequence->m_numberOfSamples; ++k)
            {
                char* destination = m_streamBuffers[streamId].get() + columnIndex * stride + i * sampleSize;
                if (storageType == StorageType::dense)
                {
                    PackDenseSample(destination, sequence, k, elementSize, sampleSize);
                }
                else // sparse
                {
                    PackSparseSample(destination, sequence, k, elementSize, sampleSize);
                }

                columnIndex++;
            }
        }
    }
}

void SequenceModePacker::PackSparseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t elementSize, size_t /*sampleSize*/)
{
    SparseSequenceDataPtr s = static_pointer_cast<SparseSequenceData>(sequence);
    size_t nonZeroCount = s->m_indices[sample].size();
    for (size_t nonZeroIndex = 0; nonZeroIndex < nonZeroCount; ++nonZeroIndex)
    {
        memcpy(
            (char*)destination + s->m_indices[sample][nonZeroIndex] * elementSize,
            (const char*)(s->m_data) + nonZeroIndex * elementSize,
            elementSize);
    }
}

void SequenceModePacker::PackDenseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t /*elementSize*/, size_t sampleSize)
{
    memcpy(destination, (char*)(sequence->m_data) + sample * sampleSize, sampleSize);
}

MBLayoutPtr SequenceModePacker::CreateStreamMBLayout(size_t streamId, const std::vector<std::vector<SequenceWrapperPtr>>& preparedSequences)
{
    // TODO: This should simply call InitAsPackedSequences, but his breaks tests right now.
    // TODO: Firstly we pass the tests, then we change this particular logic.

    MBLayoutPtr minibatchLayout = std::make_shared<MBLayout>();
    minibatchLayout->Init(preparedSequences.size(), m_maxLength);

    for (int i = 0; i < m_preparedSequences.size(); ++i)
    {
        size_t begin = 0;
        size_t end = 0;
        // Fill sequences
        for (int j = 0; j < m_preparedSequences[i].size(); ++j)
        {
            end = begin + m_preparedSequences[i][j]->m_dataPerStream[streamId]->m_numberOfSamples;
            minibatchLayout->AddSequence(NEW_SEQUENCE_ID, i, begin, end);
            begin = end;
        }

        // Fill gaps
        if (end != m_maxLength)
        {
            minibatchLayout->AddGap(i, end, m_maxLength);
        }
    }

    return minibatchLayout;
}

size_t SequenceModePacker::GetSampleSize(StreamDescriptionPtr stream)
{
    assert(stream != nullptr);
    size_t elementSize = GetSizeByType(stream->m_elementType);
    return stream->m_sampleLayout->GetNumElements() * elementSize;
}

std::shared_ptr<char> SequenceModePacker::AllocateBuffer(size_t numElements, size_t elementSize)
{
    return std::shared_ptr<char>(
        reinterpret_cast<char*>(m_memoryProvider->Alloc(elementSize, numElements)),
        [this](char* p)
    {
        m_memoryProvider->Free(p);
    });
}

}}}
