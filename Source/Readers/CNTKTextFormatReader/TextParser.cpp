//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <inttypes.h>
#include <cfloat>
#include "Indexer.h"
#include "TextParser.h"
#include "TextReaderConstants.h"

#define isSign(c) ((c == '-' || c == '+'))
#define isE(c) ((c == 'e' || c == 'E'))

namespace Microsoft { namespace MSR { namespace CNTK {

inline bool isDelimiter(char c)
{
    return c == VALUE_DELIMITER || c == NAME_PREFIX || c == COLUMN_DELIMITER ||
        c == INDEX_DELIMITER || c == ROW_DELIMITER || c == CARRIAGE_RETURN;
}

enum State
{
    Init = 0,
    Sign,
    IntegralPart,
    Period,
    FractionalPart,
    TheLetterE,
    ExponentSign,
    Exponent
};

template <class ElemType>
class TextParser<ElemType>::TextDataChunk : public Chunk, public std::enable_shared_from_this<Chunk>
{
public:
    explicit TextDataChunk(const ChunkDescriptor& descriptor, TextParser* parser);

    // Gets sequences by id.
    void GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result) override;

    // A map from sequence ids to the sequence data.
    std::map<size_t, SequenceBuffer> m_sequenceMap;

    // chunk id (copied from the descriptor)
    size_t m_id;
    // Keeps track of how many times GetSequence was called.
    // When this counter value reaches the number of sequences in 
    // the this chunk, it can be safely unloaded.
    size_t m_sequenceRequestCount;

    // a non-owned pointer to the parser that created this chunk
    TextParser* m_parser;
};


template <class ElemType>
struct TextParser<ElemType>::StreamInfo
{
    StorageType m_type;
    size_t m_sampleDimension;
};

template <class ElemType>
TextParser<ElemType>::TextParser(const TextConfigHelper& helper) :
TextParser(helper.GetFilePath(), helper.GetStreams())
{
    SetTraceLevel(helper.GetTraceLevel());
    SetMaxAllowedErrors(helper.GetMaxAllowedErrors());
    SetChunkCacheSize(helper.GetNumChunksToCache());
    SetChunkSize(helper.GetChunkSize());
    SetSkipSequenceIds(helper.ShouldSkipSequenceIds());

    Initialize();
}

template <class ElemType>
TextParser<ElemType>::TextParser(const std::wstring& filename, const vector<StreamDescriptor>& streams) : 
    m_filename(filename),
    m_file(nullptr),
    m_streamInfos(streams.size()),
    m_indexer(nullptr),
    m_fileOffsetStart(0),
    m_fileOffsetEnd(0),
    m_buffer(new char[BUFFER_SIZE + 1]),
    m_bufferStart(nullptr),
    m_bufferEnd(nullptr),
    m_pos(nullptr),
    m_chunkSizeBytes(0),
    m_chunkCacheSize(0),
    m_traceLevel(TraceLevel::Error),
    m_hadWarnings(false),
    m_numAllowedErrors(0),
    m_skipSequenceIds(false),
    m_numRetries(5)
{
    assert(streams.size() > 0);

    m_maxAliasLength = 0;

    for (size_t i = 0; i < streams.size(); ++i)
    {
        const StreamDescriptor& stream = streams[i];
        const string& alias = stream.m_alias;
        if (m_maxAliasLength < alias.length())
        {
            m_maxAliasLength = alias.length();
        }
        m_aliasToIdMap[alias] = i;
        m_streamInfos[i].m_type = stream.m_storageType;
        m_streamInfos[i].m_sampleDimension = stream.m_sampleDimension;

        auto streamDescription = std::make_shared<StreamDescription>(stream);
        streamDescription->m_sampleLayout = std::make_shared<TensorShape>(stream.m_sampleDimension);
        m_streams.push_back(streamDescription);
    }

    assert(m_maxAliasLength > 0);

    m_scratch = unique_ptr<char[]>(new char[m_maxAliasLength + 1]);
}

template <class ElemType>
TextParser<ElemType>::~TextParser()
{
    if (m_file)
    {
        fclose(m_file);
    }
}

template <class ElemType>
void TextParser<ElemType>::PrintWarningNotification()
{
    if (m_hadWarnings && m_traceLevel < Warning)
    {
        fprintf(stderr,
            "A number of warnings were generated while reading input data, "
            "to see them please set 'traceLevel' to a value greater or equal to %d.\n", Warning);
    }
}

template <class ElemType>
void TextParser<ElemType>::Initialize()
{
    if (m_indexer != nullptr)
    {
        return;
    }

    attempt(m_numRetries, [this]()
    {
        m_file = fopenOrDie(m_filename, L"rbS");
    });

    if (funicode(m_file))
    {
        RuntimeError("Found a UTF-16 BOM at the beginning of the input file (%ls). "
            "UTF-16 encoding is currently not supported.", m_filename.c_str());
    }

    m_indexer = make_unique<Indexer>(m_file, m_skipSequenceIds, m_chunkSizeBytes);

    attempt(m_numRetries, [this]()
    {
        m_indexer->Build();
    });

    // it's still possible that the actual input data does not have sequence id column.
    m_skipSequenceIds = !m_indexer->HasSequenceIds();

    assert(m_indexer != nullptr);

    int64_t position = _ftelli64(m_file);
    if (position == -1L)
    {
        RuntimeError("Error retrieving current position in the input file (%ls).", m_filename.c_str());
    }

    m_fileOffsetStart = position;
    m_fileOffsetEnd = position;
}

template <class ElemType>
ChunkDescriptions TextParser<ElemType>::GetChunkDescriptions()
{
    assert(m_indexer != nullptr);

    const auto& index = m_indexer->GetIndex();

    ChunkDescriptions result;
    result.reserve(index.size());
    for (auto const& chunk : index)
    {
        result.push_back(shared_ptr<ChunkDescription>(
            new ChunkDescription {
                chunk.m_id,
                chunk.m_numberOfSamples,
                chunk.m_numberOfSequences
        }));
    }

    return result;
}

template <class ElemType>
void TextParser<ElemType>::GetSequencesForChunk(size_t chunkId, std::vector<SequenceDescription>& result)
{
    const auto& index = m_indexer->GetIndex();
    const auto& chunk = index[chunkId];
    result.reserve(chunk.m_sequences.size());

    for (auto const& s : chunk.m_sequences)
    {
        result.push_back(
        {
            s.m_id,
            s.m_numberOfSamples,
            s.m_chunkId,
            s.m_isValid,
            s.m_key
        });
    }
}

template <class ElemType>
TextParser<ElemType>::TextDataChunk::TextDataChunk(const ChunkDescriptor& descriptor, TextParser* parser) :
    m_parser(parser)
{
    m_id = descriptor.m_id;
    m_sequenceRequestCount = 0;
}

template <class ElemType>
void TextParser<ElemType>::TextDataChunk::GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result)
{
    auto it = m_sequenceMap.find(sequenceId);
    assert(it != m_sequenceMap.end());
    ++m_sequenceRequestCount;
    result.reserve(m_parser->m_streamInfos.size());
    const auto& sequenceData = it->second;
    for (size_t j = 0; j < m_parser->m_streamInfos.size(); ++j)
    {
        InputStreamBuffer* input = sequenceData[j].get();
        const StreamInfo& stream = m_parser->m_streamInfos[j];
        SequenceDataPtr data;
        if (stream.m_type == StorageType::dense)
        {
            auto denseData = make_shared<DenseSequenceData>();
            denseData->m_sampleLayout = m_parser->m_streams[j]->m_sampleLayout;
            data = denseData;
        }
        else
        {
            auto sparseData = make_shared<SparseSequenceData>();
            SparseInputStreamBuffer* sparseInput = static_cast<SparseInputStreamBuffer*>(input);
            sparseData->m_indices = sparseInput->m_indices.data();
            sparseData->m_nnzCounts.reserve(sparseInput->m_nnzCounts.size());
            copy(sparseInput->m_nnzCounts.begin(), sparseInput->m_nnzCounts.end(),
                back_inserter(sparseData->m_nnzCounts));
            sparseData->m_totalNnzCount = sparseInput->m_totalNnzCount;
            assert(input->m_numberOfSamples == sparseInput->m_nnzCounts.size());
            data = sparseData;
        }

        data->m_data = input->m_buffer.data();
        data->m_numberOfSamples = input->m_numberOfSamples;
        data->m_chunk = shared_from_this();
        data->m_id = sequenceId;
        result.push_back(data);
    }
}

template <class ElemType>
ChunkPtr TextParser<ElemType>::GetChunk(size_t chunkId)
{
    ChunkPtr chunk;
    auto it = m_chunkCache.find(chunkId);
    if (it != m_chunkCache.end())
    {
        chunk = it->second;
    }
    else
    {
        const auto& chunkDescriptor = m_indexer->GetIndex()[chunkId];
        auto textChunk = make_shared<TextDataChunk>(chunkDescriptor, this);

        attempt(m_numRetries, [this, &textChunk, &chunkDescriptor]()
        {
            LoadChunk(textChunk, chunkDescriptor);
        });

        if (m_chunkCacheSize > 0 && m_chunkCache.size() == m_chunkCacheSize)
        {
            size_t candidateId = SIZE_MAX;
            size_t minNumSequencesLeft = SIZE_MAX;
            for (const auto& it : m_chunkCache)
            {
                const auto& chunk = *(it.second.get());
                size_t numSequencesUsed = 0;
                numSequencesUsed += chunk.m_sequenceRequestCount;
                size_t numSequencesLeft = chunk.m_sequenceMap.size() - numSequencesUsed;
                if (numSequencesLeft < minNumSequencesLeft)
                {
                    minNumSequencesLeft = numSequencesLeft;
                    candidateId = it.first;
                }
            }
            assert(candidateId != SIZE_MAX);
            m_chunkCache.erase(candidateId);
        }

        if (m_chunkCacheSize > 0)
        {
            m_chunkCache[chunkId] = textChunk;
        }

        chunk = textChunk;
    }
    return chunk;
}

template <class ElemType>
void TextParser<ElemType>::LoadChunk(TextChunkPtr& chunk, const ChunkDescriptor& descriptor)
{
    for (const auto& sequenceDescriptor : descriptor.m_sequences)
    {
        chunk->m_sequenceMap.insert(make_pair(
            sequenceDescriptor.m_id,
            LoadSequence(!m_skipSequenceIds, sequenceDescriptor)));
    }
}

template <class ElemType>
void TextParser<ElemType>::IncrementNumberOfErrorsOrDie()
{
    if (m_numAllowedErrors == 0)
    {
        PrintWarningNotification();
        RuntimeError("Reached the maximum number of allowed errors"
            " while reading the input file (%ls).",
            m_filename.c_str());
    }
    --m_numAllowedErrors;
}

template <class ElemType>
bool TextParser<ElemType>::TryRefillBuffer()
{
    size_t bytesRead = fread(m_buffer.get(), 1, BUFFER_SIZE, m_file);

    if (bytesRead == (size_t)-1)
    {
        PrintWarningNotification();
        RuntimeError("Could not read from the input file (%ls).", m_filename.c_str());
    }

    if (!bytesRead)
    {
        return false;
    }

    m_fileOffsetStart = m_fileOffsetEnd;
    m_fileOffsetEnd += bytesRead;
    m_bufferStart = m_buffer.get();
    m_pos = m_bufferStart;
    m_bufferEnd = m_bufferStart + bytesRead;
    return true;
}

template <class ElemType>
void TextParser<ElemType>::SetFileOffset(int64_t offset)
{
    int rc = _fseeki64(m_file, offset, SEEK_SET);
    if (rc)
    {
        PrintWarningNotification();
        RuntimeError("Error seeking to position %" PRId64 " in the input file (%ls).",
            offset, m_filename.c_str());
    }

    m_fileOffsetStart = offset;
    m_fileOffsetEnd = offset;

    TryRefillBuffer();
}

template <class ElemType>
typename TextParser<ElemType>::SequenceBuffer TextParser<ElemType>::LoadSequence(bool verifyId, const SequenceDescriptor& sequenceDsc)
{
    auto fileOffset = sequenceDsc.m_fileOffsetBytes;

    if (fileOffset < m_fileOffsetStart || fileOffset > m_fileOffsetEnd)
    {
        SetFileOffset(fileOffset);
    }

    size_t bufferOffset = fileOffset - m_fileOffsetStart;
    m_pos = m_bufferStart + bufferOffset;
    size_t bytesToRead = sequenceDsc.m_byteSize;

    if (verifyId)
    {
        size_t id;
        if (!TryReadUint64(id, bytesToRead) || id != sequenceDsc.m_id)
        {
            PrintWarningNotification();
            RuntimeError("Did not find the expected sequence (id = %" PRIu64 ") %ls.",
                sequenceDsc.m_id, GetFileInfo().c_str());
        }
    }

    SequenceBuffer sequence;

    // TODO: reuse loaded sequences instead of creating new ones!
    for (auto const & stream : m_streamInfos)
    {
        if (stream.m_type == StorageType::dense)
        {
            sequence.push_back(make_unique<DenseInputStreamBuffer>(
                stream.m_sampleDimension * sequenceDsc.m_numberOfSamples));
        }
        else
        {
            sequence.push_back(make_unique<SparseInputStreamBuffer>());
        }
    }

    size_t numRowsRead = 0, expectedRowCount = sequenceDsc.m_numberOfSamples;
    for (size_t i = 0; i < expectedRowCount; i++)
    {
        if ((TryReadRow(sequence, bytesToRead)))
        {
            ++numRowsRead;
        }
        else
        {
            IncrementNumberOfErrorsOrDie();
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Could not read a row (# %" PRIu64 ")"
                    " while loading sequence (id = %" PRIu64 ") %ls.\n",
                    i + 1, sequenceDsc.m_id, GetFileInfo().c_str());
            }
        }

        if (!bytesToRead && numRowsRead < expectedRowCount)
        {
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Exhausted all input"
                    " expected for the current sequence (id = %" PRIu64 ") %ls,"
                    " but only read %" PRId64 " out of %" PRId64 " expected rows.\n",
                    sequenceDsc.m_id, GetFileInfo().c_str(), numRowsRead, expectedRowCount);
            }
            break;
        }
    }

    // Double check if there are empty input streams.
    // TODO this handling needs to be graceful, but currently CNTK complains when we return empty sequences.
    bool hasEmptyInputs = false, hasDuplicateInputs = false;

    for (size_t i = 0; i < sequence.size(); ++i)
    {
        if (sequence[i]->m_numberOfSamples == 0)
        {
            fprintf(stderr,
                "ERROR: Input ('%ls') is empty in sequence (id = %" PRIu64 ") %ls.\n",
                m_streams[i]->m_name.c_str(), sequenceDsc.m_id, GetFileInfo().c_str());
            hasEmptyInputs = true;
        }

        if (sequence[i]->m_numberOfSamples > expectedRowCount)
        {
            hasDuplicateInputs = true;
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Input ('%ls') contains more samples than expected"
                    " (%" PRId64 " vs. %" PRId64 ") for sequence (id = %" PRIu64 ") %ls.\n",
                    m_streams[i]->m_name.c_str(), sequence[i]->m_numberOfSamples, expectedRowCount,
                    sequenceDsc.m_id, GetFileInfo().c_str());
            }
        }
    }

    if (hasEmptyInputs)
    {
        PrintWarningNotification();
        RuntimeError("Malformed input file. Bailing out.");
    }

    if (hasDuplicateInputs)
    {
        IncrementNumberOfErrorsOrDie();
    }

    if (m_traceLevel >= Info)
    {
        fprintf(stderr,
            "INFO: Finished loading sequence (id = %" PRIu64 ") %ls,"
            " successfully read %" PRIu64 " out of expected %" PRIu64 " rows.\n",
            sequenceDsc.m_id, GetFileInfo().c_str(), numRowsRead, expectedRowCount);
    }

    return sequence;
}

template <class ElemType>
bool TextParser<ElemType>::TryReadRow(SequenceBuffer& sequence, size_t& bytesToRead)
{
    while (bytesToRead && CanRead() && isdigit(*m_pos))
    {
        // skip sequence ids
        ++m_pos;
        --bytesToRead;
    }

    size_t numSampleRead = 0;
    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        if (c == COLUMN_DELIMITER || c == VALUE_DELIMITER || c == CARRIAGE_RETURN)
        {
            // skip column and value separators, as well as carriage returns.
            ++m_pos;
            --bytesToRead;
            continue;
        }

        if (c == ROW_DELIMITER)
        {
            // found the end of row, skip the delimiter, return.
            ++m_pos;
            --bytesToRead;

            if (numSampleRead == 0 && ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Empty input row %ls.\n", GetFileInfo().c_str());
            }
            else if (numSampleRead > m_streams.size() && ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Input row %ls contains more"
                    " samples than expected (%" PRId64 " vs. %" PRId64 ").\n",
                    GetFileInfo().c_str(), numSampleRead, m_streams.size());
            }

            return numSampleRead > 0;
        }

        if (TryReadSample(sequence, bytesToRead))
        {
            numSampleRead++;
        }
        else
        {
            // skip over until the next sample/end of row
            SkipToNextInput(bytesToRead);
        }
    }

    if (ShouldWarn())
    {
        fprintf(stderr,
            "WARNING: Exhausted all input expected for the current sequence"
            " while reading an input row %ls."
            " Possibly, a trailing newline is missing.\n", GetFileInfo().c_str());
    }
    return false;
}

// Reads one sample (an pipe-prefixed input identifier followed by a list of values)
template <class ElemType>
bool TextParser<ElemType>::TryReadSample(SequenceBuffer& sequence, size_t& bytesToRead)
{
    assert(m_pos < m_bufferEnd);

    // prefix check.
    if (*m_pos != NAME_PREFIX)
    {
        if (ShouldWarn())
        {
            fprintf(stderr,
                "WARNING: Unexpected character('%c') in place of a name prefix ('%c')"
                " in an input name %ls.\n",
                *m_pos, NAME_PREFIX, GetFileInfo().c_str());
        }
        IncrementNumberOfErrorsOrDie();
        return false;
    }

    // skip name prefix
    ++m_pos;
    --bytesToRead;

    if (bytesToRead && CanRead() && *m_pos == ESCAPE_SYMBOL)
    {
        // A vertical bar followed by the number sign (|#) is treated as an escape sequence, 
        // everything that follows is ignored until the next vertical bar or the end of 
        // row, whichever comes first.
        ++m_pos;
        --bytesToRead;
        return false;
    }

    size_t id;
    if (!TryGetInputId(id, bytesToRead))
    {
        IncrementNumberOfErrorsOrDie();
        return false;
    }

    const StreamInfo& stream = m_streamInfos[id];

    if (stream.m_type == StorageType::dense)
    {
        DenseInputStreamBuffer* data = reinterpret_cast<DenseInputStreamBuffer*>(sequence[id].get());
        vector<ElemType>& values = data->m_buffer;
        size_t size = values.size();
        assert(size % stream.m_sampleDimension == 0);
        if (!TryReadDenseSample(values, stream.m_sampleDimension, bytesToRead))
        {
            // expected a dense sample, but was not able to fully read it, ignore it.
            if (values.size() != size)
            {
                //clean up the buffer
                values.resize(size);
            }
            IncrementNumberOfErrorsOrDie();
            return false;
        }
        // everything went well, increment the number of samples.
        ++data->m_numberOfSamples;
    }
    else
    {
        SparseInputStreamBuffer* data = reinterpret_cast<SparseInputStreamBuffer*>(sequence[id].get());
        vector<ElemType>& values = data->m_buffer;
        vector<IndexType>& indices = data->m_indices;
        assert(values.size() == indices.size());
        size_t size = values.size();
        if (!TryReadSparseSample(values, indices, stream.m_sampleDimension, bytesToRead))
        {
            // expected a sparse sample, but something went south, ignore it.
            if (values.size() != size)
            {
                //clean up the buffer
                values.resize(size);
            }
            if (indices.size() != size)
            {
                //clean up the buffer
                indices.resize(size);
            }

            IncrementNumberOfErrorsOrDie();
            return false;
        }
        assert(values.size() == indices.size());
        ++data->m_numberOfSamples;
        IndexType count = static_cast<IndexType>(values.size() - size);
        data->m_nnzCounts.push_back(count);
        data->m_totalNnzCount += count;
    }

    return true;
}

template <class ElemType>
bool TextParser<ElemType>::TryGetInputId(size_t& id, size_t& bytesToRead)
{
    char* scratchIndex = m_scratch.get();

    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        // an input id can be followed by a value marker, end of line (also, carriage return),
        // column separator or the name prefix of the following input.
        if (c <= VALUE_DELIMITER || c == NAME_PREFIX)
        {
            size_t size = scratchIndex - m_scratch.get();
            if (size)
            {
                string name(m_scratch.get(), size);
                auto it = m_aliasToIdMap.find(name);
                if (it != m_aliasToIdMap.end())
                {
                    id = it->second;
                    return true;
                }

                if (ShouldWarn())
                {
                    fprintf(stderr,
                        "WARNING: Invalid input name ('%s') %ls.\n",
                        name.c_str(), GetFileInfo().c_str());
                }
            }
            else if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Input name prefix ('%c') is followed by"
                    " an invalid character ('%c') %ls.\n",
                    NAME_PREFIX, c, GetFileInfo().c_str());
            }

            return false;
        }
        else if (scratchIndex < (m_scratch.get() + m_maxAliasLength))
        {
            *scratchIndex = c;
            ++scratchIndex;
        }
        else
        {
            // the current string length is already equal to the maximum expected length,
            // yet it's not followed by a delimiter.
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Did not find a valid input name %ls.\n",
                    GetFileInfo().c_str());
            }
            return false;
        }

        ++m_pos;
        --bytesToRead;
    }

    if (ShouldWarn())
    {
        fprintf(stderr,
            "WARNING: Exhausted all input expected for the current sequence"
            " while reading an input name %ls.\n", GetFileInfo().c_str());
    }
    return false;
}

template <class ElemType>
bool TextParser<ElemType>::TryReadDenseSample(vector<ElemType>& values, size_t sampleSize, size_t& bytesToRead)
{
    size_t counter = 0;
    ElemType value;

    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        // return as soon as we hit a non-printable or a name prefix
        if (c < VALUE_DELIMITER || c == NAME_PREFIX)
        {
            if (counter > sampleSize)
            {
                if (ShouldWarn())
                {
                    fprintf(stderr,
                        "WARNING: Dense sample (size = %" PRId64 ") %ls"
                        " exceeds the expected size (%" PRId64 ").\n",
                        counter, GetFileInfo().c_str(), sampleSize);
                }
                return false;
            }

            // For dense matrices, it should be possible to input only the left part
            // if the suffix is sparse. Fill up the rest with zeros.
            if (counter < sampleSize)
            {
                if (ShouldWarn())
                {
                    fprintf(stderr,
                        "WARNING: A dense sample %ls has a sparse suffix "
                        "(expected size = %" PRId64 ", actual size = %" PRId64 ").\n",
                        GetFileInfo().c_str(), sampleSize, counter);
                }
                for (; counter < sampleSize; ++counter)
                {
                    values.push_back(0.0f);
                }
            }

            return true;
        }

        if (c == VALUE_DELIMITER)
        {
            // skip value delimiters
            ++m_pos;
            --bytesToRead;
            continue;
        }

        if (!TryReadRealNumber(value, bytesToRead))
        {
            // bail out.
            return false;
        }

        values.push_back(value);
        ++counter;
    }

    IncrementNumberOfErrorsOrDie();
    if (ShouldWarn())
    {
        fprintf(stderr,
            "WARNING: Exhausted all input expected for the current sequence"
            " while reading a dense sample %ls.\n", GetFileInfo().c_str());
    }
    return false;
}

template <class ElemType>
bool TextParser<ElemType>::TryReadSparseSample(std::vector<ElemType>& values, std::vector<IndexType>& indices,
    size_t sampleSize, size_t& bytesToRead)
{
    size_t index = 0;
    ElemType value;

    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        // return as soon as we hit a non-printable or a name prefix
        if (c < VALUE_DELIMITER || c == NAME_PREFIX)
        {
            // empty sparse samples are allowed ("|InputeName_1|InputName2...")
            return true;
        }

        if (c == VALUE_DELIMITER)
        {
            // skip value delimiters
            ++m_pos;
            --bytesToRead;
            continue;
        }

        // read next sparse index
        if (!TryReadUint64(index, bytesToRead))
        {
            // bail out.
            return false;
        }

        if (index > sampleSize)
        {
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Sparse index value (%" PRIu64 ") %ls"
                    " exceeds the expected sample size (%" PRIu64 ").\n",
                    index, GetFileInfo().c_str(), sampleSize);
            }
            // bail out.
            return false;
        }

        // an index must be followed by a delimiter
        c = *m_pos;
        if (c != INDEX_DELIMITER)
        {
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Unexpected character('%c')"
                    " in place of the index delimiter ('%c')"
                    " after a sparse value index (%" PRId64 ") %ls.\n",
                    c, INDEX_DELIMITER, index, GetFileInfo().c_str());
            }
            return false;
        }

        // skip index delimiter
        ++m_pos;
        --bytesToRead;

        // read the corresponding value
        if (!TryReadRealNumber(value, bytesToRead))
        {
            // bail out.
            return false;
        }

        values.push_back(value);
        indices.push_back(static_cast<IndexType>(index));
    }

    if (ShouldWarn())
    {
        fprintf(stderr,
            "WARNING: Exhausted all input expected for the current sequence"
            " while reading a sparse sample %ls.\n", GetFileInfo().c_str());
    }

    return false;
}

template <class ElemType>
void TextParser<ElemType>::SkipToNextValue(size_t& bytesToRead)
{
    while (bytesToRead && CanRead())
    {
        char c = *m_pos;
        // skip everything until we hit either a value marker, an input marker or the end of row.
        if (c == VALUE_DELIMITER || c == ROW_DELIMITER || c == NAME_PREFIX)
        {
            return;
        }
        ++m_pos;
        --bytesToRead;
    }
}

template <class ElemType>
void TextParser<ElemType>::SkipToNextInput(size_t& bytesToRead)
{
    while (bytesToRead && CanRead())
    {
        char c = *m_pos;
        // skip everything until we hit either an input marker or the end of row.
        if (c == NAME_PREFIX || c == ROW_DELIMITER)
        {
            return;
        }
        ++m_pos;
        --bytesToRead;
    }
}

template <class ElemType>
bool TextParser<ElemType>::TryReadUint64(size_t& value, size_t& bytesToRead)
{
    value = 0;
    bool found = false;
    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        if (!isdigit(c))
        {
            if (isDelimiter(c))
            {
                return found;
            }
            
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Unexpected character('%c') in a uint64 value %ls.\n",
                    c, GetFileInfo().c_str());
            }

            return false;
        }

        found |= true;

        size_t temp = value;
        value = value * 10 + (c - '0');
        if (temp > value)
        {
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Overflow while reading a uint64 value %ls.\n",
                    GetFileInfo().c_str());
            }

            return false;
        }

        ++m_pos;
        --bytesToRead;
    }

    if (ShouldWarn())
    {
        fprintf(stderr,
            "WARNING: Exhausted all input expected for the current sequence"
            " while reading a uint64 value %ls.\n", GetFileInfo().c_str());
    }
    return false;
}



// TODO: better precision (at the moment we're at parity with UCIFast)?
// Assumes that bytesToRead is greater than the number of characters 
// in the string representation of the floating point number
// (i.e., the string is followed by one of the delimiters)
// Post condition: m_pos points to the first character that 
// cannot be parsed as part of a floating point number.
// Returns true if parsing was successful.
template <class ElemType>
bool TextParser<ElemType>::TryReadRealNumber(ElemType& value, size_t& bytesToRead)
{
    State state = State::Init;
    double coefficient = .0, number = .0, divider = .0;
    bool negative = false;

    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        switch (state)
        {
        case State::Init:
            // the number must either start with a number or a sign
            if (isdigit(c))
            {
                state = IntegralPart;
                number = (c - '0');
            }
            else if (isSign(c))
            {
                state = Sign;
                negative = (c == '-');
            }
            else
            {
                if (ShouldWarn())
                {
                    fprintf(stderr,
                        "WARNING: Unexpected character ('%c')"
                        " in a floating point value %ls.\n",
                        c, GetFileInfo().c_str());
                }
                return false;
            }
            break;
        case Sign:
            // the sign must be followed by a number
            if (isdigit(c))
            {
                state = IntegralPart;
                number = (c - '0');
            }
            else
            {
                if (ShouldWarn())
                {
                    fprintf(stderr,
                        "WARNING: A sign symbol is followed by an invalid character('%c')"
                        " in a floating point value %ls.\n",
                        c, GetFileInfo().c_str());
                }
                return false;
            }
            break;
        case IntegralPart:
            if (isdigit(c))
            {
                number = number * 10 + (c - '0');
            }
            else if (c == '.')
            {
                state = Period;
            }
            else if (isE(c))
            {
                state = TheLetterE;
                coefficient = (negative) ? -number : number;
                number = 0;
            }
            else
            {
                value = static_cast<ElemType>((negative) ? -number : number);
                return true;
            }
            break;
        case Period:
            if (isdigit(c))
            {
                state = FractionalPart;
                coefficient = number;
                number = (c - '0');
                divider = 10;
            }
            else
            {
                value = static_cast<ElemType>((negative) ? -number : number);
                return true;
            }
            break;
        case FractionalPart:
            if (isdigit(c))
            {
                // TODO: ignore if number of precision digits > FLT_[MANT_]DIG/DBL_[MANT_]DIG
                // no state change
                number = number * 10 + (c - '0');
                divider *= 10;
            }
            else if (isE(c))
            {
                state = TheLetterE;
                coefficient += (number / divider);
                if (negative)
                {
                    coefficient = -coefficient;
                }
            }
            else
            {
                coefficient += (number / divider);
                value = static_cast<ElemType>((negative) ? -coefficient : coefficient);
                return true;
            }
            break;
        case TheLetterE:
            // followed with optional minus or plus sign and nonempty sequence of decimal digits
            if (isdigit(c))
            {
                state = Exponent;
                negative = false;
                number = (c - '0');
            }
            else if (isSign(c))
            {
                state = ExponentSign;
                negative = (c == '-');
            }
            else
            {
                if (ShouldWarn())
                {
                    fprintf(stderr,
                        "WARNING: An exponent symbol is followed by"
                        " an invalid character('%c')"
                        " in a floating point value %ls.\n", c, GetFileInfo().c_str());
                }
                return false;
            }
            break;
        case ExponentSign:
            // exponent sign must be followed by a number
            if (isdigit(c))
            {
                state = Exponent;
                number = (c - '0');
            }
            else
            {
                if (ShouldWarn())
                {
                    fprintf(stderr,
                        "WARNING: An exponent sign symbol followed by"
                        " an unexpected character('%c')"
                        " in a floating point value %ls.\n", c, GetFileInfo().c_str());
                }
                return false;
            }
            break;
        case Exponent:
            if (isdigit(c))
            {
                // no state change
                number = number * 10 + (c - '0');
            }
            else
            {
                // TODO: check the exponent value (see FLT_[MAX/MIN]_10_EXP).
                double exponent = (negative) ? -number : number;
                value = static_cast<ElemType>(coefficient * pow(10.0, exponent));
                return true;
            }
            break;
        default:
            LogicError("Reached an invalid state while reading a floating point value %ls.\n",
                GetFileInfo().c_str());
        }

        ++m_pos;
        --bytesToRead;
    }

    if (ShouldWarn())
    {
        fprintf(stderr,
            "WARNING: Exhausted all input expected for the current sequence"
            " while reading a floating point value %ls.\n", GetFileInfo().c_str());
    }

    return false;
}

template <class ElemType>
void TextParser<ElemType>::SetTraceLevel(unsigned int traceLevel)
{
    m_traceLevel = traceLevel;
}

template <class ElemType>
void TextParser<ElemType>::SetMaxAllowedErrors(unsigned int maxErrors)
{
    m_numAllowedErrors = maxErrors;
}

template <class ElemType>
void TextParser<ElemType>::SetSkipSequenceIds(bool skip)
{
    m_skipSequenceIds = skip;
}

template <class ElemType>
void TextParser<ElemType>::SetChunkCacheSize(unsigned int size)
{
    m_chunkCacheSize = size;
}

template <class ElemType>
void TextParser<ElemType>::SetChunkSize(size_t size)
{
    m_chunkSizeBytes = size;
}

template <class ElemType>
void TextParser<ElemType>::SetNumRetries(unsigned int numRetries)
{
    m_numRetries = numRetries;
}

template <class ElemType>
std::wstring TextParser<ElemType>::GetFileInfo()
{
    std::wstringstream info;
    info << L"at offset " << GetFileOffset() << L" in the input file (" << m_filename << L")";
    return info.str();
}

template class TextParser<float>;
template class TextParser<double>;
}}}
