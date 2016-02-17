//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"
#include "HTKDataDeserializer.h"
#include "biggrowablevectors.h"
#include "CorpusDescriptor.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class MLFDataDeserializer : public IDataDeserializer
{
public:
    MLFDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, const std::wstring& streamName);

    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override;
    virtual const SequenceDescriptions& GetSequenceDescriptions() const override;
    const SequenceDescription* GetSequenceDescriptionByKey(const KeyType& key) override;

    virtual ChunkPtr GetChunk(size_t) override;

private:
    // Inner classes for frames, utterances and chunks.
    struct MLFUtterance;
    struct MLFFrame;
    class MLFChunk;

    std::vector<SequenceDataPtr> GetSequenceById(size_t sequenceId);

    std::map<wstring, size_t> m_keyToSequence;
    TensorShapePtr m_layout;

    msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE> m_classIds;
    msra::dbn::biggrowablevector<size_t> m_utteranceIndex;

    // All sequences = frames, this deserializer provides.
    // This interface will be changed when the randomizer asks
    // timeline in sequences.
    msra::dbn::biggrowablevector<MLFFrame> m_frames;
    SequenceDescriptions m_sequences;

    // Type of the data this serializer provdes.
    ElementType m_elementType;

    // Streams, this deserializer provides. A single mlf stream.
    std::vector<StreamDescriptionPtr> m_streams;
};

}}}
