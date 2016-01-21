//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>

#include "Transformer.h"
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class NoRandomizer : public Transformer
{
public:
    NoRandomizer(DataDeserializerPtr deserializer);
    virtual ~NoRandomizer()
    {
    }

    virtual void Initialize(TransformerPtr next, const ConfigParameters& readerConfig) override;
    virtual void StartEpoch(const EpochConfiguration& config) override;
    virtual Sequences GetNextSequences(size_t count) override;
    virtual std::vector<StreamDescriptionPtr> GetStreams() const override
    {
        return m_deserializer->GetStreams();
    }

private:
    // Deserializer and information on the original timeline
    DataDeserializerPtr m_deserializer;
    size_t m_numSequences;
    size_t m_numChunks;
    size_t m_numSamples;
    bool m_frameMode;                                 // true iff only single-sample sequences

    // Per-epoch configuration
    size_t m_workerRank;
    size_t m_numberOfWorkers;
    size_t m_epochSize;
    size_t m_samplePositionInEpoch;

    // Sweep information
    size_t m_sweep;
    size_t m_sweepStartInSamples; // TODO do we need it?
    size_t m_sequencePositionInSweep;

    // Check that timeline has only valid sequences of non-zero length
    // with incrementing IDs and non-decreasing chunk identifiers.
    bool TimelineIsValidForRandomization(const SequenceDescriptions& timeline) const;

    bool AdvanceToNextPositionForThisWorker();
};
} } }
