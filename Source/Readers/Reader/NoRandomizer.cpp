//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS

#include "NoRandomizer.h"
#include <algorithm>
#include <utility>
#include <iostream>

#include "DataReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

bool NoRandomizer::TimelineIsValidForRandomization(const SequenceDescriptions& timeline) const
{
    SequenceDescription previous = { SIZE_MAX, 0, 0, true };

    auto it = std::find_if_not(timeline.begin(), timeline.end(),
                               [&](const SequenceDescription* current)
                               {
                                   bool result = current->m_isValid
                                       && previous.m_id + 1 == current->m_id
                                       && previous.m_chunkId <= current->m_chunkId
                                       && current->m_chunkId <= previous.m_chunkId + 1
                                       && 0 < current->m_numberOfSamples;
                                   previous = *current;
                                   return result;
                               });
    return it == timeline.end();
}

//
// Public methods
//

NoRandomizer::NoRandomizer(DataDeserializerPtr deserializer)
    : m_deserializer(deserializer), m_sweep(SIZE_MAX), m_sequencePositionInSweep(SIZE_MAX), m_samplePositionInEpoch(SIZE_MAX), m_epochSize(SIZE_MAX)
{
    assert(deserializer != nullptr);
    const SequenceDescriptions& timeline = m_deserializer->GetSequenceDescriptions();
    assert(TimelineIsValidForRandomization(timeline));

    m_numSequences = timeline.back()->m_id + 1;
    m_numChunks = timeline.back()->m_chunkId + 1;

    // Determine total and maximum number of samples
    size_t maxNumberOfSamples = 0;
    m_numSamples = 0;
    for (const auto& seqDesc : timeline)
    {
        maxNumberOfSamples = max(maxNumberOfSamples, seqDesc->m_numberOfSamples);
        m_numSamples += seqDesc->m_numberOfSamples;
    }

    // Frame mode to the randomizer just means there are only single-sample sequences
    m_frameMode = (maxNumberOfSamples == 1);
}

void NoRandomizer::Initialize(TransformerPtr next, const ConfigParameters& readerConfig)
{
    // Not used
    UNREFERENCED_PARAMETER(next);
    UNREFERENCED_PARAMETER(readerConfig);
}

void NoRandomizer::StartEpoch(const EpochConfiguration& config)
{
    m_deserializer->StartEpoch(config);

    m_workerRank = config.m_workerRank;
    m_numberOfWorkers = config.m_numberOfWorkers;

    // eldak: check partial minibatches.
    if (config.m_totalEpochSizeInSamples == requestDataSize)
    {
        m_epochSize = m_numSamples;
    }
    else
    {
        m_epochSize = config.m_totalEpochSizeInSamples;
    }

    // TODO add some asserts on EpochConfiguration
    m_samplePositionInEpoch = 0;
    size_t timeframe = m_epochSize * config.m_epochIndex;
    assert(m_frameMode);           // TODO not (tested) yet
    assert(timeframe != SIZE_MAX); // used as special value for init
};

bool NoRandomizer::AdvanceToNextPositionForThisWorker()
{
    const SequenceDescriptions& timeline = m_deserializer->GetSequenceDescriptions();

    while (m_samplePositionInEpoch < m_epochSize)
    {
        const auto& seqDesc = timeline[m_sequencePositionInSweep];

        if ((seqDesc->m_chunkId % m_numberOfWorkers) == m_workerRank)
        {
            // Got one
            break;
        }

        m_samplePositionInEpoch += seqDesc->m_numberOfSamples;
        m_sequencePositionInSweep++;
    }

    return m_epochSize <= m_samplePositionInEpoch;
}

Sequences NoRandomizer::GetNextSequences(size_t count)
{
    assert(m_samplePositionInEpoch != SIZE_MAX); // SetEpochConfiguration() must be called first

    const SequenceDescriptions& timeline = m_deserializer->GetSequenceDescriptions();

    std::vector<size_t> ids;
    bool endOfEpoch = false;
    Sequences result;

    while (ids.size() < count)
    {
        endOfEpoch = AdvanceToNextPositionForThisWorker();
        if (endOfEpoch)
        {
            break;
        }
        else
        {
            assert(m_sequencePositionInSweep < m_numSequences);
            ids.push_back(m_sequencePositionInSweep);
            const auto& seqDesc = timeline[m_sequencePositionInSweep];
            m_samplePositionInEpoch += seqDesc->m_numberOfSamples;
            m_sequencePositionInSweep++;
        }
    };

    result.m_endOfEpoch = endOfEpoch;

    if (ids.size() == 0)
    {
        return result;
    }

    // TODO chunking?

    // Get data
    result.m_data = m_deserializer->GetSequencesById(ids);
    return result;
};
} } }
