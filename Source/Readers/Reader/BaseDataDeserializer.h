//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Base class for data deserializers.
// Has default implementation for a subset of methods.
class BaseDataDeserializer : public DataDeserializer
{
public:
    BaseDataDeserializer()
    {
    }
    virtual ~BaseDataDeserializer()
    {
    }

    // Sets configuration for the current epoch.
    void StartEpoch(const EpochConfiguration& /*config*/) override{};

    // Gets descriptions of all sequences the deserializer can produce.
    const Timeline& GetSequenceDescriptions() const override
    {
        if (m_sequences.empty())
        {
            FillSequenceDescriptions(m_sequences);
        }
        return m_sequences;
    }

    // Is be called by the randomizer for prefetching the next chunk.
    // By default IO read-ahead is not implemented.
    void RequireChunk(size_t /*chunkIndex*/) override{};

    // Is be called by the randomizer for releasing a prefetched chunk.
    // By default IO read-ahead is not implemented.
    void ReleaseChunk(size_t /*chunkIndex*/) override{};

protected:
    // Fills the timeline with sequence descriptions.
    // Inherited classes should provide the complete Sequence descriptions for all input data.
    virtual void FillSequenceDescriptions(Timeline& timeline) const = 0;
    std::vector<StreamDescriptionPtr> m_streams;

private:
    BaseDataDeserializer(const BaseDataDeserializer&) = delete;
    BaseDataDeserializer& operator=(const BaseDataDeserializer&) = delete;

    mutable Timeline m_sequences;
};
} } }
