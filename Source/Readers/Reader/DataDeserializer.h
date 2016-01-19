//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "Reader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Defines main properties of a sequence. This data structure is used
// to define the global timeline of all input data.
struct SequenceDescription
{
    size_t id;              // Sequence id, uniquely identifies the sequence.
    size_t numberOfSamples; // Number of samples in a sequence.
    size_t chunkId;         // Each sequence belongs to an I/O chunk, how chunk is defined is specific to a particular data deserializer.
    bool isValid;
};
typedef std::vector<const SequenceDescription*> Timeline;

// Defines sequence data and its layout.
// We support dense and sparse sequences.
// The storageType in the corresponding stream description defines what type of SequenceData
// data deserializer or transformer provides.
struct SequenceData
{
    SequenceData()
        : data(nullptr)
    {
    }

    void* data;
};
typedef std::shared_ptr<SequenceData> SequenceDataPtr;

// Dense sequence. Corresponds to the StorageType::dense of the stream.
// All samples are store in the 'data' member as a continuous array.
// The layout of samples are described in the sampleLayout.
// All samples in the sequence should have the same layout.
struct DenseSequenceData : SequenceData
{
    DenseSequenceData()
        : numberOfSamples(0)
    {
    }

    TensorShapePtr sampleLayout; // Sample layout
    size_t numberOfSamples;      // Number of samples in the sequence
};
typedef std::shared_ptr<DenseSequenceData> DenseSequenceDataPtr;

// Dense sequence. Corresponds to the StorageType::css_sparse of the stream.
// All non zero values are store in the 'data' member as a continuous array.
// The corresponding row indices are stored in 'indices'.
// All samples in the sequence should have the same layout.
struct SparseSequenceData : SequenceData
{
    std::vector<std::vector<size_t>> indices;
};
typedef std::shared_ptr<SparseSequenceData> SparseSequenceDataPtr;

// Data deserializers are intimately familiar with a particular input formats and responsbille for reading the serialized data
// into sequences in memory. Very often data for different streams (i.e. features/lattices) reside in the same physical storage (file),
// so the data deserializer can expose not a single but several streams. Examples of data include image data deserializer
// or htkmlf data deserializer .
class DataDeserializer
{
public:
    // Describes streams the data deserializer produces.
    virtual std::vector<StreamDescriptionPtr> GetStreams() const = 0;

    // Sets epoch configuration.
    virtual void StartEpoch(const EpochConfiguration& config) = 0;

    // Retrieve global timeline the data deserializer can produce.
    virtual const Timeline& GetSequenceDescriptions() const = 0;

    // Gets sequences by id.
    // The return value can be used until the next call to GetSequencesById.
    virtual std::vector<std::vector<SequenceDataPtr>> GetSequencesById(const std::vector<size_t>& ids) = 0;

    // Require chunk.
    virtual void RequireChunk(size_t chunkIndex) = 0;

    // Release chunk.
    virtual void ReleaseChunk(size_t chunkIndex) = 0;

    virtual ~DataDeserializer() = 0 {};
};

typedef std::shared_ptr<DataDeserializer> DataDeserializerPtr;
} } }
