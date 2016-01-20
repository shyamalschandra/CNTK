//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include <memory>
#include "Sequences.h"
#include "TensorShape.h"

namespace Microsoft { namespace MSR { namespace CNTK {

typedef std::shared_ptr<TensorShape> TensorShapePtr;

struct MBLayout;
typedef std::shared_ptr<MBLayout> MBLayoutPtr;

// Configuration for the current epoch.
// Each time the epoch is started CNTK should communicate the configuration to the reader.
struct EpochConfiguration
{
    size_t m_numberOfWorkers;          // Number of the Open MPI workers for the current epoch
    size_t m_workerRank;               // Rank of the Open MPI worker, worker rank has to be less the the number of workers
    size_t m_minibatchSizeInSamples;   // Maximum minibatch size for the epoch in samples
    size_t m_totalEpochSizeInSamples;  // Total size of the epoch in samples
    size_t m_epochIndex;                    // Current epoch index [0 .. max number of epochs)
};

// Supported primitive element types, will be extended in the future
enum class ElementType
{
    tfloat,  // single precision
    tdouble, // double precision
    tatom    // sizeof(atom) == 1 constitute of blobs -> sequences of atoms (i.e. used for lattices, hmmm, etc.)
};

// Supported storage types.
enum class StorageType
{
    dense,
    sparse_csc,
};

typedef size_t StreamId;

// This class describes a particular stream: its name, element type, storage, etc.
struct StreamDescription
{
    std::wstring m_name;           // Unique name of the stream
    StreamId m_id;                 // Unique identifier of the stream
    StorageType m_storageType;     // Storage type of the stream
    ElementType m_elementType;     // Element type of the stream
    TensorShapePtr m_sampleLayout; // Layout of the sample for the stream
                                 // If not specified - can be specified per sequence
};
typedef std::shared_ptr<StreamDescription> StreamDescriptionPtr;

// Input data.
struct Stream
{
    void* m_data;         // Continues array of data. Can be encoded in dense or sparse format
    size_t m_dataSize;    // Dat size
    MBLayoutPtr m_layout; // Layout out of the data
};
typedef std::shared_ptr<Stream> StreamPtr;

// Represents a single minibatch, that contains information about all streams.
struct Minibatch
{
    bool m_endOfEpoch;                // Signifies that the end of epoch has been reached.
    std::vector<StreamPtr> m_data;    // Minibatch data

    Minibatch() : m_endOfEpoch(false)
    {
    }
};

// Main Reader interface. The border interface between the CNTK and Reader.
// TODO: possibly except the matrices in the ReadMinibatch.
class Reader
{
public:
    // Describes the streams this reader produces.
    virtual std::vector<StreamDescriptionPtr> GetStreams() = 0;

    // Starts a new epoch.
    virtual void StartEpoch(const EpochConfiguration& config) = 0;

    // Reads a minibatch that contains data across all streams.
    virtual Minibatch ReadMinibatch() = 0;

    virtual ~Reader() = 0 {};
};

typedef std::shared_ptr<Reader> ReaderPtr;
}}}
