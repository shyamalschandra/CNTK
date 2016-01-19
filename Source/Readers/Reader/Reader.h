//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include <memory>
#include "Sequences.h"
#include "DataTensor.h"

namespace Microsoft { namespace MSR { namespace CNTK {

struct ImageLayout;
typedef std::shared_ptr<ImageLayout> TensorShapePtr;

struct MBLayout;
typedef std::shared_ptr<MBLayout> MBLayoutPtr;

// Configuration for the current epoch.
// Each time the epoch is started CNTK should communicate the configuration to the reader.
struct EpochConfiguration
{
    size_t numberOfWorkers; // Number of the Open MPI workers for the current epoch
    size_t workerRank;      // Rank of the Open MPI worker, worker rank has to be less the the number of workers
    size_t minibatchSize;   // Minibatch size for the epoch
    size_t totalSize;       // Total size of the epoch in samples
    size_t index;           // Current epoch index [0 .. max number of epochs)
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
    std::wstring name;           // Uniquer name of the stream
    StreamId id;                 // Unique identifier of the stream
    StorageType storageType;     // Storage type of the stream
    ElementType elementType;     // Element type of the stream
    TensorShapePtr sampleLayout; // Layout of the sample for the stream
                                 // If not specified - can be specified per sequence
};
typedef std::shared_ptr<StreamDescription> StreamDescriptionPtr;

// Input data.
struct Stream
{
    void* data;         // Continues array of data. Can be encoded in dense or sparse format
    size_t dataSize;    // Dat size
    MBLayoutPtr layout; // Layout out of the data
};
typedef std::shared_ptr<Stream> StreamPtr;

// Represents a single minibatch, that contains information about all streams.
struct Minibatch
{
    bool atEndOfEpoch;                // Signifies the end of the epoch
    std::vector<StreamPtr> minibatch; // Minibatch data

    Minibatch()
        : atEndOfEpoch(false)
    {
    }

    Minibatch(Minibatch&& other)
        : atEndOfEpoch(std::move(other.atEndOfEpoch)), minibatch(std::move(other.minibatch))
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
} } }
