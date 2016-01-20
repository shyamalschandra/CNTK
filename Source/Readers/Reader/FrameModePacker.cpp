//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include "FrameModePacker.h"
#include "ElementTypeUtils.h"

namespace Microsoft { namespace MSR { namespace CNTK {

FrameModePacker::FrameModePacker(
    MemoryProviderPtr memoryProvider,
    TransformerPtr transformer,
    size_t minibatchSize,
    const std::vector<StreamDescriptionPtr>& streams)
    : m_transformer(transformer), m_mbSize(minibatchSize), m_outputStreams(streams), m_minibatchLayout(std::make_shared<MBLayout>()), m_memoryProvider(memoryProvider)
{
    m_inputStreams = m_transformer->GetStreams();
    assert(m_inputStreams.size() == m_outputStreams.size());
    assert(
        std::find_if(
            m_outputStreams.begin(),
            m_outputStreams.end(),
            [](const StreamDescriptionPtr& s)
            {
                return s->m_storageType == StorageType::sparse_csc;
            }) == m_outputStreams.end());

    for (const auto& stream : streams)
    {
        assert(stream->m_elementType == ElementType::tfloat || stream->m_elementType == ElementType::tdouble);
        m_streamBuffers.push_back(
            AllocateBuffer(m_mbSize * stream->m_sampleLayout->GetNumElements(), GetSizeByType(stream->m_elementType)));
    }
}

Minibatch FrameModePacker::ReadMinibatch()
{
    assert(m_mbSize > 0);

    Minibatch m;
    m.m_atEndOfEpoch = false;

    auto sequences = m_transformer->GetNextSequences(m_mbSize);

    if (sequences.m_endOfEpoch)
    {
        m.m_atEndOfEpoch = true;
    }

    for (size_t i = 0; i < sequences.m_data.size(); i++)
    {
        assert(m_streamBuffers.size() == sequences.m_data[i].size());
        for (int j = 0; j < sequences.m_data[i].size(); ++j)
        {
            size_t elementSize = GetSizeByType(m_inputStreams[j]->m_elementType);
            size_t dimensions = m_inputStreams[j]->m_sampleLayout->GetNumElements() * elementSize;
            auto source = reinterpret_cast<const char*>(sequences.m_data[i][j]->m_data);
            if (m_inputStreams[j]->m_storageType == StorageType::dense)
            {
                auto data = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[i][j]);
                assert(data.m_numberOfSamples == 1);

                std::copy(
                    source,
                    source + dimensions,
                    m_streamBuffers[j].get() + dimensions * i);
            }
            else if (m_inputStreams[j]->m_storageType == StorageType::sparse_csc)
            {
                auto data = reinterpret_cast<SparseSequenceData&>(*sequences.m_data[i][j]);
                assert(data.m_indices.size() == 1);

                std::fill(m_streamBuffers[j].get() + i * dimensions, m_streamBuffers[j].get() + (i + 1) * dimensions, 0);
                size_t nonZeroCount = data.m_indices[0].size();
                for (size_t nonZeroIndex = 0; nonZeroIndex < nonZeroCount; ++nonZeroIndex)
                {
                    size_t rowIndex = data.m_indices[0][nonZeroIndex];
                    char* destination = m_streamBuffers[j].get() + dimensions * i + rowIndex * elementSize;
                    std::copy(source + nonZeroIndex * elementSize, source + (nonZeroIndex + 1) * elementSize, destination);
                }
            }
            else
            {
                RuntimeError("Storage type %d is not supported.", m_inputStreams[j]->m_storageType);
            }
        }
    }

    if (sequences.m_data.size() == 0)
    {
        return m;
    }

    m_minibatchLayout->InitAsFrameMode(sequences.m_data.size());
    for (int i = 0; i < m_outputStreams.size(); ++i)
    {
        size_t dimensions = m_outputStreams[i]->m_sampleLayout->GetNumElements() * GetSizeByType(m_outputStreams[i]->m_elementType);
        auto stream = std::make_shared<Stream>();
        stream->m_data = m_streamBuffers[i].get();
        stream->m_dataSize = sequences.m_data.size() * dimensions;
        stream->m_layout = m_minibatchLayout;

        m.m_minibatch.push_back(stream);
    }

    return m;
}

std::shared_ptr<char> FrameModePacker::AllocateBuffer(size_t numElements, size_t elementSize)
{
    return std::shared_ptr<char>(
        reinterpret_cast<char*>(m_memoryProvider->Alloc(elementSize, numElements)),
        [this](char* p)
        {
            m_memoryProvider->Free(p);
        });
}
} } }
