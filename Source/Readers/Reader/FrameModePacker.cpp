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
                return s->storageType == StorageType::sparse_csc;
            }) == m_outputStreams.end());

    for (const auto& stream : streams)
    {
        assert(stream->elementType == ElementType::tfloat || stream->elementType == ElementType::tdouble);
        m_streamBuffers.push_back(
            AllocateBuffer(m_mbSize * stream->sampleLayout->GetNumElements(), GetSizeByType(stream->elementType)));
    }
}

Minibatch FrameModePacker::ReadMinibatch()
{
    assert(m_mbSize > 0);

    Minibatch m;
    m.atEndOfEpoch = false;

    auto images = m_transformer->GetNextSequences(m_mbSize);

    if (images.m_endOfEpoch)
    {
        m.atEndOfEpoch = true;
    }

    for (size_t i = 0; i < images.m_data.size(); i++)
    {
        assert(m_streamBuffers.size() == images.m_data[i].size());
        for (int j = 0; j < images.m_data[i].size(); ++j)
        {
            size_t elementSize = GetSizeByType(m_inputStreams[j]->elementType);
            size_t dimensions = m_inputStreams[j]->sampleLayout->GetNumElements() * elementSize;
            const char* source = reinterpret_cast<char*>(images.m_data[i][j]->data);
            if (m_inputStreams[j]->storageType == StorageType::dense)
            {
                DenseSequenceData& data = reinterpret_cast<DenseSequenceData&>(*images.m_data[i][j]);
                assert(data.numberOfSamples == 1);

                std::copy(
                    source,
                    source + dimensions,
                    m_streamBuffers[j].get() + dimensions * i);
            }
            else if (m_inputStreams[j]->storageType == StorageType::sparse_csc)
            {
                SparseSequenceData& data = reinterpret_cast<SparseSequenceData&>(*images.m_data[i][j]);
                assert(data.indices.size() == 1);

                std::fill(m_streamBuffers[j].get() + i * dimensions, m_streamBuffers[j].get() + (i + 1) * dimensions, 0);
                size_t nonZeroCount = data.indices[0].size();
                for (size_t nonZeroIndex = 0; nonZeroIndex < nonZeroCount; ++nonZeroIndex)
                {
                    size_t rowIndex = data.indices[0][nonZeroIndex];
                    char* destination = m_streamBuffers[j].get() + dimensions * i + rowIndex * elementSize;
                    std::copy(source + nonZeroIndex * elementSize, source + (nonZeroIndex + 1) * elementSize, destination);
                }
            }
            else
            {
                RuntimeError("Storage type %d is not supported.", m_inputStreams[j]->storageType);
            }
        }
    }

    if (images.m_data.size() == 0)
    {
        return m;
    }

    m_minibatchLayout->Init(images.m_data.size(), 1);
    for (int i = 0; i < m_outputStreams.size(); ++i)
    {
        size_t dimensions = m_outputStreams[i]->sampleLayout->GetNumElements() * GetSizeByType(m_outputStreams[i]->elementType);
        auto stream = std::make_shared<Stream>();
        stream->data = m_streamBuffers[i].get();
        stream->dataSize = images.m_data.size() * dimensions;
        stream->layout = m_minibatchLayout;

        m.minibatch.push_back(stream);
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
