//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "ImageReader.h"
#include "commandArgUtil.h"
#include "ImageConfigHelper.h"
#include "ImageTransformers.h"
#include "BlockRandomizer.h"
#include "ImageDataDeserializer.h"
#include "FrameModePacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {

ImageReader::ImageReader(
    MemoryProviderPtr provider,
    const ConfigParameters& config)
    : m_provider(provider), m_seed(0)
{
    // In the future, deserializers and transformers will be dynamically loaded
    // from external libraries based on the configuration/brain script.
    // We will provide ability to implement the transformer and
    // deserializer interface not only in C++ but in scripting languages as well.

    ImageConfigHelper configHelper(config);
    m_streams = configHelper.GetStreams();
    assert(m_streams.size() == 2);
    auto deserializer = std::make_shared<ImageDataDeserializer>(config);
    auto randomizer = std::make_shared<BlockRandomizer>(0, SIZE_MAX, deserializer);

    auto cropper = std::make_shared<CropTransformer>();
    auto scaler = std::make_shared<ScaleTransformer>();
    auto mean = std::make_shared<MeanTransformer>();

    cropper->Initialize(randomizer, config);
    scaler->Initialize(cropper, config);
    mean->Initialize(scaler, config);
    m_transformer = mean;
}

std::vector<StreamDescriptionPtr> ImageReader::GetStreams()
{
    assert(!m_streams.empty());
    return m_streams;
}

void ImageReader::StartEpoch(const EpochConfiguration& config)
{
    assert(config.minibatchSize > 0);
    assert(config.totalSize > 0);

    m_transformer->StartEpoch(config);
    m_packer = std::make_shared<FrameModePacker>(
        m_provider,
        m_transformer,
        config.minibatchSize,
        m_streams);
}

Minibatch ImageReader::ReadMinibatch()
{
    assert(m_packer != nullptr);
    return m_packer->ReadMinibatch();
}
} } }
