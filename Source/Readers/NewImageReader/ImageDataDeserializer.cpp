//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "ImageDataDeserializer.h"
#include "ImageConfigHelper.h"
#include <opencv2/opencv.hpp>

namespace Microsoft { namespace MSR { namespace CNTK {

class ImageDataDeserializer::LabelGenerator
{
public:
    virtual void ReadLabelDataFor(SparseSequenceData& data, size_t classId) = 0;
    virtual ~LabelGenerator()
    {
    }
};

template <class TElement>
class TypedLabelGenerator : public ImageDataDeserializer::LabelGenerator
{
public:
    TypedLabelGenerator()
        : m_value(1)
    {
    }

    virtual void ReadLabelDataFor(SparseSequenceData& data, size_t classId) override
    {
        data.indices.resize(1);
        data.indices[0] = std::vector<size_t>{classId};
        data.data = &m_value;
    }

private:
    TElement m_value;
};

ImageDataDeserializer::ImageDataDeserializer(const ConfigParameters& config)
{
    ImageConfigHelper configHelper(config);
    m_streams = configHelper.GetStreams();
    assert(m_streams.size() == 2);
    const auto& label = m_streams[configHelper.GetLabelStreamId()];
    const auto& feature = m_streams[configHelper.GetFeatureStreamId()];

    label->storageType = StorageType::sparse_csc;
    feature->storageType = StorageType::dense;

    m_featureElementType = feature->elementType;
    size_t labelDimension = label->sampleLayout->GetHeight();

    if (label->elementType == ElementType::tfloat)
    {
        m_labelGenerator = std::make_shared<TypedLabelGenerator<float>>();
    }
    else if (label->elementType == ElementType::tdouble)
    {
        m_labelGenerator = std::make_shared<TypedLabelGenerator<double>>();
    }
    else
    {
        RuntimeError("Unsupported label element type %ull.", label->elementType);
    }

    CreateSequenceDescriptions(configHelper.GetMapPath(), labelDimension);
}

void ImageDataDeserializer::CreateSequenceDescriptions(std::string mapPath, size_t labelDimension)
{
    UNREFERENCED_PARAMETER(labelDimension);

    std::ifstream mapFile(mapPath);
    if (!mapFile)
    {
        RuntimeError("Could not open %s for reading.", mapPath.c_str());
    }

    std::string line{""};

    ImageSequenceDescription description;
    description.numberOfSamples = 1;
    description.isValid = true;
    for (size_t cline = 0; std::getline(mapFile, line); cline++)
    {
        std::stringstream ss{line};
        std::string imgPath;
        std::string clsId;
        if (!std::getline(ss, imgPath, '\t') || !std::getline(ss, clsId, '\t'))
        {
            RuntimeError("Invalid map file format, must contain 2 tab-delimited columns: %s, line: %d.", mapPath.c_str(), cline);
        }

        description.id = cline;
        description.chunkId = cline;
        description.path = imgPath;
        description.classId = std::stoi(clsId);
        assert(description.classId < labelDimension);
        m_imageSequences.push_back(description);
    }
}

std::vector<StreamDescriptionPtr> ImageDataDeserializer::GetStreams() const
{
    return m_streams;
}

std::vector<std::vector<SequenceDataPtr>> ImageDataDeserializer::GetSequencesById(const std::vector<size_t>& ids)
{
    assert(0 < ids.size());

    std::vector<std::vector<SequenceDataPtr>> result;

    m_currentImages.resize(ids.size());
    m_labels.resize(ids.size());
    result.resize(ids.size());

#pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < ids.size(); ++i)
    {
        assert(ids[i] < m_imageSequences.size());
        const auto& imageSequence = m_imageSequences[ids[i]];

        // Construct image
        m_currentImages[i] = std::move(cv::imread(imageSequence.path, cv::IMREAD_COLOR));
        cv::Mat& cvImage = m_currentImages[i];
        assert(cvImage.isContinuous());

        // Convert element type.
        // TODO in original image reader, this conversion happened later. Should we support all native CV element types to be able to match this behavior?
        int dataType = m_featureElementType == ElementType::tfloat ? CV_32F : CV_64F;
        if (cvImage.type() != CV_MAKETYPE(dataType, cvImage.channels()))
        {
            cvImage.convertTo(cvImage, dataType);
        }

        auto image = std::make_shared<DenseSequenceData>();
        image->data = cvImage.ptr();
        image->sampleLayout = std::make_shared<ImageLayout>(ImageLayoutWHC(cvImage.cols, cvImage.rows, cvImage.channels()));
        image->numberOfSamples = 1;
        assert(imageSequence.numberOfSamples == image->numberOfSamples);

        // Construct label
        if (m_labels[i] == nullptr)
        {
            m_labels[i] = std::make_shared<SparseSequenceData>();
        }
        m_labelGenerator->ReadLabelDataFor(*m_labels[i], imageSequence.classId);

        result[i] = std::move(std::vector<SequenceDataPtr>{image, m_labels[i]});
    }

    return result;
}

void ImageDataDeserializer::FillSequenceDescriptions(Timeline& timeline) const
{
    timeline.resize(m_imageSequences.size());
    std::transform(m_imageSequences.begin(), m_imageSequences.end(), timeline.begin(), [](const ImageSequenceDescription& desc)
                   {
                       return &desc;
                   });
}
} } }
