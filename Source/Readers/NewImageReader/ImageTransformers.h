//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <set>
#include <unordered_map>
#include <random>
#include <opencv2/opencv.hpp>

#include "Transformer.h"
#include "ConcStack.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class ConfigParameters;

// Base class for image transformations based on OpenCV.
// Currently supports only dense data format.
class BaseTransformer : public Transformer
{
public:
    BaseTransformer();

    // Initializes the transformer.
    virtual void Initialize(TransformerPtr next, const ConfigParameters& readerConfig) override;

    // Description of streams that the transformer provides.
    virtual std::vector<StreamDescriptionPtr> GetStreams() const override;

    // Gets next "count" sequences. Sequences contains data for all streams.
    virtual Sequences GetNextSequences(size_t count) override;

    // Sets configuration for the current epoch.
    virtual void StartEpoch(const EpochConfiguration& config) override;

protected:
    using UniRealT = std::uniform_real_distribution<double>;
    using UniIntT = std::uniform_int_distribution<int>;

    // Applies transformation to the image.
    virtual void Apply(cv::Mat& from) = 0;

    // Seed  getter.
    unsigned int GetSeed() const
    {
        return m_seed;
    }

    const std::vector<StreamId>& GetFeatureStreamIds() const;
    std::vector<StreamDescriptionPtr> m_streams;

private:
    // Applies transformation to the sequence.
    SequenceDataPtr Apply(const DenseSequenceData& sequence, StreamDescriptionPtr stream, cv::Mat& buffer);

    std::vector<StreamId> m_featureStreamIds;
    TransformerPtr m_next;
    unsigned int m_seed;
    std::vector<std::vector<cv::Mat>> m_buffer;
};

class CropTransformer : public BaseTransformer
{
public:
    CropTransformer();
    virtual void Initialize(TransformerPtr next, const ConfigParameters& readerConfig) override;

protected:
    virtual void Apply(cv::Mat& mat) override;

private:
    enum class CropType
    {
        Center = 0,
        Random = 1
    };
    enum class RatioJitterType
    {
        None = 0,
        UniRatio = 1,
        UniLength = 2,
        UniArea = 3
    };

    void InitFromConfig(const ConfigParameters& config);
    CropType ParseCropType(const std::string& src);
    RatioJitterType ParseJitterType(const std::string& src);
    cv::Rect GetCropRect(CropType type, int crow, int ccol, double cropRatio, std::mt19937& rng);

    conc_stack<std::unique_ptr<std::mt19937>> m_rngs;
    CropType m_cropType;
    double m_cropRatioMin;
    double m_cropRatioMax;
    RatioJitterType m_jitterType;
    bool m_hFlip;
};

class ScaleTransformer : public BaseTransformer
{
public:
    ScaleTransformer();
    virtual void Initialize(TransformerPtr next, const ConfigParameters& readerConfig) override;

private:
    void InitFromConfig(const ConfigParameters& config);
    virtual void Apply(cv::Mat& mat) override;

    using StrToIntMapT = std::unordered_map<std::string, int>;
    StrToIntMapT m_interpMap;
    std::vector<int> m_interp;

    conc_stack<std::unique_ptr<std::mt19937>> m_rngs;
    int m_dataType;
    size_t m_imgWidth;
    size_t m_imgHeight;
    size_t m_imgChannels;
};

class MeanTransformer : public BaseTransformer
{
public:
    MeanTransformer();
    virtual void Initialize(TransformerPtr next, const ConfigParameters& readerConfig) override;

private:
    virtual void Apply(cv::Mat& mat) override;
    void InitFromConfig(const ConfigParameters& config);

    cv::Mat m_meanImg;
};
} } }
