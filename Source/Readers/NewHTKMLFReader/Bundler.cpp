//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "Bundler.h"
#include "ConfigHelper.h"
#include "HTKDataDeserializer.h"
#include "MLFDataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

std::vector<DataDeserializerPtr> CreateDeserializers(const ConfigParameters& readerConfig,
                                                     bool framemode,
                                                     size_t elementSize)
{
    std::vector<std::wstring> featureNames;
    std::vector<std::wstring> labelNames;

    std::vector<std::wstring> notused;
    ConfigHelper::GetDataNamesFromConfig(readerConfig, featureNames, labelNames, notused, notused);
    if (featureNames.size() < 1 || labelNames.size() < 1)
    {
        // eldak: Don't we support unsupervised training?
        InvalidArgument("network needs at least 1 input and 1 output specified!");
    }

    std::vector<HTKDataDeserializerPtr> featureDeserializers;
    std::vector<MLFDataDeserializerPtr> labelDeserializers;

    for (const auto& featureName : featureNames)
    {
        auto deserializer = std::make_shared<HTKDataDeserializer>(readerConfig(featureName), elementSize, framemode, featureName);
        featureDeserializers.push_back(deserializer);
    }

    assert(featureDeserializers.size() == 1);

    for (const auto& labelName : labelNames)
    {
        auto deserializer = std::make_shared<MLFDataDeserializer>(readerConfig(labelName), elementSize,
                                                                  featureDeserializers[0].get(), framemode, labelName);

        labelDeserializers.push_back(deserializer);
    }

    assert(labelDeserializers.size() == 1);

    std::vector<DataDeserializerPtr> deserializers;
    deserializers.insert(deserializers.end(), featureDeserializers.begin(), featureDeserializers.end());
    deserializers.insert(deserializers.end(), labelDeserializers.begin(), labelDeserializers.end());

    // Checking end sequences.
    size_t expected = deserializers[0]->GetSequenceDescriptions().size();
    std::vector<bool> isValid(expected, true);
    for (auto d : deserializers)
    {
        const auto& sequences = d->GetSequenceDescriptions();
        if (sequences.size() != expected)
        {
            RuntimeError("We have some invalid alignment\n");
        }

        foreach_index (i, sequences)
        {
            isValid[i] = isValid[i] && sequences[i]->isValid;
            assert(isValid[i]);
        }
    }

    // shouldn't this be checked (again) later? more utterances can be invalidated...
    size_t invalidUtts = 0;
    foreach_index (i, isValid)
    {
        if (!isValid[i])
        {
            invalidUtts++;
        }
    }
    assert(invalidUtts == 0); // For us it's zero

    if (invalidUtts > isValid.size() / 2)
    {
        RuntimeError("minibatchutterancesource: too many files with inconsistent durations, assuming broken configuration\n");
    }
    else if (invalidUtts > 0)
    {
        fprintf(stderr, "Found inconsistent durations across feature streams in %llu out of %llu files\n", invalidUtts, isValid.size());
    }

    return deserializers;
}

Bundler::Bundler(
    const ConfigParameters& readerConfig,
    bool framemode,
    int verbosity,
    DataDeserializerPtr driver,
    std::vector<DataDeserializerPtr> deserializers)
    : m_deserializers(deserializers), m_driver(driver)
{
    m_framemode = framemode;
    m_chunksinram = 0;
    m_verbosity = readerConfig(L"verbosity", 2);
    m_verbosity = verbosity; // not needed

    std::vector<StreamDescriptionPtr> streams;
    for (auto d : deserializers)
    {
        for (auto i : d->GetStreams())
        {
            StreamDescriptionPtr stream = std::make_shared<StreamDescription>();
            stream->id = streams.size();
            stream->name = i->name;
            stream->sampleLayout = i->sampleLayout;
            streams.push_back(stream);
        }
    }

    m_streams = streams;
}

void Bundler::RequireChunk(size_t chunkindex)
{
    // currently simply redirect
    // todo: we should have a mapping per deserializer actually.
    for (const auto& d : m_deserializers)
    {
        d->RequireChunk(chunkindex);
    }
}

void Bundler::ReleaseChunk(size_t chunkIndex)
{
    // currently simply redirect
    // todo: we should have a mapping per deserializer actually.
    for (const auto& d : m_deserializers)
    {
        d->ReleaseChunk(chunkIndex);
    }
}

const Timeline& Bundler::GetSequenceDescriptions() const
{
    // TODO: we probably will take different deserializers from here.
    return m_driver->GetSequenceDescriptions();
}

std::vector<StreamDescriptionPtr> Bundler::GetStreams() const
{
    return m_streams;
}

std::vector<std::vector<SequenceDataPtr>> Bundler::GetSequencesById(const std::vector<size_t>& ids)
{
    assert(ids.size() == 1); // TODO
    std::vector<std::vector<SequenceDataPtr>> result;
    result.push_back(std::vector<SequenceDataPtr>{});
    for (auto& d : m_deserializers)
    {
        auto r = d->GetSequencesById(ids);
        result[0].insert(result[0].end(), r[0].begin(), r[0].end());
    }
    return result;
}

void Bundler::StartEpoch(const EpochConfiguration& /*config*/)
{
    // TODO do we keep SetEpochConfiguration(), now empty?
}
} } }
