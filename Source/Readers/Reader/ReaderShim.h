//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ReaderShim.h: header for shim wrapping the new reader interface
//

#pragma once

#include <map>
#include <string>
#include "DataReader.h"
#include "commandArgUtil.h"
#include "Reader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

typedef ReaderPtr (*ReaderFactory)(const ConfigParameters& parameters);

template <class ElemType>
class ReaderShim : public IDataReader<ElemType>
{
    ReaderPtr m_reader;
    ReaderFactory m_factory;
    bool m_endOfEpoch;

    MBLayoutPtr m_layout;

    std::map<std::wstring, size_t> m_nameToStreamId;
    std::vector<StreamDescriptionPtr> m_streams;

public:
    explicit ReaderShim(ReaderFactory factory);
    virtual ~ReaderShim()
    {
    }

    virtual void Init(const ScriptableObjects::IConfigRecord& /*config*/) override
    {
        assert(false);
    }
    virtual void Init(const ConfigParameters& config) override;

    virtual void Destroy() override
    {
        delete this;
    }

    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples) override;
    virtual void StartDistributedMinibatchLoop(size_t requestedMBSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples) override;

    virtual bool SupportsDistributedMBRead() const override
    {
        return true;
    }

    virtual bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices) override;

    virtual bool DataEnd(EndDataType endDataType) override;

    void CopyMBLayoutTo(MBLayoutPtr) override;

    virtual size_t GetNumParallelSequences() override;
};
} } }
