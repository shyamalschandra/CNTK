//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

// Memory provider. It is injected by CNTK into the reader.
// Should be used for allocating the stream data returned by the reader.
class MemoryProvider
{
public:
    virtual void* Alloc(size_t element, size_t numberOfElements) = 0;
    virtual void Free(void* ptr) = 0;

    virtual ~MemoryProvider() = 0
    {
    }
};

typedef std::shared_ptr<MemoryProvider> MemoryProviderPtr;
} } }
