//
// <copyright file="NonlinearityNodes.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include <unordered_set>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <algorithm>
#include <assert.h>
#include <atomic>
#include <sstream>
#include <iostream>

#include "Basics.h"
#include "Matrix.h"
#include "ComputationNode.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // -----------------------------------------------------------------------
    // NonlinearityNodeBase (input) -- abstract base class that holds what's shared between non-linearity nodes like Sigmoid
    // -----------------------------------------------------------------------

    // shared base for all elemen-twise non-linearities
    // What this adds over a ComputationNode<ElemType> is a member m_gradient for temp use by derived classes.
    // TODO: Remove the Evaluate and Partial overrides from here entirely, as they don't really add value after all the code simplifications.
    template<class ElemType>
    class NonlinearityNodeBase : public ComputationNode<ElemType>, public NumInputs<1>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        //virtual ComputationNodeBase * NewThis(DEVICEID_TYPE deviceId, const wstring & name) = 0;
        NonlinearityNodeBase(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        // TODO: with FrameRange, this code has now been reduced so much that there is no need to have these overrides here; they can just be implemented in the derived classes directly.
        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            assert(inputIndex == 0); inputIndex;
            auto gradient = Inputs(0)->GradientSlice(frameRange);
            ComputeInputPartialV(*m_gradient, Inputs(0)->ValueSlice(frameRange), gradient, GradientSlice(frameRange));
        }

        // derived class implement the actual non-linear operation
        virtual void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) = 0;

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            auto values = ValueSlice(frameRange);
            EvaluateThisNodeV(values, Inputs(0)->ValueSlice(frameRange));
        }

        // derived class implement the actual non-linear operation
        virtual void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues) = 0;

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            ValidateUnaryMap(isFinalValidationPass);
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<NonlinearityNodeBase<ElemType>>(nodeP);
                *node->m_gradient = *m_gradient;
            }
        }

        // request matrices that are needed for gradient computation
        virtual void RequestMatricesBeforeGradientComp(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeGradientComp(matrixPool);
            RequestMatrixFromPool(m_gradient, matrixPool);
        }

        // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        virtual void ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool)
        {
            Base::ReleaseMatricesAfterGradientComp(matrixPool);
            ReleaseMatrixToPool(m_gradient, matrixPool);
        }
    protected:
        shared_ptr<Matrix<ElemType>> m_gradient;
    };

#define UsingNonlinearityNodeBaseMembers UsingComputationNodeMembersBoilerplate; using Base::m_gradient

    // -----------------------------------------------------------------------
    // RectifiedLinearNode (input) -- ReLU non-linearity
    // -----------------------------------------------------------------------

    template<class ElemType>
    class RectifiedLinearNode : public NonlinearityNodeBase<ElemType>
    {
        typedef NonlinearityNodeBase<ElemType> Base; UsingNonlinearityNodeBaseMembers;
        static const std::wstring TypeName() { return L"RectifiedLinear"; }
    public:
        RectifiedLinearNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNodeBase<ElemType>(deviceId, name)
        { }

        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) override
        {
            gradient.AssignLinearRectifierDerivativeOf(inputFunctionValues);
#if DUMPOUTPUT
            inputGradientValues.Print("RecitifiedLinearNode-Partial-in");
#endif
            inputGradientValues.AddElementProductOf(gradientValues, gradient);
#if DUMPOUTPUT
            inputGradientValues.Print("RecitifiedLinearNode-Partial-out");
#endif
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)
        {
            functionValues.AssignTruncateBottomOf(inputFunctionValues, 0);
#if NANCHECK
            functionValues.HasNan("RectifiedLinear");
#endif
#if DUMPOUTPUT
            functionValues.Print("RectifiedLinearNode");
#endif
        }
    };

    template class RectifiedLinearNode<float>;
    template class RectifiedLinearNode<double>;

    // -----------------------------------------------------------------------
    // SigmoidNode (input) -- sigmoid non-linearity
    // -----------------------------------------------------------------------

    template<class ElemType>
    class SigmoidNode : public NonlinearityNodeBase<ElemType>
    {
        typedef NonlinearityNodeBase<ElemType> Base; UsingNonlinearityNodeBaseMembers;
        static const std::wstring TypeName() { return L"Sigmoid"; }
    public:
        SigmoidNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNodeBase<ElemType>(deviceId, name)
        { }

        // we should get rid of this code dup, need to unify the -V functions
        void ComputeInputPartialMap(const size_t inputIndex)
        {
            assert(inputIndex == 0); inputIndex;
            ComputeInputPartialS(*m_gradient, Inputs(0)->GradientValues(), GradientValues(), FunctionValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            assert(inputIndex == 0); inputIndex;

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            ComputeInputPartialS(*m_gradient, sliceInputGrad, sliceOutputGrad, sliceOutputValue);
        }

        // should be:
        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) { gradient; inputFunctionValues;  inputGradientValues;  gradientValues;  LogicError("wrong signature :( need to unify code more"); }
        // but is:
        /*virtual*/ void ComputeInputPartialS(Matrix<ElemType>& gradient, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)
        {
            gradient.AssignSigmoidDerivativeOf(functionValues);
            inputGradientValues.AddElementProductOf(gradientValues, gradient);
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)
        {
            functionValues.AssignSigmoidOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Sigmoid");
#endif
        }
    };

    template class SigmoidNode<float>;
    template class SigmoidNode<double>;

    // -----------------------------------------------------------------------
    // TanhNode (input) -- tanh non-linearity
    // -----------------------------------------------------------------------

    template<class ElemType>
    class TanhNode : public NonlinearityNodeBase<ElemType>
    {
        typedef NonlinearityNodeBase<ElemType> Base; UsingNonlinearityNodeBaseMembers;
        static const std::wstring TypeName() { return L"Tanh"; }
    public:
        TanhNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNodeBase<ElemType>(deviceId, name)
        { }

        // TODO: unify signature & get rid of code dup
        void ComputeInputPartialMap(const size_t inputIndex)
        {
            assert(inputIndex == 0); inputIndex;
            ComputeInputPartialS(*m_gradient, Inputs(0)->GradientValues(), GradientValues(), FunctionValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            assert(inputIndex == 0); inputIndex;

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            ComputeInputPartialS(*m_gradient, sliceInputGrad, sliceOutputGrad, sliceOutputValue);
        }

        // should be:
        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) { gradient; inputFunctionValues;  inputGradientValues;  gradientValues;  LogicError("wrong signature :( need to unify code more"); }
        // but is:
        /*virtual*/ void ComputeInputPartialS(Matrix<ElemType>& gradient, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)
        {
            gradient.AssignElementProductOf(functionValues, functionValues); // v .* v
            gradient.AssignDifferenceOf(1, gradient); // 1-v^2

            inputGradientValues.AddElementProductOf(gradientValues, gradient); // += d .* ((1-v) .* v))
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)
        {
            functionValues.AssignTanhOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Tanh");
#endif
        }
    };

    template class TanhNode<float>;
    template class TanhNode<double>;

    // -----------------------------------------------------------------------
    // LogNode (input) -- component-wise log() of input
    // -----------------------------------------------------------------------

    template<class ElemType>
    class LogNode : public NonlinearityNodeBase<ElemType>
    {
        typedef NonlinearityNodeBase<ElemType> Base; UsingNonlinearityNodeBaseMembers;
        static const std::wstring TypeName() { return L"Log"; }
    public:
        LogNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNodeBase<ElemType>(deviceId, name)
        { }

        // TODO: get rid of code dup
        void ComputeInputPartialMap(const size_t inputIndex)
        {
            assert(inputIndex == 0); inputIndex;
            ComputeInputPartialS(*m_gradient, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            assert(inputIndex == 0); inputIndex;

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            Matrix<ElemType> sliceInputValue = Inputs(0)->ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            ComputeInputPartialS(*m_gradient, sliceInputGrad, sliceInputValue, sliceOutputGrad);
        }

        // should be:
        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) { gradient; inputFunctionValues;  inputGradientValues;  gradientValues;  LogicError("wrong signature :( need to unify code more"); }
        // but is:
        /*virtual*/ void ComputeInputPartialS(Matrix<ElemType>& gradient, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& inputFunctionValues, const Matrix<ElemType>& gradientValues)
        {
            gradient.AssignElementInverseOf(inputFunctionValues); // 1/x (x is input to log(x))

            inputGradientValues.AddElementProductOf(gradientValues, gradient);
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)
        {
            functionValues.AssignLogOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Log");
#endif
        }
    };

    template class LogNode<float>;
    template class LogNode<double>;

    // -----------------------------------------------------------------------
    // ExpNode (input) -- component-wise exp() of input
    // -----------------------------------------------------------------------

    template<class ElemType>
    class ExpNode : public NonlinearityNodeBase<ElemType>
    {
        typedef NonlinearityNodeBase<ElemType> Base; UsingNonlinearityNodeBaseMembers;
        static const std::wstring TypeName() { return L"Exp"; }
    public:
        ExpNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNodeBase<ElemType>(deviceId, name)
        { }

        // TODO: get rid of code dup
        void ComputeInputPartialMap(const size_t inputIndex)
        {
            assert(inputIndex == 0); inputIndex;
            ComputeInputPartialS(*m_gradient, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            assert(inputIndex == 0); inputIndex;

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            Matrix<ElemType> sliceInputValue = Inputs(0)->ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            ComputeInputPartialS(*m_gradient, sliceInputGrad, sliceInputValue, sliceOutputGrad);
        }

        // should be:
        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) { gradient; inputFunctionValues;  inputGradientValues;  gradientValues;  LogicError("wrong signature :( need to unify code more"); }
        // but is:
        /*virtual*/ void ComputeInputPartialS(Matrix<ElemType>& gradient, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& inputFunctionValues, const Matrix<ElemType>& gradientValues)
        {
            gradient.AssignExpOf(inputFunctionValues); // Exp(x) is its own partial
            inputGradientValues.AddElementProductOf(gradientValues, gradient);
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)
        {
            functionValues.AssignExpOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Exp");
#endif
        }
    };

    template class ExpNode<float>;
    template class ExpNode<double>;

    // -----------------------------------------------------------------------
    // CosineNode (input) -- component-wise cos() of input
    // -----------------------------------------------------------------------

    template<class ElemType>
    class CosineNode : public NonlinearityNodeBase<ElemType>
    {
        typedef NonlinearityNodeBase<ElemType> Base; UsingNonlinearityNodeBaseMembers;
        static const std::wstring TypeName() { return L"Cosine"; }
    public:
        CosineNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNodeBase<ElemType>(deviceId, name)
        { }

        // TODO: code dup
        void ComputeInputPartialMap(const size_t inputIndex)
        {
            assert(inputIndex == 0); inputIndex;
            ComputeInputPartialS(*m_gradient, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            assert(inputIndex == 0); inputIndex;

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            Matrix<ElemType> sliceInputValue = Inputs(0)->ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            ComputeInputPartialS(*m_gradient, sliceInputGrad, sliceInputValue, sliceOutputGrad);
        }


        // should be:
        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) { gradient; inputFunctionValues;  inputGradientValues;  gradientValues;  LogicError("wrong signature :( need to unify code more"); }
        // but is:
        /*virtual*/ void ComputeInputPartialS(Matrix<ElemType>& gradient, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& inputFunctionValues, const Matrix<ElemType>& gradientValues)
        {
            gradient.AssignNegativeSineOf(inputFunctionValues); // -sin(x) (x is input to Cosine(x))
            inputGradientValues.AddElementProductOf(gradientValues, gradient);
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)
        {
            functionValues.AssignCosineOf(inputFunctionValues);
#if NANCHECK
            functionValues.HasNan("Cosine");
#endif
        }
    };

    template class CosineNode<float>;
    template class CosineNode<double>;

    // -----------------------------------------------------------------------
    // SoftmaxNode (input) -- soft-max over input vector(s)
    // -----------------------------------------------------------------------

    //we assume it's  column-wise by default
    //the derivative will increase the Matrix<ElemType> size to the power of column size and should not be used.
    template<class ElemType>
    class SoftmaxNode : public NonlinearityNodeBase<ElemType>
    {
        typedef NonlinearityNodeBase<ElemType> Base; UsingNonlinearityNodeBaseMembers;
        static const std::wstring TypeName() { return L"Softmax"; }
    public:
        SoftmaxNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNodeBase<ElemType>(deviceId, name)
        { }

        // TODO: code dup
        void ComputeInputPartialMap(const size_t inputIndex)
        {
            assert(inputIndex == 0); inputIndex;
            ComputeInputPartialS(*m_gradient, *m_diff, Inputs(0)->GradientValues(), GradientValues(), FunctionValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            assert(inputIndex == 0); inputIndex;

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            ComputeInputPartialS(*m_gradient, *m_diff, sliceInputGrad, sliceOutputGrad, sliceOutputValue);
        }

        // should be:
        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) { gradient; inputFunctionValues;  inputGradientValues;  gradientValues;  LogicError("wrong signature :( need to unify code more"); }
        // but is:
        /*virtual*/ void ComputeInputPartialS(Matrix<ElemType>& gradient, Matrix<ElemType>& diff, Matrix<ElemType>& inputGradientValues,
            const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)
        {
            gradient.AssignInnerProductOf(gradientValues, functionValues, true);
            diff.AssignDifferenceOf(gradientValues, gradient);

            inputGradientValues.AddElementProductOf(diff, functionValues);
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)
        {
            functionValues.AssignLogSoftmaxOf(inputFunctionValues, true);
            functionValues.InplaceExp();
#if NANCHECK
            functionValues.HasNan("SoftMax");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            ValidateUnaryMap(isFinalValidationPass);
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<SoftmaxNode<ElemType>>(nodeP);
                *node->m_diff = *m_diff;
            }
        }
        //request matrices that are needed for gradient computation
        virtual void RequestMatricesBeforeGradientComp(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeGradientComp(matrixPool);
            RequestMatrixFromPool(m_diff, matrixPool);
        }

        //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        virtual void ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool)
        {
            Base::ReleaseMatricesAfterGradientComp(matrixPool);
            ReleaseMatrixToPool(m_diff, matrixPool);
        }
    private:
        shared_ptr<Matrix<ElemType>> m_diff;
    };

    template class SoftmaxNode<float>;
    template class SoftmaxNode<double>;

    // -----------------------------------------------------------------------
    // LogSoftmaxNode (input) -- log of soft-max over input vector(s)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class LogSoftmaxNode : public NonlinearityNodeBase<ElemType>
    {
        typedef NonlinearityNodeBase<ElemType> Base; UsingNonlinearityNodeBaseMembers;
        static const std::wstring TypeName() { return L"LogSoftmax"; }
    public:
        LogSoftmaxNode(DEVICEID_TYPE deviceId, const wstring & name) :
            NonlinearityNodeBase<ElemType>(deviceId, name)
        { }

        // TODO: code dup
        void ComputeInputPartialMap(const size_t inputIndex)
        {
            assert(inputIndex == 0); inputIndex;
            ComputeInputPartialS(*m_gradient, *m_softmax, Inputs(0)->GradientValues(), GradientValues(), FunctionValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            assert(inputIndex == 0); inputIndex;

            Matrix<ElemType> sliceInputGrad = Inputs(0)->GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            ComputeInputPartialS(*m_gradient, *m_softmax, sliceInputGrad, sliceOutputGrad, sliceOutputValue);
        }

        // should be:
        /*virtual*/ void ComputeInputPartialV(Matrix<ElemType>& gradient, const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues) { gradient; inputFunctionValues;  inputGradientValues;  gradientValues;  LogicError("wrong signature :( need to unify code more"); }
        // but is:
        /*virtual*/ void ComputeInputPartialS(Matrix<ElemType>& gradient, Matrix<ElemType>& softmax, Matrix<ElemType>& inputGradientValues,
            const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& functionValues)
        {
            softmax.AssignExpOf(functionValues);
            Matrix<ElemType>::VectorSum(gradientValues, gradient, true);
            softmax.RowElementMultiplyWith(gradient);
            Matrix<ElemType>::AddScaledDifference(1.0, gradientValues, softmax, inputGradientValues);
        }

        /*virtual*/ void EvaluateThisNodeV(Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues)
        {
            functionValues.AssignLogSoftmaxOf(inputFunctionValues, true);
#if NANCHECK
            functionValues.HasNan("LogSoftMax");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            ValidateUnaryMap(isFinalValidationPass);
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<LogSoftmaxNode<ElemType>>(nodeP);
                *node->m_softmax = *m_softmax;
            }
        }
        //request matrices that are needed for gradient computation
        virtual void RequestMatricesBeforeGradientComp(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeGradientComp(matrixPool);
            RequestMatrixFromPool(m_softmax, matrixPool);
        }

        //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        virtual void ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool)
        {
            Base::ReleaseMatricesAfterGradientComp(matrixPool);
            ReleaseMatrixToPool(m_softmax, matrixPool);
        }
    private:
        shared_ptr<Matrix<ElemType>> m_softmax;
    };

    template class LogSoftmaxNode<float>;
    template class LogSoftmaxNode<double>;

    // -----------------------------------------------------------------------
    // GMMLogLikelihoodNode (unnormedPrior, means, logStdDevs, features) -- GMM log LL over input vector(s)
    // -----------------------------------------------------------------------

    //calculates: the log likelihood of a feature given GMM parameters
    template<class ElemType>
    class GMMLogLikelihoodNode : public ComputationNode<ElemType>, public NumInputs<4>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"GMMLogLikelihood"; }
    public:
        GMMLogLikelihoodNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name)
        { }

        void ComputeInputPartialMap(const size_t inputIndex)
        {
            switch (inputIndex)
            {
            case 0:
                ComputeInputPartialUnnormedPrior(Inputs(0)->GradientValues(), GradientValues(), *m_prior, *m_posterior, *m_temp);
                break;
            case 1:
                ComputeInputPartialMean(Inputs(1)->GradientValues(), GradientValues(), *m_normedDeviationVectors, *m_posterior, *m_temp);
                break;
            case 2:
                ComputeInputPartialLogStddev(Inputs(2)->GradientValues(), GradientValues(), *m_normedDeviation, *m_posterior, *m_temp);
                break;
            case 3:
                ComputeInputPartialFeature(Inputs(3)->GradientValues(), GradientValues(), *m_normedDeviationVectors, *m_posterior, *m_temp);
                break;
            default:
                InvalidArgument("GMMLogLikelihoodNode only takes four inputs.");
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            //get the right slice 
            const size_t colsPrior = Inputs(0)->GetNumCols();

            Matrix<ElemType> sliceGradientValue = DataSlice(*m_gradientValues, frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> slicePosterior = DataSlice(*m_posterior, frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            switch (inputIndex)
            {
            case 0:
            {
                if (colsPrior == 1)
                        ComputeInputPartialUnnormedPrior(Inputs(0)->GradientValues(), sliceGradientValue, *m_prior, slicePosterior, *m_temp);
                else
                {
                    Matrix<ElemType> sliceUnnormedPriorGradient = Inputs(0)->GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
                        Matrix<ElemType> slicePrior = DataSlice(*m_prior, frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
                        ComputeInputPartialUnnormedPrior(sliceUnnormedPriorGradient, sliceGradientValue, slicePrior, slicePosterior, *m_temp);
                }
            }
            break;
            case 1:
            {
                      Matrix<ElemType> sliceNormedDeviationVectors = DataSlice(*m_normedDeviationVectors, frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
                if (colsPrior == 1)
                        ComputeInputPartialMean(Inputs(1)->GradientValues(), sliceGradientValue, sliceNormedDeviationVectors, slicePosterior, *m_temp);
                else
                {
                    Matrix<ElemType> sliceMeanGradient = Inputs(1)->GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
                        ComputeInputPartialMean(sliceMeanGradient, sliceGradientValue, sliceNormedDeviationVectors, slicePosterior, *m_temp);
                }
            }
            break;
            case 2:
            {
                    Matrix<ElemType> sliceNormedDeviation = DataSlice(*m_normedDeviation, frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
                if (colsPrior == 1)
                        ComputeInputPartialLogStddev(Inputs(2)->GradientValues(), sliceGradientValue, sliceNormedDeviation, slicePosterior, *m_temp);
                else
                {
                    Matrix<ElemType> sliceLotStddevGradient = Inputs(2)->GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
                    ComputeInputPartialLogStddev(sliceLotStddevGradient, sliceGradientValue, sliceNormedDeviation, slicePosterior, *m_temp);
                }
            }
            break;
            case 3:
            {
                Matrix<ElemType> sliceNormedDeviationVectors = DataSlice(*m_normedDeviationVectors, frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
                Matrix<ElemType> sliceFeatureGradient = Inputs(3)->GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
                ComputeInputPartialFeature(sliceFeatureGradient, sliceGradientValue, sliceNormedDeviationVectors, slicePosterior, *m_temp);
            }
            break;
            default:
                InvalidArgument("GMMLogLikelihoodNode criterion only takes four inputs.");
            }
        }

        /*TODO: merge with call site*/void ComputeInputPartialUnnormedPrior(Matrix<ElemType>& unnormedPriorGradientValues, const Matrix<ElemType>& gradientValues,
            const Matrix<ElemType>& prior, const Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
        {
            temp.AssignDifferenceOf(posterior, prior);
            temp.RowElementMultiplyWith(gradientValues);
            if (prior.GetNumCols() == posterior.GetNumCols())
                unnormedPriorGradientValues += temp;
            else if (prior.GetNumCols() == 1)
                Matrix<ElemType>::MultiplyAndAdd(temp, false, ConstOnes(posterior.GetNumCols(), 1, unnormedPriorGradientValues.GetDeviceId()), false, unnormedPriorGradientValues);
            else
                RuntimeError("GMMLogLikelihoodNode: UnnormedPrior should either have same number of columns as the features or have only one column.");
        }

        /*TODO: merge with call site*/void ComputeInputPartialMean(Matrix<ElemType>& meanGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& normedDeviationVectors,
            Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
        {
            size_t numComponent = posterior.GetNumRows();
            size_t numSamples = posterior.GetNumCols();
            size_t featureSize = normedDeviationVectors.GetNumRows() / numComponent;

            temp.SetValue(normedDeviationVectors); //recall normedDeviationVectors <-- (x-u_c)/(stddev^2)
            temp.Reshape(featureSize, numSamples* numComponent);

            posterior.Reshape(1, numSamples* numComponent);
            temp.RowElementMultiplyWith(posterior); //temp <-- posterior * (x-u_c)/(stddev^2)

            posterior.Reshape(numComponent, numSamples);  //reshape back
            temp.Reshape(featureSize * numComponent, numSamples); //reshape back

            temp.RowElementMultiplyWith(gradientValues);

            if (numSamples == meanGradientValues.GetNumCols())
                meanGradientValues += temp;
            else if (meanGradientValues.GetNumCols() == 1)
                Matrix<ElemType>::MultiplyAndAdd(temp, false, ConstOnes(numSamples, 1, meanGradientValues.GetDeviceId()), false, meanGradientValues);
            else
                RuntimeError("GMMLogLikelihoodNode: stddev should either have same number of columns as the features or have only one column.");
        }

        /*TODO: merge with call site*/void ComputeInputPartialLogStddev(Matrix<ElemType>& logStddevGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& normedDeviation,
            const Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
        {
            size_t numComponent = posterior.GetNumRows();
            size_t numSamples = posterior.GetNumCols();

            temp.AssignDifferenceOf(normedDeviation, (ElemType)numComponent);
            temp.ElementMultiplyWith(posterior);
            temp.RowElementMultiplyWith(gradientValues);
            if (logStddevGradientValues.GetNumCols() == numSamples)
                logStddevGradientValues += temp;
            else if (logStddevGradientValues.GetNumCols() == 1)
                Matrix<ElemType>::MultiplyAndAdd(temp, false, ConstOnes(numSamples, 1, logStddevGradientValues.GetDeviceId()), false, logStddevGradientValues);
            else
                RuntimeError("GMMLogLikelihoodNode: stddev should either have same number of columns as the features or have only one column.");
        }

        /*TODO: merge with call site*/void ComputeInputPartialFeature(Matrix<ElemType>& featureGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& normedDeviationVectors,
            Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
        {
            size_t numComponent = posterior.GetNumRows();
            size_t numSamples = posterior.GetNumCols();
            size_t featureSize = normedDeviationVectors.GetNumRows() / numComponent;

            temp.SetValue(normedDeviationVectors);
            temp *= -1;
            temp.Reshape(featureSize, numSamples* numComponent);
            posterior.Reshape(1, numSamples* numComponent);
            temp.RowElementMultiplyWith(posterior);

            posterior.Reshape(numComponent, numSamples);
            temp.Reshape(featureSize * numComponent, numSamples);
            temp.RowElementMultiplyWith(gradientValues);

            for (int i = 0; i < numComponent; i++)
                featureGradientValues.AddWithRowSliceValuesOf(temp, i*featureSize, featureSize);
        }

        virtual void UpdateFunctionMBSize() override
        {
            Base::UpdateFunctionMBSize();

            size_t numCols = Inputs(3)->GetNumCols();
            size_t numComponents = Inputs(0)->GetNumRows();
            size_t colsPrior = Inputs(0)->GetNumCols();
            size_t featureSize = Inputs(3)->GetNumRows();

            m_prior->Resize(numComponents, colsPrior);
            m_stddev->Resize(numComponents, colsPrior);
            m_normedDeviation->Resize(numComponents, numCols);
            m_normedDeviationVectors->Resize(numComponents * featureSize, numCols);
            m_posterior->Resize(numComponents, numCols);
        }

        //input0=unnormedPrior, input1=mean, input2=logstddev, input3=feature
        void EvaluateThisNodeMap()    // TODO: This is a stop-gap; in most cases, we should just be able to delete this (but need to review one by one)
        {
            // all internal matrices will be automatically resized since all of them are assigned to a value so no resize is needed here.
            EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), Inputs(3)->FunctionValues(),
                *m_prior, *m_stddev, *m_normedDeviationVectors, *m_normedDeviation, *m_posterior, *m_temp);
        }

        //input0=unnormedPrior, input1=mean, input2=logstddev, input3=feature
        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            //if (frameRange.IsAllFrames()) { EvaluateThisNodeMap(); return; }
            size_t colsPrior = Inputs(0)->GetNumCols();
            size_t numSamples = Inputs(3)->GetNumCols();

            //get the right slice 
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceFeature = Inputs(3)->ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceNormedDeviation = DataSlice(*m_normedDeviation, frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceNormedDeviationVectors = DataSlice(*m_normedDeviationVectors, frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> slicePosterior = DataSlice(*m_posterior, frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            if (colsPrior == 1)
            {
                EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), sliceFeature,
                    *m_prior, *m_stddev, sliceNormedDeviationVectors, sliceNormedDeviation, slicePosterior, *m_temp);
            }
            else if (colsPrior == numSamples)
            {
                Matrix<ElemType> sliceUnnormedPrior = Inputs(0)->ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
                Matrix<ElemType> sliceMean = Inputs(1)->ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
                Matrix<ElemType> sliceLogstddev = Inputs(2)->ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

                Matrix<ElemType> slicePrior = DataSlice(*m_prior, frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
                Matrix<ElemType> sliceStddev = DataSlice(*m_stddev, frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

                EvaluateThisNodeS(sliceOutputValue, sliceUnnormedPrior, sliceMean, sliceLogstddev, sliceFeature,
                    slicePrior, sliceStddev, sliceNormedDeviationVectors, sliceNormedDeviation, slicePosterior, *m_temp);
            }
            else  //should not reach the code since validation should fail already
                RuntimeError("GMMLogLikelihoodNode: UnnormedPrior should either have same number of columns as the features or have only one column.");
        }

        //input0=unnormedPrior, input1=mean, input2=logstddev, input3=feature
        //If we want to speed up we need to replace following code with a several specialized GPU functions
        /*TODO: merge with call site*/void EvaluateThisNodeS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& unnormedPrior, const Matrix<ElemType>& mean, Matrix<ElemType>& logstddev,
            const Matrix<ElemType>& feature, Matrix<ElemType>& prior, Matrix<ElemType>& stddev, Matrix<ElemType>& normedDeviationVectors,
            Matrix<ElemType>& normedDeviation, Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
        {
            int numComponent = unnormedPrior.GetNumRows();
            size_t numSamples = feature.GetNumCols();
            size_t featureDim = feature.GetNumRows();

            //compute prior which is softmax of unnormedPrior
            prior.AssignLogSoftmaxOf(unnormedPrior, true);  //log prior

            prior.InplaceExp();

            //compute stddev
            stddev.AssignExpOf(logstddev);

#if DUMPOUTPUT
            unnormedPrior.Print("unnormedPrior", 0, min(5, unnormedPrior.GetNumRows() - 1), 0, min(10, unnormedPrior.GetNumCols() - 1));
            mean.Print("mean", 0, min(5, mean.GetNumRows() - 1), 0, min(10, mean.GetNumCols() - 1));
            logstddev.Print("logstddev", 0, min(5, logstddev.GetNumRows() - 1), 0, min(10, logstddev.GetNumCols() - 1));

            prior.Print("prior", 0, min(5, prior.GetNumRows() - 1), 0, min(10, prior.GetNumCols() - 1));
            stddev.Print("stddev", 0, min(5, stddev.GetNumRows() - 1), 0, min(10, stddev.GetNumCols() - 1));
#endif

            //compute normedDeviation <-- ||x-u_c||^2/(stddev^2)
            normedDeviationVectors.AssignRepeatOf(feature, numComponent, 1);
            normedDeviationVectors -= mean; //each column of the mean has multiple mean components
            normedDeviationVectors.Reshape(featureDim, numSamples* numComponent);  //now each column is feature-mean_i

            normedDeviation.AssignVectorNorm2Of(normedDeviationVectors, true);
            normedDeviation ^= 2;
            temp.AssignRepeatOf(stddev, 1, numSamples / stddev.GetNumCols());  //stddev.GetNumCols() is either 1 or =numSamples
            temp.Reshape(1, temp.GetNumElements());  //one stddev value for each component for each sample
            temp ^= 2;
            normedDeviation.ElementDivideBy(temp);  //normedDeviation and temp have same dim (1, numSamples* numComponent)

            //compute  normedDeviationVectors <-- (x-u_c)/(stddev^2)
            normedDeviationVectors.RowElementDivideBy(temp);  //divide twice
            normedDeviationVectors.Reshape(featureDim*numComponent, numSamples);  //reshape back

            //compute per-component likelihood
            posterior.AssignProductOf(-0.5f, normedDeviation); //posterior  <-- -||x-u_c||^2/(stddev^2)/2 and in (1, numSamples* numComponent) dim
            temp.InplaceLog();
            temp *= ((ElemType)numComponent / 2.0f); //temp <-- stddev^c and in (1, numSamples* numComponent) dim
            posterior -= temp;  // posterior  <-- exp[-||x-u_c||^2/(stddev^2)/2]/(stddev^c)
            posterior -= (ElemType)(numComponent / 2.0f*log(TWO_PI)); //likelihood for each component and sample is now computed and stored in posterior
            posterior.InplaceExp(); //posterior  <-- exp(-||x-u_c||^2/(stddev^2)/2)

            normedDeviation.Reshape(numComponent, numSamples);  //reshape back
            posterior.Reshape(numComponent, numSamples);  //reshape back

            //compute posterior <-- prior_i * likelihood_i
            if (unnormedPrior.GetNumCols() == numSamples)  //each sample has different prior
                posterior.ElementMultiplyWith(prior);
            else  //all samples share the same prior
                posterior.ColumnElementMultiplyWith(prior);

            //compute GMM log-likelihood
            Matrix<ElemType>::Multiply(ConstOnes(1, numComponent, posterior.GetDeviceId()), false, posterior, false, functionValues);  //functionValues <-- total likelihood
            posterior.RowElementDivideBy(functionValues); //posterior <-- per-comp likelihood / total likelihood
            functionValues.InplaceLog(); //log likelihood

#if DUMPOUTPUT
            temp.Print("temp", 0, min(5, temp.GetNumRows() - 1), 0, min(10, temp.GetNumCols() - 1));
            normedDeviation.Print("normedDeviation", 0, min(5, normedDeviation.GetNumRows() - 1), 0, min(10, normedDeviation.GetNumCols() - 1));

            posterior.Print("posterior", 0, min(5, posterior.GetNumRows() - 1), 0, min(10, posterior.GetNumCols() - 1));
            functionValues.Print("functionValues", 0, min(5, functionValues.GetNumRows() - 1), 0, min(10, functionValues.GetNumCols() - 1));

            functionValues.Print("GMMLogLikelihoodNode");
#endif

#if NANCHECK
            functionValues.HasNan("GMMLogLikelihood");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            size_t rows[4], cols[4];
            for (int i = 0; i < 4; i++)
            {
                rows[i] = Inputs(i)->GetNumRows();
                cols[i] = Inputs(i)->GetNumCols();
            }

            if (isFinalValidationPass)
            {
                if (cols[0] != cols[1] || cols[0] != cols[2])
                    LogicError("GMMLogLikelihoodNode: UnnormedPrior (first input), mean (second input), and logStddev (third input) should have same number of columns.");

                if (cols[0] != 1 && cols[0] != cols[3])
                    LogicError("GMMLogLikelihoodNode: UnnormedPrior (first input) should either have same number of columns as the features (fourth input) or have only one column.");

                if (rows[0] != rows[2])
                    LogicError("GMMLogLikelihoodNode: UnnormedPrior (first input) should have same dimension as logStddev (third input), i.e., all dimensions in each Gaussian component share the same stddev.");

                if (rows[1] != rows[0] * rows[3])
                    LogicError("GMMLogLikelihoodNode: the number of rows in mean (second input) should equal rows(unnormedPrior(first input) * rows(feature(fourth input)).");
            }

            Resize(1, cols[3]);
            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(3, false);

            m_outputImageLayout = ImageLayout();
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<GMMLogLikelihoodNode<ElemType>>(nodeP);
                *node->m_prior = *m_prior;
                *node->m_normedDeviation = *m_normedDeviation;
                *node->m_normedDeviationVectors = *m_normedDeviationVectors;
                *node->m_stddev = *m_stddev;
                *node->m_posterior = *m_posterior;
            }
        }

        //request matrices needed to do node function value evaluation
        virtual void RequestMatricesBeforeEval(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeEval(matrixPool);
            RequestMatrixFromPool(m_prior, matrixPool);
            RequestMatrixFromPool(m_normedDeviation, matrixPool);
            RequestMatrixFromPool(m_normedDeviationVectors, matrixPool);
            RequestMatrixFromPool(m_stddev, matrixPool);
            RequestMatrixFromPool(m_posterior, matrixPool);
            RequestMatrixFromPool(m_temp, matrixPool);
        }

        //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        virtual void ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool)
        {
            Base::ReleaseMatricesAfterGradientComp(matrixPool);
            ReleaseMatrixToPool(m_prior, matrixPool);
            ReleaseMatrixToPool(m_normedDeviation, matrixPool);
            ReleaseMatrixToPool(m_normedDeviationVectors, matrixPool);
            ReleaseMatrixToPool(m_stddev, matrixPool);
            ReleaseMatrixToPool(m_posterior, matrixPool);
            ReleaseMatrixToPool(m_temp, matrixPool);
        }
    protected:
        shared_ptr<Matrix<ElemType>> m_prior;
        shared_ptr<Matrix<ElemType>>m_normedDeviation;
        shared_ptr<Matrix<ElemType>> m_normedDeviationVectors;
        shared_ptr<Matrix<ElemType>> m_stddev;
        shared_ptr<Matrix<ElemType>> m_posterior;
        shared_ptr<Matrix<ElemType>> m_temp;
    };

    template class GMMLogLikelihoodNode<float>;
    template class GMMLogLikelihoodNode<double>;

    // -----------------------------------------------------------------------
    // DropoutNode (input) -- perform drop-out
    // Output is scaled such that no post-scaling is necessary.
    // -----------------------------------------------------------------------

    template<class ElemType>
    class DropoutNode : public ComputationNode<ElemType>, public NumInputs<1>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"Dropout"; }
    public:
        DropoutNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name),
            m_dropoutRate(0)
        {
            m_randomSeed = (unsigned long)CreateUniqId();
        }

        void ComputeInputPartialMap(const size_t inputIndex)
        {
            if (inputIndex > 0)
                InvalidArgument("Dropout operation only takes one input.");
            ComputeInputPartialS(m_dropoutRate, Inputs(0)->GradientValues(), *m_maskOfDropout, GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            Matrix<ElemType> sliceMask = Matrix<ElemType>();
            if (m_dropoutRate > 0)
            {
                sliceMask = DataSlice(*m_maskOfDropout, frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            }

            ComputeInputPartialS(m_dropoutRate, sliceInput0Grad, sliceMask, sliceOutputGrad);
        }

        /*TODO: merge with call site*/void ComputeInputPartialS(const double dropoutRate, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& maskOfDropout, const Matrix<ElemType>& gradientValues)
        {
            if (dropoutRate > 0)
            {
                inputGradientValues.AddElementProductOf(gradientValues, maskOfDropout);
            }
            else
            {
                inputGradientValues += gradientValues;
            }
        }

        void EvaluateThisNodeMap()    // TODO: This is a stop-gap; in most cases, we should just be able to delete this (but need to review one by one)
        {
            EvaluateThisNodeS(m_dropoutRate, m_randomSeed, FunctionValues(), *m_maskOfDropout, Inputs(0)->FunctionValues());
        }
        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            //if (frameRange.IsAllFrames()) { EvaluateThisNodeMap(); return; }
            Matrix<ElemType> sliceInput0Value = Inputs(0)->ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputValue = Matrix <ElemType>();

            Matrix<ElemType> sliceMask = Matrix<ElemType>();
            if (m_dropoutRate > 0)
            {
                Resize(Inputs(0));
                m_maskOfDropout->Resize(Inputs(0)->GetNumRows(), Inputs(0)->GetNumCols());
                sliceMask = DataSlice(*m_maskOfDropout, frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            }

            sliceOutputValue = ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            EvaluateThisNodeS(m_dropoutRate, m_randomSeed, sliceOutputValue, sliceMask, sliceInput0Value);
        }

        /*TODO: merge with call site*/void EvaluateThisNodeS(const double dropoutRate, unsigned long& randomSeed, Matrix<ElemType>& functionValues, Matrix<ElemType>& maskOfDropout, const Matrix<ElemType>& inputFunctionValues)
        {
            if (dropoutRate > 0)
            {
                maskOfDropout.Resize(inputFunctionValues.GetNumRows(), inputFunctionValues.GetNumCols());

                maskOfDropout.SetUniformRandomMask((ElemType)dropoutRate, (ElemType)(1.0 / (1.0 - dropoutRate)), randomSeed);
                randomSeed += 1073807359;  //1073807359 is a very large prime number to avoid collision with other dropout nodes

                functionValues.AssignElementProductOf(maskOfDropout, inputFunctionValues);
#if NANCHECK
                functionValues.HasNan("DropOut");
#endif
            }
            else
            {
                // TODO: Is this tested? In the past, for dropoutrate == 0 it would just override FunctionValues() to return the input; which now breaks stuff.
                functionValues.SetValue(inputFunctionValues);
            }
        }

        //virtual const Matrix<ElemType>& FunctionValues() const override
        //{
        //    if (m_dropoutRate > 0)
        //        return *m_functionValues;
        //    else
        //        return Inputs(0)->FunctionValues();
        //}
        //
        //virtual Matrix<ElemType>& FunctionValues() override
        //{
        //    if (m_dropoutRate > 0)
        //        return *m_functionValues;
        //    else
        //        return Inputs(0)->FunctionValues();
        //}

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            ValidateUnaryMap(isFinalValidationPass);
            m_maskOfDropout->Resize(Inputs(0)->GetNumRows(), Inputs(0)->GetNumCols());
        }

        void SetDropoutRate(const double val)
        {
            if (val < 0 || val >= 1)
                LogicError("DropoutRate must be >= 0 and < 1.");
            m_dropoutRate = val;
        }

        void SetRandomSeed(const unsigned long val)
        {
            m_randomSeed = (unsigned long)val;
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<DropoutNode<ElemType>>(nodeP);
                node->m_dropoutRate = m_dropoutRate;
                node->m_randomSeed = m_randomSeed;
                node->m_maskOfDropout = m_maskOfDropout;
            }
        }
        //request matrices needed to do node function value evaluation
        virtual void RequestMatricesBeforeEval(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeEval(matrixPool);
            RequestMatrixFromPool(m_maskOfDropout, matrixPool);
        }

        //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        virtual void ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool)
        {
            Base::ReleaseMatricesAfterGradientComp(matrixPool);
            ReleaseMatrixToPool(m_maskOfDropout, matrixPool);
        }
    private:
        double m_dropoutRate;
        unsigned long m_randomSeed;

        shared_ptr<Matrix<ElemType>> m_maskOfDropout;
    };

    template class DropoutNode<float>;
    template class DropoutNode<double>;

}}}