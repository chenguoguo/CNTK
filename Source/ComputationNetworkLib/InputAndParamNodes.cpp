//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Basics.h"
#include "InputAndParamNodes.h"
#include "File.h"        // for LoadMatrixFromTextFile()
#include "TensorShape.h" // for SmallVector<>

#include <string>
#include <iostream>

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// LearnableParameter (/*no input*/)
// represents weight matrices and biases
// TODO: add -Node to the class name
// -----------------------------------------------------------------------

// BUGBUG: If called after random init, this will reset to 0.
// TODO: Need to remember the init parameters, and do it here.
template <class ElemType>
void LearnableParameter<ElemType>::InitShape(const TensorShape& shape)
{
    SetDims(shape, false);
    if (!m_isSparse)
    {
        UpdateFunctionValuesSize(); // this allocates the matrix
        Value().SetValue(0); // TODO: invalidate instead
    }
}

// constructor from config
template <class ElemType>
LearnableParameter<ElemType>::LearnableParameter(const ScriptableObjects::IConfigRecordPtr configp) :
    LearnableParameter(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"shape"), configp->Get(L"isSparse"), configp->Get(L"isSparse"))
{
    // TODO: Change dimensions to take a generic tensor instead. That will be a (minor) breaking change that will require fix-ups when converting from NDL to BrainScript.
    AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    // parameters[rows, [cols=1]] plus other optional parameters (learningRateMultiplier=[1|0|float], init=[uniform|gaussian|fixedvalue], initValueScale=[1|float], value=[0|float])
    if (configp->Exists(L"learningRateMultiplier"))
        SetLearningRateMultiplier(configp->Get(L"learningRateMultiplier"));
    else if (configp->Exists(L"needsGradient") || configp->Exists(L"needGradient") || configp->Exists(L"computeGradient"))
        InvalidArgument("Deprecated parameter names needsGradient|needGradient|computeGradient are not supported in BrainScript. Use learningRateMultiplier instead.");

    wstring initString = configp->Get(L"init");
    if (initString == L"fixedValue")
        Value().SetValue((ElemType) configp->Get(L"value"));
    else if (initString == L"uniform" || initString == L"gaussian")
    {
        // TODO: add these options also to old NDL
        static unsigned long randomSeed = 1;
        int forcedRandomSeed = configp->Get(L"randomSeed"); // forcing a specific random seed is useful for testing to get repeatable initialization independent of evaluation order
        InitRandom((initString == L"uniform"), forcedRandomSeed < 0 ? randomSeed++ : (unsigned long) forcedRandomSeed, configp->Get(L"initValueScale"), configp->Get(L"initOnCPUOnly"));
    }
    else if (initString == L"fromFile")
    {
        wstring initFromFilePath = configp->Get(L"initFromFilePath");
        if (initFromFilePath.empty())
            RuntimeError("initFromFilePath parameter must be provided when using \"fromFile\" initialization method");
        InitFromFile(initFromFilePath);
    }
    else if (initString == L"fromFst")
    {
        wstring fstFilePath = configp->Get(L"fstFilePath");
        if (fstFilePath.empty())
            RuntimeError("fstFilePath parameter must be provided when using \"fromFst\" initialization method");
        wstring smapFilePath = configp->Get(L"smapFilePath");
        if (smapFilePath.empty())
            RuntimeError("smapFilePath parameter must be provided when using \"fromFst\" initialization method");
        InitFromFst(fstFilePath, smapFilePath);
    }
    else if (initString == L"fromSmap")
    {
        wstring fstFilePath = configp->Get(L"fstFilePath");
        if (fstFilePath.empty())
            RuntimeError("fstFilePath parameter must be provided when using \"fromFst\" initialization method");
        wstring smapFilePath = configp->Get(L"smapFilePath");
        if (smapFilePath.empty())
            RuntimeError("smapFilePath parameter must be provided when using \"fromFst\" initialization method");
        InitFromSmap(fstFilePath, smapFilePath);
    }
    else if (initString == L"fromLiteral")
    {
        wstring initFromLiteral = configp->Get(L"initFromLiteral");
        if (initFromLiteral.empty())
            RuntimeError("initFromLiteral parameter must be provided when using \"fromLiteral\" initialization method");
        size_t numRows, numCols;
        auto array = File::LoadMatrixFromStringLiteral<ElemType>(msra::strfun::utf8(initFromLiteral), numRows, numCols);
        InitFromArray(array, numRows, numCols);
    }
    else
        RuntimeError("init must be one of the values of [ uniform | gaussian | fixedValue | fromFile ]");
}

// initialize with random numbers
// if 'initOnCPUOnly' then always init on CPU, making initialization consistent across both (for testing)
template <class ElemType>
void LearnableParameter<ElemType>::InitRandom(const bool uniformInit,
                                                const unsigned long randomSeed,
                                                const ElemType initValueScale,
                                                bool initOnCPUOnly)
{
    // fprintf(stderr, "%d x %d: %d  %ls\n", (int)GetNumRows(), (int)GetNumCols(), (int)randomSeed, NodeName().c_str());

    // the random seed offset is set via the "randomSeedOffset" parameter in config
    if (initOnCPUOnly)
        Value().TransferToDeviceIfNotThere(CPUDEVICE, true);
#if 1   // this more complex version is needed to repro test cases generated with an older version
    auto& value = GetSampleLayout().GetRank() > 2 ? Value() : ValueAsMatrix();
#else
    auto& value = Value();
#endif
    if (uniformInit)
    {
        // TODO: move these hidden extra factors out from here and into NDL, and make them visible in BS
        ElemType randRange = 0.05f * initValueScale;
        value.SetUniformRandomValue(-randRange, randRange, randomSeed);
    }
    else
    {
        size_t inputSize = value.GetNumCols();
        ElemType randInitstd = 0.2f * initValueScale / sqrt(ElemType(inputSize));
        value.SetGaussianRandomValue(0, randInitstd, randomSeed);
    }
    if (initOnCPUOnly)
        Value().TransferToDeviceIfNotThere(m_deviceId, true);
}

// initialize by reading a matrix from a text file
template <class ElemType>
void LearnableParameter<ElemType>::InitFromFile(const wstring& initFromFilePath)
{
    size_t numRows, numCols;
    auto array = File::LoadMatrixFromTextFile<ElemType>(initFromFilePath, numRows, numCols);
    InitFromArray(array, numRows, numCols);
}

// initialize by reading a matrix from a FST file
template <class ElemType>
void LearnableParameter<ElemType>::InitFromFst(const wstring& fstFilePath, const wstring& smapFilePath)
{
    map<string, int> idx4senone;
    Read_senone_map(smapFilePath.c_str(), idx4senone);

    int nstates, transM_nz, *transM_row, *transM_col;
    ElemType *transM_val;

    // sparse matrix for the mapping from state to senone
    int nsenones, smap_nz, *smap_row, *smap_col;
    ElemType *smap_val;

    nsenones = (int)idx4senone.size();
    auto input = LoadTfstFile(fstFilePath.c_str(), idx4senone);
    Graph2matrix(input, transM_val, transM_row, transM_col, nstates, transM_nz, smap_val, smap_row, smap_col, smap_nz, nsenones, NULL);
    
    //Value().SwitchToMatrixType(SPARSE, matrixFormatSparseCSC, false);
    //Value().SetMatrixFromCSCFormat(transM_col, transM_row, transM_val, transM_nz, nRows, nCols);
    Value().SwitchToMatrixType(SPARSE, matrixFormatSparseCSR, false);
    Value().SetMatrixFromCSRFormat(transM_row, transM_col, transM_val, transM_nz, nstates, nstates);
}

// initialize by reading a matrix from a Smap file
template <class ElemType>
void LearnableParameter<ElemType>::InitFromSmap(const wstring& fstFilePath, const wstring& smapFilePath)
{
    map<string, int> idx4senone;
    Read_senone_map(smapFilePath.c_str(), idx4senone);

    int nstates, transM_nz, *transM_row, *transM_col;
    ElemType *transM_val;

    // sparse matrix for the mapping from state to senone
    int nsenones, smap_nz, *smap_row, *smap_col;
    ElemType *smap_val;

    nsenones = (int)idx4senone.size();

    auto input = LoadTfstFile(fstFilePath.c_str(), idx4senone);
    Graph2matrix(input, transM_val, transM_row, transM_col, nstates, transM_nz, smap_val, smap_row, smap_col, smap_nz, nsenones, NULL);

    Value().SwitchToMatrixType(SPARSE, matrixFormatSparseCSR, false);
    Value().SetMatrixFromCSRFormat(smap_row, smap_col, smap_val, smap_nz, nsenones, nstates);
}

// initialize by reading a matrix from a text file
template <class ElemType>
void LearnableParameter<ElemType>::InitFromArray(const std::vector<ElemType>& array, size_t numRows, size_t numCols)
{
    // infer tensor dimensions from input file if not set
    // Note: The mapping of dimensions of the input matrix to tensor dimensions are somewhat confusing.
    //       The file contains a 2D matrix (one row per text line) that is saved into our column-major representation.
    //       That representation is then reshaped into a column-major tensor.
    if (GetSampleLayout().GetNumElements() == 0)    // at least one dimension is 0
    {
        auto dims = GetSampleLayout().GetDims();
        // infer rank
        if (dims.size() == 0)
            dims.push_back(0);
        if (dims.size() == 1 && numCols != 1)
            dims.push_back(0);
        // infer #rows
        if (dims[0] == 0)           // infer row dimension as input matrix row dimension
            dims[0] = numRows;      // (if already set, then mismatch will be caught in VerifyDataSize() below)
        // infer #cols: product of all dimensions but the first must match matrix #cols; if there is a single 0 position, we infer it
        size_t zeroDim = 0;         // 0 means not found
        size_t prod = 1;
        for (size_t k = 1; k < dims.size(); k++)
        {
            auto dim = dims[k];
            if (dim != 0)
                prod *= dim;
            else if (zeroDim == 0)
                zeroDim = k;
            else
                InvalidArgument("%ls %ls operation's specified shape [%s] cannot be inferred: Too many unknown dimensions.", NodeName().c_str(), OperationName().c_str(), string(GetSampleLayout()).c_str());
        }
        if (zeroDim != 0)   // we found a zero
        {
            dims[zeroDim] = numCols / prod;
            if (prod * dims[zeroDim] != numCols)
                InvalidArgument("%ls %ls operation's specified shape [%s] cannot be inferred: Tensor shape cannot hold a [%d x %d] matrix.", NodeName().c_str(), OperationName().c_str(), string(GetSampleLayout()).c_str(), (int)numRows, (int)numCols);
        }
        SetDims(TensorShape(dims), false);
    }

    // BUGBUG: We should allow to read an arbitrary tensor from a single-column file.
    //         Currently, this would cause a matrix/tensor dimension mismatch. --TODO: Is this comment up-to-date?
    Value().SetValue(numRows, numCols, m_deviceId, const_cast<ElemType*>(array.data()), matrixFlagNormal);
    // TODO: Get rid of that const_cast, as soon as after Ryan's Matrix-lib refactoring separated out SetValue() from external vs. from deep copy
    VerifyDataSize(Value());      // sanity check
}

template <class ElemType>
void LearnableParameter<ElemType>::Save(File& fstream) const /*override*/
{
    Base::Save(fstream);
    fstream << m_learningRateMultiplier;
    m_sampleLayout.Save(fstream);
    fstream << Value();
}

template <class ElemType>
void LearnableParameter<ElemType>::Load(File& fstream, size_t modelVersion) /*override*/
{
    Base::Load(fstream, modelVersion);

    TensorShape sampleLayout;

    if (modelVersion >= CNTK_MODEL_VERSION_3)
    {
        fstream >> m_learningRateMultiplier;
        sampleLayout.Load(fstream);
    }
    else // legacy format(s)
    {
        bool parameterUpdateRequired;
        fstream >> parameterUpdateRequired;
        SetLearningRateMultiplier((float)parameterUpdateRequired);

        size_t rows, cols;
        fstream >> rows >> cols;
        if (rows != 0) // legacy file format
            sampleLayout = TensorShape(rows, cols);
        else
        {
            sampleLayout.Load(fstream, /*acceptLegacyFormat=*/true);
            if (cols > 1) // in some legacy format, last tensor dimension was split off as an explicit column dimension
                sampleLayout.AppendInPlace(sampleLayout.GetRank(), cols);
        }
    }

    LoadValue(fstream);
    SetDims(sampleLayout, false); // note: call this after LoadValue() since LoadValue() overwrites m_sampleLayout
    VerifyDataSize(Value());      // sanity check
}

// computation functions don't do anything for parameter nodes
template <class ElemType>
/*virtual*/ void LearnableParameter<ElemType>::UpdateFunctionMBSize() /*override*/
{
}

template <class ElemType>
/*virtual*/ void LearnableParameter<ElemType>::ForwardProp(const FrameRange&) /*override*/
{
}

template <class ElemType>
/*virtual*/ void LearnableParameter<ElemType>::BackpropTo(const size_t /*inputIndex*/, const FrameRange&) /*override*/
{
    LogicError("%ls %ls operation is a leaf node. BackpropTo() should never be called.", NodeName().c_str(), OperationName().c_str());
}

template <class ElemType>
/*virtual*/ void LearnableParameter<ElemType>::Validate(bool isFinalValidationPass) /*override*/
{
    Base::Validate(isFinalValidationPass);
    m_pMBLayout = nullptr; // this node does not hold mini-batch data
}

// called from ComputationNode::ValidateInferInputDimsFrom()
// In case of an error, this function just backs out without updating.
// The caller must verify the dimensions.
// This is a bit weird since it is called after this node has been Validated once.
// BUGBUG: This will clear out any random initialization to 0. So currently this is not usable for most cases.
template <class ElemType>
void LearnableParameter<ElemType>::InferInputDimsFrom(const TensorShape& otherShape)
{
    const auto& thisShape = GetSampleLayout();

    // see where we stand with our shape
    bool hasMissingDims = thisShape.GetRank() == 0 || thisShape.GetNumElements() == 0;
    if (!hasMissingDims) // all there--nothing to infer
        return;
    
    // infer at least one dimension
    if (otherShape.GetRank() == 0 || otherShape.GetNumElements() == 0)
        return; // LogicError("ValidateInferInputDimsFrom: Inferred dimensions must not be empty.");
    
    // if no dimensions have been set at all, copy otherShape
    // Don't verify dimensions in this case, because the node may have explicitly been defined as a vector of 0 elements.
    bool hasAnyDim = false;
    for (auto dim : thisShape.GetDims())
        hasAnyDim |= dim != 0;
    if (!hasAnyDim)          // just use it straight
        InitShape(otherShape);
    else if (hasMissingDims) // we got a pre-existing shape: If it has zeroes, we fill them in from otherShape
    {
        if (thisShape.GetRank() != 0 && thisShape.GetRank() != otherShape.GetRank())
            return; // LogicError("ValidateInferInputDimsFrom: Inferred dimensions must match in rank.");
        SmallVector<size_t> newDims = thisShape.GetDims();
        for (size_t i = 0; i < thisShape.GetRank(); i++)
            if (newDims[i] == 0)
                newDims[i] = otherShape[i];
        InitShape(TensorShape(newDims));
    }
    fprintf(stderr, "%ls %ls operation: Tensor shape was inferred as [%s].\n", NodeName().c_str(), OperationName().c_str(), string(GetSampleLayout()).c_str());
}

template <class ElemType>
/*virtual*/ void LearnableParameter<ElemType>::DumpNodeInfo(const bool printValues, const bool printMetadata, File& fstream) const /*override*/
{
    if (printMetadata)
    {
        Base::DumpNodeInfo(printValues, printMetadata, fstream);

        char str[4096];
        sprintf(str, "[%lu,%lu]  ", GetAsMatrixNumRows(), GetAsMatrixNumCols());
        fstream << string(str);
        sprintf(str, "learningRateMultiplier=%f  NeedsGradient=%s", m_learningRateMultiplier, m_learningRateMultiplier>0 ? "true" : "false"); // TODO: update NDL to accept a better matching name as well
        fstream << string(str);
    }

    PrintNodeValuesToFile(printValues, printMetadata, fstream);
}

template class LearnableParameter<float>;
template class LearnableParameter<double>;

template <class ElemType>
void LearnableParameter<ElemType>::Graph2matrix(const vector<DataArc> input, ElemType *&_A, int *&_IA, int *&_JA, int &N, int &nz, ElemType *&smap_A, int *&smap_IA, int *& smap_JA, int &smap_nz, int numSenone, const wchar_t *transfile)
{

    map<string, pair<ElemType, ElemType>> translp4;
    if (transfile) {
        FILE *tin = _wfopen(transfile, L"r");
        if (!tin) {
            cout << "unable to open " << transfile << endl;
            exit(1);
        }
        const int slen = 1000;
        char buff[slen];
        float slp, flp;
        while (fscanf(tin, "%s%f%f", buff, &slp, &flp) == 3) {
            assert(slp <= 0 && flp <= 0); // log-probs are negative
            if (translp4.count(buff)) {
                cout << "Error: transition probabilities defined multiple times for " << buff << endl;
                exit(1);
            }
            translp4[buff] = pair<ElemType, ElemType>(exp(-slp), exp(-flp));
        }
        fclose(tin);
    }

    map<int, vector<int> > states4senone;

    int arcstate = 1;  // state 0 will be the start state
    int start_state = -1;
    vector<vector<arc>> arcs;
    map<int, ElemType> cost4final_state;
    int curr_state = 0, fs = 0;
    arc ca;
    vector<arc> carcs;
    smap_nz = 0;
    for (auto dataArc : input)
    {
        if (dataArc.Senone < 0)
        {
            int state = dataArc.From;
            fs = state;
            assert(cost4final_state.count(state) == 0);
            cost4final_state[state] = dataArc.Cost;
        }
        else
        {
            fs = dataArc.From;
            ca.source = dataArc.From;
            ca.destination = dataArc.To;
            ca.lm_cost = dataArc.Cost;
            ca.logsp = ca.logfp = 1.0;
            ca.statenum = arcstate++;
            states4senone[dataArc.Senone].push_back(ca.statenum);
            smap_nz++;
        }
        if (start_state < 0) start_state = dataArc.From;
        if (fs != curr_state) {
            assert(fs == curr_state + 1);  // att format ordered this way
            arcs.push_back(carcs);  // store the arcs for the previous state
            carcs.clear();
            curr_state = fs;
        }
        if (dataArc.Senone >= 0)
            carcs.push_back(ca);
    }

    // don't forget to store the arcs associated with the last state!
    arcs.push_back(carcs);  // store the arcs for the previous state

    assert(cost4final_state.size() != 0);
    assert(start_state == 0);

    const int finalstate = arcstate;  // make a fresh state to be the final state in the graph

    // now write the matrix

    // each arc becomes a state
    // the LM prob and forward transition probs are applied on transition out of the state
    // self-loop prob is applied for the on-diagonal transition

    vector<ElemType> A;
    vector<int> IA, JA;
    // add a notional start state
    assert(arcs[0].size());
    assert(arcs[0].front().source == 0);  // openfst should number the first state 0
    int counter = 0;
    for (size_t a = 0; a < arcs[0].size(); a++) {
        A.push_back(1.0);  // free transition out of the start state
        JA.push_back(arcs[0][a].statenum);
        counter++;
    }
    IA.push_back(0);

    // now add the rest
    for (size_t c = 0; c < arcs.size(); c++) {  // for each "from state"
        assert(arcs[c].size());
        for (size_t a = 0; a < arcs[c].size(); a++) {
            const int add = IA.back() + counter;
            IA.push_back(add);
            counter = 0;

            const arc &curr = arcs[c][a];
            const int dest = curr.destination;
            const vector<arc> &succs = arcs[dest];
            assert(succs.size());
            assert(succs[0].source == dest);  // should be 1-1 mapping of states to sets of successor arcs
            bool added_selfloop = false;
            for (size_t s = 0; s < succs.size(); s++) {
                if (s > 0)
                    assert(succs[s].statenum > succs[s - 1].statenum);
                if (succs[s].statenum > curr.statenum && !added_selfloop) {
                    A.push_back(curr.logsp);  // transition to curr.statenum
                    JA.push_back(curr.statenum);
                    added_selfloop = true;
                    counter++;
                }
                A.push_back(curr.lm_cost*curr.logfp);  // transition to succs[s].statenum
                JA.push_back(succs[s].statenum);
                counter++;
            }
            if (cost4final_state.count(dest)) {  // add transition to finalstate
                A.push_back(cost4final_state[dest]);
                JA.push_back(finalstate);
                counter++;
            }
        }
    }
    const int add = IA.back() + counter;
    IA.push_back(add);
    IA.push_back(add);  // the final state has no transitions out, and no self-transition
    assert(IA.back() == A.size());
    assert(A.size() == JA.size());
    assert(IA.size() == finalstate + 2);
    cout << "Transition matrix has " << (finalstate + 1) << " states and " << A.size() << " nozeros " << endl;

    _A = new ElemType[A.size()];
    _IA = new int[IA.size()];
    _JA = new int[JA.size()];
    copy(&A[0], &A[0] + A.size(), _A);
    copy(&IA[0], &IA[0] + IA.size(), _IA);
    copy(&JA[0], &JA[0] + JA.size(), _JA);
    N = finalstate + 1;
    nz = (int)A.size();

    assert(smap_nz == arcstate - 1);  // -1 because arcstate 0 is a dummy start state
    // map the matrix that maps from states to senones
    smap_A = new ElemType[smap_nz];
    smap_IA = new int[numSenone + 1];
    smap_IA[0] = 0;
    smap_JA = new int[smap_nz];
    int elem = 0;
    bool seen_all = true;
    for (size_t s = 0, smp = 1; s<numSenone; s++, smp++) {
        if (states4senone.find((int)s) == states4senone.end()) {  // graph build w/ small vocab - not all senones present
            seen_all = false;
            smap_IA[smp] = smap_IA[smp - 1];
            continue;
        }
        const vector<int> &states = states4senone[(int)s];
        for (size_t j = 0; j < states.size(); j++) {
            assert(j == 0 || states[j]>states[j - 1]);
            assert(states[j] > 0 && states[j] < finalstate);
            smap_A[elem] = 1.0;
            smap_JA[elem] = states[j];
            elem++;
        }
        smap_IA[smp] = smap_IA[smp - 1] + (int)states.size();
    }
    if (!seen_all) {
        cout << "Warning: not all senones present in graph" << endl;
    }
}
}}}
