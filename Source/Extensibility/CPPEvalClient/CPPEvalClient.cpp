//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPPEvalClient.cpp : Sample application using the evaluation interface from C++
//

#include "stdafx.h"
#include "eval.h"
#include <memory>

using namespace Microsoft::MSR::CNTK;
using namespace std;

// Used for retrieving the model appropriate for the element type (float / double)
template<typename ElemType>
using GetEvalProc = void(*)(IEvaluateModel<ElemType>**);

typedef std::pair<std::wstring, std::vector<float>*> MapEntry;
typedef std::map<std::wstring, std::vector<float>*> Layer;

/// <summary>
/// Program for demonstrating how to run model evaluations using the native evaluation interface
/// </summary>
/// <description>
/// This program is a native C++ client using the native evaluation interface
/// located in the <see cref="eval.h"/> file.
/// The CNTK evaluation dll (EvalDLL.dll), must be found through the system's path. 
/// The other requirement is that Eval.h be included
/// In order to run this program the model must already exist in the example. To create the model,
/// first run the example in <CNTK>/Examples/Image/MNIST. Once the model file 01_OneHidden is created,
/// you can run this client.
/// This program demonstrates the usage of the Evaluate method requiring the input and output layers as parameters.
int _tmain(int argc, _TCHAR* argv[])
{
    // Get the binary path (current working directory)
    argc = 0;
    std::wstring arg0(argv[0]);
    std::string app(arg0.begin(), arg0.end());
    std::string path = app.substr(0, app.rfind("\\"));

    // Load the eval library
    std::wstring dllname(L"evaldll.dll");
    auto hModule = LoadLibrary(dllname.c_str());
    if (hModule == nullptr)
    {
        const std::wstring msg(L"Cannot find library" + dllname);
        const std::string ex(msg.begin(), msg.end());
        throw new std::exception(ex.c_str());
    }

    // Get the factory method to the evaluation engine
    std::string func = "GetEvalF";
    auto procAddress = GetProcAddress(hModule, func.c_str());
    auto getEvalProc = (GetEvalProc<float>)procAddress;

    // Native model evaluation instance
    IEvaluateModel<float> *model;
    getEvalProc(&model);

    // This relative path assumes launching from CNTK's binary folder
    const std::string modelWorkingDirectory = path + "\\..\\..\\Examples\\Image\\MNIST\\Data\\";
    const std::string modelFilePath = modelWorkingDirectory + "..\\Output\\Models\\01_OneHidden";

    // Load model
    model->CreateNetwork("modelPath=\"" + modelFilePath + "\"");

    // Generate dummy input values in the appropriate structure and size
    std::vector<float> inputs;
    for (int i = 0; i < 28 * 28; i++)
    {
        inputs.push_back(static_cast<float>(i % 255));
    }

    // Allocate the output values layer
    std::vector<float> outputs;

    // Setup the maps for inputs and outputs
    Layer inputLayer;
    inputLayer.insert(MapEntry(L"features", &inputs));
    Layer outputLayer;
    outputLayer.insert(MapEntry(L"ol.z", &outputs));

    // We can call the evaluate method and get back the results (single layer)...
    model->Evaluate(inputLayer, outputLayer);

    // Output the results
    for each (auto& value in outputs)
    {
        fprintf(stderr, "%f\n", value);
    }

    return 0;
}