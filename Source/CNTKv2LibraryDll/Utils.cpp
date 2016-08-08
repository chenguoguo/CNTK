//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "File.h"

using namespace std;

namespace CNTK
{
    template <typename T>
    T* CreateDataPtr(const T& value)
    {
        return new T(value);
    }

    template <>
    NDArrayView* CreateDataPtr<NDArrayView>(const NDArrayView& value)
    {
        NDArrayView* viewPtr = new NDArrayView(value.GetDataType(), value.Shape(), DeviceDescriptor::CPUDevice());
        viewPtr->CopyFrom(value);
        return viewPtr;
    }

    template <typename T>
    void DictionaryValue::AllocateDataPtr(const T& value)
    {
        static_assert(is_same<T, NDShape>::value ||
                      is_same<T, wstring>::value ||
                      is_same<T, vector<DictionaryValue>>::value ||
                      is_same<T, Dictionary>::value ||
                      is_same<T, NDArrayView>::value,
                      "AllocateDataPtr called with invalid type");
        m_data.m_ptr = CreateDataPtr<T>(value);
    }

    template <typename T>
    void DictionaryValue::FreePtrAsType()
    {
        T* typedPtr = reinterpret_cast<T*>(m_data.m_ptr);
        delete typedPtr;

        m_data.m_ptr = nullptr;
    }

    template <typename ElementType> 
    bool CheckViewsEquivalent(NDArrayView& view1, NDArrayView& view2)
    {
        if (view1.GetDataType() != view2.GetDataType() ||
            view1.Shape() != view2.Shape())
        {
            return false;
        }

        ElementType* data1 = nullptr;
        ElementType* data2 = nullptr;
        if ((view1.Device().Type() == DeviceKind::CPU))
        {
            data1 = view1.WritableDataBuffer<ElementType>();
            data2 = view2.WritableDataBuffer<ElementType>();
        }
        else
        {
            auto temp1CpuDataView = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), view1.Shape(), DeviceDescriptor::CPUDevice());
            temp1CpuDataView->CopyFrom(view1);
            data1 = temp1CpuDataView->WritableDataBuffer<ElementType>();

            auto temp2CpuDataView = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), view2.Shape(), DeviceDescriptor::CPUDevice());
            temp2CpuDataView->CopyFrom(view2);
            data2 = temp2CpuDataView->WritableDataBuffer<ElementType>();
        }

        size_t numElements = view1.Shape().TotalSize();

        for (size_t i = 0; i < numElements; ++i)
        {
            if (data1[i] != data2[i])
            {
                return false;
            }
        }
        return true;
    }

    bool DictionaryValue::operator==(const DictionaryValue& other) const
    {
        if (this == &other)
        {
            return true;
        }

        if (m_valueType != other.m_valueType)
        {
            return false;
        }
        
        switch (m_valueType)
        {
        case DictionaryValue::Type::Bool:
            return (m_data.m_boolean == other.m_data.m_boolean);
        case DictionaryValue::Type::SizeT:
            return (m_data.m_sizeT == other.m_data.m_sizeT);
        case DictionaryValue::Type::Float:
            return (m_data.m_float == other.m_data.m_float);
        case DictionaryValue::Type::Double:
            return (m_data.m_double == other.m_data.m_double);
        case DictionaryValue::Type::String:
        {
            wstring* strPtr1 = reinterpret_cast<wstring*>(m_data.m_ptr);
            wstring* strPtr2 = reinterpret_cast<wstring*>(other.m_data.m_ptr);
            return (*strPtr1 == *strPtr2);
        }
        case DictionaryValue::Type::NDShape:
        {
            NDShape* shapePtr1 = reinterpret_cast<NDShape*>(m_data.m_ptr);
            NDShape* shapePtr2 = reinterpret_cast<NDShape*>(other.m_data.m_ptr);
            return (*shapePtr1 == *shapePtr2);
        }
        case DictionaryValue::Type::Vector:
        {   
            vector<DictionaryValue>* vectorPtr1 = reinterpret_cast<vector<DictionaryValue>*>(m_data.m_ptr);
            vector<DictionaryValue>* vectorPtr2 = reinterpret_cast<vector<DictionaryValue>*>(other.m_data.m_ptr);
            return (*vectorPtr1 == *vectorPtr2);
        }
        case DictionaryValue::Type::Dictionary:
        {
            Dictionary* dictPtr1 = reinterpret_cast<Dictionary*>(m_data.m_ptr);
            Dictionary* dictPtr2 = reinterpret_cast<Dictionary*>(other.m_data.m_ptr);
            return (*dictPtr1 == *dictPtr2);
        }
        case DictionaryValue::Type::NDArrayView:
        {
            NDArrayView* viewPtr1 = reinterpret_cast<NDArrayView*>(m_data.m_ptr);
            NDArrayView* viewPtr2 = reinterpret_cast<NDArrayView*>(other.m_data.m_ptr);

            switch (viewPtr1->GetDataType())
            {
            case DataType::Float:
                return CheckViewsEquivalent<float>(*viewPtr1, *viewPtr2);
            case DataType::Double:
                return CheckViewsEquivalent<double>(*viewPtr1, *viewPtr2);
            default:
                NOT_IMPLEMENTED;
            }
        }
        default:
            NOT_IMPLEMENTED;
        }
    }
    
    bool DictionaryValue::operator!=(const DictionaryValue& other) const
    {
        return !(*this == other);    
    }

    
    Microsoft::MSR::CNTK::File& operator<<(Microsoft::MSR::CNTK::File& stream, const NDShape& us)
    {
        auto size = us.NumAxes();
        stream << size;
        for (auto i = 0; i < size; i++)
        {
            stream << us[i];
        }
        return stream;
    }

    template <typename T>
    void Write(Microsoft::MSR::CNTK::File& stream, const NDArrayView& view)
    {
        assert(view.Device().Type() == DeviceKind::CPU);

        auto numElements = view.Shape().TotalSize();
        const T* buffer = view.DataBuffer<T>();
        for (auto i = 0; i < numElements; ++i)
        {
            stream << buffer[i];
        }
    }

    template <typename T>
    void Read(Microsoft::MSR::CNTK::File& stream, NDArrayView& view)
    {
        assert(view.Device().Type() == DeviceKind::CPU);
        
        auto numElements = view.Shape().TotalSize();
        T* buffer = view.WritableDataBuffer<T>();
        for (auto i = 0; i < numElements; ++i)
        {
            stream >> buffer[i];
        }
    }

    Microsoft::MSR::CNTK::File& operator>>(Microsoft::MSR::CNTK::File& stream, DictionaryValue& us)
    {
        size_t version;
        stream >> version;

        stream >> us.m_valueType;

        switch (us.ValueType())
        {
        case DictionaryValue::Type::Bool:
            stream >> us.m_data.m_boolean;
            break;
        case DictionaryValue::Type::SizeT:
            stream >> us.m_data.m_sizeT;
            break;
        case DictionaryValue::Type::Float:
            stream >> us.m_data.m_float;
            break;
        case DictionaryValue::Type::Double:
            stream >> us.m_data.m_double;
            break;
        case DictionaryValue::Type::String:
        {
            wstring* strPtr = new wstring();
            stream >> *strPtr;
            us.m_data.m_ptr = strPtr;
            break;
        }
        case DictionaryValue::Type::NDShape:
        {
            size_t size;
            stream >> size;
            NDShape* shapePtr = new NDShape(size);
            for (auto i = 0; i < size; i++)
            {
                stream >> shapePtr->operator[](i);
            }
            us.m_data.m_ptr = shapePtr;
            break;
        }
        case DictionaryValue::Type::Vector:
        {   
            size_t size;
            stream >> size;
            vector<DictionaryValue>* vectorPtr = new vector<DictionaryValue>(size);
            for (auto i = 0; i < size; i++)
            {
                stream >> vectorPtr->at(i);
            }
            us.m_data.m_ptr = vectorPtr;
            break;
        }
        case DictionaryValue::Type::Dictionary:
        {
            Dictionary* dictPtr = new Dictionary();
            stream >> *dictPtr;
            us.m_data.m_ptr = dictPtr;
            break;
        }
        case DictionaryValue::Type::NDArrayView:
        {
            DataType dtype;
            stream >> dtype;

            size_t size;
            stream >> size;
            NDShape shape(size);
            for (auto i = 0; i < size; i++)
            {
                stream >> shape[i];
            }

            NDArrayView* viewPtr = new NDArrayView(dtype, shape, DeviceDescriptor::CPUDevice());
            switch (dtype)
            {
            case DataType::Float:
                Read<float>(stream, *viewPtr);
                break;
            case DataType::Double:
                Read<double>(stream, *viewPtr);
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(dtype));
            }

            us.m_data.m_ptr = viewPtr;
            break;
        }
        default:
            NOT_IMPLEMENTED;
        }
        return stream;
    }

    Microsoft::MSR::CNTK::File& operator<<(Microsoft::MSR::CNTK::File& stream, const DictionaryValue& us)
    {
        stream << us.version;

        stream << us.ValueType();

        switch (us.ValueType())
        {
        case DictionaryValue::Type::Bool:
            stream << us.m_data.m_boolean;
            break;
        case DictionaryValue::Type::SizeT:
            stream << us.m_data.m_sizeT;
            break;
        case DictionaryValue::Type::Float:
            stream << us.m_data.m_float;
            break;
        case DictionaryValue::Type::Double:
            stream << us.m_data.m_double;
            break;
        case DictionaryValue::Type::String:
        {
            wstring* stringPtr = reinterpret_cast<wstring*>(us.m_data.m_ptr);
            stream << *stringPtr;
            break;
        }
        case DictionaryValue::Type::NDShape:
        {
            NDShape* shapePtr = reinterpret_cast<NDShape*>(us.m_data.m_ptr);
            stream << *shapePtr;
            break;
        }
        case DictionaryValue::Type::Vector:
        {
            vector<DictionaryValue>* vectorPtr =
                reinterpret_cast<vector<DictionaryValue>*>(us.m_data.m_ptr);
            auto size = vectorPtr->size();
            stream << size;
            for (auto i = 0; i < size; i++)
            {
                stream << vectorPtr->at(i);
            }
            break;
        }
        case DictionaryValue::Type::Dictionary:
        {
            Dictionary* dictPtr = reinterpret_cast<Dictionary*>(us.m_data.m_ptr);
            stream << *dictPtr;
            break;
        }
        case DictionaryValue::Type::NDArrayView:
        {
            NDArrayView* viewPtr = reinterpret_cast<NDArrayView*>(us.m_data.m_ptr);
            stream << viewPtr->GetDataType();
            stream << viewPtr->Shape();
            switch (viewPtr->GetDataType())
            {
            case DataType::Float:
                Write<float>(stream, *viewPtr);
                break;
            case DataType::Double:
                Write<double>(stream, *viewPtr);
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(viewPtr->GetDataType()));
            }
            break;
        }
        default:
            NOT_IMPLEMENTED;
        }
        return stream;
    }

    Dictionary::Dictionary()
        : m_dictionaryData(new unordered_map <wstring, DictionaryValue>)
    {
    }

    Dictionary::~Dictionary()
    {
    }

    Dictionary::Dictionary(const Dictionary& other)
    {
        *this = other;
    }

    Dictionary& Dictionary::operator=(const Dictionary& other)
    {
        assert(this != &other);
        m_dictionaryData.reset(new unordered_map<wstring, DictionaryValue>(*(other.m_dictionaryData)));
        return *this;
    }

    Dictionary::Dictionary(Dictionary&& other)
        : m_dictionaryData(nullptr)
    {
        *this = move(other);
    }

    Dictionary& Dictionary::operator=(Dictionary&& other)
    {
        assert(this != &other);

        m_dictionaryData = other.m_dictionaryData;
        other.m_dictionaryData = nullptr;

        return *this;
    }

    DictionaryValue& Dictionary::operator[](const wchar_t* key)
    {
        return (*m_dictionaryData)[key];
    }

    DictionaryValue Dictionary::operator[](const wchar_t* key) const
    {
        return m_dictionaryData->at(key);
    }

    bool Dictionary::Contains(const wchar_t* key) const
    {
        return (m_dictionaryData->find(key) != m_dictionaryData->end());
    }

    bool Dictionary::operator==(const Dictionary& other) const
    {
        if (this == &other)
        {
            return true;
        }

        if (m_dictionaryData->size() != other.m_dictionaryData->size())
        {
            return false;
        }
        
        for (auto kv : *m_dictionaryData)
        {
            auto result = other.m_dictionaryData->find(kv.first);
            if (result == other.m_dictionaryData->end() || kv.second != result->second)
            {
                return false;
            }
        }

        return true;
    }
    
    bool Dictionary::operator!=(const Dictionary& other) const
    {
        return !(*this == other);    
    }

    Microsoft::MSR::CNTK::File& operator<<(Microsoft::MSR::CNTK::File& stream, const Dictionary& us)
    {
        stream << us.version;
        stream << us.m_dictionaryData->size();
        for (auto kv : *(us.m_dictionaryData))
        {
            stream << kv.first;
            stream << kv.second;
        }
        return stream;
    }

    Microsoft::MSR::CNTK::File& operator>>(Microsoft::MSR::CNTK::File& stream, Dictionary& us)
    {
        size_t version;
        stream >> version;
        size_t size;
        stream >> size;
        us.m_dictionaryData->reserve(size);
        for (auto i = 0; i < size; i++)
        {
            wstring key;
            stream >> key;
            stream >> us[key];
        }
        return stream;
    }
     
    template void DictionaryValue::AllocateDataPtr<NDShape>(const NDShape& value);
    template void DictionaryValue::AllocateDataPtr<vector<DictionaryValue>>(const vector<DictionaryValue>& value);
    template void DictionaryValue::AllocateDataPtr<wstring>(const wstring& value);
    template void DictionaryValue::AllocateDataPtr<Dictionary>(const Dictionary& value);
    template void DictionaryValue::AllocateDataPtr<NDArrayView>(const NDArrayView& value);

    template void DictionaryValue::FreePtrAsType<NDShape>();
    template void DictionaryValue::FreePtrAsType<vector<DictionaryValue>>();
    template void DictionaryValue::FreePtrAsType<wstring>();
    template void DictionaryValue::FreePtrAsType<Dictionary>();
    template void DictionaryValue::FreePtrAsType<NDArrayView>();
}
