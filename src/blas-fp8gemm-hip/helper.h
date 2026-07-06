/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <vector>
#include <functional>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>

inline void checkHipStatus(hipError_t status) {
    if (status != hipSuccess) {
        printf("hip API failed with status %d: %s\n", status, hipGetErrorString(status));
        throw std::logic_error("hip API failed");
    }
}

inline void checkHipblasStatus(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        printf("hipBLAS API failed with status %d\n", status);
        throw std::logic_error("hipBLAS API failed");
    }
}

// FP8 E4M3 encoding differs by architecture: gfx950 uses the OCP format,
// gfx94x uses the FNUZ format. Detect which one the current device needs so
// that the host-side reference encoding/decoding matches the device layout.
inline bool fp8IsOcp() {
    int device = 0;
    checkHipStatus(hipGetDevice(&device));
    hipDeviceProp_t prop;
    checkHipStatus(hipGetDeviceProperties(&prop, device));
    return std::string(prop.gcnArchName).find("gfx95") != std::string::npos;
}

// Encode a float into an 8-bit E4M3 code using the library FP8 types, which
// apply the correct rounding for each format (OCP vs FNUZ).
inline uint8_t encodeE4M3(float v, bool ocp) {
    if (ocp) { hipblaslt_f8      t(v); uint8_t b; std::memcpy(&b, (void*)&t, 1); return b; }
    else     { hipblaslt_f8_fnuz t(v); uint8_t b; std::memcpy(&b, (void*)&t, 1); return b; }
}

// Decode an 8-bit E4M3 code back to float for the given format.
inline float decodeE4M3(uint8_t b, bool ocp) {
    if (ocp) { hipblaslt_f8      t(0.0f); std::memcpy((void*)&t, &b, 1); return float(t); }
    else     { hipblaslt_f8_fnuz t(0.0f); std::memcpy((void*)&t, &b, 1); return float(t); }
}

template <typename InType, typename CType, typename OutType, typename ComputeType>
struct TestBench {
    using SampleRunner = std::function<void()>;

    TestBench(int m, int n, int k,
            ComputeType alpha = ComputeType{0.0f}, ComputeType beta = ComputeType{0.0f},
            size_t workspaceSize = 1024 * 1024 * 4, int N = 1,
            ComputeType Ascale = ComputeType{1.0f}, ComputeType Bscale = ComputeType{1.0f},
            ComputeType Cscale = ComputeType{1.0f}, ComputeType Dscale = ComputeType{1.0f}) :
        m(m), n(n), k(k), N(N), alpha(alpha), beta(beta), workspaceSize(workspaceSize), 
        Ahost(m * k * N), Bhost(n * k * N), Chost(m * n * N), Dhost(m * n * N),
        AscaleHost(Ascale), BscaleHost(Bscale), CscaleHost(Cscale), DscaleHost(Dscale) {
        checkHipblasStatus(hipblasLtCreate(&ltHandle));
        checkHipStatus(hipMalloc(reinterpret_cast<void**>(&Adev), m * k * N * sizeof(InType)));
        checkHipStatus(hipMalloc(reinterpret_cast<void**>(&Bdev), n * k * N  * sizeof(InType)));
        checkHipStatus(hipMalloc(reinterpret_cast<void**>(&Cdev), m * n * N  * sizeof(CType)));
        checkHipStatus(hipMalloc(reinterpret_cast<void**>(&Ddev), m * n * N  * sizeof(OutType)));
        checkHipStatus(hipMalloc(&workspace, workspaceSize));
        checkHipStatus(hipStreamCreate(&stream));

        // Currently only fp8 supports per-tensor scaling
        perTensorScalingEnabled = std::is_same<InType, hipblaslt_f8_fnuz>::value || std::is_same<InType, hipblaslt_bf8_fnuz>::value;

        // The host reference must encode/decode FP8 in the same format the
        // device uses (OCP on gfx950, FNUZ on gfx94x).
        fp8Ocp = fp8IsOcp();

        if (perTensorScalingEnabled) {
            checkHipStatus(hipMalloc(reinterpret_cast<void**>(&AscaleDev), sizeof(*AscaleDev)));
            checkHipStatus(hipMalloc(reinterpret_cast<void**>(&BscaleDev), sizeof(*BscaleDev)));
            checkHipStatus(hipMalloc(reinterpret_cast<void**>(&CscaleDev), sizeof(*CscaleDev)));
            checkHipStatus(hipMalloc(reinterpret_cast<void**>(&DscaleDev), sizeof(*DscaleDev)));
            checkHipStatus(hipMalloc(reinterpret_cast<void**>(&DamaxDev), sizeof(*DamaxDev)));
        }

        fillData();
    }

    ~TestBench() {
        checkHipblasStatus(hipblasLtDestroy(ltHandle));
        checkHipStatus(hipFree(Adev));
        checkHipStatus(hipFree(Bdev));
        checkHipStatus(hipFree(Cdev));
        checkHipStatus(hipFree(Ddev));
        checkHipStatus(hipFree(workspace));
        if (perTensorScalingEnabled) {
            checkHipStatus(hipFree(AscaleDev));
            checkHipStatus(hipFree(BscaleDev));
            checkHipStatus(hipFree(CscaleDev));
            checkHipStatus(hipFree(DscaleDev));
            checkHipStatus(hipFree(DamaxDev));
        }
        checkHipStatus(hipStreamDestroy(stream));
    }

    // Uniform value used for A and B so the exact GEMM result is known and
    // FP8-representable. 2^-6 is a normal value in both E4M3 formats, and with
    // all scales = 1 and beta = 0 the output D = FILL_VALUE^2 * K stays within
    // the FP8 range and is exactly representable for every benchmark shape.
    static constexpr float FILL_VALUE = 0.015625f; // 2^-6

    void fillData() {
        uint8_t aByte = encodeE4M3(FILL_VALUE, fp8Ocp);
        uint8_t *ap = reinterpret_cast<uint8_t*>(Ahost.data());
        uint8_t *bp = reinterpret_cast<uint8_t*>(Bhost.data());
        for (int i = 0; i < m * k * N; i++) ap[i] = aByte;
        for (int i = 0; i < n * k * N; i++) bp[i] = aByte;
        // beta = 0, so C is unused; keep it zeroed.
        std::memset((void*)Chost.data(), 0, Chost.size() * sizeof(CType));
    }

    // CPU reference check. With uniform inputs, alpha = 1, beta = 0 and all
    // per-tensor scales = 1, every output element equals FILL_VALUE^2 * K.
    bool verify(double relTol = 1e-2) {
        const uint8_t *dp = reinterpret_cast<const uint8_t*>(Dhost.data());
        double expected = (double)FILL_VALUE * (double)FILL_VALUE * (double)k;
        double maxErr = 0.0;
        for (int i = 0; i < m * n * N; i++)
            maxErr = std::max(maxErr, std::fabs((double)decodeE4M3(dp[i], fp8Ocp) - expected));
        bool ok = maxErr <= relTol * (std::fabs(expected) + 1.0);
        printf("Verification: expected %.6g, max abs error %.4g -> %s\n",
               expected, maxErr, ok ? "PASS" : "FAIL");
        return ok;
    }

    void copyDataToDevice() {
        checkHipStatus(hipMemcpyAsync(Adev, Ahost.data(), Ahost.size() * sizeof(Ahost[0]), hipMemcpyHostToDevice, stream));
        checkHipStatus(hipMemcpyAsync(Bdev, Bhost.data(), Bhost.size() * sizeof(Bhost[0]), hipMemcpyHostToDevice, stream));
        checkHipStatus(hipMemcpyAsync(Cdev, Chost.data(), Chost.size() * sizeof(Chost[0]), hipMemcpyHostToDevice, stream));
        if (perTensorScalingEnabled) {
            checkHipStatus(hipMemcpyAsync(AscaleDev, &AscaleHost, sizeof(AscaleHost), hipMemcpyHostToDevice));
            checkHipStatus(hipMemcpyAsync(BscaleDev, &BscaleHost, sizeof(BscaleHost), hipMemcpyHostToDevice));
            checkHipStatus(hipMemcpyAsync(CscaleDev, &CscaleHost, sizeof(CscaleHost), hipMemcpyHostToDevice));
            checkHipStatus(hipMemcpyAsync(DscaleDev, &DscaleHost, sizeof(DscaleHost), hipMemcpyHostToDevice));
            checkHipStatus(hipMemcpyAsync(DamaxDev, &DamaxHost, sizeof(DamaxHost), hipMemcpyHostToDevice));
        }
    }

    void copyDataFromDevice() {
        checkHipStatus(hipMemcpyAsync(Dhost.data(), Ddev, Dhost.size() * sizeof(Dhost[0]), hipMemcpyDeviceToHost, stream));
    }

    void streamSynchronize() {
        checkHipStatus(hipStreamSynchronize(stream));
    }

    void run(const SampleRunner& runSample) {
        copyDataToDevice();

        runSample();

        copyDataFromDevice();
        streamSynchronize();
    }

    bool perTensorScalingEnabled;
    bool fp8Ocp;
    int m, n, k, N;
    ComputeType alpha, beta;
    size_t workspaceSize;
    std::vector<InType> Ahost, Bhost;
    std::vector<CType> Chost;
    std::vector<OutType> Dhost;
    void *workspace;
    InType *Adev, *Bdev;
    CType *Cdev;
    OutType *Ddev;
    hipStream_t stream;
    hipblasLtHandle_t ltHandle;
    ComputeType AscaleHost, BscaleHost, CscaleHost, DscaleHost, DamaxHost;
    ComputeType *AscaleDev, *BscaleDev, *CscaleDev, *DscaleDev, *DamaxDev;
};

template <>
inline void TestBench<__half, __half,  __half, float>::fillData() {
    for (int i = 0; i < m * k * N; i++) Ahost[i] = __float2half_rn(i);
    for (int i = 0; i < n * k * N; i++) Bhost[i] = __float2half_rn(i);
    for (int i = 0; i < n * k * N; i++) Chost[i] = __float2half_rn(-i);
}

template <>
inline void TestBench<__half, __half, __half, hipComplex>::fillData() {
    for (int i = 0; i < m * k * N; i++) Ahost[i] = __float2half_rn(i/100.);
    for (int i = 0; i < n * k * N; i++) Bhost[i] = __float2half_rn(i/100.);
    for (int i = 0; i < n * k * N; i++) Chost[i] = __float2half_rn(-i/100.);
}
