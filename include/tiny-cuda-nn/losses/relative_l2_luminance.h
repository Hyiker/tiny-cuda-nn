/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   relative_l2_luminance.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Hacky implementation of the relative l2 loss based on the LUMINANCE of a six-channel prediction
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/loss.h>

namespace tcnn {

template <typename T>
__global__ void relative_l2_luminance_loss(
	const uint32_t n_elements,
	const uint32_t stride,
	const uint32_t dims,
	const float loss_scale,
	const T* __restrict__ predictions,
	const float* __restrict__ targets,
	float* __restrict__ values,
	T* __restrict__ gradients,
	const float* __restrict__ data_pdf = nullptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t intra_elem_idx = i % stride;
	const uint32_t inter_elem_idx = i / stride;
	if (intra_elem_idx >= dims) {
		values[i] = 0;
		gradients[i] = 0;
		return;
	}

	const uint32_t target_idx = inter_elem_idx * dims + intra_elem_idx;

	const uint32_t n_total = n_elements / stride * dims;

	const float prediction = (float)predictions[i];

	float r = (float)predictions[i - intra_elem_idx + 0];
	float g = (float)predictions[i - intra_elem_idx + 1];
	float b = (float)predictions[i - intra_elem_idx + 2];
	if (dims >= 6) {
		r += (float)predictions[i - intra_elem_idx + 3];
		g += (float)predictions[i - intra_elem_idx + 4];
		b += (float)predictions[i - intra_elem_idx + 5];
	}
	const float luminance = (0.299f * r + 0.587f * g + 0.114f * b);

	const float prediction_sq_plus_epsilon = luminance * luminance + 0.01f;

	const float pdf = data_pdf ? data_pdf[target_idx] : 1;
	const float difference = prediction - targets[target_idx];

	values[i] = difference * difference / prediction_sq_plus_epsilon / pdf / n_total;

	float gradient = 2 * difference / prediction_sq_plus_epsilon / pdf;
	gradients[i] = (T)(loss_scale * gradient / n_total);
}

template <typename T>
class RelativeL2LuminanceLoss : public Loss<T> {
public:
	void evaluate(
		cudaStream_t stream,
		const float loss_scale,
		const GPUMatrix<T>& prediction,
		const GPUMatrix<float>& target,
		GPUMatrix<float>& values,
		GPUMatrix<T>& gradients,
		const GPUMatrix<float>* data_pdf = nullptr
	) const override {
		const uint32_t dims = target.m();
		const uint32_t stride = prediction.m();

		CHECK_THROW(prediction.n() == target.n());
		CHECK_THROW(values.m() == stride);
		CHECK_THROW(gradients.m() == stride);
		CHECK_THROW(!data_pdf || data_pdf->m() == dims);

		linear_kernel(relative_l2_luminance_loss<T>, 0, stream,
			prediction.n_elements(),
			stride,
			dims,
			loss_scale,
			prediction.data(),
			target.data(),
			values.data(),
			gradients.data(),
			data_pdf ? data_pdf->data() : nullptr
		);
	}

	void update_hyperparams(const json& params) override { }

	json hyperparams() const override {
		return {
			{"otype", "RelativeL2Luminance"},
		};
	}

	std::string generate_device_function(const std::string& name, uint32_t n_dims) const override {
		return this->generate_device_function_from_body(name, n_dims, dfmt(1, R"(
				vec<{N_DIMS}> prediction_fp = prediction;

				vec3 rgb = prediction_fp.slice<0, 3>();
				{DOUBLE_RGB}

				float luminance = dot(rgb, vec3(0.299f, 0.587f, 0.114f));

				auto diff = prediction_fp - target;
				auto scale = (1.0f / (float)n_elements) / (luminance * luminance + 1e-2f) / pdf;
				if (value) {{
					*value = diff * diff * scale;
				}}

				return (2.0f * loss_scale) * diff * scale;
			)",
			"N_DIMS"_a = n_dims,
			"DOUBLE_RGB"_a = n_dims >= 6 ? "rgb += prediction_fp.slice<3, 3>();" : ""
		));
	}
};

}
