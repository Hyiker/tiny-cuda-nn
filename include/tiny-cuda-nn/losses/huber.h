/** @file   huber.h
 *  @brief  Implementation of the Huber loss and its gradient
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/loss.h>

namespace tcnn {

template <typename T>
__global__ void huber_loss(
	const uint32_t n_elements,
	const uint32_t stride,
	const uint32_t dims,
	const float loss_scale,
	const float delta,
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
	const float pdf = data_pdf ? data_pdf[target_idx] : 1.0f;
	const float difference = prediction - targets[target_idx];
	const float abs_difference = fabsf(difference);

	float value_i;
	float gradient;

	if (abs_difference <= delta) {
		value_i = 0.5f * difference * difference;
		gradient = difference;
	} else {
		value_i = delta * abs_difference - 0.5f * delta * delta;
		gradient = delta * copysignf(1.0f, difference);
	}

	values[i] = value_i / pdf / n_total;
	gradients[i] = (T)(loss_scale * gradient / pdf / n_total);
}

template <typename T>
class HuberLoss : public Loss<T> {
private:
	float m_delta = 1.0f;

public:
	HuberLoss(const json& params = {}) {
		update_hyperparams(params);
	}

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

		linear_kernel(huber_loss<T>, 0, stream,
			prediction.n_elements(),
			stride,
			dims,
			loss_scale,
			m_delta,
			prediction.data(),
			target.data(),
			values.data(),
			gradients.data(),
			data_pdf ? data_pdf->data() : nullptr
		);
	}

	void update_hyperparams(const json& params) override {
		if (params.contains("delta")) {
			m_delta = params.value("delta", 1.0f);
		}
	}

	json hyperparams() const override {
		return {
			{"otype", "Huber"},
			{"delta", m_delta},
		};
	}

	std::string generate_device_function(const std::string& name, uint32_t n_dims) const override {
		return this->generate_device_function_from_body(name, n_dims, dfmt(1, R"(
				auto diff = vec<{N_DIMS}>(prediction) - target;
				auto scale = (1.0f / (float)n_elements) / pdf;
				const float delta = {DELTA};

				if (value) {{
					vec<{N_DIMS}> val;
					for (int i = 0; i < {N_DIMS}; ++i) {{
						auto abs_diff_i = abs(diff[i]);
						if (abs_diff_i <= {DELTA}) {{
							val[i] = 0.5f * diff[i] * diff[i];
						}} else {{
							val[i] = delta * abs_diff_i - 0.5f * delta * delta;
						}}
					}}
					*value = val * scale;
				}}

				vec<{N_DIMS}> grad;
				for (int i = 0; i < {N_DIMS}; ++i) {{
					grad[i] = clamp(diff[i], -delta, delta);
				}}

				return loss_scale * grad * scale;
			)",
			"N_DIMS"_a = n_dims,
			"DELTA"_a = m_delta
		));
	}
};

}
