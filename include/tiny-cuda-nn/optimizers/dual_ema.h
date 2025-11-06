#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/optimizer.h>

#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

namespace tcnn {

template <typename T>
__global__ void dema_step_full_precision(
	const uint32_t n_elements,
	const float dema_encoding_decay,
	const float dema_network_decay,
	const float dema_encoding_debias_old,
	const float dema_encoding_debias_new,
	const float dema_network_debias_old,
	const float dema_network_debias_new,
	const uint32_t n_network_params,
	const T* __restrict__ weights,
	T* __restrict__ weights_dema,
	float* __restrict__ tmp
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) {
		return;
	}

	float filtered_val = i < n_network_params ?
		(((float)tmp[i] * dema_network_decay * dema_network_debias_old + (float)weights[i] * (1 - dema_network_decay)) *
	     dema_network_debias_new) :
		(((float)tmp[i] * dema_encoding_decay * dema_encoding_debias_old + (float)weights[i] * (1 - dema_encoding_decay)) *
	     dema_encoding_debias_new);
	tmp[i] = filtered_val;
	weights_dema[i] = (T)filtered_val;
}

template <typename T>
__global__ void dema_step_half_precision(
	const uint32_t n_elements,
	const float dema_encoding_decay,
	const float dema_network_decay,
	const float dema_encoding_debias_old,
	const float dema_encoding_debias_new,
	const float dema_network_debias_old,
	const float dema_network_debias_new,
	const uint32_t n_network_params,
	const T* __restrict__ weights,
	T* __restrict__ weights_dema
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) {
		return;
	}

	float filtered_val = i < n_network_params ?
		(((float)weights_dema[i] * dema_network_decay * dema_network_debias_old + (float)weights[i] * (1 - dema_network_decay)) *
	     dema_network_debias_new) :
		(((float)weights_dema[i] * dema_encoding_decay * dema_encoding_debias_old + (float)weights[i] * (1 - dema_encoding_decay)) *
	     dema_encoding_debias_new);
	weights_dema[i] = (T)filtered_val;
}

template <typename T> class DualEmaOptimizer : public Optimizer<T> {
public:
	DualEmaOptimizer(const json& params) {
		m_nested.reset(create_optimizer<T>(params.value("nested", json::object())));
		update_hyperparams(params);
	}

	void allocate(uint32_t n_weights, const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
		m_nested->allocate(n_weights, layer_sizes);

		if (n_weights <= m_weights_dema.size()) {
			return;
		}

		m_weights_dema.resize(n_weights);
		m_weights_dema.memset(0);

		if (m_full_precision) {
			m_tmp.resize(n_weights);
			m_tmp.memset(0);
		}
	}

	void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
		m_nested->step(stream, loss_scale, weights_full_precision, weights, gradients);
	}

	void step_ema(cudaStream_t stream, T* weights, int step_offset=0) {
		uint32_t current_step = m_nested->step() + step_offset;

		float dema_encoding_debias_old = 1 - (float)std::pow(m_dema_encoding_decay, current_step - 1);
		float dema_encoding_debias_new = 1.0f / (1 - (float)std::pow(m_dema_encoding_decay, current_step));

		float dema_network_debias_old = 1 - (float)std::pow(m_dema_network_decay, current_step - 1);
		float dema_network_debias_new = 1.0f / (1 - (float)std::pow(m_dema_network_decay, current_step));

		T* nested_custom_weights = m_nested->custom_weights();

		if (nested_custom_weights) {
			weights = nested_custom_weights;
		}

		if (m_full_precision) {
			linear_kernel(
				dema_step_full_precision<T>,
				0,
				stream,
				n_weights(),
				m_dema_encoding_decay,
				m_dema_network_decay,
				dema_encoding_debias_old,
				dema_encoding_debias_new,
				dema_network_debias_old,
				dema_network_debias_new,
				m_n_network_params,
				weights,
				m_weights_dema.data(),
				m_tmp.data()
			);
		} else {
			linear_kernel(
				dema_step_half_precision<T>,
				0,
				stream,
				n_weights(),
				m_dema_encoding_decay,
				m_dema_network_decay,
				dema_encoding_debias_old,
				dema_encoding_debias_new,
				dema_network_debias_old,
				dema_network_debias_new,
				m_n_network_params,
				weights,
				m_weights_dema.data()
			);
		}
	}


	float learning_rate() const override { return m_nested->learning_rate(); }

	void set_learning_rate(float val) override { m_nested->set_learning_rate(val); }

	uint32_t step() const override { return m_nested->step(); }

	uint32_t n_weights() const override { return m_nested->n_weights(); }

	T* custom_weights() const override { return m_weights_dema.data(); }

	size_t n_nested() const override { return 1; }

	void set_n_network_params(uint32_t n_network_params) { m_n_network_params = n_network_params; }

	const std::shared_ptr<Optimizer<T>>& nested(size_t idx) const override {
		CHECK_THROW(idx == 0);
		return m_nested;
	}

	void update_hyperparams(const json& params) override {
		if (params.contains("encoding_decay")) {
			m_dema_encoding_decay = params["encoding_decay"];
		}

		if (params.contains("network_decay")) {
			m_dema_network_decay = params["network_decay"];
		}

		if (params.contains("full_precision")) {
			m_full_precision = params["full_precision"];
		}

		if (params.contains("nested")) {
			m_nested->update_hyperparams(params["nested"]);
		}
	}

	json hyperparams() const override {
		return {
			{"otype",          "DualEMA"              },
			{"nested",         m_nested->hyperparams()},
			{"encoding_decay", m_dema_encoding_decay  },
			{"network_decay",  m_dema_network_decay   },
			{"full_precision", m_full_precision       },
		};
	}

	json serialize() const override {
		json data;
		data["nested"] = m_nested->serialize();
		data["weights_dema_binary"] = m_weights_dema;
		return data;
	}

	void deserialize(const json& data) override {
		m_weights_dema = data["weights_dema_binary"];

		if (m_full_precision) {
			m_tmp.resize(m_weights_dema.size());
			linear_kernel(cast_from<T>, 0, nullptr, m_weights_dema.size(), m_weights_dema.data(), m_tmp.data());
		}

		m_nested->deserialize(data["nested"]);
	}

private:
	uint32_t m_n_network_params = 0u;

	float m_dema_encoding_decay = 0.0f;
	float m_dema_network_decay = 0.99f;
	bool m_full_precision = false;
	std::shared_ptr<Optimizer<T>> m_nested;

	GPUMemory<T> m_weights_dema;
	GPUMemory<float> m_tmp;
};

} // namespace tcnn
