#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <set>
#include <vector>

#include "NetworkGroup.h"
#include "Logger.h"
#include "WeightLogger.h"
#include "SoundCurrentGenerator.h"

namespace Juerka::SoundRecognition::Main
{
	void run(bool, bool, bool) noexcept;
};


int main(void) noexcept
{
	bool is_run_parallel(true);
	bool is_monitor_performance(false);
	bool is_record_weights(false);


	Juerka::SoundRecognition::Main::run
	(
		is_run_parallel,
		is_monitor_performance,
		is_record_weights
	);

	return EXIT_SUCCESS;
}

namespace Juerka::SoundRecognition::Main
{
	void run
	(
		bool is_run_parallel,
		bool is_monitor_performance,
		bool is_record_weights
	) noexcept
	{
		const size_t training_layer(4);
		const size_t output_layer(2);
		const size_t network_size(training_layer + output_layer);
		const size_t training_on_time(500);
		const size_t training_off_time(500);
		const Juerka::CommonNet::step_time_t VALIDATION_TIME_DURATION = 4000;

		const double current_input_normalizer(3.5);
		//const double synaptic_current_normalizer(8);
		//const double current_input_normalizer(6);
		//const double synaptic_current_normalizer(6);
		//const double current_input_normalizer(2);
		//const double synaptic_current_normalizer(15);
		//const double current_input_normalizer(0.6);
		//const double synaptic_current_normalizer(3.5);
		//const double current_input_normalizer(0.05);
		//const double synaptic_current_normalizer(20);
		//const Juerka::CommonNet::neuron_t window(100);

		std::vector< std::array<std::vector<Juerka::CommonNet::neuron_t>, 2> > target_neuron_list(network_size);
		std::vector< std::array<std::vector<Juerka::CommonNet::elec_t>, 2> > synaptic_current_list(network_size);
		std::vector< std::array<std::multimap<Juerka::CommonNet::neuron_t, Juerka::CommonNet::neuron_t>, 2> > strong_edge_list(network_size);
		Juerka::SoundRecognition::SoundCurrentGenerator current_generator(1000, "voice.dat");
		Juerka::SoundRecognition::SoundCurrentGenerator current_generator_sine(1000, "sine.dat");
		std::vector<double> sound_current_input;

		Juerka::CommonNet::NetworkGroup ng(network_size, is_run_parallel, is_monitor_performance, is_record_weights);
		Juerka::Utility::Logger logger(network_size);
		Juerka::Utility::WeightLogger weight_logger(network_size);

		ng.set_update_weights(false, 0, training_layer);

		for (size_t j = 0; j < training_layer; j += 1)
		{
			for (size_t k = 0; k < Juerka::CommonNet::SerialNet::Ne / 1; k += 1)
			{
				target_neuron_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back(k);
			}
		}

		for (Juerka::CommonNet::step_time_t i = 0; i < Juerka::CommonNet::TIME_END; i += 1)
		{
			for (int j = 0; j < training_layer; j += 1)
			{
//				target_neuron_list[j][Juerka::CommonNet::INPUT_SIDE].clear();
				synaptic_current_list[j][Juerka::CommonNet::INPUT_SIDE].clear();
			}

			if (0 == (i % (training_on_time + training_off_time)))
			{
				current_generator.reset_data_point_index();
				current_generator_sine.reset_data_point_index();
			}

			const bool is_training_with_sound{ (i % (training_on_time + training_off_time)) < training_on_time };
			const bool is_training_with_voice{ 0 == ((i / (training_on_time + training_off_time)) % 2) };

			if (is_training_with_sound)
			{
				if (is_training_with_voice)
				{
					//train network with voice

					sound_current_input.clear();
					current_generator.generate_current(sound_current_input);

					size_t j_max(std::min(sound_current_input.size(), training_layer));

					for (int j = 0; j < j_max; j += 1)
					{
						for (int k = 0; k < Juerka::CommonNet::SerialNet::Ne / 1; k += 1)
						{
//							target_neuron_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back(k);
							size_t neuron_index{ (j * (Juerka::CommonNet::SerialNet::Ne / j_max) + k) % sound_current_input.size() };
							synaptic_current_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back(sound_current_input[neuron_index] * current_input_normalizer);
						}
					}
				}
				else
				{
					//train network with sine wave

					sound_current_input.clear();
					current_generator_sine.generate_current(sound_current_input);

					size_t j_max(std::min(sound_current_input.size(), training_layer));

					for (int j = 0; j < j_max; j += 1)
					{
						for (int k = 0; k < Juerka::CommonNet::SerialNet::Ne / 1; k += 1)
						{
//							target_neuron_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back(k);
							size_t neuron_index{ (j * (Juerka::CommonNet::SerialNet::Ne / j_max) + k) % sound_current_input.size() };
							synaptic_current_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back(sound_current_input[neuron_index] * current_input_normalizer);
						}
					}
				}

				target_neuron_list[training_layer + 0][Juerka::CommonNet::INPUT_SIDE].clear();
				synaptic_current_list[training_layer + 0][Juerka::CommonNet::INPUT_SIDE].clear();

				target_neuron_list[training_layer + 1][Juerka::CommonNet::INPUT_SIDE].clear();
				synaptic_current_list[training_layer + 1][Juerka::CommonNet::INPUT_SIDE].clear();


				for (int j = 0; j < training_layer; j += 1)
				{
					//output of previous step
					const auto& output_target_neuron_list(target_neuron_list[j][Juerka::CommonNet::OUTPUT_SIDE]);
					const auto& output_synaptic_current_list(synaptic_current_list[j][Juerka::CommonNet::OUTPUT_SIDE]);

					assert(output_target_neuron_list.size() == output_synaptic_current_list.size());

					size_t index_offset;

					if (is_training_with_voice)
					{
						index_offset = 0;
					}
					else
					{
						index_offset = 1;
					}

					for (size_t k = 0; k < output_target_neuron_list.size(); k += 1)
					{
						//Juerka::CommonNet::neuron_t neuron_index(output_target_neuron_list[k]);
						Juerka::CommonNet::neuron_t neuron_index{ (j * (Juerka::CommonNet::SerialNet::Ne / training_layer) + output_target_neuron_list[k]) % Juerka::CommonNet::SerialNet::Ne };

						if (output_target_neuron_list[k] < Juerka::CommonNet::SerialNet::Ne)
						{
							//target_neuron_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(neuron_index);
							//target_neuron_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(j);
							//target_neuron_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back((j + Juerka::CommonNet::SerialNet::Ne - window / 2 + (neuron_index % window)) % Juerka::CommonNet::SerialNet::Ne);
							target_neuron_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back((neuron_index) % Juerka::CommonNet::SerialNet::Ne);
							//synaptic_current_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(output_synaptic_current_list[k] * synaptic_current_normalizer);
							synaptic_current_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(10.0);
						}
					}
				}
			}

			ng.run(i, target_neuron_list, synaptic_current_list, strong_edge_list);
			logger.log(i, target_neuron_list, synaptic_current_list);
			weight_logger.log(i, strong_edge_list);
			if(0 == (i%1000)) std::cout << i << std::endl;
		}

		ng.set_update_weights(false, 0, network_size);

		size_t validate_step_time(1000);
		size_t counter_for_voice(0);
		size_t counter_for_sine(0);

		size_t i_offset(Juerka::CommonNet::TIME_END);

		for (Juerka::CommonNet::step_time_t i = 0 + i_offset; i < (VALIDATION_TIME_DURATION + i_offset); i += 1)
		{
			for (int j = 0; j < training_layer; j += 1)
			{
//				target_neuron_list[j][Juerka::CommonNet::INPUT_SIDE].clear();
				synaptic_current_list[j][Juerka::CommonNet::INPUT_SIDE].clear();
			}
			
			if (0 == (i%validate_step_time))
			{
				std::cout << i << ' ' << counter_for_voice << ' ' << counter_for_sine << std::endl;

				if (counter_for_voice > counter_for_sine)
				{
					std::cout << "voice" << std::endl;
				}
				else if (counter_for_voice < counter_for_sine)
				{
					std::cout << "sine" << std::endl;
				}

				counter_for_voice = 0;
				counter_for_sine = 0;

				current_generator.reset_data_point_index();
				current_generator_sine.reset_data_point_index();
			}

			const bool is_validate_with_sound{ (i % (training_on_time + training_off_time)) < training_on_time };
			const bool is_validate_with_voice{ 0 == ((i / (validate_step_time)) % 2) };

			if (is_validate_with_sound)
			{
				if (is_validate_with_voice)
				{
					//validate network with voice

					sound_current_input.clear();
					current_generator.generate_current(sound_current_input);

					size_t j_max(std::min(sound_current_input.size(), training_layer));

					for (int j = 0; j < j_max; j += 1)
					{
						for (int k = 0; k < Juerka::CommonNet::SerialNet::Ne / 1; k += 1)
						{
							//target_neuron_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back(k);
							size_t neuron_index{ (j * (Juerka::CommonNet::SerialNet::Ne / j_max) + k) % sound_current_input.size() };
							synaptic_current_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back(sound_current_input[neuron_index] * current_input_normalizer);
						}
					}
				}
				else
				{
					//validate network with sine wave

					sound_current_input.clear();
					current_generator_sine.generate_current(sound_current_input);

					size_t j_max(std::min(sound_current_input.size(), training_layer));

					for (int j = 0; j < j_max; j += 1)
					{
						for (int k = 0; k < Juerka::CommonNet::SerialNet::Ne / 1; k += 1)
						{
							//target_neuron_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back(k);
							size_t neuron_index{ (j * (Juerka::CommonNet::SerialNet::Ne / j_max) + k) % sound_current_input.size() };
							synaptic_current_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back(sound_current_input[neuron_index] * current_input_normalizer);
						}
					}
				}

				target_neuron_list[training_layer + 0][Juerka::CommonNet::INPUT_SIDE].clear();
				synaptic_current_list[training_layer + 0][Juerka::CommonNet::INPUT_SIDE].clear();

				target_neuron_list[training_layer + 1][Juerka::CommonNet::INPUT_SIDE].clear();
				synaptic_current_list[training_layer + 1][Juerka::CommonNet::INPUT_SIDE].clear();

				for (int j = 0; j < training_layer; j += 1)
				{
					//output of previous step
					const auto& output_target_neuron_list(target_neuron_list[j][Juerka::CommonNet::OUTPUT_SIDE]);
					const auto& output_synaptic_current_list(synaptic_current_list[j][Juerka::CommonNet::OUTPUT_SIDE]);

					assert(output_target_neuron_list.size() == output_synaptic_current_list.size());

					for (size_t index_offset = 0; index_offset < output_layer; index_offset += 1)
					{
						for (size_t k = 0; k < output_target_neuron_list.size(); k += 1)
						{
							//Juerka::CommonNet::neuron_t neuron_index(output_target_neuron_list[k]);
							Juerka::CommonNet::neuron_t neuron_index{ (j * (Juerka::CommonNet::SerialNet::Ne / training_layer) + output_target_neuron_list[k]) % Juerka::CommonNet::SerialNet::Ne };

							if (output_target_neuron_list[k] < Juerka::CommonNet::SerialNet::Ne)
							{
								//target_neuron_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(neuron_index);
								//target_neuron_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(j);
								//target_neuron_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back((j + Juerka::CommonNet::SerialNet::Ne - window / 2 + (neuron_index % window)) % Juerka::CommonNet::SerialNet::Ne);
								target_neuron_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back((neuron_index) % Juerka::CommonNet::SerialNet::Ne);
								//synaptic_current_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(output_synaptic_current_list[k] * synaptic_current_normalizer);
								synaptic_current_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(10.0);
							}
						}
					}
				}
			}
			size_t index_offset(0);

			size_t exc_size(target_neuron_list[training_layer + index_offset][Juerka::CommonNet::OUTPUT_SIDE].size());

			for (size_t k = 0; k < exc_size; k += 1)
			{
				Juerka::CommonNet::size_t neuron_index(target_neuron_list[training_layer + index_offset][Juerka::CommonNet::OUTPUT_SIDE][k]);

				if (neuron_index < Juerka::CommonNet::SerialNet::Ne)
				{
					counter_for_voice += 1;
				}
			}
			//counter_for_voice += target_neuron_list[training_layer + index_offset][Juerka::CommonNet::OUTPUT_SIDE].size();
			//std::cout << i << " v " << counter_for_voice << std::endl;

			index_offset += 1;

			exc_size = target_neuron_list[training_layer + index_offset][Juerka::CommonNet::OUTPUT_SIDE].size();

			for (size_t k = 0; k < exc_size; k += 1)
			{
				Juerka::CommonNet::size_t neuron_index(target_neuron_list[training_layer + index_offset][Juerka::CommonNet::OUTPUT_SIDE][k]);

				if (neuron_index < Juerka::CommonNet::SerialNet::Ne)
				{
					counter_for_sine += 1;
				}
			}
			//counter_for_sine += target_neuron_list[training_layer + index_offset][Juerka::CommonNet::OUTPUT_SIDE].size();

			//std::cout << i << " s " << counter_for_sine << std::endl;

			ng.run(i, target_neuron_list, synaptic_current_list, strong_edge_list);
			logger.log(i, target_neuron_list, synaptic_current_list);
			weight_logger.log(i, strong_edge_list);
		}

		if (counter_for_voice > counter_for_sine)
		{
			std::cout << "voice" << std::endl;
		}
		else if (counter_for_voice < counter_for_sine)
		{
			std::cout << "sine" << std::endl;
		}
	}
}
