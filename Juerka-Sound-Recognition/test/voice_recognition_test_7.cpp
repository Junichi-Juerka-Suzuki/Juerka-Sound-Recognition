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
		//const size_t training_layer(0);
		//const size_t output_layer(2);
		const size_t network_size{ 2 };
		const size_t training_on_time{ 800 };
		const size_t training_off_time{ 200 };
		const Juerka::CommonNet::step_time_t VALIDATION_TIME_DURATION{ 4000 };

		//const double current_input_normalizer_table[network_size] = { 1.0 / 100, 2.75 / 20 };
		const double current_input_normalizer_table[network_size] { 2.75/50 , 1.0/8  };
		//const double EXC_BASE_CURRENT{ 7.5 };
		//const double INH_CURRENT{ -35 };
		const double INH_CURRENT{ -5 };
		const double INH_LABEL_CURRENT{ -5 };
		const size_t EXC_NUM{ 800 };
		const size_t INH_NUM{ 1600 };
		const size_t NO_SOUND_BLOCK{ 200 };
		double current_input_normalizer{ 0.0 };

		std::vector< std::array<std::vector<Juerka::CommonNet::neuron_t>, 2> > target_neuron_list(network_size);
		std::vector< std::array<std::vector<Juerka::CommonNet::elec_t>, 2> > synaptic_current_list(network_size);
		std::vector< std::array<std::multimap<Juerka::CommonNet::neuron_t, Juerka::CommonNet::neuron_t>, 2> > strong_edge_list(network_size);
		Juerka::SoundRecognition::SoundCurrentGenerator current_generator(1000, "voice.dat");
		Juerka::SoundRecognition::SoundCurrentGenerator current_generator_sine(1000, "sine.dat");
		std::vector<double> sound_current_input;

		Juerka::CommonNet::NetworkGroup ng(network_size, is_run_parallel, is_monitor_performance, is_record_weights);
		Juerka::Utility::Logger logger(network_size);
		Juerka::Utility::WeightLogger weight_logger(network_size);

		size_t exc_counter[network_size] = { 0 };
		size_t inh_counter[network_size] = { 0 };
		//ng.set_update_weights(false, 0, training_layer);

		for (Juerka::CommonNet::step_time_t i = 0; i < Juerka::CommonNet::TIME_END; i += 1)
		{
			for (int j = 0; j < network_size; j += 1)
			{
				target_neuron_list[j][Juerka::CommonNet::INPUT_SIDE].clear();
				synaptic_current_list[j][Juerka::CommonNet::INPUT_SIDE].clear();
			}

			if (0 == (i % (training_on_time + training_off_time)))
			{
				current_generator.reset_data_point_index();
				current_generator_sine.reset_data_point_index();
			}

			const bool is_training_with_sound{ (i % (training_on_time + training_off_time)) < training_on_time };
			const bool is_training_with_voice{ 0 == ((i / (training_on_time + training_off_time)) % 2) };

			sound_current_input.clear();

			if (is_training_with_voice)
			{
				current_generator.generate_current(sound_current_input);
				current_input_normalizer = current_input_normalizer_table[0];
			}
			else
			{
				current_generator_sine.generate_current(sound_current_input);
				current_input_normalizer = current_input_normalizer_table[1];
			}

			if (is_training_with_sound)
			{
				for (int j = 0; j < network_size; j += 1)
				{
					for (int k = 0; k < EXC_NUM; k += 1)
					{
						target_neuron_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back((exc_counter[j]) % Juerka::CommonNet::SerialNet::Ne);
						size_t neuron_index{ exc_counter[j] % (sound_current_input.size() - NO_SOUND_BLOCK)};
						synaptic_current_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back(sound_current_input[neuron_index] * current_input_normalizer);
						exc_counter[j] += 1;
					}
				}

				for (int j = 0; j < network_size; j += 1)
				{
					if (is_training_with_voice)
					{
						//if (0 == j)
						if (1 == j)
						{
							for (int k = 0; k < Juerka::CommonNet::SerialNet::Ne; k += 1)
							{
								target_neuron_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back(k);
								synaptic_current_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back(INH_LABEL_CURRENT);
							}
						}
					}
					else
					{
						//if (1 == j)
						if (0 == j)
						{
							for (int k = 0; k < Juerka::CommonNet::SerialNet::Ne; k += 1)
							{
								target_neuron_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back(k);
								synaptic_current_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back(INH_LABEL_CURRENT);
							}
						}
					}
				}


				for (int j = 0; j < network_size; j += 1)
				{
					//output of previous step
					const auto& output_target_neuron_list(target_neuron_list[j][Juerka::CommonNet::OUTPUT_SIDE]);
					const auto& output_synaptic_current_list(synaptic_current_list[j][Juerka::CommonNet::OUTPUT_SIDE]);

					assert(output_target_neuron_list.size() == output_synaptic_current_list.size());

					for (size_t index_offset = 0; index_offset < network_size; index_offset += 1)
					{
						if ((index_offset) == j)
						{
							continue;
						}

						//double iadd{ 0.0 };
						double iadd{ INH_CURRENT };

						//if (is_training_with_voice)
						//{
						//	if (1 == index_offset)
						//	{
						//		iadd = INH_CURRENT;
						//	}
						//}
						//else
						//{
						//	if (0 == index_offset)
						//	{
						//		iadd = INH_CURRENT;
						//	}
						//}

						if (0.0 != iadd)
						{
							for (size_t offset = 0; offset < INH_NUM; offset += 1)
							{
								//Juerka::CommonNet::neuron_t neuron_index(output_target_neuron_list[k]);
								//Juerka::CommonNet::neuron_t neuron_index{ (output_target_neuron_list[k] * Juerka::CommonNet::SerialNet::Ne / Juerka::CommonNet::SerialNet::Ni + offset) % Juerka::CommonNet::SerialNet::Ne };
								Juerka::CommonNet::neuron_t neuron_index{ (inh_counter[j]) % Juerka::CommonNet::SerialNet::Ne};

								//target_neuron_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(neuron_index);
								//target_neuron_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(j);
								//target_neuron_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back((j + Juerka::CommonNet::SerialNet::Ne - window / 2 + (neuron_index % window)) % Juerka::CommonNet::SerialNet::Ne);
								target_neuron_list[index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(neuron_index);
								//synaptic_current_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(output_synaptic_current_list[k] * synaptic_current_normalizer);
								synaptic_current_list[index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(iadd);

								inh_counter[j] += 1;
							}
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

		const size_t validate_step_time(1000);
		const size_t i_offset(Juerka::CommonNet::TIME_END);

		std::array<size_t, network_size> counter_for_sound = { 0 };
		size_t counter_for_voice(0);
		size_t counter_for_sine(0);
		const Juerka::CommonNet::step_time_t i_init{ 0 + i_offset };
		for (Juerka::CommonNet::step_time_t i = i_init; i < (VALIDATION_TIME_DURATION + i_offset); i += 1)
		{
			for (int j = 0; j < network_size; j += 1)
			{
				target_neuron_list[j][Juerka::CommonNet::INPUT_SIDE].clear();
				synaptic_current_list[j][Juerka::CommonNet::INPUT_SIDE].clear();
			}
			
			if (0 == (i%validate_step_time))
			{
				counter_for_voice = counter_for_sound[0];
				counter_for_sine = counter_for_sound[1];

				if (counter_for_voice > counter_for_sine)
				{
					std::cout << "voice" << std::endl;
				}
				else if (counter_for_voice < counter_for_sine)
				{
					std::cout << "sine" << std::endl;
				}
				else
				{
					if (i_init != i)
					{
						std::cout << "even" << std::endl;
					}
				}

				if (i_init != i)
				{
					std::cout << counter_for_voice << ' ' << counter_for_sine << std::endl;
				}

				counter_for_voice = 0;
				counter_for_sine = 0;

				for (size_t j = 0; j < network_size; j += 1)
				{
					counter_for_sound[j] = 0;
				}

				current_generator.reset_data_point_index();
				current_generator_sine.reset_data_point_index();
			}

			const bool is_validate_with_sound{ (i % (training_on_time + training_off_time)) < training_on_time };
			const bool is_validate_with_voice{ 0 == ((i / (validate_step_time)) % 2) };

			sound_current_input.clear();

			if (is_validate_with_sound)
			{
				if (is_validate_with_voice)
				{
					current_generator.generate_current(sound_current_input);
					current_input_normalizer = current_input_normalizer_table[0];
				}
				else
				{
					current_generator_sine.generate_current(sound_current_input);
					current_input_normalizer = current_input_normalizer_table[1];
				}

				for (int j = 0; j < network_size; j += 1)
				{
					for (int k = 0; k < EXC_NUM; k += 1)
					{
						target_neuron_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back((exc_counter[j]) % Juerka::CommonNet::SerialNet::Ne);
						size_t neuron_index{ exc_counter[j] % (sound_current_input.size()-NO_SOUND_BLOCK)};
						synaptic_current_list[j][Juerka::CommonNet::INPUT_SIDE].emplace_back(sound_current_input[neuron_index] * current_input_normalizer);
						exc_counter[j] += 1;
					}
				}
			}

			for (int j = 0; j < network_size; j += 1)
			{
				//output of previous step
				const auto& output_target_neuron_list(target_neuron_list[j][Juerka::CommonNet::OUTPUT_SIDE]);
				const auto& output_synaptic_current_list(synaptic_current_list[j][Juerka::CommonNet::OUTPUT_SIDE]);

				assert(output_target_neuron_list.size() == output_synaptic_current_list.size());

				for (size_t index_offset = 0; index_offset < network_size; index_offset += 1)
				{
					if ((index_offset) == j)
					{
						continue;
					}

					double iadd{ INH_CURRENT };

					if (0.0 != iadd)
					{
						for (size_t offset = 0; offset < INH_NUM; offset += 1)
						{
							//Juerka::CommonNet::neuron_t neuron_index(output_target_neuron_list[k]);
							//Juerka::CommonNet::neuron_t neuron_index{ (output_target_neuron_list[k] * Juerka::CommonNet::SerialNet::Ne / Juerka::CommonNet::SerialNet::Ni + offset) % Juerka::CommonNet::SerialNet::Ne };
							Juerka::CommonNet::neuron_t neuron_index{ (inh_counter[j]) % Juerka::CommonNet::SerialNet::Ne };

							//target_neuron_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(neuron_index);
							//target_neuron_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(j);
							//target_neuron_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back((j + Juerka::CommonNet::SerialNet::Ne - window / 2 + (neuron_index % window)) % Juerka::CommonNet::SerialNet::Ne);
							target_neuron_list[index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(neuron_index);
							//synaptic_current_list[training_layer + index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(output_synaptic_current_list[k] * synaptic_current_normalizer);
							synaptic_current_list[index_offset][Juerka::CommonNet::INPUT_SIDE].emplace_back(iadd);

							inh_counter[j] += 1;
						}
					}
				}
			}

			for (size_t index_offset = 0; index_offset < network_size; index_offset += 1)
			{
				size_t spike_count(target_neuron_list[index_offset][Juerka::CommonNet::OUTPUT_SIDE].size());

				for (size_t k = 0; k < spike_count; k += 1)
				{
					Juerka::CommonNet::neuron_t neuron_index(target_neuron_list[index_offset][Juerka::CommonNet::OUTPUT_SIDE][k]);

					if (neuron_index < Juerka::CommonNet::SerialNet::Ne)
					{
						counter_for_sound[index_offset] += 1;
					}
				}
			}

			ng.run(i, target_neuron_list, synaptic_current_list, strong_edge_list);
			logger.log(i, target_neuron_list, synaptic_current_list);
			weight_logger.log(i, strong_edge_list);
		}

		counter_for_voice = counter_for_sound[0];
		counter_for_sine = counter_for_sound[1];

		if (counter_for_voice > counter_for_sine)
		{
			std::cout << "voice" << std::endl;
		}
		else if (counter_for_voice < counter_for_sine)
		{
			std::cout << "sine" << std::endl;
		}
		else
		{
			std::cout << "even" << std::endl;
		}

		if(1)
		{
			std::cout << counter_for_voice << ' ' << counter_for_sine << std::endl;
		}
	}
}
