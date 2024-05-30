#include <array>
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
		std::uint_fast32_t network_size(100);

		std::vector< std::array<std::vector<Juerka::CommonNet::neuron_t>, 2> > target_neuron_list(network_size);
		std::vector< std::array<std::vector<Juerka::CommonNet::elec_t>, 2> > synaptic_current_list(network_size);
		std::vector< std::array<std::multimap<Juerka::CommonNet::neuron_t, Juerka::CommonNet::neuron_t>, 2> > strong_edge_list(network_size);
		Juerka::SoundRecognition::SoundCurrentGenerator current_generator(1000, "voice.dat");
		Juerka::SoundRecognition::SoundCurrentGenerator current_generator_sine(1000, "sine.dat");
		std::vector<double> sound_current_input;

		Juerka::CommonNet::NetworkGroup ng(network_size, is_run_parallel, is_monitor_performance, is_record_weights);
		Juerka::Utility::Logger logger(network_size);
		Juerka::Utility::WeightLogger weight_logger(network_size);

		for(Juerka::CommonNet::step_time_t i=0; i<Juerka::CommonNet::TIME_END; i+=1)
		{
			for (int j = 0; j < network_size; j += 1)
			{
				target_neuron_list[j][Juerka::CommonNet::INPUT_SIDE].clear();
				synaptic_current_list[j][Juerka::CommonNet::INPUT_SIDE].clear();
			}

			bool is_run_with_voice(false);

			if (is_run_with_voice)
			{
				if ((i % 100) < 50)
				{
					sound_current_input.clear();
					current_generator.generate_current(sound_current_input);

					for (int j = 0; j < sound_current_input.size(); j += 1)
					{
						target_neuron_list[0][Juerka::CommonNet::INPUT_SIDE].emplace_back(j);
						synaptic_current_list[0][Juerka::CommonNet::INPUT_SIDE].emplace_back(sound_current_input[j]);
					}
				}
			}
			else
			{
				if (i < 500)
				{
					if ((i % 100) < 50)
					{
						sound_current_input.clear();
						current_generator.generate_current(sound_current_input);

						for (int j = 0; j < sound_current_input.size(); j += 1)
						{
							target_neuron_list[0][Juerka::CommonNet::INPUT_SIDE].emplace_back(j);
							synaptic_current_list[0][Juerka::CommonNet::INPUT_SIDE].emplace_back(sound_current_input[j]);
						}
					}
				}
				else
				{
					if ((i % 100) < 50)
					{
						sound_current_input.clear();
						current_generator_sine.generate_current(sound_current_input);

						for (int j = 0; j < sound_current_input.size(); j += 1)
						{
							target_neuron_list[0][Juerka::CommonNet::INPUT_SIDE].emplace_back(j);
							synaptic_current_list[0][Juerka::CommonNet::INPUT_SIDE].emplace_back(sound_current_input[j]);
						}
					}
				}
			}

			if (i < 500)
			{
				if ((5 <= (i % 100)) && (55 > (i % 100)))
				{
					for (int j = 600; j < 800; j += 1)
					{
						target_neuron_list[0][Juerka::CommonNet::INPUT_SIDE].emplace_back(j);
						synaptic_current_list[0][Juerka::CommonNet::INPUT_SIDE].emplace_back(15.0);
					}
				}
			}

			ng.run(i, target_neuron_list, synaptic_current_list, strong_edge_list);
			logger.log(i, target_neuron_list, synaptic_current_list);
			weight_logger.log(i, strong_edge_list);
			std::cout << i << std::endl;
		}
	}
}
