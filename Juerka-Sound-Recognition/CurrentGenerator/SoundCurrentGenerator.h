#pragma once
#include <algorithm>
#include <complex>
#include <iostream>
#include <string>

#include "Fft.h"
#include "SoundDataManager.h"

namespace Juerka::SoundRecognition
{
	using std::complex;
	using std::min;
	using std::string;
	using std::vector;

	class SoundCurrentGenerator
	{
		size_t data_series_size;
		size_t sound_data_point_index;
		SoundDataManager sound_data_manager;
		double multiplying_factor;
		size_t sound_data_delta;

	public:
		SoundCurrentGenerator(size_t arg_data_series_size, string&& filename)
			:
			data_series_size(arg_data_series_size),
			sound_data_point_index(0),
			multiplying_factor(DEFAULT_MULTIPLYING_FACTOR),
			sound_data_delta(3)
		{
			sound_data_manager.clear_sound_data();
			sound_data_manager.load_sound_file(move(filename), data_series_size);
			sound_data_manager.normalize_sound_data();
		}

		void reset_data_point_index(void)
		{
			sound_data_point_index = 0;
		}

		void generate_current(vector<double>& out)
		{
			vector<sound_t> time_domain_sound_data;
			vector<double> time_domain_data;
			vector<complex<double>> frequency_domain_data;

			for (size_t i = 0; i < data_series_size; i += 1)
			{
				if (time_domain_data.size() >= data_series_size)
				{
					break;
				}
				time_domain_data.emplace_back(0.0);
			}

			for (size_t i = 0; i < data_series_size + 2; i += 1)
			{
				if (frequency_domain_data.size() >= (data_series_size + 2))
				{
					break;
				}
				frequency_domain_data.emplace_back(0.0);
			}

			size_t end_sound_data_point_index(sound_data_point_index + data_series_size);
			size_t max_data_point(sound_data_manager.get_sound_data_length());

			if (end_sound_data_point_index > max_data_point)
			{
				sound_data_point_index = 0;
				end_sound_data_point_index = min(max_data_point, data_series_size);
			}

			sound_data_manager.get_sound_data_slice
			(
				time_domain_sound_data,
				sound_data_point_index,
				end_sound_data_point_index
			);

			for (size_t i = 0; i < time_domain_data.size(); i++)
			{
				time_domain_data[i] = 0.0;
			}

			{
				size_t i{ 0 };

				for (auto it = time_domain_sound_data.begin(); it != time_domain_sound_data.end(); it++)
				{
					time_domain_data[i] = static_cast<double>(*it);
					i += 1;
				}
			}

			for (size_t i = 0; i < frequency_domain_data.size(); i++)
			{
				frequency_domain_data[i] = 0.0;
			}

			do_dft(time_domain_data, frequency_domain_data);

			for (size_t i = 0; i < (time_domain_data.size() / 2); i++)
			{
				out.emplace_back(sqrt(abs(frequency_domain_data[i]) * multiplying_factor));
			}

			sound_data_point_index += sound_data_delta;
		}

		// copy
		SoundCurrentGenerator(const SoundCurrentGenerator&) = default;
		SoundCurrentGenerator& operator = (const SoundCurrentGenerator&) = default;
		// move
		SoundCurrentGenerator(SoundCurrentGenerator&&) noexcept = default;
		SoundCurrentGenerator& operator = (SoundCurrentGenerator&&) noexcept = default;
		// dtor
		virtual ~SoundCurrentGenerator() noexcept = default;
	};
}