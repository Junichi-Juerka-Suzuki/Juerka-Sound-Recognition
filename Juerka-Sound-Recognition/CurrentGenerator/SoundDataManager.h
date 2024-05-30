#pragma once

#include <algorithm>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "Common.h"

namespace Juerka::SoundRecognition
{
	using std::back_inserter;
	using std::copy;
	using std::ifstream;
	using std::max_element;

	using std::string;
	using std::transform;
	using std::vector;

	class SoundDataManager
	{
		vector<sound_t> data_store;

	public:
		void clear_sound_data(void)
		{
			data_store.clear();
		}

		size_t get_sound_data_length(void)
		{
			return data_store.size();
		}

		void normalize_sound_data(void)
		{
			const auto max_it(max_element(data_store.begin(), data_store.end()));
			const auto max_value(*max_it);

			if (0.0 != max_value)
			{
				transform
				(
					data_store.begin(),
					data_store.end(),
					data_store.begin(),
					[max_value](double a) { return a / max_value; }
				);
			}
		}

		void get_sound_data_slice(vector<sound_t>& out, size_t start_index, size_t end_sentinel)
		{
			copy(&data_store[start_index], &data_store[end_sentinel], back_inserter(out));
		}

		bool load_sound_file(string&& filename)
		{
			ifstream ifs(filename);
			sound_t sound;

			bool is_sound_loaded(false);

			while (ifs >> sound)
			{
				data_store.emplace_back(sound);
				is_sound_loaded = true;
			}

			return is_sound_loaded;
		}

		SoundDataManager() noexcept = default;
		// copy
		SoundDataManager(const SoundDataManager&) = default;
		SoundDataManager& operator = (const SoundDataManager&) = default;
		// move
		SoundDataManager(SoundDataManager&&) noexcept = default;
		SoundDataManager& operator = (SoundDataManager&&) noexcept = default;
		// dtor
		virtual ~SoundDataManager() noexcept = default;
	};
}
