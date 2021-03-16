#include <cstdint>
#include <cstdio>

#include "voxels.h"
#include "screen.h"

#include "och_simplex_noise.h"
#include "och_fmt.h"

#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

constexpr int log2_sz = 8;

constexpr int sz = 1 << log2_sz;

uint8_t slice[sz * sz];

class window : public olc::PixelGameEngine
{
public:

	float total_t = 0;

	uint8_t max = 0, min = 255;

	window() { sAppName = "Test"; }

	bool OnUserCreate() override
	{
		return true;
	}

	bool OnUserUpdate(float t) override
	{
		if (GetKey(olc::ENTER).bPressed || GetKey(olc::ESCAPE).bPressed)
			return false;

		total_t += t * 64;

		if (total_t >= (float)sz)
		{
			total_t -= (float)sz;
			och::print("\nmin: {}, max: {}\n", min, max);
		}

		get_slice(slice, (uint32_t) total_t, sz);

		for (int y = 0; y != sz; ++y)
			for (int x = 0; x != sz; ++x)
			{
				uint8_t col = slice[x + y * sz];

				if (col < min) min = col;
				if (col > max) max = col;

				Draw(x, y, olc::Pixel(col, col, col));
			}

		return true;
	}
};

const och::stringview screen_id("screen");

int main(int argc, const char** argv)
{


	launch_voxels(log2_sz, 0, 0);
	
	window w;
	
	if (w.Construct(sz, sz, 1, 1))
		w.Start();
}
