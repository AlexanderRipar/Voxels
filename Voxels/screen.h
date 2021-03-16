#pragma once

#include <cstdint>

#include <och_utf8.h>

namespace wnd
{
	enum styles : uint32_t
	{
		closeable       = 0x0001,
		minimizable     = 0x0002,
		maximizable     = 0x0004,
		resizeable      = 0x0008,

		full_menu      = closeable | minimizable | maximizable,

		init_minimized = 0x8000,
		init_maximized = 0xC000,
	};
}

void define_screen(och::stringview id);

struct screen
{
	void* system_screen_handle;

	uint8_t vk_events[226]{};

	screen(och::stringview id, uint32_t width, uint32_t height, uint32_t initial_x, uint32_t initial_y, uint32_t style = 0, och::stringview name = och::stringview(nullptr, 0, 0));

	int32_t handle_message(uint32_t msg_id = 0);

	void show();

	void maximize();

	void minimize();

	void set_title(och::stringview s);
};
