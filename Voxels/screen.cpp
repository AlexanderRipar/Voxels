#include "screen.h"

#include <Windows.h>
#include <cstdint>

#include "och_range.h"
#include "och_fmt.h"

#include "virtual_keys.h"

LRESULT screen_function(HWND screen, UINT msg, WPARAM wp, LPARAM lp)
{
	if (msg == WM_CLOSE)
	{
		PostQuitMessage(0);
		return 0;
	}

	DefWindowProcA(screen, msg, wp, lp);
}

void define_screen(och::stringview id)
{
	WNDCLASSA w{};

	w.style = CS_OWNDC;

	w.lpfnWndProc = screen_function;

	w.hInstance = GetModuleHandle(nullptr);

	w.lpszClassName = id.raw_cbegin();

	RegisterClassA(&w);
}

uint32_t derive_style(uint32_t style)
{
	uint32_t s = 0;

	if (style & wnd::closeable)
		s |= WS_SYSMENU;
	if (style & wnd::minimizable)
		s |= WS_MINIMIZEBOX;
	if (style & wnd::maximizable)
		s |= WS_MAXIMIZEBOX;
	if (style & wnd::resizeable)
		s |= WS_SIZEBOX;

	if (style & wnd::init_minimized)
		s |= WS_MINIMIZE;
	else if (style & wnd::init_maximized)
		s |= WS_MAXIMIZE;

	return s;
}

uint32_t derive_style_ex(uint32_t style)
{
	return 0;
}

screen::screen(och::stringview id, uint32_t width, uint32_t height, uint32_t initial_x, uint32_t initial_y, uint32_t style, och::stringview name) :
	system_screen_handle{ CreateWindowExA(derive_style_ex(style), id.raw_cbegin(), name.raw_cbegin(), derive_style(style), initial_x, initial_y, width, height, nullptr, nullptr, GetModuleHandleA(nullptr), nullptr) } {}

int32_t screen::handle_message(uint32_t msg_id)
{
	uint32_t processed_cnt = 0;

	MSG m{};

	int32_t message_info = GetMessageA(&m, reinterpret_cast<HWND>(system_screen_handle), msg_id, msg_id);

	if (message_info > 0)
	{
		switch (m.message)
		{
		case WM_KEYDOWN:
		{
			if (vk_events[m.wParam] == 1)
				vk_events[m.wParam] = 2;
			else
				vk_events[m.wParam] = 1;

			break;
		}
		case WM_KEYUP:
		{
			vk_events[m.wParam] = 0;

			break;
		}
		case WM_CHAR:
		{
			och::print("{:c}", m.wParam);
			break;
		}
		}

		if (uint32_t err = GetLastError())
			och::print("ERR: {}\n", err);

		TranslateMessage(&m);
		DispatchMessageA(&m);
	}

	return message_info;
}

void screen::show() { ShowWindow(reinterpret_cast<HWND>(system_screen_handle), SW_SHOW); }

void screen::maximize() { ShowWindow(reinterpret_cast<HWND>(system_screen_handle), SW_MAXIMIZE); }

void screen::minimize() { ShowWindow(reinterpret_cast<HWND>(system_screen_handle), SW_MINIMIZE); }

void screen::set_title(och::stringview s) { SetWindowTextA(reinterpret_cast<HWND>(system_screen_handle), s.raw_cbegin()); }
