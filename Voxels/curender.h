#pragma once

#include <cstdint>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
//#include <D3Dcompiler.h>
//#include <DirectXMath.h>
#include "d3dx12.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"

#include "och_lib.h"

//#include "och_setints_gpu.cuh"
#include "och_simplex_noise_gpu.cuh"

//DEBUG SWITCH
#define GRAPHICS_DEBUG

void dump_and_flee(uint64_t error_number, const char* error_name, const char* error_desc, const char* src_file, int line_number)
{
	//Remove path from file-name
	int last_backslash = -1;

	for (int i = 0; src_file[i]; ++i)
		if (src_file[i] == '\\')
			last_backslash = i;

	const char* filename = src_file + last_backslash + 1;
	och::print("\nERROR ({0} | 0x{0:X}): {1}\n\n{2}\n\nFile: {3}\nLine: {4}\n\n", error_number, error_name, error_desc, filename, line_number);

	exit(1);
}

void dump_and_flee(const char* message, const char* src_file, int line_number)
{
	och::print("\nRUNTIME-ERROR:\n\n{}\n\nFile: {}\nLine: {}\n\n", message, src_file, line_number);
}

void check_(HRESULT err, const char* src_file, int line_number)
{
	if (!FAILED(err))
		return;

	char error_buf[1024];

	uint64_t error_code = static_cast<uint64_t>(err);

	const char* error_desc;

	if (!FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, err, 0, error_buf, sizeof(error_buf), nullptr))
		error_desc = "[[No error information available. Error is HRESULT]]";
	else
		error_desc = error_buf;

	dump_and_flee(error_code, "", error_desc, src_file, line_number);
}

void check_(cudaError_t err, const char* src_file, int line_number)
{
	if (err == cudaSuccess)
		return;

	dump_and_flee(static_cast<uint64_t>(err), cudaGetErrorName(err), cudaGetErrorString(err), src_file, line_number);
}

#define check(x) check_(x, __FILE__, __LINE__);

#define panic(msg) dump_and_flee(msg, __FILE__, __LINE__);

LRESULT CALLBACK window_function(HWND window, UINT msg, WPARAM wp, LPARAM lp);

struct render_data
{

	/*////////////////////////////////////////////////////////////////////////*/
	/*//////////////////////////////////DATA//////////////////////////////////*/
	/*////////////////////////////////////////////////////////////////////////*/

	static constexpr uint8_t m_frame_cnt = 2;

	static constexpr const wchar_t* m_window_class_name = L"OCHVXWN";
	const wchar_t* m_window_title;
	HWND m_window;
	RECT m_window_rect;

	//RENDER STATE
	ID3D12Device2* m_device;
	ID3D12CommandQueue* m_cmd_queue;
	IDXGISwapChain4* m_swapchain;
	ID3D12Resource* m_backbuffers[m_frame_cnt];
	ID3D12GraphicsCommandList* m_cmd_list;
	ID3D12CommandAllocator* m_cmd_allocators[m_frame_cnt];
	ID3D12DescriptorHeap* m_rtv_desc_heap;

	ID3D12Fence* m_fence;
	uint64_t m_fence_values[m_frame_cnt]{};
	uint64_t m_curr_fence_value = 0;
	HANDLE m_fence_event;

	//TEMPORAL RENDER STATE
	uint16_t m_rtv_desc_size;
	uint16_t m_window_width = 1280;
	uint16_t m_window_height = 720;
	uint8_t m_curr_frame = 0;

	//FLAGS
	bool m_vsync = true;
	bool m_supports_tearing;
	bool m_is_fullscreen = false;
	bool m_is_initialized = false;

	//INPUT
	uint64_t m_keystates[4]{};
	int16_t m_mouse_x;
	int16_t m_mouse_y;
	int16_t m_mouse_scroll;
	int16_t m_mouse_h_scroll;

	//CUDA INTEROP
	cudaExternalMemory_t m_cu_external_memory_handles[m_frame_cnt];
	cudaSurfaceObject_t m_cu_surfaces[m_frame_cnt];
	cudaExternalSemaphore_t m_cu_fence;
	HANDLE m_cu_backbuffer_shared_handles[m_frame_cnt];
	HANDLE m_cu_fence_shared_handle;

	//D3D12 DEBUG
#ifdef GRAPHICS_DEBUG
	ID3D12Debug* m_debug_interface;
#endif

	/*////////////////////////////////////////////////////////////////////////*/
	/*///////////////////////////////CTOR / DTOR//////////////////////////////*/
	/*////////////////////////////////////////////////////////////////////////*/

	render_data(uint32_t width, uint32_t height, const wchar_t* title)
	{
		och::print("Initializing...\n");

		//Timer for initialization
		och::timer initialization_timer;

		//Output-Codepage to unicode (UTF-8)
		SetConsoleOutputCP(65001);

		//Set DPI to be separate for each screen, to allow for flexible tearing-behaviour
		SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

		//Initialize D3D12 Debugger
		{
			#ifdef GRAPHICS_DEBUG

				check(D3D12GetDebugInterface(IID_PPV_ARGS(&m_debug_interface)));

				m_debug_interface->EnableDebugLayer();

			#endif // GRAPHICS_DEBUG
		}

		union
		{
			char cuda[8];
			LUID d3d12{ (DWORD)~0, (LONG)~0 };
		} luid;

		//Select the best cuda device by major ver, minor ver, sm-count and global mem.
		//Device-LUID is stored in the above luid union
		{
			int32_t cuda_dev_cnt;

			check(cudaGetDeviceCount(&cuda_dev_cnt));

			int32_t best_major_ver = -1;
			int32_t best_minor_ver = -1;
			int32_t best_sm_cnt = -1;
			uint64_t best_gmem_bytes = 0;

			uint32_t best_idx = -1;

			for (int32_t i = 0; i != cuda_dev_cnt; ++i)
			{
				cudaDeviceProp prop;

				cudaError_t e2 = cudaGetDeviceProperties(&prop, i);

				if (e2 != cudaSuccess)
					check(e2);

				if (prop.major >= best_major_ver && prop.minor >= best_minor_ver && prop.multiProcessorCount >= best_sm_cnt && prop.totalGlobalMem >= best_gmem_bytes)
				{
					best_major_ver = prop.major;
					best_minor_ver = prop.minor;
					best_sm_cnt = prop.multiProcessorCount;
					best_gmem_bytes = prop.totalGlobalMem;

					best_idx = i;

					for (int j = 0; j != 8; ++j)
						luid.cuda[j] = prop.luid[j];
				}
			}

			cudaSetDevice(best_idx);
		}

		//Initialize window
		{
			m_window_width = width;
			m_window_height = height;
			m_window_title = title;

			WNDCLASSEXW window_class{};

			window_class.cbSize = sizeof(window_class);
			window_class.style = CS_VREDRAW | CS_HREDRAW;
			window_class.lpfnWndProc = window_function;
			window_class.hInstance = GetModuleHandleW(nullptr);
			window_class.lpszClassName = m_window_class_name;
			RegisterClassExW(&window_class);

			int32_t screen_width = GetSystemMetrics(SM_CXSCREEN);

			int32_t screen_height = GetSystemMetrics(SM_CYSCREEN);

			RECT window_rect = { 0, 0, static_cast<LONG>(m_window_width), static_cast<LONG>(m_window_height) };

			AdjustWindowRect(&window_rect, WS_OVERLAPPEDWINDOW, FALSE);

			int32_t actual_width = window_rect.right - window_rect.left;

			int32_t actual_height = window_rect.bottom - window_rect.top;

			// Center the window within the screen. Clamp to 0, 0 for the top-left corner.
			int32_t window_x = (screen_width - actual_width) >> 1;
			if (window_x < 0) window_x = 0;

			int32_t window_y = (screen_height - actual_height) >> 1;
			if (window_y < 0) window_y = 0;

			m_window = CreateWindowExW(0, m_window_class_name, m_window_title, WS_OVERLAPPEDWINDOW, window_x, window_y, actual_width, actual_height, nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);

			SetWindowLongPtrW(m_window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));

			GetWindowRect(m_window, &m_window_rect);
		}

		IDXGIFactory4* dxgi_factory;

		//Create DXGI-factory COM-Interface, which is stored in the above pointer
		{
			uint32_t factory_flags = 0;

			#ifdef GRAPHICS_DEBUG

				factory_flags = DXGI_CREATE_FACTORY_DEBUG;

			#endif // GRAPHICS_DEBUG

			check(CreateDXGIFactory2(factory_flags, IID_PPV_ARGS(&dxgi_factory)));
		}

		//Query whether tearing is supported
		{
			IDXGIFactory5* dxgi_factory_5;

			check(dxgi_factory->QueryInterface(&dxgi_factory_5));

			BOOL is_allowed;

			if (FAILED(dxgi_factory_5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &is_allowed, sizeof(is_allowed))))
				m_supports_tearing = false;

			m_supports_tearing = is_allowed;

			dxgi_factory_5->Release();
		}

		//Get the DXGI adapter  corresponding to the luid of the previously selected cuda-device and use it to create a D3D12-Device
		{
			IDXGIAdapter4* dxgi_adapter;

			IDXGIAdapter1* adapter_1;

			check(dxgi_factory->EnumAdapterByLuid(luid.d3d12, IID_PPV_ARGS(&adapter_1)));

			//Check if the chosen device actually supports D3D12
			check(D3D12CreateDevice(adapter_1, D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr));

			check(adapter_1->QueryInterface(&dxgi_adapter));

			check(D3D12CreateDevice(dxgi_adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device)));

			#ifdef GRAPHICS_DEBUG

				//Set up warnings
				ID3D12InfoQueue* info_queue;

				check(m_device->QueryInterface(&info_queue));

				info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
				info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
				info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, true);

				D3D12_MESSAGE_SEVERITY info_severity = D3D12_MESSAGE_SEVERITY_INFO;

				D3D12_INFO_QUEUE_FILTER message_filter{};
				message_filter.DenyList.pSeverityList = &info_severity;
				message_filter.DenyList.NumSeverities = 1;

				check(info_queue->PushStorageFilter(&message_filter));

				info_queue->Release();

			#endif // GRAPHICS_DEBUG

			adapter_1->Release();

			dxgi_adapter->Release();
		}

		//Create D3D12 command-queue
		{
			D3D12_COMMAND_QUEUE_DESC queue_desc{};

			queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
			queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
			queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
			queue_desc.NodeMask = 0;

			check(m_device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&m_cmd_queue)));
		}

		//Create DXGI swapchain
		{
			DXGI_SWAP_CHAIN_DESC1 swapchain_desc{};

			swapchain_desc.Width = width;
			swapchain_desc.Height = height;
			swapchain_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			swapchain_desc.Stereo = false;
			swapchain_desc.SampleDesc = { 1, 0 };
			swapchain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
			swapchain_desc.BufferCount = m_frame_cnt;
			swapchain_desc.Scaling = DXGI_SCALING_STRETCH;//CUSTOMIZE
			swapchain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
			swapchain_desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;//CUSTOM
			if (m_supports_tearing)
				swapchain_desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

			IDXGISwapChain1* swapchain_1;

			check(dxgi_factory->CreateSwapChainForHwnd(m_cmd_queue, m_window, &swapchain_desc, nullptr, nullptr, &swapchain_1));

			check(dxgi_factory->MakeWindowAssociation(m_window, DXGI_MWA_NO_ALT_ENTER));

			check(swapchain_1->QueryInterface(&m_swapchain));

			swapchain_1->Release();
		}

		//Get index of current backbuffer from swapchain
		m_curr_frame = m_swapchain->GetCurrentBackBufferIndex();

		//Create Render Target View descriptor heap for use by the swapchain
		{
			D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};

			heap_desc.NumDescriptors = m_frame_cnt;
			heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
			heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
			heap_desc.NodeMask = 0;

			check(m_device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&m_rtv_desc_heap)));
		}

		//Query size of RTV descriptors created above
		m_rtv_desc_size = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

		//Bind backbuffers to swapchain and to local reference
		{
			CD3DX12_CPU_DESCRIPTOR_HANDLE rtv_handle(m_rtv_desc_heap->GetCPUDescriptorHandleForHeapStart());

			for (int32_t i = 0; i != m_frame_cnt; ++i)
			{
				ID3D12Resource* backbuffer;

				check(m_swapchain->GetBuffer(i, IID_PPV_ARGS(&backbuffer)));

				m_device->CreateRenderTargetView(backbuffer, nullptr, rtv_handle);

				m_backbuffers[i] = backbuffer;

				rtv_handle.Offset(m_rtv_desc_size);
			}
		}

		//Map backbuffers for access by cuda
		map_backbuffers_for_cuda();

		//Create command allocators for each backbuffer, to allow cycling through them
		{
			for (int32_t i = 0; i != m_frame_cnt; ++i)
				check(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS((m_cmd_allocators + i))));
		}

		//Create command list, which initially uses the allocator corresponding to the swapchain's current backbuffer
		{
			check(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_cmd_allocators[m_curr_frame], nullptr, IID_PPV_ARGS(&m_cmd_list)));

			check(m_cmd_list->Close());
		}

		//Create a shared D3D12 fence which can also be accessed by cuda
		m_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_fence));

		//Map the previously created fence for access by cuda
		{
			check(m_device->CreateSharedHandle(m_fence, nullptr, GENERIC_ALL, nullptr, &m_cu_fence_shared_handle));

			cudaExternalSemaphoreHandleDesc fence_desc{};
			fence_desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
			fence_desc.handle.win32.handle = m_cu_fence_shared_handle;
			fence_desc.flags = 0;

			check(cudaImportExternalSemaphore(&m_cu_fence, &fence_desc));
		}

		//Create an awaitable event handle for the fence
		{
			m_fence_event = CreateEventW(nullptr, false, false, nullptr);

			if (!m_fence_event)
				panic("Could not create fence-event");
		}

		//Release temporary COM-Interfaces
		dxgi_factory->Release();

		//Indicate initialization has finished
		m_is_initialized = true;

		//Finito output
		och::print("Finished in {}\n", initialization_timer.read());
	}

	~render_data()
	{
		//Cuda stuff
		unmap_backbuffers_for_cuda();

		cudaDestroyExternalSemaphore(m_cu_fence);

		CloseHandle(m_cu_fence_shared_handle);


		//D3D12/DXGI stuff
		m_fence->Release();

		m_cmd_list->Release();

		for (int i = 0; i != m_frame_cnt; ++i)
		{
			m_backbuffers[i]->Release();

			m_cmd_allocators[i]->Release();
		}

		m_rtv_desc_heap->Release();

		m_swapchain->Release();

		m_cmd_queue->Release();

		m_device->Release();

#ifdef GRAPHICS_DEBUG
		m_debug_interface->Release();
#endif // GRAPHICS_DEBUG


		CloseHandle(m_fence_event);
	}

	/*////////////////////////////////////////////////////////////////////////*/
	/*/////////////////////////RENDERING AND HELPERS//////////////////////////*/
	/*////////////////////////////////////////////////////////////////////////*/

	void run()
	{
		och::print("Running...\n");

		ShowWindow(m_window, SW_SHOW);

		MSG msg{};

		while (GetMessageW(&msg, m_window, 0, 0) > 0)
		{
			TranslateMessage(&msg);
			DispatchMessageW(&msg);
		}

		wait_for_gpu();

		och::print("Finished\n");
	}

	void unmap_backbuffers_for_cuda()
	{
		for (int32_t i = 0; i != m_frame_cnt; ++i)
		{
			//This call also invalidates all arrays and surfaces created from this external memory
			check(cudaDestroyExternalMemory(m_cu_external_memory_handles[i]));

			CloseHandle(m_cu_backbuffer_shared_handles[i]);
		}
	}

	void map_backbuffers_for_cuda()
	{
		for (int32_t i = 0; i != m_frame_cnt; ++i)
		{
			check(m_device->CreateSharedHandle(m_backbuffers[i], nullptr, GENERIC_ALL, nullptr, (m_cu_backbuffer_shared_handles + i)));

			D3D12_RESOURCE_DESC buffer_desc = m_backbuffers[i]->GetDesc();

			D3D12_RESOURCE_ALLOCATION_INFO buffer_info = m_device->GetResourceAllocationInfo(0, 1, &buffer_desc);

			cudaExternalMemoryHandleDesc cu_handle_desc{};
			cu_handle_desc.flags = cudaExternalMemoryDedicated;
			cu_handle_desc.handle.win32.handle = m_cu_backbuffer_shared_handles[i];
			cu_handle_desc.size = buffer_info.SizeInBytes;
			cu_handle_desc.type = cudaExternalMemoryHandleTypeD3D12Resource;

			check(cudaImportExternalMemory((m_cu_external_memory_handles + i), &cu_handle_desc));

			cudaExternalMemoryMipmappedArrayDesc cu_arr_desc{};

			cu_arr_desc.offset = 0;
			cu_arr_desc.formatDesc = cudaCreateChannelDesc<uchar4>();
			cu_arr_desc.extent = make_cudaExtent(m_window_width, m_window_height, 0);
			cu_arr_desc.flags = cudaArraySurfaceLoadStore;
			cu_arr_desc.numLevels = 1;

			//This does not have to be stored for deletion, as it is automatically invalidated by calling cudaDestroyExternalMemory anyways
			cudaMipmappedArray_t cu_mipmapped_array;

			check(cudaExternalMemoryGetMappedMipmappedArray(&cu_mipmapped_array, m_cu_external_memory_handles[i], &cu_arr_desc));

			//This does not have to be stored for deletion, as it is automatically invalidated by calling cudaDestroyExternalMemory anyways
			cudaArray_t cu_array;

			check(cudaGetMipmappedArrayLevel(&cu_array, cu_mipmapped_array, 0));

			cudaResourceDesc res_desc{};

			res_desc.resType = cudaResourceTypeArray;
			res_desc.res.array.array = cu_array;

			check(cudaCreateSurfaceObject(m_cu_surfaces + i, &res_desc));
		}
	}

	void update_rtvs()
	{
		CD3DX12_CPU_DESCRIPTOR_HANDLE rtv_handle(m_rtv_desc_heap->GetCPUDescriptorHandleForHeapStart());

		for (int32_t i = 0; i != m_frame_cnt; ++i)
		{
			ID3D12Resource* backbuffer;

			check(m_swapchain->GetBuffer(i, IID_PPV_ARGS(&backbuffer)));

			m_device->CreateRenderTargetView(backbuffer, nullptr, rtv_handle);

			m_backbuffers[i] = backbuffer;

			rtv_handle.Offset(m_rtv_desc_size);
		}

		unmap_backbuffers_for_cuda();

		map_backbuffers_for_cuda();
	}

	//uint64_t signal(ID3D12CommandQueue* queue, ID3D12Fence* fence, uint64_t& fence_value)
	//{
	//	uint64_t value_for_signal = ++fence_value;
	//
	//	check(queue->Signal(fence, value_for_signal));
	//
	//	return fence_value;
	//}

	//void wait_for_fence(ID3D12Fence* fence, uint64_t value_to_await, HANDLE fence_event)
	//{
	//	if (fence->GetCompletedValue() < value_to_await)
	//	{
	//		check(fence->SetEventOnCompletion(value_to_await, fence_event));
	//
	//		WaitForSingleObject(fence_event, INFINITE);
	//	}
	//}

	void wait_for_gpu()
	{
		++m_fence_values[m_curr_frame];

		check(m_cmd_queue->Signal(m_fence, m_fence_values[m_curr_frame]));

		if (m_fence->GetCompletedValue() < m_fence_values[m_curr_frame])
		{
			check(m_fence->SetEventOnCompletion(m_fence_values[m_curr_frame], m_fence_event));

			WaitForSingleObject(m_fence_event, INFINITE);
		}
	}

	void update()
	{
		static uint64_t elapsed_frames = 0;
		static och::time last_report_time = och::time::now();

		++elapsed_frames;

		och::time now = och::time::now();

		if ((now - last_report_time).seconds())
		{
			wchar_t buf[64]{};
			
			wchar_t* curr = buf + 62;//Leave last char null
			
			*curr-- = u']';
			*curr-- = u's';
			*curr-- = u'p';
			*curr-- = u'f';
			*curr-- = u' ';
			
			while (elapsed_frames >= 10)
			{
				*curr-- = u'0' + elapsed_frames % 10;
				elapsed_frames /= 10;
			}
			
			*curr-- = u'0' + static_cast<wchar_t>(elapsed_frames);
			
			*curr-- = u'[';
			*curr = u' ';
			
			const wchar_t* prev_title = m_window_title;
			
			int prev_len = 0;
			
			while (prev_title[prev_len])
				++prev_len;
			
			curr -= prev_len;
			
			for (int i = 0; i != prev_len; ++i)
				curr[i] = prev_title[i];
			
			SetWindowTextW(m_window, curr);

			elapsed_frames = 0;

			last_report_time = now;
		}
	}

	void render()
	{
		//WAIT FOR FRAME TO FINISH
		//PRESENT
		//RENDER

		ID3D12CommandAllocator* cmd_allocator = m_cmd_allocators[m_curr_frame];
		ID3D12Resource* backbuffer = m_backbuffers[m_curr_frame];
		
		cmd_allocator->Reset();

		m_cmd_list->Reset(cmd_allocator, nullptr);

		//Render
		CD3DX12_RESOURCE_BARRIER clear_barrier = CD3DX12_RESOURCE_BARRIER::Transition(backbuffer, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);

		m_cmd_list->ResourceBarrier(1, &clear_barrier);

		/*////////////////////////////////////////////////////////////////////////*/
		/*//////////////////////////////////CUDA//////////////////////////////////*/
		/*////////////////////////////////////////////////////////////////////////*/

		dim3 threads_per_block(64, 64);
		dim3 blocks_per_grid((m_window_width + 63) / 64, (m_window_height + 63) / 64);

		//ABGR
		//MY new favourite colour: 0xFF007FFF (ABGR)
		//check(launch_set_surface_to(threads_per_block, blocks_per_grid, draw_color, m_cu_surfaces[m_curr_frame], m_window_width, m_window_height));

		static float z_offset = 0.0F;

		uint2 surface_dim{ m_window_width, m_window_height };
		float3 offset{ 0.0F, 0.0F, z_offset };
		float2 step{ 1.0F / 256.0F, 1.0F / 256.0F };

		z_offset += 1.0F / 2048.0F;

		launch_simplex_3d_surface2d_grayscale_argb(threads_per_block, blocks_per_grid, m_cu_surfaces[m_curr_frame], surface_dim, offset, step, 0);

		cudaDeviceSynchronize();

		/*////////////////////////////////////////////////////////////////////////*/
		/*////////////////////////////////END CUDA////////////////////////////////*/
		/*////////////////////////////////////////////////////////////////////////*/

		//Present
		CD3DX12_RESOURCE_BARRIER present_barrier = CD3DX12_RESOURCE_BARRIER::Transition(backbuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);

		m_cmd_list->ResourceBarrier(1, &present_barrier);

		check(m_cmd_list->Close());

		ID3D12CommandList* const cmd_lists[]{ m_cmd_list };

		m_cmd_queue->ExecuteCommandLists(1, cmd_lists);

		int32_t sync_interval = static_cast<int32_t>(m_vsync & !m_supports_tearing);

		int32_t present_flags = m_supports_tearing && m_vsync ? DXGI_PRESENT_ALLOW_TEARING : 0;

		check(m_swapchain->Present(sync_interval, present_flags));

		//Signal fence on completion of 'Present'
		check(m_cmd_queue->Signal(m_fence, ++m_curr_fence_value));
		m_fence_values[m_curr_frame] = m_curr_fence_value;

		m_curr_frame = m_swapchain->GetCurrentBackBufferIndex();

		//Wait for current buffer to complete. TODO: Could be moved to top of function, to minimize blocking
		if (m_fence->GetCompletedValue() < m_fence_values[m_curr_frame])
		{
			check(m_fence->SetEventOnCompletion(m_fence_values[m_curr_frame], m_fence_event));
		
			WaitForSingleObject(m_fence_event, INFINITE);
		}
	}

	void resize(uint16_t new_width, uint16_t new_height)
	{
		m_window_width = new_width;
		m_window_height = new_height;

		wait_for_gpu();

		for (int32_t i = 0; i != m_frame_cnt; ++i)
		{
			m_backbuffers[i]->Release();

			m_fence_values[i] = m_fence_values[m_curr_frame];
		}

		DXGI_SWAP_CHAIN_DESC swapchain_desc{};

		check(m_swapchain->GetDesc(&swapchain_desc));

		check(m_swapchain->ResizeBuffers(m_frame_cnt, m_window_width, m_window_height, swapchain_desc.BufferDesc.Format, swapchain_desc.Flags));

		m_curr_frame = m_swapchain->GetCurrentBackBufferIndex();

		update_rtvs();
	}

	//TODO: Crashes sporadically
	void set_fullscreen(bool fullscreen)
	{
		if (m_is_fullscreen == fullscreen)
			return;

		m_is_fullscreen = fullscreen;

		if(fullscreen)
		{
			GetWindowRect(m_window, &m_window_rect);

			uint32_t window_style = 0;

			SetWindowLongPtrW(m_window, GWL_STYLE, window_style);

			HMONITOR monitor = MonitorFromWindow(m_window, MONITOR_DEFAULTTONEAREST);

			MONITORINFOEXW mon_info{};

			mon_info.cbSize = sizeof(MONITORINFOEXW);

			GetMonitorInfoW(monitor, &mon_info);

			RECT& mr = mon_info.rcMonitor;

			SetWindowPos(m_window, HWND_TOP, mr.left, mr.top, mr.right - mr.left, mr.bottom - mr.top, SWP_FRAMECHANGED | SWP_NOACTIVATE);

			ShowWindow(m_window, SW_MAXIMIZE);
		}
		else
		{
			SetWindowLongPtrW(m_window, GWL_STYLE, WS_OVERLAPPEDWINDOW);

			SetWindowPos(m_window, HWND_NOTOPMOST, m_window_rect.left, m_window_rect.top, m_window_rect.right - m_window_rect.left, m_window_rect.bottom - m_window_rect.top, SWP_FRAMECHANGED | SWP_NOACTIVATE);

			ShowWindow(m_window, SW_NORMAL);
		}
	}



	/*////////////////////////////////////////////////////////////////////////*/
	/*/////////////////////////////INPUT / OUTPUT/////////////////////////////*/
	/*////////////////////////////////////////////////////////////////////////*/

	void set_key(uint8_t vk) noexcept
	{
		m_keystates[vk >> 6] |= 1ull << (vk & 63);
	}

	void unset_key(uint8_t vk) noexcept
	{
		m_keystates[vk >> 6] &= ~(1ull << (vk & 63));
	}

	void update_mouse_pos(int64_t lparam) noexcept
	{
		m_mouse_x = static_cast<int16_t>(lparam & 0xFFFF);
		m_mouse_y = static_cast<int16_t>((lparam >> 16) & 0xFFFF);
	}

	bool key_is_down(uint8_t vk) const noexcept
	{
		return m_keystates[vk >> 6] & (1ull << (vk & 63));
	}
};

LRESULT CALLBACK window_function(HWND window, UINT msg, WPARAM wp, LPARAM lp)
{
	render_data* rd_ptr = reinterpret_cast<render_data*>(GetWindowLongPtrW(window, GWLP_USERDATA));

	if (!rd_ptr || !rd_ptr->m_is_initialized)
		return DefWindowProcW(window, msg, wp, lp);

	render_data& rd = *rd_ptr;

		switch (msg)
		{
		case WM_PAINT:
			rd.update();
			rd.render();
			break;

		case WM_SIZE:
			RECT wr;
			GetClientRect(window, &wr);
			rd.resize(static_cast<uint16_t>(wr.right - wr.left), static_cast<uint16_t>(wr.bottom - wr.top));
			break;

		case WM_DESTROY:
			PostQuitMessage(0);
			break;

		case WM_KEYDOWN:
		case WM_SYSKEYDOWN:
			switch (wp)
			{
			case och::vk::enter:
				if (!rd.key_is_down(och::vk::alt))
					break;
			case och::vk::f11:
				rd.set_fullscreen(!rd.m_is_fullscreen);
				break;
			case och::vk::escape:
				PostQuitMessage(0);
				
				break;
			case och::vk::key_v:
				rd.m_vsync = !rd.m_vsync;
				break;
			}
			rd.set_key(static_cast<uint8_t>(wp));
			break;

		case WM_KEYUP:
		case WM_SYSKEYUP:
			rd.unset_key(static_cast<uint8_t>(wp));
			break;

		case WM_LBUTTONDOWN:
		case WM_RBUTTONDOWN:
		case WM_MBUTTONDOWN:
		case WM_XBUTTONDOWN:
			rd.update_mouse_pos(lp);
			rd.set_key(static_cast<uint8_t>(msg - (msg >> 1) - (msg == WM_XBUTTONDOWN && (wp & (1 << 16)))));//Figure out key from low four bits of message
			break;

		case WM_LBUTTONUP:
		case WM_RBUTTONUP:
		case WM_MBUTTONUP:
		case WM_XBUTTONUP:
			rd.update_mouse_pos(lp);
			rd.unset_key(static_cast<uint8_t>((msg >> 1) - (msg == WM_XBUTTONUP && (wp & (1 << 16)))));//Figure out key from low four bits of message
			break;

		case WM_MOUSEHWHEEL:
			rd.update_mouse_pos(lp);
			rd.m_mouse_h_scroll += static_cast<int16_t>(wp >> 16);
			break;

		case WM_MOUSEWHEEL:
			rd.update_mouse_pos(lp);
			rd.m_mouse_scroll += static_cast<int16_t>(wp >> 16);
			break;

		case WM_MOUSEMOVE:
			rd.update_mouse_pos(lp);
			break;

		default:
			return DefWindowProcW(window, msg, wp, lp);
		}

		return 0;
}
