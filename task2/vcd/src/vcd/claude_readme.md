I've refactored the code into small, focused functions. Here's what each function is responsible for:

## Core Algorithm Functions:
- **`compute_phi()`** & **`compute_xi()`**: Mathematical functions for VCD
- **`get_pixel_safe()`**: Safe pixel access with boundary checking
- **`compute_vcd_delta_for_pixel()`**: VCD computation for a single pixel
- **`compute_sobel_gradients()`**: Sobel gradient computation for a single pixel

## Iteration and Control Functions:
- **`perform_vcd_iteration()`**: Execute one VCD iteration in parallel
- **`should_continue_vcd()`**: Check VCD convergence criteria
- **`apply_sobel_operator()`**: Apply Sobel to all pixels in parallel

## Image Processing Functions:
- **`load_and_validate_image()`**: Load image file and validate format
- **`allocate_double_arrays()`**: Allocate memory for double arrays
- **`convert_to_double_parallel()`**: Convert uint8 to double in parallel
- **`convert_from_double_parallel()`**: Convert double to uint8 in parallel
- **`write_output_image()`**: Write processed image to file

## High-Level Orchestration Functions:
- **`apply_image_processing()`**: Apply VCD and/or Sobel algorithms
- **`print_parallel_info()`**: Display MPI/OpenMP configuration
- **`vcd_parallel()`** & **`sobel_parallel()`**: Main algorithm entry points

## Benefits of This Structure:

1. **Single Responsibility**: Each function has one clear purpose
2. **Testability**: Individual components can be tested separately
3. **Maintainability**: Easy to modify specific parts without affecting others
4. **Reusability**: Functions can be reused in different contexts
5. **Readability**: The main flow is much clearer to follow
6. **Debugging**: Easier to isolate and fix issues in specific functions

The code now follows clean architecture principles while maintaining the same parallel performance characteristics as the original implementation.