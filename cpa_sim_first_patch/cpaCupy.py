import cupy as cp
import numpy as np
from ipdb import set_trace

def get_cp_dtype(np_dtype):
    if np_dtype == np.float32:
        return cp.float32
    elif np_dtype == np.float64:
        return cp.float64
    else:
        raise ValueError("Unsupported dtype. Only np.float32 and np.float64 are supported.")

def compute_correlation_matrix(covUnnormalized, trace_var_sum, hypothesis_var_sum, numTraces, dtype=np.float32, debug=False):
    # Cast inputs to the chosen precision
    covUnnormalized = covUnnormalized.astype(dtype)
    trace_var_sum = trace_var_sum.astype(dtype)
    hypothesis_var_sum = hypothesis_var_sum.astype(dtype)
    normalization_factor = dtype(numTraces - 1)

    # Variance computation
    var_samples1 = trace_var_sum / np.sqrt(np.float64(normalization_factor))
    variance1 = (hypothesis_var_sum / np.sqrt(normalization_factor)).flatten()

    # Small epsilon to avoid division by zero
    epsilon = dtype(1e-10)

    # Concatenate and normalize the covariance matrix
    final_cov = covUnnormalized / normalization_factor

    # Square root of variances with epsilon for stability
    sqrt_var_samples1 = (np.maximum(var_samples1, epsilon))
    sqrt_variance1 = (np.maximum(variance1, epsilon))

    # Compute correlation matrix with broadcasting
    tmp = sqrt_variance1[:, np.newaxis] * sqrt_var_samples1
    final_corr = final_cov / tmp
    if debug:
        set_trace()  # Debugging breakpoint, can be removed later


    # Handle NaNs and clip values to [-1, 1]
    final_corr = np.nan_to_num(final_corr) #np.clip(np.nan_to_num(final_corr), -1.0, 1.0)

    return final_corr

# Define the kernel using a workaround for handling float16 types
popc_kernel = cp.ElementwiseKernel(
    in_params='float16 input',
    out_params='float16 output',
    operation=r'''
    // Convert float16 input to __half for compatibility
    __half h_input = __float2half(input);
    // Calculate population count after conversion to ushort
    output = __half2float(__popc(__half_as_ushort(h_input)));
    ''',
    name='popc_kernel'
)
#use inplace trick

hd_kernel = cp.ElementwiseKernel(
    in_params='float16 a, float16 b',
    out_params='float16 output',
    operation=r'''
      // Convert inputs to __half
      __half ha = __float2half(a);
      __half hb = __float2half(b);
      // Extract their 16-bit representations
      unsigned short ba = __half_as_ushort(ha);
      unsigned short bb = __half_as_ushort(hb);
      // XOR to get differing bits
      unsigned short diff = ba ^ bb;
      // Count number of 1 bits (Hamming distance)
      int dist = __popc(diff);
      // Cast to half (float16)
      output = __half2float((float)dist);
    ''',
    name='hd_kernel'
)

#use inplace trick
# Define the element-wise kernel using ElementwiseKernel
hfm_kernel = cp.ElementwiseKernel(
    'float16 inp, float16 value, float16 prevSum',  # Input types
    'float16 res',                      # Output type
    'res = __half2float(__hfma(__float2half(inp), __float2half(value), __float2half(prevSum)));',                  # The operation to perform
    'hfm'                              # The kernel name
)

def CPA(traceMatrix, inputs, candidates, prevSums, leakage_model, batch_size=200, dtype=np.float32, debug=False):
    """
    Preprocesses the power and hypothesis matrices for correlation analysis.
    This function works in either float32 or float64 mode as specified by the dtype argument.

    Args:
        traceMatrix (numpy.ndarray): Matrix of power traces with shape (n_traces, n_samples).
        inputs (numpy.ndarray): 1D array used for hypothesis computation.
        candidates (numpy.ndarray): 2D array of candidate hypotheses.
        prevSums (numpy.ndarray): 1D array for the in-place HW trick.
        batch_size (int): Number of hypothesis rows to process at a time.
        dtype (data-type): Either np.float32 or np.float64.

    Returns:
        numpy.ndarray: The computed correlation matrix.
    """
    numCandidates = candidates.shape[0]
    numTraces = traceMatrix.shape[0]
    cp_dtype = get_cp_dtype(dtype)

    # Transfer data to GPU using the proper CuPy dtype for inputs to the kernels (float16)
    # The kernels still use float16, so we convert those arrays to float16.
    inputs_gpu = cp.asarray(inputs)
    candidates_gpu = cp.asarray(candidates)
    prevSums_gpu = cp.asarray(prevSums)

    # Center the trace matrix in the desired precision and then transfer to GPU
    centered = traceMatrix.astype(np.float64) - np.mean(traceMatrix, axis=0, keepdims=True).astype(dtype)
    traceMatCentered_gpu = cp.asarray(centered, dtype=cp.float64)

    # Compute the norm (L2) of the centered trace matrix (variance per sample)
    trace_var_sum = cp.linalg.norm(traceMatCentered_gpu, ord=2, axis=0)
    
    traceMatCentered_gpu = traceMatCentered_gpu.astype(cp_dtype)
    
    # Prepare an array to accumulate hypothesis variances
    hypothesis_var_sum = cp.zeros(numCandidates, dtype=cp_dtype)

    lst = []

    # Process candidates in batches
    for start in range(0, numCandidates, batch_size):
        end = min(start + batch_size, numCandidates)
        
        # Get current batch of hypotheses (as float16 for GPU kernel)
        candidatesBatch_gpu = candidates_gpu[start:end, :]

        # Compute hypothesis batch on the GPU using the HW in-place trick
        hypothesisBatch_gpu = hfm_kernel(candidatesBatch_gpu, inputs_gpu.flatten(), prevSums_gpu)
        
        # Apply the popcount kernel in-place
        if leakage_model == 'HW':
            popc_kernel(hypothesisBatch_gpu, hypothesisBatch_gpu)
        elif leakage_model == 'HD':
            hd_kernel(prevSums_gpu, hypothesisBatch_gpu, hypothesisBatch_gpu)
        
        # Convert hypothesis batch to the desired precision for further computation
        hypothesisBatch_gpu = hypothesisBatch_gpu.astype(cp_dtype)
        if debug:
            set_trace()

        # Center the hypothesis batch by subtracting the mean along axis 1
        mean_gpu = cp.mean(hypothesisBatch_gpu, axis=1, keepdims=True)
        cp.subtract(hypothesisBatch_gpu, mean_gpu, out=hypothesisBatch_gpu)

        # Calculate the norm (L2) of each hypothesis row
        hypothesis_var_sum[start:end] = cp.linalg.norm(hypothesisBatch_gpu, ord=2, axis=1)

        # Compute the covariance batch on the GPU (matrix multiplication)
        covBatch_gpu = cp.dot(hypothesisBatch_gpu, traceMatCentered_gpu)
        if debug:
            set_trace()
        cp.cuda.Stream.null.synchronize()

        # Move the covariance batch to CPU memory
        lst.append(covBatch_gpu.get())

        # Delete temporary variables and synchronize GPU stream
        del candidatesBatch_gpu, hypothesisBatch_gpu, mean_gpu, covBatch_gpu
        cp.cuda.Stream.null.synchronize()

    del traceMatCentered_gpu

    # Concatenate all covariance batches from each batch iteration
    covUnnormalized = np.concatenate(lst, axis=0)
    # Compute the final correlation matrix using the specified precision
    final_corr = compute_correlation_matrix(covUnnormalized, trace_var_sum.get(), hypothesis_var_sum.get(), numTraces, dtype, debug)
    return final_corr