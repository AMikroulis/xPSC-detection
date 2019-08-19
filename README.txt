xPSC-detection
Template correlation-based detection of postsynaptic currents.

The purpose of this function is to detect postsynaptic currents in a voltage-clamp recording, based on similarity to a template, allowing for batch processing of recordings.

It is using the correlation coefficient (normalized cross-correlation of the recording and the template) and returns the results for the 0.5 to 0.95 range in 0.05 steps.

Input:
    data_channel should be a numpy array of 16-bit integers (straight from the digitizer works ok). (*)

		template should be a floating point or double precision array of a single event (it should be as precise as possible). If you put multiple events in succession you may have a very bad detection rate.
		
		D_scale for HEKA EPC9 I channel in pA = 3.125e-1, HEKA V channel in mV = 3.125e-2, may vary between amplifiers, so check first.
     
    file_name_base is a base file name. You may want to define this programmatically. All result file names will start with this and append their corresponding name and correlation coefficient cut-off if applicable.
     
    sampling_rate is sampling rate in Hz (samples / sec). Note that trace and template (and filter kernel, if used) sampling rates should be equal.
    
    min_iei is the minimum latency (in samples) at which discerning distinct, partially overlaid events is possible. Depends on the recording noise level. Adjust as needed.
    
    filter_kernel expects (nothing or 0 or) a floating-point array of coefficients for a FIR filter. Omitting it will not filter the data (ok, it will filter it but with a [1]), specifying 0 will make a 400Hz low-pass filter for 10kHz sampling rate.
    
(*) If you have the real numbers already (floating point) in pA, specify D_scale = 0 , so the scaling does not mess up your data.

Output:
Result files are placed in the folder indicated by the user.
A text file with time points, amplitudes and 20-80% rise-times is generated for further processing.
You may want to tweak the plots to your needs.
internal_output is an array/list intentionally left free for the user to export anything they may want in their code after calling the function. 
All output assumes pA for the current recording. If you have a different unit you will need to adjust the names and labels of the graphs. Search for current, pA, I in here and change it to your measured quantity and units.


Notes:
There is a low-pass FIR filter in the beginning (can be cancelled or replaced at runtime), and a simple baseline correction (detrending – fitting a straight line between several equally-spaced points and subtracting them from the recording).
There is a percentile-based exclusion of high-amplitude events; no way to cancel at runtime – has to be disabled in the code, if not needed.
There is a Gaussian Mixture-based rejection of low-amplitude events. It can be cancelled at runtime. Or repurposed if needed.

Output takes a lot of space (1 full 16-bit recording for each correlation coefficient cut-off – this is just for monitoring purposes). Keep in mind if low on disk space.
MKL-enabled numpy is strongly recommended.
Most of the code was initially written with a 10kHz sampling rate in mind. 
Anaconda versions on Windows 7, 8, 8.1  and 10 run well. Not tested on Linux.

