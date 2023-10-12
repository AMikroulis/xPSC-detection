### xPSC-detection

<i> Template correlation-based detection of postsynaptic currents.</i>

The purpose of this function is to detect postsynaptic currents in a voltage-clamp recording, based on similarity to a template, allowing for batch processing of recordings.

It is using the correlation coefficient (normalized cross-correlation of the recording and the template) and returns the results for the 0.5 to 0.95 range in 0.05 steps.



#### Input:

<b>data_channel</b> should be a numpy array of 16-bit integers (straight from the digitizer works ok). <b>(*)</b>
	
<b>template</b> should be a floating point or double precision array of a single event (it should be as precise as possible). If you put multiple events in succession you may have a very bad detection rate.
		
<b>D_scale</b> for HEKA EPC9 I channel in pA = 3.125e-1, may vary between amplifiers, so check first.
     
<b>file_name_base</b> is a base file name. You may want to define this programmatically. All result file names will start with this and append their corresponding name and correlation coefficient cut-off if applicable.
     
<b>sampling_rate</b> is sampling rate in Hz (samples / sec). Note that trace and template (and filter kernel, if used) sampling rates should be equal.
    
<b>min_iei</b> is the minimum latency (in ms) at which discerning distinct, partially overlaid events is possible. Depends on the recording noise level. Adjust as needed.
    
<b>peak_search_interval</b> & <b>decay_search_interval</b> are the limits (in ms) to look for peak </i>(from start of matched event)</i> and decay <i>(from peak of matched event)</i>.

<b>filter_kernel</b> expects (nothing or 0 or) a floating-point array of coefficients for a FIR filter. Omitting it will not filter the data (ok, it will filter it but with a [1]), specifying 0 will make a 400Hz low-pass filter for 10 kHz sampling rate.
    
<b>clustering_override</b> = True disables the additional Gaussian mixture classifier for low-amplitude rejection (likely noise) --- keep it set to True if you select events at a later stage with more criteria.

<i><b>(*)</b> If you have the real numbers already (floating point) in pA, specify D_scale = 0 , so the scaling does not mess up your data.</i>


#### Output:
Result files are placed in the folder indicated by the user.<br/>
A text file with time points, amplitudes, 20-80% rise/decay-times and half-width is generated for further processing.<br/>
You may want to tweak the plots to your needs.<br/>
<b>internal_output</b> is an array/list intentionally left free for the user to export anything they may want in their code after calling the function.<br/>
All output assumes pA for the current recording. If you have a different unit you will need to adjust the names and labels of the graphs.

#### Notes:
There is a low-pass FIR filter in the beginning (can be canceled or replaced at runtime), and a simple baseline correction (detrending – fitting a straight line between several equally-spaced points and subtracting them from the recording).<br/>
There is a percentile-based exclusion of high-amplitude events; no way to cancel at runtime – has to be disabled in the code, if not needed.<br/>
There is a Gaussian Mixture-based rejection of low-amplitude events. It can be canceled at runtime. Or repurposed if needed.<br/>

Output takes a lot of space (1 full 16-bit recording for each correlation coefficient cut-off – this is just for monitoring purposes, and can be disabled by specifying <b>skip_write = True</b>). Keep in mind if low on disk space.<br/>
MKL-enabled numpy is strongly recommended.<br/>
Anaconda versions on Windows 10 (and 7) run well. Not tested on Linux.<br/>

[![DOI](https://zenodo.org/badge/202560114.svg)](https://zenodo.org/badge/latestdoi/202560114)

