import numpy as npy
import scipy as scy
from scipy import signal as ssy
from matplotlib import pyplot as plt
import sklearn as sklrn
from sklearn import mixture
import os
import sys
from tkinter import *
from tkinter import filedialog



def cc_detection(data_channel, template, file_name_base = '', sampling_rate = 10000.0, min_iei = 15.0, peak_search_interval = 10.0, decay_search_interval = 20.0, D_scale = 3.1250000e-1, filter_kernel = [1], clustering_override = True, skip_write = False, internal_output = []):
    #
    # data_channel should be an array of 16-bit integers (straight from the digitizer works ok). *can use 32/64bit floats as well, but read D_scale notes below.
    # template should be a floating point or double precision array of a single event size (it should be as precise as possible, averaged events or fitted events preferably).
    # (template should start at 0pA --- add a template -= template[0] if needed)
    # D_scale for HEKA I channel in pA = 3.125e-1 (1pA/mV) ### (amplifier output range (in V)) / 2^(bits per sample)
    #
    # If you have the real numbers already (floating point) in pA, specify D_scale = 0 , so the scaling does not mess up your data
    # 
    # file_namme_base is a base file name. You may want to define this programmatically. Every result file name will start with this.
    # 
    # sampling_rate is sampling rate in Hz (samples / sec). Note that trace and template (and filter kernel, if used) sampling rates should be equal -  or you may get nonsensical results.
    # min_iei is the smallest latency (in ms) at which discerning distinct, partially overlaid events is possible. Depends on the recording noise level, the sampling rate and shape of the events. Adjust as needed.
    # peak_search_interval & decay_search_interval are the limits (in ms) to look for peak (from start of matched event) and decay (from peak of matched event).
    # filter_kernel expects (nothing or 0 or) a floating point array of the kernel of a FIR filter. Omitting it will not filter the data (ok, it will filter it but with a [1], which does nothing at all), specifying 0 will make a 400Hz low-pass filter for 10kHz sampling rate.
    # clustering_override = True disables the additional Gaussian mixture classifier for low-amplitude rejection (likely noise) --- keep it set to True if you select events at a later stage with more criteria.
    # 
    # skip_write = False will output the full recording with an indicator channel. Specify True if you don't need to check, to save space.
    # internal_output is an array/list intentionally left free for the user to export anything they may want in their code after calling the function. 
    #
    # All output assumes pA for the current recording. If you have a different unit you will need to adjust the names and labels of the graphs. Search for current, pA, I in here and change it to your measured quantity and units.
    #
    # Output takes a lot of space (1 full 16-bit recording for each correlation coefficient cut-off – this is just for monitoring purposes). Keep in mind if low on disk space. It can be disabled by specifying skip_write = True.
    # MKL-enabled numpy is strongly recommended.
    # 

    try:
        mwin = Tk()
        mwin.minsize(350,200)
        mwin.title("cc_detection")
        outputfolder = filedialog.askdirectory()
        print(outputfolder)
        if not os.path.exists(outputfolder+'\\xpsc_results\\'):
            os.makedirs(outputfolder+'\\xpsc_results\\')


        source_directory = outputfolder 
        new_file_path = source_directory + '\\xPSC_results\\' 
        file_path = new_file_path + file_name_base
        print(file_path)
        mwin.destroy()
        mwin.quit()

    except:
        print('file path error - exiting')
        return 1

    if npy.size(data_channel) < 2 * npy.size(template):
        print('recording too short - exiting')
        return 2

    if filter_kernel == 0 :
        fir = ssy.remez(2799,[0,400,420,5000],[1,0],[1,10],10000,'bandpass',2048,256)
        fir = npy.convolve(fir,fir,'full')
    else:
        fir = npy.asfarray(filter_kernel)

    
    ### read rec
    if D_scale == 0:
        scaled_D = data_channel
    else:
        scaled_D = data_channel * D_scale
    
    plt.clf()
    plt.plot(npy.arange(0,npy.size(scaled_D)/sampling_rate, 1.0/sampling_rate), scaled_D)
    plt.xlabel('time (s)')
    plt.ylabel('current (pA)')
    plt.savefig(file_path+'_I.png')
    
    BL_sub = ssy.detrend(scaled_D, axis=-1, type='linear', bp=npy.arange(0,npy.size(scaled_D),int(sampling_rate)) )
    plt.clf()
    plt.plot(BL_sub, color='#b0b0b0')
    plt.title('baseline = 0')
    plt.savefig(file_path+'_baseline fix.png')

    f_sc_data = ssy.oaconvolve(BL_sub, fir, 'valid')
    plt.plot(f_sc_data, color = '#402060')
    plt.title('low-pass')
    plt.xlabel('time (samples)')
    plt.ylabel('current (pA)')
    plt.savefig(file_path+'_filtered.png')
    
    if D_scale >0:
        int_f_sc_data = npy.round(f_sc_data/D_scale, 0).astype('int16')
    else: ### default to mV or 1 unit per mV if the scale  was already precalculated in the data channel
        int_f_sc_data = npy.round(f_sc_data/3.125e-1, 0).astype('int16')

    def correl(array1,vector2, winlength):
        unitkernel = npy.ones(winlength)
        array1sq = npy.square(array1)
        array1ss = ssy.oaconvolve(array1sq, unitkernel,'valid')
        array1Eucl = npy.sqrt(array1ss)
        
        vector2Eucl = npy.sqrt(npy.sum(npy.square(vector2)))
    
        cc = npy.correlate(array1,vector2) / (array1Eucl * vector2Eucl)     # normalized to -1..+1
        return cc


    f_sc_data = f_sc_data.astype('float32')
    

    print('cross-correlation:')

    template_window = npy.size(template)
    corr = correl(f_sc_data,template,template_window)

    #d_fscI = npy.gradient(f_sc_data) # first derivative if you need it..


    corr_positive = (corr+npy.abs(corr))/2

    t_start = 0
    t_stop = min(npy.size(corr), npy.size(f_sc_data))
    t_step = 1.0/ sampling_rate

    t = npy.arange(t_start*t_step,t_stop*t_step,t_step)
    
    plt.clf()
    plt.hist(corr,color='#8020a0',bins=20)
    plt.savefig(file_path+'_ccoef.png')
    
    def extract(c_cutoff):              ### NEEDS TO BE NESTED!! no array in the input args!
        localpeaks = ssy.argrelextrema(corr_positive,npy.greater)
        localpeaksampl = corr_positive[localpeaks]
        ccoeff_cutoff = c_cutoff
        contamination_limit = int(sampling_rate * min_iei / 1000) #in samples
        selpeaks = localpeaksampl>ccoeff_cutoff

        scan_number = 1

        removed_elements = 0

        for i in range(1,npy.size(selpeaks)):
            if (localpeaks[0][i] - localpeaks[0][i-1])<contamination_limit and selpeaks[i] == True:
                selpeaks[i] = False
                removed_elements += 1
            else:
                pass
        print('scan #'+ str(scan_number) + ' , eliminated ' + str(removed_elements) + ' triggers')
        scan_number+=1        


        while removed_elements>1 :
            removed_elements = 0
            for i in range(1,npy.size(selpeaks)):
                if (localpeaks[0][i] - localpeaks[0][i-1])<contamination_limit and selpeaks[i] == True:
                    selpeaks[i] = False
                    removed_elements += 1
                else:
                    removed_elements += 0
            print('scan #'+ str(scan_number) + ' , eliminated ' + str(removed_elements) + ' triggers')
            scan_number+=1        
            if removed_elements == 0:
                break

        flocpeaks = localpeaks[0][:]
        fflocpeaks = npy.array([])
        for k in range(npy.size(flocpeaks)):
            if selpeaks[k] == True:
                fflocpeaks = npy.append(fflocpeaks,flocpeaks[k])

        plt.clf()
        plt.hist((1000.0/sampling_rate)*npy.diff(fflocpeaks),bins = 80, cumulative = 1, histtype='step', density = 1, color='purple')
        plt.xlabel('IEI (ms)')
        plt.ylabel('observations (rel. fraction)')
        plt.title('IEI - cumulative distribution\ncorr. coeff. > ' + str(npy.round(ccoeff_cutoff, 2)))
        plt.savefig(file_path+'_IEI_CD_cc_'+str(npy.round(ccoeff_cutoff,2))+'.png')

        print('\n'+str(npy.size(fflocpeaks)) + ' events at c.coeff>'+str(npy.round(ccoeff_cutoff,2)))
        print(str(npy.round(npy.size(f_sc_data)/sampling_rate, 3)) + 's total duration')
        print('\nraw frequency = ' + str(npy.round(sampling_rate * npy.size(fflocpeaks)/npy.size(f_sc_data),2)) + ' events/s')

        plt.clf()
        for l in range(npy.size(fflocpeaks)):
            currentpeak = int(fflocpeaks[l])
            try:
                plt.plot((1000.0/sampling_rate)*sampling_rate*t[0:template_window], f_sc_data[currentpeak:currentpeak+template_window])
            except:
                fflocpeaks = fflocpeaks[:l]
                break

        plt.title('all detected events\ncorr. coeff. > ' + str(ccoeff_cutoff))
        plt.xlabel('time (ms)')
        plt.ylabel('current (pA)')
        plt.savefig(file_path+'_events_cc_'+str(npy.round(ccoeff_cutoff,2))+'.png')

        return(fflocpeaks)



    nevents = []
    amplrange = []
    minrange = []
    maxrange = []
    sdrange = []
    evwvmin = []
    evwvmax = []
    evmedian = []
    evwvp10 = []
    evwvp90 = []
    falsepos = []
    nevents = []
    rt20 = []
    rt80 = []
    rt2080 = []
    dt20 = []
    dt80 = []
    dt2080 = []
    hw = []

    trcampl = []

    peak_interval = int(sampling_rate * peak_search_interval / 1000)
    decay_interval = int(sampling_rate * decay_search_interval / 1000)
    if decay_interval > (template_window - peak_interval):
        decay_interval = template_window - peak_interval
        print('decay time limit set to ' + str(round(decay_interval * 1000 / sampling_rate, 3)) + ' ms (template duration limit)')

    if scy.integrate.trapz(template) < 0 :
        template_sign = -1
    else:
        template_sign = +1

    fulltracesd = npy.std(f_sc_data)

    for ccr_th in npy.arange(0.5,1,0.05):
        linconcat = npy.array([])
        eventwv = npy.zeros((template_window - 1))
        evwvmin = []
        evwvmax = []
        evwvrange = []
        fitfail = 0
        nevents_ = 0
        trcampl = []
        rt20 = []
        rt80 = []
        rt2080 = []
        dt20 = []
        dt80 = []
        dt2080 = []
        hw = []

        if skip_write == False:
            prooffile = open(file_path+'_scan_'+str(npy.round(ccr_th,2))+'.dat', 'wb')
        scan_array = npy.zeros(npy.size(f_sc_data))


        detected = extract(ccr_th)

        if npy.size(detected) == 0:
            print('No events @ '+ str(npy.round(ccr_th,2)))
            amplrange.append(amplrange[-1])
            minrange.append(minrange[-1])
            maxrange.append(maxrange[-1])
            evwvp10.append(evwvp10[-1])
            evwvp90.append(evwvp90[-1])
            sdrange.append(sdrange[-1])
            evmedian.append(evmedian[-1])
            nevents.append(nevents_)
            falsepos.append(nevents_)

            continue
        

        for evc in detected.tolist():
            evc = int(evc)
            eventwv = f_sc_data[evc:evc+template_window] 
            linconcat = npy.append(linconcat,f_sc_data[evc:evc+template_window]) 
            evwvmin.append(npy.min(f_sc_data[evc:evc+template_window])) 
            evwvmax.append(npy.max(f_sc_data[evc:evc+template_window])) 
            evwvrange.append(evwvmin[-1] - evwvmax[-1])

            
            trc0 = f_sc_data[evc]
            if template_sign == -1:
                trc100 = npy.min(f_sc_data[evc:evc+peak_interval])
            else:
                trc100 = npy.max(f_sc_data[evc:evc+peak_interval])
            trc20t = evc
            trc50t = evc
            trc80t = evc+peak_interval
            trc100t = peak_interval
            try:
                for trc_k in range(evc,evc+peak_interval):
                    if f_sc_data[trc_k] == trc100:
                        trc100t = trc_k
                        break


                for trc_k in range(evc,evc+peak_interval):
                    if template_sign * (f_sc_data[trc_k] - (trc0 + 0.20*(trc100-trc0))) >= 0:
                        trc20t = trc_k
                        break

                for trc_k in range(evc,evc+peak_interval):
                    if template_sign * (f_sc_data[trc_k] - (trc0 + 0.50*(trc100-trc0))) >= 0:
                        trc50t = trc_k
                        break

                for trc_k in range(trc20t,trc100t):
                    if template_sign * (f_sc_data[trc_k] - (trc0 + 0.80*(trc100-trc0))) >= 0:
                        trc80t = trc_k
                        break
            except:
                print('rise-time not found/imprecise for event @ t = '+str(evc/sampling_rate)+ ' s')
                pass

            rt20.append(trc20t*1000.0/sampling_rate)
            rt80.append(trc80t*1000.0/sampling_rate)
            rt2080.append((trc80t-trc20t)*1000.0/sampling_rate) ## in ms
            trcampl.append(trc100 - trc0)

            #decays
            
            tdc0 = trc100
            tdc0t = trc100t # peak is the start of the decay time calculation
            tdc100 = trc0
            tdc20t = tdc0t 
            tdc50t = tdc0t
            tdc80t = tdc0t+decay_interval
            tdc100t = evc+template_window

            try:
                for tdc_k in range(tdc0t,tdc0t+decay_interval):
                    if template_sign * (f_sc_data[tdc_k] - trc0) <= 0:
                        tdc100t = tdc_k
                        break
                
                for tdc_k in range(tdc0t,tdc0t+decay_interval):
                    if template_sign * (f_sc_data[tdc_k] - (tdc100 + 0.80*(tdc0-tdc100))) <= 0:
                        tdc20t = tdc_k
                        break

                for tdc_k in range(tdc0t,tdc0t+decay_interval):
                    if template_sign * (f_sc_data[tdc_k] - (tdc100 + 0.50*(tdc0-tdc100))) <= 0:
                        tdc50t = tdc_k
                        break

                for tdc_k in range(tdc20t,tdc100t):
                    if template_sign * (f_sc_data[tdc_k] - (tdc100 + 0.20*(tdc0-tdc100))) <= 0:
                        tdc80t = tdc_k
                        break
            except:
                print('decay-time not found for event @ t = '+str(evc/sampling_rate)+ ' s --set to measured maximum')
                pass

            dt20.append(tdc20t*1000.0/sampling_rate)
            dt80.append(tdc80t*1000.0/sampling_rate)
            dt2080.append((tdc80t-tdc20t)*1000.0/sampling_rate) ## in ms for 10kHz sampling rate
            hw.append((tdc50t-trc50t)*1000.0/sampling_rate)

            nevents_ = nevents_ + 1


        if nevents_ == 0 :
            amplrange.append(amplrange[-1])
            minrange.append(minrange[-1])
            maxrange.append(maxrange[-1])
            evwvp10.append(evwvp10[-1])
            evwvp90.append(evwvp90[-1])
            sdrange.append(sdrange[-1])
            evmedian.append(evmedian[-1])
            nevents.append(nevents_)
            falsepos.append(nevents_)
        else:
            amplrange.append(npy.average(npy.asarray(evwvrange)))
            minrange.append(npy.min(npy.asarray(evwvrange)))
            maxrange.append(npy.max(npy.asarray(evwvrange)))
            evwvp10.append(npy.percentile(npy.asarray(evwvrange),10))
            evwvp90.append(npy.percentile(npy.asarray(evwvrange),90))
            sdrange.append(template_sign * 2.5 * (npy.size(f_sc_data)/(npy.size(f_sc_data)-nevents_ * (template_window - 1))) * (fulltracesd - nevents_ * (template_window - 1) * npy.std(linconcat)/npy.size(f_sc_data)))
            evmedian.append(npy.median(npy.asarray(evwvrange)))
            nevents.append(nevents_)
            falsepos.append(fitfail)
        ev_n = 0
        evc = 0
        ev_f= 0
        eventsrecfile = open(file_path+'_events_t_A_rt_'+str(npy.round(ccr_th,2))+'.csv','w')
        eventsrecfile.write('time (ms);amplitude (pA);rt20-80 (ms);dt20-80 (ms);hw (ms)')
        eventsrecfile.write('\r')
        try:
            singlesd = (npy.size(f_sc_data)/(npy.size(f_sc_data)-nevents_ * (template_window - 1))) * (fulltracesd - nevents_ * (template_window - 1) * npy.std(linconcat)/npy.size(f_sc_data))
        except:
            singlesd = 0.0
        
        cluster_low_mean = 0.0
        cluster_main_mean = 0.0
        cluster_high_mean = 0.0

        cluster_high = []
        cluster_main = []
        cluster_low = []

        vrampl = [] ###amplitude array within the valid range (only mid-lows)
        vrdetected = []

        ### highs first:
        ev_n = 0
        ev_high = 0
        evc = 0
        
        for evc in detected.tolist():
            evc = int(evc)
            #scan_array placeholder
            vrampl.append(trcampl[ev_n])
            vrdetected.append(evc)
            if npy.abs(trcampl[ev_n])>npy.percentile(npy.abs(trcampl),0.95):    # looking for events with amplitudes in the top 5%, and > 5x the average event amplitude
                if npy.abs(trcampl[ev_n])>= 5* npy.abs(npy.mean(trcampl)):      # 2-step if -  slightly faster, I would guess
                    ev_high += 1
                    cluster_high_mean = ((ev_high-1)*cluster_high_mean + trcampl[ev_n])/ev_high
                    cluster_high.append(evc)
                    vrampl.pop(-1)
                    vrdetected.pop(-1)
            ev_n += 1
        
        ### lows next:
        ev_n = 0
        ev_low = 0
        ev_main = 0
        evc = 0

        cluster_main_ampl = []
        cluster_low_ampl = []

        for evc in vrdetected:
            evc = int(evc)
            #scan_array placeholder
            #provisionally append all to the main cluster, then pop them as they get reassigned:
            cluster_main.append(evc)
            ev_main += 1
            #cluster_main_mean = (cluster_main_mean*(ev_main-1)+vrampl[ev_n])/ev_main
            cluster_main_ampl.append(vrampl[ev_n])


            if npy.abs(vrampl[ev_n]) < 0.5 * singlesd:                              # this thingy needs some adjustment. I'm currently using a 0.01x and  lowest 0.5-1%
                if npy.abs(vrampl[ev_n])< npy.percentile(npy.abs(vrampl),0.01):
                    ev_low += 1
                    
                    cluster_low.append(evc)
                    cluster_low_ampl.append(vrampl[ev_n])

                    ### reverting main cluster
                    cluster_main.pop(-1)
                    ev_main -= 1
                    
                    cluster_main_ampl.pop(-1)
            ev_n += 1
        
        if npy.size(cluster_low_ampl) > 0:
            cluster_low_mean = npy.mean(cluster_low_ampl)
        else:
            cluster_low_mean = 0.0
        if npy.size(cluster_main_ampl) > 0:
            cluster_main_mean = npy.mean(cluster_main_ampl)         # in my version, I have replaced the npy.mean(.....) here with npy.percentile(npy.abs(cluster_main_ampl), 0.1) to detect even smaller events
        else:
            if npy.size(amplrange)>0:
                cluster_main_ampl = amplrange[-1]
            else:
                cluster_main_ampl = 10.0

        ### mids-lows 
        
        ### these gaussian mix model specs works fine, but it is very eager to  reject events (put them into the small amplitude category. See above for a potential fix, if you have the same issue. It may be better suited to separating different amplitude events.
        ### n_components = 2 means 2 categories (small/noise and big/valid events)
        ###  (could dial in the rise-time in the future to make it a bit more robust, but for now it's working ok)

        gmr = mixture.GaussianMixture(n_components = 2, covariance_type = 'full', means_init = [[cluster_low_mean],[cluster_main_mean]]).fit(npy.reshape(vrampl,(-1,1)))
        if clustering_override == True:
            clusters = npy.zeros(npy.size(vrampl), 'int64')
        else:
            clusters = gmr.predict(npy.reshape(vrampl,(-1,1)))
        evc = 0
        cluster_main = []
        cluster_low = []
        cluster_main_ampl = []
        cluster_low_ampl = []

        temp_cl1 = []
        temp_cl2 = []
        temp_cl1i = []
        temp_cl2i = []

        ev_n = 0

        for evc in vrdetected:
            evc = int(evc)
            
            if clusters[ev_n] == 0:
                temp_cl1.append(vrampl[ev_n])
                temp_cl1i.append(evc)

            else:
                temp_cl2.append(vrampl[ev_n])
                temp_cl2i.append(evc)

            ev_n += 1


        if npy.size(temp_cl1)*npy.size(temp_cl2) == 0:
            if npy.size(temp_cl1) > 0 :
                cluster_main = temp_cl1i
                cluster_low = []
                cluster_main_ampl = temp_cl1
                cluster_low_ampl = []
            else:
                if npy.size(temp_cl2) > 0:
                    cluster_main = temp_cl2i
                    cluster_low = []
                    cluster_main_ampl = temp_cl2
                    cluster_low_ampl = []
                else:
                    print('no events')
                    
                    break
        else:

            if npy.abs(npy.mean(temp_cl1)) < npy.abs(npy.mean(temp_cl2)):
                cluster_main = temp_cl2i
                cluster_low = temp_cl1i
                cluster_main_ampl = temp_cl2
                cluster_low_ampl = temp_cl1

            else:
                cluster_main = temp_cl1i
                cluster_low = temp_cl2i
                cluster_main_ampl = temp_cl1
                cluster_low_ampl = temp_cl2

        
        ev_n = 0
        evc = 0
        ev_f = 0    # left-over from debugging, I think; otherwise unused - can remove

        for evc in detected.tolist():
            evc = int(evc)
            scan_array[evc] = -1
            if  cluster_main.count(evc)>0:
                eventsrecfile.write(str(evc*1000.0/sampling_rate) + ';' + str(trcampl[ev_n]) + ';' + str(rt2080[ev_n]) + ';' + str(dt2080[ev_n]) + ';' + str(hw[ev_n])+'\r')
                scan_array[evc] = 1
                ev_f += 1
            ev_n += 1

        eventsrecfile.close()
        if skip_write == False:
            npy.dstack((int_f_sc_data,scan_array)).astype('int16').tofile(prooffile)
            prooffile.close()
        
        cc_scan_file = open(file_path+'_'+str(nevents_)+'_raw_ALL_corr_'+str(npy.round(ccr_th,2))+'.dat','wb')
        linconcat.astype('float32').tofile(cc_scan_file)
        cc_scan_file.close()

        print(str(npy.round(ccr_th,2)) +' : '+ str(nevents_) +' - '+ str(fitfail))
    
    sd1x = npy.array(sdrange)/2.5
    sd3x = sd1x*1.5

    plt.clf()
    plt.semilogy(npy.arange(0.5,1,0.05)[:len(nevents)],(nevents), 'b', npy.arange(0.5,1,0.05)[:len(nevents)],falsepos,'r')
    plt.xlabel('correlation coeff')
    plt.ylabel('# events')
    plt.savefig(file_path+'_xPSC_detection.png')
    
    plt.clf()
    plt.plot(npy.arange(0.5,1,0.05)[:len(nevents)],amplrange, 'm',npy.arange(0.5,1,0.05)[:len(nevents)],evwvp10,'#808080',npy.arange(0.5,1,0.05)[:len(nevents)],evwvp90,'#808080',npy.arange(0.5,1,0.05)[:len(nevents)],sdrange, '#8020f0',npy.arange(0.5,1,0.05)[:len(nevents)],sd3x, '#401080',npy.arange(0.5,1,0.05)[:len(nevents)],sd1x, 'r.',npy.arange(0.5,1,0.05)[:len(nevents)],evmedian,'#20c0f0') #,npy.arange(0.05,1,0.05),maxrange,'#000000',npy.arange(0.05,1,0.05),minrange, '#000000',
    plt.xlabel('correlation coeff')
    plt.ylabel('amplitude (pA)')
    plt.savefig(file_path+'_xPSC_descriptives.png')
    
    internal_output.append(0)

    ### return whatever
    return 0





### Apostolis Mikroulis
### apostolos.mikroulis [at] med.lu.se
###
### If you have any problems please try debug mode to see where the issue is.

