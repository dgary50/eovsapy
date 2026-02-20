'''
   Module for plotting the median of the front-end RF detector voltages
   from the stateframe SQL database, as a crude flare monitor'''
#
# Runtime note (DPP cron task):
#   This code is executed on DPP by cron to create flare monitor web plots in
#   /common/webplots/flaremon every 10 minutes from 13:00-02:00 UT.
#
#   Cron entry:
#   2,12,22,32,42,52 0,1,2,13,14,15,16,17,18,19,20,21,22,23 * * * \
#   touch /data1/TPCAL/LOG/FLM$(date +\%Y\%m\%d).log; \
#   /bin/bash /home/user/test_svn/shell_scripts/flare_monitor.sh \
#   >> /data1/TPCAL/LOG/FLM$(date +\%Y\%m\%d).log 2>&1
#
#   Wrapper script: /home/user/test_svn/shell_scripts/flare_monitor.sh
#   #! /bin/bash
#   export PYTHONPATH=/home/user/test_svn/python:/common/python/current:/common/python:/common/python/packages/pipeline
#   /common/anaconda2/bin/python /common/python/current/flare_monitor.py
#
# History:
#   2014-Dec-20  DG
#      First written.
#   2014-Dec-21  DG
#      Added annotation and information about source.
#   2014-Dec-22  DG
#      Cleaned up error handling so that procedure will work as cron job.
#   2014-Dec-24  DG
#      Added printing of date, for cron log file.
#   2014-Dec-26  DG
#      Fix bug when there are no gaps in scan_off_times.  Also set xlim to 15-24 h
#   2015-Feb-20  DG
#      Add v38/v39 stateframe table update.
#   2015-Mar-10  DG
#      Add v39/v42 stateframe table update.
#   2015-Mar-29  DG
#      Add v42/v45 stateframe table update.
#   2015-Mar-31  DG
#      Add v45/v46 stateframe table update.
#   2015-Apr-02  DG
#      Finally made this version independent!  Added calls to new routine
#      dbutil.find_table_version().
#   2015-May-29  DG
#      Converted from using datime() to using Time() based on astropy.
#   2015-Jul-04  DG
#      Added xdata_display() routine and plotting of cross-correlation
#      amplitude in files named /common/webplots/flaremon/XSP*.png
#   2015-Aug-30  DG
#      Added flaremeter() routine to calculate medians of cross-correlation
#      amplitudes across all baselines, polarizations, and frequencies. Added
#      code to plot this median information on flare_monitor plot.  Also,
#      extend timerange of plots to 13 UT on current day to 02 UT on next day.
#      A new series of binary files are created containing the flaremeter
#      information for each day, currently in /common/webplots/flaremon/flaremeter/.
#   2015-Sep-06  DG
#      Added code in xdata_display and __main__ to read fdb files into next 
#      UT day, so that scans that extend past 24 UT are fully plotted.
#   2015-Sep-07  DG
#      Attempt to fix bug in extending date past 24 UT
#   2016-Jun-30  DG
#      Update to work with 16-ant-correlator data and new routine read_idb()
#      Also does correct scaling of x-corr level in case of extraneous inf
#   2016-Jul-15  DG
#      Add sk_flag to xdata display.
#   2016-Aug-04  DG
#      After update of numpy, my medians no longer worked.  Changed to nanmedian.
#   2017-Mar-20  DG
#      Changes to get this working for 300 MHz correlator.  Skip SK flagging, skip
#      call to flaremeter() [just takes too long!], and skip get_history() call.
#   2017-Apr-06  DG
#      Changes to put identifying text on plot for more types of calibration.
#   2017-Aug-11  DG
#      Changes to rationalize the handling of the scan start-stop information in
#      flare_monitor(), and tweaks to the line plot.
#   2017-Sep-08  DG
#      Fixed small bug that caused crash on error return from xdata_display()
#   2021-Nov-13  DG
#      Added RT_flare_monitor(), which reads a file RT_latest.txt and creates
#      a summary file RT_<datstr>.txt in the same /data1/RT folder on the dpp.
#   2022-Feb-09  DG
#      Changed the flare_monitor() code to keep data if within 0.1 units of
#      previous measurement instead of stringent 0.01 units.
#   2022-Mar-05  DG
#      Due to a (hopefully) temporary outage of the SQL server, I wrote a
#      new version of get_projects() called get_projects_nosql() that
#      does almost the same thing using the FDB files.  Calling get_projects()
#      with nosql=True also works.  The __main__ routine has some changes
#      to avoid reading from SQL.
#   2022-Jul-26  DG
#      Fix error in projdict['EOS'] returned from get_projects_nosql().
#   2022-Sep-05  DG
#      For some reason, rd_fdb is returning an extra blank file now.
#      I added a check for that in get_projects_nosql().
#   2023-Aug-16  DG
#      Fixed problem with plots not showing up past new-day boundary
#      Note by SY: the problem still exists after the fix.
#    2025-Oct-03  SY
#      Add utc2pst to fix the problem of not showing plots past new-day boundary.
#    2026-Feb-19  SY
#      Added flare-monitor-specific TP calibration for small IDB file lists used
#      in XSP plot generation. Uses temporary files under /data1/dgary/HSO and
#      falls back to nearest earlier SQL TP calibration when current-day TP cal
#      is unavailable.
#      Updated XSP plotting to use TP-calibrated total-power spectrogram in SFU
#      when calibration is available (without creating FITS files). Falls back
#      to cross-correlation amplitude plot when calibration path fails.

import numpy as np
from eovsapy.util import Time, get_idbdir


def RT_flare_monitor():
    ''' Read "real-time" data file once and obtain the median over antenna
        and frequency, appending the result to /data1/RT/RT_<date>.txt.
        
        Returns ut times in plot_date format and median values.
    '''
    tp = np.zeros((100, 16))
    amp = np.zeros((100, 16))
    f = open('/data1/RT/RT_latest.txt', 'r')
    lines = f.readlines()
    f.close()
    mjd = Time.now().mjd - 0.5
    datstr = Time(mjd, format='mjd').iso[:10]
    tstr = lines[1].split(':')[1][1:7]
    tstr = tstr[:2] + ':' + tstr[2:4] + ':' + tstr[4:]
    print(datstr + ' ' + tstr + ' (' + Time.now().iso[:19] + ')')
    t = Time(datstr + ' ' + tstr)

    try:
        # Read last line of output and compare with input. Skip if the same.
        f = open('/data1/RT/RT_' + datstr + '.txt', 'r')
        lastline = f.readlines()[-1]
        f.close()
        lastt = Time(lastline[:19])
        if lastt == t:
            # Times are the same, so do not write to output file
            return
    except:
        pass

    # Read entire file and form median values of TP and AMP        
    for i, line in enumerate(lines[10::10]):
        tp[i] = np.array(line.split()).astype(float)
    for i, line in enumerate(lines[11::10]):
        amp[i] = np.array(line.split()).astype(float)
    tpmed = np.median(tp[:, :13])
    ampmed = np.median(amp[:, :13])

    # Open output file for appending, write median values, and close.
    f = open('/data1/RT/RT_' + datstr + '.txt', 'a')
    f.write(t.iso + ' {:8.4f} {:8.4f}'.format(tpmed, ampmed) + '\n')
    f.close()


def get_projects(t, nosql=False):
    ''' Read all projects from SQL for the current date and return a summary
        as a dictionary with keys Timestamp, Project, and EOS (another timestamp)
    '''
    if nosql == True:
        return get_projects_nosql(t)
    from eovsapy import dbutil
    # timerange is 12 UT to 12 UT on next day, relative to the day in Time() object t
    trange = Time([int(t.mjd) + 12. / 24, int(t.mjd) + 36. / 24], format='mjd')
    tstart, tend = trange.lv.astype('str')
    cnxn, cursor = dbutil.get_cursor()
    mjd = t.mjd
    # Get the project IDs for scans during the period
    verstrh = dbutil.find_table_version(cursor, trange[0].lv, True)
    if verstrh is None:
        print('No scan_header table found for given time.')
        cnxn.close()
        return {}
    query = 'select Timestamp,Project from hV' + verstrh + '_vD1 where Timestamp between ' + tstart + ' and ' + tend + ' order by Timestamp'
    projdict, msg = dbutil.do_query(cursor, query)
    if msg != 'Success':
        print(msg)
        cnxn.close()
        return {}
    elif len(projdict) == 0:
        # No Project ID found, so return data and empty projdict dictionary
        print('SQL Query was valid, but no Project data were found.')
        cnxn.close()
        return {}
    projdict['Timestamp'] = projdict['Timestamp'].astype('float')  # Convert timestamps from string to float
    for i in range(len(projdict['Project'])): projdict['Project'][i] = projdict['Project'][i].replace('\x00', '')
    projdict.update({'EOS': projdict['Timestamp'][1:]})
    projdict.update({'Timestamp': projdict['Timestamp'][:-1]})
    projdict.update({'Project': projdict['Project'][:-1]})
    cnxn.close()
    return projdict


def get_projects_nosql(t):
    ''' Read all projects from FDB file for the current date and return a summary
        as a dictionary with keys Timestamp, Project, and EOS (another timestamp)
    '''
    from eovsapy import dump_tsys as dt
    # timerange is 12 UT to 12 UT on next day, relative to the day in Time() object t
    trange = Time([int(t.mjd) + 12. / 24, int(t.mjd) + 36. / 24], format='mjd')
    tstart = t.iso[2:10].replace('-', '') + '120000'
    t2 = Time(int(t.mjd) + 1, format='mjd')
    tend = t2.iso[2:10].replace('-', '') + '120000'
    fdb = dt.rd_fdb(t)
    fdb2 = dt.rd_fdb(t2)
    if fdb == {}:
        # No FDB file found, so return empty project dictionary
        print('No Project data [FDB file] found for the given date.')
        return {}
    if fdb2 == {}:
        pass
    else:
        #  Concatenate the two dicts into one
        if fdb['FILE'][-1] == '':
            fdb = dict([(k, np.concatenate((fdb.get(k, [])[:-1], fdb2.get(k, [])))) for k in set(fdb) | set(fdb2)])
        else:
            fdb = dict([(k, np.concatenate((fdb.get(k, []), fdb2.get(k, [])))) for k in set(fdb) | set(fdb2)])
            # Get "good" indexes for times between 12 UT on date and 12 UT on next date
    gidx, = np.where(np.logical_and(fdb['SCANID'] > tstart, fdb['SCANID'] < tend))
    scanid, idx = np.unique(fdb['SCANID'][gidx], return_index=True)
    sidx = gidx[idx]  # Indexes into fdb for the start of each scan
    eidx = np.concatenate((sidx[1:] - 1, np.array([gidx[-1]])))  # Indexes into fdb for the end of each scan
    # Get the project IDs for scans during the period
    projdict = {'Timestamp': fdb['ST_TS'][sidx].astype(float),
                'Project': fdb['PROJECTID'][sidx],
                'EOS': fdb['EN_TS'][eidx].astype(float)}
    return projdict


def flare_monitor(t):
    ''' Get all front-end power-detector voltages for the given day
        from the stateframe SQL database, and obtain the median of them, 
        to use as a flare monitor.
        
        Returns ut times in plot_date format and median voltages.
    '''
    from eovsapy import dbutil
    # timerange is 12 UT to 12 UT on next day, relative to the day in Time() object t
    trange = Time([int(t.mjd) + 12. / 24, int(t.mjd) + 36. / 24], format='mjd')
    tstart, tend = trange.lv.astype('str')
    cnxn, cursor = dbutil.get_cursor()
    mjd = t.mjd
    verstr = dbutil.find_table_version(cursor, tstart)
    if verstr is None:
        print('No stateframe table found for given time.')
        cnxn.close()
        return tstart, [], {}
    query = 'select Timestamp,Ante_Fron_FEM_HPol_Voltage,Ante_Fron_FEM_VPol_Voltage from fV' + verstr + '_vD15 where timestamp between ' + tstart + ' and ' + tend + ' order by timestamp'
    data, msg = dbutil.do_query(cursor, query)
    if msg != 'Success':
        print(msg)
        cnxn.close()
        return tstart, [], {}
    for k, v in data.items():
        data[k].shape = (len(data[k]) // 15, 15)
    hv = []
    try:
        ut = Time(data['Timestamp'][:, 0].astype('float'), format='lv').plot_date
    except:
        print('Error for time', t.iso)
        print('Query:', query, ' returned msg:', msg)
        print('Keys:', data.keys())
        print(data['Timestamp'][0, 0])
    hfac = np.median(data['Ante_Fron_FEM_HPol_Voltage'].astype('float'), 0)
    vfac = np.median(data['Ante_Fron_FEM_VPol_Voltage'].astype('float'), 0)
    for i in range(4):
        if hfac[i] > 0:
            hv.append(data['Ante_Fron_FEM_HPol_Voltage'][:, i] / hfac[i])
        if vfac[i] > 0:
            hv.append(data['Ante_Fron_FEM_VPol_Voltage'][:, i] / vfac[i])
    # import pdb; pdb.set_trace()
    flm = np.median(np.array(hv), 0)
    good = np.where(abs(flm[1:] - flm[:-1]) < 0.1)[0]

    projdict = get_projects(t)
    cnxn.close()
    return ut[good], flm[good], projdict


def _get_flaremon_tmp_outpath(t, files):
    ''' Return a temporary output path for calibrated scan files.

        Use the same base location as all-day processing:
        /data1/dgary/HSO/YYYYMMDD/
    '''
    import os
    import tempfile

    datstr = None
    try:
        basename = os.path.basename(files[0])
        if basename.startswith('IDB') and len(basename) >= 11:
            datstr = basename[3:11]
    except:
        datstr = None
    if datstr is None:
        datstr = t.iso[:10].replace('-', '')
    base_outpath = '/data1/dgary/HSO/' + datstr + '/'
    if not os.path.isdir(base_outpath):
        try:
            os.makedirs(base_outpath)
        except:
            pass
    return tempfile.mkdtemp(prefix='flaremon_', dir=base_outpath) + '/'


def _cleanup_flaremon_tmp(outpath):
    import os
    import shutil

    if outpath is None:
        return
    try:
        if os.path.isdir(outpath):
            shutil.rmtree(outpath)
    except:
        pass


def read_idb_calibrated(files, t):
    ''' Read and calibrate a small list of IDB files for flare_monitor().

        Uses pipeline_cal.udb_corr(calibrate=True), which applies TP
        calibration and automatically falls back to the nearest earlier SQL
        calibration (e.g. previous day) when same-day TP calibration does not
        exist.

        Returns:
          out, calibrated
            out: read_idb dictionary
            calibrated: True if TP calibration path succeeded
    '''
    import os
    from eovsapy import read_idb as ri

    outpath = None
    filelist = [str(file) for file in files]
    try:
        from eovsapy import pipeline_cal as pc
        from eovsapy import udb_util as uu
        outpath = _get_flaremon_tmp_outpath(t, filelist)
        # Calibrate each IDB file separately so downstream read_idb keeps the
        # expected ~600-record chunk behavior per file.
        calfiles = []
        skipped = []
        for src in filelist:
            try:
                ok, ok_files, bad_files = uu.valid_miriad_dataset([src])
                if len(ok_files) == 0:
                    skipped.append(src)
                    print('Skipping invalid/incomplete IDB dataset: {0}'.format(src))
                    continue
                calfile = pc.udb_corr([src], calibrate=True, outpath=outpath, attncal=False)
                if isinstance(calfile, str) and calfile != '':
                    calfiles.append(calfile)
                else:
                    skipped.append(src)
                    print('Warning: TP calibration returned invalid output path for {0}: {1}'.format(src, calfile))
            except Exception as ferr:
                skipped.append(src)
                print('Warning: TP calibration failed for {0}: {1}'.format(src, ferr))

        if len(calfiles) > 0:
            print('TP-calibrated scan files: {0} (skipped {1})'.format(len(calfiles), len(skipped)))
            out = ri.read_idb(calfiles)
            if out:
                return out, True
            print('Warning: no data returned from calibrated files.')
        else:
            print('Warning: no calibrated files produced (skipped {0}).'.format(len(skipped)))

    except Exception as err:
        print('Warning: flare monitor TP calibration setup failed: {0}'.format(err))
    finally:
        _cleanup_flaremon_tmp(outpath)

    print('Falling back to limited uncalibrated IDB read to avoid memory blow-up.')
    try:
        valid = []
        for src in filelist:
            try:
                if os.path.isdir(src):
                    valid.append(src)
            except Exception:
                pass
        if len(valid) == 0:
            return {}, False
        nfile = min(len(valid), 6)
        out = ri.read_idb(valid[-nfile:], navg=10)
        return out, False
    except Exception as err2:
        print('Fallback uncalibrated read failed: {0}'.format(err2))
        return {}, False


def _tp_sfu_spectrogram(out):
    ''' Build TP spectrogram (nf x nt) in SFU from calibrated read_idb output.

        Follows the antenna-selection strategy in pipeline_cal.allday_process(),
        but does not write FITS files.
    '''
    from eovsapy import pipeline_cal as pc
    from eovsapy.util import nearest_val_idx

    pdat = np.array(out['p'], copy=True)
    nant, npol, nf, nt = pdat.shape
    if nt == 0 or nf == 0:
        return None

    if out['time'][0] < Time('2025-05-22').jd:
        nsolant = 13
    else:
        nsolant = 15
    nsolant = min(nsolant, nant)

    # Use only data from tracking antennas, as in allday_process().
    try:
        azeldict = pc.get_sql_info(Time(out['time'], format='jd')[[0, -1]])
        tidx = nearest_val_idx(out['time'], azeldict['Time'].jd)
        tracking = azeldict['TrackFlag'].T
        for i in range(nsolant):
            pdat[i, :, :, np.logical_not(tracking[i, tidx])] = np.nan
    except Exception as err:
        print('Warning: tracking mask unavailable for TP SFU plot: {0}'.format(err))

    med = np.nanmean(np.nanmedian(pdat[:nsolant], 3), 1)  # nant, nf
    medspec = np.nanmedian(med, 0)  # nf
    nbest = min(8, nsolant)
    best = np.arange(nbest)
    good = np.where(np.isfinite(medspec))[0]
    if len(good) >= 3:
        try:
            poly = np.polyfit(out['fghz'][good], medspec[good], 2)
            spec = np.polyval(poly, out['fghz']).repeat(nsolant).reshape(nf, nsolant)
            stdev = np.nanstd(med - np.transpose(spec), 1)
            best = stdev.argsort()[:nbest]
        except Exception as err:
            print('Warning: best-antenna selection failed for TP SFU plot: {0}'.format(err))

    # Final median total-power dynamic spectrum in SFU.
    return np.nanmean(np.nanmedian(pdat[best], 0), 0)


def _get_xsp_cache_file(t, files):
    '''Return per-day cache file path for XSP spectrogram products.'''
    import os

    datstr = t.iso[:10].replace('-', '')
    try:
        basename = os.path.basename(files[0])
        if basename.startswith('IDB') and len(basename) >= 11:
            datstr = basename[3:11]
    except Exception:
        pass
    cache_dir = '/tmp/flaremon_cache'
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    return os.path.join(cache_dir, 'XSP_cache_' + datstr + '.npz')


def _load_xsp_cache(cache_file):
    '''Load cached spectrogram payload from npz file.'''
    import os

    if not os.path.exists(cache_file):
        return None
    try:
        with np.load(cache_file, allow_pickle=False) as npz:
            required = ['scanid', 'mode', 'files', 'fghz', 'times_jd', 'pdata']
            for key in required:
                if key not in npz:
                    return None
            out = {
                'scanid': str(npz['scanid'].tolist()),
                'mode': str(npz['mode'].tolist()),
                'files': [str(f) for f in npz['files'].tolist()],
                'fghz': np.asarray(npz['fghz'], dtype=float),
                'times_jd': np.asarray(npz['times_jd'], dtype=float),
                'pdata': np.asarray(npz['pdata'], dtype=float),
            }
            if out['pdata'].ndim != 2:
                return None
            if out['pdata'].shape[0] != len(out['fghz']):
                return None
            if out['pdata'].shape[1] != len(out['times_jd']):
                return None
            return out
    except Exception as err:
        print('Warning: failed to load XSP cache {0}: {1}'.format(cache_file, err))
        return None


def _save_xsp_cache(cache_file, scanid, mode, files, fghz, times_jd, pdata):
    '''Save spectrogram payload cache to npz.'''
    np.savez_compressed(
        cache_file,
        scanid=np.asarray(scanid),
        mode=np.asarray(mode),
        files=np.asarray(files),
        fghz=np.asarray(fghz, dtype=float),
        times_jd=np.asarray(times_jd, dtype=float),
        pdata=np.asarray(pdata, dtype=float),
    )


def _spectrogram_from_out(out, cal_ok, scanid):
    '''Convert read_idb output to a plottable spectrogram payload.'''
    mode = 'xdata'
    pdata = None
    if cal_ok:
        pdata = _tp_sfu_spectrogram(out)
        if pdata is not None:
            print('Successfully created TP spectrogram for scan ID {0}'.format(scanid))
            mode = 'tp'
    if mode != 'tp':
        data = out['x']
        pdata = np.nansum(np.nansum(np.abs(data[0:11, :]), 1), 0)
    return mode, np.asarray(out['fghz'], dtype=float), np.asarray(out['time'], dtype=float), np.asarray(pdata, dtype=float)


def _merge_spectrogram_payload(fghz, old_times, old_pdata, new_times, new_pdata):
    '''Append and de-duplicate spectrogram time samples.'''
    times = np.concatenate((old_times, new_times))
    pdata = np.concatenate((old_pdata, new_pdata), 1)
    order = np.argsort(times)
    times = times[order]
    pdata = pdata[:, order]
    key = np.round(times * 86400.).astype(np.int64)
    _, idx = np.unique(key, return_index=True)
    keep = np.sort(idx)
    return fghz, times[keep], pdata[:, keep]


def xdata_display(t, ax=None):
    ''' Given the time as a Time object, search the FDB file for files
        associated with the scan for that time and create a dynamic spectrogram
        on the axis specified by ax, or on a new plot if no ax. If the requested
        time is more than 10 minutes after the last file of that scan, returns
        None to indicate no plot.
        
        Skip SK flagging [2017-Mar-20 DG]
    '''
    import time, os
    from eovsapy import dump_tsys
    # import get_X_data2 as gd
    from eovsapy import spectrogram_fit as sp
    import astropy.units as u

    utc2pst = -8 * u.hour  # The difference between UTC and pst time. The diff between pst and pdt does not matter here.
    t_pst = t + utc2pst
    fdb = dump_tsys.rd_fdb(t_pst)
    # Get files from next day, in case scan extends past current day
    t1 = Time(t.mjd + 1, format='mjd') + utc2pst
    fdb1 = dump_tsys.rd_fdb(t1)
    # Concatenate the two days (if the second day exists)
    if fdb1 != {}:
        for key in fdb.keys():
            if key in fdb1.keys():
                fdb[key] = np.concatenate((fdb[key], fdb1[key]))

    # Find unique scan IDs
    scans, idx = np.unique(fdb['SCANID'], return_index=True)

    # Limit to scans in 'NormalObserving' mode
    good, = np.where(fdb['PROJECTID'][idx] == 'NormalObserving')
    if len(good) > 0:
        scans = scans[good]
    else:
        print('No NormalObserving scans found.')
        return None, None, None, None

    # Find scanID that starts earlier than, but closest to, the current time
    for i, scan in enumerate(scans):
        print(scan)
        dt = t - Time(time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(scan, '%y%m%d%H%M%S')))
        if dt.sec > 0.:
            iout = i
    scan = scans[iout]

    # Find files for this scan
    fidx, = np.where(fdb['SCANID'] == scan)
    tlevel = None
    bflag = None
    if len(fidx) > 0:
        files = fdb['FILE'][fidx]
        files_st = Time(fdb['ST_TS'][fidx].astype(float), format='lv')
        files_subdir = [st.to_datetime().strftime('%Y%m%d') for st in files_st]
        # Find out how old last file of this scan is, and proceed only if less than 20 minutes
        # earlier than the time given in t.
        try:
            dt = t - Time(time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(files[-1], 'IDB%Y%m%d%H%M%S')))
        except:
            dt = 10000.  # Forces skip of plot creation
            print('Unexpected FDB file format.')
            scan = None
        if dt.sec < 1200.:
            # This is a currently active scan, so create the figure
            path = '/data1/IDB/'
            filelist = files
            if not os.path.isdir(path + files[0]):
                # Look in /dppdata1
                # path = '/data1/eovsa/fits/IDB/'+datstr+'/'
                path = get_idbdir(t)
                files = []
                found = 0
                for i, file in enumerate(filelist):
                    files.append(os.path.join(path, files_subdir[i], file))
                    if os.path.isdir(files[-1]):
                        found += 1
                if found == 0:
                    # if not os.path.isdir(files[0]):
                    print('No files found for this scan ID', scan)
                    scan = None
                    times = None
                    return scan, tlevel, bflag, times
            else:
                files = [os.path.join(path, file) for file in filelist]
            cache_file = _get_xsp_cache_file(t, files)
            cache = _load_xsp_cache(cache_file)
            mode = None
            fghz = None
            times_jd = None
            pdata = None

            if cache is not None and cache['scanid'] == str(scan):
                cached_files = cache['files']
                new_files = [f for f in files if f not in cached_files]
                if len(new_files) == 0:
                    print('Using cached XSP payload from {0}'.format(cache_file))
                    print('Cache scan: {0}. Requested scan: {1}.'.format(cache['scanid'], scan))
                    print('Cache reuse: {0} files reused, 0 new files.'.format(len(cached_files)))
                    print('Cached files: {0}'.format([os.path.basename(f) for f in cached_files]))
                    mode = cache['mode']
                    fghz = cache['fghz']
                    times_jd = cache['times_jd']
                    pdata = cache['pdata']
                else:
                    print('Cache hit for scan {0}: {1} files reused from cache, {2} new files to process.'.format(
                        scan, len(cached_files), len(new_files)
                    ))
                    print('Cache file: {0}'.format(cache_file))
                    print('Cached files: {0}'.format([os.path.basename(f) for f in cached_files]))
                    print('New files: {0}'.format([os.path.basename(f) for f in new_files]))
                    out_new, cal_ok_new = read_idb_calibrated(new_files, t)
                    if out_new:
                        mode_new, fghz_new, times_jd_new, pdata_new = _spectrogram_from_out(out_new, cal_ok_new, scan)
                        same_mode = (mode_new == cache['mode'])
                        same_fghz = len(fghz_new) == len(cache['fghz']) and (
                            len(fghz_new) == 0 or np.nanmax(np.abs(fghz_new - cache['fghz'])) < 1e-6
                        )
                        if same_mode and same_fghz:
                            fghz, times_jd, pdata = _merge_spectrogram_payload(
                                cache['fghz'], cache['times_jd'], cache['pdata'], times_jd_new, pdata_new
                            )
                            mode = mode_new
                            _save_xsp_cache(cache_file, scan, mode, files, fghz, times_jd, pdata)
                            print('Updated XSP cache with new files.')
                        else:
                            print('Cache payload incompatible with new data. Rebuilding from full file list.')
                    else:
                        print('No payload returned for new files. Reusing existing cache.')
                        mode = cache['mode']
                        fghz = cache['fghz']
                        times_jd = cache['times_jd']
                        pdata = cache['pdata']
            elif cache is not None:
                print('Cache exists but scan mismatch. Cache scan: {0}, requested scan: {1}. Rebuilding from files.'.format(
                    cache['scanid'], scan
                ))

            if mode is None:
                # data, uvw, fghz, times = gd.get_X_data(files)
                out, cal_ok = read_idb_calibrated(files, t)
                if not out:
                    print('No data returned from read_idb for scan ID {0}'.format(scan))
                    scan = None
                    times = None
                    return scan, tlevel, bflag, times
                print('Files read at', Time.now())
                print(files)
                mode, fghz, times_jd, pdata = _spectrogram_from_out(out, cal_ok, scan)
                _save_xsp_cache(cache_file, scan, mode, files, fghz, times_jd, pdata)
                print('Wrote XSP cache to {0}'.format(cache_file))

            times = Time(times_jd, format='jd')
            if ax is not None:
                datstr = times[0].iso[:10]
                ax.set_xlabel('Time [UT on ' + datstr + ']')
                ax.set_ylabel('Frequency [GHz]')

            if mode == 'tp':
                if ax is not None:
                    ax.set_title('EOVSA Total Power for ' + datstr)
                finite = pdata[np.isfinite(pdata)]
                finite = finite[np.where(finite > 0.0)]
                dmin = 1.0
                dmax = None
                if len(finite) > 0:
                    dmin = max(1.0, np.nanpercentile(finite, 5))
                    dmax = np.nanpercentile(finite, 95)
                sp.plot_spectrogram(fghz, times, pdata,
                                    ax=ax, logsample=None, cbar=True, dmin=dmin, dmax=dmax)
                if ax is not None:
                    ax.text(
                        0.01, 0.98,
                        'TP calibration from previous day. Not for science use.',
                        transform=ax.transAxes, ha='left', va='top', fontsize=8,
                        color='white',
                        bbox={'facecolor': 'black', 'alpha': 0.35, 'edgecolor': 'none', 'pad': 2.0}
                    )
            else:
                if ax is not None:
                    ax.set_title('EOVSA Summed Cross-Correlation Amplitude for ' + datstr)
                finite = pdata[np.isfinite(pdata)]
                dmax = None
                if len(finite) > 0:
                    dmax = np.nanpercentile(finite, 95)
                sp.plot_spectrogram(fghz, times, pdata,
                                    ax=ax, logsample=None, xdata=True, cbar=True, dmax=dmax)
            # tlevel, bflag = flaremeter(data)
        else:
            print('Time', dt.sec, 'is > 1200 s after last file of last NormalObserving scan.  No plot created.')
            scan = None
            times = None
    else:
        print('No files found for this scan ID', scan)
        scan = None
    return scan, tlevel, bflag, times


def flaremeter(data):
    ''' Obtain median of data across baselines, polarizations, and frequencies to create a
        time series indicated whether a flare has occurred.  Values returned will be close
        to unity if no flare.  Returns:
            tlevel:      Array of levels at each time, nominally near unity
            bflag:       Array of flags indicating nominal background (where True) or
                            elevated background (where False) indicating possible flare
    '''
    nbl, npol, nf, nt = data.shape
    tlevel = np.zeros(nt, 'float')
    background = np.sqrt(np.abs(data[:, 0, :, :]) ** 2 + np.abs(data[:, 1, :, :]) ** 2)
    init_bg = np.nanmedian(background, 2)  # Initially take background as median over entire time range
    bflag = np.ones(nt, 'bool')  # flags indicating "good" background times (not in flare)
    for i in range(nt):
        good, = np.where(bflag[:i] == True)  # List of indexes of good background times up to current time
        ngood = len(good)  # Truncate list of indexes to last 100 elements (or fewer)
        if ngood > 100:
            good = good[ngood - 100:]
            # Calculate median over good background times
            bg = np.nanmedian(background[:, :, good], 2)
        else:
            # If there haven't been 100 times with good backgrounds yet, just use the initial one.
            # This is supposed to avoid startup transients.
            bg = init_bg
        # Generate levels for each baseline and frequency for this time
        level = np.sqrt(abs(data[:, 0, :, i]) ** 2 + abs(data[:, 1, :, i]) ** 2) / bg
        # Take median over baseline and frequency to give a single number for this time
        tlevel[i] = np.nanmedian(level)
        if tlevel[i] > 1.05:
            # If the level of the current time is higher than 1.05, do not include this time in future backgrounds
            bflag[i] = False
    return tlevel, bflag


def cleanup(bflag):
    ''' Cleans up the background flag array to remove rapid fluctuations
        and provide better in-flare designations.
    '''
    return bflag


def get_history(times, tlevel, bflag):
    ''' Given newly determined tlevel and bflag, see if a file already
        exists for this date and append or replace with new information, 
        if so, otherwise create a new file.
        
        File created is a binary file of records, 9 bytes per record: 
           float time, float tlevel, bool bflag
        Returns data for entire day (contents of any existing file plus
        the new data)
    '''
    import glob
    from eovsapy import dump_tsys
    import struct

    datstr = times[0].iso[:10].replace('-', '')
    filename = '/common/webplots/flaremon/flaremeter/FLM' + datstr + '.dat'
    if len(glob.glob(filename)) == 1:
        # Filename exists, so read entire file at once
        f = open(filename, 'rb')
        buf = f.read()
        f.close()
        nrec = len(buf) // 13  # 13 bytes per record: double time, float level, bool flag
        t = np.zeros(nrec, 'double')
        l = np.zeros(nrec, 'float')
        b = np.zeros(nrec, 'bool')
        for i in range(nrec):
            t[i], l[i], b[i] = struct.unpack('dfB', buf[i * 13:(i + 1) * 13])
        # Since unique also sorts, and takes the first instance, it should be enough to
        # concatenate times with t and get unique indexes
        times_lv = np.concatenate(((times.lv + 0.5).astype('int'), (t + 0.5).astype('int')))
        tlevel = np.concatenate((tlevel, l))
        bflag = np.concatenate((bflag, b))
        blah, idx = np.unique(times_lv, return_index=True)
        times = Time(times_lv[idx], format='lv')
        tlevel = tlevel[idx]
        bflag = bflag[idx]

    # Open same filename for writing (overwrites contents if file exists)
    f = open(filename, 'wb')
    for i in range(len(times)):
        f.write(struct.pack('dfB', *(times[i].lv, tlevel[i], bflag[i])))
    f.close()
    return times, tlevel, bflag


def rd_RT(filename=None):
    import os
    from eovsapy.util import Time
    if filename is None:
        datstr = Time.now().iso[:10]
        filename = '/data1/RT/RT_' + datstr + '.txt'
    if not os.path.exists(filename):
        print('File', filename, 'not found.')
        return {}
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    t = []
    amp = []
    tp = []
    from eovsapy.util import Time
    for line in lines:
        t.append(Time(line[:19]).plot_date)
        tp.append(float(line[24:31]))
        amp.append(float(line[32:-3]))
    t = np.array(t)
    dt = t[1:] - t[:-1]
    jmp, = np.where(dt < -0.75)
    if len(jmp) == 1:
        # This is a day jump, so add one to later times
        t[jmp[0] + 1:] += 1

    return {'time': t, 'tp': tp, 'amp': amp}


if __name__ == "__main__":
    ''' For non-interactive use, use a backend that does not require a display
        Usage examples:
          python /common/python/eovsapy-src/eovsapy/flare_monitor.py
          python /common/python/eovsapy-src/eovsapy/flare_monitor.py --timestamp "2026-02-19 20:10:00"
          python /common/python/eovsapy-src/eovsapy/flare_monitor.py --timestamp "2026-02-19T20:10:00" --skip-xsp
          python /common/python/eovsapy-src/eovsapy/flare_monitor.py --rt
    '''
    import argparse
    import glob, shutil
    import matplotlib, sys

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description='Generate flare-monitor XSP PNG files.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python flare_monitor.py\n'
            '  python flare_monitor.py --timestamp "2026-02-19 20:10:00"\n'
            '  python flare_monitor.py --timestamp "2026-02-19T20:10:00" --skip-xsp\n'
            '  python flare_monitor.py --rt'
        )
    )
    parser.add_argument(
        '-t', '--timestamp',
        help='UTC time string parseable by eovsapy.util.Time.'
    )
    parser.add_argument(
        '--skip-xsp', action='store_true',
        help='Skip generating XSP20*.png.'
    )
    parser.add_argument(
        '--rt', action='store_true',
        help='Run RT_flare_monitor() and exit.'
    )
    args = parser.parse_args()

    if args.rt:
        RT_flare_monitor()
        sys.exit(0)

    if args.timestamp:
        try:
            t = Time(args.timestamp)
        except Exception as err:
            parser.error('Cannot interpret --timestamp {!r}: {}'.format(args.timestamp, err))
    else:
        t = Time.now()
    skip = args.skip_xsp
    print(t.iso[:19], ': ', )
    # if (t.mjd % 1) < 3./24:
    # # Special case of being run at or before 3 AM (UT), so change to late "yesterday" to finish out
    # # the previous UT day
    # imjd = int(t.mjd)
    # t = Time(float(imjd-0.001),format='mjd')

    if not skip:
        # Check if cross-correlation plot already exists
        f, ax = plt.subplots(1, 1)
        f.set_size_inches(12.5, 5)
        scanid, tlevel, bflag, times = xdata_display(t, ax)
        if times is None:
            plt.close(f)
        else:
            plt.savefig('/common/webplots/flaremon/XSP20' + scanid + '.png', bbox_inches='tight')
            plt.close(f)
            print('Plot written to /common/webplots/flaremon/XSP20' + scanid + '.png')
            bflag = cleanup(bflag)
        # See if a file for this date already exists, and if so, read it and 
        # append or replace with the newly determined levels
        # times, tlevel, bflag = get_history(times, tlevel, bflag)

    # Zombie legacy FLM block intentionally disabled (no file output, only extra I/O/log noise).
    # Copy the most recent two files to fixed names so that the web page can find them.
    flist = np.sort(glob.glob('/common/webplots/flaremon/XSP20????????????.png'))
    if len(flist) > 0:
        shutil.copy(flist[-1], '/common/webplots/flaremon/XSP_latest.png')
    if len(flist) > 1:
        shutil.copy(flist[-2], '/common/webplots/flaremon/XSP_later.png')
