'''
Minimal spectrogram plotting utilities used by flare_monitor.
'''

import numpy as np


def _resample_2d(fghz, ut, tsys, fnew, kx, ky):
    '''Interpolate a spectrogram on a regular (fghz, ut) grid.'''
    from scipy.interpolate import RectBivariateSpline

    fghz = np.asarray(fghz, dtype=float)
    ut = np.asarray(ut, dtype=float)
    tsys = np.asarray(tsys, dtype=float)

    # RectBivariateSpline requires interpolation orders less than grid size.
    if len(fghz) < 2 or len(ut) < 2:
        return np.tile(tsys[:1, :], (len(fnew), 1))
    kx = min(kx, len(fghz) - 1)
    ky = min(ky, len(ut) - 1)
    spline = RectBivariateSpline(fghz, ut, tsys, kx=kx, ky=ky)
    return spline(fnew, ut)


def log_sample(fghz, ut, tsys):
    '''Resample a spectrogram to logarithmic frequency spacing.'''
    nf = len(fghz)
    fghzl = np.logspace(np.log10(fghz[0]), np.log10(fghz[-1]), nf)
    out = _resample_2d(fghz, ut, tsys, fghzl, kx=3, ky=3)
    return fghzl, out


def lin_sample(fghz, ut, tsys):
    '''Resample a spectrogram to linear frequency spacing.'''
    nf = len(fghz)
    fghzl = np.linspace(fghz[0], fghz[-1], nf)
    out = _resample_2d(fghz, ut, tsys, fghzl, kx=1, ky=1)
    return fghzl, out


def plot_spectrogram(fghz, ut, tsys, ax=None, cbar=True, logsample=False, **kwargs):
    '''Create a standard EOVSA spectrogram plot.

    kwargs:
      dmin, dmax, xlabel, ylabel, title, xdata
    '''
    import matplotlib.dates
    import matplotlib.colors as mcolors
    import matplotlib.pylab as plt

    utd = ut.plot_date
    datstr = ut[0].iso[:10]

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel('Time [UT on ' + datstr + ']')
        ax.set_ylabel('Frequency [GHz]')
        ax.set_title('EOVSA Total Power for ' + datstr)
        if kwargs.get('xdata', False):
            ax.set_title('EOVSA Summed Cross-Correlation Amplitude for ' + datstr)

    ax.xaxis.set_tick_params(width=1.5, size=10, which='both')
    ax.yaxis.set_tick_params(width=1.5, size=10, which='both')
    if logsample:
        fghzl, tsysl = log_sample(fghz, utd, tsys)
        ax.set_yscale('log')
        minor_formatter = plt.LogFormatter(base=10, labelOnlyBase=False)
        ax.yaxis.set_minor_formatter(minor_formatter)
    elif logsample is None:
        fghzl, tsysl = fghz, tsys
    else:
        fghzl, tsysl = lin_sample(fghz, utd, tsys)

    dmin = kwargs.get('dmin', 1.0)
    dmax = kwargs.get('dmax', np.nanmax(tsys))
    if dmax is None:
        dmax = np.nanmax(tsysl)
    xdata = kwargs.get('xdata', False)

    if xdata:
        data = np.clip(tsysl, dmin, dmax)
        im = ax.pcolormesh(utd, fghzl, data, shading='auto')
    else:
        vmin = max(float(dmin), np.finfo(float).tiny)
        vmax = float(dmax)
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin * 10.0
        data = np.clip(tsysl, vmin, vmax)
        im = ax.pcolormesh(utd, fghzl, data, shading='auto',
                           norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))

    if cbar:
        cbar_label = 'Flux Density [sfu]'
        if xdata:
            cbar_label = 'Amplitude [arb. units]'
        cbar_pad = kwargs.get('cbar_pad', 0.015)
        cbar_fraction = kwargs.get('cbar_fraction', 0.05)
        ax.figure.colorbar(im, ax=ax, label=cbar_label, pad=cbar_pad, fraction=cbar_fraction)
    ax.xaxis_date()
    span_minutes = 0.0
    if len(utd) > 1:
        span_minutes = (float(utd[-1]) - float(utd[0])) * 24.0 * 60.0
    if span_minutes <= 20.0:
        locator = matplotlib.dates.MinuteLocator(interval=1)
        formatter = matplotlib.dates.DateFormatter('%H:%M:%S')
    elif span_minutes <= 120.0:
        locator = matplotlib.dates.MinuteLocator(interval=5)
        formatter = matplotlib.dates.DateFormatter('%H:%M')
    elif span_minutes <= 360.0:
        locator = matplotlib.dates.MinuteLocator(interval=15)
        formatter = matplotlib.dates.DateFormatter('%H:%M')
    else:
        locator = matplotlib.dates.AutoDateLocator(minticks=4, maxticks=8, interval_multiples=True)
        try:
            formatter = matplotlib.dates.ConciseDateFormatter(locator, show_offset=False)
        except TypeError:
            formatter = matplotlib.dates.ConciseDateFormatter(locator)
        except AttributeError:
            formatter = matplotlib.dates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    if 'xlabel' in kwargs:
        if kwargs['xlabel'] == 'auto':
            ax.set_xlabel('Time [UT on ' + datstr + ']')
        else:
            ax.set_xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        if kwargs['ylabel'] == 'auto':
            ax.set_ylabel('Frequency [GHz]')
        else:
            ax.set_ylabel(kwargs['ylabel'])
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    return ax
