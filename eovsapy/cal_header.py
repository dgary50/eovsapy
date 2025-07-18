#
# Routines for implementing XML descriptions of calibration data.
# 
# History:
#   2015-Mar-29  DG
#     First written (for Solpnt).
#   2015-Mar-30  DG
#     Added FGHz and Poln arrays to Solpnt
#   2015-Mar-31  DG
#     Added DCM_base_attn().  Changed name Solpnt to TPcal, and changed 
#     routine names to end in "...2xml"  Also added cal_types dictionary.
#   2015-Apr-01  DG
#     Forgot to commit changes to database!  Added commit at bottom of
#     send_xml2sql().
#   2015-May-29  DG
#      Converted from using datime() to using Time() based on astropy.
#   2016-Apr-07  DG
#      Added get_size() and write_cal() routines.  Also added some selection
#      keywords (type and t) to send_xml2sql() routine.
#   2016-Apr-09  DG
#      Added read_cal() and read_cal_xml() routines.  Also binary buffer
#      writing routines *2sql() for the various types of calibration data.
#      Changed the total power type to prototype, because I need to break
#      the scheme a little (need to change number of frequencies, ants in
#      the type definition on the fly).  For the "production" TP calibration,
#      a number of things will change, but I cannot know exactly how until
#      I can successfully record the relevant data.
#   2016-05-05  DG
#      Added a fifth cal type for equalizer gains, and routines
#      eq_gain2xml() and eq_gain2sql()
#   2016-05-21  DG
#      Change to dcm_master_table2sql() to allow output from one of the
#      better adc_cal2.py routines (get_dcm_attn()).
#   2016-08-03  DG
#      Added two type definitions for DCM_Attn_Val and FEM_Attn_Val and
#      their associated xml and sql routines.
#   2016-10-17  DG
#      Added a dla_update2sql() routine to make updates to the delay center
#      easier.
#   2016-11-21  DG
#      Added dla_censql2table() routine to read a delay center calibration
#      from the SQL database and (optionally) send it to the ACC, which
#      is needed for dppxmp. Also renamed dla_cen2sql() -> dla_centable2sql().
#   2017-01-08  DG
#      Fixed bug where XY delay in dla_update2sql() was being treated as 
#      relative to ant 1Y.  It is already relative to ant 1X, but ant 1Y 
#      has to be able to change.
#   2017-03-05  DG
#      Given the latest methods for determining delays (using delay_widget()),
#      the dla_update2sql() routine is changed to expect delays in ns, and
#      already relative to Ant 1.  A non-zero Ant1 delay issues an error.
#   2017-04-13  DG
#      Changed order of parameters in dla_censql2table() so that first one
#      is time.  This guards against accidentally writing a file to the ACC.
#   2017-05-13  DG
#      Changed names of fem_attn* and dcm_attn* routines to remove "val" from
#      the names.  Updated fem_attn2sql() to create a default table if called
#      without a table.
#   2017-05-13  SJ
#      Added an eighth cal type for reference calibration, and routines
#      refcal2sql() and refcal2xml()
#   2017-05-14  SJ
#      Added read_calX() routine.
#   2017-07-08  DG
#      Added lorx argument to dla_update2sql(), to enable scheme to write
#      Ant 14 low-frequency receiver delays into the Ant 15 slot.  The schedule
#      will interpret the Ant 15 slot as Ant 14 whenever the low-frequency
#      receiver is in place (I hope).
#   2017-08-06  DG
#      Added new calibration types tpcal (replaces proto_tpcal, adding
#      auto-correlation information) and xy_phasecal, containing the X-Y delay 
#      phase vs. frequency for each antenna.
#   2017-10-05  NK
#      Modified fem_attn_val2xml() and fem_attn_val2sql(), to write the new XML
#      definition and binary buffer that are compatible with current
#      GAINCALTEST measurements into SQL database
#   2017-12-05  DG
#      Modified what is returned from read_cal(), so that the SQL timestamp is
#      appended to both the XML and data buffer.  This should be transparent
#      to previous uses of the routine.  Having the SQL timestamp is useful for
#      some error checking.
#   2018-01-01  DG
#      Changed definition of X-Y Phase Calibration to add new Xi_Rot phase offset, 
#      so changed the version to 2.0.  Corresponding changes to  xy_phasecal2xml() 
#      and xy_phasecal2sql()
#   2018-02-03  SJ
#      Added an twelfth cal type for super reference calibration (with band 4), and updated routines
#      refcal_sp2xml()
#   2018-02-23  DG
#      Added delete_cal() routine for deleting SQL records of a given type for a given time.
#   2019-02-22  DG
#      Updated DCM Master table to version 2.0, increasing from 34 to 52 bands to reflect new IF Filters
#   2019-07-18  DG
#      Updated Refcal and Phasecal types to version 2.0, increasing from 34 to 52 bands (xml defs written to 2019-02-22)
#   2020-05-12  DG
#      Added copy_cal() routine for copying SQL records of a given type from one time to another.
#   2020-05-21 OG
#      Added rstnflux2xml() routine for writing header information for RSTN noon flux values.
#      Added cal_type 12 for the RSTN flux values
#   2020-08-22  DG
#      Since I often want to check the date/time of a calibration record, I added a "verbose"
#      keyword to read_cal(), to print this information if desired (i.e. if verbose=True).
#   2022-03-06  DG
#      Added ACCdlatable2dict() and dla_update2table() routines to handle the interim period when
#      the SQL server is not available.  The new scheme that is meant to allow us to take data is to
#      keep a record of the delay centers in the folder /nas4/Tables/Delays/ and maintain the most
#      current one on the ACC as /parm/delay_centers.txt (which is the file read by dppxmp).
#   2022-03-07  DG
#      Added ACC_DCMtable2dict() routines to handle the interim period when
#      the SQL server is not available.  The new scheme that is meant to allow us to take data is to
#      keep a record of the DCM attenuations in the folder /nas4/Tables/DCM_master/ and maintain the most
#      current one on the ACC as /parm/DCM_master_table.txt (which is the file read by schedule).
#   2022-03-28  DG
#      Multiple changes to bring this routine into the eovsapy Python 3 package.
#   2022-05-22 DG
#      Many further changes to deal with multiple databases inluding the cloud database.
#   2023-01-06  DG
#      Fixed a number of issues with read_calX() [and commented out unused read_cal_xmlX()]
#   2023-02-17  DG
#      Added a new calibration type 13, SKYcal, and added skycal2xml() and skycal2sql() routines to write
#      it to the abin table.
#   2025-06-10  DG
#      Many changes to make it all work with 16 antennas.  It should still be able to read and write 
#      older records.

import struct, sys, os
from .util import Time, extract
from . import dbutil, read_xml2
import numpy as np

def cal_types(t=Time.now()):
    ''' Routine that defines all of the basic "long-term" calibration
        /information types as a dictionary.  These types and descriptions 
        will be written into the Description field of the aBin table.
        A new type can be added at the end--there is no significance to
        the type number--it is just a unique ordinal.

        Although not strictly necessary, in case one of these definitions
        is changed in any way, it is good practice to increment the version
        number, given as the last element of each type.        
    '''
    # Set cal_type versions based on date
    if t.mjd < Time('2018-01-01').mjd:
        # All version 1 before 2018-01-01
        verlist = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif t.mjd < Time('2019-02-22').mjd:
        # Type 11 becomes version 2.0 between 2018-01-01 and 2019-02-22
        verlist = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0]
    elif t.mjd < Time('2025-05-22').mjd:
        # Types 2, 8, and 9 become version 2.0 between 2019-02-22 and 2025-05-04
        verlist = [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0]
    else:
        # Types 2, 3, 4, 6, 7, 8, 9, 10, 11, and 13 all increment by 1 after 2025-05-04
        verlist = [1.0, 3.0, 2.0, 2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 1.0, 3.0, 1.0, 1.0]

    return {1: ['Prototype total power calibration (output of SOLPNTCAL)', 'proto_tpcal2xml', verlist[0]],
            2: ['DCM master base attenuation table [units=dB]', 'dcm_master_table2xml', verlist[1]],  # Version 2.0 (2019-02-22)
            3: ['DCM base attenuation table [units=dB]', 'dcm_table2xml', verlist[2]],
            4: ['Delay centers [units=ns]', 'dlacen2xml', verlist[3]],
            5: ['Equalizer gains', 'eq_gain2xml', verlist[4]],
            6: ['DCM attenuator values [units=dB]', 'dcm_attn_val2xml', verlist[5]],
            7: ['FEM attenuator values [units=dB]', 'fem_attn_val2xml', verlist[6]],
            8: ['Reference calibration', 'refcal2xml', verlist[7]],   # Version 2.0 (2019-02-22)
            9: ['Daily phase calibration', 'phacal2xml', verlist[8]],  # Version 2.0 (2019-02-22)
           10: ['Total power calibration', 'tpcal2xml', verlist[9]],
           11: ['X-Y phase calibration', 'xy_phasecal2xml', verlist[10]],  # Changed the definition of this, so version is 2.0 (2018-01-01)
           12: ['RSTN noon flux', 'rstnflux2xml', verlist[11]],
           13: ['Sky calibration', 'skycal2xml', verlist[12]]}


def str2bin(string):
    ''' Convert the given string to a binary packed byte buffer '''
    return struct.pack(str(len(string) + 1) + 's', bytes(string + '\n',encoding='ascii'))


def proto_tpcal2xml(nant, nfrq):
    ''' Writes the XML description of the prototype total power calibration binary
        data (SOLPNTCAL result).  Returns a binary representation of the
        text file, for putting into the SQL database.  For the prototype
        data, the format varies due to variable numbers of antennas/frequencies
    '''
    version = cal_types()[1][2]

    buf = b''
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name>TPcal</Name>')
    buf += str2bin('<NumElts>5</NumElts>')

    # Timestamp (double) [s, in LabVIEW format]
    # Start time of SOLPNT observation on which calibration is based
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Timestamp</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Version of this XML file.  This number should be incremented each
    # time there is a change to the structure of this file.
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Version</Name>')
    buf += str2bin('<Val>' + str(version) + '</Val>')
    buf += str2bin('</DBL>')

    # Array of frequencies in GHz (nfrq)
    buf += str2bin('<Array>')
    buf += str2bin('<Name>FGHz</Name>')
    buf += str2bin('<Dimsize>' + str(nfrq) + '</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # Array of Poln (2)
    # Polarization list (Miriad definition) (signed int)
    #     1: Stokes I
    #     2: Stokes Q
    #     3: Stokes U
    #     4: Stokes V
    #    -1: Circular RR
    #    -2: Circular LL
    #    -3: Circular RL
    #    -4: Circular LR
    #    -5: Linear XX
    #    -6: Linear YY
    #    -7: Linear XY
    #    -8: Linear YX
    #     0: Not used
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Poln</Name>')
    buf += str2bin('<Dimsize>2</Dimsize>\n<I32>\n<Name></Name>\n<Val></Val>\n</I32>')
    buf += str2bin('</Array>')

    # Array of clusters for each antenna (nant)
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Antenna</Name>')
    buf += str2bin('<Dimsize>' + str(nant) + '</Dimsize>')

    # Cluster containing information for one antenna
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name></Name>')
    buf += str2bin('<NumElts>3</NumElts>')

    # Antenna number (1-13).
    buf += str2bin('<U16>')
    buf += str2bin('<Name>Antnum</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</U16>')

    # Calibration factors (nfrq x 2) = nfreq x npol
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Calfac</Name>')
    buf += str2bin(
        '<Dimsize>' + str(nfrq) + '</Dimsize><Dimsize>2</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # Offsun values (448 x 2) = nfreq x npol
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Offsun</Name>')
    buf += str2bin(
        '<Dimsize>' + str(nfrq) + '</Dimsize><Dimsize>2</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # End cluster
    buf += str2bin('</Cluster>')  # End Calinfo cluster
    buf += str2bin('</Array>')  # End Antenna array
    buf += str2bin('</Cluster>')  # End TPcal cluster

    return buf


def tpcal2xml(nant, nfrq):
    ''' Writes the XML description of the total power calibration binary
        data (SOLPNTCAL result), for both power and auto-correlation.  
        Returns a binary representation of the text file, for putting into the SQL 
        database.  The format varies due to variable numbers of antennas/frequencies.
    '''
    version = cal_types()[10][2]

    buf = b''
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name>TPcal</Name>')
    buf += str2bin('<NumElts>5</NumElts>')

    # Timestamp (double) [s, in LabVIEW format]
    # Start time of SOLPNT observation on which calibration is based
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Timestamp</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Version of this XML file.  This number should be incremented each
    # time there is a change to the structure of this file.
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Version</Name>')
    buf += str2bin('<Val>' + str(version) + '</Val>')
    buf += str2bin('</DBL>')

    # Array of frequencies in GHz (nfrq)
    buf += str2bin('<Array>')
    buf += str2bin('<Name>FGHz</Name>')
    buf += str2bin('<Dimsize>' + str(nfrq) + '</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # Array of Poln (2)
    # Polarization list (Miriad definition) (signed int)
    #     1: Stokes I
    #     2: Stokes Q
    #     3: Stokes U
    #     4: Stokes V
    #    -1: Circular RR
    #    -2: Circular LL
    #    -3: Circular RL
    #    -4: Circular LR
    #    -5: Linear XX
    #    -6: Linear YY
    #    -7: Linear XY
    #    -8: Linear YX
    #     0: Not used
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Poln</Name>')
    buf += str2bin('<Dimsize>2</Dimsize>\n<I32>\n<Name></Name>\n<Val></Val>\n</I32>')
    buf += str2bin('</Array>')

    # Array of clusters for each antenna (nant)
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Antenna</Name>')
    buf += str2bin('<Dimsize>' + str(nant) + '</Dimsize>')

    # Cluster containing information for one antenna
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name></Name>')
    buf += str2bin('<NumElts>5</NumElts>')

    # Antenna number (1-13).
    buf += str2bin('<U16>')
    buf += str2bin('<Name>Antnum</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</U16>')

    # Total Power Calibration factors (nfrq x 2) = nfreq x npol
    buf += str2bin('<Array>')
    buf += str2bin('<Name>TPCalfac</Name>')
    buf += str2bin(
        '<Dimsize>' + str(nfrq) + '</Dimsize><Dimsize>2</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # Auto-Correlation Calibration factors (nfrq x 2) = nfreq x npol
    buf += str2bin('<Array>')
    buf += str2bin('<Name>ACCalfac</Name>')
    buf += str2bin(
        '<Dimsize>' + str(nfrq) + '</Dimsize><Dimsize>2</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # Total Power Offsun values (nfrq x 2) = nfreq x npol
    buf += str2bin('<Array>')
    buf += str2bin('<Name>TPOffsun</Name>')
    buf += str2bin(
        '<Dimsize>' + str(nfrq) + '</Dimsize><Dimsize>2</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # Auto-Correlation Offsun values (nfrq x 2) = nfreq x npol
    buf += str2bin('<Array>')
    buf += str2bin('<Name>ACOffsun</Name>')
    buf += str2bin(
        '<Dimsize>' + str(nfrq) + '</Dimsize><Dimsize>2</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # End cluster
    buf += str2bin('</Cluster>')  # End Calinfo cluster
    buf += str2bin('</Array>')  # End Antenna array
    buf += str2bin('</Cluster>')  # End TPcal cluster

    return buf

def skycal2xml(nant, nfrq):
    ''' Writes the XML description of the receiver background binary
        data (SKYCAL_ANAL result), for both power and auto-correlation.  
        Returns a binary representation of the text file, for putting into the SQL 
        database.  The format varies due to variable numbers of antennas/frequencies.
    '''
    version = cal_types()[13][2]

    buf = b''
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name>SKYcal</Name>')
    buf += str2bin('<NumElts>5</NumElts>')

    # Timestamp (double) [s, in LabVIEW format]
    # Start time of SKYCAL observation on which calibration is based
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Timestamp</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Version of this XML file.  This number should be incremented each
    # time there is a change to the structure of this file.
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Version</Name>')
    buf += str2bin('<Val>' + str(version) + '</Val>')
    buf += str2bin('</DBL>')

    # Array of frequencies in GHz (nfrq)
    buf += str2bin('<Array>')
    buf += str2bin('<Name>FGHz</Name>')
    buf += str2bin('<Dimsize>' + str(nfrq) + '</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # Array of Poln (2)
    # Polarization list (Miriad definition) (signed int)
    #     1: Stokes I
    #     2: Stokes Q
    #     3: Stokes U
    #     4: Stokes V
    #    -1: Circular RR
    #    -2: Circular LL
    #    -3: Circular RL
    #    -4: Circular LR
    #    -5: Linear XX
    #    -6: Linear YY
    #    -7: Linear XY
    #    -8: Linear YX
    #     0: Not used
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Poln</Name>')
    buf += str2bin('<Dimsize>2</Dimsize>\n<I32>\n<Name></Name>\n<Val></Val>\n</I32>')
    buf += str2bin('</Array>')

    # Array of clusters for each antenna (nant)
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Antenna</Name>')
    buf += str2bin('<Dimsize>' + str(nant) + '</Dimsize>')

    # Cluster containing information for one antenna
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name></Name>')
    buf += str2bin('<NumElts>3</NumElts>')

    # Antenna number (1-nant).
    buf += str2bin('<U16>')
    buf += str2bin('<Name>Antnum</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</U16>')

    # Backend receiver power with full 62 dB frontend attenuation (nfrq x 2) = nfreq x npol
    buf += str2bin('<Array>')
    buf += str2bin('<Name>RcvrBgd</Name>')
    buf += str2bin(
        '<Dimsize>' + str(nfrq) + '</Dimsize><Dimsize>2</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # Auto-Correlation with full 62 dB frontend attenuation (nfrq x 2) = nfreq x npol
    buf += str2bin('<Array>')
    buf += str2bin('<Name>ACRcvrBgd</Name>')
    buf += str2bin(
        '<Dimsize>' + str(nfrq) + '</Dimsize><Dimsize>2</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # End cluster
    buf += str2bin('</Cluster>')  # End Calinfo cluster
    buf += str2bin('</Array>')  # End Antenna array
    buf += str2bin('</Cluster>')  # End Skycal cluster

    return buf


def xy_phasecal2xml(t=Time.now()):
    ''' Writes the XML description of the X-Y delay phase calibration binary
        data (get_xy_corr results), for antennas 1 to nant plus the 27-m.  
        Returns a binary representation of the text file, for putting into the SQL 
        database.
        
        Version 2 adds xi_rot phase offset (2018-01-01)
        Version 3 is for 15 2-m antennas and 27-m in slot 16
    '''
    version = cal_types(t)[11][2]

    if version == 2:
        nant = 13
    elif version == 3:
        nant = 15
    buf = b''
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name>XYcal</Name>')
    buf += str2bin('<NumElts>5</NumElts>')

    # Timestamp (double) [s, in LabVIEW format]
    # Start time of PHASECAL observation on which calibration is based
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Timestamp</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Version of this XML file.  This number should be incremented each
    # time there is a change to the structure of this file.
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Version</Name>')
    buf += str2bin('<Val>' + str(version) + '</Val>')
    buf += str2bin('</DBL>')

    # Fixed size array of frequencies in GHz (500)
    buf += str2bin('<Array>')
    buf += str2bin('<Name>FGHz</Name>')
    buf += str2bin('<Dimsize>500</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # Fixed size array of frequencies in GHz (500)
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Xi_Rot</Name>')
    buf += str2bin('<Dimsize>500</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # Array of X-Y phases (nant x 500) for each of nant antennas
    buf += str2bin('<Array>')
    buf += str2bin('<Name>XYphase</Name>')
    buf += str2bin('<Dimsize>500</Dimsize><Dimsize>'+str(nant+1)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')  # End XYphase array
    buf += str2bin('</Cluster>')  # End XYcal cluster

    return buf


def dcm_master_table2xml(t=Time.now()):
    ''' Writes the XML description of the DCM master base attenuation 
        table (created by pcapture.py).  Returns a binary representation 
        of the text file, for putting into the SQL database.  The version 
        number must be incremented each time there is a change to the 
        structure of this header.  Updated to version 2 (16 antenna version)
    '''
    version = cal_types(t)[2][2]

    if version == 2:
        nant = 15
    elif version == 3:
        nant = 16

    buf = b''
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name>DCMMasterBaseAttn</Name>')
    buf += str2bin('<NumElts>4</NumElts>')

    # Timestamp (double) [s, in LabVIEW format]
    # Time of creation of the table (close to packet capture time)
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Timestamp</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Version of this XML file.  This number should be incremented each
    # time there is a change to the structure of this file.
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Version</Name>')
    buf += str2bin('<Val>' + str(version) + '</Val>')
    buf += str2bin('</DBL>')

    # List of bands (52), with band number (1-52) if used, 0 if not.
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Bands</Name>')
    buf += str2bin('<Dimsize>52</Dimsize>\n<U16>\n<Name></Name>\n<Val></Val>\n</U16>')
    buf += str2bin('</Array>')

    # Array of base attenuations [dB] (52 x nant*2).  Attenuations for unmeasured
    # antennas and/or bands are set to nominal value of 10 dB.  Values are
    # ordered as Ant1x, Ant1y, Ant2x, Ant2y, ..., Ant16x, Ant16y
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Attenuation</Name>')
    buf += str2bin('<Dimsize>'+str(nant*2)+'</Dimsize><Dimsize>52</Dimsize>\n<U16>\n<Name></Name>\n<Val></Val>\n</U16>')
    buf += str2bin('</Array>')

    # End cluster
    buf += str2bin('</Cluster>')  # End DCMMasterBaseAttn cluster

    return buf


def dcm_table2xml(t=Time.now()):
    ''' Writes the XML description of the DCM base attenuation table 
        (derived from the DCM master base attenuation table and the
        current frequency sequence.  Returns a binary representation of the
        text file, for putting into the SQL database.  The version number
        must be incremented each time there is a change to the structure 
        of this header. Updated to version 2 (16 antenna version)
    '''
    version = cal_types(t)[3][2]

    if version == 1:
        nant = 15
    elif version == 2:
        nant = 16
    buf = b''
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name>DCMBaseAttn</Name>')
    buf += str2bin('<NumElts>3</NumElts>')

    # Timestamp (double) [s, in LabVIEW format]
    # Time of creation of the table (close to packet capture time)
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Timestamp</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Version of this XML file.  This number should be incremented each
    # time there is a change to the structure of this file.
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Version</Name>')
    buf += str2bin('<Val>' + str(version) + '</Val>')
    buf += str2bin('</DBL>')

    # Array of base attenuations [dB] (50 x nant*2).  Attenuations for unmeasured
    # antennas and/or bands are set to nominal value of 10 dB.  Values are
    # ordered as Ant1x, Ant1y, Ant2x, Ant2y, ..., Ant16x, Ant16y
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Attenuation</Name>')
    buf += str2bin('<Dimsize>'+str(nant*2)+'</Dimsize><Dimsize>50</Dimsize>\n<U16>\n<Name></Name>\n<Val></Val>\n</U16>')
    buf += str2bin('</Array>')

    # End cluster
    buf += str2bin('</Cluster>')  # End DCMBaseAttn cluster

    return buf


def dlacen2xml(t=Time.now()):
    ''' Writes the XML description of the Delay Centers table (currently
        created by hand).  Returns a binary representation of the xml
        text file, for putting into the SQL database.  The version number
        must be incremented each time there is a change to the structure 
        of this header. Updated to version 2 (16 antenna version)
    '''
    version = cal_types(t)[4][2]

    if version == 1:
        nchan = 15
    elif version == 2:
        nchan = 17

    buf = b''
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name>Delaycenters</Name>')
    buf += str2bin('<NumElts>3</NumElts>')

    # Timestamp (double) [s, in LabVIEW format]
    # Time of creation of the table (precise time not critical)
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Timestamp</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Version of this XML file.  This number should be incremented each
    # time there is a change to the structure of this file.
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Version</Name>')
    buf += str2bin('<Val>' + str(version) + '</Val>')
    buf += str2bin('</DBL>')

    # List of delay centers [ns] (2 x nchan).
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Delaycen_ns</Name>')
    buf += str2bin('<Dimsize>2</Dimsize><Dimsize>'+str(nchan)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # End cluster
    buf += str2bin('</Cluster>')  # End Delaycenters cluster

    return buf


def eq_gain2xml():
    ''' Writes the XML description of the equalizer gain table.  Returns 
        a binary representation of the xml, for putting into the SQL 
        database.  The version number must be incremented each time there 
        is a change to the structure of this header. This was never used,
        so was not updated for 16 antennas.  Will need updating if used.
    '''
    version = cal_types()[5][2]

    buf = b''
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name>EQ_Gain</Name>')
    buf += str2bin('<NumElts>3</NumElts>')

    # Timestamp (double) [s, in LabVIEW format]
    # Time of creation of the table (precise time not critical)
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Timestamp</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Version of this XML file.  This number should be incremented each
    # time there is a change to the structure of this file.
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Version</Name>')
    buf += str2bin('<Val>' + str(version) + '</Val>')
    buf += str2bin('</DBL>')

    # List of equalizer gains (nant x npol x nband) (2 x 2 x 34).  This
    # can be extended to handle 128 subbands/band, if needed.  Note inverted
    # order of dimensions
    buf += str2bin('<Array>')
    buf += str2bin('<Name>EQ_Coeff</Name>')
    buf += str2bin(
        '<Dimsize>34</Dimsize><Dimsize>2</Dimsize><Dimsize>2</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # End cluster
    buf += str2bin('</Cluster>')  # End EQ_Gain cluster

    return buf


def dcm_attn_val2xml():
    ''' Writes the XML description of the DCM attenuator values as 
        measured by DCMATTNTEST, and analyzed by DCM_attn_anal().  The
        values are complex numbers, in dB, for each of the attenuator 
        bits (2, 4, 8 and 16).  Returns a binary representation of the 
        xml, for putting into the SQL database.  The version number 
        must be incremented each time there is a change to the structure 
        of this header. This was never used but should work for 16 ants.
    '''
    version = cal_types()[6][2]

    buf = b''
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name>DCM_Attn_Val</Name>')
    buf += str2bin('<NumElts>4</NumElts>')

    # Timestamp (double) [s, in LabVIEW format]
    # Time of creation of the table (precise time not critical)
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Timestamp</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Version of this XML file.  This number should be incremented each
    # time there is a change to the structure of this file.
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Version</Name>')
    buf += str2bin('<Val>' + str(version) + '</Val>')
    buf += str2bin('</DBL>')

    # List of real part of attenuations (nant x npol x nbits) (4 x 2 x 16).
    # Note inverted order of dimensions
    buf += str2bin('<Array>')
    buf += str2bin('<Name>DCM_Attn_Real</Name>')
    buf += str2bin(
        '<Dimsize>4</Dimsize><Dimsize>2</Dimsize><Dimsize>16</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # List of imaginary part of attenuations (nant x npol x nbits) (4 x 2 x 16).
    # Note inverted order of dimensions
    buf += str2bin('<Array>')
    buf += str2bin('<Name>DCM_Attn_Imag</Name>')
    buf += str2bin(
        '<Dimsize>4</Dimsize><Dimsize>2</Dimsize><Dimsize>16</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # End cluster
    buf += str2bin('</Cluster>')  # End DCM_Attn_Val cluster

    return buf


def fem_attn_val2xml(t=Time.now()):
    ''' Writes the XML description of the FEM attenuator values as 
        measured by GAINCALTEST, and analyzed by ???.  The
        values are attenuation values corresponding to the level 1-8
        (each 2 dB nominal step), for each antenna, frequency, and 
        polarization. Returns a binary representation of the 
        xml, for putting into the SQL database.  The version number 
        must be incremented each time there is a change to the structure 
        of this header.  Updated for version 2 (16 antennas).
    '''
    version = cal_types()[7][2]

    if version == 1:
        nant = 15
    elif version == 2:
        nant = 16

    buf = b''
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name>FEM_Attn_Val</Name>')
    buf += str2bin('<NumElts>4</NumElts>')

    # Timestamp (double) [s, in LabVIEW format]
    # Time of creation of the table (precise time not critical)
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Timestamp</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Version of this XML file.  This number should be incremented each
    # time there is a change to the structure of this file.
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Version</Name>')
    buf += str2bin('<Val>' + str(version) + '</Val>')
    buf += str2bin('</DBL>')

    # Fixed size array of frequencies in GHz (500)
    buf += str2bin('<Array>')
    buf += str2bin('<Name>FGHz</Name>')
    buf += str2bin('<Dimsize>500</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # List of real part of attenuations (nant x npol x nfreq x natt) (8 x 500 x 2 x nant).
    # Note inverted order of dimensions
    buf += str2bin('<Array>')
    buf += str2bin('<Name>FEM_Attn_Real</Name>')
    buf += str2bin(
        '<Dimsize>500</Dimsize><Dimsize>2</Dimsize><Dimsize>'+str(nant)+'</Dimsize><Dimsize>8</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # End cluster
    buf += str2bin('</Cluster>')  # End DCM_Attn_Val cluster

    return buf


def refcal2xml(t=Time.now()):
    ''' Writes the XML description of the reference calibration table.
        The values are complex numbers.
        Returns a binary representation of the xml text file, for 
        putting into the SQL database.  The version number
        must be incremented each time there is a change to the structure 
        of this header.  Updated for version 2 (16 antennas).
    '''
    version = cal_types()[8][2]

    if version == 2:
        nant = 15
    elif version == 3:
        nant = 16

    buf = b''
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name>REFCAL</Name>')
    buf += str2bin('<NumElts>10</NumElts>')

    # Timestamp (double) [s, in LabVIEW format]
    # Time of creation of the table (precise time not critical)
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Timestamp</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Version of this XML file.  This number should be incremented each
    # time there is a change to the structure of this file.
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Version</Name>')
    buf += str2bin('<Val>' + str(version) + '</Val>')
    buf += str2bin('</DBL>')

    # Timestamp of the gaincal (double) of [s, in LabVIEW format]
    # Time of creation of the table (precise time not critical)
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>T_gcal</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Timestamp of the begin time of refcal (double) of [s, in LabVIEW format]
    # Time of creation of the table (precise time not critical)
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>T_beg</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Timestamp of the end time of refcal (double) of [s, in LabVIEW format]
    # Time of creation of the table (precise time not critical)
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>T_end</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # List of averaged band frequencies in GHz.
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Fghz</Name>')
    buf += str2bin(
        '<Dimsize>52</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # List of real part of reference calibration (nant x npol x nband) (nant x 2 x 52).
    # Note inverted order of dimensions
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Refcal_Real</Name>')
    buf += str2bin(
        '<Dimsize>52</Dimsize><Dimsize>2</Dimsize><Dimsize>'+str(nant)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # List of imaginary part of reference calibration (nant x npol x nband) (nant x 2 x 52).
    # Note inverted order of dimensions
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Refcal_Imag</Name>')
    buf += str2bin(
        '<Dimsize>52</Dimsize><Dimsize>2</Dimsize><Dimsize>'+str(nant)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # List of sigmas of reference calibration (nant x npol x nband) (nant x 2 x 52).
    # Note inverted order of dimensions
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Refcal_Sigma</Name>')
    buf += str2bin(
        '<Dimsize>52</Dimsize><Dimsize>2</Dimsize><Dimsize>'+str(nant)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # List of flags of reference calibration (nant x npol x nband) (nant x 2 x 52).
    # Note inverted order of dimensions
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Refcal_Flag</Name>')
    buf += str2bin(
        '<Dimsize>52</Dimsize><Dimsize>2</Dimsize><Dimsize>'+str(nant)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # End cluster
    buf += str2bin('</Cluster>')  # End Refcal cluster

    return buf

#def refcal2xml52(t=Time.now()):
#    ''' Writes the XML description of the 52-band reference calibration table.
#        The values are complex numbers.
#        Returns a binary representation of the xml text file, for 
#        putting into the SQL database.  The version number
#        must be incremented each time there is a change to the structure 
#        of this header.  Updated to work with version 3 (16 antennas)
#    '''
#    version = cal_types(t)[12][2]
#
#    if version == 2:
#        nant = 15
#    elif version == 3:
#        nant = 16
#
#    buf = b''
#    buf += str2bin('<Cluster>')
#    buf += str2bin('<Name>REFCAL</Name>')
#    buf += str2bin('<NumElts>10</NumElts>')
#
#    # Timestamp (double) [s, in LabVIEW format]
#    # Time of creation of the table (precise time not critical)
#    buf += str2bin('<DBL>')
#    buf += str2bin('<Name>Timestamp</Name>')
#    buf += str2bin('<Val></Val>')
#    buf += str2bin('</DBL>')
#
#    # Version of this XML file.  This number should be incremented each
#    # time there is a change to the structure of this file.
#    buf += str2bin('<DBL>')
#    buf += str2bin('<Name>Version</Name>')
#    buf += str2bin('<Val>' + str(version) + '</Val>')
#    buf += str2bin('</DBL>')
#
#    # Timestamp of the gaincal (double) of [s, in LabVIEW format]
#    # Time of creation of the table (precise time not critical)
#    buf += str2bin('<DBL>')
#    buf += str2bin('<Name>T_gcal</Name>')
#    buf += str2bin('<Val></Val>')
#    buf += str2bin('</DBL>')
#
#    # Timestamp of the begin time of refcal (double) of [s, in LabVIEW format]
#    # Time of creation of the table (precise time not critical)
#    buf += str2bin('<DBL>')
#    buf += str2bin('<Name>T_beg</Name>')
#    buf += str2bin('<Val></Val>')
#    buf += str2bin('</DBL>')
#
#    # Timestamp of the end time of refcal (double) of [s, in LabVIEW format]
#    # Time of creation of the table (precise time not critical)
#    buf += str2bin('<DBL>')
#    buf += str2bin('<Name>T_end</Name>')
#    buf += str2bin('<Val></Val>')
#    buf += str2bin('</DBL>')
#
#    # List of averaged band frequencies in GHz.
#    buf += str2bin('<Array>')
#    buf += str2bin('<Name>Fghz</Name>')
#    buf += str2bin(
#        '<Dimsize>34</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
#    buf += str2bin('</Array>')
#
#    # List of real part of reference calibration (nant x npol x nband) (nant x 2 x 52).
#    # Note inverted order of dimensions
#    buf += str2bin('<Array>')
#    buf += str2bin('<Name>Refcal_Real</Name>')
#    buf += str2bin(
#        '<Dimsize>52</Dimsize><Dimsize>2</Dimsize><Dimsize>'+str(nant)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
#    buf += str2bin('</Array>')
#
#    # List of imaginary part of reference calibration (nant x npol x nband) (nant x 2 x 52).
#    # Note inverted order of dimensions
#    buf += str2bin('<Array>')
#    buf += str2bin('<Name>Refcal_Imag</Name>')
#    buf += str2bin(
#        '<Dimsize>52</Dimsize><Dimsize>2</Dimsize><Dimsize>'+str(nant)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
#    buf += str2bin('</Array>')
#
#    # List of sigmas of reference calibration (nant x npol x nband) (nant x 2 x 52).
#    # Note inverted order of dimensions
#    buf += str2bin('<Array>')
#    buf += str2bin('<Name>Refcal_Sigma</Name>')
#    buf += str2bin(
#        '<Dimsize>52</Dimsize><Dimsize>2</Dimsize><Dimsize>'+str(nant)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
#    buf += str2bin('</Array>')
#
#    # List of flags of reference calibration (nant x npol x nband) (nant x 2 x 52).
#    # Note inverted order of dimensions
#    buf += str2bin('<Array>')
#    buf += str2bin('<Name>Refcal_Flag</Name>')
#    buf += str2bin(
#        '<Dimsize>52</Dimsize><Dimsize>2</Dimsize><Dimsize>'+str(nant)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
#    buf += str2bin('</Array>')
#
#    # End cluster
#    buf += str2bin('</Cluster>')  # End Refcal cluster
#
#    return buf


def phacal2xml(t=Time.now()):
    ''' Writes the XML description of the phase calibration table.
        The values are complex numbers.
        Returns a binary representation of the xml text file, for 
        putting into the SQL database.  The version number
        must be incremented each time there is a change to the structure 
        of this header.  Updated to work with version 3 (16 antennas)
    '''
    version = cal_types(t)[9][2]

    if version == 2:
        nant = 15
    elif version == 3:
        nant = 16

    buf = b''
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name>PHASECAL</Name>')
    buf += str2bin('<NumElts>12</NumElts>')

    # Timestamp (double) [s, in LabVIEW format]
    # Default is the timestamp of the daily phase calibration
    # Time of creation of the table (precise time not critical)
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Timestamp</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Version of this XML file.  This number should be incremented each
    # time there is a change to the structure of this file.
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Version</Name>')
    buf += str2bin('<Val>' + str(version) + '</Val>')
    buf += str2bin('</DBL>')

    # Timestamp of the reference gain calibration (double) of [s, in LabVIEW format]
    # Time of creation of the table (precise time not critical)
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>T_refcal</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # List of multi-band delay of daily phase calibration relative to refcal (nant x npol x [phase_offset, phase_slope]) (nant x 2 x 2).
    # Note inverted order of dimensions
    buf += str2bin('<Array>')
    buf += str2bin('<Name>MBD</Name>')
    buf += str2bin(
        '<Dimsize>2</Dimsize><Dimsize>2</Dimsize><Dimsize>'+str(nant)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # List of flag of multi-band delay of daily phase calibration (nant x npol x [phase_offset, phase_slope]) (nant x 2 x 2).
    # Note inverted order of dimensions
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Flag</Name>')
    buf += str2bin(
        '<Dimsize>2</Dimsize><Dimsize>2</Dimsize><Dimsize>'+str(nant)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # Timestamp of the begin time of phacal (double) of [s, in LabVIEW format]
    # Time of creation of the table (precise time not critical)
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>T_beg</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Timestamp of the end time of phacal (double) of [s, in LabVIEW format]
    # Time of creation of the table (precise time not critical)
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>T_end</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # List of averaged band frequencies in GHz.
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Fghz</Name>')
    buf += str2bin(
        '<Dimsize>52</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # List of real part of daily phase calibration (nant x npol x nband) (nant x 2 x 52).
    # Note inverted order of dimensions
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Phacal_Amp</Name>')
    buf += str2bin(
        '<Dimsize>52</Dimsize><Dimsize>2</Dimsize><Dimsize>'+str(nant)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # List of imaginary part of daily phase calibration (nant x npol x nband) (nant x 2 x 52).
    # Note inverted order of dimensions
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Phacal_Pha</Name>')
    buf += str2bin(
        '<Dimsize>52</Dimsize><Dimsize>2</Dimsize><Dimsize>'+str(nant)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # List of sigmas of daily phase calibration (nant x npol x nband) (nant x 2 x 52).
    # Note inverted order of dimensions
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Phacal_Sigma</Name>')
    buf += str2bin(
        '<Dimsize>52</Dimsize><Dimsize>2</Dimsize><Dimsize>'+str(nant)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # List of flags of daily phase calibration (nant x npol x nband) (nant x 2 x 52).
    # Note inverted order of dimensions
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Phacal_Flag</Name>')
    buf += str2bin(
        '<Dimsize>52</Dimsize><Dimsize>2</Dimsize><Dimsize>'+str(nant)+'</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')

    # End cluster
    buf += str2bin('</Cluster>')  # End Phacal cluster

    return buf


def send_xml2sql(caltype=None, t=None, test=False, nant=None, nfrq=None):
    ''' Routine to send any changed calibration xml definitions to the 
        SQL Server.  The latest definition (if any) for a given type is
        checked to see if the version matches.  If not, an update is 
        stored.  This routine will typically be run whenever a definition
        is added or changed.  If type is provided (i.e. not None), only 
        the given type will be updated (and only if its internal version 
        number has changed).

        The timestamp of the new record will be set according to the Time()
        object t, if provided, or the current time if not.

        As a debugging tool, if test is True, this routine goes through the
        motions but does not write to the abin table.
    '''
    if t is None:
        t = Time.now()
    timestamp = int(t.lv)  # Current time as LabVIEW timestamp
    cnxn, cursor = dbutil.get_cursor()
    typdict = cal_types()
    if caltype:
        # If a particular type is specified, limit the action to that type
        typdict = {caltype: typdict[caltype]}
    for key in typdict.keys():
        # print 'Working on',typdict[key][0]
        # Execute the code to create the xml description for this key
        if key == 1 or key == 10 or key == 13:
            # Special case for TP calibration
            if nant is None or nfrq is None:
                print('For', typdict[key][0], 'values for both nant and nfrq are required.')
                cnxn.close()
                return
            exec('buf = ' + typdict[key][1] + '(nant=' + str(nant) + ',nfrq=' + str(nfrq) + ')')
        else:
            print('Creating buffer')
            bufcmd = 'buf = ' + typdict[key][1] + '(); f = open("/tmp/tmp.xml","wb"); f.write(buf); f.close()'
            exec(bufcmd)
        # Resulting buf must be written to a temporary file and reread
        # by xml_ptrs()
        f = open('/tmp/tmp.xml', 'rb')
        buf = f.read()
        f.close()

        mydict, xmlver = read_xml2.xml_ptrs('/tmp/tmp.xml')
        defn_version = float(key) + xmlver / 10.  # Version number expected
        # Retrieve most recent key.0 record and check its version against the expected one
        if str(type(cursor)).find('pyodbc') == -1:
            query1 = 'set textsize 2147483647 select * from abin '
            query2 = ' limit 1'
        else:
            query1 = 'set textsize 2147483647 select top 1 * from abin '
            query2 = ''
        query = query1+'where Version = ' + str(key) + '.0 and Timestamp <= ' + str(timestamp) + ' order by Timestamp desc, Id desc'+query2
        # print 'Executing query'
        outdict, msg = dbutil.do_query(cursor, query)
        print(msg)
        if msg == 'Success':
            if len(outdict) == 0:
                # This type of xml file does not yet exist in the database, so mark it for adding
                add = True
            else:
                # There is one, so see if they differ
                buf2 = outdict['Bin'][0]  # Binary representation of xml file
                if buf == buf2:
                    # This description is already there so we will skip it
                    add = False
                else:
                    add = True
        else:
            # Some kind of error occurred, so print the message and skip adding the xml description
            # print 'Query',query,'resulted in error message:',msg
            add = False
        if add:
            # This is either new or updated, so add the xml description
            # to the database
            # print 'Trying to add',typdict[key][0]
            try:
                if test:
                    print('Would have updated', typdict[key][0], 'to version', defn_version)
                else:
                    if query2 != '':
                        if cnxn.server_host == 'localhost':
                            # This is MySQL at OVRO, so write directly to abin table
                            tbl = 'abin'
                        else:
                            # This is the cloud MySQL, so write to the "staging" table
                            tbl = 'staging'
                        cnxn.cursor().execute('insert into '+tbl+' (Timestamp,Version,Description,Bin) values (%s, %s, %s, %s)',
                                   (timestamp, key, typdict[key][0], buf))
                        cnxn.commit()
                    else:
                        # This is the MS SQL at OVRO, so write into ALL THREE databases to keep them in sync
                        from eovsactl import sql2mysql
                        sql2mysql.abin2all3(timestamp, key, typdict[key][0], buf)
                    print('Type definition for', typdict[key][0], 
                          'successfully added/updated to version', defn_version, '--OK')
            except:
                print('Unknown error occurred in adding', typdict[key][0])
                print(sys.exc_info()[1])
        else:
            print('Type definition for', typdict[key][0], 'version', defn_version, 'exists--OK')
    cnxn.close()

def read_cal_xml(caltype, t=None):
    ''' Read the calibration type definition xml record of the given type, for the 
        given time (as a Time() object), or for the current time if None.

        Returns a dictionary of look-up information and its internal version.  A side-effect
        is that a file /tmp/type<n>.xml is created, where <n> is the type.
    '''
    if t is None:
        t = Time.now()
    timestamp = int(t.lv)  # Given (or current) time as LabVIEW timestamp
    typdict = cal_types()
    try:
        typinfo = typdict[caltype]
    except:
        print('Type', caltype, 'not found in type definition dictionary.')
        return {}, None
    cnxn, cursor = dbutil.get_cursor()
    if str(type(cursor)).find('pyodbc') == -1:
        query1 = 'select '
        query2 = ' limit 1'
    else:
        query1 = 'set textsize 2147483647 select top 1 '
        query2 = ''

    # Read type definition XML from abin table
    query = query1+'* from abin where abs(Version - ' + str(caltype) + '.0) < 1e-6 and Timestamp <=' + str(
        timestamp) + ' order by Timestamp desc, Id desc'+query2
    sqldict, msg = dbutil.do_query(cursor, query)
    if msg == 'Success':
        if len(sqldict) == 0:
            # This type of xml file does not yet exist in the database, so mark it for adding
            print('Type', caltype, 'not defined in abin table.')
            cnxn.close()
            return {}, None
        else:
            # There is one, so read it and the corresponding binary data
            buf = sqldict['Bin'][0]  # Binary representation of xml file
            xmlfile = '/tmp/type' + str(caltype) + '.xml'
            f = open(xmlfile, 'wb')
            f.write(buf)
            f.close()
            xmldict, thisver = read_xml2.xml_ptrs(xmlfile)
            cnxn.close()
            os.system('rm -rf {}'.format(xmlfile))
            return xmldict, thisver


#def read_cal_xmlX(caltype, t=None, verbose=True, neat=False, gettime=False):
#   ''' Read the calibration type definition xml record of the given type, for the
#       given time or time-range (as a Time() object), or for the current time if None.
#       :param caltype:
#       :param t:
#       :param verbose:
#       :param neat: If True, throw away the obsolete records if t is time range.
#       Returns a dictionary of look-up information and its internal version.  A side-effect
#       is that a file /tmp/type<n>.xml is created, where <n> is the type.
#   '''
#   if t is None:
#       t = Time.now()
#
#   try:
#       if len(t) >= 2:
#           timestamp = [int(ll.lv) for ll in t]
#           timestamp = [timestamp[0], timestamp[-1]]
#       tislist = True
#   except:
#       timestamp = int(t.lv)  # Given (or current) time as LabVIEW timestamp
#       tislist = False
#
#   typdict = cal_types()
#   try:
#       typinfo = typdict[caltype]
#   except:
#       print('Type', caltype, 'not found in type definition dictionary.')
#       return {}, None
#   cnxn, cursor = dbutil.get_cursor()
#   if str(type(cursor)).find('pyodbc') == -1:
#       query1 = 'select '
#       query2 = ' limit 1'
#   else:
#       query1 = 'set textsize 2147483647 select top 1 '
#       query2 = ''
#   # Read type definition XML from abin table
#   if tislist:
#       query = 'select * from abin where Version = ' + str(caltype) + '.0 and Timestamp between ' + str(
#           timestamp[0]) + ' and ' + str(
#           timestamp[1]) + ' order by Timestamp desc, Id desc'
#   else:
#       query = query1+'* from abin where Version = ' + str(caltype) + '.0 and Timestamp between ' + str(
#           timestamp[0]) + ' and ' + str(
#           timestamp[1]) + ' order by Timestamp desc, Id desc'+query2
#   sqldict, msg = dbutil.do_query(cursor, query)
#   cnxn.close()
#   if msg == 'Success':
#       if len(sqldict) == 0:
#           if verbose:
#               # This type of xml file does not yet exist in the database, so mark it for adding
#               print('Type', caltype, 'not defined in abin table.')
#           return {}, None
#       else:
#           if tislist:
#               tlist = [Time(ll, format='lv').iso for ll in sqldict['Timestamp']]
#               tlistc = sorted(list(set(tlist)), reverse=True)
#               if neat:
#                   idxs = [tlist.index(ll) for ll in tlistc]
#               else:
#                   idxs = list(range(len(sqldict['Timestamp'])))
#               if verbose:
#                   print('{} records are found in {} ~ {}.'.format(len(idxs), t[0].iso, t[-1].iso))
#                   for idx, ll in enumerate(idxs):
#                       t = Time(sqldict['Timestamp'][ll], format='lv')
#                       ver = sqldict['Version'][ll]
#                       print('{} ---> ver {} {}'.format(idx + 1, ver, t.iso))
#               xml, ver = [], []
#               for idx, ll in enumerate(idxs):
#                   # There is one, so read it and the corresponding binary data
#                   buf = sqldict['Bin'][ll]  # Binary representation of xml file
#                   xmlfile = '/tmp/type' + str(caltype) + '_tmp.xml'
#                   f = open(xmlfile, 'wb')
#                   f.write(buf)
#                   f.close()
#                   xmldict, thisver = read_xml2.xml_ptrs(xmlfile)
#                   xml.append(xmldict)
#                   ver.append(thisver)
#               os.system('rm -rf {}'.format(xmlfile))
#               if gettime:
#                   ts = [tlist[ll] for ll in idxs]
#                   return xml, ver, ts
#               else:
#                   return xml, ver
#           else:
#               # There is one, so read it and the corresponding binary data
#               buf = sqldict['Bin'][0]  # Binary representation of xml file
#               xmlfile = '/tmp/type' + str(caltype) + '.xml'
#               f = open(xmlfile, 'wb')
#               f.write(buf)
#               f.close()
#               xmldict, thisver = read_xml2.xml_ptrs(xmlfile)
#               return xmldict, thisver

def read_cal(caltype, t=None, verbose=False):
    ''' Read the calibration data of the given type, for the given time (as a Time() object),
        or for the current time if None.

        Returns a dictionary of look-up information and a binary buffer containing the 
        calibration record.
    '''
    if t is None:
        t = Time.now()
    timestamp = int(t.lv)  # Given (or current) time as LabVIEW timestamp
    typdict = cal_types()
    xmldict, ver = read_cal_xml(caltype, t)
    cnxn, cursor = dbutil.get_cursor()
    if str(type(cursor)).find('pyodbc') == -1:
        query1 = 'select '
        query2 = ' limit 1'
    else:
        query1 = 'set textsize 2147483647 select top 1 '
        query2 = ''

    if xmldict != {}:
        query = query1+'* from abin where abs(Version - ' + str(
            caltype + ver / 10.) + ') < 1e-6 and Timestamp <= ' + str(timestamp) + ' order by Timestamp desc, Id desc'+query2
        sqldict, msg = dbutil.do_query(cursor, query)
        cnxn.close()
        if msg == 'Success':
            if sqldict == {}:
                print('Error: Query returned no records.')
                print(query)
                return {}, None
            buf = sqldict['Bin'][0]  # Binary representation of data
            # Next two lines extends XML and buffer to add the SQL timestamp of the record read.
            # This can be useful for error checking.
            xmldict.update({'SQL_timestamp':['d',len(buf)]})   # Adds new keyword and double definition
            buf += struct.pack('d',sqldict['Timestamp'][0])    # Appends SQL timestamp to buffer
            tstr = Time(extract(buf,xmldict['Timestamp']),format='lv').iso[:19]
            sstr = Time(extract(buf,xmldict['SQL_timestamp']),format='lv').iso[:19]
            if verbose: print('Read',typdict[caltype][0],'at SQL time',sstr,'taken at',tstr)
            return xmldict, buf #str(buf)
        else:
            print('Unknown error occurred reading', typdict[caltype][0])
            print(sys.exc_info()[1])
            return {}, None
    else:
        return {}, None


def read_calX(caltype, t=None, nrecords=1, verbose=True, neat=False, gettime=False, reverse=False):
    """
    Read the calibration data for the given type at a specific time or over a time range.

    If t is a single Time object (or None, which defaults to the current time), the function
    returns the record nearest to that time. If t is provided as a Time array, list, or tuple
    of two Time objects, it returns all records within that time range.

    :param caltype: Calibration type identifier, defaults to None.
    :type caltype: int or str (optional)
    :param t: A Time object for a single time or a collection of two Time objects specifying a time range,
              defaults to the current time.
    :type t: astropy.time.Time, or list(Time) or tuple(Time) (optional)
    :param nrecords: Number of records to return when a single time is provided; set to 0 to return all records,
                     defaults to 1.
    :type nrecords: int (optional)
    :param verbose: If True, print debugging information, defaults to True.
    :type verbose: bool (optional)
    :param neat: If True, discard obsolete records when a time range is provided, defaults to False.
    :type neat: bool (optional)
    :param gettime: If True, also return a list of timestamp strings for the returned records, defaults to False.
    :type gettime: bool (optional)
    :param reverse: If True, sort results in ascending order by Timestamp; otherwise, descending, defaults to False.
    :type reverse: bool (optional)

    :raises OSError: If an error occurs during file operations.
    :raises Exception: If an error occurs during the database query or XML processing.

    :return: A tuple containing a dictionary of lookup information and a binary buffer (or list of binary buffers)
             of the calibration record. If gettime is True, a list of timestamp strings is also returned.
    :rtype: tuple(dict, bytes) or tuple(dict, list(bytes)) or tuple(dict, list(bytes), list(str))
    """
    if t is None:
        t = Time.now()

    try:
        if len(t) >= 2:
            timestamp = [int(ll.lv) for ll in t]
            timestamp = [timestamp[0], timestamp[-1]]
            xmldict, ver = read_cal_xml(caltype, t[0])
        tislist = True
    except:
        timestamp = int(t.lv)  # Given (or current) time as LabVIEW timestamp
        xmldict, ver = read_cal_xml(caltype, t)
        tislist = False
    typdict = cal_types()
    cnxn, cursor = dbutil.get_cursor()
    if str(type(cursor)).find('pyodbc') == -1:
        query1 = 'select '
        query2 = ' limit ' + str(nrecords)
    else:
        query1 = 'set textsize 2147483647 select top ' + str(nrecords) + ' ' \
            if nrecords >0 else 'set textsize 2147483647 select '
        query2 = ''

    if xmldict != {}:
        vers = caltype + ver/10.
        ver1 = vers - 0.05
        ver2 = vers + 0.05
        if tislist:
            if reverse:
                query = query1 + '* from abin where Version between ' + str(ver1) + ' and ' + str(
                        ver2) + ' and Timestamp between ' + str(timestamp[0]) + ' and ' + str(timestamp[1]) + ' order by Timestamp asc, Id desc'
            else:
                query = query1 + '* from abin where Version between ' + str(ver1) + ' and ' + str(
                        ver2) + ' and Timestamp between ' + str(timestamp[0]) + ' and ' + str(timestamp[1]) + ' order by Timestamp desc, Id desc'            
        else:
            query = query1 + '* from abin where Version between ' + str(ver1) + ' and ' + str( 
                        ver2) + ' and Timestamp <= ' + str(timestamp) + ' order by Timestamp desc, Id desc'+query2

        sqldict, msg = dbutil.do_query(cursor, query)
        cnxn.close()
        if msg == 'Success':
            if sqldict == {}:
                if verbose:
                    print('Error: Query returned no records.')
                    print(query)
                return {}, None
            if tislist:
                tlist = [Time(extract(ll, xmldict['Timestamp']), format='lv').iso for ll in
                         sqldict['Bin']]
                tlistc = sorted(list(set(tlist)), reverse=True)
                if neat:
                    idxs = [tlist.index(ll) for ll in tlistc]
                else:
                    idxs = list(range(len(sqldict['Timestamp'])))
                buf = sqldict['Bin'][idxs]
                if verbose:
                    print('{} records are found in {} ~ {}.'.format(len(buf), t[0].iso, t[-1].iso))
                    for idx, ll in enumerate(buf):
                        t = Time(extract(ll, xmldict['Timestamp']), format='lv')
                        print('{} ---> {}'.format(idx + 1, t.iso))
                if gettime:
                    ts = [tlist[ll] for ll in idxs]
                    return xmldict, buf, ts
                else:
                    return xmldict, buf
            else:
                buf = sqldict['Bin'][0]  # Binary representation of data
                return xmldict, buf #str(buf)
        else:
            if verbose:
                print('Unknown error occurred reading', typdict[caltype][0])
                print(sys.exc_info()[1])
            return {}, None
    else:
        return {}, None


def get_size(fmt):
    # Complicated, but clever routine to determine the size of my
    # non-Pythonic format string, in which arrays are indicated by
    # a number followed by a fmt substring in square [] brackets.
    # An example is:
    #   fmt = 'ddI448fI2iI13[II896fII896f]'
    # which means 'ddI448fI2iI' followed by 13 'II896fII896f'
    fs = fmt.split('[')
    flist = [fs[0]]
    if len(fs) != 1:
        # There was an open [, so find numbers at end of first element
        for f in fs[1:]:
            flist.append(f.split(']'))
    outlist = []
    out = flist[0]
    for f in flist:
        if type(f) != str:
            chk = f[1]
            outlist.append(val * f[0])
        else:
            chk = f
        for idx in range(1, len(chk) + 1):
            try:
                val2 = int(chk[-idx:])
                out = chk[:-idx]
                val = val2
            except:
                outlist.append(out)
                break
    outfmt = ''.join(outlist)
    return struct.calcsize('>' + outfmt)


def write_cal(caltype, buf, t=None):
    ''' Write the calibration data provided in data buffer buf of the given type, 
        for the given time (as a Time() object), or for the current time if None.
        Typcially, the time should refer to when the calibration data were taken,
        so the correct time object should be provided.

        The type keyword is a real number whose integer part indicates the type
        definition.  The fractional part must not be 0 (since this would indicate
        a type definition record rather than a data record).  The relevant type 
        definition is read from the database, and its total size is determined and 
        compared with the buffer size, as a sanity check.
        
        Returns True if success, or False if failure.
    '''
    if t is None:
        t = Time.now()
    timestamp = int(t.lv)  # Given (or current) time as LabVIEW timestamp
    typdict = cal_types()
    try:
        typinfo = typdict[int(caltype)]
    except:
        print('Type', int(caltype), 'not found in type definition dictionary.')
        return False
    cnxn, cursor = dbutil.get_cursor()
    # Read type definition XML from abin table and do a sanity check
    mysql = False
    if str(type(cursor)).find('pyodbc') == -1:
        mysql = True
        query = 'select * from abin where Version = ' + str(int(caltype)) + '.0 and Timestamp <=' + str(timestamp) + ' order by Timestamp desc, Id desc limit 1'
    else:
        query = 'select top 1 * from abin where Version = ' + str(int(caltype)) + '.0 and Timestamp <=' + str(timestamp) + ' order by Timestamp desc, Id desc'
    outdict, msg = dbutil.do_query(cursor, query)
    if msg == 'Success':
        if len(outdict) == 0:
            # This type of xml file does not yet exist in the database, so indicate an error
            print('Error: Type', caltype, 'not defined in abin table.')
            cnxn.close()
            return False
        else:
            # There is one, so read it and do a sanity check against binary data
            f = open('/tmp/tmp.xml', 'wb')
            f.write(outdict['Bin'][0])
            f.close()
            keys, mydict, fmt, ver = read_xml2.xml_read('/tmp/tmp.xml')
            binsize = get_size(fmt)
            if len(buf) == binsize:
                if mysql:
                    if cnxn.server_host.find('amazonaws') != -1:
                        # If we are connected to the Cloud MySQL, insert into the Staging table
                        cnxn.cursor().execute('insert into Staging (Timestamp,Version,Description,Bin) values (%s, %s, %s, %s)',
                               (timestamp, caltype + ver / 10., typinfo[0], buf))
                    else:
                        # If we are connected to the OVRO MySQL, insert into the abin table
                        cnxn.cursor().execute('insert into abin (Timestamp,Version,Description,Bin) values (%s, %s, %s, %s)',
                               (timestamp, caltype + ver / 10., typinfo[0], buf))
                else:
                    # If we are connected to the OVRO MS SQL, insert into all three databases
                    from eovsactl import sql2mysql
                    sql2mysql.abin2all3(timestamp, caltype + ver / 10., typinfo[0], buf)
                cnxn.close()
                return True
            else:
                print('Error: Size of buffer', len(buf), 'does not match this calibration type.  Expecting', binsize)
                cnxn.close()
                return False

def copy_cal(caltype, tfrom=None, tto=None):
    ''' Read the calibration data of the given type, for the given time tfrom (as a Time() object),
        or for the current time if None, and write a copy to time tto (as a Time() object, required).
    '''
    if tto is None:
        print('ERROR: Required "to" Time() object (tto) not given.')
        return
    if tfrom is None:
        tfrom = Time.now()
    try:
        mjd_from = tfrom.mjd
        mjd_to = tto.mjd
    except:
        print('ERROR: At least one of the "from" or "to" times is not valid.')
        return
    xml, buf = read_cal(caltype, tfrom)
    if buf is None:
        print('No such record could be found.')
        return
    else:
        sql_from = Time(extract(buf,xml['SQL_timestamp']),format='lv').iso
        print(cal_types()[caltype][0], 'record found at', sql_from)
    ddays = abs(mjd_from - mjd_to)
    ans = 'Y'
    if ddays > 2.:
        print('WARNING: The time difference of',ddays,'days seems large.') 
    ans = input('Are you sure you want to copy this record from '+sql_from[:19]+' to '+tto.iso[:19]+' [y/n]?')
    if ans.upper() == 'Y':
        print(write_cal(caltype,buf[:xml['SQL_timestamp'][1]],t=tto))
    return
    
def delete_cal(caltype, t=None, relax=False):
    ''' Locate the calibration record for the given time, verify that it is of the
        correct type, and request the user to chose ID to delete.  Also requires user
        to confirm deletion. 
        
        type:   Calibration type from cal_types()
        t:      The specified time in the form of a Time() object, which is the time of 
                  the SQL record, and must match exactly unless relax is True
        relax:  If True, the search matches the first record of the requested type that 
                  occurs prior to the requested time.

        Returns True if success, or False if failure.
    '''
    if t is None:
        print('A time (as a Time() object) must be provided.')
        return False
    if relax:
        try:
            xml, buf = read_cal(caltype, t=t)
            sqltime = str(int(extract(buf,xml['SQL_timestamp'])))
        except:
            print('Error reading SQL time for specified cal type')
            return False
    else:
        sqltime = str(int(t.lv))
        
    cc = list(dbutil.get_cursor())
    try:
        if str(type(cc[0])).find('pyodbc') == -1:
            print('Sorry, only OVRO users of MS SQL can delete records')
            return False
    except:
        pass
    result = []

    # Attempt to delete records from all three databases
    dbname = ['OVRO MS SQL','OVRO MySQL','Cloud MySQL']
    cc += list(dbutil.get_cursor('localhost'))
    cc += list(dbutil.get_cursor('amazonaws.com'))
    query = 'select * from abin where Timestamp = '+sqltime
    for n in range(3):
        cnxn = cc[n*2]
        cursor = cc[n*2+1]
        data, msg = dbutil.do_query(cursor, query)
        if msg == 'Success':
            if len(data) == 0:
                print('No records found in '+dbname[n])
                result.append(False)
            else:
                print('Found the following record(s) in '+dbname[n]+':')
                print('ID          Date/Time         Calibration Type')
                result.append(True)
                for i in range(len(data['Id'])):
                    ctype = int(data['Version'][i])
                    ctypestr = cal_types()[ctype][0]
                    tiso = Time(int(data['Timestamp'][i]),format='lv').iso
                    print(data['Id'][i],tiso,ctypestr)
                ids = input('Enter ID number(s) of record(s) to delete [xxxx yyyy zzzz]: ')
                try:
                    idlist = [int(i) for i in ids.split(' ')]
                    for id in idlist:
                        k, = np.where(id in data['Id'])
                        if len(k) == 1:
                            cnxn.cursor().execute('delete from abin where Id = '+str(id)+' and Timestamp = '+sqltime)
                            print('Ready to delete ID:', id)
                except:
                    print('ID list not understood.')
                    result[n] = False
        else:
            print('Query failed with message: '+msg)
            result.append(False)
    if any(result) == True:
        ans = input('Are you sure you want to delete these records [y/n]?')
        if ans.upper() == 'Y':
            for n in range(3):
                if result[n] == True:
                    if n == 0:
                        # MS SQL
                        cc[0].cursor().commit()
                    else:
                        # MySQL
                        cc[n*2].commit()
        else:
            for n in range(3):
                result[n] = False
    else:
        print('No records to delete.')
    for n in range(3):
        cc[n*2].close()
    return result

def proto_tpcal2sql(filename, t=None):
    ''' Writes prototype TP calibration data as a record into SQL server table abin

        This kind of record is type definition 1.
    '''
    typedef = 1
    ver = cal_types()[typedef][2]
    if t is None:
        t = Time.now()
    try:
        # Open and read TP calibration file
        npol, nfrq, nant = filename.replace('.dat', '').split('_')[1:]
        nant = int(nant)
        nsiz = int(npol) * int(nfrq) * nant
        nfi = int(nfrq)
        f = open(filename, 'rb')
        data = f.read()
        f.close()
        fghz = np.array(struct.unpack_from(nfrq + 'f', data, 0))
        calfac = np.array(struct.unpack_from(str(nsiz) + 'f',
                                             data, nfi * 4)).reshape(int(npol), int(nfi), nant)
        offsun = np.array(struct.unpack_from(str(nsiz) + 'f',
                                             data, (nfi + nsiz) * 4)).reshape(int(npol), int(nfi), nant)
    except:
        print('Error: Could not open/read file', filename)
        return False
    # For TP calibration, must explicitly write xml for this nant and nfrq, since
    # the definition changes if nant and nfrq change
    send_xml2sql(typedef, t, nant=nant, nfrq=nfi)
    # Write timestamp
    buf = struct.pack('d', int(t.lv))
    # Write version number
    buf += struct.pack('d', ver)
    buf += struct.pack('I', nfi)  # Length of frequency array
    buf += struct.pack(nfrq + 'f', *fghz)
    buf += struct.pack('I', 2)  # Length of frequency array
    buf += struct.pack('2i', *np.array([-5, -6]))  # Polarizations XX and YY
    buf += struct.pack('I', nant)  # Length of antenna array
    for i in range(nant):
        # Antenna number
        buf += struct.pack('H', i + 1)
        # Calibration factors
        buf += struct.pack('I', nfi)  # Number of frequencies
        buf += struct.pack('I', 2)  # Number of polarizations
        buf += struct.pack(str(nfi * 2) + 'f', *(calfac[:, :, i].reshape(nfi * 2)))
        # Offsun level
        buf += struct.pack('I', nfi)  # Number of frequencies
        buf += struct.pack('I', 2)  # Number of polarizations
        buf += struct.pack(str(nfi * 2) + 'f', *(offsun[:, :, i].reshape(nfi * 2)))
    return write_cal(typedef, buf, t)

def tpcal2sql(tpcal_dict, t=None):
    ''' Writes TP calibration data from the input dictionary, as a record into 
        the SQL server table abin.  The timestamp of the SQL record is given by
        parameter t, or the timestamp of the start of the SOLPNTCAL if None.

        The dictionary has the keys:
        'fghz': List of frequencies, in GHz (nf)
        'timestamp': The 'lv' format timestamp of the start of the SOLPNTCAL
        'tpcalfac': The calibration factors for total power [sfu/corrrelator-unit] (npol,nf,nant)
        'accalfac': The calibration factors for auto-correlation [sfu/correlator-unit] (npol,nf,nant)
        'tpoffsun': The offsun total power, for subtraction prior to applying 
                      calibration [correlator-units] (npol,nf,nant)
        'acoffsun': The offsun auto-correlation, for subtraction prior to applying 
                      calibration [correlator-units] (npol,nf,nant)

        This kind of record is type definition 10.
    '''
    typedef = 10
    ver = cal_types()[typedef][2]
    npol, nf, nant = tpcal_dict['tpcalfac'].shape
    if t is None:
        t = Time(tpcal_dict['timestamp'],format='lv')
    # For TP calibration, must explicitly write xml for this nant and nfrq, since
    # the definition changes if nant and nfrq change
    send_xml2sql(typedef, t, nant=nant, nfrq=nf)
    # Write timestamp of data
    tdata = Time(tpcal_dict['timestamp'],format='lv')
    buf = struct.pack('d', int(tdata.lv))
    # Write version number
    buf += struct.pack('d', ver)
    buf += struct.pack('I', nf)  # Length of frequency array
    buf += struct.pack(str(nf) + 'f', *tpcal_dict['fghz'])
    buf += struct.pack('I', 2)  # Length of frequency array
    buf += struct.pack('2i', *np.array([-5, -6]))  # Polarizations XX and YY
    buf += struct.pack('I', nant)  # Length of antenna array
    for i in range(nant):
        # Antenna number
        buf += struct.pack('H', i + 1)
        # Total Power Calibration factors
        buf += struct.pack('I', nf)  # Number of frequencies
        buf += struct.pack('I', 2)  # Number of polarizations
        buf += struct.pack(str(nf * 2) + 'f', *(tpcal_dict['tpcalfac'][:, :, i].reshape(nf * 2)))
        # Auto-Correlation Calibration factors
        buf += struct.pack('I', nf)  # Number of frequencies
        buf += struct.pack('I', 2)  # Number of polarizations
        buf += struct.pack(str(nf * 2) + 'f', *(tpcal_dict['accalfac'][:, :, i].reshape(nf * 2)))
        # Total Power Offsun level
        buf += struct.pack('I', nf)  # Number of frequencies
        buf += struct.pack('I', 2)  # Number of polarizations
        buf += struct.pack(str(nf * 2) + 'f', *(tpcal_dict['tpoffsun'][:, :, i].reshape(nf * 2)))
        # Auto-Correlation Offsun level
        buf += struct.pack('I', nf)  # Number of frequencies
        buf += struct.pack('I', 2)  # Number of polarizations
        buf += struct.pack(str(nf * 2) + 'f', *(tpcal_dict['acoffsun'][:, :, i].reshape(nf * 2)))
    return write_cal(typedef, buf, t)

def skycal2sql(skycal_dict, t=None):
    ''' Writes Skycal calibration data from the input dictionary, as a record into 
        the SQL server table abin.  The timestamp of the SQL record is given by
        parameter t, or the timestamp of the start of the SKYCAL if None.

        The dictionary has the keys:
        'fghz': List of frequencies, in GHz (nf)
        'timestamp': The 'lv' format timestamp of the start of the SKYCAL
        'rcvr_bgd': The receiver power with full 62 dB frontend attnuation [corrrelator-unit] (nant,npol,nf)
        'rcvr_bgd_auto': The same for auto-correlation [correlator-unit] (nant,npol,nf)

        This kind of record is type definition 13.
    '''
    typedef = 13
    ver = cal_types()[typedef][2]
    nant, npol, nf = skycal_dict['rcvr_bgd'].shape
    if t is None:
        t = Time(skycal_dict['timestamp'],format='lv')
    # For Skycal calibration, must explicitly write xml for this nant and nfrq, since
    # the definition changes if nant and nfrq change
    send_xml2sql(typedef, t, nant=nant, nfrq=nf)
    # Write timestamp of data
    tdata = Time(skycal_dict['timestamp'],format='lv')
    buf = struct.pack('d', int(tdata.lv))
    # Write version number
    buf += struct.pack('d', ver)
    buf += struct.pack('I', nf)  # Length of frequency array
    buf += struct.pack(str(nf) + 'f', *skycal_dict['fghz'])
    buf += struct.pack('I', 2)  # Length of frequency array
    buf += struct.pack('2i', *np.array([-5, -6]))  # Polarizations XX and YY
    buf += struct.pack('I', nant)  # Length of antenna array
    for i in range(nant):
        # Antenna number
        buf += struct.pack('H', i + 1)
        # Receiver Background
        buf += struct.pack('I', nf)  # Number of frequencies
        buf += struct.pack('I', 2)  # Number of polarizations
        buf += struct.pack(str(nf * 2) + 'f', *(skycal_dict['rcvr_bgd'][i, :, :].reshape(nf * 2)))
        # Auto-Correlation Receiver Background
        buf += struct.pack('I', nf)  # Number of frequencies
        buf += struct.pack('I', 2)  # Number of polarizations
        buf += struct.pack(str(nf * 2) + 'f', *(skycal_dict['rcvr_bgd_auto'][i, :, :].reshape(nf * 2)))
    return write_cal(typedef, buf, t)

def xy_phasecal2sql(xyphase_dict, t=None):
    ''' Writes X-Y calibration data from the input dictionary, as a record into 
        the SQL server table abin

        The dictionary has the keys:
        'fghz': List of frequencies, in GHz (nf)
        'timestamp': The 'lv' format timestamp of the start of the PHASECAL
        'xyphase': The X-Y delay phase returned by get_xy_corr() routine [radians] (nant,nf)
        'xi_rot': The phase offset that appears when the Ant14 feed is rotated [radians] (nf)

        The record is written as 500 frequencies, zero-filled for nf:500
        This kind of record is type definition 11.
    '''
    typedef = 11
    nant, nf = xyphase_dict['xyphase'].shape
    if t is None:
        # Write timestamp of data
        t = Time(xyphase_dict['timestamp'],format='lv')
    ver = cal_types(t)[typedef][2]
    if ver <= 2:
        nant = 14
    elif ver == 3:
        nant = 16
    buf = struct.pack('d', int(t.lv))
    # Write version number
    buf += struct.pack('d', ver)
    # Write frequency array
    buf += struct.pack('I', 500)  # Length of frequency array
    flist = np.zeros(500,np.float64)
    flist[:nf] = xyphase_dict['fghz']
    buf += struct.pack('500f', *flist)
    # Write xi_rot array
    buf += struct.pack('I',500)  # Length of xi_rot array
    xi_rot = np.zeros(500,np.float64)
    xi_rot[:nf] = xyphase_dict['xi_rot']
    buf += struct.pack('500f', *xi_rot)
    # Write xyphase
    buf += struct.pack('I', nant)  # Length of antenna array
    buf += struct.pack('I', 500)  # Length of frequency array
    xyout = np.zeros((nant,500),np.float64)
    xyout[:,:nf] = xyphase_dict['xyphase']
    for i in range(nant):
        buf += struct.pack('500f', *xyout[i])
    return write_cal(typedef, buf, t)


def dcm_master_table2sql(filename, tbl=None, t=None):
    ''' Writes a DCM master base attenuation calibration table as a record into 
        SQL server table abin. filename can either be a txt file (DCM_master_table.txt) 
        or a DCM_list (from the output of adc_cal2.make_DCM_table()) or a table from
        the output of adc_cal2.set_dcm_attn() [use filename='' and give the table as tbl].

        This kind of record is type definition 2.
    '''
    if t is None:
        t = Time.now()
    typedef = 2
    ver = cal_types(t)[typedef][2]

    if ver == 2:
        nant = 15
    elif ver == 3:
        nant = 16
    if tbl is None:
        # Check the format of the input file and see if it is a file or a python list
        try:
            # Open and read DCM_master_table.txt file
            if type(filename) is str:
                f = open(filename, 'r')
                lines = f.readlines()
                f.close()
            if type(filename) is list:
                lines = filename
            # Read file of attenuations (34 non-comment lines with band + nant*2 attns)
            bands = np.zeros(52, 'int')
            attn = np.zeros((52, nant*2), 'float')
            for line in lines:
                if line[0] != '#':
                    band, rline = line.strip().split(':')
                    attn[int(band) - 1] = list(map(int, rline.split()))
                    bands[int(band) - 1] = band
        except:
            print('Error: Could not open/read file', filename)
            return False
    else:
        # Standard table was input, so interpret as output from adc_cal.set_dcm_attn()
        bands = np.linspace(1, 52, 52).astype(int)
        attn = tbl[:, :nant*2]

    # Write timestamp
    buf = struct.pack('d', int(t.lv))
    # Write version number
    buf += struct.pack('d', ver)
    buf += struct.pack('I', 52)
    buf += struct.pack('52H', *bands)
    buf += struct.pack('I', 52)
    buf += struct.pack('I', nant*2)
    for i in range(52):
        buf += struct.pack(str(nant*2)+'H', *attn[i])
    return write_cal(typedef, buf, t)


def dcm_table2sql(filename, t=None):
    ''' Writes an instance of DCM table attenuation calibration for the 
        current scan as a record into SQL server table abin. filename can 
        either be a txt file (DCM_table.txt) or a list

        This kind of record is type definition 3.
    '''
    if t is None:
        t = Time.now()
    typedef = 3
    ver = cal_types(t)[typedef][2]

    if ver == 1:
        nant = 15
    elif ver == 2:
        nant = 16

    try:
        if type(filename) is str:
            # Open and read DCM_table.txt file
            f = open(filename, 'r')
            lines = f.readlines()
            f.close()
        if type(filename) is list:
            lines = filename
        # Read file of attenuations (50 non-comment lines with nant*2 attns)
        attn = np.zeros((50, nant*2), 'float')
        for i, line in enumerate(lines):
            attn[i] = list(map(int, line.split()))
    except:
        print('Error: Could not open/read file', filename)
        return False
    # Write timestamp
    buf = struct.pack('d', int(t.lv))
    # Write version number
    buf += struct.pack('d', ver)
    buf += struct.pack('I', 50)
    buf += struct.pack('I', nant*2)
    for i in range(50):
        buf += struct.pack(str(nant*2)+'H', *attn[i])
    return write_cal(typedef, buf, t)


def dla_update2sql(dla_update, xy_delay=None, t=None, lorx=False):
    ''' Write delay_center updates to SQL server table abin,
        with the timestamp given by Time() object t (or current time, if none)

        Input:
          dla_update   a 14-element array of delay differences [ns], already
                          converted to be relative to Ant 1
          xy_delay     an optional 14-element array of delay differences [ns] in 
                          Y relative to X
        Optional argument:
          lorx         if True, ONLY the Ant 14 delay update is applied, and ONLY
                          to the Ant 15 slot, which is the delay setting to use
                          for Ant 14 Lo-frequency receiver.
                          
        Updated to version 2 (16 antenna version)
    '''
    if dla_update[0] != 0.0:
        print('First delay in list is not zero.  Delays must be relative to Ant 1')
        return False
    if t is None:
        # If no time is defined, use the current time
        t = Time.now()
    typedef = 4
    ver = cal_types()[typedef][2]

    if ver == 1:
        nant = 14  # Old data had 15 slots but used the last one for Ant 14 LoRX delays
    elif ver == 2:
        nant = 16  # New data have 17 slots and use the last one for Ant 16 (Ant A) LoRX delays

    if xy_delay is None:
        # If no xy_delay was given, use zeros
        xy_delay = np.zeros(nant, dtype=float)
    # Read the SQL database to get delay_centers current at the given time
    xml, buf = read_cal(4, t)
    dcen = extract(buf, xml['Delaycen_ns'])
    # Apply corrections, forcing them to be relative to ant 1 (reference antenna),
    rel_dla_ns = dla_update
    xy_dla_ns = xy_delay  # XY delay is not relative to Ant 1
    if lorx:
        # This is the case of updating the Lo-frequency receiver delays ONLY
        # Only change Ant 15 entries, using the Ant 14 information
        dcen[nant, 0] -= rel_dla_ns[nant-1]
        dcen[nant, 1] -= rel_dla_ns[nant-1] + xy_dla_ns[nant-1]
    else:
        dcen[:nant, 0] -= rel_dla_ns
        dcen[:nant, 1] -= rel_dla_ns + xy_dla_ns

    # Write timestamp
    buf = struct.pack('d', int(t.lv))
    # Write version number
    buf += struct.pack('d', ver)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', nant+1)
    for i in range(nant+1):
        buf += struct.pack('2f', *dcen[i])
    return write_cal(typedef, buf, t)


def dla_censql2table(t=None, acc=False):
    ''' Reads current database contents for delay centers and writes
        them out as a text table to a fixed location, /tmp/delay_centers.txt,
        and optionally sends the file to the ACC (for use by dppxmp).

        acc   boolean: if True, write the file to the ACC. This only
              works if at OVRO, and requires the eovsactl package.
    '''
    import time
    typedef = 4
    if t is None:
        t = Time.now()
    else:
        # Ensure that an old table is not written to ACC
        print('Warning! Specifying a time disables writing the file to the ACC.')
        print('FTP the file /tmp/delay_centers.txt by hand if that is what you intend.')
        acc = False
    ver = cal_types(t)[typedef][2]

    if ver == 1:
        nant = 15
    elif ver == 2:
        nant = 16

    xml, buf = read_cal(typedef, t)
    delays = extract(buf, xml['Delaycen_ns'])
    timestr = Time(int(extract(buf, xml['Timestamp'])), format='lv').iso
    f = open('/tmp/delay_centers.txt', 'w')
    f.write('# Antenna delay centers, in nsec, relative to Ant 1\n')
    f.write('#     Date: ' + timestr + '\n')
    f.write('# Note: For historical reasons, dppxmp needs four header lines\n')
    f.write('# Ant  X Delay[ns]  Y Delay[ns]\n')
    fmt = '{:4d}   {:10.3f}   {:10.3f}\n'
    for i in range(nant+1):
        f.write(fmt.format(i + 1, *delays[i]))
    f.close()
    if acc:
        time.sleep(1)  # Make sure file has time to be closed.
        try:
            from eovsactl.accFTP import ACC_stor
            ACC_stor('/tmp/delay_centers.txt')
        except:
            print('FTP to ACC failed.  Note this can only work at OVRO')

def dla_update2table(dla_update, xy_delay=None, t=None, lorx=False, acc=True):
    ''' Write delay_center updates to a standard delay_centers.txt file
        in the /nas4/Tables/Delays folder.  Optionally send the table to the ACC

        Input:
          dla_update   a 14-element array of delay differences [ns], already
                          converted to be relative to Ant 1
          xy_delay     an optional 14-element array of delay differences [ns] in 
                          Y relative to X
        Optional argument:
          t            the Time object from which to get the Time string for
                          the file header (uses the current time if None)
          lorx         if True, ONLY the Ant 14 delay update is applied, and ONLY
                          to the Ant 15 slot, which is the delay setting to use
                          for Ant 14 Lo-frequency receiver.
          acc          if True (default), the table is also written to the ACC

        Updated to work with 16 antennas.  No need to generalize for backward
        compatibility, since this only applies to the current time.
    '''
    import time
    if dla_update[0] != 0.0:
        print('First delay in list is not zero.  Delays must be relative to Ant 1')
        return False
    if t is None:
        # If no time is defined, use the current time
        t = Time.now()
    if xy_delay is None:
        # If no xy_delay was given, use zeros
        xy_delay = np.zeros(14, dtype=float)
    # Read the current ACC table to get delay_centers current at the given time
    dladict = ACCdlatable2dict()
    if dladict == {}:
        print('Error retrieving the delay_centers.txt from the ACC')
        return False
    dcen = dladict['Delaycen_ns']
    # Apply corrections, forcing them to be relative to ant 1 (reference antenna),
    rel_dla_ns = dla_update
    xy_dla_ns = xy_delay  # XY delay is not relative to Ant 1
    if lorx:
        # This is the case of updating the Lo-frequency receiver delays ONLY
        # Only change Ant 15 entries, using the Ant 14 information
        dcen[16, 0] -= rel_dla_ns[13]
        dcen[16, 1] -= rel_dla_ns[13] + xy_dla_ns[13]
    else:
        dcen[:16, 0] -= rel_dla_ns
        dcen[:16, 1] -= rel_dla_ns + xy_dla_ns

    timestr = t.iso
    datstr = timestr[:19].replace('-','').replace(' ','_').replace(':','')
    filename = '/nas4/Tables/Delays/delay_centers_'+datstr+'.txt'
    f = open(filename, 'w')
    f.write('# Antenna delay centers, in nsec, relative to Ant 1\n')
    f.write('#     Date: ' + timestr + '\n')
    f.write('# Note: For historical reasons, dppxmp needs four header lines\n')
    f.write('# Ant  X Delay[ns]  Y Delay[ns]\n')
    fmt = '{:4d}   {:10.3f}   {:10.3f}\n'
    for i in range(17):
        f.write(fmt.format(i + 1, *dcen[i]))
    f.close()
    if acc:
        time.sleep(1)  # Make sure file has time to be closed.
        try:
            from eovsactl.accFTP import ACC_stor
            ACC_stor('/tmp/delay_centers.txt')
        except:
            print('FTP to ACC failed.  Note this can only work at OVRO')
    
def dla_centable2sql(filename='/tmp/delay_centers_tmp.txt', t=None):
    ''' Write delays given in file filename to SQL server table abin
        with the timestamp given by Time() object t (or current time, if none)

        This kind of record is type definition 4.

        Updated to handle version 2 (16 antennnas)
    '''
    if t is None:
        t = Time.now()
    typedef = 4
    ver = cal_types(t)[typedef][2]
    
    if ver == 1:
        nant = 15
    elif ver == 2:
        nant = 16

    try:
        # Open and read file of delay_center values
        f = open(filename, 'r')
        lines = f.readlines()
        f.close()
        # Read file of delays (nant+1 non-comment lines with ant, dlax, dlay)
        tau_ns = np.zeros((nant+1, 2), 'float')
        for line in lines:
            if line[0] != '#':
                ant, xdla, ydla = line.strip().split()
                tau_ns[int(ant) - 1] = np.array([float(xdla), float(ydla)])
    except:
        print('Error: Could not open/read file', filename)
        return False

    # Write timestamp
    buf = struct.pack('d', int(t.lv))
    # Write version number
    buf += struct.pack('d', ver)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', nant+1)
    for i in range(nant+1):
        buf += struct.pack('2f', *tau_ns[i])
    return write_cal(typedef, buf, t)


def eq_gain2sql(coeff, ver=1.0, t=None):
    ''' Write coefficients to SQL server table abin, with the timestamp 
        given by Time() object t (or current time, if none)

        This kind of record is type definition 5.  This data type
        can have many "versions," i.e. tables, for different source
        types.  In particular, we will have one for calibrators and
        one for the Sun.  
           Version 1.0 => calibrator/blank sky
           Version 2.0 => Sun

        This type of record was never used.  Will have to be updated
        for 16 antennas if used in the future.   
    '''
    typedef = 5
    if t is None:
        t = Time.now()

    # Write timestamp
    buf = struct.pack('d', int(t.lv))
    # Write version number
    buf += struct.pack('d', ver)
    buf += struct.pack('I', 34)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', 2)
    for i in range(2):
        for j in range(2):
            buf += struct.pack('34f', *coeff[i, j])
    return write_cal(typedef, buf, t)


def dcm_attn_val2sql(attn, ver=1.0, t=None):
    ''' Write measured DCM attenuation values (dB) to SQL server table
        abin, with the timestamp given by Time() object t (or current
        time, if none).

        This kind of record is type definition 6.
    '''
    if t is None:
        t = Time.now()
    typedef = 6
    ver = cal_types(t)[typedef][2]

    # Write timestamp
    buf = struct.pack('d', int(t.lv))
    # Write version number
    buf += struct.pack('d', ver)

    # Write real part of table
    rattn = np.real(attn)
    buf += struct.pack('I', 4)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', 16)
    for i in range(16):
        for j in range(2):
            buf += struct.pack('4f', *rattn[i, j])
    # Write imag part of table
    iattn = np.imag(attn)
    buf += struct.pack('I', 4)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', 16)
    for i in range(16):
        for j in range(2):
            buf += struct.pack('4f', *iattn[i, j])
    return write_cal(typedef, buf, t)


def fem_attn_val2sql(attn, ver=1.0, t=None):
    ''' Write measured FEM attenuation values (dB) to SQL server table
        abin, with the timestamp given by Time() object t (or the time
        of GAINCALTEST, if none).
        
        attn   a dictionary returned by get_attncal() routine, which
               has the keys:
               'attn': The attenuation values calculated from GAINCALTEST
                       (natt, nant, npol, nfrq)
               'fghz': List of frequencies, in GHz
               'time': The timestamp of GAINCALTEST scan

        This kind of record is type definition 7.
        
        An update maybe needed to address the case where attn is None.
        
        Updated to work with 16 antennas.
    '''
    if t is None:
        t = attn[0]['time']
    typedef = 7
    ver = cal_types(t)[typedef][2]
    
    if ver == 1:
        nant = 15
    elif ver == 2:
        nant = 16

    natn, nsolant, npol, nfrq = attn[0]['attn'].shape

#    if attn is None:  # What should the default timestamp & fghz be??
#        # Create a default attn array, with nominal values
#        attnvals = np.array([2.,4.,6.,8.,10.,12.,14.,16.])  # The 8 attenuation values
#        attn = np.zeros((8,15,2,500))
#        attn[:,:,:,:] = attnvals
    # Write timestamp
    buf = struct.pack('d', int(attn[0]['time'].lv))
    # Write version number
    buf += struct.pack('d', ver)
    # Write frequencies
    buf += struct.pack('f', 500) # Length of frequency array
    flist = np.zeros(500,np.float64)
    flist[:nfrq] = attn[0]['fghz']
    buf += struct.pack('500f', *flist)

    # Write the attenuation table
    attn_pad = np.zeros((8,nant,2,500))
    attn_pad[:,:nsolant,:,:nfrq] = attn[0]['attn']
    buf += struct.pack('f', 500)
    buf += struct.pack('f', 2)
    buf += struct.pack('f', nant)
    buf += struct.pack('f', 8)
    for i in range(8):
        for j in range(nant):
            for k in range(2):
                buf += struct.pack('500f', *attn_pad[i, j, k])

    return write_cal(typedef,buf,t)


def refcal2sql(rfcal, timestamp=None):
    ''' Write reference calibration to SQL server table
        abin, with the timestamp given by Time() object t (or current
        time, if none).
        rfcal: a dict ('t_bg', 'vis', 't_gcal', 'timestamp', 'flag', 't_ed', 'fghz')

        This kind of record is type definition 8.
    '''
    if timestamp:
        t = int(timestamp.lv)
    else:
        if 'timestamp' in list(rfcal.keys()):
            t = int(rfcal['timestamp'].lv)
        else:
            t = int(Time.now().lv)
    typedef = 8

    if not 'vis' in list(rfcal.keys()):
        raise KeyError('Key "vis" not exist')
    ver = cal_types()[typedef][2]

    if ver == 2:
        nant = 15
    elif ver == 3:
        nant = 16

    if 't_gcal' in list(rfcal.keys()):
        tgcal = int(rfcal['t_gcal'].lv)
    else:
        tgcal = -1

    if 't_bg' in list(rfcal.keys()):
        tbg = int(rfcal['t_bg'].lv)
    else:
        tbg = t

    if 't_ed' in list(rfcal.keys()):
        ted = int(rfcal['t_ed'].lv)
    else:
        ted = t

    if 'flag' in list(rfcal.keys()):
        flag = rfcal['flag']
    else:
        flag = np.zeros_like(np.real(rfcal['vis']))

    if 'sigma' in list(rfcal.keys()):
        sigma = rfcal['sigma']
    else:
        sigma = np.zeros_like(np.real(rfcal['vis']))

    # Write timestamp
    buf = struct.pack('d', t)
    # Write version number
    buf += struct.pack('d', ver)
    # Write timestamp of gaincal
    buf += struct.pack('d', tgcal)
    # Write timestamp of begin time of refcal
    buf += struct.pack('d', tbg)
    # Write timestamp of end time of refcal
    buf += struct.pack('d', ted)

    # Get number of frequencies (either 34 or 52)
    nf = len(rfcal['fghz'])
    # Write table of the averaged band frequencies
    buf += struct.pack('I', nf)
    buf += struct.pack(str(nf)+'f', *rfcal['fghz'])

    # Write real part of table
    rrfcal = np.real(rfcal['vis'])
    buf += struct.pack('I', nf)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', nant)
    for i in range(nant):
        for j in range(2):
            buf += struct.pack(str(nf)+'f', *rrfcal[i, j])

    # Write imag part of table
    irfcal = np.imag(rfcal['vis'])
    buf += struct.pack('I', nf)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', nant)
    for i in range(nant):
        for j in range(2):
            buf += struct.pack(str(nf)+'f', *irfcal[i, j])

    # Write Sigma of table
    buf += struct.pack('I', nf)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', nant)
    for i in range(nant):
        for j in range(2):
            buf += struct.pack(str(nf)+'f', *sigma[i, j])

    # Write Flag table
    flag = np.array(flag, dtype=float)
    buf += struct.pack('I', nf)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', nant)
    for i in range(nant):
        for j in range(2):
            buf += struct.pack(str(nf)+'f', *flag[i, j])
    t = Time(t, format='lv')
    print('sending refcal of {} to SQL database.'.format(t.iso))
    return write_cal(typedef, buf, t)
    # return buf


def phacal2sql(phcal, timestamp=None):
    ''' Write daily phase calibration to SQL server table
        abin, with the timestamp given by Time() object t (or current
        time, if none).
        rfcal: a dict (phacal, flag, t_mid, t_bg, t_ed, t_dla, t_refcal, mbd0, mbd1, fghz)

        This kind of record is type definition 9.
    '''
    if timestamp:
        t = int(timestamp.lv)
    else:
        if 't_pha' in list(phcal.keys()):
            t = int(phcal['t_pha'].lv)
        else:
            t = int(Time.now().lv)
    typedef = 9
    if not 'phacal' in list(phcal.keys()):
        raise KeyError('Key "phacal" not exist')
    ver = cal_types(Time(t,format='lv'))[typedef][2]

    if ver == 2:
        nant = 15
    elif ver == 3:
        nant = 16

    if 't_ref' in list(phcal.keys()):
        trefcal = int(phcal['t_ref'].lv)
    else:
        trefcal = -1

    if 't_bg' in list(phcal['phacal'].keys()):
        tbg = int(phcal['phacal']['t_bg'].lv)
    else:
        tbg = -1

    if 't_ed' in list(phcal['phacal'].keys()):
        ted = int(phcal['phacal']['t_ed'].lv)
    else:
        ted = -1

    # Get number of frequencies (either 34 or 52)
    nf = len(phcal['phacal']['fghz'])
    # Write timestamp
    buf = struct.pack('d', t)
    # Write version number
    buf += struct.pack('d', ver)
    # Write timestamp of reference calibration
    buf += struct.pack('d', trefcal)

    # Write multi-band delay of table
    mbd = np.concatenate((np.expand_dims(phcal['poff'],2), np.expand_dims(phcal['pslope'],2)),axis=2)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', nant)
    for i in range(nant):
        for j in range(2):
            buf += struct.pack('2f', *mbd[i, j])

    # Write the flag of multi-band delay of table
    flag = np.concatenate((np.expand_dims(phcal['flag'], 2), np.expand_dims(phcal['flag'], 2)), axis=2)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', nant)
    for i in range(nant):
        for j in range(2):
            buf += struct.pack('2f', *flag[i, j])

    # Write timestamp of begin time of phacal
    buf += struct.pack('d', tbg)
    # Write timestamp of end time of phacal
    buf += struct.pack('d', ted)

    # Write table of the averaged band frequencies
    buf += struct.pack('I', nf)
    buf += struct.pack(str(nf)+'f', *phcal['phacal']['fghz'])

    # Write amplitude of phcal
    aphcal = np.real(phcal['phacal']['amp'])
    buf += struct.pack('I', nf)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', nant)
    for i in range(nant):
        for j in range(2):
            buf += struct.pack(str(nf)+'f', *aphcal[i, j])

    # Write phase of phcal
    pphcal = np.array(phcal['phacal']['pha'])
    buf += struct.pack('I', nf)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', nant)
    for i in range(nant):
        for j in range(2):
            buf += struct.pack(str(nf)+'f', *pphcal[i, j])

    # Write Sigma of phcal
    sigma = np.array(phcal['phacal']['sigma'])
    buf += struct.pack('I', nf)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', nant)
    for i in range(nant):
        for j in range(2):
            buf += struct.pack(str(nf)+'f', *sigma[i, j])

    # Write Flag of phcal
    phacal_flag = np.array(phcal['phacal']['flag'], dtype=float)
    buf += struct.pack('I', nf)
    buf += struct.pack('I', 2)
    buf += struct.pack('I', nant)
    for i in range(nant):
        for j in range(2):
            buf += struct.pack(str(nf)+'f', *phacal_flag[i, j])

    t = Time(t, format='lv')
    print('sending phacal of {} to SQL database.'.format(t.iso))
    return write_cal(typedef, buf, t)
    # return buf

def rstnflux2xml():
    ''' Writes the XML description of the RSTN noon flux table.
        Returns a binary representation of the xml
        text file, for putting into the SQL database.  The version number
        must be incremented each time there is a change to the structure 
        of this header.
    '''
    version = cal_types()[12][2]

    buf = b''
    buf += str2bin('<Cluster>')
    buf += str2bin('<Name>RSTNflux</Name>')
    buf += str2bin('<NumElts>4</NumElts>')

    # Timestamp (double) [s, in LabVIEW format]
    # Time of creation of the table (precise time not critical)
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Timestamp</Name>')
    buf += str2bin('<Val></Val>')
    buf += str2bin('</DBL>')

    # Version of this XML file.  This number should be incremented each
    # time there is a change to the structure of this file.
    buf += str2bin('<DBL>')
    buf += str2bin('<Name>Version</Name>')
    buf += str2bin('<Val>' + str(version) + '</Val>')
    buf += str2bin('</DBL>')
    
    # List of frequencies (9 element array).
    buf += str2bin('<Array>')
    buf += str2bin('<Name>FGHz</Name>')
    buf += str2bin('<Dimsize>9</Dimsize>\n<SGL>\n<Name></Name>\n<Val></Val>\n</SGL>')
    buf += str2bin('</Array>')
    
    # List of fluxes (9 x 7).
    buf += str2bin('<Array>')
    buf += str2bin('<Name>Flux</Name>')
    buf += str2bin('<Dimsize>7</Dimsize><Dimsize>9</Dimsize>\n<I16>\n<Name></Name>\n<Val></Val>\n</I16>')
    buf += str2bin('</Array>')

    # End cluster
    buf += str2bin('</Cluster>')  # End RSTNflux cluster

    return buf

def rstnflux2sql(data, t=None):
    ''' Write the RSTN frequencies and fluxes to SQL server table abin
        with the timestamp given by Time() object t (or current time, if none)

        This kind of record is type definition 12.
    '''
    typedef = 12
    ver = cal_types()[typedef][2]
    if t is None:
        t = Time.now()
    
    # Write timestamp
    buf = struct.pack('d', int(data[0].lv))
    # Write version number
    buf += struct.pack('d', ver)
    # Write the frequency list
    buf += struct.pack('I', 9)
    buf += struct.pack('9f', *data[1])
    # Write the flux values 
    buf += struct.pack('I', 9)
    buf += struct.pack('I', 7)
    for i in range(9):
        buf += struct.pack('7h', *data[2][i])
    return write_cal(typedef, buf, t)

    

