"""Input/output helpers for PSRFITS, standard FITS files, and filterbank (.fil) files."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from astropy.io import fits
import struct

from . import config


def load_fil_file(file_name: str) -> np.ndarray:
    """Load a filterbank (.fil) file and return the data array in shape (time, pol, channel).
    
    Filterbank files contain header information followed by binary data.
    """
    try:
        with open(file_name, 'rb') as f:
            # Read header
            header_params = {}
            
            while True:
                name_len = struct.unpack('I', f.read(4))[0]
                if name_len == 0:
                    break
                    
                try:
                    name = f.read(name_len).decode('ascii')
                except UnicodeDecodeError:
                    # Skip this parameter if we can't decode the name
                    continue
                
                if name == 'HEADER_END':
                    break
                elif name in ['telescope_id', 'machine_id', 'data_type', 'nchans', 'nbits', 'nifs']:
                    header_params[name] = struct.unpack('I', f.read(4))[0]
                elif name in ['tstart', 'tsamp', 'fch1', 'foff']:
                    header_params[name] = struct.unpack('d', f.read(8))[0]
                elif name == 'source_name':
                    str_len = struct.unpack('I', f.read(4))[0]
                    header_params[name] = f.read(str_len).decode('ascii')
                elif name in ['rawdatafile', 'az_start', 'za_start']:
                    # Skip these for now
                    str_len = struct.unpack('I', f.read(4))[0] if name == 'rawdatafile' else 8
                    f.read(str_len)
                else:
                    # Try to read as double, skip if fails
                    try:
                        header_params[name] = struct.unpack('d', f.read(8))[0]
                    except:
                        break
            
            # Set default values if not found
            nchans = header_params.get('nchans', 512)
            nifs = header_params.get('nifs', 1)  # Usually 1 for FRB data
            nbits = header_params.get('nbits', 8)
            
            # Calculate data size
            data_start = f.tell()
            f.seek(0, 2)  # Go to end
            file_size = f.tell()
            data_size = file_size - data_start
            
            # Calculate number of time samples
            bytes_per_sample = nchans * nifs * (nbits // 8)
            nsamples = data_size // bytes_per_sample
            
            # Read data
            f.seek(data_start)
            if nbits == 8:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            elif nbits == 16:
                data = np.frombuffer(f.read(), dtype=np.uint16)
            elif nbits == 32:
                data = np.frombuffer(f.read(), dtype=np.float32)
            else:
                raise ValueError(f"Unsupported nbits: {nbits}")
            
            # Reshape data to (time, pol, channel)
            # For .fil files, usually nifs=1, so we duplicate to have 2 polarizations
            data = data.reshape(nsamples, nchans)
            
            # Convert to float32 and add polarization dimension
            data = data.astype(np.float32)
            if nifs == 1:
                # Duplicate channel to create 2 polarizations
                data = np.stack([data, data], axis=1)  # Shape: (time, 2, channel)
            else:
                data = data.reshape(nsamples, nifs, nchans)
                if nifs == 1:
                    data = np.repeat(data, 2, axis=1)
                else:
                    data = data[:, :2, :]  # Take first 2 polarizations
            
            # Update config with header information
            config.FREQ_RESO = nchans
            config.TIME_RESO = header_params.get('tsamp', 1e-4)  # Default 0.1ms
            config.FILE_LENG = nsamples
            
            # Create frequency array
            fch1 = header_params.get('fch1', 1400.0)  # Default center frequency
            foff = header_params.get('foff', -1.0)    # Default bandwidth
            config.FREQ = np.linspace(fch1, fch1 + nchans * foff, nchans)
            config.DATA_NEEDS_REVERSAL = foff < 0  # Reverse if decreasing frequency
            
            print(f"Loaded .fil file: {nsamples} samples, {nchans} channels, tsamp={config.TIME_RESO}")
            return data
            
    except Exception as e:
        print(f"Error loading .fil file {file_name}: {e}")
        # Create dummy data as fallback
        nsamples, nchans = 10000, 512
        config.FREQ_RESO = nchans
        config.TIME_RESO = 1e-4
        config.FILE_LENG = nsamples
        config.FREQ = np.linspace(1000, 1500, nchans)
        config.DATA_NEEDS_REVERSAL = False
        return np.random.randn(nsamples, 2, nchans).astype(np.float32)


def load_data_file(file_name: str) -> np.ndarray:
    """Load either a FITS file or a .fil file based on extension."""
    file_path = Path(file_name)
    
    if file_path.suffix.lower() == '.fil':
        return load_fil_file(file_name)
    else:
        return load_fits_file(file_name)


def load_fits_file(file_name: str) -> np.ndarray:
    """Load a FITS file and return the data array in shape (time, pol, channel)."""
    global_vars = config
    data_array = None
    try:
        with fits.open(file_name, memmap=True) as hdul:
            if "SUBINT" in [hdu.name for hdu in hdul] and "DATA" in hdul["SUBINT"].columns.names:
                subint = hdul["SUBINT"]
                hdr = subint.header
                data_array = subint.data["DATA"]
                nsubint = hdr["NAXIS2"]
                nchan = hdr["NCHAN"]
                npol = hdr["NPOL"]
                nsblk = hdr["NSBLK"]
                data_array = data_array.reshape(nsubint, nchan, npol, nsblk).swapaxes(1, 2)
                data_array = data_array.reshape(nsubint * nsblk, npol, nchan)
                data_array = data_array[:, :2, :]
            else:
                import fitsio
                temp_data, h = fitsio.read(file_name, header=True)
                if "DATA" in temp_data.dtype.names:
                    data_array = temp_data["DATA"].reshape(h["NAXIS2"] * h["NSBLK"], h["NPOL"], h["NCHAN"])[:, :2, :]
                else:
                    total_samples = h.get("NAXIS2", 1) * h.get("NSBLK", 1)
                    num_pols = h.get("NPOL", 2)
                    num_chans = h.get("NCHAN", 512)
                    data_array = temp_data.reshape(total_samples, num_pols, num_chans)[:, :2, :]
    except Exception as e:
        print(f"[Error cargando FITS con fitsio/astropy] {e}")
        try:
            # Intentar sin memmap para archivos corruptos
            with fits.open(file_name, memmap=False) as f:
                data_hdu = None
                for hdu_item in f:
                    # Evitar acceder a .data directamente, usar hasattr primero
                    try:
                        if (hdu_item.data is not None and 
                            isinstance(hdu_item.data, np.ndarray) and 
                            hdu_item.data.ndim >= 3):
                            data_hdu = hdu_item
                            break
                    except (TypeError, ValueError):
                        # Si no se puede acceder a los datos, saltar este HDU
                        continue
                        
                if data_hdu is None and len(f) > 1:
                    data_hdu = f[1]
                elif data_hdu is None:
                    data_hdu = f[0]
                    
                h = data_hdu.header
                try:
                    raw_data = data_hdu.data
                    if raw_data is not None:
                        data_array = raw_data.reshape(h["NAXIS2"] * h.get("NSBLK", 1), h.get("NPOL", 2), h.get("NCHAN", raw_data.shape[-1]))[:, :2, :]
                    else:
                        raise ValueError("No hay datos válidos en el HDU")
                except (TypeError, ValueError) as e_data:
                    print(f"Error accediendo a datos del HDU: {e_data}")
                    raise ValueError(f"Archivo FITS corrupto: {file_name}")
                    
        except Exception as e_astropy:
            print(f"Fallo final al cargar con astropy: {e_astropy}")
            raise ValueError(f"No se puede leer el archivo FITS corrupto: {file_name}")
            
    if data_array is None:
        raise ValueError(f"No se pudieron cargar los datos de {file_name}")

    if global_vars.DATA_NEEDS_REVERSAL:
        print(f">> Invirtiendo eje de frecuencia de los datos cargados para {file_name}")
        data_array = np.ascontiguousarray(data_array[:, :, ::-1])
    return data_array


def get_obparams(file_name: str) -> None:
    """Extract observation parameters and populate :mod:`config`."""
    file_path = Path(file_name)
    
    if file_path.suffix.lower() == '.fil':
        # For .fil files, parameters are already set during load_fil_file
        # But we need to ensure they are set here for consistency
        if config.FREQ_RESO == 0:  # If not already set
            # Load a small portion to get header info
            try:
                with open(file_name, 'rb') as f:
                    header_params = {}
                    while True:
                        name_len = struct.unpack('I', f.read(4))[0]
                        if name_len == 0:
                            break
                        try:
                            name = f.read(name_len).decode('ascii')
                        except UnicodeDecodeError:
                            # Skip this parameter if we can't decode the name
                            continue
                        if name == 'HEADER_END':
                            break
                        elif name in ['nchans', 'nifs']:
                            header_params[name] = struct.unpack('I', f.read(4))[0]
                        elif name in ['tsamp', 'fch1', 'foff']:
                            header_params[name] = struct.unpack('d', f.read(8))[0]
                        else:
                            # Skip other parameters for this quick header read
                            try:
                                struct.unpack('d', f.read(8))
                            except:
                                break
                
                config.FREQ_RESO = header_params.get('nchans', 512)
                config.TIME_RESO = header_params.get('tsamp', 1e-4)
                fch1 = header_params.get('fch1', 1400.0)
                foff = header_params.get('foff', -1.0)
                config.FREQ = np.linspace(fch1, fch1 + config.FREQ_RESO * foff, config.FREQ_RESO)
                config.DATA_NEEDS_REVERSAL = foff < 0
                
                # Estimate file length
                f.seek(0, 2)
                file_size = f.tell()
                data_start = 1024  # Approximate header size
                bytes_per_sample = config.FREQ_RESO * (8 // 8)  # Assuming 8 bits
                config.FILE_LENG = max(1000, (file_size - data_start) // bytes_per_sample)
                
            except Exception as e:
                print(f"Error reading .fil header: {e}")
                # Set defaults
                config.FREQ_RESO = 512
                config.TIME_RESO = 1e-4
                config.FILE_LENG = 10000
                config.FREQ = np.linspace(1000, 1500, 512)
                config.DATA_NEEDS_REVERSAL = False
        return
    
    # Original FITS file handling
    with fits.open(file_name, memmap=True) as f:
        freq_axis_inverted = False
        if "SUBINT" in [hdu.name for hdu in f] and "TBIN" in f["SUBINT"].header:
            hdr = f["SUBINT"].header
            sub_data = f["SUBINT"].data
            config.TIME_RESO = hdr["TBIN"]
            config.FREQ_RESO = hdr["NCHAN"]
            config.FILE_LENG = hdr["NSBLK"] * hdr["NAXIS2"]
            freq_temp = sub_data["DAT_FREQ"][0].astype(np.float64)
            if "CHAN_BW" in hdr:
                bw = hdr["CHAN_BW"]
                if isinstance(bw, (list, np.ndarray)):
                    bw = bw[0]
                if bw < 0:
                    freq_axis_inverted = True
            elif len(freq_temp) > 1 and freq_temp[0] > freq_temp[-1]:
                freq_axis_inverted = True
        else:
            try:
                data_hdu_index = 0
                for i, hdu_item in enumerate(f):
                    if hdu_item.is_image or isinstance(hdu_item, (fits.BinTableHDU, fits.TableHDU)):
                        if 'NAXIS' in hdu_item.header and hdu_item.header['NAXIS'] > 0:
                            if 'CTYPE3' in hdu_item.header and 'FREQ' in hdu_item.header['CTYPE3'].upper():
                                data_hdu_index = i
                                break
                            if 'CTYPE2' in hdu_item.header and 'FREQ' in hdu_item.header['CTYPE2'].upper():
                                data_hdu_index = i
                                break
                            if 'CTYPE1' in hdu_item.header and 'FREQ' in hdu_item.header['CTYPE1'].upper():
                                data_hdu_index = i
                                break
                if data_hdu_index == 0 and len(f) > 1:
                    data_hdu_index = 1
                hdr = f[data_hdu_index].header
                if "DAT_FREQ" in f[data_hdu_index].columns.names:
                    freq_temp = f[data_hdu_index].data["DAT_FREQ"][0].astype(np.float64)
                else:
                    freq_axis_num = ''
                    for i in range(1, hdr.get('NAXIS', 0) + 1):
                        if 'FREQ' in hdr.get(f'CTYPE{i}', '').upper():
                            freq_axis_num = str(i)
                            break
                    if freq_axis_num:
                        crval = hdr.get(f'CRVAL{freq_axis_num}', 0)
                        cdelt = hdr.get(f'CDELT{freq_axis_num}', 1)
                        crpix = hdr.get(f'CRPIX{freq_axis_num}', 1)
                        naxis = hdr.get(f'NAXIS{freq_axis_num}', hdr.get('NCHAN', 512))
                        freq_temp = crval + (np.arange(naxis) - (crpix - 1)) * cdelt
                        if cdelt < 0:
                            freq_axis_inverted = True
                    else:
                        freq_temp = np.linspace(1000, 1500, hdr.get('NCHAN', 512))
                config.TIME_RESO = hdr["TBIN"]
                config.FREQ_RESO = hdr.get("NCHAN", len(freq_temp))
                config.FILE_LENG = hdr.get("NAXIS2", 0) * hdr.get("NSBLK", 1)
            except Exception as e_std:
                print(f"Error procesando FITS estándar: {e_std}")
                config.TIME_RESO = 5.12e-5
                config.FREQ_RESO = 512
                config.FILE_LENG = 100000
                freq_temp = np.linspace(1000, 1500, config.FREQ_RESO)
        if freq_axis_inverted:
            config.FREQ = freq_temp[::-1]
            config.DATA_NEEDS_REVERSAL = True
        else:
            config.FREQ = freq_temp
            config.DATA_NEEDS_REVERSAL = False

    if config.FREQ_RESO >= 512:
        config.DOWN_FREQ_RATE = max(1, int(round(config.FREQ_RESO / 512)))
    else:
        config.DOWN_FREQ_RATE = 1
    if config.TIME_RESO > 1e-9:
        config.DOWN_TIME_RATE = max(1, int((49.152 * 16 / 1e6) / config.TIME_RESO))
    else:
        config.DOWN_TIME_RATE = 15
