#!/usr/bin/env python3
"""Script de debugging básico para identificar problemas."""

import sys
import traceback
import time

def test_imports():
    """Probar imports básicos."""
    print("=== TESTING IMPORTS ===")
    try:
        import numpy as np
        print("✓ numpy OK")
    except Exception as e:
        print(f"✗ numpy ERROR: {e}")
        
    try:
        import torch
        print(f"✓ torch OK (CUDA available: {torch.cuda.is_available()})")
    except Exception as e:
        print(f"✗ torch ERROR: {e}")
        
    try:
        import psutil
        print("✓ psutil OK")
    except Exception as e:
        print(f"✗ psutil ERROR: {e}")
        
    try:
        from DRAFTS import config
        print(f"✓ config OK (DEBUG: {config.DEBUG})")
    except Exception as e:
        print(f"✗ config ERROR: {e}")

def test_config():
    """Probar configuración."""
    print("\n=== TESTING CONFIG ===")
    try:
        from DRAFTS import config
        print(f"DEBUG: {config.DEBUG}")
        print(f"DEBUG_DATA: {config.DEBUG_DATA}")
        print(f"DEBUG_MEMORY: {config.DEBUG_MEMORY}")
        print(f"DEBUG_TIMING: {config.DEBUG_TIMING}")
        print(f"FRB_TARGETS: {config.FRB_TARGETS}")
        print(f"DATA_DIR: {config.DATA_DIR}")
        print(f"DEVICE: {config.DEVICE}")
    except Exception as e:
        print(f"Error accediendo config: {e}")
        traceback.print_exc()

def test_data_loading():
    """Probar carga de datos."""
    print("\n=== TESTING DATA LOADING ===")
    try:
        from DRAFTS.io import load_data_file, get_obparams
        from DRAFTS import config
        
        # Buscar el archivo .fil
        data_file = config.DATA_DIR / "3097_0001_00_8bit.fil"
        if data_file.exists():
            print(f"✓ Archivo encontrado: {data_file}")
            
            print("Probando get_obparams...")
            get_obparams(str(data_file))
            print(f"  FREQ_RESO: {config.FREQ_RESO}")
            print(f"  TIME_RESO: {config.TIME_RESO}")
            print(f"  FILE_LENG: {config.FILE_LENG}")
            
            print("Probando load_data_file...")
            start_time = time.time()
            data = load_data_file(str(data_file))
            load_time = time.time() - start_time
            print(f"✓ Datos cargados en {load_time:.2f}s")
            print(f"  Forma: {data.shape}")
            print(f"  Tipo: {data.dtype}")
            print(f"  Min/Max: {data.min():.6f}/{data.max():.6f}")
        else:
            print(f"✗ Archivo no encontrado: {data_file}")
            
    except Exception as e:
        print(f"Error en carga de datos: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("=== DEBUGGING PIPELINE ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {sys.path[0]}")
    
    test_imports()
    test_config()
    test_data_loading()
    
    print("\n=== DEBUGGING COMPLETE ===")
