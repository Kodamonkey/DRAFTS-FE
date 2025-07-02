#!/usr/bin/env python3
"""Script mínimo para probar carga de configuración."""

import sys
import os

def test_minimal():
    print("=== TEST MÍNIMO ===")
    print(f"Python: {sys.version}")
    print(f"Directorio: {os.getcwd()}")
    
    try:
        print("Importando config...")
        from DRAFTS import config
        print(f"✓ Config importado")
        print(f"  DEBUG: {config.DEBUG}")
        print(f"  FRB_TARGETS: {config.FRB_TARGETS}")
        print(f"  DEVICE: {config.DEVICE}")
        
        print("Probando funciones de búsqueda...")
        from DRAFTS.pipeline import _find_data_files
        files = _find_data_files("3097_0001")
        print(f"✓ Archivos encontrados: {[f.name for f in files]}")
        
        # NO cargar modelos aún
        print("✓ Test básico completado sin problemas")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal()
