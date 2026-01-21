#!/bin/bash

# --- CONFIGURACIÓN ---
OUTPUT_DIR="USB_KIT_WINDOWS/packages"
PYTHON_VER="312"  # Ojo: Pon "310" o "311" si tu Python Portable es diferente
# ---------------------

echo "📦 Preparando kit para Windows desde Pop!_OS..."

# 1. Limpiamos la lista de cosas exclusivas de Linux que podrían colarse
# (Aunque uv compile --platform windows es listo, a veces se cuelan pexpect/ptyprocess)
grep -vE "ptyprocess|pexpect|uvloop" requirements_windows.txt > requirements_final.txt

# 2. Crear carpeta
mkdir -p "$OUTPUT_DIR"

# 3. Descargar forzando binarios de Windows
# Usamos pip estándar para engañar al servidor y decir que somos Windows
echo "⬇️  Descargando archivos .whl para Windows (amd64)..."

pip download \
    -r requirements_final.txt \
    --dest "$OUTPUT_DIR" \
    --platform win_amd64 \
    --python-version "$PYTHON_VER" \
    --only-binary=:all: \
    --no-deps

# 4. Copiar archivos esenciales al KIT
cp requirements_final.txt "USB_KIT_WINDOWS/"
# Si tienes el codigo aquí, podrías copiarlo también:
# cp -r mi_carpeta_codigo "USB_KIT_WINDOWS/"

echo "✅ ¡Hecho! La carpeta 'USB_KIT_WINDOWS' está lista."
echo "   NO OLVIDES: Añadir 'uv.exe' y la carpeta 'python_portable' a esa carpeta manualmente."
