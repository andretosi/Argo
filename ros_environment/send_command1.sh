#!/bin/bash

# Interrompe lo script se un comando fallisce
set -e

# --- CONFIGURAZIONE ---
WORKSPACE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "--- 1. Attivazione del Virtual Environment ---"
source "$WORKSPACE_DIR/.venv/bin/activate"

echo "--- 2. Caricamento ROS 2 Jazzy ---"
source /opt/ros/jazzy/setup.bash

echo "--- 3. Spostamento nella cartella di lavoro ---"
cd "$WORKSPACE_DIR/spotmicro_control"

if [ ! -f "install/setup.bash" ]; then
    echo "ERRORE: Non trovo install/setup.bash. Devi fare 'colcon build' almeno una volta!"
    return 1
fi

echo "--- 4. Caricamento dell'overlay locale ---"
source install/setup.bash

echo "--- 5. Lancio il codice per il controller ---"
python3 src/send_command.py

