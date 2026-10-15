#!/bin/bash

# Interrompe lo script se un comando fallisce (es. se la build dà errore)
set -e

# --- CONFIGURAZIONE ---
# Modifica questo percorso se la tua cartella è altrove (es. ~/Desktop/...)
WORKSPACE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "--- 1. Attivazione del Virtual Environment ---"
source "$WORKSPACE_DIR/.venv/bin/activate"

echo "--- 2. Caricamento ROS 2 Jazzy ---"
source /opt/ros/jazzy/setup.bash

echo "--- 3. Spostamento nella cartella di lavoro ---"
cd "$WORKSPACE_DIR/spot_micro"

echo "--- 4. Compilazione del workspace ---"
# Uso --symlink-install così le modifiche future ai file Python/Yaml non richiederanno rebuild
colcon build --symlink-install

echo "--- 5. Caricamento dell'overlay locale ---"
source install/setup.bash

echo "--- 6. Avvio della simulazione ---"
ros2 launch mobile_robot gazebo_model.launch.py
