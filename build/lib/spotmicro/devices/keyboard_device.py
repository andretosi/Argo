#Assignee: virgina


#class Keyboard(Device):
#    def __init__(self):
#        raise NotImplementedError("Keyboard device is not implemented yet")


import sys
import termios
import tty
import select
import time

from spotmicro.devices.device import Device
from spotmicro.agent.input import Input

class Keyboard(Device):
  
    def __init__(self):
        super().__init__()
        self._enabled = False
        self._orig_settings = None
        
        # --- PARAMETRI ---
        self.VAL_SPEED = 1
        self.VAL_TURN = 1
        # Quanto tempo (in secondi) un tasto resta 'attivo' dopo essere stato premuto.
        self.KEY_TIMEOUT = 0.3 

        # --- STATO INTERNO ---
        # Memorizziamo l'ultima volta che abbiamo visto un comando
        self.last_time_x = 0.0
        self.last_time_y = 0.0
        self.last_time_w = 0.0
        
        # Valori attuali da inviare
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_w = 0.0
        
        self.is_stopped = True # Per gestire le stampe di debug

    def update(self):
        if not self._enabled:
            self._orig_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._enabled = True
            #print("\n[Keyboard] CONTROLLO ATTIVO: Supporto combinazioni W+A, W+D, ecc.")

    def read(self) -> Input:
        if not self._enabled:
            return Input(0.0, 0.0, 0.0)

        now = time.time()
        
        # 1. LEGGI TUTTI I CARATTERI NEL BUFFER
        # Invece di leggerne uno solo, leggo tutto quello che c'è
        # catturo combinazioni rapide.
        while True:
            dr, _, _ = select.select([sys.stdin], [], [], 0)
            if not dr:
                break # Nessun altro carattere in attesa
            
            char = sys.stdin.read(1).lower()
            
            # Aggiorna lo stato e il timestamp in base al tasto
            if char == 'w': 
                self.current_x = self.VAL_SPEED
                self.last_time_x = now
            elif char == 's': 
                self.current_x = -self.VAL_SPEED
                self.last_time_x = now
            elif char == 'q': 
                self.current_w = self.VAL_TURN
                self.last_time_w = now
            elif char == 'e': 
                self.current_w = -self.VAL_TURN
                self.last_time_w = now
            elif char == 'a': 
                self.current_y = self.VAL_SPEED
                self.last_time_y = now
            elif char == 'd': 
                self.current_y = -self.VAL_SPEED
                self.last_time_y = now

        # 2. CONTROLLA SCADENZA (DECAY)
        # Se è passato troppo tempo dall'ultimo segnale di un asse, azzeralo.
        
        if now - self.last_time_x > self.KEY_TIMEOUT:
            self.current_x = 0.0
        
        if now - self.last_time_y > self.KEY_TIMEOUT:
            self.current_y = 0.0
            
        if now - self.last_time_w > self.KEY_TIMEOUT:
            self.current_w = 0.0

        # 3. GESTIONE STAMPE (DEBUG)
        # Se tutti i valori sono zero
        #if self.current_x == 0 and self.current_y == 0 and self.current_w == 0:
        #    if not self.is_stopped:
        #        print(" -> STOP (0.0, 0.0, 0.0)")
        #        self.is_stopped = True
        #else:
            # Se ci stiamo muovendo
        #    print(f" -> INPUT COMBINATO: x={self.current_x:.1f}, y={self.current_y:.1f}, w={self.current_w:.1f}")
        #    self.is_stopped = False

        return Input(self.current_x, self.current_y, self.current_w)

    def reset(self):
        self.last_time_x = 0.0
        self.last_time_y = 0.0
        self.last_time_w = 0.0
        
        # Valori attuali da inviare
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_w = 0.0
        
        self.is_stopped = True # Per gestire le stampe di debug

    def __del__(self):
        if self._enabled and self._orig_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._orig_settings)
            #print("[Keyboard] Terminale ripristinato.")


