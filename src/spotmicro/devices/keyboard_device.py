#Assignee: virgina


#class Keyboard(Device):
#    def __init__(self):
#        raise NotImplementedError("Keyboard device is not implemented yet")


import sys
import termios
import tty
import select
import time
<<<<<<< HEAD
=======
import math
>>>>>>> 403050b (Keyboard update)

from spotmicro.devices.device import Device
from spotmicro.agent.input import Input

class Keyboard(Device):
  
    def __init__(self):
        super().__init__()
        self._enabled = False
        self._orig_settings = None
<<<<<<< HEAD
        
        # --- PARAMETRI ---
        self.VAL_SPEED = 1
        self.VAL_TURN = 1
        # Quanto tempo (in secondi) un tasto resta 'attivo' dopo essere stato premuto.
        self.KEY_TIMEOUT = 0.3 
=======

        
        # --- PARAMETRI ---
        self.VAL_SPEED = 1
        self.VAL_TURN = 1 #parametro da regolare per sincronizzarlo alla rotazione
        # Quanto tempo (in secondi) un tasto resta 'attivo' dopo essere stato premuto.
        self.KEY_TIMEOUT = 0.1 
>>>>>>> 403050b (Keyboard update)

        # --- STATO INTERNO ---
        # Memorizziamo l'ultima volta che abbiamo visto un comando
        self.last_time_x = 0.0
        self.last_time_y = 0.0
        self.last_time_w = 0.0
        
        # Valori attuali da inviare
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_w = 0.0
        
<<<<<<< HEAD
=======
        #bussola virtuale
        self.yaw = 0.0
        self.last_update = time.time()
        self.YAW_RATE= 0.045

>>>>>>> 403050b (Keyboard update)
        self.is_stopped = True # Per gestire le stampe di debug

    def update(self):
        if not self._enabled:
            self._orig_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._enabled = True
            #print("\n[Keyboard] CONTROLLO ATTIVO: Supporto combinazioni W+A, W+D, ecc.")

    def read(self) -> Input:
<<<<<<< HEAD
        if not self._enabled:
            return Input(0.0, 0.0, 0.0)

        now = time.time()
=======
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        if not self._enabled:
            return Input(0.0, 0.0, 0.0)

>>>>>>> 403050b (Keyboard update)
        
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
<<<<<<< HEAD
                self.current_x = self.VAL_SPEED
                self.last_time_x = now
            elif char == 's': 
                self.current_x = -self.VAL_SPEED
=======
                self.current_x = -self.VAL_SPEED
                self.last_time_x = now
            elif char == 's': 
                self.current_x = self.VAL_SPEED
>>>>>>> 403050b (Keyboard update)
                self.last_time_x = now
            elif char == 'q': 
                self.current_w = self.VAL_TURN
                self.last_time_w = now
            elif char == 'e': 
                self.current_w = -self.VAL_TURN
                self.last_time_w = now
            elif char == 'a': 
<<<<<<< HEAD
                self.current_y = self.VAL_SPEED
                self.last_time_y = now
            elif char == 'd': 
                self.current_y = -self.VAL_SPEED
=======
                self.current_y = -self.VAL_SPEED
                self.last_time_y = now
            elif char == 'd': 
                self.current_y = self.VAL_SPEED
>>>>>>> 403050b (Keyboard update)
                self.last_time_y = now

        # 2. CONTROLLA SCADENZA (DECAY)
        # Se è passato troppo tempo dall'ultimo segnale di un asse, azzeralo.
        
        if now - self.last_time_x > self.KEY_TIMEOUT:
            self.current_x = 0.0
        
        if now - self.last_time_y > self.KEY_TIMEOUT:
            self.current_y = 0.0
            
        if now - self.last_time_w > self.KEY_TIMEOUT:
            self.current_w = 0.0

<<<<<<< HEAD
=======
        #Trasformazione coordinate
        self.yaw += self.current_w * self.YAW_RATE*dt
        #self.yaw = (self.yaw + math.pi)%(2*math.pi) - math.pi
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

        #vx = -self.current_x * math.cos(self.yaw ) - self.current_y * math.sin(self.yaw )
        #vy = self.current_x * math.sin(self.yaw ) - self.current_y * math.cos(self.yaw )
        
        #vx = -self.current_x
        #vy = - self.current_y
        # vx, vy sono velocità nel frame locale del robot:
        # avanti = -self.current_x, destra = -self.current_y
        local_vx = -self.current_x
        local_vy = -self.current_y

        # Ruota verso il frame globale secondo la stessa formula del KinematicGhost
        vx = local_vx * math.cos(self.yaw) - local_vy * math.sin(self.yaw)
        vy = local_vx * math.sin(self.yaw) + local_vy * math.cos(self.yaw)

        print(f"yaw={self.yaw:.2f} vx={vx:.2f} vy={vy:.2f}")
        print(f"yaw={self.yaw:.2f}")

>>>>>>> 403050b (Keyboard update)
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
<<<<<<< HEAD

        return Input(self.current_x, self.current_y, self.current_w)

    def reset(self):
=======
        vx = max(min(vx, 1.0), -1.0)
        vy = max(min(vy, 1.0), -1.0)
        vw = max(min(self.current_w, 1.0), -1.0)
        #vw = self.current_w
        
        return Input(vx, vy, vw)
    
        

    def reset(self):
        self.yaw = 0.0
>>>>>>> 403050b (Keyboard update)
        self.last_time_x = 0.0
        self.last_time_y = 0.0
        self.last_time_w = 0.0
        
        # Valori attuali da inviare
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_w = 0.0
<<<<<<< HEAD
        
=======
        self.last_update = time.time()

>>>>>>> 403050b (Keyboard update)
        self.is_stopped = True # Per gestire le stampe di debug

    def __del__(self):
        if self._enabled and self._orig_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._orig_settings)
<<<<<<< HEAD
            #print("[Keyboard] Terminale ripristinato.")


=======
            #print("[Keyboard] Terminale ripristinato.")
>>>>>>> 403050b (Keyboard update)
