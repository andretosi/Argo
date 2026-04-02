import numpy as np
# Importa la classe SpotmicroEnv, che definisce l'ambiente di simulazione del robot.
from spotmicro.env.spotmicro_env_mujoco import SpotmicroEnv

# classe per memorizzare lo stato tra un timestep e l'altro
class RewardState:
    def __init__(self):
        # Insieme per memorizzare i contatti precedenti dei piedi del robot. Attualmente non utilizzato.
        self.prev_contacts = set() # INUTILIZZATO
        # Posizione precedente della base del robot, inizializzata a zero.
        self.prev_base_position = np.array([0.0, 0.0, 0.0]) # placeholder che viene aggiornato al primo step

    def populate(self, env: SpotmicroEnv): # INUTILIZZATO
        # Questo metodo dovrebbe aggiornare lo stato della ricompensa con le informazioni dell'ambiente, ma attualmente è vuoto.
        return

# Funzione per aumentare gradualmente un valore da 0 a 1, utile per introdurre gradualmente componenti della reward.
def fade_in(current_step, start, scale=2.0): # INUTILIZZATO
    # Se il passo corrente è inferiore a quello di inizio, ritorna 0.
    if current_step < start:
        return 0.0
    # Calcola un valore che cresce esponenzialmente verso 1.
    return 1.0 - np.exp(-scale * (current_step - start) / 1_000_000)

# Funzione principale che calcola la ricompensa totale per un dato stato e azione.
def reward_function(env: SpotmicroEnv, action: np.ndarray) -> tuple[float, dict]:
    # Estrae l'orientamento del robot (roll, pitch, yaw) dallo stato dell'agente.
    roll, pitch, _ = env.agent.state.roll_pitch_yaw 
    # Definizione di un errore percentuale, attualmente non utilizzato nel calcolo della reward.
    percentage_error = 0.2 # INUTILIZZATO

    ##########################
    ### Errors and metrics ###
    ##########################

    
    # Calcola l'errore quadratico percentuale della velocità lineare rispetto a quella target.
    lin_vel_sq_perc_error = (np.linalg.norm(env.agent.controller.input.as_array[:2] - env.agent.state.linear_velocity[:2])/
                             np.linalg.norm(env.agent.controller.input.as_array[:2]) + 1e-5)** 2

    # Calcola l'errore della velocità angolare normalizzato rispetto alla massima velocità angolare possibile.
    ang_vel_error = ((env.agent.controller.input.w - env.agent.state.angular_velocity[2]) / env.agent.max_angular_velocity) ** 2 # CHANGED
    # Su questo ci devo riflettere, noi diamo anche indicazioni rotazionali? Che indicazioni sulla rotazione diamo?
    # In generale come stabiliamo le velocità target? Dove sono decise? Da chi?

    # Misura quanto le articolazioni si allontanano dalla posizione di riposo (homing position).
    deviation_penalty = np.linalg.norm(env.agent.state.joint_positions - env.agent.homing_positions) ** 2  # INUTILIZZATO , con peso 0 alla fine
    # Penso che bisogna aggiungere un range di movimento che NON penalizziamo e solo alcuni estremi che sono penalizzati,
    # legati a constraint fisiche, non a constraint di "scomodità", quelle verrano decie dalla parte
    # che si occupa del risparmio energetico.

    # === HEIGHT PENALTY: Approccio basato su Propriocezione===
    # Calcola l'altezza del corpo RELATIVA al centroide dei piedi, proiettata
    # lungo l'asse verticale del robot. Questo approccio è robusto su terreni
    # accidentati perché non dipende dall'altezza assoluta (Z globale).
    body_to_feet_height = env.agent.get_body_to_feet_height_projected()
    height_penalty = (body_to_feet_height - env.agent.target_body_to_feet_height) ** 2


    # Calcola la variazione media dell'azione rispetto all'azione precedente per penalizzare movimenti bruschi.
    action_rate = np.mean(action - env.agent.previous_action) ** 2
    
    # Calcola il quadrato della velocità verticale per penalizzare movimenti sussultori.
    vertical_velocity_sq =  env.agent.state.linear_velocity[2] ** 2
    
    # Penalità per l'inclinazione del robot (roll e pitch), per promuovere la stabilità.
    stabilization_penalty = roll ** 2 + pitch ** 2
    # Penso vada bene, unico dubbio, il robot non dovrebbe impararlo da solo?


    # Calcola lo sforzo normalizzato totale dei motori, per penalizzare un consumo eccessivo di energia.
    total_normalized_effort = np.sum([(j.effort / j.max_torque) ** 2 for j in env.agent.motor_joints]) / len(env.agent.motor_joints)
    # Come è calcolato j.effort? da dove viene? Dalla simulazione o lo abbiamo calcolato noi
    # da altre info inidirette? Super importante chiarire sta cosa

    # Calcola le penalità derivate basate sulle metriche precedenti.
    tolerance = 0.3
    # non capisco il senso della tolerance
    
    # Ricompensa per la velocità lineare, che decresce quadraticamente con l'errore e limitata inferiormente.
    # In altre parole, l'agente ha un reward=1 se segue eprfettamente la velocità, altrimenti questa decade quadraticamente fino ad un valore minimo di -0.5 (ossia una penalità). Il parametro di toleranze determina l'ampiezza della parabola, ossia quanto preciso vogliamo che sia il tracciamento
    lin_vel_reward = max(1.0 - ((lin_vel_sq_perc_error / tolerance) **2), -0.5)

    

    # Calcola lo spostamento della base del robot rispetto al timestep precedente.
    delta_pos = env.agent.state.base_position - env.reward_state.prev_base_position
    
    # Calcola il progresso nella direzione della velocità target.
    progress = np.dot(delta_pos[:2], env.agent.controller.input.as_array[:2]) / (np.linalg.norm(env.agent.controller.input.as_array[:2])+1e-6)
    
    # Limita il progresso per ogni step per evitare valori troppo grandi e stabilizzare l'apprendimento.
    progress = np.clip(progress / env.sim_frequency, -0.5, 0.5)
    # Perchè normalizza per la freq di simulazione?

    # === Final Reward ===
    # Dizionario che raccoglie tutte le componenti della reward con i rispettivi pesi.
    reward_dict = {
        "linear_vel_reward": 12 * lin_vel_reward,
        "progress_reward": 1.5 - fade_in(env.num_steps, 100_000) * progress, #1.5 ->0.5
        "angular_vel_penalty": -5 * ang_vel_error,
        "action_rate_penalty": -2 * action_rate,
        "height_penalty": -3 * min(height_penalty, 1.0),
        "stabilization_penalty": -3 * min(stabilization_penalty, 1.0),
        "effort_penalty": -1.5 * total_normalized_effort,
        "deviation_penalty": -0.0 * deviation_penalty,
        "vertical_motion_penalty": -0.5 * vertical_velocity_sq,
    }
    # Calcola la ricompensa totale come somma di tutte le componenti pesate.
    total_reward = sum(reward_dict.values())

    # Registra i valori delle singole componenti della reward per il monitoraggio.
    env.log_rewards(reward_dict)
    
    # Ritorna la ricompensa totale e il dizionario con le componenti separate.
    return total_reward, reward_dict



#### RIFLESSIONI #####
# Dobbiamo commentare le varie funzioni helper, tipo quando passo sopra
# "env.agent.state.roll_pitch_yaw" dovrebbe comparire cosa ogni cosa fa
# potrebbe occuparsene @mariomori04 che lo sta già facendo per conto suo,
# faciliterebbe anche l'onboarding oltre che rendere il codice più
# facile per i noi del futuro (e di adesso)

# Usare la huber loss function invece del quadrato?

# Riprendere il ragionamento con copilot