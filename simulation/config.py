strategy_mapping = {'AC':'AlwaysCooperate', 'AD':'AlwaysDefect', 'TFT':'TitForTat', 'Random':'random', 
                    'CD':'Alternating', 'GEa':'GilbertElliot_typea', 'GEb':'GilbertElliot_typeb', 'GEc':'GilbertElliot_typec', 'GEd':'GilbertElliot_typed', 'GEe':'GilbertElliot_typee',
                    'QLS':'Q-Learning eps-greedy', 'QLUT':'Q-Learning eps-greedy', 'QLDE':'Q-Learning eps-greedy', 'QLVE_e':'Q-Learning eps-greedy', 'QLVE_k':'Q-Learning eps-greedy', 'QLVM':'Q-Learning eps-greedy', 'QLAL':'Q-Learning eps-greedy',
                    'QLaUT':'Q-Learning eps-greedy', 'QLmDE':'Q-Learning eps-greedy', 'QLVie':'Q-Learning eps-greedy', 'QLVagg':'Q-Learning eps-greedy', 'QLaAL':'Q-Learning eps-greedy'}
moral_mapping = {'AC':None, 'AD':None, 'TFT':None, 'Random':None, 
                 'CD':None, 'GEa':None, 'GEb':None, 'GEc':None, 'GEd':None, 'GEe':None,
                 'QLS':None, 'QLUT':'Utilitarian', 'QLDE':'Deontological', 'QLVE_e':'VirtueEthics_equality', 'QLVE_k':'VirtueEthics_kindness', 'QLVM':'VirtueEthics_mixed', 'QLAL':'Altruist',
                 'QLaUT':'anti-Utilitarian', 'QLmDE':'malicious_Deontological', 'QLVie':'VirtueEthics_inequality', 'QLVagg':'VirtueEthics_aggressiveness', 'QLaAL':'anti-Altruist'}


#PAYOFFMAT_IPD = [ [(3,3),(1,4)] , [(4,1),(2,2)] ] #IPD game - b:c=2:1
#PAYOFFMAT_IPD = [ [(4,4),(1,5)] , [(5,1),(2,2)] ] #IPD game - b:c=3:1
PAYOFFMAT_IPD = [ [(3,3),(0,4)] , [(4,0),(1,1)] ] #IPD game - b:c=3:1, payoffmatwith0
#PAYOFFMAT_IPD = [ [(11,11),(1,12)] , [(12,1),(2,2)] ] #IPD game - b:c=10:1
#PAYOFFMAT_IPD = [ [(21,21),(1,22)] , [(22,1),(2,2)] ] #IPD game - b:c=20:1

PAYOFFMAT_VOLUNTEER = [ [(4,4),(2,5)] , [(5,2),(1,1)] ] #VOLUNTEER game 
PAYOFFMAT_STAGHUNT = [ [(5,5),(1,4)] , [(4,1),(2,2)] ] #STAGHUNT game 

my_game = 'IPD' #IPD, VOLUNTEER, STAGHUNT
study_type = 'population_withselection' #['dyadic', 'population', 'population_withselection]

master_seed = 1 #used to instantiate the random number generator 
gamma = 0.99 #discount factor for Q-learning / DQN 

epsdecay = False #whether to implement epsilon decay or not over the iterations  
eps_dil_final = 0.05
eps_sel_final = 0.1
if epsdecay == True: 
    eps_dil_initial = 1 #proportion of random exploration. If epsdecay = True, this is the initial value of eps at iteration=1
    eps_sel_initial = 1 
else: #keep eps the same throughout 
    eps_dil_initial = eps_dil_final
    eps_sel_initial = eps_sel_final

#### parameters for tabular qlearning study 
mixed_beta = 0.5 #parameter for mixed agent QLVM to weigh the two rewards
alpha0 = 0.5 #0.01 #0.9 #initial learning rate alpha for tabular Q-learning 
decay = 0 # alpha decay  for tabular Q-learning 

episodes = True #whether to run the QNetwork updates in episodes or not (if not, then it happens on every iteration)
player2_learns = True
dilemma_state_uses_identity = False #True 
state_pairs = False #NOTE need to finish implementation with state_pairs and episodes !!! 

random_matching = False #if False, will use game.partner_selection 

#seed_DQN = 1 #can use this to separately set seed for nn weights intialisation and sampling from memory
#plot_targetQ = False #whether we want to store target net Q-values instead of local net Q-values (for plotting)

action_size = 2 #number of possible actions for the dilemma game 
#set state size - if dilemma_state_uses_identity, this will be modified in main.py 
if state_pairs: 
    state_size_game = 2 #number of observations for the dilemma game (2 = pair of actions from past iteration)
else: 
    state_size_game = 1 #(1 = only the opponent's last move)
#population_size will be defined via destination_folder parsing through argparse 


network_size = 256 #size of the single hidden layer in DQN
use_target_network = False #True 

#LR = 0.001 #learning rate for stochastic gradient descent / AdamW optimizer in DQN 
LR_selection = 0.001 #learning rate for stochastic gradient descent / AdamW optimizer in SELECTION DQN
LR_dilemma = 0.001 #learning rate for stochastic gradient descent / AdamW optimizer in DILEMMA DQN
TAU = 1 #0.005 #1e-3 #1 #update rate of the target network

UPDATE_EVERY_selection = 1 #every C steps reset targetQ = localQ 
UPDATE_EVERY_dilemma = 1 #every C steps reset targetQ = localQ 
BUFFER_SIZE = 256 #256 possible memories in the buffer - but note this gets refreshed at the end of every episode 
#BATCH_SIZE = 1 #number of memories to be sampled at every learning step --> now this is calculated flexibly depending on the number of experiences in the buffer from the past episode 

record_Qvalues = False #'both' or False 
#record_Qvalues_every = 10 #store every X Qvalues to disc (to reduce memory requirements) - TO DO fix this 
record_Loss = False #'both' #'dilemma' #'selection' #False 
record_selection_state = True
record_dilemma_nextstate = False
record_eps = False 
record_reason = True 
record_RNs = False 
profile_time = True 


testing = False 


#### define parameters for moral players 
xi = 5 #the constant that defines reward & punishment values for norm-based agents 
