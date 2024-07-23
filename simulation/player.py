from explorationpolicy import ExplorationPolicy
import numpy as np 


from qnetwork import QNetwork
from buffer import ReplayBuffer, SelectionReplayBuffer
from config import * 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd 
import itertools


class Player:
    def __init__(self, strategy, population_size, selection_size, selection_state_size, state_size, moral_type = None, dil_DQN_weights_seed = None, dil_memory_sampling_seed = None, sel_DQN_weights_seed = None, sel_memory_sampling_seed = None): 
        #self.reset() #defined below
        self.strategy = strategy #long title desribing the strategy 
        self.moral_type = moral_type
        
        self.mixed_beta = mixed_beta #will only be used for QLVM agent 

        self.RNs = {} #see rn_generator.py for explanation of keys & values stored in this Random Number dictionary 

        #TO DO self.QV from expplorationpolicy.py should be stored within the agent instead 
        self.alg_type = None #to be defined within main.py


        #initialise Q-table for tabularQ:
        self.Q_values = None #for tabular q-learning

        self.population_size = population_size
        self.selection_size = selection_size
        self.selection_state_size = selection_state_size
        self.state_size = state_size

        self.reason_selection = None 

        #TO DO add filtering for non-static players here 

        if 'Q-Learning' in self.strategy: 
            #define from config parameters 
            self.eps_dil_initial = eps_dil_initial
            self.eps_sel_initial = eps_sel_initial
            self.epsdecay = epsdecay

            self.eps_selection = None #initialise parameter for later storage of current eps_sel values 

            #initialise Qvalues storage if applicable
            self.record_Qvalues = record_Qvalues #defined in config
            #NB set the above manually when we want to export Q values
            #NOTE we only save expected Q-values at the moment, not target ones 

            if self.record_Qvalues == 'both': 
                self.Qvalues_expected_dilemma = {}
                #self.Qvalues_expected_dilemma[('state', 'action', 'iteration')] = 'Q-value dilemma'
                self.Qvalues_expected_selection = {}
                #self.Qvalues_expected_selection[('state', 'action', 'iteration')] = 'Q-value selection'
        else:
            self.record_Qvalues = None

        #self.t_step = 0
        self.t_step_selection = 0
        self.t_step_dilemma = 0
        #TO DO !!!!C check why looping over a breakpoint here creates more loops than population_size !!!! 

        #self.possible_opponents = None
        self.possible_opponent_indices = None #local indices 
        self.possible_opponent_indices_global = None #global indices 
        self.index_mapping_localtoglobal = None #mapping local to global indices 
        self.index_mapping_globaltolocal = None #mapping global to local indices 

        self.current_state = None #this will maintain the opponent's previous move 
        self.current_selection_state = None #this will maintain moves for each possible opponent in population, and gets updated whenever we play this opponent (selectOR or selectED)
        self.current_selection_idx = None #int index of the opponent selected - mapped to the GLOBAL population index 
        self.next_selection_state = None 

        self.latest_move = None 
        self.current_reward_selection = None 

        self.loss_history = {} #None 
        self.selection_loss_history = {} #None #
        self.current_loss = None #CHECK 
        self.current_selection_loss = None
        self.running_selection_loss = None
        self.running_loss = None
        
        #create memory objects - even for non-learning players 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device 
        
        self.RNs['dil_memory_sampling'] = dil_memory_sampling_seed
        self.RNs['sel_memory_sampling'] = sel_memory_sampling_seed

        #self.memory = ReplayBuffer(device=self.device, action_size=action_size, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=self.RNs['memory_sampling'])



        if 'Q-Learning' in self.strategy: 
            
            # Q-Network for DQN - these will be set within main.py:
            #self.DQN_weights_seed = DQN_weights_seed
            #self.memory_sampling_seed = memory_sampling_seed

            #TO DO test this !! 
            self.RNs['dil_DQN_weights'] = dil_DQN_weights_seed
            self.RNs['sel_DQN_weights'] = sel_DQN_weights_seed

            self.qnetwork_local = QNetwork(state_size=self.state_size, action_size=action_size, seed=self.RNs['dil_DQN_weights']).to(self.device) #NOTE the target and local network are initialised with the same seed
            if use_target_network:
                self.qnetwork_target = QNetwork(state_size=self.state_size, action_size=action_size, seed=self.RNs['dil_DQN_weights']).to(self.device) 
                self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict()) #NOTE what does this do??? 
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR_dilemma)
            self.dilemma_memory = ReplayBuffer(device=self.device, action_size=action_size, buffer_size=BUFFER_SIZE, seed=self.RNs['dil_memory_sampling'])
        
            self.selection_qnetwork_local = QNetwork(state_size=self.selection_state_size, action_size=self.selection_size, seed=self.RNs['sel_DQN_weights']).to(self.device) #TO DO make different weights
            if use_target_network:
                self.selection_qnetwork_target = QNetwork(state_size=self.selection_state_size, action_size=self.selection_size, seed=self.RNs['sel_DQN_weights']).to(self.device) 
                self.selection_qnetwork_target.load_state_dict(self.selection_qnetwork_local.state_dict()) #NOTE what does this do??? 
            self.selection_optimizer = optim.Adam(self.selection_qnetwork_local.parameters(), lr=LR_selection)
            self.selection_memory = SelectionReplayBuffer(device=self.device, selection_size=self.selection_size, buffer_size=BUFFER_SIZE, seed=self.RNs['sel_memory_sampling'])
            #batch_size is 1 from the point of view of each agent 
                #TO DO make sure the seed used everywhere is consistent 
            #QUESTION should this include random weights? currently it is just random? 

    if False: 
        def get_last_move(self, game, global_history): #NOT NEEDED
            ''' get opponent and learn their last move ''' #NOTE not used as we feed this to the function explicitly as 'state' 
            opponent = game.opponents[self]
            opponent_idx = game.players.index(opponent)
            last_move = global_history[f'action_player{opponent_idx}'][-1] 
            #will return None if there is no move
            print('last_move: ', last_move)
            return last_move

    def make_exploratory_move(self, iteration_dil, num_iter): 
        '''this function is used in the online learning setting - when the player is making a move according to an ExplorationPolicy'''
        #current_state_chain = list(itertools.chain(*self.current_state))
        if dilemma_state_uses_identity: #TO UPDATE 
            #translate index into 1hot vector within the player's state 
            identity = self.current_state[-1] #this uses the custom index 
            #TO DO UNDERSTAND WHY IDENTITY HERE IS 1 at iteration 0 !!!! ???? 
            identity = torch.tensor(identity, dtype=torch.int64) 
            #torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).squeeze(0).long().to(self.device)
            identity_1hot = F.one_hot(identity, num_classes=(self.population_size-1)) 
            #torch.nn.functional.one_hot(identity, num_classes=2) #also works 
            #join the two lists: 
            if state_pairs: 
                current_state_chain = list(self.current_state[0:2]) 
            else: #if state notpairs 
                current_state_chain = list(self.current_state[0:1]) 
            for i in identity_1hot.tolist(): #TO DO use itertools here 
                current_state_chain.append(i)
        else:
            current_state_chain = self.current_state
            #current_state = self.current_selection_state[opponent_index]
            #current_state_chain = current_state
 
        my_expl_policy = ExplorationPolicy(state_chain=current_state_chain, selection_state_chain=None, policy_type='eps_greedy') # QV=self.Q_values
        my_expl_policy.alg_type = self.alg_type #define if using tabularQ or DQN 
        my_expl_policy.player = self 
        move, eps, reason = my_expl_policy.use_policy(iteration_dil=iteration_dil, total_iterations=num_iter) #eps0=self.eps0, epsdecay=self.epsdecay
        return int(bool(move)), eps, reason
    
    def make_fixed_move(self, state):
        '''this function can be used to play a fixed strategy - not exploring'''
        if self.strategy == 'random':
            RN = self.RNs['static_random']
            return int(RN < 0.5)
        
        elif self.strategy == 'TitForTat': 
            '''respond to opponent's last move using a reactive strategy based on 1 last move of the opponent'''
            #react to last move - NOTE currently we assume a state will always be given. If not - add the code below: 
            #if state is None: #if this is the initial move
            #    return int(self.initial_move) #TO DO check that we are not duplicating initial_move in other functions 
            #else:
            #if self.t_step_dilemma == 1: #cooperate on the first move 
            if self.latest_move == None: #if this is the first iteration, play randomy
                return int(False) 
            else: 
                return int(bool(state[0])) #take the first element of the state, i.e. the opponent's previous move
        
        elif self.strategy == 'AlwaysCooperate':
            return int(False)
            
        elif self.strategy == 'AlwaysDefect':
            return int(True) 
        
        if self.latest_move == None: #if this is the first iteration, play randomy
            return int(np.random.choice([False, True]))
        else: 
            if self.strategy == 'Alternating':
                latest_move = bool(self.latest_move) 
                return int(not latest_move) #1-latest_move #alternate moves
        
            elif self.strategy == 'GilbertElliot_typea': #Cooperate most of the time 
                if self.latest_move == 0: #if previously Cooperated
                    return int(np.random.choice([False, True], p=[0.99, 0.01]))
                elif self.latest_move == 1: #if previously Defected 
                    return int(np.random.choice([False, True], p=[0.99, 0.01]))
                
            elif self.strategy == 'GilbertElliot_typeb': #Cooperate lots at the start, then always Defect 
                if self.latest_move == 0: #if previously Cooperated
                    return int(np.random.choice([False, True], p=[0.8, 0.2]))
                elif self.latest_move == 1: #if previously Defected 
                    return int(True) #always defect from now on
            
            elif self.strategy == 'GilbertElliot_typec': #80% chance of alternating moves, 20% chance of staying with the same move
                if self.latest_move == 0: #if previously Cooperated
                    return int(np.random.choice([False, True], p=[0.2, 0.8]))
                elif self.latest_move == 1: #if previously Defected 
                    return int(np.random.choice([False, True], p=[0.8, 0.2]))
                
            elif self.strategy == 'GilbertElliot_typed': #80% chance of Defecting
                if self.latest_move == 0: #if previously Cooperated
                    return int(np.random.choice([False, True], p=[0.2, 0.8]))
                elif self.latest_move == 1: #if previously Defected 
                    return int(np.random.choice([False, True], p=[0.2, 0.8]))
                
            elif self.strategy == 'GilbertElliot_typee': #80% chance of Cooperating
                if self.latest_move == 0: #if previously Cooperated
                    return int(np.random.choice([False, True], p=[0.8, 0.2]))
                elif self.latest_move == 1: #if previously Defected 
                    return int(np.random.choice([False, True], p=[0.8, 0.2]))

            
    #def record(self, game):
    #    self.games_played.append(game) #NOT USED
    #    opponent = game.opponents[self]
    #    self.players_played.append(opponent) #NOT USED
        
    def intrinsic_reward(self, game, m1, m2, state):
        ''' Calculate intrinsic reward for the player based on their moral_type, and the last two moves m1 & m2.
        '''
        #create the baseline individual payoffs, as defined in the IPD game
        payoffs = (game.payoffmat[m1][m2]) 
        
        #extract integers from the tuple with payoffs
        pay1 = payoffs[0]
        pay2 = payoffs[1]

        
        ## PROSOCIAL AGENTS ##
        if self.moral_type == 'Utilitarian':
            pay1_intrinsic = pay1 + pay2
            
        elif self.moral_type == 'Deontological':
            #check if I followed the norm conditional cooperation
            if type(state) == list or type(state) == tuple: #if using state_pairs, or dilemmastateusesidentity, ignore
            #if len(state>1): #if using state_pairs, or dilemmastateusesidentity, ignore
                opponent_last_move = state[0]
            else: 
                opponent_last_move = state
                
            if opponent_last_move == 0: 
                if m1 == 1: #if I (player1) defected against a cooperator (based on 1 previous move of the opponent), get punished
                    pay1_intrinsic = -xi
                else: 
                    pay1_intrinsic = 0
            else: 
                pay1_intrinsic = 0
            
            #check if player followed the external norm reciprocity - give what you receive
            #if yes - reward = +5, otherwise reward = -5 
            #if m1 == state: #if I reciprocated/copied my opponent's previous move
            #    pay1_intrinsic = 5
            #else:
            #    pay1_intrinsic=-5

            #can add more norms here 


        elif self.moral_type == 'VirtueEthics_equality':
            #pay1_intrinsic = (min(pay1, pay2) +1) / (max(pay1, pay2) +1) #OLD VERSION
            pay1_intrinsic = 1 - ((abs(pay1 - pay2)) / (pay1 + pay2)) #a simplification of the Gini coefficient for 2 players

        elif self.moral_type == 'VirtueEthics_kindness':
            if m1 == False: #if this agent cooperated, get rewarded
                pay1_intrinsic = xi
            else: 
                pay1_intrinsic = 0

        elif self.moral_type == 'VirtueEthics_mixed':
            mixed_beta = int(self.mixed_beta)
            k_normalised = xi/xi
            if m1 == False: #if this agent cooperated
                pay1_intrinsic = mixed_beta * (1 - ((abs(pay1 - pay2)) / (pay1 + pay2))) + (1-mixed_beta) * k_normalised 
            else: 
                pay1_intrinsic = mixed_beta * (1 - ((abs(pay1 - pay2)) / (pay1 + pay2))) 
        
        elif self.moral_type == 'Altruist':
            pay1_intrinsic = pay2

        ## MALICIOUS AGENTS ## 
        elif self.moral_type == 'anti-Utilitarian':
            pay1_intrinsic = -(pay1 + pay2)

        elif self.moral_type == 'malicious_Deontological':
            if type(state) == list or type(state) == tuple: #if using state_pairs, or dilemmastateusesidentity, ignore
            #if state_pairs:
                opponent_last_move = state[0]
            else: 
                opponent_last_move = state 
                
            if opponent_last_move == 0: 
                if m1 == 1: #if I (player1) defected against a cooperator (based on 1 previous move of the opponent), get punished
                    pay1_intrinsic = xi
                else: 
                    pay1_intrinsic = 0
            else: 
                pay1_intrinsic = 0

        elif self.moral_type == 'VirtueEthics_inequality':
            pay1_intrinsic = (abs(pay1 - pay2)) / (pay1 + pay2)

        elif self.moral_type == 'VirtueEthics_aggressiveness':
            if m1 == True: #if this agent defected, get rewarded
                pay1_intrinsic = xi
            else: 
                pay1_intrinsic = 0

        elif self.moral_type == 'anti-Altruist':
            pay1_intrinsic = -pay2

    # TO DO - Rescale rewards to all be on the same scale ?? 

        ## TRADITIONAL SELFISH AGENT ## 
        elif self.moral_type == None: 
            pay1_intrinsic = None
            
        return pay1_intrinsic

        
    def total_reward(self, game, m1, m2):
        return 0 
    #TO DO - implement set of weights alpha & beta on entrinsic vs intrinsic reward 


    def update_Qnetwork(self, states, actions, rewards, next_states, gamma): #experiences
        """Update dilemma action value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s') tuples - NOTE this uses state_index instead of the 2-item state tself 
            gamma (float): discount factor
        """
        # Obtain random minibatch of tuples from D
        #states, actions, rewards, next_states = experiences

        #TO DO transform action to 1-hot encoding 
        if state_pairs: 
            actions_1hot = F.one_hot(actions, num_classes=action_size).squeeze(1) #NEW
        else: #if state not pairs  
            if self.dilemma_memory.batch_size > 1: 
                actions_1hot = F.one_hot(actions, num_classes=action_size).squeeze(1)
            elif self.dilemma_memory.batch_size == 1: 
                actions_1hot = F.one_hot(actions, num_classes=action_size).squeeze(0)
        #if using state_index 
        #states_chain_tensor = F.one_hot(states, num_classes=state_size).squeeze(1) #NEW
        #next_states_chain_tensor = F.one_hot(next_states, num_classes=state_size).squeeze(1) #NEW

        #reformat state so it 
        if False: #don't think we need this any more 
            if self.dilemma_memory.batch_size > 1: #if BATCH_SIZE > 1:
                #TO DO Check this - is chain the right procedure to use? 
                states_chain = [list(itertools.chain(*state)) for state in states]
                next_states_chain = [list(itertools.chain(*state)) for state in states]

                states_chain_tensor = states_chain.float()
                #states_chain_tensor = torch.tensor(states_chain, dtype=torch.float32)
                next_states_chain_tensor = next_states_chain.float()
                #next_states_chain_tensor = torch.tensor(next_states_chain, dtype=torch.float32)

        #elif self.dilemma_memory.batch_size == 1: #elif BATCH_SIZE == 1:
        if dilemma_state_uses_identity:
            if state_pairs: 
                #create states_chain_tensor
                identity = states[-1] #already a tensor 
                #identity = torch.tensor(identity, dtype=torch.int64)
                identity_1hot = F.one_hot(identity, num_classes=(self.population_size-1))
                current_state_chain = states[0:2].tolist()
                for i in identity_1hot.tolist(): #TO DO use itertools here 
                    current_state_chain.append(i)
                states_chain_tensor = torch.tensor(current_state_chain, dtype=torch.int).long().float().to(self.device)

                #create next_states_chain_tensor
    #TO DO make this update a step later - so that next_state[-1] is not None! 
                identity = next_states[-1]
                #identity = torch.tensor(identity, dtype=torch.int64)
                identity_1hot = F.one_hot(identity, num_classes=(self.population_size-1))
                next_state_chain = next_states[0:2].tolist()
                for i in identity_1hot.tolist(): #TO DO use itertools here 
                    next_state_chain.append(i)
                next_states_chain_tensor = torch.tensor(next_state_chain, dtype=torch.int).long().float().to(self.device)
            else: #if state not pairs
#TO DO fix this since removing state_pairs 
                #create states_chain_tensor
                identity = states[-1] #already a tensor 
                identity_1hot = F.one_hot(identity, num_classes=(self.population_size-1))
                current_state_chain = states[0:1].tolist()
                for i in identity_1hot.tolist(): #TO DO use itertools here 
                    current_state_chain.append(i)
                states_chain_tensor = torch.tensor(current_state_chain, dtype=torch.int).long().float().to(self.device)
                    
                #create next_states_chain_tensor
    #TO DO make this update a step later - so that next_state[-1] is not None! 
                identity = next_states[-1]
                #identity = torch.tensor(identity, dtype=torch.int64)
                identity_1hot = F.one_hot(identity, num_classes=(self.population_size-1))
                next_state_chain = next_states[0:1].tolist()
                for i in identity_1hot.tolist(): #TO DO use itertools here 
                    next_state_chain.append(i)
                next_states_chain_tensor = torch.tensor(next_state_chain, dtype=torch.int).long().float().to(self.device)


        else: #if not using identity in dilemma state
        #TO DO understand if this new shape is preferred to the shape of states itself 
        # - tensor[[]] is the same dimension as the 1-hot encoded actions tensor [[]]
            if state_pairs: 
                states_chain_tensor = states.reshape(1, self.state_size).float().to(self.device)
        #TO DO check that test is of len 4 in the population_size=3 case 
                next_states_chain_tensor = next_states.reshape(1, self.state_size).float().to(self.device)
            else: #if states not using pairs 
                states_chain_tensor = states.float().to(self.device)
                next_states_chain_tensor = next_states.float().to(self.device)

        #transform state to a float so it is the same format as network weights 
        #states_1hot = states_1hot.float().to(self.device)
        #next_states_1hot = next_states_1hot.float().to(self.device)

    
        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        ## Expected values of actions for non_final_next_states are computed based on the "older" target_net; selecting their best reward with max(1)[0].
        if use_target_network:
            if self.dilemma_memory.batch_size > 1: #if BATCH_SIZE > 1:
                q_targets_next = self.qnetwork_target(next_states_chain_tensor).detach().max(1)[0].unsqueeze(1) 
            elif self.dilemma_memory.batch_size == 1: #elif BATCH_SIZE == 1:
                q_targets_next = self.qnetwork_target(next_states_chain_tensor).detach().max()#.unsqueeze(0)
        else: 
            if self.dilemma_memory.batch_size > 1: #if BATCH_SIZE > 1:
                q_targets_next = self.qnetwork_local(next_states_chain_tensor).detach().max(1)[0].unsqueeze(1) 
            elif self.dilemma_memory.batch_size == 1: #elif BATCH_SIZE == 1:
                q_targets_next = self.qnetwork_local(next_states_chain_tensor).detach().max().reshape(1) 
        #TO DO check with batch > 1 may need to unsqueeze 
        #NOTE TO DO - test max(1)[0] here - currently it is not being used given history = 1 !!!! 
        #.detach() removes the gradient info from the output of the qnetwork; 
        # .unsqueeze() reformats it slightly to add a fake batch dimension (torch only takes minibathces of samples)
    #TO DO understand the line above!! 

        ### Calculate target value from bellman equation
        q_targets = rewards + gamma * q_targets_next

        ### Calculate expected value from local network
        #if self.qnetwork_local(states_chain_tensor).shape[0] >1: 
#TO DO understns why shapes of self.qnetwork_local(states_chain_tensor) (shape [2]) and actions_1hot (shape [1,2]) do not match up here 
        #q_expected = self.qnetwork_local(states_chain_tensor)[actions_1hot.bool()].unsqueeze(1) 
        if self.dilemma_memory.batch_size > 1: #if BATCH_SIZE > 1:
            #q_expected = torch.reshape(self.qnetwork_local(states_chain_tensor), [1,2])[actions_1hot.bool()].unsqueeze(1) 
            q_expected = self.qnetwork_local(states_chain_tensor)[actions_1hot.bool()].unsqueeze(1) 
        elif self.dilemma_memory.batch_size == 1: # elif BATCH_SIZE == 1:
            #q_expected = torch.reshape(self.qnetwork_local(states_chain_tensor), [1,2])[actions_1hot.bool()] 
            q_expected = self.qnetwork_local(states_chain_tensor)[actions_1hot.bool()] 

            #.gather(1, actions_1hot) #TO DO CHECK that this works with training on >1 states at once (BATCH_SIZE >1) 
            #NOTE this should be outputting q-values for each action rather than the actions to take ... !!! 
            #NOTE check why we use .gather() on local network but not on target
        #else: 
        #    q_expected = self.qnetwork_local(states_chain_tensor)[actions_1hot.bool()].unsqueeze(1)
        
        ### Loss calculation (we used MSE loss)
        criterion = nn.MSELoss() #or moothL1Loss()
        loss = criterion(q_expected, q_targets) #loss function takes in (intput, target)
        #e.g. using nn.functional instead: 
        #loss = F.mse_loss(q_expected, q_targets) 

        ### Optimize the model 
        self.optimizer.zero_grad() #Zero the existing gradient buffers of all parameters
        loss.backward() #TO DO check this is backpropagating through target network only 
#NOTE the above returns a number, not within an array of size 1 
        #backpropagate the error
        #the backward() function, where gradients are computed, is automatically defined using autograd 
        # after calling .backward(), all Tensors in the graph that have requires_grad=True will have their .grad Tensor accumulated with the gradient
        self.optimizer.step() #update the weights using method from self.optimizer

        # ------------------- update target network ------------------- #
        if self.t_step_dilemma % UPDATE_EVERY_dilemma == 0:
#QUESTION should this be updating every t iter globally or as far as this player is concerned? 
            if use_target_network:
                self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)   
            else: 
                self.soft_update(self.qnetwork_local, self.qnetwork_local, TAU)   


        #TO DO only update this every 100 iterations 
        if self.record_Qvalues == 'both': 

            if torch.cuda.is_available():
                states_chain_tensor = states_chain_tensor.cpu()
                actions = actions.cpu()
                q_targets = q_targets.cpu()
                q_expected = q_expected.cpu()

            states_chain_tensor_int = states_chain_tensor.int()

            #TO DO fix this !!! state not currently being saved correctly 
            #store latest Q-values 
            #temp_list = list(zip(str(states.numpy()), tuple(actions.numpy()), q_targets.squeeze(1).numpy(), q_expected.detach().squeeze(1).numpy()))
            if self.dilemma_memory.batch_size > 1: #if BATCH_SIZE > 1:
                #temp_list = list(zip(str(states_chain_tensor.squeeze(1).numpy()), tuple(actions.squeeze(1).numpy()), q_targets.squeeze(1).numpy(), q_expected.detach().squeeze(1).numpy()))
                data = {'state_reshaped':states_chain_tensor_int.squeeze(1).numpy(), 'action':actions.squeeze(1).numpy(), 'q_target':q_targets.squeeze(1).numpy(), 'q_expected':q_expected.detach().squeeze(1).numpy()}
            elif self.dilemma_memory.batch_size == 1: #elif BATCH_SIZE == 1:
                #temp_list = list(zip(str(states_chain_tensor.numpy()), tuple(actions.numpy()), q_targets.squeeze(1).numpy(), q_expected.detach().squeeze(1).numpy()))
                #temp_list = list(zip(str(states_chain_tensor.numpy()), tuple(actions.numpy()), q_targets.numpy(), q_expected.detach().numpy()))
                #temp_list = list(zip(states_chain_tensor_int.numpy(), actions.numpy(), q_targets.numpy(), q_expected.detach().numpy()))
                data = {'state_reshaped':states_chain_tensor_int.numpy(), 'action':actions.numpy(), 'q_target':q_targets.numpy(), 'q_expected':q_expected.detach().numpy()}

            #temp_list = list(zip(states.squeeze(1).numpy(), actions.squeeze(1).numpy(), q_targets.squeeze(1).numpy(), q_expected.detach().squeeze(1).numpy()))
            #temp_df = pd.DataFrame(temp_list, columns=['state_reshaped', 'action', 'q_target', 'q_expected'])
            temp_df = pd.DataFrame(data)


            temp_df['state_reshaped_str'] = temp_df['state_reshaped'].astype(str)
            #store just the latest Q-value from this batch,  each row is indexed by state itself:
            temp_new = temp_df.groupby(['state_reshaped_str', 'action']).tail(1).reset_index(drop=True)  

            for row_idx in temp_new.index:
                state = temp_new.iloc[row_idx]['state_reshaped']
                action = temp_new.iloc[row_idx]['action']

                #self.Qvalues_expected[list(state), selection] = round(temp_df.iloc[row_idx]['q_expected'], 2)
                #self.Qvalues_expected['to_be_index'].apply(lambda x: ''.join(str(e) for e in x))
                if state_pairs: 
                    state_stored = ''.join(str(e) for e in state)
                else: #if state not pairs
                    state_stored = state
#TO DO incorporate recowd_Qvalues_every here !!!! 
                self.Qvalues_expected_dilemma[str(tuple([state_stored, action, self.t_step_dilemma]))] = round(temp_df.iloc[row_idx]['q_expected'], 2)
                #NOTE TO DO check that the above works!!!! 


        self.current_loss = loss
        #return loss


    def soft_update(self, local_model, target_model, TAU):
            """Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target

            Params
            ======
                local_model (PyTorch model): weights will be copied from
                target_model (PyTorch model): weights will be copied to
                tau (float): interpolation parameter 
            """
            #for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            #    target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)

            # When TAU=1, target_net_state_dict[key] =  policy_net_state_dict[key]*1

            target_net_state_dict = target_model.state_dict()
            policy_net_state_dict = local_model.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_model.load_state_dict(target_net_state_dict)
    

    def act_DQN(self, state_chain):
        """ Takes in a state (2D tuple), converts it into a usable tensor, 
        and outputs an action from the current learn policy network. """
        #state = state_index.float().unsqueeze(0).to(self.device)
        #states = torch.from_numpy(np.vstack(state_index)).long().to(self.device)
        #if using_state_index:
        #state = torch.tensor(state_index, dtype=torch.int).long()
        #state_1hot = F.one_hot(state, num_classes=4).float().to(self.device)  
        #USING STATE ITSELF 
        state_tensor = torch.tensor(state_chain, dtype=torch.int).long()
        state_tensor = state_tensor.float().to(self.device)
        if state_pairs == False: 
            state_tensor = state_tensor.unsqueeze(0)
        #TO DO update action here to fit IPD env
        self.qnetwork_local.eval() #switch to eval mode 
        #NOTE check if I need eval() and train() here or if torch.no_grad does the same 
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train() #switch back to train mode for future training        
        # Epsilon-greedy action selection

        #TO DO add check if both 0, then choose randomly

        #alternative way to take action:
        # action_values.max(1)[1].view(1, 1)  
        #TO DO update for n>1
        
        return np.argmax(action_values.cpu().data.numpy())
    

    def make_exploratory_selection(self, iteration_sel, num_iter): 
        '''this function is used in the online learning setting - when the player is making a move according to an ExplorationPolicy'''
        #current_selection_state_index = game.selection_state_index_converter.index(self.current_selection_state)
        if state_pairs: 
            current_selection_state_chain = list(itertools.chain(*self.current_selection_state))
        else: 
            current_selection_state_chain = self.current_selection_state

        my_expl_policy = ExplorationPolicy(state_chain=[], selection_state_chain=current_selection_state_chain, policy_type='eps_greedy') # QV=self.Q_values
        my_expl_policy.alg_type = self.alg_type #define if using tabularQ or DQN 
        my_expl_policy.player = self 
        selection_idx, eps_selection, reason_selection = my_expl_policy.use_policy_selection(iteration_sel=iteration_sel, total_iterations=num_iter) #eps_selection0=self.eps_selection0, epsdecay=self.epsdecay
        #note the above usues the GLOBAL index for the opponent 
        return selection_idx, eps_selection, reason_selection
    
    def make_fixed_selection(self):
        selection_idx = int(1) #force-select AlwaysCooperate oppponent - note this is the global idx
        reason_selection = 'fixed selection of AC player'
        return  selection_idx, reason_selection
    
    def select_DQN(self, current_selection_state_chain):
        """ Takes in a state (2D tuple), converts it into a usable tensor, 
        and outputs an action from the currently learnt policy network. """
        #state = state_index.float().unsqueeze(0).to(self.device)
        #states = torch.from_numpy(np.vstack(state_index)).long().to(self.device)
        selection_state_tensor = torch.tensor(current_selection_state_chain, dtype=torch.int).long()
        selection_state_tensor = selection_state_tensor.float().to(self.device)
        #selection_state_1hot = F.one_hot(selection_state, num_classes=selection_state_size).float().to(self.device)  

        #TO DO update action here to fit IPD env
        self.qnetwork_local.eval() #switch to eval mode #TO DO check if this is needed? perhaps line below does the same thing 
        with torch.no_grad():
            selection_values = self.selection_qnetwork_local(selection_state_tensor)
        self.selection_qnetwork_local.train() #switch back to train mode for future training        
        # Epsilon-greedy action selection

        #TO DO add check if both 0, then choose randomly

        #alternative way to take action:
        # action_values.max(1)[1].view(1, 1)  
        #TO DO update for n>1

        #print('selection_DQN: ', np.argmax(selection_values.cpu().data.numpy()))
        
        #TO DO check that the below will return local index
        return np.argmax(selection_values.cpu().data.numpy())


    def update_selection_Qnetwork(self, selection_states, selections, selection_rewards, selection_next_states, gamma): #experiences
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s') tuples - NOTE this now uses the multi-dimensional state, not the state_index any more 
            gamma (float): discount factor
        """
        # Obtain random minibatch of tuples from D
        #states, actions, rewards, next_states = experiences

        #TO DO check that transforming action to 1-hot encoding works for the NN output 
        #num_state_classes = selection_state_size #len(selection_state_index_converter)
        if state_pairs: 
            selections_1hot = F.one_hot(selections, num_classes=self.population_size-1).squeeze(1) #NEW
        else: 
            selections_1hot = F.one_hot(selections, num_classes=self.population_size-1).squeeze(0)

        #states_1hot = F.one_hot(selection_states, num_classes=num_state_classes).squeeze(1) #NEW
        #next_states_1hot = F.one_hot(selection_next_states, num_classes=num_state_classes).squeeze(1) #NEW
#NOTE in the state_pairs=False case, the above selections_1hot object if of shape [1,2], but states_chain_tensor, for example, is of shape [2] !!! 

        #reformat state so it fits the NN dimensions 
        if False: #selection memory batch cannot be >1 
            if BATCH_SIZE > 1:
                states_chain = [list(itertools.chain(*state)) for state in selection_states]
                next_states_chain = [list(itertools.chain(*state)) for state in selection_next_states]

                states_chain_tensor = states_chain.float()
                #states_chain_tensor = torch.tensor(states_chain, dtype=torch.float32)
                next_states_chain_tensor = next_states_chain.float()
                #next_states_chain_tensor = torch.tensor(next_states_chain, dtype=torch.float32)

        #elif BATCH_SIZE == 1:
        if state_pairs:
            states_chain_tensor = selection_states.reshape(1, self.selection_state_size)
            next_states_chain_tensor = selection_next_states.reshape(1, self.selection_state_size)
        else: #if states not using pairs 
            states_chain_tensor = selection_states
            next_states_chain_tensor = selection_next_states

            #transform state to a float so it is the same format as network weights 
            #state = T.tensor([state], dtype=T.float).to(self.critic.device)
            #states_1hot = states_1hot.float().to(self.device)
            #next_states_1hot = next_states_1hot.float().to(self.device)
            #states_chain = [chain.float().to(self.device) for chain in states_chain]
            #next_states_chain = next_states_chain.float().to(self.device)
            #next_states_chain = [chain.float().to(self.device) for chain in next_states_chain]
        
        
        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        ## Expected values of actions for non_final_next_states are computed based on the "older" target_net; selecting their best reward with max(1)[0].
        if use_target_network:
            if state_pairs: 
                q_targets_next = self.selection_qnetwork_target(next_states_chain_tensor).detach().max(1)[0].unsqueeze(1) 
            else: #if state not pairs 
                q_targets_next = self.selection_qnetwork_target(next_states_chain_tensor).detach().max()
        else:
            if state_pairs: 
                q_targets_next = self.selection_qnetwork_local(next_states_chain_tensor).detach().max(1)[0].unsqueeze(1) 
            else: #if state not pairs 
                q_targets_next = self.selection_qnetwork_local(next_states_chain_tensor).detach().max()

#TO DO understand why max(1) works here but not in dilemma network update 
#the above returns tensor([[0.5848, 0.0959]], grads) --> tensor([[0.5848]]) after unsqueezing 
#note selection_rewards here is in shape tensor([3.]) ... --> how do these match up? 

        #TO DO check with batch > 1 may need to undqueeze 
        #NOTE TO DO - test max(1)[0] here - currently it is not being used given history = 1 !!!! 
        #.detach() removes the gradient info from the output of the qnetwork; 
        # .unsqueeze() reformats it slightly to add a fake batch dimension (torch only takes minibathces of samples)
    #TO DO understand the line above!! 
        ### Calculate target value from bellman equation
        q_targets = selection_rewards + gamma * q_targets_next
        ### Calculate expected value from local network
        if self.selection_qnetwork_local(states_chain_tensor).shape[0] >1:
            if state_pairs: 
                q_expected = self.selection_qnetwork_local(states_chain_tensor)[selections_1hot.bool()].unsqueeze(1) 
            #.gather(1, actions_1hot) #TO DO CHECK that this works with training on >1 states at once (BATCH_SIZE >1) 
            #NOTE this should be outputting q-values for each action rather than the actions to take !!! 
            #NOTE check why we use .gather() on local network but not on target
            else: #if state not pairs
                q_expected = self.selection_qnetwork_local(states_chain_tensor)[selections_1hot.bool()]
        else: 
            q_expected = self.selection_qnetwork_local(states_chain_tensor)[selections_1hot.bool()]
        
        ### Loss calculation (we used MSE loss)
        criterion = nn.MSELoss() #or moothL1Loss()
        loss = criterion(q_expected, q_targets) #loss function takes in (intput, target)
        #e.g. using nn.functional instead: 
        #loss = F.mse_loss(q_expected, q_targets) 

        ### Optimize the model 
        self.selection_optimizer.zero_grad() #Zero the existing gradient buffers of all parameters
        loss.backward() #TO DO check this is backpropagating through target network only 
        #backpropagate the error
        #the backward() function, where gradients are computed, is automatically defined using autograd 
        # after calling .backward(), all Tensors in the graph that have requires_grad=True will have their .grad Tensor accumulated with the gradient
        self.selection_optimizer.step() #update the weights using method from self.optimizer

        # ------------------- update target network ------------------- #
        if self.t_step_selection % UPDATE_EVERY_selection == 0: 
            #NOTE should UPDATE_EVERY be the same value as population_size-1? 
            if use_target_network:
                self.soft_update(self.selection_qnetwork_local, self.selection_qnetwork_target, TAU)   
            else:
                self.soft_update(self.selection_qnetwork_local, self.selection_qnetwork_local, TAU)   


        self.current_selection_loss = loss#.detach()

        

        if self.record_Qvalues == 'both': 
            #store latest Q-values 
            #states_chain = [list(itertools.chain(*state)) for state in selection_states] #TO DO test this 
            #states_chain_tuple_int = tuple(selection_states.int().numpy()) #tuple(states_chain_tensor)

            if torch.cuda.is_available():
                states_chain_tensor = states_chain_tensor.cpu()
                #selection_states = selection_states.cpu()
                selections = selections.cpu()
                q_targets = q_targets.cpu()
                q_expected = q_expected.cpu()

            #TO DO solve error that pops up on cluster: TypeError: only integer tensors of a single element can be converted to an index
            if False: #selection memory batch cannot be >1 
                if BATCH_SIZE > 1:
                    states_chain_tensor_int = states_chain_tensor.int()
                    temp_list = list(zip(states_chain_tensor_int.squeeze(1).numpy(), selections.squeeze(1).numpy(), q_targets.squeeze(1).numpy(), q_expected.detach().squeeze(1).numpy()))
                    temp_df_sel = pd.DataFrame(temp_list, columns=['state_reshaped', 'selection', 'q_target', 'q_expected'])
                    temp_df_sel['state_reshaped'] = temp_df_sel['state_reshaped'].astype(str)

            #elif BATCH_SIZE == 1:
            if state_pairs: 
                states_chain_tensor_int = states_chain_tensor.int()
                #temp_list = list(zip(states_chain_tensor_int.numpy(), selections.numpy(), q_targets.squeeze(1).numpy(), q_expected.detach().squeeze(1).numpy()))
                temp_df_sel = pd.DataFrame({'state_reshaped':states_chain_tensor_int.numpy(), 'selection':selections.numpy(), 'q_target':q_targets.numpy(), 'q_expected':q_expected.detach().numpy()})
                temp_df_sel['state_reshaped'] = temp_df_sel['state_reshaped'].astype(str)
            else: #if state not pairs
                if len(states_chain_tensor)>1: 
                    states_chain_tuple_int = tuple(states_chain_tensor.int().numpy()) #tuple(states_chain_tensor)
                else: 
                    states_chain_tuple_int = int(states_chain_tensor)
                #temp_list = list(zip(str(states_chain_tuple_int), selections.numpy(), q_targets.numpy(), q_expected.detach().numpy()))
                temp_df_sel = pd.DataFrame({'state_reshaped':str(states_chain_tuple_int), 'selection':selections.numpy(), 'q_target':q_targets.numpy(), 'q_expected':q_expected.detach().numpy()})

            #temp_list = list(zip(states_chain.squeeze(1).numpy(), selections.squeeze(1).numpy(), q_targets.squeeze(1).numpy(), q_expected.detach().squeeze(1).numpy()))
            #temp_df_sel = pd.DataFrame(temp_list, columns=['state_reshaped', 'selection', 'q_target', 'q_expected'])
            #TO DO check that state_reshaped is the right term here - or should it be simply state? 
            #temp_df_sel['state_reshaped_str'] = temp_df_sel['state_reshaped'].astype(str)
            #store just the latest Q-value from this batch,  each row is indexed by state itself:
            temp_new = temp_df_sel.groupby(['state_reshaped', 'selection']).tail(1)#.reset_index(drop=True)  

            #TO DO debug whether these are being saved correctly
            for row_idx in temp_new.index:
                state = temp_new.iloc[row_idx]['state_reshaped']
                selection = temp_new.iloc[row_idx]['selection']

                #self.Qvalues_expected[list(state), selection] = round(temp_df_sel.iloc[row_idx]['q_expected'], 2)
                #self.Qvalues_expected['to_be_index'].apply(lambda x: ''.join(str(e) for e in x))
                if state_pairs: 
                    state_str = ''.join(str(e) for e in state)
                else:
                    state_str = str(state)
                self.Qvalues_expected_selection[str(tuple([state_str, selection, self.t_step_selection]))] = round(temp_df_sel.iloc[row_idx]['q_expected'].item(), 2)
                #TO DO check that the above works!!!! 

                #self.Qvalues_targets[state, selection] = round(temp_df_sel.iloc[row_idx]['q_target'], 2)

        #return loss

