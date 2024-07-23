import numpy as np 
#import torch
#import torch.nn.functional as F

from config import episodes, eps_dil_initial, epsdecay, eps_dil_final, eps_sel_initial, eps_sel_final #, selection_state_size

 
class ExplorationPolicy:   
    def __init__(self, state_chain, selection_state_chain, policy_type): 
        #self.state_index = state_index
        self.state_chain = state_chain
        #self.selection_state_index = selection_state_index
        self.current_selection_state_chain = selection_state_chain
        self.policy_type = policy_type
        # random numbers and Q_values will be passed via player 
        # self.RNs = RNs
        # self.QV = QV

        self.alg_type = None #determines if this is tabularQ or DQN 
        self.player = None #make a player attribute available for DQN action selection 

    def use_policy(self, iteration_dil, total_iterations): #eps0, epsdecay

        if self.policy_type == 'eps_greedy': 
            '''eps-greedy exploration policy - to be used in taking actions during learning'''
            prob = self.player.RNs['eps_prob'] #.RNs[1]
            if not epsdecay: 
                eps = eps_dil_initial
            else: # if I need to implement eps_decay and eps0 has been pre-defined
                #eps_dil_initial & eps_dil_final are read in from config
                if int(iteration_dil) < 20000: #decay for the irst 20K iteration_dil
                    r = max((20000-int(iteration_dil))/20000, 0)
                    eps = (eps_dil_initial - eps_dil_final)*r + eps_dil_final
                else: #no longer decay after 2000-0 iteration_dil 
                    eps = eps_dil_final 
                #r = max((int(total_iterations)-int(iteration_dil))/int(total_iterations), 0)
                #eps = (eps_dil_initial - eps_dil_final)*r + eps_dil_final

            if prob <= eps: 
                #make a random move with probability eps 
                reason = 'random, dilemma eps'
                #temp = [self.player.RNs['eps_move'], int(self.player.RNs['eps_move'] < 0.5)]
                return int(self.player.RNs['eps_move'] < 0.5), eps, reason #temp #prob
            else:             
                #move optimally based on current Q-value estimates, if they are not empty 

                if self.alg_type == 'tabularQ': #if using tabularQ: 
                    if not np.any(self.player.Q_values[self.state_index]): #if Q-values for this state are empty
                        reason = 'random, empty Q-values'
                        return int(self.player.RNs['empty_Q'] < 0.5), eps, reason #make a random move
                    else: 
                        optimal_policy = np.argmax(self.player.Q_values, axis=1) #list(np.argmax(self.QV, axis=1))
                        reason = 'greedy action'
                        return int(bool(optimal_policy[self.state_index])), eps, reason

                elif self.alg_type == 'DQN': #if using DQN:
                    #check if non-0 Q-values are available for this state from the Q-network
                    #NOTE check DQN syntax here 
                    #state = torch.tensor(self.state_index, dtype=torch.int).long()
                    #state_1hot = F.one_hot(state, num_classes=4).float()#.to(self.device)  

                    #TO DO update action here to fit IPD env
                    #self.player.qnetwork_local.eval()
                    #TO DO check what the above line does 
                    #with torch.no_grad():
                    #    action_values = self.player.qnetwork_local(state_1hot)
                    #self.player.qnetwork_local.train()        # Epsilon-greedy action selection
                    #TO DO check what the above line does 

                    #TO DO add check if both 0, then choose randomly

                    #if action_values.cpu().data.numpy() == [0, 0]:
                    #if [0, 0] in self.player.Qvalues_expected[self.state_index]:
                    #if len(self.player.memory) < BATCH_SIZE:
                    if episodes: 
                        if (self.player.t_step_dilemma == 0) or (self.player.t_step_dilemma < 1): #if haven't filled up memory yet
                            reason = 'random, non-full dilemma memory'
                            #temp2 = [self.player.RNs['empty_Q'], int(self.player.RNs['empty_Q'] < 0.5)]
                            return int(self.player.RNs['empty_Q'] < 0.5), eps, reason #temp2 #prob #make a random move
                        else: 
                            reason = 'greedy move'
                            #action = self.player.act_DQN(self.state_index) 
                            action = self.player.act_DQN(self.state_chain) 
                            return int(action), eps, reason
                    
                    elif episodes == False: #use dilemma_memory length 
                        if len(self.player.dilemma_memory) < 1:
                            #TO DO add option for player.memory as well - as in dyadic case 
                            reason = 'random, non-full dilemma memory'
                            #temp2 = [self.player.RNs['empty_Q'], int(self.player.RNs['empty_Q'] < 0.5)]
                            return int(self.player.RNs['empty_Q'] < 0.5), eps, reason #temp2 #prob #make a random move
                        else: 
                            reason = 'greedy move'
                            #action = self.player.act_DQN(self.state_index) 
                            action = self.player.act_DQN(self.state_chain) 
                            return int(action), eps, reason



    def use_policy_selection(self, iteration_sel, total_iterations): #eps_selection0, epsdecay

        if self.policy_type == 'eps_greedy': 
            '''eps-greedy exploration policy - to be used in taking actions during learning'''
            prob = self.player.RNs['selection_eps_prob'] #.RNs[1] #TO DO add RN stream for selection 
            if not epsdecay: 
                eps_selection = eps_sel_initial #try 0.05 #0.01 #0.001 
            else: # if I need to implement eps_decay and eps0 has been pre-defined
                #eps_sel_initial & eps_sel_initial get passed in from config 
                if int(iteration_sel) < 10000: #decay for the irst 20K iteration_dil
                    r = max((10000-int(iteration_sel))/10000, 0)
                    eps_selection = (eps_sel_initial - eps_sel_final)*r + eps_sel_final
                else: #no longer decay 
                    eps_selection = eps_sel_final 
                
               # r = max((int(total_iterations)-int(iteration_sel))/int(total_iterations), 0)
               # eps_selection = (eps_sel_initial - eps_sel_final)*r + eps_sel_final

            if prob <= eps_selection: 
                #make a random move with probability eps 
                reason_selection = 'random, selection eps'
                opponent_idx = self.player.RNs['selection_eps_move'] #local index
                self.player.current_selection_idx = opponent_idx #update player's latest selection with custom index 
                selection_idx = self.player.index_mapping_localtoglobal[opponent_idx] #global index
                #opponent = game.population[selection_idx]
                return int(selection_idx), eps_selection, reason_selection

            else:             
                #move optimally based on current Q-value estimates, if they are not empty 

                if self.alg_type == 'DQN': #if using DQN:
                    #check if non-0 Q-values are available for this state from the Q-network
                    #NOTE check DQN syntax here 
                    #sel_state = torch.tensor(self.selection_state_index, dtype=torch.int).long()
                    #sel_state_1hot = F.one_hot(sel_state, num_classes=selection_state_size).float()#.to(self.device)  

                    #TO DO update action here to fit IPD env
                    #self.player.qnetwork_local.eval()
                    #TO DO check what the above line does 
                    #with torch.no_grad():
                    #    action_values = self.player.qnetwork_local(state_1hot)
                    #self.player.qnetwork_local.train()        # Epsilon-greedy action selection
                    #TO DO check what the above line does 

                    #TO DO add check if both 0, then choose randomly

                    #if action_values.cpu().data.numpy() == [0, 0]:
                    #if [0, 0] in self.player.Qvalues_expected[self.state_index]:
                    if episodes: 
                        if (self.player.t_step_selection == 0) or (self.player.t_step_selection < 1):
                            reason_selection = 'random, non-full selection memory'
                            opponent_idx = self.player.RNs['selection_empty_Q'] #make a random selection
                            self.player.current_selection_idx = opponent_idx #update player's latest selection with custom index 
                            selection_idx = self.player.index_mapping_localtoglobal[opponent_idx] #global index
                            #opponent = game.population[selection_idx]
                            return int(selection_idx), eps_selection, reason_selection

                        else: 
                            reason_selection = 'greedy selection'
                            opponent_idx = self.player.select_DQN(self.current_selection_state_chain) #TO DO make sure this returns local index for this selecting player 
                            self.player.current_selection_idx = opponent_idx #update player's latest selection with custom index 
                            #TO DO check that current_selection_idx is being updated correctly 
                            selection_idx = self.player.index_mapping_localtoglobal[opponent_idx] #global index
                            #opponent = game.population[selection_idx]
                            return int(selection_idx), eps_selection, reason_selection
                        
                    elif episodes == False: 
                        if len(self.player.selection_memory) < 1:
                            reason_selection = 'random, non-full selection memory'
                            opponent_idx = self.player.RNs['selection_empty_Q'] #make a random selection
                            self.player.current_selection_idx = opponent_idx #update player's latest selection with custom index 
                            selection_idx = self.player.index_mapping_localtoglobal[opponent_idx] #global index
                            #opponent = game.population[selection_idx]
                            return int(selection_idx), eps_selection, reason_selection
                
                        else: 
                            reason_selection = 'greedy selection'
                            opponent_idx = self.player.select_DQN(self.current_selection_state_chain) #TO DO make sure this returns local index for this selecting player 
                            self.player.current_selection_idx = opponent_idx #update player's latest selection with custom index 
                            #TO DO check that current_selection_idx is being updated correctly 
                            selection_idx = self.player.index_mapping_localtoglobal[opponent_idx] #global index
                            #opponent = game.population[selection_idx]
                            return int(selection_idx), eps_selection, reason_selection



