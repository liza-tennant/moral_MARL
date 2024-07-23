import numpy as np 

from config import episodes, dilemma_state_uses_identity, record_eps, record_reason, record_dilemma_nextstate, state_pairs
#import torch.nn.functional as F
#import torch 

class Game_for_population_learning:
    '''This is the environment used by the two agents during learning - the game that they play, e.g. Iterated Prisoners Dilemma. 
    It takes in states, allows each player to make a selection ('partner_selection' function) or to take an action (in the 'step' function) and returns payoffs & next state'''
    
    def __init__(self, population, payoffmat):
        self.population = population #this is kept in a fixed order
        self.payoffmat = payoffmat
        self.history = list() #TO DO make this a circular array / queue instead 

        self.env_state = None #this will record the latest moves of all players in thepopulation --> frm this, each player will create their current_selection_state (by removing themselves) 
        self.selection_pairs = []

    def game_reward(self, m1, m2): #this is the EXTRINSIC REWARD from the game environment
        pay1, pay2 = self.payoffmat[m1][m2][0], self.payoffmat[m1][m2][1]
        # return a mapping of each player index to its payoff
        return [pay1, pay2]
    
    if False: 
        def step_mixed(self, state_player1, state_player2, iteration, global_history, num_iter, RNG):
            '''function for the two learning agents to interact with the environment simultaneously (i.e. play the game).
            For each player, it takes a state as input and:
            - generates an action (based on exploration_policy),
            - computes reward (game (extrinsic), intrinsic (moral - only for player1), total, collective)
            - computes a next state.
            
            The function returns the two actions, two s', game, total and global rewards.
            Apart from returning these variables, it also updates the global_history object (defined outside this class) with more detailed results.'''

            #generate 2*4 random numbers, then use whichever is needed. Generate all of them to make sure we go through the RN list consistently 
            #only using some of the RNs generated, since one of the players is static
            self.player1.RNs = {'empty_Q': RNG.player1_streams[0].uniform(0,1), 'eps_prob': RNG.player1_streams[0].uniform(0,1), 'eps_move': RNG.player1_streams[2].uniform(0,1), 'random_state': None}
            #'DQN_weights': RNG.player1_streams[5].integers(low=1,high=100, size=1), 'memory_sampling': RNG.player1_streams[6].integers(low=1,high=100, size=1) 
            #'static_random': RNG.player1_streams[3].uniform(0,1), 
            self.player2.RNs = {'static_random': RNG.player1_streams[3].uniform(0,1), 'random_state': None}
            #player1_RN_1 = RNG.player1_streams[0].uniform(0,1) #random move when Q-table is empty
            #player1_RN_2 = RNG.player1_streams[1].uniform(0,1) #probability to compare against eps
            #player1_RN_3 = RNG.player1_streams[2].uniform(0,1) #random move due to eps
            #player2_RN_4 = RNG.player2_streams[3].uniform(0,1) #move for a static agent with strategy==’random’
            #player1_RN_5 = RNG.player1_streams[4].uniform(0,1) #seed for Q-network weights initialisation 
            #player1_RN_6 = RNG.player1_streams[5].uniform(0,1) #seed for sampling experience 

            state_index_player1 = self.state_index_converter[state_player1]
            
            action_player1, eps_player1, reason_player1 = self.player1.make_exploratory_move(iteration=iteration, num_iter=num_iter) #state_index = state_index_player1, 
            action_player2 = self.player2.make_fixed_move(state=state_player2)

            #save the key information as next_state for each agent #NOTE we record state with opponent's move first, then own movement
            next_state_player1 = (action_player2, action_player1)
            next_state_player2 = (action_player1, action_player2) #note this does not really get used as player2 is static 

            #calculate reward - extrinsic (from the game scores), intrinsic (based on moral rule of the player), collective
            reward_game_player1 = self.game_reward(m1=action_player1, m2=action_player2)[0]
            reward_game_player2 = self.game_reward(m1=action_player2, m2=action_player1)[0]

            reward_intrinsic_player1 = self.player1.intrinsic_reward(self, m1=action_player1, m2=action_player2, state=state_player1)

            reward_collective = self.collective_reward(m1=action_player1, m2=action_player2)
            reward_gini = self.reward_gini(m1=action_player1, m2=action_player2)
            reward_min = self.reward_min(m1=action_player1, m2=action_player2) 

            #append values to the history dataframe - used for plotting later 
            global_history.loc[iteration, ['state_player1', 'action_player1', 'state_player2', 'action_player2']] = [
                state_player1, action_player1, state_player2, action_player2]

            global_history.loc[iteration, ['reward_game_player1', 'next_state_player1', 'reward_game_player2', 'next_state_player2']] = [
                reward_game_player1, next_state_player1, reward_game_player2, next_state_player2]

            global_history.loc[iteration, ['reward_intrinsic_player1', 'reward_collective', 'reward_gini', 'reward_min']] = [
                reward_intrinsic_player1, reward_collective, reward_gini, reward_min]

            global_history.loc[iteration, ['eps_player1', 'reason_player1']] = [
                eps_player1, reason_player1]
                #NOTE if we do not use str() here, this throws and error about creating np arrray from ragged nested sequences - ignore for now

            #determine which reward the player will be learning from based on their moral type:
            reward_learning_player1 = 0 #initialise

            if self.player1.moral_type == None: 
                reward_learning_player1 = reward_game_player1
            else: #if moral_type=='Utilitarian' or 'Deontological' or 'VirtueEthics': 
                reward_learning_player1 = reward_intrinsic_player1
            #if need to debug - print action, next_state and rewards here        
            #print('One step done')

            global_history.loc[iteration, ['reward_learning_player1']] = [reward_learning_player1] 

            
            return int(action_player1), next_state_player1, next_state_player2, reward_learning_player1

    def step_simplified(self, player1, player2, p1_idx, p2_idx, iteration_dil, global_history, num_iter, RNG): #p2_idx_by_p1, p1_idx_by_p2, 
        '''function for the two learning agents to interact with the environment simultaneously (i.e. play the game).
        For each player, it takes a state as input and:
        - generates an action (based on exploration_policy),
        - computes reward (game (extrinsic), intrinsic (moral), total, collective)
        - computes a next state.
        
        The function returns the two actions, two s', game, total and global rewards.
        Apart from returning these variables, it also updates the global_history object (defined outside this class) with more detailed results.
        
        Note, for the DQN player, this will play randomly until the buffer is filled, then play according to the eps-greedy policy '''

        #generate 2*4 random numbers, then use whichever is needed. Generate all of them to make sure we go through the RN list consistently
        player1.RNs.update({'empty_Q': RNG.player_streams[f'player{p1_idx}'][0].uniform(0,1), 'eps_prob': RNG.player_streams[f'player{p1_idx}'][1].uniform(0,1), 'eps_move': RNG.player_streams[f'player{p1_idx}'][2].uniform(0,1), 'random_state': None})
        if player2.strategy not in ['AlwaysCooperate', 'AlwaysDefect', 'TitforTat', 'Random', 'Alternating', 'GilbertElliot_typea', 'GilbertElliot_typeb', 'GilbertElliot_typec', 'GilbertElliot_typed', 'GilbertElliot_typee']:
            player2.RNs.update({'empty_Q': RNG.player_streams[f'player{p2_idx}'][0].uniform(0,1), 'eps_prob': RNG.player_streams[f'player{p2_idx}'][1].uniform(0,1), 'eps_move': RNG.player_streams[f'player{p2_idx}'][2].uniform(0,1), 'random_state': None})        
       

        if player1.strategy in ['AlwaysCooperate', 'AlwaysDefect', 'TitforTat', 'Random', 'Alternating', 'GilbertElliot_typea', 'GilbertElliot_typeb', 'GilbertElliot_typec', 'GilbertElliot_typed', 'GilbertElliot_typee']:
            action_player1 = player1.make_fixed_move(state=player1.current_state)
            eps_player1 = None 
            reason_player1 = None
        elif 'Q-Learning' in player1.strategy: 
            action_player1, eps_player1, reason_player1 = player1.make_exploratory_move(iteration_dil=iteration_dil, num_iter=num_iter) #opponent_index = p2_idx_by_p1,  #use player's .current_state attribute 
        
        if player2.strategy in ['AlwaysCooperate', 'AlwaysDefect', 'TitforTat', 'Random', 'Alternating', 'GilbertElliot_typea', 'GilbertElliot_typeb', 'GilbertElliot_typec', 'GilbertElliot_typed', 'GilbertElliot_typee']:
            action_player2 = player2.make_fixed_move(state=player2.current_state)
            eps_player2 = None 
            reason_player2 = None
        elif 'Q-Learning' in player2.strategy: 
            action_player2, eps_player2, reason_player2 = player2.make_exploratory_move(iteration_dil=iteration_dil, num_iter=num_iter) #opponent_index = p1_idx_by_p2, 

        #save the key information as next_state for each agent - of shape (action_opponent, action_own)
        if state_pairs:
            next_state_player1 = [action_player2, action_player1]
            next_state_player2 = [action_player1, action_player2]  
            if dilemma_state_uses_identity:
                next_state_player1.append(player1.index_mapping_globaltolocal[p2_idx]) 
                next_state_player2.append(player2.index_mapping_globaltolocal[p1_idx]) #PLACEHOLDER #p1_idx_for_p2)
                
                next_state_player1 = tuple(next_state_player1)
                next_state_player2 = tuple(next_state_player2)
        else: #if state not using pairs but just the opponent's move 
            if dilemma_state_uses_identity:
                next_state_player1 = (action_player2, player1.index_mapping_globaltolocal[p2_idx])
                next_state_player2 = (action_player1, player2.index_mapping_globaltolocal[p1_idx]) 
            else: #if not using identity 
                next_state_player1 = action_player2
                next_state_player2 = action_player1  


        #calculate reward - extrinsic (from the game scores), intrinsic (based on moral rule of the player), collective
        reward_game_player1 = self.game_reward(m1=action_player1, m2=action_player2)[0]
        reward_game_player2 = self.game_reward(m1=action_player2, m2=action_player1)[0]

#!!!! NOTE TO DO fix intrinsic reward calculation if dilemma_state_uses_identity==True !!! 

        reward_intrinsic_player1 = player1.intrinsic_reward(self, m1=action_player1, m2=action_player2, state=player1.current_state)
        reward_intrinsic_player2 = player2.intrinsic_reward(self, m1=action_player2, m2=action_player1, state=player1.current_state)

        ########################################################################################
        #### append values to the global_history dataframe - used for analysis & plotting later 

        #titled for each player  
        if player1.strategy == 'Q-Learning eps-greedy':
            if player1.moral_type == None: 
                title_p1 = 'Selfish' #['Selfish']
            else: 
                title_p1 = player1.moral_type #[player1.moral_type]
        elif player1.strategy in ['AlwaysCooperate', 'AlwaysDefect', 'TitforTat', 'Random', 'Alternating', 'GilbertElliot_typea', 'GilbertElliot_typeb', 'GilbertElliot_typec', 'GilbertElliot_typed', 'GilbertElliot_typee']:
            title_p1 = player1.moral_type #[player1.strategy]
        

        if player2.strategy == 'Q-Learning eps-greedy':
            if player2.moral_type == None: 
                title_p2 = 'Selfish'
            else: 
                title_p2 = player2.moral_type
        elif player2.strategy in ['AlwaysCooperate', 'AlwaysDefect', 'TitforTat', 'Random', 'Alternating', 'GilbertElliot_typea', 'GilbertElliot_typeb', 'GilbertElliot_typec', 'GilbertElliot_typed', 'GilbertElliot_typee']:
            title_p2 = player2.strategy
  
        if record_dilemma_nextstate == True: 
            global_history.loc[iteration_dil, ['next_state_player1', 'next_state_player2']] = [
            str(tuple(next_state_player1)), str(tuple(next_state_player2))]
        #if record_dilemma_nextstate == 'TEST': 
        #    global_history.loc[iteration, ['next_state_player1']] = str(tuple([12, 345]))
        #    global_history.loc[iteration, ['next_state_player2']] = str(tuple([54, 321]))

        if record_eps == True:
            global_history.loc[iteration_dil, ['eps_player1', 'eps_player2']] = [
            eps_player1, eps_player2]

        if record_reason == True:
            global_history.loc[iteration_dil, ['reason_player1', 'reason_player2']] = [
            reason_player1, reason_player2]
        

        #determine which reward the player will be learning from based on their moral type:
        reward_learning_player1 = 0 #initialise
        reward_learning_player2 = 0
        if player1.moral_type == None: 
            reward_learning_player1 = reward_game_player1
        else: #if moral_type=='Utilitarian' or 'Deontological' or 'VirtueEthics': 
            reward_learning_player1 = reward_intrinsic_player1
        if player2.moral_type == None: 
            reward_learning_player2 = reward_game_player2
        else: #if moral_type=='Utilitarian' or 'Deontological' or 'VirtueEthics': 
            reward_learning_player2 = reward_intrinsic_player2


        global_history.loc[iteration_dil, ['title_p1', 'title_p2', 'idx_p1', 'idx_p2',
                                       'state_player1', 'action_player1', 'state_player2', 'action_player2', 
                                       'reward_game_player1', 'reward_game_player2', 'reward_intrinsic_player1', 'reward_intrinsic_player2', 
                                       'reward_learning_player1', 'reward_learning_player2']
                                       ] = [title_p1, title_p2, p1_idx, p2_idx,
                                            str(player1.current_state), action_player1, str(player2.current_state), action_player2, 
                                            reward_game_player1, reward_game_player2, reward_intrinsic_player1, reward_intrinsic_player2,
                                            reward_learning_player1, reward_learning_player2] 
        
        #return two actions and next_states for both players (based on this exact interaction), and the rewards they will learn from
        return int(action_player1), int(action_player2), next_state_player1, next_state_player2, reward_learning_player1, reward_learning_player2
    

    def random_matching(self, leading_player_idx, RNG):
        leading_player = self.population[leading_player_idx]
        #randomly match partner to leading_player_idx
        self.RNs = {'sample_opponent': RNG.game_streams[1].choice(leading_player.possible_opponent_indices, 1).item()} #sample a random number to represent next leading_player index 
        opponent_idx = self.RNs['sample_opponent'] #local index
        leading_player.current_selection_idx = opponent_idx

        selection_idx = leading_player.index_mapping_localtoglobal[opponent_idx]
        #opponent = self.population[selection_idx] #global index 

        reason_selection = 'random matching'
        #record data
        if record_reason==True:
            leading_player.reason_selection = reason_selection

        return selection_idx #, opponent, 
    
    def partner_selection(self, leading_player_idx, iteration_sel, num_iter, global_history, RNG, forced_choice=False): #manually seet forced_choice=True here if testing with forced choice 
        #NOTE: opponent_idx = local index for leading_player specifically; selection_idx = global index for entire population
        
        leading_player = self.population[leading_player_idx]

        #set selection RNs for this leading player 
        leading_player.RNs.update({'selection_empty_Q': RNG.player_streams[f'player{leading_player_idx}'][7].choice(leading_player.possible_opponent_indices, 1).item(), 
                                   'selection_eps_prob': RNG.player_streams[f'player{leading_player_idx}'][8].uniform(0,1), 
                                   'selection_eps_move': RNG.player_streams[f'player{leading_player_idx}'][9].choice(leading_player.possible_opponent_indices, 1).item()})

        #leading_player chooses an opponent - if leading_player is a Q-Learning player 
        if 'Q-Learning' in leading_player.strategy:
            if forced_choice: #use for testing 
                selection_idx, reason_selection = leading_player.make_fixed_selection() #note this is the global idx 
                #update player's attribute 
                opponent = self.population[selection_idx]
                opponent_idx = leading_player.index_mapping_globaltolocal[selection_idx] #local idx
                leading_player.current_selection_idx = opponent_idx

                #record data in global_history 
                if record_reason == True:
                    global_history.loc[iteration_sel, ['reason_selection_player1']] = [
                        reason_selection]
        
                return leading_player, opponent, selection_idx
            elif episodes == False: #if running actual experiment with selection 
                selection_idx, eps_selection, reason_selection = leading_player.make_exploratory_selection(iteration_sel, num_iter) #use player's current_selection_state_index
                opponent = self.population[selection_idx] # using global index
                opponent_idx = leading_player.index_mapping_globaltolocal[selection_idx] #local idx
                leading_player.current_selection_idx = opponent_idx
                
                #record data
                if record_eps == True:
                    global_history.loc[iteration_sel, ['eps_selection_player1']] = [
                        eps_selection]
                if record_reason == True:
                    global_history.loc[iteration_sel, ['reason_selection_player1']] = [
                        reason_selection]
                    
                return leading_player, opponent, selection_idx
            else: #if using espides 
                selection_idx, eps_selection, reason_selection = leading_player.make_exploratory_selection(iteration_sel, num_iter) #use player's current_selection_state_index
                opponent_idx = leading_player.index_mapping_globaltolocal[selection_idx] #local idx
                leading_player.current_selection_idx = opponent_idx

                #record data
                if record_eps == True:
                    leading_player.eps_selection = eps_selection
                    #global_history.loc[iteration_sel, ['eps_selection_player1']] = [
                    #    eps_selection]
                if record_reason == True:
                    leading_player.reason_selection = reason_selection
                    #global_history.loc[iteration_sel, ['reason_selection_player1']] = [
                    #    reason_selection]
        
                return selection_idx#, reason_selection

        elif leading_player.strategy in ['AlwaysCooperate', 'AlwaysDefect', 'TitforTat', 'Random', 
                                         'Alternating', 'GilbertElliot_typea', 'GilbertElliot_typeb', 'GilbertElliot_typec', 'GilbertElliot_typed', 'GilbertElliot_typee']: #random selection for static (non-learning) players 
            
            #randomly match partner to p1_idx
            self.RNs = {'sample_opponent': RNG.game_streams[1].choice(leading_player.possible_opponent_indices, 1).item()} #sample a random number to represent next leading_player index 
            opponent_idx = self.RNs['sample_opponent'] #local index
            selection_idx = leading_player.index_mapping_localtoglobal[opponent_idx]
            #opponent = self.population[selection_idx] #global index 

            reason_selection = 'random matching'

            #record data
            if record_reason==True:
                leading_player.reason_selection = reason_selection
                #global_history.loc[iteration_sel, ['reason_selection_player1']] = [
                #    reason_selection]
    
            return selection_idx#, reason_selection

        