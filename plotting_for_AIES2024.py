#plotting for AIES 2024 paper 

#plotting rewards (mean and cumulative mean with SD); actions (C or D) and actions types (in response to state) 
import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib.pyplot import figure
import dataframe_image as dfi
import os 
import glob 
from collections import Counter 
import pickle 
#import pickle5 as pickle #for earlier versions of python
from ast import literal_eval as make_tuple
from ordered_set import OrderedSet

sns.set(font_scale=2)

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

from statistics import mean 
import networkx as nx
from math import isnan
from collections import Counter

    
####################################
#### globally useful variables ####
####################################


title_mapping = {'AC':'AlwaysCooperate', 'AD':'AlwaysDefect', 'TFT':'TitForTat', 'Random':'random', 
                 'CD':'Alternating', 'GEa':'GilbertElliot_typea', 'GEb':'GilbertElliot_typeb', 'GEc':'GilbertElliot_typec', 'GEd':'GilbertElliot_typed', 'GEe':'GilbertElliot_typee',
                 'QLS':'Selfish', 'QLUT':'Utilitarian', 'QLDE':'Deontological', 'QLVE_e':'VirtueEthics_equality', 'QLVE_k':'VirtueEthics_kindness', 'QLVM':'VirtueEthics_mixed', 'QLAL':'Altruist',
                 'QLaUT':'anti-Utilitarian', 'QLmDE':'malicious_Deontological', 'QLVie':'VirtueEthics_inequality', 'QLVagg':'VirtueEthics_aggressiveness', 'QLaAL':'anti-Altruist'}
              
color_mapping_longtitle = {'Selfish':'orange', 'AlwaysCooperate':'green', 'AlwaysDefect':'orchid', 'TitfroTat':'orangered', 'Random':'grey', 
                           'Alternating':'deeppink', 'GilbertElliot_typea':'limegreen', 'GilbertElliot_typeb':'forestgreen', 'GilbertElliot_typec':'springgreen', 'GilbertElliot_typed':'magenta', 'GilbertElliot_typee':'lightgreen', 
			               'Utilitarian':'cornflowerblue', 'Deontological':'aqua', 'VirtueEthics_equality':'mediumorchid', 'VirtueEthics_kindness':'lime',
                            'anti-Utilitarian':'blue', 'malicious_Deontological':'lightseagreen', 'VirtueEthics_inequality':'purple', 'VirtueEthics_aggressiveness':'forestgreen'}
                 
movingaverage = True 
movingN = 200 #overwrite this as needed 
n_runs_global = 20 



population_size_approx = 16 

title_mapping_forpaper_long = {'AlwaysCooperate':'AC', 'AlwaysDefect':'AD', 'TitForTat':'TFT', 'random':'Random', 
                 'Alternating':'CD', 'GilbertElliot_typea':'GEa', 'GilbertElliot_typeb':'GEb', 'GilbertElliot_typec':'GEc', 'GilbertElliot_typed':'GEd', 'GilbertElliot_typee':'GEe',
                 'Selfish':'S', 'Utilitarian':'Ut', 'Deontological':'De', 'VirtueEthics_equality':'V-Eq', 'VirtueEthics_kindness':'V-Ki',
                 'anti-Utilitarian':'aUt', 'malicious_Deontological':'mDe', 'VirtueEthics_inequality':'V-In', 'VirtueEthics_aggressiveness':'V-Ag'}


title_mapping_forpaper_short = {'AC':'AC', 'AD':'AD', 'TFT':'TFT', 'Random':'Random', 
                 'CD':'Al', 'GEa':'GEa', 'GEb':'GEb', 'GEc':'GEc', 'GEd':'GEd', 'GEe':'GEe',
                 'QLS':'S', 'QLUT':'Ut', 'QLDE':'De', 'QLVE_e':'V-Eq', 'QLVE_k':'V-Ki',
                 'QLaUT':'aUt', 'QLmDE':'mDe', 'QLVie':'V-In', 'QLVagg':'V-Ag'}

title_mapping_forpaper_shorttolong  = {v: k for k, v in title_mapping_forpaper_long.items()}


title_mapping_forpaper_longtoclean = {'AlwaysCooperate':'AC', 'AlwaysDefect':'AD', 'TitForTat':'TFT', 'random':'Random', 
                 'Alternating':'CD', 'GilbertElliot_typea':'GEa', 'GilbertElliot_typeb':'GEb', 'GilbertElliot_typec':'GEc', 'GilbertElliot_typed':'GEd', 'GilbertElliot_typee':'GEe',
                 'Selfish':'Selfish', 'Utilitarian':'Utilitarian', 'Deontological':'Deontological', 'VirtueEthics_equality':'Virtue-Equality', 'VirtueEthics_kindness':'Virtue-Kindness',
                 'anti-Utilitarian':'anti-Utilitarian', 'malicious_Deontological':'malicious-Deontological', 'VirtueEthics_inequality':'Virtue-Inequality', 'VirtueEthics_aggressiveness':'Virtue-Aggressive'}


sns.set(font_scale=2)


#######################
#### population_plots ####
#######################


def get_titles_for_population(destination_folder, print_titles=False):

    titles = [] 
    temp_titles = destination_folder.split('__')[0]

    for item in temp_titles.split('_'): 
        items = item.split('x')
        if print_titles: 
            print(items)
        for n_repeats in range(int(items[0])):
            if items[1] in ['AC', 'AD', 'TFT', 'Random', 'CD', 'GEa', 'GEb', 'GEc', 'GEd', 'GEe']:
                titles.append(str(items[1]))
            else: 
                titles.append(str('QL'+items[1]))
    
    #TO DO sort out VEe, VEk --> VE_E, VE_k !! 
    titles_new = []
    for title in titles: 
        if 'VE' in title: 
            titles_new.append(title[0:4]+'_'+title[4:])
        else:
            titles_new.append(title)

    return titles_new

def reformat_reward_for_population(destination_folder, n_runs, population_size, long_titles, episodes=True):
    '''create dfs for each moral player type, cregardless of whether they acted as player1 or player2'''
    colnames = ['run'+str(i+1) for i in range(n_runs)]

    if episodes:
        num_iter_new = num_iter * population_size
    else: 
        num_iter_new = num_iter

    for idx in range(population_size): #initialise the results dataframes for each player idx
        globals()[f'R_game_idx{idx}'] = pd.DataFrame(columns=colnames, index=range(num_iter_new))
        globals()[f'Rcumul_game_idx{idx}'] = pd.DataFrame(columns=colnames, index=range(num_iter_new))

        globals()[f'R_intr_idx{idx}'] = pd.DataFrame(columns=colnames, index=range(num_iter_new))
        globals()[f'Rcumul_intr_idx{idx}'] = pd.DataFrame(columns=colnames, index=range(num_iter_new))
    
    if False: 
        R_collective = pd.DataFrame(columns=colnames)
        Rcumul_collective = pd.DataFrame(columns=colnames)
        R_gini = pd.DataFrame(columns=colnames)
        Rcumul_gini = pd.DataFrame(columns=colnames)
        R_min = pd.DataFrame(columns=colnames)
        Rcumul_min = pd.DataFrame(columns=colnames)
    

    for run in os.listdir(str(destination_folder+'/history')):
        run_name = str(run).strip('.csv')
        run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)), index_col=0)[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'reward_game_player1', 'reward_game_player2', 'reward_intrinsic_player1', 'reward_intrinsic_player2']]
        run_df['title_p1'].fillna('Selfish', inplace=True)
        run_df['title_p2'].fillna('Selfish', inplace=True)

        #create for player1 & player2 for each index 
        for idx in range(population_size): 
            ##################################
            #### game reward ####
            ##################################
            globals()[f'R_game_player1_idx{idx}'] = run_df[run_df['idx_p1']==idx]['reward_game_player1'].astype(int)
            globals()[f'R_game_player2_idx{idx}'] = run_df[run_df['idx_p2']==idx]['reward_game_player2'].astype(int)
            globals()[f'R_game_run_idx{idx}'] = globals()[f'R_game_player1_idx{idx}'].append(globals()[f'R_game_player2_idx{idx}'])
            globals()[f'R_game_run_idx{idx}'] = globals()[f'R_game_run_idx{idx}'].sort_index() #sort index, conveer to int  
            globals()[f'R_game_idx{idx}'][run_name] = globals()[f'R_game_run_idx{idx}'] ##insert result for this particular run into the overall df
            globals()[f'Rcumul_game_idx{idx}'][run_name] = np.cumsum(globals()[f'R_game_run_idx{idx}'])

            ##################################
            #### intrinsic reward ####
            ##################################
            globals()[f'R_intr_player1_idx{idx}'] = run_df[run_df['idx_p1']==idx]['reward_intrinsic_player1']
            globals()[f'R_intr_player2_idx{idx}'] = run_df[run_df['idx_p2']==idx]['reward_intrinsic_player2']
            globals()[f'R_intr_run_idx{idx}'] = globals()[f'R_intr_player1_idx{idx}'].append(globals()[f'R_intr_player2_idx{idx}'])
            globals()[f'R_intr_run_idx{idx}'] = globals()[f'R_intr_run_idx{idx}'].sort_index() #sort index, conveer to int  
            globals()[f'R_intr_idx{idx}'][run_name] = globals()[f'R_intr_run_idx{idx}'] ##insert result for this particular run into the overall df
            globals()[f'Rcumul_intr_idx{idx}'][run_name] = np.cumsum(globals()[f'R_intr_run_idx{idx}'])

        if False: 
            ##################################
            #### collective, gini & min reward ####
            ##################################
            R_collective_run = run_df['reward_collective'] #not filtering by player idx any more
            R_collective[run_name] = R_collective_run.sort_index()
            Rcumul_collective[run_name] = np.cumsum(R_collective_run)

            R_gini_run = run_df['reward_gini'] #not filtering by player idx any more
            R_gini[run_name] = R_gini_run.sort_index()
            Rcumul_gini[run_name] = np.cumsum(R_gini_run)

            R_min_run = run_df['reward_min'] #not filtering by player idx any more
            R_min[run_name] = R_min_run.sort_index()
            Rcumul_min[run_name] = np.cumsum(R_min_run)

    #save the reformatted data
    if not os.path.isdir(f'{destination_folder}/player_rewards'):
            os.makedirs(f'{destination_folder}/player_rewards')

    print('save results...')
    for idx in range(population_size): #save results
        idx_title = long_titles[idx]

        globals()[f'R_game_idx{idx}'].to_csv(f'{destination_folder}/player_rewards/R_game_{idx_title}_{idx}.csv')
        globals()[f'Rcumul_game_idx{idx}'].to_csv(f'{destination_folder}/player_rewards/Rcumul_game_{idx_title}_{idx}.csv')

        globals()[f'R_intr_idx{idx}'].to_csv(f'{destination_folder}/player_rewards/R_intr_{idx_title}_{idx}.csv')
        globals()[f'Rcumul_intr_idx{idx}'].to_csv(f'{destination_folder}/player_rewards/Rcumul_intr_{idx_title}_{idx}.csv')

    if False: 
        if not os.path.isdir(f'{destination_folder}/player_outcomes'):
            os.makedirs(f'{destination_folder}/player_outcomes')

        R_collective.to_csv(f'{destination_folder}/player_outcomes/R_collective.csv')
        Rcumul_collective.to_csv(f'{destination_folder}/player_outcomes/Rcumul_collective.csv')
        R_gini.to_csv(f'{destination_folder}/player_outcomes/R_gini.csv')
        Rcumul_gini.to_csv(f'{destination_folder}/player_outcomes/Rcumul_gini.csv')
        R_min.to_csv(f'{destination_folder}/player_outcomes/R_min.csv')
        Rcumul_min.to_csv(f'{destination_folder}/player_outcomes/Rcumul_min.csv')




#######################
#### plots per episode ####
#######################




def episode_plot_cooperation_population_v2(destination_folder, titles, n_runs, num_iter, with_CI=True, reduced=False):
    '''explore the actions taken by players at every step out of num_iter - what percentage are cooperating? 
    if reduced==True, plot every 5th value'''
    
    if len(titles)<=6:
        titles_short = [title.replace('QL','') for title in titles] 
        temp = ''
        for t in titles_short: 
            temp += (t+',')
        population_list = temp.strip(',')
    else:  
        population_list = destination_folder.replace('results/', '')

    #combine each player's actions into one df: 
    result_population = pd.DataFrame(index=range(num_iter)) 

    for run_idx in range(n_runs):
        run_idx += 1
        run_df = pd.read_csv(destination_folder+f'/history/run{run_idx}.csv', index_col=0)[['episode', 'action_player1', 'action_player2']]

        #create boolean variabnle for cooperative selections amde 
        run_df['action_player1_C'] = run_df.apply(lambda row: row['action_player1']==0, axis=1)
        run_df['action_player2_C'] = run_df.apply(lambda row: row['action_player2']==0, axis=1)

        #group by episode 
        run_grouped = run_df.groupby('episode').agg({'action_player1_C': 'sum', 'action_player2_C': 'sum', 'action_player1':'count', 'action_player2':'count'})
        run_grouped['action_C'] = run_grouped['action_player1_C'] + run_grouped['action_player2_C']
        run_grouped['actions_total'] = run_grouped['action_player1'] + run_grouped['action_player2']
        #TO DO check the denominator is 40 
        run_grouped['%_C'] = run_grouped['action_C'] * 100 / run_grouped['actions_total']

        result_population[f'run{run_idx}'] = run_grouped['%_C']


    if with_CI:
        means = result_population.mean(axis=1)
        sds = result_population.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)
        result_population['%_cooperate_mean'] = means
        result_population['%_cooperate_ci'] = ci
    else: 
        result_population['%_cooperate_mean'] = result_population.mean(axis=1)

    if reduced: 
        #keep every 5th row 
        result_population = result_population[result_population.index % 5 == 0]
        reduced_option = ' (every 5)'
    else: 
        reduced_option = None

    #plot results 
    plt.figure(dpi=80, figsize=(15, 6)) 
    if with_CI:
                plt.plot(result_population.index[:], result_population['%_cooperate_mean'], label=f'{population_list}', color='darkgreen', linewidth=0.05)
                plt.fill_between(result_population.index[:], result_population['%_cooperate_mean']-result_population['%_cooperate_ci'], result_population['%_cooperate_mean']+result_population['%_cooperate_ci'], facecolor='lightgreen', linewidth=0.04, alpha=1)
    else: 
        plt.plot(result_population.index[:], result_population['%_cooperate_mean'], label=f'{population_list}', color='darkgreen', linewidth=0.05)

    if reduced_option: 
        plt.title(f'Cooperation in population \n {population_list} '+ reduced_option) 
    else: 
        plt.title(f'Cooperation in population \n {population_list} ') 
    plt.gca().set_ylim([0, 100])
    plt.ylabel(r'% cooperating per episode'+ '  \n (mean over '+str(n_runs)+r' runs +- CI)')
    plt.xlabel('Episode')


    if not os.path.isdir(f'{destination_folder}/plots'):
        os.makedirs(f'{destination_folder}/plots')

    plt.savefig(f'{destination_folder}/plots/episode_cooperation_{population_list}.png', bbox_inches='tight')

def episode_plot_cooperation_perrun(destination_folder, titles, n_runs, num_iter):
    
    if len(titles)<=6:
        titles_short = [title.replace('QL','') for title in titles] 
        temp = ''
        for t in titles_short: 
            temp += (t+',')
        population_list = temp.strip(',')
    else:  
        population_list = destination_folder.replace('results/', '')

    #combine each player's actions into one df: 
    result_population = pd.DataFrame(index=range(num_iter)) 

    for run_idx in range(n_runs):
        run_idx += 1
        run_df = pd.read_csv(destination_folder+f'/history/run{run_idx}.csv', index_col=0)[['episode', 'action_player1', 'action_player2']]

        #create boolean variabnle for cooperative selections amde 
        run_df['action_player1_C'] = run_df.apply(lambda row: row['action_player1']==0, axis=1)
        run_df['action_player2_C'] = run_df.apply(lambda row: row['action_player2']==0, axis=1)

        #group by episode 
        run_grouped = run_df.groupby('episode').agg({'action_player1_C': 'sum', 'action_player2_C': 'sum', 'action_player1':'count', 'action_player2':'count'})
        run_grouped['action_C'] = run_grouped['action_player1_C'] + run_grouped['action_player2_C']
        run_grouped['actions_total'] = run_grouped['action_player1'] + run_grouped['action_player2']
        #TO DO check the denominator is 40 
        run_grouped['%_C'] = run_grouped['action_C'] * 100 / run_grouped['actions_total']

        result_population[f'run{run_idx}'] = run_grouped['%_C']

    #plot results 
    for run in result_population.columns:

        plt.figure(dpi=80, figsize=(15, 6)) 
        plt.plot(result_population.index[:], result_population[run], label=f'{population_list}', color='darkgreen', linewidth=0.05)
        plt.title(f'Cooperation in population \n {population_list} ') 
        plt.gca().set_ylim([0, 100])
        plt.ylabel(r'% cooperating per episode'+ '  \n (every run separate)')
        plt.xlabel('Episode')  



def episode_plot_action_pairs_population_new(destination_folder, titles, n_runs, option=None, combine_CDDC=False):
    '''if reduced, plot only every 10th value. 
       if combine_CDDC, ignore order of action pairs, and plot CD and DC (exploitation in one direction) together. This ignores whether the selector or the selected is the defector.'''
    if not os.path.isdir(str(destination_folder)+'/plots'):
        os.makedirs(str(destination_folder)+'/plots')

    if not os.path.isdir(str(destination_folder)+'/plots/outcome_plots'):
        os.makedirs(str(destination_folder)+'/plots/outcome_plots')

    if not os.path.isdir('results/outcome_plots/actions'):
            os.makedirs('results/outcome_plots/actions')


    if 'action_pairs.csv' not in os.listdir(str(destination_folder+'/')): 
    #if True: 
        colnames = ['run'+str(i+1) for i in range(n_runs)]
        #append 'episode' to the end of the list 
        #colnames.append('episode')
        actions_player1 = pd.DataFrame(columns=colnames, index=range(num_iter*population_size_approx))
        actions_player2 = pd.DataFrame(columns=colnames,  index=range(num_iter*population_size_approx))

        print('gathering actions from each run ...')
        for run in os.listdir(str(destination_folder+'/history')):
            if '.csv' in run:
                run_name = str(run).strip('.csv')
                run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
                actions_player1[run_name] = run_df['action_player1']
                actions_player1['episode'] = run_df['episode']
                actions_player2[run_name] = run_df['action_player2']
                #actions_player2['episode'] = run_df['episode']

        results = pd.DataFrame(columns=colnames)

        print('combining actions from each run into pairs ...')
        for colname in colnames: #loop over every run
            str_value = actions_player1[colname].astype(str) + ',' + actions_player2[colname].astype(str)
            str_value = str_value.str.replace('1', 'D')
            str_value = str_value.str.replace('0', 'C')
            str_value = str_value.str.replace(' ', '')
            if combine_CDDC:
                str_value = str_value.replace('C,D', 'C,D/D,C').replace('D,C', 'C,D/D,C') #when the order of actions does not matter, combine CD/DC into one color
            results[colname] = str_value

        #results['episode'] = actions_player1['episode']

        print('storing action_pairs to disc...')
        results.to_csv(str(destination_folder+'/action_pairs.csv'))
    else: 
        print('importing existing action_pairs file...')
        results = pd.read_csv(str(destination_folder+'/action_pairs.csv'), index_col=0)


    print('calculating value_counts for each action pair...')
    results_counts = results.apply(lambda x: x.value_counts(), axis=1)
    #results_counts = results.transpose().apply(pd.value_counts).transpose()#[1:] #plot after first episode, as in the first apisode they are reacting to default state=0
    results_counts['episode'] = pd.read_csv(str(destination_folder+'/history/run1.csv'), index_col=0)['episode'] #actions_player1['episode']
    results_counts_grouped = results_counts.groupby('episode').sum() #sum all the counts of each pair across the entire episode
    #y-axis = number of runs * population_size (i.e., action pairs in a single episode)

    #results_counts_grouped.dropna(axis=1, how='all', inplace=True)

    titles = [title.replace('QL','') for title in titles] 
    titles = [title.replace('Ethics','').replace('_','-').replace('VE-e','VE_e').replace('VE-k','VE_k') for title in titles] 
    temp = ''
    for t in titles: 
        temp += (t+',')
    temp = temp.strip(',')
    if len(titles)>6: 
        #temp = destination_folder.replace('results/', '').replace('__iter30000_runs20', '')
        most_common_numbers = Counter(titles).most_common(2)
        if most_common_numbers[0][1] == most_common_numbers[1][1]:
            print('more than one majority value')
            population_forpaper = f'uniform'
            popsize_forpaper = f', size {len(titles)}'
        else: 

            #population = destination_folder.replace('___iter30000_runs20', '').replace('results/', '')
            #long_titles = get_titles_for_population(destination_folder)
            majority_title_short = most_common_numbers[0][0] #max(set(titles), key = titles.count)
            majority_title = title_mapping['QL'+majority_title_short]
            population_forpaper = f'majority-{title_mapping_forpaper_long[majority_title]}'
            popsize_forpaper = ''

    #plt.figure(figsize=(20, 15), dpi=100)
    if num_iter > 20000:
        plt.figure(dpi=80, figsize=(10, 4))
    else:
        plt.figure(dpi=80, figsize=(5, 4))
    
    if combine_CDDC:
        color = {'C,C':'#28641E', 'C,D/D,C':'#FFFFFF', 'D,D':'#8E0B52'}
    else: 
        color = {'C,C':'#28641E', 'C,D':'#B0DC82', 'D,C':'#EEAED4', 'D,D':'#8E0B52'}

    print('plotting ...')
    sns.set(font_scale=3.5)
    #plot as an area plot 
    total_actionpairs = results_counts_grouped.sum(axis=1)[0]
    #plt.rcParams.update({'font.size':20})
    results_counts_grouped.plot.area(stacked=True, ylabel = '# action pairs \n (over 20 runs)', #rot=45,
        xlabel='Episode', #each episode includes N iterations, where N = population_size (eahc player makes a selection once)
        xticks=[0,15000,30000], yticks=[0,total_actionpairs/2,total_actionpairs],
        color=color, linewidth=0.01, alpha=0.9,
        #color={'C, C':'#28641E', 'C, D':'#B0DC82', 'D, C':'#EEAED4', 'D, D':'#8E0B52'}, linewidth=0.05, alpha=0.9,
        #color={'C, C':'royalblue', 'C, D':'lightblue', 'D, C':'yellow', 'D, D':'orange'},
        title = str('Action pairs: \n ' + population_forpaper + ' population'+  popsize_forpaper)
        ).legend(fontsize=18, loc='center') #Pairs of simultaneous actions over time: \n '+

    #plt.savefig(f'{destination_folder}/plots/action_pairs.png', bbox_inches='tight')
    if not option: #if plotting main results
        population = destination_folder.split('/')[1]
    else: #if plotting extra parameter search for beta in QLVM 
        population = destination_folder
        #pair += str(option)
    
    plt.savefig(f'results/outcome_plots/actions/episode_actions_pairs_popul{population_forpaper}.png', bbox_inches='tight')




def reformat_a_for_population(destination_folder, n_runs, population_size, num_iter, long_titles, episodes=True):
        '''create dfs for each moral player type, regardless of whether they acted as player1 or player2'''
        colnames = ['run'+str(i+1) for i in range(n_runs)]

        if episodes:
            num_iter_new = num_iter * population_size
        else: 
            num_iter_new = num_iter

        for idx in range(population_size): #initialise the results dataframes for each player idx
            globals()[f'results_idx{idx}'] = pd.DataFrame(columns=colnames, index=range(num_iter_new))
        
        path = str(destination_folder+'/history/'+'*.csv')
        
        for run in glob.glob(path): #os.listdir():
            print(f'proecssing run {run}')
            run_name = str(run).replace(str(destination_folder)+'/history/', '')
            if '.csv' in run_name:
                run_name = str(run_name).strip('.csv')
            #try: 
                run_df = pd.read_csv(run)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
                #run_df = pd.read_csv(str(destination_folder+'/history/'+str(run)))#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
                
                #run_df['title_p1'].fillna('Selfish', inplace=True)
                #run_df['title_p2'].fillna('Selfish', inplace=True)
                run_df = run_df[run_df['idx_p1']!=run_df['idx_p2']] #drop self-selections - FOR EARLIER VERSIONS
            #except: 
            #    break 

                #create for player1 & player2 for each index 
                for idx in range(population_size): 
                    globals()[f'results_player1_idx{idx}'] = run_df[run_df['idx_p1']==idx]['action_player1'].astype(str)
                    globals()[f'results_player2_idx{idx}'] = run_df[run_df['idx_p2']==idx]['action_player2'].astype(str)
                    globals()[f'results_run_idx{idx}'] = globals()[f'results_player1_idx{idx}'].append(globals()[f'results_player2_idx{idx}'])

                    globals()[f'results_run_idx{idx}'] = globals()[f'results_run_idx{idx}'].str.replace('1', 'D')
                    globals()[f'results_run_idx{idx}'] = globals()[f'results_run_idx{idx}'].str.replace('0', 'C')
                    globals()[f'results_idx{idx}'][run_name] = globals()[f'results_run_idx{idx}'] #insert result for this particular run into the overall df 
            
        #save the reformatted data
        if not os.path.isdir(f'{destination_folder}/player_actions'):
                os.makedirs(f'{destination_folder}/player_actions')

        print('saving action csvs for each player_idx to disc')
        for idx in range(population_size): #save results
            #idx_title = run_df.loc[run_df['idx_p1']==idx]['title_p1']
            #idx_title = idx_title[idx_title.first_valid_index()]
            idx_title = long_titles[idx]
            globals()[f'results_idx{idx}'].to_csv(f'{destination_folder}/player_actions/{idx_title}_{idx}.csv')



def episode_plot_cooperative_selections_whereavailable(destination_folder, titles, n_runs):
    '''create cooperative_selection bool variable for each run, then plot % cooperative selections'''
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')
    if len(titles)>6: 
        population_list = destination_folder.replace('results/', '')
    
    runnames_intermediate = ['run'+str(i+1) for i in range(n_runs)]
    data_coop_selections = pd.DataFrame(columns=runnames_intermediate)
    data_selections = pd.DataFrame(columns=runnames_intermediate)
    data_availables = pd.DataFrame(columns=runnames_intermediate)

    path = str(destination_folder+'/history/'+'*.csv')
        
    for run in glob.glob(path): #for run in os.listdir(str(destination_folder+'/history')):
        run_name = str(run).strip('.csv').split('/')[-1]
        try: 
            run_df = pd.read_csv(str(run), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
            #drop NAs from the df - i.e. drop rows where a selection was not actually made 
            run_df_clean = run_df[run_df['selected_prev_move'].notna()]
            
            #create boolean variable for cooperative selections amde 
            run_df_clean['cooperative_sel_made'] = run_df_clean.apply(lambda row: row['selected_prev_move']==0, axis=1)

            #create boolean variable for cooperative partners available using env_state 
            run_df_clean['cooperative_partners_available_new'] = run_df_clean.apply(lambda row: row['cooperative_sel_available']==True, axis=1)

            #group by episode 
            run_grouped = run_df_clean[['episode', 'cooperative_partners_available_new', 'cooperative_sel_made']].groupby('episode').agg({
                'cooperative_partners_available_new': 'sum', 'cooperative_sel_made': 'sum'})

            run_grouped['%_coop_sel'] = run_grouped['cooperative_sel_made'] * 100 / run_grouped['cooperative_partners_available_new']
            #fill in 0-division as 0 --> does this make sense?? 
            #run_grouped['%_coop_sel'] = run_grouped.apply(lambda row: 0 if row['cooperative_sel_made']==0 and row['cooperative_sel_available']==0, axis=1)

            data_coop_selections[run_name] = run_grouped['%_coop_sel']
            data_selections[run_name] = run_grouped['cooperative_sel_made']
            data_availables[run_name] = run_grouped['cooperative_partners_available_new']

        except:
            print('failed at run :', run_name)
            break 

    mean_coop_sel = data_coop_selections.mean(axis=1)
    sd_coop_sel = data_coop_selections.std(axis=1)
    ci = 1.96 * sd_coop_sel/np.sqrt(n_runs)

    mean_sel = data_selections.mean(axis=1)
    mean_avail = data_availables.mean(axis=1)

    if len(titles) < 3:
        if 'AC' in titles: 
            linewidth=0.9
        elif 'AD' in titles:
            linewidth=0.9
        else: 
            linewidth = 0.9
    else: 
        linewidth = 0.07

    #plot results 
    plt.figure(dpi=80, figsize=(15, 6))
    #plt.plot(result.index[:], result[:], label=r'% cooperative selections', color='blue', linewidth=linewidth)
    plt.plot(mean_coop_sel, label=r'mean % coop. selections', color='blue', linewidth=linewidth)
    plt.fill_between(mean_coop_sel.index[:], mean_coop_sel-ci, mean_coop_sel+ci, facecolor='lightblue', alpha=0.9)
    plt.title(f'Cooperative selections in population \n {population_list}') # \n (percentage cooperated over '+str(n_runs)+r' runs)'
    plt.gca().set_ylim([-5, 105])
    plt.ylabel("% cooperative partners selected \n (those that C in last move) \n over all cooperative partners available  \n (mean over "+str(n_runs)+f" runs +- 95% CI")
    plt.xlabel('Episode')

    #save the plot
    if not os.path.isdir(f'{destination_folder}/plots'):
            os.makedirs(f'{destination_folder}/plots')

    plt.savefig(f'{destination_folder}/plots/episode_cooperative_selections_{population_list}_whereavailable.png', bbox_inches='tight')

    population_learners = [x for x in titles if x not in ['AC', 'AD', 'TFT', 'Random', 'CD', 'GEa','GEb', 'GEc', 'GEd', 'GEe']]
    
    #plot cooperative selections made - absolute number out of n_runs
    plt.figure(dpi=80, figsize=(15, 6))
    #plt.plot(data_selections['sum_across_runs'].index[:], data_selections['sum_across_runs'][:], label=r'num cooperative selections made', color='blueviolet', linewidth=0.07)
    plt.plot(mean_sel, label=r'num cooperative selections made', color='blueviolet', linewidth=linewidth)
    plt.title(f'Cooperative selections made in population \n {population_list}') # \n (percentage cooperated over '+str(n_runs)+r' runs)'
    plt.gca().set_ylim([-0.1, len(population_learners)+0.1]) #plt.gca().set_ylim([-0.5, n_runs+0.5])
    plt.ylabel("num cooperative selections made  \n (those that C in last move)  \n (mean over "+str(n_runs)+r" runs)")
    plt.xlabel('Episode')

    if ('AC' in titles or 'AD' in titles): 
        linewidth=0.9
    else: 
        linewidth = 0.07

    #plot cooperative selections available - absolute number out of n_runs
    plt.figure(dpi=80, figsize=(15, 6))
    #plt.plot(data_availables['sum_across_runs'].index[:], data_availables['sum_across_runs'][:], label=r'num cooperative opponents available', color='indigo', linewidth=0.07)
    plt.plot(mean_avail, label=r'num cooperative opponents available', color='indigo', linewidth=linewidth)
    plt.title(f'Cooperative partners available in population \n {population_list}') # \n (percentage cooperated over '+str(n_runs)+r' runs)'
    plt.gca().set_ylim([-0.1, len(population_learners)+0.1]) #plt.gca().set_ylim([-0.5, n_runs+0.5])
    plt.ylabel("num cooperative partners available \n (those that C in last move)  \n (mean over "+str(n_runs)+r" runs)")
    plt.xlabel('Episode')

def episode_plot_cooperative_selections_perrun(destination_folder, titles, n_runs):
    '''create cooperative_selection bool variable for each run, then plot % cooperative selections'''
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')
    if len(titles)>6: 
        population_list = destination_folder.replace('results/', '')
    
    runnames_intermediate = ['run'+str(i+1) for i in range(n_runs)]
    data_coop_selections = pd.DataFrame(columns=runnames_intermediate)
    data_selections = pd.DataFrame(columns=runnames_intermediate)
    data_availables = pd.DataFrame(columns=runnames_intermediate)

    path = str(destination_folder+'/history/'+'*.csv')
        
    for run in glob.glob(path): #for run in os.listdir(str(destination_folder+'/history')):
        run_name = str(run).strip('.csv').split('/')[-1]
        try: 
            run_df = pd.read_csv(str(run), index_col=0)#[['title_p1', 'title_p2', 'idx_p1', 'idx_p2', 'state_player1', 'action_player1', 'state_player2', 'action_player2']]
            #drop NAs from the df - i.e. drop rows where a selection was not actually made 
            run_df_clean = run_df[run_df['selected_prev_move'].notna()]
            
            #create boolean variable for cooperative selections amde 
            run_df_clean['cooperative_sel_made'] = run_df_clean.apply(lambda row: row['selected_prev_move']==0, axis=1)

            #create boolean variable for cooperative partners available using env_state 
            run_df_clean['cooperative_partners_available_new'] = run_df_clean.apply(lambda row: row['cooperative_sel_available']==True, axis=1)

            #group by episode 
            run_grouped = run_df_clean[['episode', 'cooperative_partners_available_new', 'cooperative_sel_made']].groupby('episode').agg({
                'cooperative_partners_available_new': 'sum', 'cooperative_sel_made': 'sum'})

            run_grouped['%_coop_sel'] = run_grouped['cooperative_sel_made'] * 100 / run_grouped['cooperative_partners_available_new']
            #fill in 0-division as 0 --> does this make sense?? 
            #run_grouped['%_coop_sel'] = run_grouped.apply(lambda row: 0 if row['cooperative_sel_made']==0 and row['cooperative_sel_available']==0, axis=1)

            data_coop_selections[run_name] = run_grouped['%_coop_sel']
            data_selections[run_name] = run_grouped['cooperative_sel_made']
            data_availables[run_name] = run_grouped['cooperative_partners_available_new']

        except:
            print('failed at run :', run_name)
            break 

    
    for run in data_coop_selections.columns:
        plt.figure(dpi=80, figsize=(15, 6))
        plt.plot(data_coop_selections[run], linewidth=0.08)# alpha=0.5
        plt.title(f'Cooperative selections in population \n {population_list}, {run}') 
        #plt.gca().set_ylim([-5, 105])
        plt.ylabel("% cooperative partners selected \n (those that C in last move) \n over all cooperative partners available  \n (every run separate")
        plt.xlabel('Episode')



def episode_plot_actions_eachplayerinpopulation(destination_folder, titles, n_runs, num_iter, population_size):
    '''explore the actions taken by players aggregated per episode - what percentage are cooperating? '''
    
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')
    if len(titles)>6: 
        population_list = destination_folder.replace('results/', '')

    if False: 
        if population_size == 2: 
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(6*population_size,2.5))
            linewidth = 0.06
        elif population_size == 3: 
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(6*population_size,2.5))
            linewidth = 0.05
        elif population_size == 4: 
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharey=True, sharex=True, figsize=(6*population_size,2.5))
            linewidth = 0.03
        elif population_size == 5: 
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True, figsize=(6*population_size,2.5))
            linewidth = 0.01
        elif population_size == 6: 
            fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=1, ncols=6, sharey=True, sharex=True, figsize=(6*population_size,2.5))
            linewidth = 0.01
        elif population_size == 7: 
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(nrows=1, ncols=7, sharey=True, sharex=True, figsize=(3*population_size,2.5))
            linewidth = 0.01
        elif population_size == 10: 
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(nrows=1, ncols=10, sharey=True, sharex=True, figsize=(3*population_size,2.5))
            linewidth = 0.005
        elif population_size == 14: 
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14) = plt.subplots(nrows=1, ncols=14, sharey=True, sharex=True, figsize=(3*population_size,2.5))
            linewidth = 0.005
        elif population_size == 20: 
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20) = plt.subplots(nrows=1, ncols=20, sharey=True, sharex=True, figsize=(3*population_size,2.5))
            linewidth = 0.005 #0.0005
        elif population_size == 11: 
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11) = plt.subplots(nrows=1, ncols=11, sharey=True, sharex=True, figsize=(3*population_size,2.5))
            linewidth = 0.005
        elif population_size == 21: 
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21) = plt.subplots(nrows=1, ncols=21, sharey=True, sharex=True, figsize=(3*population_size,2.5))
            linewidth = 0.005
        
    fig, axs = plt.subplots(nrows=1, ncols=population_size, sharey=True, sharex=True, figsize=(3*population_size,2.5))
    linewidth = 0.005
    #results_idx = pd.DataFrame(index=range(num_iter))

    episodes_column = np.repeat(range(num_iter), population_size)

    #plot for each player type: 
    for idx in range(len(titles)):
        title = title_mapping[titles[idx]] #get long title 

        if title in ['AlwaysCooperate', 'AlwaysDefect', 'GilbertElliot_typeb']:
            linewidth_new = 1.5
        else: 
            linewidth_new = linewidth

        actions_idx = pd.read_csv(f'{destination_folder}/player_actions/{title}_{idx}.csv', index_col=0).sort_index()
        results_idx = pd.DataFrame(index=range(num_iter)) #actions_idx.index)

        #manually add episode to the actions_df 
#TO DO check that this will work correctly in all cases and with NAs in Selfish player's actions !!! 

        #mask D actions, only keeping Cc so we can the count 
        actions_C = actions_idx.mask(actions_idx.ne('C'))
        actions_C['episode'] = episodes_column
        
        actions_C_episode = actions_C.groupby('episode').count()

        actions_idx['episode'] = episodes_column
        actions_all_episode = actions_idx.groupby('episode').count()
        

        #calculate % of 100 agents (runs) that cooperate at every step  out of the 10000
        results_idx['%_cooperate'] = actions_C_episode.sum(axis='columns') / actions_all_episode.sum(axis='columns') * 100


        #ax = locals()["ax"+str(idx+1)]
        ax = axs[idx]
        ax.set_title(f'\n {title} \n player{idx}') #f'\n {title} player{idx}'
        #plt.title(f'Cooperation by {title} player{idx} \n from population {population_list}') #(\n percent cooperated over '+str(n_runs)+r' runs)'

        if movingaverage: 
            linewidth_new = 0.5
            movingaverage_option = f'\n (moving average, window {movingN})'

            movMean = results_idx.rolling(window=movingN,center=True,min_periods=1).mean()
            #movStd = results_idx.rolling(window=movingNN,center=True,min_periods=1).std()
            ## get mean +/- 1.96*std/sqrt(N)
            #confIntPos = movMean + 1.96 * movStd / np.sqrt(N)
            #confIntNeg = movMean - 1.96 * movStd / np.sqrt(N)

            ax.plot(results_idx.index[:], movMean, color='darkgreen', linewidth=linewidth_new) #label=f'{title}_{idx}', 

        else: #if plotting every value 
            movingaverage_option = ''
            #plot results 
            ax.plot(results_idx.index[:], results_idx['%_cooperate'], color='darkgreen', linewidth=linewidth_new) #label=f'{title}_{idx}', 
        
        ax.set_ylim([-10, 110])
        ax.set(xlabel='Episode', ylabel='') #ylabel='Percentage cooperating \n (over '+str(n_runs)+r' runs)'

        #plt.savefig(f'{destination_folder}/plots/cooperation_{title}_{idx}.png', bbox_inches='tight')

    axs[0].set(ylabel=r'% cooperating'+ f'\n(over {n_runs} runs) {movingaverage_option}')
    #plt.ylabel(r'% cooperating'+ f'\n(over {n_runs} runs)')
    plt.suptitle(f'% cooperating for each player from population {population_list} \n', fontsize=28, y=1.4)

    if not os.path.isdir(f'{destination_folder}/plots'):
        os.makedirs(f'{destination_folder}/plots')

    plt.savefig(f'{destination_folder}/plots/episode_cooperation_eachplayer_{movingaverage_option}.png', bbox_inches='tight')


 


def plot_actions_aggbyplayertype_avgacrossruns(destination_folder, titles, n_runs, num_iter, population_size, title='Selfish'):
    '''explore the actions taken by player types aggregated per episode - what percentage are cooperating? '''

    if False:     
        titles_short = [title.replace('QL','') for title in titles] 
        temp = ''
        for t in titles_short: 
            temp += (t+',')
        population_list = temp.strip(',')
        if len(titles)>6: 
            population_list = destination_folder.replace('results/', '')

    linewidth = 0.005
    
    results_allplayersthistype = pd.DataFrame(index=range(num_iter))
    episodes_column = np.repeat(range(num_iter), population_size)
    indices_this_type = [i for i, x in enumerate(long_titles) if x == title]

    #plot for each player type: 
    for idx in indices_this_type:

        actions_idx = pd.read_csv(f'{destination_folder}/player_actions/{title}_{idx}.csv', index_col=0).sort_index()
        results_idx = pd.DataFrame(index=range(num_iter)) #actions_idx.index)

        #manually add episode to the actions_df 
#TO DO check that this will work correctly in all cases and with NAs in Selfish player's actions !!! 

        #mask D actions, only keeping Cc so we can the count 
        actions_C = actions_idx.mask(actions_idx.ne('C'))
        actions_C['episode'] = episodes_column
        
        actions_C_episode = actions_C.groupby('episode').count()

        actions_idx['episode'] = episodes_column
        actions_all_episode = actions_idx.groupby('episode').count()

        #calculate % of 100 agents (runs) that cooperate at every step out of the 10000
        results_idx['%_cooperate'] = actions_C_episode.sum(axis='columns') / actions_all_episode.sum(axis='columns') * 100

        results_allplayersthistype[f'{title}_{idx}'] = results_idx['%_cooperate']

    results_allplayersthistype['mean'] = results_allplayersthistype.mean(axis=1)
    results_allplayersthistype['sd'] = results_allplayersthistype.std(axis=1)
    results_allplayersthistype['ci'] = 1.96 * results_allplayersthistype['sd']/np.sqrt(n_runs)

    #plot results 
    plt.figure(dpi=80, figsize=(7, 6)) #3, 2.5
    
    if movingaverage: 
        movingaverage_option = f'\n (moving average, window {movingN})'
        linewidth = 0.5

        movMean = results_allplayersthistype['mean'].rolling(window=movingN,center=True,min_periods=1).mean()
        #movStd = results_allplayersthistype['sd'].rolling(window=movingN,center=True,min_periods=1).std()
        movStd = results_allplayersthistype['mean'].rolling(window=movingN,center=True,min_periods=1).std()
        # get mean +/- 1.96*std/sqrt(N)
        #confIntPos = movMean + 1.96 * movStd / np.sqrt(movingN)
        #confIntNeg = movMean - 1.96 * movStd / np.sqrt(movingN)
        confIntPos = movMean + 1.96 * movStd / np.sqrt(n_runs)
        confIntNeg = movMean - 1.96 * movStd / np.sqrt(n_runs)


        plt.plot(results_allplayersthistype.index[:], movMean, color='darkgreen', linewidth=linewidth)
        plt.fill_between(results_allplayersthistype.index[:], confIntPos, confIntNeg, color='lightgreen', linewidth=linewidth, alpha=0.8) #label=f'{title}_{idx}',    
        
    else: #if plotting all values 
        plt.plot(results_allplayersthistype.index[:], results_allplayersthistype['mean'], color='darkgreen', linewidth=linewidth)
        plt.fill_between(results_allplayersthistype.index[:], results_allplayersthistype['mean']+results_allplayersthistype['ci'], results_allplayersthistype['mean']-results_allplayersthistype['ci'], color='lightgreen', linewidth=linewidth, alpha=0.8) #label=f'{title}_{idx}',    
        
    plt.title(f'% Cooperating (mean over {len(indices_this_type)} {title} players) ') 
    plt.gca().set_ylim([-10, 110])
    plt.ylabel(r'% Cooperating'+ 
                f'\n (mean over '+str(n_runs)+f' runs \n & {len(indices_this_type)} {title} players' + r'+- CI)')
    plt.xlabel('Episode')


    if not os.path.isdir(f'{destination_folder}/plots'):
        os.makedirs(f'{destination_folder}/plots')

    plt.savefig(f'{destination_folder}/plots/episode_actions_agg{title}_avgacrossruns_{movingaverage_option}.png', bbox_inches='tight')




def episode_plot_reward_eachplayerinpopulation(destination_folder, titles, n_runs, num_iter, population_size, which='game', first_100=False):
    '''explore the actions taken by players aggregated per episode - what percentage are cooperating? '''
    
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')
    if len(titles)>6: 
        population_list = destination_folder.replace('results/', '')

    if which == 'intrinsic':
        sharey = False
    elif which == 'game':
        sharey=True
    fig, axs = plt.subplots(nrows=1, ncols=population_size, sharey=sharey, sharex=True, figsize=(3*population_size,2.5))
    #linewidth = 0.0007
    linewidth = 0.001

    #results_idx = pd.DataFrame(index=range(num_iter))

    episodes_column = np.repeat(range(num_iter), population_size)

    #plot for each player type: 
    for idx in range(len(titles)):
        title = title_mapping[titles[idx]] #get long title 

        if title in ['AlwaysCooperate', 'AlwaysDefect', 'GilbertElliot_typeb']:
            linewidth_new = 1.5
        elif first_100:
            linewidth_new = 0.5
        else: 
            linewidth_new = linewidth


        if first_100:
            results_idx = pd.DataFrame(index=range(100)) #actions_idx.index)
        else:
            results_idx = pd.DataFrame(index=range(num_iter*population_size)) #actions_idx.index)


        for run in range(1, n_runs+1): 
            if first_100:
                run_df = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0).sort_index()[0:100]
            else:
                run_df = pd.read_csv(f'{destination_folder}/history/run{run}.csv',index_col=0).sort_index()
            small_run_df_p1 = run_df[run_df['idx_p1']==idx]
            rewards_idx_p1 = small_run_df_p1[[f'reward_{which}_player1']]
            small_run_df_p2 = run_df[run_df['idx_p2']==idx]
            rewards_idx_p2 = small_run_df_p2[[f'reward_{which}_player2']]

            rewards_idx = rewards_idx_p1[f'reward_{which}_player1'].append(rewards_idx_p2[f'reward_{which}_player2'])
            rewards_idx.sort_index(inplace=True)
            rewards_idx_cumsum = np.cumsum(rewards_idx)
            results_idx[f'run{run}'] = rewards_idx_cumsum

        #calculate average across runs for every iteration 
        means_reward = results_idx.mean(axis='columns')
        sds_reward =  results_idx.std(axis='columns')
        CIs_reward = 1.96 * sds_reward/np.sqrt(n_runs)

        #results_idx['episode'] = episodes_column
        #results_idx_episode = results_idx.groupby('episode').count()

        #### TO EDIT FROM HERE
        if False: 
            #mask D actions, only keeping Cc so we can the count 
            actions_C = actions_idx.mask(actions_idx.ne('C'))
            actions_C['episode'] = episodes_column
            
            actions_C_episode = actions_C.groupby('episode').count()

            actions_idx['episode'] = episodes_column
            actions_all_episode = actions_idx.groupby('episode').count()
            

            #calculate % of 100 agents (runs) that cooperate at every step  out of the 10000
            results_idx['%_cooperate'] = actions_C_episode.sum(axis='columns') / actions_all_episode.sum(axis='columns') * 100

        #ax = locals()["ax"+str(idx+1)]
        ax = axs[idx]
        ax.set_title(f'\n {title} \n player{idx}') #f'\n {title} player{idx}'
        #plt.title(f'Rewards for {title} player{idx} \n from population {population_list}') #(\n percent cooperated over '+str(n_runs)+r' runs)'

        #plot results 
        ax.plot(results_idx.index[:], means_reward, color='blue', linewidth=linewidth_new, alpha=0.5) #label=f'{title}_{idx}', 
        ax.fill_between(results_idx.index[:], means_reward-CIs_reward, means_reward+CIs_reward, facecolor='lightblue', alpha=0.5)

        #ax.set_ylim([-5, num_iter*population_size*4])
        ax.set(xlabel='Iteration', ylabel='') #ylabel='Percentage cooperating \n (over '+str(n_runs)+r' runs)'

        #plt.savefig(f'{destination_folder}/plots/cooperation_{title}_{idx}.png', bbox_inches='tight')

    axs[0].set(ylabel=f'Cumulative {which[0].upper()+which[1:]} Reward \n(mean over {n_runs} runs)')
    #plt.ylabel(r'% cooperating'+ f'\n(over {n_runs} runs)')
    if first_100:
        string = 'FIRST 100 -'
    else:
        string=''
    plt.suptitle(f'{string}Cumulative {which[0].upper()+which[1:]} Reward for each player from population {population_list} \n', fontsize=28, y=1.4)

    plt.savefig(f'{destination_folder}/plots/episode_cumul{which}reward_eachplayer.png', bbox_inches='tight')


def episode_plot_collective_outcomes_population(destination_folder, titles, n_runs, num_iter, population_size):
    '''explore the actions taken by players aggregated per episode - what percentage are cooperating? '''
    
    titles_short = [title.replace('QL','') for title in titles] 
    temp = ''
    for t in titles_short: 
        temp += (t+',')
    population_list = temp.strip(',')
    if len(titles)>6: 
        population_list = destination_folder.replace('results/', '')

 
    results_collective_reward = pd.DataFrame(index=range(num_iter))
    results_collective_reward_cumulative = pd.DataFrame(index=range(num_iter))
    results_gini_reward = pd.DataFrame(index=range(num_iter))
    results_min_reward = pd.DataFrame(index=range(num_iter))

    
    for run_idx in range(n_runs):
        run_idx += 1
        run_df = pd.read_csv(destination_folder+f'/history/run{run_idx}.csv', index_col=0)[['episode', 'reward_collective', 'reward_gini', 'reward_min']]

        #group by episode 
        run_grouped = run_df.groupby('episode').agg({'reward_collective': 'sum', 'reward_gini':'mean', 'reward_min':'mean'})

        results_collective_reward[f'run{run_idx}'] = run_grouped['reward_collective']
        results_collective_reward_cumulative[f'run{run_idx}'] = np.cumsum(run_grouped['reward_collective'])
        results_gini_reward[f'run{run_idx}'] = run_grouped['reward_gini']
        results_min_reward[f'run{run_idx}'] = run_grouped['reward_min']

    #do COLLECTIVE_REWARD first - non-cumulative 
    results_collective_reward['collective_R_mean'] = results_collective_reward.mean(axis=1)
    sds = results_collective_reward.std(axis=1)
    results_collective_reward['collective_R_ci'] = 1.96 * sds/np.sqrt(n_runs)

    #plot results - non-cumulative 
    plt.figure(dpi=80, figsize=(15, 6)) 
    plt.plot(results_collective_reward.index[:], results_collective_reward['collective_R_mean'], label=f'{population_list}', color='purple', linewidth=0.05)
    plt.fill_between(results_collective_reward.index[:], results_collective_reward['collective_R_mean']-results_collective_reward['collective_R_ci'], results_collective_reward['collective_R_mean']+results_collective_reward['collective_R_ci'], facecolor='lightpink', linewidth=0.04, alpha=1)
   
    plt.title(f'Collective reward (mean) in population \n {population_list} ') 
    plt.gca().set_ylim([2*population_size, 6*population_size])
    plt.ylabel(r'Collective reward (mean per episode)'+ '  \n (mean over '+str(n_runs)+r' runs +- CI)')
    plt.xlabel('Episode')
    
    
    if not os.path.isdir(f'{destination_folder}/plots'):
        os.makedirs(f'{destination_folder}/plots')

    plt.savefig(f'{destination_folder}/plots/episode_collective_R_{population_list}.png', bbox_inches='tight')


    #do COLLECTIVE_REWARD - cumulative 
    results_collective_reward_cumulative['collective_R_mean'] = results_collective_reward_cumulative.mean(axis=1)
    sds = results_collective_reward_cumulative.std(axis=1)
    results_collective_reward_cumulative['collective_R_ci'] = 1.96 * sds/np.sqrt(n_runs)

    #plot results - cumulative 
    plt.figure(dpi=80, figsize=(15, 6)) 
    plt.plot(results_collective_reward_cumulative.index[:], results_collective_reward_cumulative['collective_R_mean'], label=f'{population_list}', color='purple', linewidth=0.05)
    plt.fill_between(results_collective_reward_cumulative.index[:], results_collective_reward_cumulative['collective_R_mean']-results_collective_reward_cumulative['collective_R_ci'], results_collective_reward_cumulative['collective_R_mean']+results_collective_reward_cumulative['collective_R_ci'], facecolor='lightpink', linewidth=0.04, alpha=1)
   
    plt.title(f'Collective reward (cumulative) in population \n {population_list} ') 
    #plt.gca().set_ylim([0, max(results_collective_reward_cumulative['collective_R_mean'])+1000])
    plt.ylabel(r'Collective reward (cumulative)'+ '  \n (mean over '+str(n_runs)+r' runs +- CI)')
    plt.xlabel('Episode')
    
    plt.savefig(f'{destination_folder}/plots/episode_collective_R_cumul_{population_list}.png', bbox_inches='tight')

    
    

    #do GINI_REWARD - non-cumulative 
    results_gini_reward['gini_R_mean'] = results_gini_reward.mean(axis=1)
    sds = results_gini_reward.std(axis=1)
    results_gini_reward['gini_R_ci'] = 1.96 * sds/np.sqrt(n_runs)

    #plot results - non-cumulative 
    plt.figure(dpi=80, figsize=(15, 6)) 
    plt.plot(results_gini_reward.index[:], results_gini_reward['gini_R_mean'], label=f'{population_list}', color='orange', linewidth=0.05)
    plt.fill_between(results_gini_reward.index[:], results_gini_reward['gini_R_mean']-results_gini_reward['gini_R_ci'], results_gini_reward['gini_R_mean']+results_gini_reward['gini_R_ci'], facecolor='yellow', linewidth=0.04, alpha=1)
   
    plt.title(f'Gini reward (mean) in population \n {population_list} ') 
    plt.gca().set_ylim([0, 1])
    plt.ylabel(r'Gini reward (mean per episode)'+ '  \n (mean over '+str(n_runs)+r' runs +- CI)')
    plt.xlabel('Episode')

    plt.savefig(f'{destination_folder}/plots/episode_gini_R_{population_list}.png', bbox_inches='tight')


    #do MIN_REWARD - non-cumulative 
    results_min_reward['min_R_mean'] = results_min_reward.mean(axis=1)
    sds = results_min_reward.std(axis=1)
    results_min_reward['min_R_ci'] = 1.96 * sds/np.sqrt(n_runs)

    #plot results - non-cumulative 
    plt.figure(dpi=80, figsize=(15, 6)) 
    plt.plot(results_min_reward.index[:], results_min_reward['min_R_mean'], label=f'{population_list}', color='blue', linewidth=0.05)
    plt.fill_between(results_min_reward.index[:], results_min_reward['min_R_mean']-results_min_reward['min_R_ci'], results_min_reward['min_R_mean']+results_min_reward['min_R_ci'], facecolor='lightblue', linewidth=0.04, alpha=1)
   
    plt.title(f'Min reward (mean) in population \n {population_list} ') 
    plt.gca().set_ylim([0, 3])
    plt.ylabel(r'Min reward (mean per episode)'+ '  \n (mean over '+str(n_runs)+r' runs +- CI)')
    plt.xlabel('Episode')

    plt.savefig(f'{destination_folder}/plots/episode_min_R_{population_list}.png', bbox_inches='tight')



#analyse how often each opponent is selected
def plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range='full'): 
    '''note episode_range uses non-0 indexing, so we need to convert human-readable indices to python-readable'''
    palette = [color_mapping_longtitle[t] for t in long_titles]
    order = list(zip(long_titles, range(population_size)))
    indices_str = [str(i) for i in list(range(population_size))] 
    order = ['_'.join(z) for z in zip(long_titles, indices_str)]
    
    results = pd.DataFrame(index=order)

    if episode_range == 'full':
        episode_idxs = list(range(0,num_iter-1))
        range_label = 'across full episode range'
    elif len(episode_range) == 1: 
        episode_idxs = [i-1 for i in episode_range]
        range_label = f'on episode {episode_idxs[0]+1}'
    elif len(episode_range) > 1:
        episode_idxs = list(range(episode_range[0]-1,episode_range[1]-1))
        range_label = f'on episodes {episode_range[0]+1}-{episode_range[1]+1}'


    for run in range(1,n_runs+1):  
        df = pd.read_csv(f'{destination_folder}/history/run{run}.csv', index_col=0)[['episode', 'title_p2', 'idx_p2']]
        small_df = df.loc[ df['episode'].isin(episode_idxs) ]
        small_df['title_idx_p2'] = small_df['title_p2'] + '_' + small_df['idx_p2'].astype(str)

        results[f'run{run}'] = small_df['title_idx_p2'].value_counts()
        #means = results.mean(axis=1)
        #sds = results.std(axis=1)
        #cis = 1.96 * sds/np.sqrt(n_runs)


    chart = sns.barplot(data=results.T, order=order, palette=palette) 
    #chart = sns.countplot(data=results, x=results['run0'], order=order, palette=palette) #sns.countplot(results_df.stack().reset_index(drop=True)) #x=results_df['title_p2'],
    #sns.set(font_scale = 0.5)
    chart.set_xticklabels(
        chart.get_xticklabels(), 
        rotation=45, 
        horizontalalignment='right',
        fontweight='light',
        fontsize='small'
        )
    chart.set_title(f'Number of times each player is selected \n {range_label} \n (average across {n_runs} runs)')
    chart.set(ylabel='Number of times selected')

    plt.savefig(f'{destination_folder}/plots/episode_selections_episoderange{episode_range}.png', bbox_inches='tight')







################################################
#### combine different populations on one plot ###
################################################

movingN = 1000 

def manypopulations_plot_cooperation_aggbyplayertype_avgacrossruns(POPULATIONS, num_iter, title='Selfish'):
    '''plot cooperation levels by this speicifc player title across many populationns. 
    color each population by the color of the majority player in it. '''
    episodes_column = np.repeat(range(num_iter), population_size_approx)
    linewidth=1.2 #0.009

    results_manypopulations = pd.DataFrame(index=range(num_iter))
    ci_manypopulations = pd.DataFrame(index=range(num_iter))

    for destination_folder in POPULATIONS:  
        titles = get_titles_for_population(destination_folder)
        population_size = len(titles)
        long_titles = [title_mapping[title] for title in titles]
        indices_this_type = [i for i, x in enumerate(long_titles) if x == title]

        results_allplayersthistype = pd.DataFrame(index=range(num_iter))
        ci_allplayersthistype = pd.DataFrame(index=range(num_iter))

        population = destination_folder.replace('___iter30000_runs20', '')
        print(f'calculating for populaiton {population}')
        #plot for each player type: 
        for idx in indices_this_type:

            actions_idx = pd.read_csv(f'results/{destination_folder}/player_actions/{title}_{idx}.csv', index_col=0).sort_index()
            results_idx = pd.DataFrame(index=range(num_iter)) #actions_idx.index)
            ci_idx = pd.DataFrame(index=range(num_iter))

            #manually add episode to the actions_df 
        #TO DO check that this will work correctly in all cases and with NAs in Selfish player's actions !!! 

            #mask D actions, only keeping Cc so we can the count 
            actions_C = actions_idx.mask(actions_idx.ne('C'))
            actions_C['episode'] = episodes_column
                
            actions_C_episode = actions_C.groupby('episode').count()

            actions_idx['episode'] = episodes_column
            actions_all_episode = actions_idx.groupby('episode').count()

            #calculate % of 100 agents (runs) that cooperate at every step out of the 10000
            #results_idx['%_cooperate'] = actions_C_episode.sum(axis='columns') / actions_all_episode.sum(axis='columns') * 100
            percentages_df = actions_C_episode / actions_all_episode * 100
            percentages_df_rolling = percentages_df.rolling(window=movingN,center=True,min_periods=1).mean()

            results_idx['%_cooperate'] = percentages_df_rolling.mean(axis=1) 
            ci_idx['%_cooperate'] = 1.96 * percentages_df_rolling.std(axis=1) / np.sqrt(n_runs_global)

            results_allplayersthistype[f'{title}_{idx}'] = results_idx['%_cooperate']
            ci_allplayersthistype[f'{title}_{idx}'] = ci_idx['%_cooperate']

        #if applicable, average allplayersthistype 
        if len(indices_this_type) > 1: 
            results_allplayersthistype['mean'] = results_allplayersthistype.mean(axis=1)
            ci_allplayersthistype['mean'] = ci_allplayersthistype.mean(axis=1)
        else: 
            results_allplayersthistype['mean'] = results_allplayersthistype#[columns[0]]
            ci_allplayersthistype['mean'] = ci_allplayersthistype#[columns[0]]

        #results_allplayersthistype['mean'] = results_allplayersthistype.mean(axis=1)
        #results_allplayersthistype['sd'] = results_allplayersthistype.std(axis=1)
        #results_allplayersthistype['ci'] = 1.96 * results_allplayersthistype['sd']/np.sqrt(n_runs)


        if False: 
            movingaverage_option = f'\n (moving average, window {movingN})'
            linewidth = 0.8 #overwrite the linewidth with a larger value than above 

            movMean = results_allplayersthistype['mean'].rolling(window=movingN,center=True,min_periods=1).mean()
            #movStd = results_allplayersthistype['sd'].rolling(window=movingN,center=True,min_periods=1).std()
            movStd = results_allplayersthistype['mean'].rolling(window=movingN,center=True,min_periods=1).std()
            # get mean +/- 1.96*std/sqrt(N)
            #confIntPos = movMean + 1.96 * movStd / np.sqrt(movingN)
            #confIntNeg = movMean - 1.96 * movStd / np.sqrt(movingN)
            confInt = 1.96 * movStd / np.sqrt(n_runs)

            results_manypopulations[population] = movMean
            ci_manypopulations[population] = confInt

            #plt.plot(results_allplayersthistype.index[:], movMean, color='darkgreen', linewidth=linewidth)
            #plt.fill_between(results_allplayersthistype.index[:], confIntPos, confIntNeg, color='lightgreen', linewidth=linewidth, alpha=0.8) #label=f'{title}_{idx}',    
            
        #if plotting all values
        movingaverage_option = f'\n (moving avg., window {movingN})'

        results_manypopulations[population] = results_allplayersthistype['mean']
        ci_manypopulations[population] = ci_allplayersthistype['mean']

    print('plotting...')
    #plot results across populations:
    plt.figure(dpi=80, figsize=(6, 6)) #3, 2.5
    sns.set(font_scale=2.5)
    
    for destination_folder in POPULATIONS: 
        population = destination_folder.replace('___iter30000_runs20', '') 
        short_titles = get_titles_for_population(destination_folder)
        majority_title_short = max(set(short_titles), key = short_titles.count)
        majority_title = title_mapping[majority_title_short]

        population_forpaper = f'majority-{title_mapping_forpaper_long[majority_title]}'

        plt.plot(results_manypopulations.index[:], results_manypopulations[population], linewidth=linewidth, alpha=1, label=population_forpaper, color=color_mapping_longtitle[majority_title])
        plt.fill_between(results_manypopulations.index[:], results_manypopulations[population]+ci_manypopulations[population], results_manypopulations[population]-ci_manypopulations[population], linewidth=linewidth, alpha=0.1, color=color_mapping_longtitle[majority_title])
    #plt.legend(loc='lower center', fontsize="10", bbox_to_anchor=(0.5, -.60))
    # change the line width for the legend
    leg = plt.legend(title='Population', loc='right', bbox_to_anchor=(1.70, 0.5), fontsize="21")
    for line in leg.get_lines():
        line.set_linewidth(5.0)
    title_forplot = title_mapping_forpaper_longtoclean[title] #title_mapping_forpaper_long[title]
    plt.title(f'Cooperation by {title_forplot} player(s) \n across {len(POPULATIONS)} populations ') #% 
    plt.gca().set_ylim([-10, 110])
    plt.ylabel(f'% {title_forplot} Players Cooperating'+ 
                f'\n (mean per episode) {movingaverage_option} \n (mean over 20 runs '+ r'+- CI)')
    plt.xlabel('Episode')

    plt.savefig(f'outcome_plots/manypopulaitons_actions_agg{title}_avgacrossruns_{movingaverage_option}_new.png', bbox_inches='tight')
    

def manypopulations_plot_reward_aggbyplayertype_avgacrossruns(POPULATIONS, titles, n_runs, num_iter, population_size, which='game', cumulative = False, title='Selfish', agg='sum'):
    ''' This plots the non-cumulative reward for 
    which = 'game' or 'intr' '''
    episodes_column = np.repeat(range(num_iter), population_size)

    results_manypopulations = pd.DataFrame(index=range(num_iter))
    ci_manypopulations = pd.DataFrame(index=range(num_iter))

    for destination_folder in POPULATIONS: 

        print('looping over population:')
        titles = get_titles_for_population(destination_folder)
        population_size = len(titles)
        long_titles = [title_mapping[title] for title in titles]
        indices_this_type = [i for i, x in enumerate(long_titles) if x == title]

        results_allplayersthistype = pd.DataFrame(index=range(num_iter))

        population = destination_folder.replace('___iter30000_runs20', '')

        #plot for each player type: 
        for idx in indices_this_type:
            idx_title = long_titles[idx]

            rewards_idx = pd.read_csv(f'results/{destination_folder}/player_rewards/R_{which}_{idx_title}_{idx}.csv', index_col=0).sort_index()
            results_idx = pd.DataFrame(index=range(num_iter)) #actions_idx.index)
    #TO DO check that this will work correctly in all cases and with NAs in Selfish player's actions !!! 
            
            #manually add episode to the actions_df 
            rewards_idx['episode'] = episodes_column
            if agg == 'mean': 
                rewards_idx_episode = rewards_idx.groupby('episode').mean() #calcualte average reward per episode (on every run) 
            elif agg == 'sum': 
                rewards_idx_episode = rewards_idx.groupby('episode').sum() #calcualte total reward per episode (on every run) 

            if cumulative: 
                rewards_idx_episode_cumul = np.cumsum(rewards_idx_episode, axis=0)
                #rewards_idx_episode[f'R_{which}_cumul_mean'] = rewards_idx_episode_cumul.mean(axis=1)
                results_allplayersthistype[f'{title}_{idx}'] = rewards_idx_episode_cumul.mean(axis=1)

            else: #if not using cumulative reward values 
                #rewards_idx_episode[f'R_{which}_mean'] = rewards_idx_episode.mean(axis=1)
                results_allplayersthistype[f'{title}_{idx}'] = rewards_idx_episode.mean(axis=1)

            #results_allplayersthistype[f'{title}_{idx}'] = rewards_idx_episode[f'R_{which}_mean']

                

        results_allplayersthistype['mean'] = results_allplayersthistype.mean(axis=1)
        results_allplayersthistype['sd'] = results_allplayersthistype.std(axis=1)
        results_allplayersthistype['ci'] = 1.96 * results_allplayersthistype['sd']/np.sqrt(n_runs)

        results_manypopulations[population] = results_allplayersthistype['mean']
        ci_manypopulations[population] = results_allplayersthistype['ci']

    print(f'plotting {which} reward...')
    #plot results across populations:
    if cumulative: 
        linewidth=0.4
    else: 
        linewidth=0.01
    plt.figure(dpi=80, figsize=(7, 6)) #3, 2.5
    for destination_folder in POPULATIONS: 
        population = destination_folder.replace('___iter30000_runs20', '') 
        plt.plot(results_manypopulations.index[:], results_manypopulations[population], linewidth=linewidth, alpha=0.9, label=population)
        plt.fill_between(results_manypopulations.index[:], results_manypopulations[population]+ci_manypopulations[population], results_manypopulations[population]-ci_manypopulations[population], color='lightgrey', linewidth=linewidth, alpha=0.8)    
    #plt.legend(loc='lower center', fontsize="10", bbox_to_anchor=(0.5, -.60))
    # change the line width for the legend
    leg = plt.legend(loc='lower center', fontsize="10", bbox_to_anchor=(0.5, -.60))
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    if cumulative: 
        cumul_option = 'Cumulative'
    else: #if cumulative == False 
        cumul_option = '' 
        if agg == 'mean':
            if which == 'game': 
                plt.gca().set_ylim([0, 4])
    plt.title(f'{cumul_option} {which.capitalize()} Reward for {title} players \n across {len(POPULATIONS)} populations') 
    plt.ylabel(f'{cumul_option} {which.capitalize()} Reward'+ 
                f'\n ({agg} per episode)' +
                f'\n (average across {str(n_runs)} runs ' + r'+- CI)') 
    plt.xlabel('Episode')

    if not os.path.isdir('outcome_plots'):
        os.makedirs('outcome_plots')

    plt.savefig(f'outcome_plots/manypopulaitons_{cumul_option}{which}reward_agg{title}_avgacrossruns.png', bbox_inches='tight')
    


movingN = 200 
 
def manypopulations_calculate_cooperation_allplayers_avgacrossruns(POPULATIONS, num_iter):
    '''explore the actions taken by players at every step out of num_iter - what percentage are cooperating?
    % per episode 
    Note this uses the global variable movingN'''

    results_manypopulations = pd.DataFrame(index=range(num_iter))
    ci_manypopulations = pd.DataFrame(index=range(num_iter))

    
    for destination_folder in POPULATIONS: 
        n_runs = len(glob.glob('results/'+destination_folder+'/history/*.csv'))
        population = destination_folder.replace('___iter30000_runs20', '') 
        print(f'processing {population} with {n_runs} runs')

        result_population = pd.DataFrame(index=range(num_iter)) 

        for run_idx in range(n_runs):
            run_idx += 1
            run_df = pd.read_csv(f'results/{destination_folder}/history/run{run_idx}.csv', index_col=0)[['episode', 'action_player1', 'action_player2']]

            #create boolean variabnle for cooperative selections amde 
            run_df['action_player1_C'] = run_df.apply(lambda row: row['action_player1']==0, axis=1)
            run_df['action_player2_C'] = run_df.apply(lambda row: row['action_player2']==0, axis=1)

            #group by episode 
            run_grouped = run_df.groupby('episode').agg({'action_player1_C': 'sum', 'action_player2_C': 'sum', 'action_player1':'count', 'action_player2':'count'})
            run_grouped['action_C'] = run_grouped['action_player1_C'] + run_grouped['action_player2_C']
            run_grouped['actions_total'] = run_grouped['action_player1'] + run_grouped['action_player2']
            run_grouped['%_C'] = run_grouped['action_C'] * 100 / run_grouped['actions_total']

            
            #EITHER add raw result to output df 
            #result_population[f'run{run_idx}'] = run_grouped['%_C']

            #OR calcualte moving average for each run before storing data 
            rolling_mean_thisrun = run_grouped['%_C'].rolling(window=movingN,center=True,min_periods=1).mean()
            result_population[f'run{run_idx}'] = rolling_mean_thisrun

        #average the rolling means across runs and calcualte CI 
        means = result_population.mean(axis=1)
        sds = result_population.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)
        results_manypopulations[population] = means
        ci_manypopulations[population] = ci

    #save data 
    if not os.path.isdir('outcome_data'):
        os.makedirs('outcome_data')

    if not os.path.isdir('outcome_data/cooperation_allplayers'):
        os.makedirs('outcome_data/cooperation_allplayers')
 
    results_manypopulations.to_csv('outcome_data/cooperation_allplayers/results_manypopulations.csv')
    ci_manypopulations.to_csv('outcome_data/cooperation_allplayers/ci_manypopulations.csv')

def manypopulations_plot_cooperation_allplayers_avgacrossruns(POPULATIONS, num_iter):
    '''explore the actions taken by players at every step out of num_iter - what percentage are cooperating? 
    Plot all populations on one plot.'''
    
    #read in data 
    results_manypopulations = pd.read_csv('outcome_data/cooperation_allplayers/results_manypopulations.csv', index_col=0)
    ci_manypopulations = pd.read_csv('outcome_data/cooperation_allplayers/ci_manypopulations.csv', index_col=0)

    #plot results 
    plt.figure(dpi=80, figsize=(6, 6)) 
    linewidth = 1 #0.1 #0.06
    sns.set(font_scale=2.5)

    for destination_folder in POPULATIONS: 
        population = destination_folder.replace('___iter30000_runs20', '') 
        print(f'plotting population {population}')
        short_titles = get_titles_for_population(destination_folder)
        majority_title_short = max(set(short_titles), key = short_titles.count)
        majority_title = title_mapping[majority_title_short]

        population_forpaper = f'majority-{title_mapping_forpaper_long[majority_title]}'


        if False: 
            movingaverage_option = f'\n (moving average, window {movingN})'
            linewidth = 0.8 #overwrite the linewidth with a larger value than above 

            movMean = results_manypopulations[population].rolling(window=movingN,center=True,min_periods=1).mean()
            #TO DO figure out if I can use the std of the mean or if I need to use the std of the raw data
            movStd = results_manypopulations[population].rolling(window=movingN,center=True,min_periods=1).std()
            # get mean +/- 1.96*std/sqrt(N)
            #confIntPos = movMean + 1.96 * movStd / np.sqrt(movingN)
            #confIntNeg = movMean - 1.96 * movStd / np.sqrt(movingN)
            confInt = 1.96 * movStd / np.sqrt(n_runs_global)

            results_manypopulations[population] = movMean
            ci_manypopulations[population] = confInt

            plt.plot(results_manypopulations.index[:], movMean, linewidth=linewidth, alpha=0.8, label=population_forpaper, color=color_mapping_longtitle[majority_title])
            plt.fill_between(results_manypopulations.index[:], movMean-confInt, movMean+confInt, linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  
    
        #plotting the moving average from stored data
        movingaverage_option = f'\n (moving avg., window {movingN})'
        #linewidth = 0.8 #overwrite the linewidth with a larger value than above 

        plt.plot(results_manypopulations.index[:], results_manypopulations[population], linewidth=linewidth, alpha=0.8, label=population_forpaper, color=color_mapping_longtitle[majority_title])
        plt.fill_between(results_manypopulations.index[:], results_manypopulations[population]-ci_manypopulations[population], results_manypopulations[population]+ci_manypopulations[population], linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  
    
    # change the line width for the legend
    leg = plt.legend(title='Population', loc='right', bbox_to_anchor=(1.70, 0.5), fontsize="21")
    for line in leg.get_lines():
        line.set_linewidth(5.0)

    plt.title(f'Cooperation by all player types \n across {len(POPULATIONS)} populations ') 
    plt.gca().set_ylim([-5, 105])
    plt.ylabel(r'% All Players Cooperating'+ '\n (mean per episode) '
                f'\n (mean over 20 runs '+ r'+- CI)' +movingaverage_option) #fontsize=25
    plt.xlabel('Episode')

    if not os.path.isdir('outcome_plots'):
        os.makedirs('outcome_plots')

    plt.savefig('outcome_plots/episode_cooperation_ALLpopulations.pdf', bbox_inches='tight')
    plt.savefig('outcome_plots/episode_cooperation_ALLpopulations.png', bbox_inches='tight')



def manypopulations_calculate_outcomes_allplayers_avgacrossruns(POPULATIONS, num_iter):
    '''For each outcome type (collective reward, collective reward cumulative, ginin reward, min reward):
    calculate outcomes for each population (average across episodes) and store them in a dataframe in a folder outcome_data.'''
    results_manypopulations_collective_reward = pd.DataFrame(index=range(num_iter))
    results_manypopulations_collective_reward_cumulative = pd.DataFrame(index=range(num_iter))
    results_manypopulations_gini_reward = pd.DataFrame(index=range(num_iter))
    results_manypopulations_min_reward = pd.DataFrame(index=range(num_iter))

    ci_manypopulations_collective_reward = pd.DataFrame(index=range(num_iter))
    ci_manypopulations_collective_reward_cumulative = pd.DataFrame(index=range(num_iter))
    ci_manypopulations_gini_reward = pd.DataFrame(index=range(num_iter))
    ci_manypopulations_min_reward = pd.DataFrame(index=range(num_iter))

    #loop over each population and create store the mean and CI across runs for each outcome
    for destination_folder in POPULATIONS: 
        n_runs = len(glob.glob('results/'+destination_folder+'/history/*.csv'))
        population = destination_folder.replace('___iter30000_runs20', '') 
        print(f'processing {population} with {n_runs} runs')

        result_population_collective_reward = pd.DataFrame(index=range(num_iter)) 
        result_population_collective_reward_cumulative = pd.DataFrame(index=range(num_iter)) 
        result_population_gini_reward = pd.DataFrame(index=range(num_iter)) 
        result_population_min_reward = pd.DataFrame(index=range(num_iter)) 

        for run_idx in range(n_runs):
            #print(f'run {run_idx}')
            run_idx += 1
            run_df = pd.read_csv(f'results/{destination_folder}/history/run{run_idx}.csv', index_col=0)[['episode', 'reward_collective', 'reward_gini', 'reward_min']]

            #group by episode 
            run_grouped = run_df.groupby('episode').agg({'reward_collective': 'sum', 'reward_gini':'mean', 'reward_min':'mean'})

            #EITHER add raw result to output df 
            #result_population_collective_reward[f'run{run_idx}'] = run_grouped['reward_collective']
            #result_population_collective_reward_cumulative[f'run{run_idx}'] = np.cumsum(run_grouped['reward_collective'])
            #result_population_gini_reward[f'run{run_idx}'] = run_grouped['reward_gini']
            #result_population_min_reward[f'run{run_idx}'] = run_grouped['reward_min']

            #OR calcualte moving average for each run before storing data 
            result_population_collective_reward[f'run{run_idx}'] = run_grouped['reward_collective'].rolling(window=movingN,center=True,min_periods=1).mean()
            result_population_collective_reward_cumulative[f'run{run_idx}'] = np.cumsum(run_grouped['reward_collective'])
            result_population_gini_reward[f'run{run_idx}'] = run_grouped['reward_gini'].rolling(window=movingN,center=True,min_periods=1).mean()
            result_population_min_reward[f'run{run_idx}'] = run_grouped['reward_min'].rolling(window=movingN,center=True,min_periods=1).mean()


        #save each outcome (mean & CI) for this population to the appropriate manypopulations df 
        means = result_population_collective_reward.mean(axis=1)
        sds = result_population_collective_reward.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)
        results_manypopulations_collective_reward[population] = means
        ci_manypopulations_collective_reward[population] = ci

        means = result_population_collective_reward_cumulative.mean(axis=1)
        sds = result_population_collective_reward_cumulative.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)
        results_manypopulations_collective_reward_cumulative[population] = means
        ci_manypopulations_collective_reward_cumulative[population] = ci

        means = result_population_gini_reward.mean(axis=1)
        sds = result_population_gini_reward.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)
        results_manypopulations_gini_reward[population] = means
        ci_manypopulations_gini_reward[population] = ci

        means = result_population_min_reward.mean(axis=1)
        sds = result_population_min_reward.std(axis=1)
        ci = 1.96 * sds/np.sqrt(n_runs)
        results_manypopulations_min_reward[population] = means
        ci_manypopulations_min_reward[population] = ci

    #store data to disc 
    print('storing data to outcome_data folder')
    if not os.path.isdir('outcome_data'):
        os.makedirs('outcome_data')

    results_manypopulations_collective_reward.to_csv('outcome_data/results_manypopulations_collective_reward.csv')
    results_manypopulations_collective_reward_cumulative.to_csv('outcome_data/results_manypopulations_collective_reward_cumulative.csv')
    results_manypopulations_gini_reward.to_csv('outcome_data/results_manypopulations_gini_reward.csv')
    results_manypopulations_min_reward.to_csv('outcome_data/results_manypopulations_min_reward.csv')

    ci_manypopulations_collective_reward.to_csv('outcome_data/ci_manypopulations_collective_reward.csv')
    ci_manypopulations_collective_reward_cumulative.to_csv('outcome_data/ci_manypopulations_collective_reward_cumulative.csv')
    ci_manypopulations_gini_reward.to_csv('outcome_data/ci_manypopulations_gini_reward.csv')    
    ci_manypopulations_min_reward.to_csv('outcome_data/ci_manypopulations_min_reward.csv')

def manypopulations_plot_outcomes_allplayers_avgacrossruns(POPULATIONS, num_iter):
    '''explore the social outcomes over time - using data stores in outcome_data.  
    Plot all populations on one plot.'''
    

    #read in outcomes data
    results_manypopulations_collective_reward = pd.read_csv('outcome_data/results_manypopulations_collective_reward.csv', index_col=0)
    results_manypopulations_collective_reward_cumulative = pd.read_csv('outcome_data/results_manypopulations_collective_reward_cumulative.csv', index_col=0)
    results_manypopulations_gini_reward = pd.read_csv('outcome_data/results_manypopulations_gini_reward.csv', index_col=0)
    results_manypopulations_min_reward = pd.read_csv('outcome_data/results_manypopulations_min_reward.csv', index_col=0)

    ci_manypopulations_collective_reward = pd.read_csv('outcome_data/ci_manypopulations_collective_reward.csv', index_col=0)
    ci_manypopulations_collective_reward_cumulative = pd.read_csv('outcome_data/ci_manypopulations_collective_reward_cumulative.csv', index_col=0)
    ci_manypopulations_gini_reward = pd.read_csv('outcome_data/ci_manypopulations_gini_reward.csv', index_col=0)    
    ci_manypopulations_min_reward = pd.read_csv('outcome_data/ci_manypopulations_min_reward.csv', index_col=0)



    #now plot each outcome in turn 
    figsize = (6, 6)
    linewidth = 1 #0.06
    sns.set(font_scale=2.5)

    #do COLLECTIVE_REWARD first - non-cumulative 
    print('plotting collective reward - non-cumulative')
    plt.figure(dpi=80, figsize=figsize) 

    for destination_folder in POPULATIONS: 
        population = destination_folder.replace('___iter30000_runs20', '') 
        print(f'plotting population {population}')
        long_titles = get_titles_for_population(destination_folder)
        majority_title_short = max(set(long_titles), key = long_titles.count)
        majority_title = title_mapping[majority_title_short]

        population_forpaper = f'majority-{title_mapping_forpaper_long[majority_title]}'

        if False: 
            movingaverage_option = f'\n (moving average, window {movingN})'
            linewidth = 0.5

            movMean = results_manypopulations_collective_reward[population].rolling(window=movingN,center=True,min_periods=1).mean()
            #movStd = results_allplayersthistype['sd'].rolling(window=movingN,center=True,min_periods=1).std()
            movStd = results_manypopulations_collective_reward[population].rolling(window=movingN,center=True,min_periods=1).std()
            #TO DO understand if we can use ci_manypopulations_collective_reward here in calcualting the moving CI 
            # get mean +/- 1.96*std/sqrt(N)
            #confIntPos = movMean + 1.96 * movStd / np.sqrt(movingN)
            #confIntNeg = movMean - 1.96 * movStd / np.sqrt(movingN)
            confIntPos = movMean + 1.96 * movStd / np.sqrt(n_runs_global)
            confIntNeg = movMean - 1.96 * movStd / np.sqrt(n_runs_global)


            plt.plot(results_manypopulations_collective_reward.index[:], movMean, linewidth=linewidth, alpha=1, label=population_forpaper, color=color_mapping_longtitle[majority_title])
            plt.fill_between(results_manypopulations_collective_reward.index[:], confIntPos, confIntNeg, 
                            linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  

        movingaverage_option = f'\n (moving avg., window {movingN})'
        plt.plot(results_manypopulations_collective_reward.index[:], results_manypopulations_collective_reward[population], linewidth=linewidth, alpha=0.7, label=population_forpaper, color=color_mapping_longtitle[majority_title])
        plt.fill_between(results_manypopulations_collective_reward.index[:], 
                            results_manypopulations_collective_reward[population]-ci_manypopulations_collective_reward[population], 
                            results_manypopulations_collective_reward[population]+ci_manypopulations_collective_reward[population], 
                            linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  

    population_size_approx = 16

    #leg = plt.legend(loc='lower center', fontsize="12", bbox_to_anchor=(0.5, -.70))
    #for line in leg.get_lines():
    #    line.set_linewidth(4.0)
    plt.title(f'Collective reward \n across {len(POPULATIONS)} populations') 
    plt.gca().set_ylim([2*population_size_approx, 6*population_size_approx])
    plt.ylabel(f'Collective reward \n (sum per episode) \n (mean over 20 runs' + r' +- CI)' + movingaverage_option) #, fontsize=25
    plt.xlabel('Episode')

    if not os.path.isdir('outcome_plots'):
        os.makedirs('outcome_plots')

    plt.savefig(f'outcome_plots/episode_collective_R_ALLpopulations_{movingaverage_option}.pdf', bbox_inches='tight')
    plt.savefig(f'outcome_plots/episode_collective_R_ALLpopulations_{movingaverage_option}.png', bbox_inches='tight')



    #plot COLLECTIVE_REWARD cumulative 
    print('plotting collective reward - cumulative')
    plt.figure(dpi=80, figsize=figsize) 
    #linewidth = 1

    for destination_folder in POPULATIONS: 
        population = destination_folder.replace('___iter30000_runs20', '') 
        print(f'plotting population {population}')
        long_titles = get_titles_for_population(destination_folder)
        majority_title_short = max(set(long_titles), key = long_titles.count)
        majority_title = title_mapping[majority_title_short]

        population_forpaper = f'majority-{title_mapping_forpaper_long[majority_title]}'

        plt.plot(results_manypopulations_collective_reward_cumulative.index[:], results_manypopulations_collective_reward_cumulative[population], linewidth=linewidth, alpha=1, label=population_forpaper, color=color_mapping_longtitle[majority_title])
        plt.fill_between(results_manypopulations_collective_reward_cumulative.index[:], 
                         results_manypopulations_collective_reward_cumulative[population]-ci_manypopulations_collective_reward_cumulative[population], 
                         results_manypopulations_collective_reward_cumulative[population]+ci_manypopulations_collective_reward_cumulative[population], 
                         linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  
    

    leg = plt.legend(title='Population', loc='right', bbox_to_anchor=(1.70, 0.5), fontsize="21")
    for line in leg.get_lines():
        line.set_linewidth(4.5)
    plt.title(f'Collective reward, cumulative \n across {len(POPULATIONS)} populations') 
    plt.ylabel(f'Collective reward \n (sum per episode), \n cumulative'+ f'\n (mean over 20 runs' + r' +- CI)')
    plt.xlabel('Episode')

    plt.savefig('outcome_plots/episode_collective_R_cumulative__ALLpopulations.pdf', bbox_inches='tight')


    #plot GINI_REWARD 
    print('plotting gini reward')
    plt.figure(dpi=80, figsize=figsize) 
    #linewidth = 0.06

    for destination_folder in POPULATIONS: 
        population = destination_folder.replace('___iter30000_runs20', '') 
        print(f'plotting population {population}')
        long_titles = get_titles_for_population(destination_folder)
        majority_title_short = max(set(long_titles), key = long_titles.count)
        majority_title = title_mapping[majority_title_short]

        population_forpaper = f'majority-{title_mapping_forpaper_long[majority_title]}'

        movingaverage_option = f'\n (moving avg., window {movingN})'
        plt.plot(results_manypopulations_gini_reward.index[:], results_manypopulations_gini_reward[population], linewidth=linewidth, alpha=0.7, label=population_forpaper, color=color_mapping_longtitle[majority_title])
        plt.fill_between(results_manypopulations_gini_reward.index[:], 
                         results_manypopulations_gini_reward[population]-ci_manypopulations_gini_reward[population], 
                         results_manypopulations_gini_reward[population]+ci_manypopulations_gini_reward[population], 
                         linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  
    

    #leg = plt.legend(loc='lower center', fontsize="12", bbox_to_anchor=(0.5, -.70))
    #for line in leg.get_lines():
    #    line.set_linewidth(4.0)
    plt.title(f'Gini reward \n across {len(POPULATIONS)} populations') 
    plt.gca().set_ylim([0, 1])
    plt.ylabel(f'Gini reward \n (mean per episode) \n (mean over 20 runs' + r' +- CI)' + movingaverage_option)
    plt.xlabel('Episode')

    plt.savefig(f'outcome_plots/episode_gini_R_ALLpopulations_{movingaverage_option}.pdf', bbox_inches='tight')
    plt.savefig(f'outcome_plots/episode_gini_R_ALLpopulations_{movingaverage_option}.png', bbox_inches='tight')


    #plot MIN_REWARD 
    print('plotting min reward')
    plt.figure(dpi=80, figsize=figsize) 
    #linewidth = 0.06

    for destination_folder in POPULATIONS: 
        population = destination_folder.replace('___iter30000_runs20', '') 
        print(f'plotting population {population}')
        long_titles = get_titles_for_population(destination_folder)
        majority_title_short = max(set(long_titles), key = long_titles.count)
        majority_title = title_mapping[majority_title_short]

        population_forpaper = f'majority-{title_mapping_forpaper_long[majority_title]}'

        movingaverage_option = f'\n (moving avg., window {movingN})'
        plt.plot(results_manypopulations_min_reward.index[:], results_manypopulations_min_reward[population], linewidth=linewidth, alpha=0.7, label=population_forpaper, color=color_mapping_longtitle[majority_title])
        plt.fill_between(results_manypopulations_min_reward.index[:], 
                         results_manypopulations_min_reward[population]-ci_manypopulations_min_reward[population], 
                         results_manypopulations_min_reward[population]+ci_manypopulations_min_reward[population], 
                         linewidth=linewidth, alpha=0.2, color=color_mapping_longtitle[majority_title])  
    

    #leg = plt.legend(title='Population', loc='right', bbox_to_anchor=(1.70, 0.5), fontsize="21")
    #for line in leg.get_lines():
    #    line.set_linewidth(4.0)
    plt.title(f'Min reward \n across {len(POPULATIONS)} populations') 
    plt.gca().set_ylim([0, 3])
    plt.ylabel(f'Min reward \n (mean per episode) \n (mean across 20 runs' + r' +- CI)' + movingaverage_option)
    plt.xlabel('Episode')

    plt.savefig(f'outcome_plots/episode_min_R_ALLpopulations_{movingaverage_option}.pdf', bbox_inches='tight')
    plt.savefig(f'outcome_plots/episode_min_R_ALLpopulations_{movingaverage_option}.png', bbox_inches='tight')

def manypopulations_plot_outcomes_legend(POPULATIONS, num_iter): 

    results_manypopulations_min_reward = pd.read_csv('outcome_data/results_manypopulations_min_reward.csv', index_col=0)

    #now plot each outcome in turn 
    figsize = (12, 6)
    linewidth = 1 #0.06
    sns.set(font_scale=2.5)

    #plot legend as though for the MIN_REWARD plot (it is the same as for other plots)
    print('plotting min reward')
    plt.figure(dpi=80, figsize=figsize) 
    #plt.figure()
    #linewidth = 0.06
    #plt.plot([], [], ' ', label=" ")

    for destination_folder in POPULATIONS: 
        population = destination_folder.replace('___iter30000_runs20', '') 
        print(f'plotting population {population}')
        long_titles = get_titles_for_population(destination_folder)
        majority_title_short = max(set(long_titles), key = long_titles.count)
        majority_title = title_mapping[majority_title_short]

        population_forpaper = f'majority-{title_mapping_forpaper_long[majority_title]}'

        movingaverage_option = f'\n (moving avg., window {movingN})'
        plt.plot(results_manypopulations_min_reward.index[:], results_manypopulations_min_reward[population], linewidth=linewidth, alpha=0.7, label=population_forpaper, color=color_mapping_longtitle[majority_title])
    

    leg = plt.legend(title='Population', loc='lower center', bbox_to_anchor=(0, -1), ncol=9, columnspacing=0.8, fontsize="21") #prop = { "size": 30 }, 
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    plt.rc('legend',fontsize='large')

    #plt.title(f'Min reward \n across {len(POPULATIONS)} populations') 
    #plt.gca().set_ylim([0, 3])
    #plt.ylabel(f'Min reward \n (mean per episode) \n (mean across 20 runs' + r' +- CI)' + movingaverage_option)
    #plt.xlabel('Episode')


    #NOTE need to manually crop the figure to only contian the legend 
    plt.savefig(f'outcome_plots/episode_outcomes_legend.pdf', bbox_inches='tight')
    plt.savefig(f'outcome_plots/episode_outcomes_legend.png', bbox_inches='tight')






def manypopulations_plot_heatmap_reward(POPULATIONS, num_iter, reward_type): 

    figsize=(8, 6.5)
    player_types = ['QL' + i for i in ['S','UT', 'DE', 'VE_e', 'VE_k', 'aUT', 'mDE', 'Vie', 'Vagg']]
    print('NOTE assuming all player types are possible in population!!')
    player_types_long = [title_mapping[title] for title in player_types]
    #populations = [f'majority {title}' for title in player_types_long]
    player_types_forpaper = [title_mapping_forpaper_short[title] for title in player_types]
    populations = [f'majority-{title}' for title in player_types_forpaper]

    matrix_rewards = pd.DataFrame(columns=populations, index=player_types_forpaper)

    for destination_folder in POPULATIONS: 

        titles = get_titles_for_population(destination_folder)

        population_size = len(titles)
        destination_folder = 'results/' + destination_folder

        long_titles = [title_mapping[title] for title in titles]

#NOTE this assumes only one player is the majority player in whole population!! 
        multiple_players_same_type_rewards = []
        majority_title = max(set(long_titles), key=long_titles.count)
        majority_title_forpaper = title_mapping_forpaper_long[majority_title]

        for idx in range(population_size):
            idx_title = long_titles[idx]
            idx_title_forpaper = title_mapping_forpaper_long[idx_title]
            #read in cumulative reward values and forward-fill missing iterations with last available value 
            reward_df = pd.read_csv(f'{destination_folder}/player_rewards/Rcumul_{reward_type}_{idx_title}_{idx}.csv', index_col=0).ffill()
            last_avail_reward = reward_df.mean(axis=1).iloc[-1]

            if long_titles.count(idx_title) > 1: #if this player occurs in the population more than once
                if not majority_title == idx_title:
                    print ('ERROR! there is more than one majority player type')
                    break 
                else: 
                    majority_title = idx_title
                multiple_players_same_type_rewards.append(last_avail_reward.astype(float))
            else: #if only one player of this type exists in the population 
                matrix_rewards.loc[idx_title_forpaper, f'majority-{majority_title_forpaper}'] = last_avail_reward.astype(float)

        #fill in value for the majority player - average across multiple players 
        multiple_players_avg_reward = mean(multiple_players_same_type_rewards)
        matrix_rewards.loc[majority_title_forpaper, f'majority-{majority_title_forpaper}'] = multiple_players_avg_reward.astype(float)

    #if plotting intrinsic reward_type, drop the Selfish player 
    if reward_type == 'intr':
        matrix_rewards.drop('S', inplace=True)
        player_types_forpaper.remove('S')

        #normalise intr rewards to be between the min and max value for eahc type of player 
        matrix_rewards_T = matrix_rewards.T
        #normalized_df = (matrix_rewards_T-matrix_rewards_T.min()) / (matrix_rewards_T.max()-matrix_rewards_T.min())
        #normalized_df = (matrix_rewards_T-0) / (matrix_rewards_T.max()-0)
        normalized_df = pd.DataFrame(index=matrix_rewards_T.index, columns=matrix_rewards_T.columns)
        for column in matrix_rewards_T.columns:
            if column in ['De', 'aUt']:
                positive_column = matrix_rewards_T[column] - min(matrix_rewards_T[column])
                #normalized_df[column] = (positive_column - 0) / (positive_column.max() - 0)
                normalized_df[column] = (positive_column - positive_column.min()) / (positive_column.max() - positive_column.min())

            elif column in ['Ut', 'V-Eq', 'V-Ki', 'mDe', 'V-In', 'V-Ag']:
                #normalized_df[column] = (matrix_rewards_T[column] - 0) / (matrix_rewards_T[column].max() - 0)
                normalized_df[column] = (matrix_rewards_T[column] - matrix_rewards_T[column].min()) / (matrix_rewards_T[column].max() - matrix_rewards_T[column].min())

        matrix_rewards_toplot = normalized_df.T
        #normalization_option = ' 0-max normalized'
        normalization_option = ' min-max normalized'

        vmax = 1
        cmap = 'pink_r' #'cividis_r' #'plasma_r' #'viridis_r' #magma_r
        reward_type = 'Intrinsic'

    else: #if plotting game reward
        vmax = matrix_rewards.max(axis=None).max()
        normalization_option = ''
        matrix_rewards_toplot = matrix_rewards
        cmap = 'bone_r' #'winter_r' #'plasma_r' #'viridis_r'
    
    for label in matrix_rewards_toplot.columns:
        matrix_rewards_toplot[label] = matrix_rewards_toplot[label].astype(float)

    plt.figure(figsize=figsize)
    sns.set(font_scale=2.8)
    s = sns.heatmap(matrix_rewards_toplot, cmap=cmap, xticklabels=populations, yticklabels=player_types_forpaper,
                    annot=False, linecolor='black', linewidths=0, square=True, vmin = 0, vmax = vmax)
    s.set(xlabel='Population', ylabel='Player Type',title=f'{reward_type.capitalize()} Rewards \n (accumulated over all episodes) \n {normalization_option}')
    #manually set limits to top and bottom of heatmap do not get cut off 
    bottom, top = s.get_ylim()
    s.set_ylim(bottom+0.5, top-0.5)

    plt.savefig(f'outcome_plots/matrix_{reward_type}_rewards.pdf', bbox_inches='tight')
    matrix_rewards_toplot.to_csv(f'outcome_data/matrix_{reward_type}_rewards_{normalization_option}.csv')




def calculate_heatmap_selections(destination_folder, num_iter, num_episodes, episode_idxs):
    titles = get_titles_for_population(destination_folder)

    population_size = len(titles)
    destination_folder = 'results/' + destination_folder

    long_titles = [title_mapping[title] for title in titles]
    #palette = [color_mapping_longtitle[t] for t in long_titles]

    #if using title AND idx 
    #indices_str = [str(i) for i in list(range(population_size))] 
    #order = ['_'.join(z) for z in zip(long_titles, indices_str)]
    #results_allruns = pd.DataFrame(index=order) #long_titles)

    results_allruns = pd.DataFrame(index=set(long_titles))


#NOTE this assumes only one player is the majority player in whole population!! 
    #multiple_players_same_type_rewards = []
    majority_title_long = max(set(long_titles), key=long_titles.count)
    majority_title = title_mapping_forpaper_long[majority_title_long]


    for run in range(1,n_runs_global+1):
        #FROM EARLIER FUNCTION 
        df = pd.read_csv(f'{destination_folder}/history/run{run}.csv', index_col=0)[['episode', 'title_p2', 'idx_p2']]
        small_df = df.loc[ df['episode'].isin(episode_idxs) ]
            
        results_allruns[f'run{run}'] = small_df['title_p2'].value_counts()

        #if using title AND idx
        #small_df['title_idx_p2'] = small_df['title_p2'] + '_' + small_df['idx_p2'].astype(str)
        #results_allruns[f'run{run}'] = small_df['title_idx_p2'].value_counts()


    #now aggregate across runs 
    results_allruns_percenttimes_selected = results_allruns.mean(axis=1) / (population_size*num_episodes)
    ci_allruns_percenttimes_selected = ( 1.96 * results_allruns.std(axis=1)/np.sqrt(n_runs_global) ) / (population_size*num_episodes)

    results_allruns_percenttimes_selected.index = [title_mapping_forpaper_long[i] for i in results_allruns_percenttimes_selected.index]
    ci_allruns_percenttimes_selected.index = [title_mapping_forpaper_long[i] for i in ci_allruns_percenttimes_selected.index]

    return majority_title, results_allruns_percenttimes_selected, ci_allruns_percenttimes_selected

def manypopulations_calculate_heatmap_selections(POPULATIONS, num_iter, episode_range=[29900, 30000]): #TO FINISH !!! 

    player_types = ['QL' + i for i in ['S','UT', 'DE', 'VE_e', 'VE_k', 'aUT', 'mDE', 'Vie', 'Vagg']]
    print('NOTE assuming all player types are possible in population!!')
    player_types_forpaper = [title_mapping_forpaper_short[title] for title in player_types]

    populations = [f'majority-{title}' for title in player_types_forpaper]

    matrix_selections = pd.DataFrame(columns=populations, index=player_types_forpaper)
    ci_selections = pd.DataFrame(columns=populations, index=player_types_forpaper)

    if episode_range == 'full':
        episode_idxs = list(range(0,num_iter))
        range_label = 'across full episode range'
    elif len(episode_range) == 1: 
        episode_idxs = [i-1 for i in episode_range]
        range_label = f'on episode {episode_idxs[0]}'
    elif len(episode_range) > 1:
        episode_idxs = list(range(episode_range[0],episode_range[1]))
        range_label = f'on episodes {episode_range[0]}-{episode_range[1]}'
        num_episodes = episode_range[1] - episode_range[0] 



    for destination_folder in POPULATIONS: 

        majority_title, results_allruns_percenttimes_selected, ci_allruns_percenttimes_selected = calculate_heatmap_selections(destination_folder, num_iter, num_episodes, episode_idxs) 
    
        #append to results dataframe 
        matrix_selections[f'majority-{majority_title}'] = results_allruns_percenttimes_selected.astype(float)
        ci_selections[f'majority-{majority_title}'] = ci_allruns_percenttimes_selected.astype(float)


    #save selections matrix to csv
    matrix_selections.to_csv(f'outcome_data/matrix_selections_episodes{episode_range}.csv')
    ci_selections.to_csv(f'outcome_data/ci_selections_episodes{episode_range}.csv')

#def plot_heatmap_selections():


def manypopulations_plot_heatmap_selections(POPULATIONS, num_iter, episode_range=[29900, 30000]):
    '''plot heatmap and more detailed bar plot of number of times each player type is selected on the final 100 episdoes in each populaiton'''
    figsize=(6, 6)

    matrix_selections = pd.read_csv(f'outcome_data/matrix_selections_episodes{episode_range}.csv', index_col=0)
    matrix_selections_percent = matrix_selections*100 

    ci_selections = pd.read_csv(f'outcome_data/ci_selections_episodes{episode_range}.csv', index_col=0)
    ci_selections_percent = ci_selections*100 

    player_types = ['QL' + i for i in ['S','UT', 'DE', 'VE_e', 'VE_k', 'aUT', 'mDE', 'Vie', 'Vagg']]
    player_types_long = [title_mapping[title] for title in player_types]

    print('NOTE assuming all player types are possible in population!!')
    player_types_forpaper = [title_mapping_forpaper_short[title] for title in player_types]
    populations = [f'majority-{title}' for title in player_types_forpaper]


    #plot selections matrix
    plt.figure(figsize=figsize)
    sns.set(font_scale=2)
    s = sns.heatmap(matrix_selections_percent, cmap='gist_yarg', 
                    #viridis, seismic, coolwarm, cool, hot, PiYG, summer, PuOr, PRGn, Blues, Greens, Purples, PuBuGn, YlGnBu, BuGn, gist_yarg, RdPu, PuRd
                     xticklabels=populations, yticklabels=player_types_forpaper, 
                    annot=False, linecolor='black', linewidths=0, square=True, vmin = 0, vmax = 100)
    s.set(xlabel='Population', ylabel='Player Type',
          title=f'Popularity of each player type \n (% times selected) \n (over last 100 episodes)')
    #manually set limits to top and bottom of heatmap do not get cut off 
    bottom, top = s.get_ylim()
    s.set_ylim(bottom+0.5, top-0.5)

    plt.savefig('outcome_plots/matrix_selections.pdf', bbox_inches='tight')


#def manypopulations_plot_bar_selections(POPULATIONS, num_iter, episode_range=[29900, 30000]):
#TO DO add calculation where we treat each player separately 

    #plot selections as bar chart 
    palette = [color_mapping_longtitle[t] for t in player_types_long]

    sns.set(font_scale=2)
    #s = matrix_selections_percent.T.plot(kind='bar', stacked=False, color=palette, width=0.8, figsize=(15, 6), edgecolor='grey')
    s = matrix_selections_percent.T.plot.bar(stacked=False, yerr=ci_selections_percent.T, color=palette, width=0.8, figsize=(15, 6), edgecolor='grey')
    #s = plt.errorbar(y=matrix_selections_percent.T, yerr=ci_selections_percent.T)
    # horizontal line indicating the threshold
    s.set(xlabel='Population', ylabel='% times selected \n (mean over last 100 episodes)',
          title=f'Popularity of each player type \n (over last 100 episodes)')
    leg = plt.legend(title='Player', loc='right', bbox_to_anchor=(1.2, 0.5)) #, fontsize="21"
    s.axhline(y=50, color='grey', linestyle='--') #label = '50% threshold'
    s.annotate('majority player constitutes \n '+r'50% of populaiton', xy=(7, 53), ha="center", va="top", fontsize=17)


    plt.savefig('outcome_plots/bar_selections.pdf', bbox_inches='tight')



def manypopulations_calculate_network_selections(POPULATIONS, num_iter, episode_range=[0, 30000]):
    '''create a matrix of when each player type selects each other player type - count number of iterations across episode_range. 
    Then average across all runs '''

    player_types = ['QL' + i for i in ['S','UT', 'DE', 'VE_e', 'VE_k', 'aUT', 'mDE', 'Vie', 'Vagg']]
    print('NOTE assuming all player types are possible in population!!')
    player_types_forpaper = [title_mapping_forpaper_short[title] for title in player_types]

    populations = [f'majority {title}' for title in player_types_forpaper]

    if episode_range == 'full':
        episode_idxs = list(range(0,num_iter))
        range_label = 'across full episode range'
    elif len(episode_range) == 1: 
        episode_idxs = [i-1 for i in episode_range]
        range_label = f'on episode {episode_idxs[0]}'
    elif len(episode_range) > 1:
        episode_idxs = list(range(episode_range[0],episode_range[1]))
        range_label = f'on episodes {episode_range[0]}-{episode_range[1]}'
        num_episodes = episode_range[1] - episode_range[0] 

    for destination_folder in POPULATIONS:

        titles = get_titles_for_population(destination_folder)
        population_size = len(titles)
        destination_folder = 'results/' + destination_folder
        long_titles = [title_mapping[title] for title in titles]
        print('assuming only one player is the majority player in whole population!!')
        majority_title_long = max(set(long_titles), key=long_titles.count)
        majority_title = title_mapping_forpaper_long[majority_title_long]
        population_tile_forpaper = f'majority {majority_title}'

        #loop over eveyr run 
        #network_selections = pd.DataFrame(columns=player_types_forpaper, index=player_types_forpaper)
        network_selections_everyrun = dict.fromkeys(list(range(1, n_runs_global+1)))

        for run in range(1,n_runs_global+1):
            #FROM EARLIER FUNCTION 
            df = pd.read_csv(f'{destination_folder}/history/run{run}.csv', index_col=0)[['episode', 'title_p1', 'title_p2']]
            small_df = df.loc[ df['episode'].isin(episode_idxs) ]
            small_df.drop('episode', axis='columns', inplace=True)

            small_df["value"]=1
            #pd.pivot_table(small_df, values="value", index=["title_p1"], columns="title_p2", fill_value=0) 
            #small_df.pivot_table(index=["title_p1"], columns="title_p2", aggfunc=lambda x: 1, fill_value=0)
            network_selections_matrix = pd.crosstab(small_df['title_p1'], small_df['title_p2'].fillna(0))
            network_selections_matrix.rename(index=title_mapping_forpaper_long, columns=title_mapping_forpaper_long, inplace=True)

            network_selections_everyrun[run] = network_selections_matrix
            
        #average numbers across all runs
        network_selections_mean = pd.concat(network_selections_everyrun.values()).reset_index().groupby("title_p1").mean()

        #store results in this population's results folder 
        network_selections_mean.to_csv(f'{destination_folder}/network_selections_mean_episodes{episode_range}.csv')



def plot_network_selections(destination_folder, num_iter, episode_range=[0, 30000]): 
    '''plot selections matrix and network graph per player TYPE (aggregarte multiple players of the same time 
    - will be skewed by one player beong predominant)'''
    titles = get_titles_for_population(destination_folder.replace('results/',''))
    if not 'results' in destination_folder: 
        destination_folder = 'results/' + destination_folder
    long_titles = [title_mapping[title] for title in titles]
    print('assuming only one player is the majority player in whole population!!')
    majority_title_long = max(set(long_titles), key=long_titles.count)
    majority_title = title_mapping_forpaper_long[majority_title_long]
    population_tile_forpaper = f'majority-{majority_title}'

    player_types_ordered = ['S', 'Ut', 'De', 'V-Eq', 'V-Ki', 'aUt', 'mDe', 'V-In', 'V-Ag']


    network_selections_mean = pd.read_csv(f'{destination_folder}/network_selections_mean_episodes{episode_range}.csv', index_col=0)
    #add a threshold so the network is not too populated - only keep edges with weights > 10000
    network_selections_mean = network_selections_mean.mask(network_selections_mean < 10000)
    #network_selections_mean.mask(network_selections_mean < network_selections_mean.quantile())

    #change order in the index 
    network_selections_mean_forplotting = network_selections_mean.reindex(index=player_types_ordered, columns=player_types_ordered)
    
    

    # 1 - plot selections matrix as heatmap 
    plt.clf()
    plt.figure(figsize=(6,6))
    sns.set(font_scale=2)
    s = sns.heatmap(network_selections_mean_forplotting, cmap='gist_yarg', 
                    #viridis, seismic, coolwarm, cool, hot, PiYG, summer, PuOr, PRGn, Blues, Greens, Purples, PuBuGn, YlGnBu, BuGn, gist_yarg, RdPu, PuRd
                    xticklabels=player_types_ordered, yticklabels=player_types_ordered, 
                    annot=False, linecolor='black', linewidths=0, square=True, 
                    vmin = 0, vmax = network_selections_mean_forplotting.max(axis=None).max())
    s.set(ylabel='Player 1 (selectOR) Type', xlabel='Player 2 (selectED) Type',
          title=f'Selection dynamics across player types \n in population {population_tile_forpaper} \n (# times selected) \n (over episodes {episode_range[0]}-{episode_range[1]})')
    #manually set limits to top and bottom of heatmap do not get cut off 
    bottom, top = s.get_ylim()
    s.set_ylim(bottom+0.5, top-0.5)

    plt.savefig(f'outcome_plots/matrix_selections_byplayertype_popul{population_tile_forpaper}.pdf', bbox_inches='tight')


    # 2 - plot selctions as network plot, circular layout 
    plt.clf()
    plt.figure(figsize=(4,3))
    temp = network_selections_mean_forplotting.replace(0, np.nan).astype('float')
    #min-max normalise 
    network_selections_mean_fornetworkplotting = temp.sub(temp.min().min()).div((temp.max().max()-temp.min().min()))
    network_selections_mean_fornetworkplotting = network_selections_mean_fornetworkplotting.mul(10)

    G=nx.from_pandas_adjacency(network_selections_mean_fornetworkplotting.astype('float'), create_using=nx.DiGraph())
    #G = nx.DiGraph(network_selections_mean_fornetworkplotting.astype('float'))

    widths = nx.get_edge_attributes(G, 'weight')
    node_colors = [color_mapping_longtitle[title_mapping_forpaper_shorttolong[node]] for node in G.nodes()]
    #elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 10000]
    #esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 10000]

    edges = {k: v for k, v in widths.items() if not isnan(v)}
    #pos = nx.spring_layout(G, k=5.95, seed=7)  # positions for all nodes - seed for reproducibility
    pos = nx.circular_layout(G) #, scale=5, center=('S', 'DE')  # positions for all nodes - seed for reproducibility
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=list(edges.keys()), width=list(edges.values()), alpha=0.5, 
                           edge_color="black", style="solid", arrows=True, arrowstyle='-|>', connectionstyle="arc3,rad=0.1", node_size=500)
    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="serif")
    plt.title(f'Selections \n in population {population_tile_forpaper}')
    plt.savefig(f'outcome_plots/network_selections_byplayertype_popul{population_tile_forpaper}.pdf', bbox_inches='tight')

#for changing the size of self-loops in network plots 
#https://stackoverflow.com/questions/74350464/how-to-better-visualize-networkx-self-loop-plot

def calculate_network_selection_bipartite(destination_folder, episode_idxs, episode_range, num_episodes, reorder_index=False):

    titles = get_titles_for_population(destination_folder)
    population_size = len(titles)
    destination_folder = 'results/' + destination_folder
    long_titles = [title_mapping[title] for title in titles]
    print('for the chart label, assuming only one player is the majority player in whole population!!')
    majority_title_long = max(set(long_titles), key=long_titles.count)
    majority_title = title_mapping_forpaper_long[majority_title_long]
    population_tile_forpaper = f'majority {majority_title}'
    print(f'running for populaiton {population_tile_forpaper}')

    #loop over eveyr run 
    #network_selections = pd.DataFrame(columns=player_types_forpaper, index=player_types_forpaper)
    network_selections_everyrun = dict.fromkeys(list(range(1, n_runs_global+1)))

    for run in range(1,n_runs_global+1):
        #FROM EARLIER FUNCTION 
        df = pd.read_csv(f'{destination_folder}/history/run{run}.csv', index_col=0)[['episode', 'idx_p1', 'title_p1', 'idx_p2', 'title_p2']]
        small_df = df.loc[ df['episode'].isin(episode_idxs) ]
        small_df.drop('episode', axis='columns', inplace=True)
        #replace long player title wiht short 
        small_df.replace({'title_p1': title_mapping_forpaper_long, 'title_p2': title_mapping_forpaper_long}, inplace=True)
        
        #reorder index so prosocial players appear first! 
        if reorder_index:
            relabel_dict = relabel_player_index(long_titles)
            small_df['idx_p1'] = small_df['idx_p1'].map(relabel_dict)
            small_df['idx_p2'] = small_df['idx_p2'].map(relabel_dict)

        #concatenate type and index for bipartite graph 
        small_df['title_idx_p1'] = small_df['idx_p1'].astype(str) + '_' + small_df['title_p1']
        small_df['title_idx_p2'] = small_df['idx_p2'].astype(str) + '_' + small_df['title_p2']

        #create unique ordered index for this population 
        labels_df = small_df[['idx_p1','title_p1']].value_counts()
        labels = ['{}_{}'.format(i, j) for i, j in labels_df.index]

        small_df["value"]=1
        network_selections_matrix_temp = pd.crosstab(small_df['title_idx_p1'], small_df['title_idx_p2'].fillna(0))
        #add columns for all labels, even those that were not present in the selections at all
        network_selections_matrix = pd.DataFrame(index=labels, columns=labels)
        #reorder columns and rows so they are in fixed order
        #network_selections_matrix = network_selections_matrix[labels]
        network_selections_matrix = network_selections_matrix.reindex(labels)
        #add crosstab values into results df with all possible columns 
        network_selections_matrix[network_selections_matrix_temp.columns] = network_selections_matrix_temp

        #store results for this run in big df 
        network_selections_everyrun[run] = network_selections_matrix
            
    #average numbers across all runs
    network_selections_mean = pd.concat(network_selections_everyrun.values()).reset_index().groupby("index").mean()
    #reorder index and columns in result 
    network_selections_mean_ordered = network_selections_mean.reindex(index=labels, columns=labels)
    #store results in this population's results folder 
    network_selections_mean_ordered.to_csv(f'{destination_folder}/network_selections_bipartite_mean_episodes{episode_range}.csv')




def manypopulations_calculate_network_selection_bipartite(POPULATIONS, num_iter, episode_range=[0, 30000]):
    '''create a matrix of when each player type selects each other player type - count number of iterations across episode_range. 
    Then average across all runs '''

    #player_types = ['QL' + i for i in ['S','UT', 'DE', 'VE_e', 'VE_k', 'aUT', 'mDE', 'Vie', 'Vagg']]
    #print('NOTE assuming all player types are possible in population!!')
    #player_types_forpaper = [title_mapping_forpaper_short[title] for title in player_types]

    #populations = [f'majority {title}' for title in player_types_forpaper]

    if episode_range == 'full':
        episode_idxs = list(range(0,num_iter))
        #range_label = 'across full episode range'
    elif len(episode_range) == 1: 
        episode_idxs = [i-1 for i in episode_range]
        #range_label = f'on episode {episode_idxs[0]}'
    elif len(episode_range) > 1:
        episode_idxs = list(range(episode_range[0],episode_range[1]))
        #range_label = f'on episodes {episode_range[0]}-{episode_range[1]}'
        num_episodes = episode_range[1] - episode_range[0] 

    for destination_folder in POPULATIONS:
        calculate_network_selection_bipartite(destination_folder, episode_idxs, episode_range, num_episodes, reorder_index=True)


def plot_network_selections_bipartite(destination_folder, num_iter, episode_range=[0, 30000], threshold=False, normalise=True, one_run=False): 
    '''plot selections matrix and network graph per INDIVIDUAL PLAYER (aggregarte multiple players of the same time 
    - will be skewed by one player beong predominant)'''

    titles = get_titles_for_population(destination_folder.replace('results/',''))
    if not 'results' in destination_folder: 
        destination_folder = 'results/' + destination_folder
    long_titles = [title_mapping[title] for title in titles]
    print('assuming only one player is the majority player in whole population!!')
    majority_title_long = max(set(long_titles), key=long_titles.count)
    majority_title = title_mapping_forpaper_long[majority_title_long]
    population_tile_forpaper = f'majority-{majority_title}'

    #player_types_ordered = ['S', 'Ut', 'De', 'V-Eq', 'V-Ki', 'aUt', 'mDe', 'V-In', 'V-Ag']
    if one_run:
        run_option = f'_run{one_run}'
    else:
        run_option=''

    network_selections_mean = pd.read_csv(f'{destination_folder}/network_selections_bipartite_mean_episodes{episode_range}{run_option}.csv', index_col=0)

    #if specified, add a threshold so the network is not too populated - only keep edges with weights > 10000
    if threshold: 
        if type(threshold) == bool: #use quantile if no numeric threshold specified 
            network_selections_mean = network_selections_mean.mask(network_selections_mean < network_selections_mean.quantile(q=0.85))
            threshold_option = f'\n threshold > 85th percentile'
        elif type(threshold) == int: 
            network_selections_mean = network_selections_mean.mask(network_selections_mean < threshold)
            threshold_option = f'\n threshold > {threshold} interactions'
        else: 
            print('threshold type not valid')
    else:
        #network_selections_mean = network_selections_mean
        threshold_option = ''

    
    # 1 - plot selections matrix as heatmap 
    plt.clf()
    plt.figure(figsize=(8,8))
    sns.set(font_scale=2)
    s = sns.heatmap(network_selections_mean, cmap='gist_yarg', 
                    #viridis, seismic, coolwarm, cool, hot, PiYG, summer, PuOr, PRGn, Blues, Greens, Purples, PuBuGn, YlGnBu, BuGn, gist_yarg, RdPu, PuRd
                    xticklabels=network_selections_mean.index, yticklabels=network_selections_mean.columns, 
                    annot=False, linecolor='black', linewidths=0.003, square=True, 
                    vmin = 0, vmax = network_selections_mean.max(axis=None).max())
                    #vmin = 0, vmax = 60)

    s.set(ylabel='Player 1 (selectOR)', xlabel='Player 2 (selectED)',
          title=f'Selection dynamics across players \n in population {population_tile_forpaper} \n (# times selected) \n (over episodes {episode_range[0]+1}-{episode_range[1]}) {threshold_option}')
    #manually set limits to top and bottom of heatmap do not get cut off 
    bottom, top = s.get_ylim()
    s.set_ylim(bottom+0.5, top-0.5)

    plt.savefig(f'outcome_plots/matrix_selections_bipartite_byplayertype_popul{population_tile_forpaper}_threshold{threshold}.pdf', bbox_inches='tight')
    plt.show()

    # 2 - plot selections as network plot, circular layout 
    plt.clf()
    plt.figure(figsize=(7,6))
    temp = network_selections_mean.fillna(0).astype('float')     #temp = network_selections_mean.replace(0, np.nan).astype('float')
    if normalise: 
        #min-max normalise for the network plot to appear more neat 
        network_selections_mean_fornetworkplotting = temp.sub(temp.min().min()).div( (temp.max().max()-temp.min().min()) )
        network_selections_mean_fornetworkplotting = network_selections_mean_fornetworkplotting.mul(5)

        #max-normalise
        #network_selections_mean_fornetworkplotting = temp.div( temp.max().max() )
    else: 
        network_selections_mean_fornetworkplotting = temp

    G=nx.from_pandas_adjacency(network_selections_mean_fornetworkplotting.astype('float'), create_using=nx.DiGraph())
    #G = nx.DiGraph(network_selections_mean_fornetworkplotting.astype('float'))

    widths = nx.get_edge_attributes(G, 'weight')
    #node_colors = ['grey' for node in G.nodes()]
    player_types = [node.split('_')[1] for node in G.nodes()]
    node_colors = [color_mapping_longtitle[title_mapping_forpaper_shorttolong[player]] for player in player_types]
    #node_colors = [color_mapping_longtitle[title_mapping_forpaper_shorttolong[node]] for node in G.nodes()]
    
    edges = {k: v for k, v in widths.items() if not isnan(v)}
    pos = nx.spring_layout(G, k=1, seed=15) #seed 5  # positions for all nodes - seed for reproducibility
    #pos = nx.circular_layout(G) #, scale=5, center=('S', 'DE')  # positions for all nodes - seed for reproducibility
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color=node_colors)
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=list(edges.keys()), width=list(edges.values()), alpha=0.5, 
                           edge_color="black", style="solid", arrows=True, arrowstyle='-|>', connectionstyle="arc3,rad=0.1", node_size=1200)
    # node labels
    #nodenames = {n:f'{n.split('_')[0]} \n {n.split('_')[1]} \n' for n in G.nodes()}
    nodenames = {n:n.split('_')[0]+'\n'+n.split('_')[1] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=nodenames, font_size=15, font_family="serif")

    plt.title(f'Selections \n in population {population_tile_forpaper} {threshold_option}')
    plt.savefig(f'outcome_plots/network_selections_bipartite_byplayertype_popul{population_tile_forpaper}_threshold{threshold}_normalise{normalise}{run_option}.pdf', bbox_inches='tight')
    plt.show()



def calculate_network_selection_bipartite_onerun(destination_folder, episode_idxs, num_episodes, run, reorder_index=False):

    titles = get_titles_for_population(destination_folder)
    population_size = len(titles)
    destination_folder = 'results/' + destination_folder
    long_titles = [title_mapping[title] for title in titles]
    print('for the chart label, assuming only one player is the majority player in whole population!!')
    majority_title_long = max(set(long_titles), key=long_titles.count)
    majority_title = title_mapping_forpaper_long[majority_title_long]
    population_tile_forpaper = f'majority {majority_title}'
    print(f'running for populaiton {population_tile_forpaper}')

    #loop over eveyr run 
    #network_selections = pd.DataFrame(columns=player_types_forpaper, index=player_types_forpaper)
    network_selections_everyrun = dict.fromkeys(list(range(1)))

    # for a single run: 
    df = pd.read_csv(f'{destination_folder}/history/run{run}.csv', index_col=0)[['episode', 'idx_p1', 'title_p1', 'idx_p2', 'title_p2']]
    small_df = df.loc[ df['episode'].isin(episode_idxs) ]
    small_df.drop('episode', axis='columns', inplace=True)
    #replace long player title wiht short 
    small_df.replace({'title_p1': title_mapping_forpaper_long, 'title_p2': title_mapping_forpaper_long}, inplace=True)
    #reorder index so prosocial players appear first! 
    if reorder_index:
        relabel_dict = relabel_player_index(long_titles)
        small_df['idx_p1'] = small_df['idx_p1'].map(relabel_dict)
        small_df['idx_p2'] = small_df['idx_p2'].map(relabel_dict)

    #concatenate type and index for bipartite graph 
    small_df['title_idx_p1'] = small_df['idx_p1'].astype(str) + '_' + small_df['title_p1']
    small_df['title_idx_p2'] = small_df['idx_p2'].astype(str) + '_' + small_df['title_p2']

    #create unique ordered index for this population 
    labels_series = small_df[['idx_p1','title_p1']].value_counts()
    labels = ['{}_{}'.format(i, j) for i, j in labels_series.index]

    small_df["value"]=1
    network_selections_matrix_temp = pd.crosstab(small_df['title_idx_p1'], small_df['title_idx_p2'].fillna(0))
    #add columns for all labels, even those that were not present in the selections at all
    network_selections_matrix = pd.DataFrame(index=labels, columns=labels)
    #reorder columns and rows so they are in fixed order
    #network_selections_matrix = network_selections_matrix[labels]
    network_selections_matrix = network_selections_matrix.reindex(labels)
    #add crosstab values into results df with all possible columns 
    network_selections_matrix[network_selections_matrix_temp.columns] = network_selections_matrix_temp

    #store results for this run in big df 
    network_selections_everyrun[run] = network_selections_matrix
            
    #average numbers across all runs
    network_selections_mean = pd.concat(network_selections_everyrun.values()).reset_index().groupby("index").mean()
    #reorder index and columns in result 
    network_selections_mean_ordered = network_selections_mean.reindex(index=labels, columns=labels)

    if False: 
        #if reorder_index:
        print('TO FINISH reorder_index')
        my_list = [i.split('_') for i in network_selections_mean_ordered.index]
        my_df = pd.DataFrame(my_list, columns=['idx_p1', 'title_p1'])

        #get target index in the desired order 
        new_list = reorder_selections_index(my_df)
        my_df.set_index('title_p1', inplace=True)

        
        new_idx_df = pd.DataFrame(new_list, columns=['title_p1'])
        new_idx_df['new_idx'] = new_idx_df.index.astype(str)
        #new_idx_df['new_idx_title'] = new_idx_df.index.astype(str) + '_' + new_idx_df['title_p1']
        new_idx_df.set_index('title_p1', inplace=True)

        final_idx_df = my_df.join(new_idx_df)
        #final_idx_df['index'] = final_idx_df['idx_p1'] + '_' + final_idx_df.index.astype(str)
        #final_idx_df.set_index('index', inplace=True)
        #final_idx_df.drop('idx_p1', axis=1, inplace=True)
        final_idx_df.set_index('idx_p1', inplace=True)
        final_idx_df.to_dict()

        #my_df.reindex(new_list)
        network_selections_mean_ordered = network_selections_mean_ordered.join(final_idx_df)

        #reorder original df 
        network_selections_mean_ordered.reindex(index=new_list, columns=new_list) 

    #store results in this population's results folder 
    network_selections_mean_ordered.to_csv(f'{destination_folder}/network_selections_bipartite_mean_episodes{episode_range}_run{run}.csv')

def relabel_player_index(long_titles):
    #my_list = [i.split('_') for i in network_selections_mean_ordered.index]
    #my_df = pd.DataFrame(my_list, columns=['idx_p1', 'title_p1'])

    #get target index in the desired order
    sort_order = ['S', 'Ut', 'De', 'V-Eq', 'V-Ki', 'aUt', 'mDe', 'V-In', 'V-Ag']
    original_titles = [title_mapping_forpaper_long[title] for title in long_titles]
    original_df = pd.DataFrame(original_titles, columns=['title_p1'])
    #original_df['title_p1_enum'] = lambda x: x.enumerate() original_df['title_p1']
    duplicated_titles = original_df[original_df['title_p1'].duplicated()]
    duplicate_single = duplicated_titles.iloc[0,0]
    #handle duplicate title in original_df
    original_df['title_p1_identifier'] = ''    
    original_df.loc[original_df['title_p1']==duplicate_single, 'title_p1_identifier'] = np.arange((original_df['title_p1']==duplicate_single).sum()) + 1
    original_df['title_p1_formatted'] = original_df['title_p1'] + original_df['title_p1_identifier'].astype(str)
    original_df.drop(['title_p1', 'title_p1_identifier'], axis=1, inplace=True)

    original_titles.sort(key = lambda i: sort_order.index(i)) # note this happens inplace 
    new_idx_df = pd.DataFrame(original_titles, columns=['title_p1'])
    new_idx_df['new_idx'] = new_idx_df.index.astype(str)
    #handle duplicate title in new_idx_df
    new_idx_df['title_p1_identifier'] = ''    
    new_idx_df.loc[new_idx_df['title_p1']==duplicate_single, 'title_p1_identifier'] = np.arange((new_idx_df['title_p1']==duplicate_single).sum()) + 1
    new_idx_df['title_p1_formatted'] = new_idx_df['title_p1'] + new_idx_df['title_p1_identifier'].astype(str)
    new_idx_df.drop(['title_p1', 'title_p1_identifier'], axis=1, inplace=True)

    new_idx_df.set_index('title_p1_formatted', inplace=True)

    #new_list = reorder_selections_index(set(small_df['title_p1']))
    final_df = original_df.reset_index(inplace=False).set_index('title_p1_formatted', inplace=False)
    final_df = final_df.join(new_idx_df) #this currently stores the original index 
    #final_df.set_index('index', inplace=True)
    final_dict = dict(zip(final_df['index'], final_df['new_idx'].astype(int))) 

    return final_dict

        






#######################
#### Setup ############
#######################


game_title = 'IPD'
record_Loss = 'both' 
record_Qvalues = 'both'



#### PART 1 = plotting for the full mixed populations: conseq+norm+selfish
#if plotting from local directory 
os.getcwd()
os.chdir('EPISODES_PartnerSelection/conseq+norm+selfish')

#if plotting form external hard drive
os.getcwd()
os.chdir("/Volumes/G-DRIVE mobile USB-C/Partner Selection/EPISODES_IPD_PartnerSelection/conseq+norm_selfish")
os.listdir()


############################################################################################
#### Vary the destination_folder variable as below for all the plots #######################
############################################################################################


destination_folder = '8xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20' 

destination_folder = '1xS_8xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20' 
destination_folder = '1xS_1xUT_8xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20' 

destination_folder = '1xS_1xUT_1xaUT_8xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20' 
destination_folder = '1xS_1xUT_1xaUT_1xDE_8xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20' 

destination_folder = '1xS_1xUT_1xaUT_1xDE_1xmDE_8xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20' 
destination_folder = '1xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_8xVie_1xVEk_1xVagg___iter30000_runs20' 

destination_folder = '1xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_8xVEk_1xVagg___iter30000_runs20' 
destination_folder = '1xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_8xVagg___iter30000_runs20' 




#############################################################################
#### preparation to run for every destination_folder  #######################
#############################################################################

num_iter = 30000

n_runs = len(glob.glob('results/'+destination_folder+'/history/*.csv'))
print('n_runs = ', n_runs)

titles = get_titles_for_population(destination_folder)

population_size = len(titles)
destination_folder = 'results/' + destination_folder

long_titles = [title_mapping[title] for title in titles]


#################################################################################
#### create formatted datasets for plotting #####################################
#################################################################################
reformat_a_for_population(destination_folder, n_runs, population_size, num_iter, long_titles) 

reformat_reward_for_population(destination_folder, n_runs, population_size, long_titles)


#################################################################################
#### functions to run for every destination_folder ##############################
#################################################################################

episode_plot_cooperative_selections_whereavailable(destination_folder, titles, n_runs)

#
episode_plot_cooperation_population_v2(destination_folder, titles, n_runs, num_iter, with_CI=True, reduced=True)
episode_plot_cooperation_perrun(destination_folder, titles, n_runs, num_iter) #each run separately 

#
episode_plot_action_pairs_population_new(destination_folder, titles, n_runs, option=None, combine_CDDC=False)


episode_plot_actions_eachplayerinpopulation(destination_folder, titles, n_runs, num_iter, population_size) 

#
plot_actions_aggbyplayertype_avgacrossruns(destination_folder, titles, n_runs, num_iter, population_size)

episode_plot_reward_eachplayerinpopulation()

#
episode_plot_collective_outcomes_population(destination_folder, titles, n_runs, num_iter, population_size)



#analyse how often each opponent is selected
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range='full') 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[500]) 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[2500]) 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[5000]) 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[7500]) 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[10000]) 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[20000]) 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[30000]) 
plot_all_opponents_selected(destination_folder, long_titles, population_size, n_runs, num_iter, episode_range=[29900,30000]) 





#################################################################################
#### plot ACROSS MANY POPULATIONS  ##############################################
#################################################################################

sns.set_style("darkgrid")
num_iter = 30000


POPULATIONS = ['8xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20', 
               '1xS_8xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20',
               '1xS_1xUT_8xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20',
               '1xS_1xUT_1xaUT_8xDE_1xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20',
               '1xS_1xUT_1xaUT_1xDE_8xmDE_1xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20',
               '1xS_1xUT_1xaUT_1xDE_1xmDE_8xVEe_1xVie_1xVEk_1xVagg___iter30000_runs20',
               '1xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_8xVie_1xVEk_1xVagg___iter30000_runs20',
               '1xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_8xVEk_1xVagg___iter30000_runs20',
               '1xS_1xUT_1xaUT_1xDE_1xmDE_1xVEe_1xVie_1xVEk_8xVagg___iter30000_runs20']



#actions - % cooperation over time
manypopulations_calculate_cooperation_allplayers_avgacrossruns(POPULATIONS, num_iter)
manypopulations_plot_cooperation_allplayers_avgacrossruns(POPULATIONS, num_iter)



#actions only by the Selfish player over time 
movingN=1000
manypopulations_plot_cooperation_aggbyplayertype_avgacrossruns(POPULATIONS, num_iter, title='Selfish')

#actions by each player type 
for player_type in set(long_titles):
    manypopulations_plot_cooperation_aggbyplayertype_avgacrossruns(POPULATIONS, num_iter, title=player_type)


#social outcomes over time
manypopulations_calculate_outcomes_allplayers_avgacrossruns(POPULATIONS, num_iter)
manypopulations_plot_outcomes_allplayers_avgacrossruns(POPULATIONS, num_iter)






#selections towards the end of trainig - which player is the most poppular? 
manypopulations_calculate_heatmap_selections(POPULATIONS, num_iter, episode_range=[29900, 30000])
manypopulations_plot_heatmap_selections(POPULATIONS, num_iter, episode_range=[29900, 30000])

#selection network among player types
manypopulations_calculate_network_selections(POPULATIONS, num_iter, episode_range=[0, 30000]) 
for destination_folder in POPULATIONS:
    plot_network_selections(destination_folder, num_iter, episode_range=[0, 30000])

#selection network among specific players -bipartite graph 
manypopulations_calculate_network_selection_bipartite(POPULATIONS, num_iter, episode_range=[0, 30000])
for destination_folder in POPULATIONS:
    plot_network_selections_bipartite(destination_folder, num_iter, episode_range=[0, 30000], threshold=False, normalise=True)
#plot the above with threshold=True for network plots, threshold=False for the heatmap plots
