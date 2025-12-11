import os
import random
import numpy as np
import pandas as pd

from get_summary import *
from orth_neighborhood_utils import *
from PredictiveCoding_Model import *
from stimulus_counterbalancing_script import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

random.seed(1)
np.random.seed(1)


def run_simulation(**kwargs):
    # run a simulation and keep only the components needed for plotting and data analysis (to avoid out-of-memory errors)
    full_simulation = Simulation(**kwargs)
    data,fname,sim_input_bottomup,sim_input_topdown = full_simulation.simulation_data, full_simulation.sim_filename, full_simulation.sim_input_bottomup, full_simulation.sim_input_topdown
    del full_simulation
    return {"simulation_data" : data,
             "sim_filename" : fname,
             "sim_input_bottomup": sim_input_bottomup,
             "sim_input_topdown": sim_input_topdown}
    # return full_simulation
def run_simulation_comp(**kwargs):
    # run a simulation and keep only the components needed for plotting and data analysis (to avoid out-of-memory errors)
    full_simulation = Simulation_comp(**kwargs)
    data,fname,sim_input_bottomup,sim_input_topdown = full_simulation.simulation_data, full_simulation.sim_filename, full_simulation.sim_input_bottomup, full_simulation.sim_input_topdown
    del full_simulation
    return {"simulation_data" : data,
             "sim_filename" : fname,
             "sim_input_bottomup": sim_input_bottomup,
             "sim_input_topdown": sim_input_topdown}
    # return full_simulation



lexicon = Lexicon()
NUM_ITERS = 20
THRESHOLD = 4.0

wordlist = np.array(lexicon.words)

##################### DEFINE STIMULI FOR ALL CONDITIONS ##################### 

## create stimuli that are counterbalanced
# quads = find_counterbalanced_quads(lexicon,min_sem_overlap = 1)
# stim_dict = create_stimulus_lists(quads, lexicon)
# verify_stimulus_lists(stim_dict, lexicon)
# np.save('./helper_txt_files/standard_stims_word_idx.npy', stim_dict['standard_idx'])
# np.save('./helper_txt_files/sem_related_word_idx_nonneighbor.npy', stim_dict['sem_related_idx'])
# np.save('./helper_txt_files/fully_unrelated_word_idx_1.npy', stim_dict['unrelated_idx'])


standard_stims_word_idx = np.load('./helper_txt_files/standard_stims_word_idx.npy')
fully_unrelated_word_idx_1 = np.load('./helper_txt_files/fully_unrelated_word_idx_1.npy')
sem_related_word_idx_nonneighbor = np.load('./helper_txt_files/sem_related_word_idx_nonneighbor.npy')

standard_stims = wordlist[standard_stims_word_idx]
unrelated_stims_1 = wordlist[fully_unrelated_word_idx_1]
semrelated_stims = wordlist[sem_related_word_idx_nonneighbor]

final_standard_idx = [list(lexicon.words).index(w) for w in standard_stims]
final_unrelated_idx_1 = [list(lexicon.words).index(w) for w in unrelated_stims_1]
final_semrelated_idx = [list(lexicon.words).index(w) for w in semrelated_stims]

shared_feats = lexicon.semfeatmatrix[:,final_standard_idx] * lexicon.semfeatmatrix[:,final_semrelated_idx] 
assert (lexicon.semfeatmatrix[:,final_unrelated_idx_1] * lexicon.semfeatmatrix[:,final_semrelated_idx]).sum() == 0

##################### LEXICAL EFFECT SIMULATIONS #####################
num_stims = 1400 #standard_stims.shape[0]
standard_simulation = run_simulation(sim_input_bottomup = standard_stims[:num_stims], clamp_iterations = NUM_ITERS, sim_filename = 'standard_simulation')

# ##################### CONTEXTUAL EFFECT SIMULATIONS #####################
# Run the semantic priming simulation
sem_priming = {'semunrelated': run_simulation(sim_input_bottomup = unrelated_stims_1[:num_stims], clamp_iterations =NUM_ITERS, blanks_before_clamp = 5, prevSim = standard_simulation, sim_filename = 'sem_priming_semunrelated5'),
                'semrelated': run_simulation(sim_input_bottomup = semrelated_stims[:num_stims], clamp_iterations =NUM_ITERS, blanks_before_clamp = 5, prevSim = standard_simulation, sim_filename = 'sem_priming_semrelated5')}

# Run the repetition priming simulation
rep_priming = {'unrepeated': run_simulation(sim_input_bottomup = unrelated_stims_1[:num_stims], clamp_iterations =NUM_ITERS, blanks_before_clamp = 5, prevSim = standard_simulation, sim_filename = 'rep_priming_unrepeated5'),
               'repeated' : run_simulation(sim_input_bottomup = standard_stims[:num_stims], clamp_iterations =NUM_ITERS, blanks_before_clamp = 5, prevSim = standard_simulation, sim_filename = 'rep_priming_repeated5')}


# Run the lexical predictability simulation
cloze_levels = {
    "low_cloze": 1/lexicon.size, 
                "med_low_cloze": 0.25,
                "med_high_cloze": 0.5,
                "high_cloze": 0.99}

cloze_simulations_preactivate = {}
cloze_simulations_bottomup = {}

for key, val in cloze_levels.items():
    # pre-activate each of the 512 standard inputs from the top down
    cloze_simulations_preactivate.update({key: run_simulation(sim_input_topdown = standard_stims[:num_stims], 
                                                              clamp_iterations =NUM_ITERS,
                                                              BU_TD_mode = "top_down", 
                                                              cloze = val, 
                                                              sim_filename = f'cloze_simulations_preact_{key}')})
    # present each of the 512 standard inputs from the bottom up
    cloze_simulations_bottomup.update({key: run_simulation(sim_input_bottomup = standard_stims[:num_stims], 
                                                           sim_input_topdown = standard_stims[:num_stims],
                                                           cloze = val,
                                                           clamp_iterations =NUM_ITERS,
                                                           BU_TD_mode = "bottom_up", 
                                                           prevSim = cloze_simulations_preactivate[key], 
                                                           sim_filename = f'cloze_simulations_bottomup{key}')})

# faster if the data is pre-saved:
# for key, val in cloze_levels.items():
#     # cloze_simulations_preactivate.update({key: run_simulation(sim_input = standard_stims, clamp_iterations =NUM_ITERS,BU_TD_mode = "top_down", cloze = val, sim_filename = f'cloze_simulations_preact_{key}')})
#     cloze_simulations_bottomup.update({key: run_simulation(sim_input = standard_stims, clamp_iterations =NUM_ITERS,BU_TD_mode = "bottom_up", sim_filename = f'cloze_simulations_bottomup{key}')})

# Run the lexical prediction violation simulation
lexical_violation = {"low_constraint_unexpected": run_simulation(sim_input_bottomup = unrelated_stims_1[:num_stims], 
                                                                 sim_input_topdown = standard_stims[:num_stims], 
                                                                 clamp_iterations =NUM_ITERS,
                                                                 BU_TD_mode = "bottom_up", 
                                                                 cloze = 1/lexicon.size,
                                                                 prevSim = cloze_simulations_preactivate["low_cloze"], 
                                                                 sim_filename = 'lexviol_LCunexp'), 
                    "high_constraint_unexpected": run_simulation(sim_input_bottomup = unrelated_stims_1[:num_stims], 
                                                                 sim_input_topdown = standard_stims[:num_stims], 
                                                                 clamp_iterations =NUM_ITERS,
                                                                 BU_TD_mode = "bottom_up", 
                                                                 cloze = 0.99,
                                                                 prevSim = cloze_simulations_preactivate["high_cloze"], 
                                                                 sim_filename = 'lexviol_HCunexp'),
                    "high_constraint_expected": run_simulation(sim_filename = f'cloze_simulations_bottomuphigh_cloze')
                    }

# Run the anticipatory semantic overlap simulation
semantic_prediction_overlap = {"semunrelated_99cloze": run_simulation(sim_input_bottomup = unrelated_stims_1[:num_stims], 
                                                                      sim_input_topdown = standard_stims[:num_stims], 
                                                                      BU_TD_mode = "bottom_up",
                                                                      cloze = 0.99,
                                                                      clamp_iterations =NUM_ITERS, 
                                                                      prevSim = cloze_simulations_preactivate["high_cloze"], 
                                                                      sim_filename = 'sempredoverlap_semunrelated_99cloze'),
                                "semrelated_99cloze": run_simulation(sim_input_bottomup = semrelated_stims[:num_stims], 
                                                                     sim_input_topdown = standard_stims[:num_stims], 
                                                                     BU_TD_mode = "bottom_up",
                                                                     cloze = 0.99,
                                                                     clamp_iterations =NUM_ITERS, 
                                                                     prevSim = cloze_simulations_preactivate["high_cloze"], 
                                                                     sim_filename = 'sempredoverlap_semrelated_99cloze')}
semantic_prediction_overlap.update({"high_constraint_expected": run_simulation(sim_filename = f'cloze_simulations_bottomuphigh_cloze')})

##################### PLOT DATA FROM ALL CONTEXTUAL SIMULATIONS #####################
all_simulations = [sem_priming, rep_priming, cloze_simulations_bottomup, lexical_violation, semantic_prediction_overlap]
all_simulations_names = ["sem_priming", "rep_priming", "cloze_simulations_bottomup", "lexical_violation", "semantic_prediction_overlap"]

def simulation_accuracy_info(the_simulation, num_iters_target_presentation=20):
    crossers = []
    for condition in the_simulation.keys():
        # retrieve the activity of the most active lexical state for each trial, ie target input, at all iterations
        most_active_state_activity_per_trial = the_simulation[condition]['simulation_data']['max_lex_state_activation'][0][:,-num_iters_target_presentation:] #(num_trials,num_iterations)
        # indicate whether the threshold was crossed for each trial
        threshold_was_crossed = np.any(most_active_state_activity_per_trial > THRESHOLD,axis = 1)
        # for each trial, find the iteration at which the threshold was crossed
        threshold_crossing_iteration_per_trial = np.argmax(most_active_state_activity_per_trial > THRESHOLD,axis = 1) [threshold_was_crossed]#(threshold_crossing_trials,20)
        # for each trial, find the identity of the lexical state that crossed the threshold
        most_active_state_identity_per_trial = the_simulation[condition]['simulation_data']['max_lex_state_identity'][0][threshold_was_crossed,-num_iters_target_presentation:]#(threshold_crossing_trials,20)
        identity_of_threshold_crosser = most_active_state_identity_per_trial[np.arange(np.sum(threshold_was_crossed)), threshold_crossing_iteration_per_trial] #(threshold_crossing_trials,)
        # retrieve the identity of the "correct" target that should have crossed the threshold
        target_identity = np.array([list(lexicon.words).index(w) for w in np.array(the_simulation[condition]['sim_input_bottomup'])[threshold_was_crossed]])
        # check how many of the lexical states that crossed the threshold matched the correct target state
        print(f'The number of threshold crossers in {the_simulation[condition]["sim_filename"]} that matched the identity was {np.sum(target_identity == identity_of_threshold_crosser)} out of {threshold_was_crossed.shape[0]}, {(np.sum(target_identity == identity_of_threshold_crosser)/threshold_was_crossed.shape[0])*100}%')
        crossers.append(target_identity == identity_of_threshold_crosser)
    return crossers



def get_threshold_crossing_info(the_simulation, condition, threshold,num_iters_target_presentation = NUM_ITERS, return_iteration_idx = False):

    # retrieve the correct, target identity
    target_identity = np.array([list(lexicon.words).index(w) for w in the_simulation[condition]['sim_input_bottomup']])
    # next, retrieve the activity of the most active lexical state for each trial, ie target input, at all iterations
    most_active_state_activity_per_trial = the_simulation[condition]['simulation_data']['max_lex_state_activation'][0][:,-num_iters_target_presentation:] #(num_trials,num_iterations)
    most_active_state_identity_per_trial = the_simulation[condition]['simulation_data']['max_lex_state_identity'][0][:,-num_iters_target_presentation:] #(num_trials,num_iterations)
    # initialize 3 vectors, indicating whether the correct item crossed the threshold and what the index-identity and word-identity of the threshold crosser was
    correct_item_identified = np.ones(the_simulation[condition]['sim_input_bottomup'].shape[0])*np.nan
    threshold_crossing_iteration_ = np.ones(the_simulation[condition]['sim_input_bottomup'].shape[0])*np.nan
    idx_identity_of_threshold_crosser = (np.ones(the_simulation[condition]['sim_input_bottomup'].shape[0])*np.nan).astype('int16')
    word_identity_of_threshold_crosser = (np.ones(the_simulation[condition]['sim_input_bottomup'].shape[0])*np.nan).astype('<U4')

    # indicate whether the threshold was crossed for each trial
    threshold_was_crossed = np.any(most_active_state_activity_per_trial > threshold,axis = 1)

    for trial_idx,word in enumerate(the_simulation[condition]['sim_input_bottomup']):
        if threshold_was_crossed[trial_idx]:
            threshold_crossing_iteration = np.argmax(most_active_state_activity_per_trial[trial_idx] > threshold)
            threshold_crossing_iteration_[trial_idx] = threshold_crossing_iteration
            idx_identity_of_threshold_crosser[trial_idx] = most_active_state_identity_per_trial[trial_idx, threshold_crossing_iteration] 
            correct_item_identified[trial_idx] = idx_identity_of_threshold_crosser[trial_idx] == target_identity[trial_idx]

    word_identity_of_threshold_crosser[~np.isnan(idx_identity_of_threshold_crosser)] = lexicon.words[idx_identity_of_threshold_crosser[~np.isnan(idx_identity_of_threshold_crosser)].astype('int32')]
    if return_iteration_idx:
        return threshold_was_crossed, correct_item_identified, word_identity_of_threshold_crosser, threshold_crossing_iteration_
    return threshold_was_crossed, correct_item_identified, word_identity_of_threshold_crosser

acc_dic = {}
for sim, simname in zip(all_simulations, all_simulations_names):
    acc_dic[simname] = {}
    for subsim_name in sim.keys():
        acc_array = []
        for a_threshold in np.linspace(0.1,5.0):
            a,b,c = get_threshold_crossing_info(sim,subsim_name, a_threshold)
            bb = np.nan_to_num(b,0)
            acc_array.append(bb.sum()/1400)
        acc_dic[simname].update({f'{subsim_name}' : acc_array})


tci_dic = {}
for sim, simname in zip(all_simulations, all_simulations_names):
    tci_dic[simname] = {}
    for subsim_name in sim.keys():
        threshold_crossing_iterations = []
        for a_threshold in np.linspace(0.1,4.0):
            boolean_threshold_crossing = sim[f'{subsim_name}']['simulation_data']['max_lex_state_activation'][0][:,-NUM_ITERS:] > a_threshold
            threshold_crossing_iterations.append(np.argmax(boolean_threshold_crossing, axis = 1))
        tci_dic[simname][subsim_name] = threshold_crossing_iterations


thresholds = np.linspace(0.1,4.0)
for simname, subsims in tci_dic.items():
    plt.figure(figsize=(12*1.3, 6*1.3))

    for subsim_name, arr_list in subsims.items():
        arr = np.array(arr_list)              # (n_thresholds, n_items)
        means = arr.mean(axis=1)
        sems  = arr.std(axis=1) / np.sqrt(arr.shape[1])

        plt.plot(thresholds, means, marker='o', label=subsim_name)
        plt.fill_between(
            thresholds,
            means - sems,
            means + sems,
            alpha=0.2
        )

    plt.title(simname)
    plt.xlabel("Threshold")
    plt.ylabel("Mean crossing iteration")
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_max_yval(all_simulations):
    max_yval_st = 0
    max_yval_err = 0
    for simulation in all_simulations:
        for sub_sim in simulation.keys():
            data_to_plot_st = np.mean(simulation[sub_sim]['simulation_data']['max_lex_state_activation'][0],axis = 0).T
            data_to_plot_err = np.mean(simulation[sub_sim]['simulation_data']['total_lexsem_err'][0],axis = 0).T
            if max(data_to_plot_st) >= max_yval_st:
                max_yval_st = max(data_to_plot_st)
            if max(data_to_plot_err) >= max_yval_err:
                max_yval_err = max(data_to_plot_err)
    return max_yval_st,max_yval_err

def top_state_means(filename, simulation, max_yval, ordered_conditions = [],condition_styles = ['r','k','b','g','r:','k:','b:','g:','r--','k--','b--','g--'],labels = [],legend_loc = 'lower right'):

    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(12*1.3, 6*1.3))
    main_ax = fig.add_subplot()
    main_ax.spines.top.set_visible(False)
    main_ax.spines.right.set_visible(False)
    leg_label = 'Most Active Lexical State'
    
    if ordered_conditions == []:
        ordered_conditions = list(simulation.keys())
    for i, (sub_sim, style, label) in enumerate(zip(ordered_conditions, condition_styles, labels)):
        num_trials = simulation[sub_sim]['simulation_data']['max_lex_state_activation'][0].shape[0]
        num_iters = simulation[sub_sim]['simulation_data']['max_lex_state_activation'][0].shape[-1]
        PRESTIMULUS_WINDOW = 21
        individual_stimulus = num_iters == NUM_ITERS +1
        if individual_stimulus:
            final_stim_duration = NUM_ITERS
            padding = PRESTIMULUS_WINDOW - 1
            full_window = num_iters + padding
            state_vals = np.block([np.ones((num_trials,padding))*0.001, simulation[sub_sim]['simulation_data']['max_lex_state_activation'][0]])
        else:
            final_stim_duration = num_iters - NUM_ITERS - 1
            final_stim_duration -= np.remainder(final_stim_duration , NUM_ITERS)
            full_window = final_stim_duration + PRESTIMULUS_WINDOW
            state_vals = simulation[sub_sim]['simulation_data']['max_lex_state_activation'][0]

        data_to_plot = np.mean(state_vals,axis = 0).T

        main_ax.plot(np.arange(-PRESTIMULUS_WINDOW + 1, final_stim_duration+0.03, 1.0), data_to_plot[-full_window:],style,label = label)
        main_ax.set_ylabel(leg_label)
        main_ax.set_xlabel('Iterations')
        main_ax.set_xticks(np.arange(-PRESTIMULUS_WINDOW+1, final_stim_duration+0.03, 5.0))
        main_ax.set_yticks(np.arange(0, max_yval+ 0.3, 0.5))
        main_ax.set_ylim(-0.3, max_yval+0.3)

    main_ax.legend(labels, fontsize = 12)
    main_ax.legend(loc=legend_loc, frameon=False)
    plt.savefig(f'./plots/{filename}_maxlexstate.png')
    plt.savefig(f'./plots/{filename}_maxlexstate.svg')


max_yval_state,max_yval_error = get_max_yval(all_simulations)


def total_error_means(filename, simulation, max_yval, ordered_conditions = [],condition_styles = ['r','k','b','g','r:','k:','b:','g:','r--','k--','b--','g--'],labels = [],legend_loc = 'upper right'):

    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(12*1.3, 6*1.3))
    main_ax = fig.add_subplot()
    main_ax.spines.top.set_visible(False)
    main_ax.spines.right.set_visible(False)
    leg_label = 'Total Lexico-semantic PE'
    
    if ordered_conditions == []:
        ordered_conditions = list(simulation.keys())
    for i, (sub_sim, style, label) in enumerate(zip(ordered_conditions, condition_styles, labels)):
        num_trials = simulation[sub_sim]['simulation_data']['total_lexsem_err'][0].shape[0]
        num_iters = simulation[sub_sim]['simulation_data']['total_lexsem_err'][0].shape[-1]
        PRESTIMULUS_WINDOW = 21
        individual_stimulus = num_iters == NUM_ITERS +1
        if individual_stimulus:
            final_stim_duration = NUM_ITERS
            padding = PRESTIMULUS_WINDOW - 1
            full_window = num_iters + padding
            err_vals = np.block([np.ones((num_trials,padding))*0.001, simulation[sub_sim]['simulation_data']['total_lexsem_err'][0]])
        else:
            final_stim_duration = num_iters - NUM_ITERS - 1
            final_stim_duration -= np.remainder(final_stim_duration , NUM_ITERS)
            full_window = final_stim_duration + PRESTIMULUS_WINDOW
            err_vals = simulation[sub_sim]['simulation_data']['total_lexsem_err'][0]

        data_to_plot = np.mean(err_vals,axis = 0).T

        main_ax.plot(np.arange(-PRESTIMULUS_WINDOW + 1, final_stim_duration+0.03, 1.0), data_to_plot[-full_window:],style,label = label)
        main_ax.set_ylabel(leg_label)
        main_ax.set_xlabel('Iterations')
        main_ax.set_xticks(np.arange(-PRESTIMULUS_WINDOW+1, final_stim_duration+0.03, 5.0))
        main_ax.set_yticks(np.arange(0, max_yval+ 10, 200))
        main_ax.set_ylim(-25, max_yval+0.3)

    main_ax.legend(labels, fontsize = 12)
    main_ax.legend(loc=legend_loc, frameon=False)
    plt.savefig(f'./plots/{filename}_total_lexsem_err.png')
    plt.savefig(f'./plots/{filename}_total_lexsem_err.svg')

def total_bias_means(filename, simulation, max_yval, ordered_conditions = [],condition_styles = ['r','k','b','g','r:','k:','b:','g:','r--','k--','b--','g--'],labels = [],legend_loc = 'upper right'):

    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(12*1.3, 6*1.3))
    main_ax = fig.add_subplot()
    main_ax.spines.top.set_visible(False)
    main_ax.spines.right.set_visible(False)
    leg_label = 'Total Lexico-semantic Bias'
    
    if ordered_conditions == []:
        ordered_conditions = list(simulation.keys())
    for i, (sub_sim, style, label) in enumerate(zip(ordered_conditions, condition_styles, labels)):
        num_trials = simulation[sub_sim]['simulation_data']['total_lexsem_bias'][0].shape[0]
        num_iters = simulation[sub_sim]['simulation_data']['total_lexsem_bias'][0].shape[-1]
        PRESTIMULUS_WINDOW = 21
        individual_stimulus = num_iters == NUM_ITERS +1
        if individual_stimulus:
            final_stim_duration = NUM_ITERS
            padding = PRESTIMULUS_WINDOW - 1
            full_window = num_iters + padding
            err_vals = np.block([np.ones((num_trials,padding))*0.001, simulation[sub_sim]['simulation_data']['total_lexsem_bias'][0]])
        else:
            final_stim_duration = num_iters - NUM_ITERS - 1
            final_stim_duration -= np.remainder(final_stim_duration , NUM_ITERS)
            full_window = final_stim_duration + PRESTIMULUS_WINDOW
            err_vals = simulation[sub_sim]['simulation_data']['total_lexsem_bias'][0]

        data_to_plot = np.mean(err_vals,axis = 0).T

        main_ax.plot(np.arange(-PRESTIMULUS_WINDOW + 1, final_stim_duration+0.03, 1.0), data_to_plot[-full_window:],style,label = label)
        main_ax.set_ylabel(leg_label)
        main_ax.set_xlabel('Iterations')
        main_ax.set_xticks(np.arange(-PRESTIMULUS_WINDOW+1, final_stim_duration+0.03, 5.0))
        main_ax.set_yticks(np.arange(0, max_yval+ 10, 200))
        main_ax.set_ylim(-25, max_yval+0.3)

    main_ax.legend(labels, fontsize = 12)
    main_ax.legend(loc=legend_loc, frameon=False)
    plt.savefig(f'./plots/{filename}_total_lexsem_bias.png')
    plt.savefig(f'./plots/{filename}_total_lexsem_bias.svg')

# plot prediction errors
total_error_means('rep_priming', rep_priming, max_yval_error, ordered_conditions = ['unrepeated','repeated'], labels = ['Non-repeated', 'Repeated'], condition_styles= ['k--','k-'])
simulation_accuracy_info(rep_priming)
total_error_means('sem_priming', sem_priming, max_yval_error, ordered_conditions = ['semunrelated','semrelated'],  labels = ['Unrelated', 'Related'], condition_styles= ['k--','k-'])
simulation_accuracy_info(sem_priming)
total_error_means('cloze_simulations', cloze_simulations_bottomup, max_yval_error, ordered_conditions = ['high_cloze', 'med_high_cloze', 'med_low_cloze', 'low_cloze'][::-1], labels = ['99% Cloze', '50% Cloze','25% Cloze', '0.06% Cloze'][::-1], condition_styles = ['k-', 'k--', 'k-.', 'k:'][::-1])
simulation_accuracy_info(cloze_simulations_bottomup)
total_error_means('lexical_violation', lexical_violation, max_yval_error, ordered_conditions = ['high_constraint_expected', 'high_constraint_unexpected', 'low_constraint_unexpected'][::-1], labels = ['99% Cloze', '99% Constraint Unexpected','0.06% Constraint Unexpected'][::-1], condition_styles = ['k-', 'r:', 'k:'][::-1],legend_loc = 'upper left')
simulation_accuracy_info(lexical_violation)
total_error_means('semantic_prediction_overlap', semantic_prediction_overlap, max_yval_error, ordered_conditions = ['high_constraint_expected','semrelated_99cloze', 'semunrelated_99cloze'][::-1],\
     labels = ['Expected', '99% Constraint Related',  '99% Constraint Unrelated'][::-1], condition_styles = ['k-', 'k--', 'k:'][::-1],legend_loc = 'upper left')
simulation_accuracy_info(semantic_prediction_overlap)

# plot bias
total_bias_means('rep_priming', rep_priming, max_yval_error, ordered_conditions = ['unrepeated','repeated'], labels = ['Non-repeated', 'Repeated'], condition_styles= ['k--','k-'])
simulation_accuracy_info(rep_priming)
total_bias_means('sem_priming', sem_priming, max_yval_error, ordered_conditions = ['semunrelated','semrelated'],  labels = ['Unrelated', 'Related'], condition_styles= ['k--','k-'])
simulation_accuracy_info(sem_priming)
total_bias_means('cloze_simulations', cloze_simulations_bottomup, max_yval_error, ordered_conditions = ['high_cloze', 'med_high_cloze', 'med_low_cloze', 'low_cloze'][::-1], labels = ['99% Cloze', '50% Cloze','25% Cloze', '0.06% Cloze'][::-1], condition_styles = ['k-', 'k--', 'k-.', 'k:'][::-1])
simulation_accuracy_info(cloze_simulations_bottomup)
total_bias_means('lexical_violation', lexical_violation, max_yval_error, ordered_conditions = ['high_constraint_expected', 'high_constraint_unexpected', 'low_constraint_unexpected'][::-1], labels = ['99% Cloze', '99% Constraint Unexpected','0.06% Constraint Unexpected'][::-1], condition_styles = ['k-', 'r:', 'k:'][::-1],legend_loc = 'upper left')
simulation_accuracy_info(lexical_violation)
total_bias_means('semantic_prediction_overlap', semantic_prediction_overlap, max_yval_error, ordered_conditions = ['high_constraint_expected','semrelated_99cloze', 'semunrelated_99cloze'][::-1],\
     labels = ['Expected', '99% Constraint Related',  '99% Constraint Unrelated'][::-1], condition_styles = ['k-', 'k--', 'k:'][::-1],legend_loc = 'upper left')
simulation_accuracy_info(semantic_prediction_overlap)

# plot behavioral trajectory
top_state_means('behav_sem_priming', sem_priming, max_yval_state, ordered_conditions = ['semunrelated','semrelated'],  labels = ['Unrelated', 'Related'], condition_styles= ['k--','k-'])
top_state_means('behav_rep_priming', rep_priming, max_yval_state, ordered_conditions = ['unrepeated','repeated'], labels = ['Non-repeated', 'Repeated'], condition_styles= ['k--','k-'])
top_state_means('behav_cloze_simulations', cloze_simulations_bottomup, max_yval_state, ordered_conditions = ['high_cloze', 'med_high_cloze', 'med_low_cloze', 'low_cloze'][::-1], labels = ['99% Cloze', '50% Cloze','25% Cloze', '0.06% Cloze'][::-1], condition_styles = ['k-', 'k--', 'k-.', 'k:'][::-1])
top_state_means('behav_lexical_violation', lexical_violation, max_yval_state, ordered_conditions = ['high_constraint_expected', 'high_constraint_unexpected', 'low_constraint_unexpected'][::-1], labels = ['99% Cloze', '99% Constraint Unexpected','0.06% Constraint Unexpected'][::-1], condition_styles = ['k-', 'r:', 'k:'][::-1],legend_loc = 'upper left')
top_state_means('behav_semantic_prediction_overlap', semantic_prediction_overlap, max_yval_state, ordered_conditions = ['high_constraint_expected','semrelated_99cloze', 'semunrelated_99cloze'][::-1],\
     labels = ['Expected', '99% Constraint Related',  '99% Constraint Unrelated'][::-1], condition_styles = ['k-', 'k--', 'k:'][::-1],legend_loc = 'upper left')


##################### CREATE CSV DATA FROM ALL SIMULATIONS #####################

def create_simulation_df(simulation, conditions_dict,simulation_name):
    # given a simulation and a conditions_dict, write a CSV file with the right columns
    sub_simulations = list(simulation.keys()) # list of conditions (e.g., unrepeated, repeated)
    number_of_conditions = len(sub_simulations)
    assert conditions_dict['factors'].shape[0] == number_of_conditions # each condition must have a name
    num_of_trials_condition1 = simulation[sub_simulations[0]]['simulation_data']['total_lexsem_err'].shape[1]
    num_iters_condition1 = simulation[sub_simulations[0]]['simulation_data']['total_lexsem_err'].shape[-1]
    for subsim in sub_simulations[1:]:
        num_of_trials_conditionK = simulation[subsim]['simulation_data']['total_lexsem_err'].shape[1]
        num_iters_conditionK = simulation[subsim]['simulation_data']['total_lexsem_err'].shape[-1]
        assert num_of_trials_conditionK == num_of_trials_condition1 # make sure all conditions have an equal number of trials
        assert num_iters_conditionK == num_iters_condition1 # make sure all conditions have an equal number of iterations per trial
    num_trials_per_condition = num_of_trials_condition1
    num_iters_per_trial = num_iters_condition1
    # get a 10-iteration time_window covering iterations 2 through 11, inclusive.
    # Note that -20 is the first iteration of the final word. -20 + 1 is the 2nd iteration; -20 + 11 is the 12th iteration. 
    time_window = np.arange(num_iters_per_trial)[-NUM_ITERS + 1:-NUM_ITERS + 11] 

    WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed = np.zeros((num_trials_per_condition*number_of_conditions,9)).astype('object')
    sim_input_all = []

    for condition_number, subsim in enumerate(sub_simulations):
        # retrieve start and end inds in the data matrix
        # confirm that these are the right ones.
        start_ind = num_trials_per_condition*condition_number 
        end_ind = num_trials_per_condition*(condition_number +1)

        sim_input_inds = np.array(get_correct_inds(simulation[subsim]['sim_input_bottomup'] ,lexicon.words))
        sim_input_all.extend(simulation[subsim]['sim_input_bottomup'])
        LexSemErr_FullTimeCourse = simulation[subsim]['simulation_data']['total_lexsem_err'][0]
        
        mean_LexSemErr = np.mean(LexSemErr_FullTimeCourse[:,time_window], axis = 1)


        WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,1] = mean_LexSemErr

        boolean_threshold_crossing = simulation[subsim]['simulation_data']['max_lex_state_activation'][0][:,-NUM_ITERS:] > THRESHOLD
        threshold_crossing_iteration = np.argmax(boolean_threshold_crossing, axis = 1)
        WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,2] = threshold_crossing_iteration
        WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,6][boolean_threshold_crossing[:,-1]] = 1
        if not np.any(np.isnan(sim_input_inds)):
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,0] = sim_input_inds
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,3] = lexicon.ONsize.T[sim_input_inds][:,0]
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,4] = lexicon.frequency.T[sim_input_inds][:,0]
            # conc = np.array([lexicon.concreteness]).T
            # conc[conc == 0] = -1 # code nonrich items as -1
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,5] = lexicon.num_semfeats[sim_input_inds]
            ThresholdWasCrossed,CorrectItemCrossed,WordThatCrossed = get_threshold_crossing_info(simulation, subsim, THRESHOLD)
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,6] = ThresholdWasCrossed # binary value indicating if any item crossed the threshold
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,7] = CorrectItemCrossed # binary value indicating if the correct item crossed the threshold
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,8] = WordThatCrossed # word string indicating which word crossed the threshold
        else:
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,0] = np.nan
            orth_overlap = np.dot(wordlist_to_orth(simulation[subsim]['sim_input_bottomup']).T, lexicon.orthmatrix) # find the raw orthographic overlap
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,3] = np.sum(orth_overlap == 3, axis = 1) # retrieve orthographic neighbors relative to the model's lexicon
            WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed[start_ind:end_ind,4:6] = np.nan
        
    cond_names = np.vstack([np.tile(conditions_dict['factors'][i], (num_trials_per_condition, 1)) for i in range(number_of_conditions)])
    cond_codes = np.vstack([np.tile(conditions_dict['coding'][i],(num_trials_per_condition,1)) for i in range(number_of_conditions)])


    df1 = pd.DataFrame(WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed, columns = 'WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_NumSemFeats_ThresholdWasCrossed_CorrectItemCrossed_WordThatCrossed'.split('_'))
    df2 = pd.DataFrame(cond_names, columns = [i+'_name' for i in conditions_dict['col_names']])
    df3 = pd.DataFrame(cond_codes, columns = [i+'_code' for i in conditions_dict['col_names']])

    final_df = pd.concat([df1, df2,df3], axis=1)
    final_df['WordInput'] = sim_input_all
    final_df.to_csv(f'./simulation_csv_files/{simulation_name}_N400_{time_window[0]}_to_{time_window[-1]}_IterationsToThreshold{THRESHOLD}.csv', index = False)
    # return final_df


##### STANDARD SIMULATION #####
# add dummy condition so that it can be passed into `create_simulation_df`
standard_simulation_withdummycondition = {}
standard_simulation_withdummycondition['dummy'] = standard_simulation
std_sim_conditions = {}
std_sim_conditions['col_names'] = np.array(['Dummy'])
std_sim_conditions['factors'] = np.array([['dummy']])
std_sim_conditions['coding'] =  np.array([[0]])
create_simulation_df(standard_simulation_withdummycondition,std_sim_conditions,'Standard_Simulation')


##### REPETITION PRIMING #####

rep_priming_conditions = {}
rep_priming_conditions['col_names'] = np.array(['Repeated'])
rep_priming_conditions['factors'] = np.array([[i] for i in list(rep_priming.keys())])
rep_priming_conditions['coding'] =  np.array([[-0.5],[0.5]])
create_simulation_df(rep_priming,rep_priming_conditions,'RepetitionPriming_Simulation')


##### SEMANTIC PRIMING #####

sem_priming_conditions = {}
sem_priming_conditions['col_names'] = np.array(['SemanticRelatedness'])
sem_priming_conditions['factors'] = np.array([[i] for i in list(sem_priming.keys())])
sem_priming_conditions['coding'] =  np.array([[-0.5],[0.5]])
create_simulation_df(sem_priming,sem_priming_conditions,'SemanticPriming_Simulation')

##### LEXICAL PREDICTABILITY #####

cloze_conditions = {}
cloze_conditions['col_names'] = np.array(['Cloze'])
cloze_conditions['factors'] = np.array([[i] for i in list(cloze_simulations_bottomup.keys())])
cloze_conditions['coding'] =  np.array([[1/1579],[0.25],[0.5],[0.99]])
create_simulation_df(cloze_simulations_bottomup,cloze_conditions,'ClozeProbability_Simulation')


##### LEXICAL PREDICTION VIOLATION #####

lexical_violation_conditions = {}
lexical_violation_conditions['col_names'] = np.array(['Constraint', 'IsExpected'])
lexical_violation_conditions['factors'] = np.array([['LowConstraint','Unexpected'],['HighConstraint','Unexpected'], ['HighConstraint','Expected']])
lexical_violation_conditions['coding'] =  np.array([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5]])
create_simulation_df(lexical_violation,lexical_violation_conditions,'LexicalViolation_Simulation')


##### ANTICIPATORY SEMANTIC OVERLAP #####
del semantic_prediction_overlap['high_constraint_expected']

semantic_prediction_overlap_conditions = {}
semantic_prediction_overlap_conditions['col_names'] = np.array(['Relatedness'])
semantic_prediction_overlap_conditions['factors'] = np.array([['SemUnrelated'],['SemRelated']])
semantic_prediction_overlap_conditions['coding'] =  np.array([[-0.5],[0.5]])
create_simulation_df(semantic_prediction_overlap,semantic_prediction_overlap_conditions,'SemanticPredictionOverlap_Simulation')

