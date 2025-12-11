import string
import numpy as np
import itertools
import matplotlib.pyplot as plt
import random

def wordlist_to_orth(wordlist):
    # takes in list of K words, outputs a 104 x K "spelling" array
    alphabet = string.ascii_lowercase
    wordids = np.array([np.array([alphabet.index(L) for L in word]) for word in wordlist])
    onehots = np.zeros((len(wordlist), len(wordlist[0])*len(alphabet)))
    for i in range(len(wordlist)):
        indices = np.add(wordids[i],np.array([0,1,2,3])*26)
        onehots[i,:][indices] = 1
    return onehots.T

def wordlist_to_ctx(word_stimlist, lexicon, cloze = 0.99, preact_resource = 2.0):
    # takes in a list of K words, assigns a high probability (p = cloze) to each of them in separate trials
    # preact_resource is the value that the pseudoprobability distribution sums to
    # seems like I don't need a word_stimlist.
    lexicon = list(lexicon)
    lexicon_indices = np.array([lexicon.index(w) for w in word_stimlist])
    low_prob = np.multiply(np.ones((len(lexicon),len(word_stimlist))), preact_resource / (len(lexicon) - 1))* (1 - cloze)
    low_prob[lexicon_indices, np.arange(len(lexicon_indices))] = preact_resource*cloze
    return low_prob

def orth_to_wordlist(onehots):
    # change an N x 104 matrix of one-hot encodings to a list of words
    alphabet = string.ascii_lowercase
    NumWords = onehots.shape[0]
    LettersPerWord = int(onehots.shape[1]/len(alphabet))
    matrix_perword = onehots.reshape((NumWords,LettersPerWord,26))
    wordlist = []
    for w in range(NumWords):
        letterlist = [alphabet[i] for i in np.where(matrix_perword[w][:])[-1]]
        wordlist.append(''.join(letterlist))
    return wordlist

def generate_stims_for_all_conditions(lexicon):
    wordlist = lexicon.words
    og = list(wordlist)
    # shared_feats = lexicon.semfeatmatrix.T @ lexicon.semfeatmatrix
    # np.save('./helper_txt_files/shared_feats.npy',shared_feats)
    shared_feats = np.load('./helper_txt_files/shared_feats.npy')
    diag_mask_feats = np.ones(shared_feats.shape) - np.eye(shared_feats.shape[0])
    only_shared_features = np.multiply(shared_feats, diag_mask_feats)


    orthrelated_matrix = lexicon.orthmatrix.T @ lexicon.orthmatrix
    shared_letters = orthrelated_matrix
    diag_mask_letters = np.ones(shared_letters.shape) - np.eye(shared_letters.shape[0])
    only_shared_letters = np.multiply(shared_letters, diag_mask_letters)


    has_semrelated_words_idx = np.ones(lexicon.orthmatrix.shape[-1], dtype = 'bool')
    sem_related_word_idx_nonneighbor = np.zeros(lexicon.orthmatrix.shape[-1], dtype = 'int32')
    has_orthrelated_words_idx = np.ones(lexicon.orthmatrix.shape[-1], dtype = 'bool')
    orth_related_word_idx_nonsemrel = np.zeros(lexicon.orthmatrix.shape[-1], dtype = 'int32')

    fully_unrelated_word_idx_1 = np.zeros(lexicon.orthmatrix.shape[-1], dtype = 'int32')
    fully_unrelated_word_idx_2 = np.zeros(lexicon.orthmatrix.shape[-1], dtype = 'int32')

    # semantically related words
    for idx, word in enumerate(lexicon.words):
        wordsemantics_related_to_current_word = only_shared_features[idx,:] != 0
        wordspelling_unrelated_to_current_word = orthrelated_matrix[idx,:] == 0
        words_related_in_semantics_but_not_spelling = np.logical_and(wordsemantics_related_to_current_word, wordspelling_unrelated_to_current_word)
        if not np.any(words_related_in_semantics_but_not_spelling):
            has_semrelated_words_idx[idx] = False
            pass
        else:
            special_inds = np.argwhere(words_related_in_semantics_but_not_spelling).T[0]
            max_num_shared_feats_under_above_constraints = np.max(only_shared_features[idx,special_inds])
            best_among_special_inds = np.argwhere(only_shared_features[idx,special_inds] == max_num_shared_feats_under_above_constraints)
            the_index = np.random.choice(best_among_special_inds.T[0])
            sem_related_word_idx_nonneighbor[idx] = special_inds[the_index]
            print(f'For the word {word}, the closest semantic but not orth neighbor is {lexicon.words[special_inds[the_index]]}')
    
    # orthographically related words
    for idx, word in enumerate(lexicon.words):
        wordsemantics_unrelated_to_current_word = only_shared_features[idx,:] == 0
        wordspelling_related_to_current_word = orthrelated_matrix[idx,:] > 0
        words_related_in_spelling_but_not_semantics = np.logical_and(wordspelling_related_to_current_word, wordsemantics_unrelated_to_current_word)
        if not np.any(words_related_in_spelling_but_not_semantics):
            has_orthrelated_words_idx[idx] = False
            pass
        else:
            special_inds = np.argwhere(words_related_in_spelling_but_not_semantics).T[0]
            max_num_shared_letters_under_above_constraints = np.max(only_shared_letters[idx,special_inds])
            best_among_special_inds = np.argwhere(only_shared_letters[idx,special_inds] == max_num_shared_letters_under_above_constraints)
            the_index = np.random.choice(best_among_special_inds.T[0])
            orth_related_word_idx_nonsemrel[idx] = special_inds[the_index]
            print(f'For the word {word}, the closest orth but not semantic neighbor is {lexicon.words[special_inds[the_index]]}')

    for idx, word in enumerate(lexicon.words):
        wordsemantics_unrelated_to_current_word = only_shared_features[idx,:] == 0
        wordspelling_unrelated_to_current_word = orthrelated_matrix[idx,:] == 0
        fully_unrelated_words = np.logical_and(wordspelling_unrelated_to_current_word, wordsemantics_unrelated_to_current_word)
        special_inds = np.argwhere(fully_unrelated_words).T[0]
        the_index_1,the_index_2 = np.random.choice(special_inds,2)
        fully_unrelated_word_idx_1[idx] = the_index_1
        fully_unrelated_word_idx_2[idx] = the_index_2
    # for each "standard stim" in the lexicon, keep it if it has a semantically related word (all words have an orth related word)
    # so we just focus on sem related stims
    standard_stims_word_idx = np.arange(lexicon.size)[has_semrelated_words_idx]
    sem_related_word_idx_nonneighbor = sem_related_word_idx_nonneighbor[has_semrelated_words_idx]
    orth_related_word_idx_nonsemrel = orth_related_word_idx_nonsemrel[has_semrelated_words_idx]
    fully_unrelated_word_idx_1_ = fully_unrelated_word_idx_1[has_semrelated_words_idx]
    fully_unrelated_word_idx_2_ = fully_unrelated_word_idx_2[has_semrelated_words_idx]
    semoverlap1 = lexicon.semfeatmatrix[:,fully_unrelated_word_idx_1_] *lexicon.semfeatmatrix[:,sem_related_word_idx_nonneighbor] 
    semoverlap2 = lexicon.semfeatmatrix[:,fully_unrelated_word_idx_2_] *lexicon.semfeatmatrix[:,sem_related_word_idx_nonneighbor] 
    semoverlap3 = lexicon.semfeatmatrix[:,fully_unrelated_word_idx_1_] *lexicon.semfeatmatrix[:,fully_unrelated_word_idx_2_] 
    semoverlap= semoverlap1 + semoverlap2 + semoverlap3
    orthoverlap1 = lexicon.orthmatrix[:,fully_unrelated_word_idx_1_] *lexicon.orthmatrix[:,sem_related_word_idx_nonneighbor] 
    orthoverlap2 = lexicon.orthmatrix[:,fully_unrelated_word_idx_2_] *lexicon.orthmatrix[:,sem_related_word_idx_nonneighbor] 
    orthoverlap3 = lexicon.orthmatrix[:,fully_unrelated_word_idx_1_] *lexicon.orthmatrix[:,fully_unrelated_word_idx_2_] 
    orthoverlap= orthoverlap1 + orthoverlap2 + orthoverlap3
    sem_overlap_indices = np.argwhere(semoverlap.any(axis = 0)).squeeze()
    orth_overlap_indices = np.argwhere(orthoverlap.any(axis = 0)).squeeze()
    all_overlap_indices = np.unique(np.append(sem_overlap_indices, orth_overlap_indices))
    not_overlap_indices = np.delete(np.arange(fully_unrelated_word_idx_1_.shape[0]), all_overlap_indices)

    sem_related_word_idx_nonneighbor = sem_related_word_idx_nonneighbor[not_overlap_indices]
    orth_related_word_idx_nonsemrel = orth_related_word_idx_nonsemrel[not_overlap_indices]
    fully_unrelated_word_idx_1_ = fully_unrelated_word_idx_1_[not_overlap_indices]
    fully_unrelated_word_idx_2_ = fully_unrelated_word_idx_2_[not_overlap_indices]
    standard_stims_word_idx = standard_stims_word_idx[not_overlap_indices]

      
    
    np.save('./helper_txt_files/standard_stims_word_idx.npy', standard_stims_word_idx)
    np.save('./helper_txt_files/fully_unrelated_word_idx_1.npy', fully_unrelated_word_idx_1_)
    np.save('./helper_txt_files/fully_unrelated_word_idx_2.npy', fully_unrelated_word_idx_2_)
    np.save('./helper_txt_files/orth_related_word_idx_nonsemrel.npy', orth_related_word_idx_nonsemrel)
    np.save('./helper_txt_files/sem_related_word_idx_nonneighbor.npy', sem_related_word_idx_nonneighbor)
