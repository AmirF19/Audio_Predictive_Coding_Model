library(lme4)
library(lmerTest)
library(tidyverse)
library(emmeans)
library(effectsize)


# prevent R from wrapping its output around too soon
options(width=300)

nrm <- function(x) {
  return((x - mean(x))/sd(x))
}


base = "D:\\Machine_learning\\delete\\PCBehav_comprehensive\\without_vis_features\\4runs_4000feats_atleast3reps\\simulation_csv_files\\"

# load data
std_sim= read_csv(paste(base, "Standard_Simulation_N400_2_to_11_IterationsToThreshold4.0.csv", sep = ""))
wrd_psd_sim= read_csv(paste(base, "Word_vs_Pseudoword_Simulation_N400_2_to_11_IterationsToThreshold4.0.csv", sep = ""))
reppriming_sim= read_csv(paste(base, "RepetitionPriming_Simulation_N400_27_to_36_IterationsToThreshold4.0.csv", sep = ""))
sempriming_sim = read_csv(paste(base, "SemanticPriming_Simulation_N400_27_to_36_IterationsToThreshold4.0.csv", sep = ""))
formpriming_sim = read_csv(paste(base, "FormPriming_Simulation_N400_27_to_36_IterationsToThreshold4.0.csv", sep = ""))
wrdpsd_formpriming_sim = read_csv(paste(base, "FormPriming_WrdPsd_Simulation_N400_27_to_36_IterationsToThreshold4.0.csv", sep = ""))
cloze_sim = read_csv(paste(base, "ClozeProbability_Simulation_N400_22_to_31_IterationsToThreshold4.0.csv", sep = ""))
lexviol_sim = read_csv(paste(base, "LexicalViolation_Simulation_N400_22_to_31_IterationsToThreshold4.0.csv", sep = ""))
sempredoverlap_sim= read_csv(paste(base, "SemanticPredictionOverlap_Simulation_N400_22_to_31_IterationsToThreshold4.0.csv", sep = ""))
orthpredoverlap_sim= read_csv(paste(base, "OrthographicPredictionOverlap_Simulation_N400_22_to_31_IterationsToThreshold4.0.csv", sep = ""))

# add the info to the wrdpsd_formpriming_sim simulation
lex= read_csv(paste(base, "lexical_characteristics\\form_priming_primetarget_info.csv", sep = ""))
lex_relative_freq = lex %>% mutate(TargetFreqMinusPrimeFreq = Target_Frequency - Prime_Frequency)
unrel_word_cond = lex_relative_freq %>% filter(Condition == "after_unrel_wrd")  
rel_word_cond = lex_relative_freq %>% filter(Condition == "after_rel_wrd")  

wrdpsd_formpriming_sim_justrel = wrdpsd_formpriming_sim %>% 
filter(PrimeIsWord_name == 'Word', FormRelatedness_name == 'Related') %>%
 mutate(TargetFreqMinusPrimeFreq = rel_word_cond$TargetFreqMinusPrimeFreq) %>%
 mutate(PrimeFreq = rel_word_cond$Prime_Frequency)

wrdpsd_formpriming_sim_justunrel = wrdpsd_formpriming_sim %>%
filter(PrimeIsWord_name == 'Word', FormRelatedness_name == 'Unrelated')%>%
 mutate(TargetFreqMinusPrimeFreq = unrel_word_cond$TargetFreqMinusPrimeFreq)%>%
 mutate(PrimeFreq = unrel_word_cond$Prime_Frequency)

wrdpsd_formpriming_sim_relfreq <- bind_rows(wrdpsd_formpriming_sim_justrel, wrdpsd_formpriming_sim_justunrel)


# normalize all quantitative variables
nrm_std_sim = std_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_reppriming_sim = reppriming_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_sempriming_sim = sempriming_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_formpriming_sim = formpriming_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_wrdpsd_formpriming_sim = wrdpsd_formpriming_sim_relfreq %>% mutate_if(is.numeric, list(norm = nrm))
nrm_cloze_sim = cloze_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_lexviol_sim = lexviol_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_sempredoverlap_sim = sempredoverlap_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_orthpredoverlap_sim = orthpredoverlap_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_wrd_psd_sim = wrd_psd_sim %>% mutate_if(is.numeric, list(norm = nrm))
options(width = 300)

######## stats ########

# Section 1.1, 1.3, 1.4
nrm_std_sim_accurate = nrm_std_sim %>% filter(CorrectItemCrossed == 1)
nrm_std_sim_accurate %>% summarize(n = n(), meanlse = mean(LexSemErr), meantci = mean(ThresholdCrossingIteration))
std_model_N400 = lm(LexSemErr ~ ONsize_norm + Frequency_norm + NumSemFeats_norm, data = nrm_std_sim_accurate)
nrm_std_sim %>% summarize(n = n(), meanlse = mean(LexSemErr), meantci = mean(ThresholdCrossingIteration))
nrm_std_sim %>% filter(ThresholdWasCrossed == TRUE) %>% group_by(CorrectItemCrossed) %>% summarize(n = n(), meanlse = mean(LexSemErr), meantci = mean(ThresholdCrossingIteration))
options(width = 100)
options(repr.plot.width=30, repr.plot.height=8)
ggplot(nrm_std_sim_accurate, aes(x = Frequency_norm, y = LexSemErr), width = 14, height = 7) +
                geom_point() +  # Add points
                labs(x = "Frequency", y = "LexSemErr") +  # Set axis labels
                theme_minimal()

# Display the plot
print(scatter_plot)

summary(std_model_N400)
std_model_behav = lm(ThresholdCrossingIteration ~ ONsize_norm + Frequency_norm + NumSemFeats_norm, data = nrm_std_sim_accurate)
summary(std_model_behav)


# # Section 1.2: in the paper, the outcome variable is LexSemErr_norm.
# psd_ONsize_model_norm = lm(LexSemErr_norm ~ ONsize_norm*IsWord_code, data = nrm_wrd_psd_sim) # this is in the paper
# summary(psd_ONsize_model_norm)

# instead, I should use `LexSemErr` itself, like this:
psd_ONsize_model = lm(LexSemErr ~ ONsize_norm*IsWord_code, data = nrm_wrd_psd_sim) 
summary(psd_ONsize_model)


# Section 2.1
options(width = 100)
nrm_reppriming_sim %>% glimpse()
nrm_reppriming_sim %>% filter(ThresholdWasCrossed == 1) %>% group_by(Repeated_code) %>% summarize(accuracy = sum(CorrectItemCrossed)/n())

nrm_reppriming_sim_accurate = nrm_reppriming_sim %>% filter(CorrectItemCrossed == 1)
nrm_reppriming_sim_accurate %>% group_by(Repeated_name) %>% summarize(n=n())
nrm_reppriming_sim %>% group_by(Repeated_name) %>% summarize(n=n())
RepPrimModel_N400 = lmer(LexSemErr ~ Repeated_code + (Repeated_code || WordInput), data = nrm_reppriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
# failed to converge:
# RepPrimModel = lmer(LexSemErr ~ Repeated_code + (Repeated_code | WordInput), data = nrm_reppriming_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(RepPrimModel_N400)
RepPrimModel_behav = lmer(ThresholdCrossingIteration ~ Repeated_code + (Repeated_code || WordInput), data = nrm_reppriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(RepPrimModel_behav)


options(width = 100)
nrm_sempriming_sim %>% filter(ThresholdWasCrossed == 1) %>% group_by(SemanticRelatedness_code) %>% summarize(accuracy = sum(CorrectItemCrossed)/n())

# Section 2.2
nrm_sempriming_sim_accurate = nrm_sempriming_sim %>% filter(CorrectItemCrossed == 1)
nrm_sempriming_sim_accurate %>% group_by(SemanticRelatedness_name) %>% summarize(n=n()/1476)
SemPrimModel_N400= lmer(LexSemErr ~ SemanticRelatedness_code + (SemanticRelatedness_code || WordInput), data = nrm_sempriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(SemPrimModel_N400)
SemPrimModel_behav= lmer(ThresholdCrossingIteration ~ SemanticRelatedness_code + (SemanticRelatedness_code || WordInput), data = nrm_sempriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(SemPrimModel_behav)



# difference between repeated and semantically primed words
repprimed = nrm_reppriming_sim_accurate %>% filter(Repeated_name == "repeated")%>% mutate(dummy_condition = "repeated")
semprimed = nrm_sempriming_sim_accurate %>% filter(SemanticRelatedness_name == "semrelated") %>% mutate(dummy_condition = "semrelated")
sem_vs_rep = full_join(semprimed, repprimed, by = "WordInds")
t.test(sem_vs_rep$LexSemErr.x, sem_vs_rep$LexSemErr.y, paired = TRUE)
t.test(sem_vs_rep$ThresholdCrossingIteration.x, sem_vs_rep$ThresholdCrossingIteration.y, paired = TRUE)

sem_vs_rep %>% glimpse()

# 
# accuracy by condition
nrm_formpriming_sim  %>% filter(ThresholdWasCrossed == 1) %>% group_by(FormRelatedness_code) %>% summarize(accuracy = sum(CorrectItemCrossed)/n())
nrm_formpriming_sim_accurate = nrm_formpriming_sim %>% filter(CorrectItemCrossed == 1)
nrm_formpriming_sim_accurate %>% group_by(FormRelatedness_name) %>% summarize(n=n()/1476)
nrm_formpriming_sim_accurate %>% group_by(FormRelatedness_code, CorrectItemCrossed) %>% summarize(threshcrossiter = mean(ThresholdCrossingIteration),n= n())

FormPrimModel_N400= lmer(LexSemErr ~ FormRelatedness_code + (FormRelatedness_code || WordInput), data = nrm_formpriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(FormPrimModel_N400)
FormPrimModel_behav= lmer(ThresholdCrossingIteration ~ FormRelatedness_code + (FormRelatedness_code || WordInput), data = nrm_formpriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(FormPrimModel_behav)
options(width = 100)
nrm_formpriming_sim  %>% group_by(FormRelatedness_code, CorrectItemCrossed) %>% summarize(threshcrossiter = mean(ThresholdCrossingIteration),n= n())

wrdpsd_formpriming_sim %>% glimpse()
wrdpsd_formpriming_sim %>% colnames()



nrm_wrdpsd_formpriming_sim %>% glimpse()

nrm_wrd_formpriming_sim_accurate = nrm_wrdpsd_formpriming_sim %>% mutate(HighFreqTarget = Frequency > median(std_sim$Frequency)) %>% filter(CorrectItemCrossed == 1)
FormPrimModel_behav= lmer(ThresholdCrossingIteration ~ Frequency_norm*TargetFreqMinusPrimeFreq_norm*FormRelatedness_code + (0 + FormRelatedness_code | WordInput), data = nrm_wrd_formpriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
options(width = 300)
summary(FormPrimModel_behav)


# Section 3.1
nrm_cloze_sim %>% filter(ThresholdWasCrossed == 1) %>% group_by(as.factor(Cloze_code)) %>% summarize(accuracy = sum(CorrectItemCrossed)/n(), n = n())
nrm_cloze_sim_accurate = nrm_cloze_sim %>% filter(CorrectItemCrossed == 1)
nrm_cloze_sim_accurate %>% group_by(Cloze_code) %>% summarize(n=n()/1476)
nrm_cloze_sim %>% group_by(as.factor(Cloze_code)) %>% summarize(accuracy = sum(ThresholdWasCrossed)/n(), n = n())

nrm_cloze_sim%>% glimpse()
ClzModel_N400 = lmer(LexSemErr ~ Cloze_code_norm + (0 + Cloze_code_norm  | WordInput), data = nrm_cloze_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(ClzModel_N400)
ClzModel_behav = lmer(ThresholdCrossingIteration ~ Cloze_code_norm + (0 + Cloze_code_norm  | WordInput), data = nrm_cloze_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(ClzModel_behav)
# Section 3.2
nrm_lexviol_sim_unexp_accurate = nrm_lexviol_sim %>% filter(CorrectItemCrossed == 1, IsExpected_code == -0.5)
nrm_lexviol_sim_unexp_accurate %>% group_by(Constraint_name) %>% summarize(n=n()/1476)
LexViolModel_N400 = lmer(LexSemErr ~ Constraint_code  + (0+ Constraint_code | WordInput), data = nrm_lexviol_sim_unexp_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))

summary(LexViolModel_N400)
LexViolModel_behav = lmer(ThresholdCrossingIteration ~ Constraint_code  + (0 + Constraint_code | WordInput), data = nrm_lexviol_sim_unexp_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(LexViolModel_behav)

# Section 3.3
PE_expected = nrm_cloze_sim %>% mutate(dummy_condition = "expected") %>% filter(Cloze_name == "high_cloze")
unexp_related_sempredoverlap = nrm_sempredoverlap_sim %>% filter(Cloze_name == "HighCloze")%>% filter(Relatedness_name == "SemRelated") %>% mutate(dummy_condition = "unexp_related")
unexp_unrelated_sempredoverlap = nrm_sempredoverlap_sim %>% filter(Cloze_name == "HighCloze")%>% filter(Relatedness_name == "SemUnrelated") %>% mutate(dummy_condition = "unexp_unrelated")
temp = full_join(PE_expected, unexp_related_sempredoverlap)
final_SPO = full_join(temp, unexp_unrelated_sempredoverlap)
# create the three conditions: expected, overlap, nonoverlap
final_SPO = final_SPO %>% mutate(dummy_condition = fct_relevel(dummy_condition,"unexp_related","unexp_unrelated", "expected"))
expected_semoverlap_nonoverlap_model_N400 = lmer(LexSemErr ~ as.factor(dummy_condition) + (1| WordInput), data = final_SPO, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000))) # nolint: line_length_linter.
summary(expected_semoverlap_nonoverlap_model_N400)
expected_semoverlap_nonoverlap_model_behav = lmer(ThresholdCrossingIteration ~ as.factor(dummy_condition) + (1| WordInput), data = final_SPO, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000))) # nolint: line_length_linter.
summary(expected_semoverlap_nonoverlap_model_behav)


# Section 3.4: Effect of contextual constraint on the anticipatory semantic overlap effect
unexp_related_sempredoverlap_99 = nrm_sempredoverlap_sim %>% filter(Cloze_name == "HighCloze")%>% filter(Relatedness_name == "SemRelated") %>% mutate(dummy_condition = "unexp_related")
unexp_unrelated_sempredoverlap_99 = nrm_sempredoverlap_sim %>% filter(Cloze_name == "HighCloze")%>% filter(Relatedness_name == "SemUnrelated") %>% mutate(dummy_condition = "unexp_unrelated")
unexp_related_sempredoverlap_50 = nrm_sempredoverlap_sim %>% filter(Cloze_name == "MedCloze")%>% filter(Relatedness_name == "SemRelated") %>% mutate(dummy_condition = "unexp_related")
unexp_unrelated_sempredoverlap_50 = nrm_sempredoverlap_sim %>% filter(Cloze_name == "MedCloze")%>% filter(Relatedness_name == "SemUnrelated") %>% mutate(dummy_condition = "unexp_unrelated")
temp2 = full_join(unexp_related_sempredoverlap_99, unexp_related_sempredoverlap_50)
temp3 = full_join(unexp_unrelated_sempredoverlap_99, unexp_unrelated_sempredoverlap_50)
finalSPO_constraint = full_join(temp2, temp3)
finalSPO_constraint = finalSPO_constraint %>% mutate(Relatedness_name = fct_relevel(Relatedness_name, "SemUnrelated", "SemRelated"))%>% mutate(Cloze_name = fct_relevel(Cloze_name, "MedCloze", "HighCloze"))
SPOconstraint_model_N400 = lmer(LexSemErr ~ as.factor(Cloze_name)*as.factor(Relatedness_name) + (1| WordInput), data = finalSPO_constraint, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(SPOconstraint_model_N400)

finalSPO_constraint %>% group_by(Relatedness_name, Cloze_name) %>% summarize(n())
finalSPO_constraint_accurate  = finalSPO_constraint %>% filter(CorrectItemCrossed == 1)
finalSPO_constraint_accurate %>% group_by(Relatedness_name, Cloze_name) %>% summarize(n()/1476)

SPOconstraint_model_behav = lmer(ThresholdCrossingIteration ~ as.factor(Cloze_name)*as.factor(Relatedness_name) + (1| WordInput), data = finalSPO_constraint, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(SPOconstraint_model_behav)
options(width=300)


# Section 3.5
unexp_related_orthpredoverlap= nrm_orthpredoverlap_sim %>% filter(IsNeighborofExpected_name == "Neighbor")%>% mutate(dummy_condition = "unexp_related")
unexp_unrelated_orthpredoverlap= nrm_orthpredoverlap_sim %>% filter(IsNeighborofExpected_name == "NonNeighbor") %>% mutate(dummy_condition = "unexp_unrelated")
PE_expected = nrm_cloze_sim  %>% filter(Cloze_name == "high_cloze") %>% mutate(dummy_condition = "expected")
unexpecteds = full_join(unexp_related_orthpredoverlap, unexp_unrelated_orthpredoverlap)
finalOPO = full_join(PE_expected, unexpecteds)
finalOPO = finalOPO %>% mutate(dummy_condition = fct_relevel(dummy_condition,"unexp_related","unexp_unrelated", "expected"))
OPO_model= lmer(LexSemErr ~ as.factor(dummy_condition) + (1 | WordInput) , data = finalOPO, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(OPO_model)
finalOPO %>% colnames()
finalOPO_accurate  = finalOPO %>% filter(CorrectItemCrossed == 1)
finalOPO_accurate %>% group_by(dummy_condition) %>% summarize(n()/1476)


# Section 3.6
nrm_orthpredoverlap_sim %>% glimpse()
OrthPredOverlapModel = lmer(LexSemErr ~ IsNeighborofExpected_code + (0 + IsNeighborofExpected_code | WordInput), data = nrm_orthpredoverlap_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(OrthPredOverlapModel)

# Section 4.1
Interactions_RepPrimModel = lmer(LexSemErr ~ ONsize_norm*Repeated_code + Frequency_norm*Repeated_code + NumSemFeats_norm*Repeated_code + ( 0 + Repeated_code | WordInput), data = nrm_reppriming_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(Interactions_RepPrimModel)
Interactions_RepPrimModel = lmer(LexSemErr ~ ONsize_norm*Repeated_code + Frequency_norm*Repeated_code + NumSemFeats_norm*Repeated_code + ( 0 + Repeated_code | WordInput), data = nrm_reppriming_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(Interactions_RepPrimModel)

# Section 4.2
split_nrm_cloze_sim = nrm_cloze_sim %>% filter(Cloze_name == 'high_cloze' | Cloze_name == 'low_cloze')
Interactions_ClzModel = lmer(LexSemErr ~ ONsize_norm*Cloze_code_norm  + Frequency_norm*Cloze_code_norm  + NumSemFeats_norm*Cloze_code_norm  + (Cloze_code_norm  || WordInput), data = split_nrm_cloze_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(Interactions_ClzModel)
Interactions_ClzModel_behav = lmer(ThresholdCrossingIteration ~ ONsize_norm*Cloze_code_norm  + Frequency_norm*Cloze_code_norm  + NumSemFeats_norm*Cloze_code_norm  + (Cloze_code_norm  || WordInput), data = split_nrm_cloze_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(Interactions_ClzModel_behav)
# failed to converge:
# splitClzModel = lmer(LexSemErr ~ ONsize_norm*Cloze_code  + Frequency_norm*Cloze_code  + Concreteness*Cloze_code  + (Cloze_code  | WordInput), data = split_nrm_cloze_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
