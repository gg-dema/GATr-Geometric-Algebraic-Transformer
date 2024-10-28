

def save_best_params_callback(study, trial):

    if study.best_trial == trial:
        best_params = study.best_params

        # save the relevant parameters

        
        