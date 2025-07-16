from logistic_regression import LogisticRegression
import streamlit as st
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
def hyperopt_objective(params,X,y,status_widget,max_iter=50):
    model = LogisticRegression(**params,class_weight={0:3, 1:1})
    model.fit(X, y)
    accuracy = model.score(X, y)
    st.session_state.iteration += 1
    iteration = st.session_state.iteration
    st.session_state.best_accuracy = max(st.session_state.best_accuracy, accuracy)
    best_accuracy = st.session_state.best_accuracy
    
    status_widget.text(f"Iteration {iteration}/{max_iter}: Accuracy = {accuracy:.4f}, Best Accuracy: {best_accuracy:.4f}")
    
    return {'loss': -accuracy, 'status' : STATUS_OK}

def hyperparameter_tuning(X, y, max_iter=50):
    space = {
        'C' : hp.loguniform('C', -4, 4),
        'max_iter' : hp.choice('max_iter', list(range(500,2001))),
    }
    trials = Trials()
    status_text = st.empty()
    best_params = fmin(
        fn = lambda params: hyperopt_objective(params, X, y, status_text,max_iter),
        space = space,
        algo = tpe.suggest, 
        max_evals = max_iter,
        trials = trials,
        rstate = np.random.default_rng(42)
    )
    
    best_params['C'] = float(best_params['C'])
    best_params['max_iter'] = max(int(best_params['max_iter']), 500)
    return best_params