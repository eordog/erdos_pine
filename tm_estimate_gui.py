#!/usr/bin/env python
# coding: utf-8

# In[1]:


import PySimpleGUI as sg
import xgboost as xgb
import pandas as pd
import numpy as np
import gc
import torch, esm
from sklearn.decomposition import PCA
import pickle as pk
import re


# In[2]:


def torch_batch_converter ():
    token_map = {'L': 0, 'A': 1, 'G': 2, 'V': 3, 'S': 4, 'E': 5, 'R': 6, 'T': 7, 'I': 8, 'D': 9, 'P': 10, 
             'K': 11, 'Q': 12, 'N': 13, 'F': 14, 'Y': 15, 'M': 16, 'H': 17, 'W': 18, 'C': 19}
    t_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    t_model.eval()  # disables dropout for deterministic results
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_model.to(device)
    
    return t_model, batch_converter, device


# In[3]:


def seq_analyse (seq, t_model, batch_converter, device):
    all_pdb_embed_pool_test = []

    from scipy.special import softmax 
    from scipy.stats import entropy

    data = [("protein1", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = t_model(batch_tokens, repr_layers=[33])

    results = results["representations"][33].detach().cpu().numpy()    
    torch_model_results = np.mean( results[0,:,:],axis=0 )       

    del batch_tokens, results, batch_labels, batch_strs
    gc.collect(); torch.cuda.empty_cache()
    
    return pd.DataFrame (torch_model_results.reshape(1,-1), columns = ['val_{}'.format(index) for index in range(1280)], index = ['0'])


# In[5]:


def pca_seq_df (df, pca_model):
    pca_list = [col for col in df.columns.tolist() if col.__contains__('val')]
    pca_pool = pk.load(open(pca_model, 'rb'))
    pca_esm = pca_pool.transform(df[pca_list])
    
    return pd.DataFrame (pca_esm, columns = ['pca_val_{}'.format(i) for i in range (pca_esm.shape[1])])


# In[4]:


def xgb_predict (df, xgb_model):
    
    dpredict = xgb.DMatrix (data=df)
    
    bst = xgb.Booster()
    bst.load_model (xgb_model)
    tm = bst.predict (dpredict)
    
    return tm[0]


# In[12]:


def main ():
    
    load_model_loc = 'XGBoost Model'
    xgb_model = load_model_loc + '/XGB_model.xgb'
    pca_model = load_model_loc+'/pca.pkl'
    
    predict_tm_bool = False
    
    layout = [
             [sg.Text('Please input protein sequence - use the one-letter code for each amino acid.'), sg.InputText('', key = 'input_text')],
             [sg.Text('', key = 'instruction')],
             [sg.Button ('Preload ESM model'),sg.Button('Predict Tm', disabled = True), sg.Button ('Exit')]
             ]
    
    window = sg.Window ('Protein Stability Prediction', layout, modal = True)
    
    while True:
        event, values = window.read()
        
        if event == 'Preload ESM model':
            try:
                print ('Loading the ESM Model. This may take a few minutes.')
                t_model, batch_converter, device = torch_batch_converter ()
                window['instruction'].update ('ESM Model loaded successfully. You can now predict the melting temperature.')
                predict_tm_bool = True
            except ExceptionError as e:
                window['instruction'].update ('Error. Could not load ESM model.')
        elif event == 'Predict Tm':
            protein_seq = values ['input_text'].upper()
            
            protein_seq_regex = r'[^A-Z]+'
            
            if protein_seq != '':
                if len (protein_seq) < 2000:
                    if re.search (protein_seq_regex, protein_seq) == None:
                        invalid_aa_regex = r'[BJOUXZ]'
                        if re.search (invalid_aa_regex, protein_seq) == None:
                            print ('Extraction of features from ESM model is occuring. This may take a few minutes if the protein is very large.')
                            df = seq_analyse (protein_seq, t_model, batch_converter, device)
                            print ('Features from ESM are undergoing PCA!')
                            pca_df = pca_seq_df (df, pca_model)
                            print ('Features are running through the XGBoost model. Almost there!')
                            tm_estimate = xgb_predict (pca_df, xgb_model)
                            
                            tm_estimate = str (round(tm_estimate,1))                            
                            
                            window['instruction'].update ('Analysis complete. Estimated melting temperature is {}.'.format (tm_estimate))
                            protein_seq = ''
                        else:
                            window['instruction'].update ('Error. Invalid Amino Acid code was entered. B, J, O, U, X, and Z are invalid amino acids.')
                            window['input_text'].update ('')
                    else:
                        window['instruction'].update ('Error. Improper sequence entered. Please only use letters. B, J, O, U, X, and Z are not valid amino acids to enter for analysis.')
                        window['input_text'].update ('')
                else:
                    window['instruction'].update ('Error. Protein sequence is too large for analysis. Please enter sequences less than 2000 amino acids in length.')
                    window['input_text'].update ('')
            else:
                window['instruction'].update ('Error. Please enter a protein sequence.')
            
            
            
        elif event == sg.WIN_CLOSED or event == 'Exit':
            break
            
        if predict_tm_bool:
            window['Predict Tm'].update (disabled = False)
    
    window.close()


if __name__ == '__main__':
    main()

