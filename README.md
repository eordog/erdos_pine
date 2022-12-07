Predicting Thermostability of Enzyme Variants
=========================================================================

Pol Arranz-Gibert, Ayush Khaitan, Richard van Krieken, and Erika Ordog (Team Pine for the Erdos Institute).

Background
----------
Enzymes, which are long chains of units called amino acids, require a suitable temperature to function. Beyond a certain temperature, called the melting temperature, they get denatured and lose their functionality. Although this temperature can be experimentally determined, it would be tremendously useful for bench side scientists and biotech companies to be able to predict this melting temperature from just the amino acid sequence.


Data
-------
The data set was provided by Novozymes as part of the Novozymes Enzyme Stability Prediction competition on Kaggle. It consists of the melting temperatures (Tm) and enzyme sequences for approximately 30,000 enzymes.
We expanded our data set by including enzymes from the data set linked [here](https://github.com/JinyuanSun/mutation-stability-data).

We extracted features such as the protein 3D structure (both obtained from experimentation and predicted by AlphaFold 3D), amino acid properties, amino acid substitution matrices, and features from a protein transformer.


Models
-----------
We performed an XGBoost (with Protein Transformer ESM), XGBoost (with Protein Transformer ESM, Amino acid properties, Amino Acid Substitution matrix, 3D Features), an NLP Hugging Face (with Protein Transformer ESM, Amino acid properties, Amino Acid Substitution matrix, 3D Features), and a third XGBoost (Protein Transformer ESM, PCA) model and compared the root mean squared errors (RMSE) and Spearman correlation coefficients of each.



Results
---------------
Our best model was the third XGBoost model with Protein Transformer ESM and principal component analysys (PCA).
We found the normalized RMSE to be 0.06 and the Pearson R to be 0.81. 

One interesting observation was in the test proteins with a true TM around 20-30 C. We graphed the distribution of the enzyme sizes on the right graph. Here we see that normally proteins in this range are supposed to be 100-300 AA in size, but in the test set they were a lot bigger (300-1000 AA). This indicates that our model is not well-optimized to handle smaller/lower melting temperature proteins. 



