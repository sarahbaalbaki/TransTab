# TransTab

Type 2 diabetes is one of the most prevalent, expensive, and debilitating conditions in the world. While machine learning has been used for specific datasets, we seek to modify and develop a framework for diabetes prediction that has transfer learning capabilities across various datasets. We combine four specific datasets that include categorical, numerical, and textual features varying from lifestyle choices to specific measurements of glucose, cholesterol, and more. We modify the baseline transtab model by building a smart data loader with LLM capabilities, as well as implementing label smoothing, a mixture of experts (MOE) layer, and specific fine-tuning. Feature importance is conducted, and results are compared to the current gold standard for tabular data, XGBoost. We are able to effectively achieve this goal with 93.5% specificity and F1 score of 81% across the merged dataset composed of the four individual datasets. Applications for this model include as a clinical decision support tool that would serve the interests of hospitals, primary care providers, health insurers, patients, and other research institutions in this field  of work. 

“AllDatasets”  - Link to all 4 original datasets as well as the merged dataset
“MergeCleanDatasets.ipynb” - Preprocessing and merging code to write the 5th (combined) dataset
“XGBoostIndividualDatasets.ipynb” - Runs the XGBoost models on the 4 individual datasets
“BaselineAndShap.py” - BaselineAndShap.py runs the baseline transtab model and feature importance for the merged Dataset 
“CustomDataLoader.ipynb”  - The smart data loader for our enhanced transtab model 
“new_utils.py”, “new_trainner.py”, “modeling_new.py” - These files work with the original transtab library, as well as our modified one (they are a superset of the corresponding files in Transtab, with some changes) 
