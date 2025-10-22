Overview
This project explores how machine learning can enhance football scouting by identifying high-potential players using data-driven techniques. We evaluated seven classification models on a dataset of 142,079 FIFA player records (2015–2022), enriched with 21 engineered features. The goal was to determine which models best support scouts in prospect identification under real-world constraints.

📊 Key Findings
- Best Performing Model:
XGBoost achieved the highest ROC AUC (0.929) and F1-score (0.615), making it the most balanced and deployable model.
- Recall vs. Precision Tradeoff:
- SVM had the highest recall (0.803), useful for casting a wide net.
- XGBoost maintained precision at 0.635, ideal for confident selection.
- AdaBoost showed extreme recall bias (0.915 recall, 0.221 precision).
- Threshold Optimization:
XGBoost’s optimal threshold was 0.284, yielding 78.6% recall and 50% precision—meeting operational scouting needs.
- Feature Importance:
- Top predictors: Skill moves (0.162), playmaking score (0.144), vision (0.065).
- Technical and cognitive traits outperformed physical attributes.
- Position-Specific Insights:
- Attackers had the highest precision (0.69).
- Goalkeepers showed the most recall variance (±7%).

🛠 Tools & Techniques
- Languages: Python
- Libraries: Scikit-learn, XGBoost, LightGBM
- Data Source: FIFA Career Mode dataset (2015–2022)
- Preprocessing: Feature engineering, outlier handling, stratified sampling
- Evaluation Metrics: ROC AUC, F1-score, precision, recall

📁 Repository Structure
football-scouting-ml/

│ data/                   # Cleaned and processed datasets
│ models/                 # Trained model files and deployment artifacts
│ notebooks/              # Jupyter notebooks for analysis and visualization
│ figures/                # Plots and evaluation charts
│ README.md               # Project documentation


📌 Recommendations
- Use SVM for initial prospect pooling and XGBoost for shortlist refinement.
- Prioritize technical and cognitive traits over physical metrics.
- Develop position-specific classifiers and integrate psychometric data for future improvements.

👤 Authors
- Elikplim Emmanuel Atinyuie
- Collins Nana Agyapong
- Edward Akuleme Adiyure
Final-year students, BSc Computer Science and Engineering
University of Mines and Technology (UMaT), Ghana

