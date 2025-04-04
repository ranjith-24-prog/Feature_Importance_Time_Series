# Feature Importance in Time Series For Energy Consumption in CNC Machine

This project applies model-agnostic and model-specific techniques to understand feature importance in energy consumption datasets from CNC machines. It compares methods such as Integrated Gradients (IG), WINIT, LIME, and Permutation Importance (PI) across different models and datasets (with/without correlation).

## Prerequisites

- Python 3.8 or later
- OS: Windows/macOS
- Hardware:
  - Minimum 8GB RAM
- Required Python libraries:
  ```
      Refer requirements.txt
  ```
- The following should also be installed:
  - PyTorch
  - Captum (for IG)
  - LIME
  - Scikit-learn
  - Streamlit
  - Matplotlib
  - Pandas, NumPy

## Installation

1. **Clone the repository**  
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

3. **Run batch analysis (optional)**  
   This processes all datasets using all techniques and stores results.
   ```
   python main.py
   ```

4. **Start Streamlit dashboard**  
   ```
   streamlit run app.py
   ```

5. **Directory Structure**
   ```
   ├── static/Dataset/                      # Input datasets
   ├── output/
   │   ├── FI_Comparison_Results/           # CSVs for technique-wise comparisons
   │   ├── FI_Comparison_Plots/             #Visualizations of technique-wise comparisons
   │   ├── FI_Dataframes/                   # feature importances as CSVs
   │   └── FI_Plots/                        # Visualizations of feature importance
   ├── Integrated_Gradient/
   ├── Permutation_Importance/
   ├── LIME/
   └── WINIT/
   ```

## ToDo / future work

-  Real-Time Implementation: Adapt and optimize models for seamless integration into live
CNC machine operations.
-  Dynamic & Cross-Machine Analysis: Analyze how feature importance evolves over time
and test generalization across CNC machines.
-  Hybrid Interpretability Models: Combine methods like IG and LIME to leverage both global
and local explanations.
-  Feedback-Driven Control Systems: Develop adaptive closed-loop systems where model
insights inform real-time machine adjustments.
-  Update Stremalit Application as per project developments

## Resources

- [Captum for Integrated Gradients](https://captum.ai/)
- [LIME Documentation](https://marcotcr.github.io/lime/)
- [Scikit-learn Permutation Importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [Streamlit for Interactive Web Apps](https://streamlit.io/)

## Documentation

- The Streamlit interface allows easy interaction with each feature importance techniques.
- Each feature importance plot and CSV file is generated with the filename indicating the model, technique, and correlation mode.
- Internally, `main.py` executes all technique functions and saves:
  - Test loss
  - Top 10 important features
  - Execution time
- Final Report link: https://code.ovgu.de/iks-ams/teaching/student-projects/wise-24-25/p1-team2-feature-importance/-/blob/ef4c14d9bbe5f171c9378c154a9fb3d8ddfa9662/documentation/AMS_Team2_Report.pdf
- Stremalit App: https://feature-importance-time-series.streamlit.app/ 


## Authors

- Kavyashree Byalya Nanjegowda : kavyashree.byalya@st.ovgu.de
- Nitin Bharadwaj Nataraj : nitin.nataraj@st.ovgu.de
- Ranjith Mahesh : ranjith.mahesh@st.ovgu.de