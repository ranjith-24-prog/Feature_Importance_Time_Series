import streamlit as st
import pandas as pd
import os

def laod_df_from_csv(output_path):
    if os.path.exists(output_path):
        # Load the DataFrame if the file exists
        df = pd.read_csv(output_path)
        st.dataframe(df)

# Streamlit sidebar navigation
st.sidebar.title("Feature Importance Techniques")
options = ["Integrated Gradients", "WINit", "LIME", "Permutation Importance"]
choice = st.sidebar.radio("Select an option", options)

# Main application
st.title("Feature Importance Analysis")
st.write("Select an option from the sidebar to view the respective feature importance analysis.")

if choice == "Permutation Importance":
    st.header("Permutation Importance")
    output_path = "output/FI_Dataframes/Permutation_Importance.csv"
    tab1, tab2 = st.tabs(["Plot", "Feature Importance"])
    with tab1:
        st.write("Permutation Importance Plot")
        st.image("output/Permutation_importance_plot.png", caption="Permutation importance Plot")

    with tab2:
        st.write("Feature Importance DataFrame")
        laod_df_from_csv(output_path)

elif choice == "Integrated Gradients":
    st.header("Integrated Gradients")
    output_path = "output/FI_Dataframes/Integrated_Gradients.csv"
    tab1, tab2 = st.tabs(["Plot", "Feature Importance"])
    with tab1:
        st.write("Integrated Gradients Plot")
        st.image("output/Integrated_Gradients_plot.png", caption="Integrated Gradients Plot")

    with tab2:
        st.write("Feature Importance DataFrame")
        laod_df_from_csv(output_path)

elif choice == "WINit":
    st.header("WINit")
    output_path = "output/FI_Dataframes/WINit.csv"
    tab1, tab2 = st.tabs(["Plot", "Feature Importance"])
    with tab1:
        st.write("WINit Plot")
        st.image("output/WINit_plot.png", caption="WINit Plot")

    with tab2:
        st.write("Feature Importance DataFrame")
        laod_df_from_csv(output_path)


elif choice == "LIME":
    st.header("LIME")
    output_path = "output/FI_Dataframes/LIME.csv"

    tab1, tab2 = st.tabs(["Plot", "Feature Importance"])
    with tab1:
        st.write("LIME Plot")
        st.image("output/LIME_plot.png", caption="LIME Plot")

    with tab2:
        st.write("Feature Importance DataFrame")
        laod_df_from_csv(output_path)


