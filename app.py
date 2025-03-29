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
    tab1, tab2, tab3 = st.tabs(["Graph", "Plot", "Feature Importance"])
    with tab1:
        st.write("Graph")
        st.image("output/FI_Comparison_Plots/comparison_plot_PI.png")
        # Load your CSV (assuming it's local)
        df = pd.read_csv("output/FI_Comparison_Results/Comparison_Results_PI.csv")

        # Extract only the relevant columns
        simplified_df = df[["Technique", "Top Features"]]

        # Display the table
        st.title("Technique and Top Features")
        st.dataframe(simplified_df, hide_index=True, use_container_width=False, height=600, width=6000)

    with tab2:
        st.write("Permutation Importance Plot")
        image_folder = "output/FI_Plots/Permutation"
        # Get all .png files
        image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
        image_files.sort()  # Optional: keep it ordered

        # Function to format image names
        def format_image_name(filename):
            name_part, rest = filename.split("_PI_", 1)
            rest = rest.replace("withcorr", "with correlation")
            rest = rest.replace("without_corr", "without correlation")
            
            # Remove underscore after DT or RF
            rest = rest.replace("DT_", "DT ").replace("RF_", "RF ")
            
            return f"{name_part} - {rest.replace('.png', '')}"

        # Create a mapping from display name to file
        display_names = [format_image_name(f) for f in image_files]
        display_to_file = dict(zip(display_names, image_files))

        # Dropdown menu
        selected_display_name = st.selectbox("Choose a plot to display:", display_names)

        # Get the actual file path
        selected_file = display_to_file[selected_display_name]
        image_path = os.path.join(image_folder, selected_file)

        # Display image
        st.image(image_path, caption=selected_display_name, use_container_width=True)

    with tab3:
        st.write("Feature Importance DataFrame")
        dataframe_folder = "output/FI_Dataframes/Permutation"
        # Get all .csv files
        dataframe_files = [f for f in os.listdir(dataframe_folder) if f.endswith(".csv")]
        dataframe_files.sort()
        # Function to format image names
        def format_dataframe_name(filename):
            name_part, rest = filename.split("_PI_", 1)
            rest = rest.replace("withcorr", "with correlation")
            rest = rest.replace("without_corr", "without correlation")
            
            # Remove underscore after DT or RF
            rest = rest.replace("DT_", "DT ").replace("RF_", "RF ")
            
            return f"{name_part} - {rest.replace('.csv', '')}"

        # Create a mapping from display name to file
        display_names = [format_dataframe_name(f) for f in dataframe_files]
        display_to_file = dict(zip(display_names, dataframe_files))

        # Dropdown menu
        selected_display_name = st.selectbox("Choose a dataframe to display:", display_names)

        # Get the actual file path
        selected_file = display_to_file[selected_display_name]
        dataframe_path = os.path.join(dataframe_folder, selected_file)
        # Display dataframe
        df = pd.read_csv(dataframe_path)
        st.dataframe(df)

elif choice == "Integrated Gradients":
    st.header("Integrated Gradients")
    output_path = "output/FI_Dataframes/Integrated_Gradients.csv"
    tab1, tab2, tab3 = st.tabs(["Graph", "Plot", "Feature Importance"])
    with tab1:
        st.write("Graph")
        st.image("output/FI_Comparison_Plots/comparison_plot_IG.png")
        # Load your CSV (assuming it's local)
        df = pd.read_csv("output/FI_Comparison_Results/Comparison_Results_IG.csv")

        # Extract only the relevant columns
        simplified_df = df[["Technique", "Top Features"]]

        # Display the table
        st.title("Technique and Top Features")
        st.dataframe(simplified_df, hide_index=True, use_container_width=False, height=600, width=6000)
    with tab2:
        st.write("Integrated Gradients Plot")
        image_folder = "output/FI_Plots/Integrated_Gradients"
        # Get all .png files
        image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
        image_files.sort()  # Optional: keep it ordered

        # Function to format image names
        def format_image_name(filename):
            name_part, rest = filename.split("_IG_", 1)
            rest = rest.replace("withcorr", "with correlation")
            rest = rest.replace("without_corr", "without correlation")
            
            # Remove underscore after DT or RF
            rest = rest.replace("FNN_", "FNN ").replace("lstm_", "LSTM ")
            
            return f"{name_part} - {rest.replace('.png', '')}"

        # Create a mapping from display name to file
        display_names = [format_image_name(f) for f in image_files]
        display_to_file = dict(zip(display_names, image_files))

        # Dropdown menu
        selected_display_name = st.selectbox("Choose a plot to display:", display_names)

        # Get the actual file path
        selected_file = display_to_file[selected_display_name]
        image_path = os.path.join(image_folder, selected_file)

        # Display image
        st.image(image_path, caption=selected_display_name, use_container_width=True)

    with tab3:
        st.write("Feature Importance DataFrame")
        dataframe_folder = "output/FI_Dataframes/Integrated_Gradients"
        # Get all .csv files
        dataframe_files = [f for f in os.listdir(dataframe_folder) if f.endswith(".csv")]
        dataframe_files.sort()
        # Function to format image names
        def format_dataframe_name(filename):
            name_part, rest = filename.split("_IG_", 1)
            rest = rest.replace("withcorr", "with correlation")
            rest = rest.replace("without_corr", "without correlation")
            
            # Remove underscore after DT or RF
            rest = rest.replace("FNN_", "FNN ").replace("lstm_", "LSTM ")
            
            return f"{name_part} - {rest.replace('.csv', '')}"

        # Create a mapping from display name to file
        display_names = [format_dataframe_name(f) for f in dataframe_files]
        display_to_file = dict(zip(display_names, dataframe_files))

        # Dropdown menu
        selected_display_name = st.selectbox("Choose a dataframe to display:", display_names)

        # Get the actual file path
        selected_file = display_to_file[selected_display_name]
        dataframe_path = os.path.join(dataframe_folder, selected_file)
        # Display dataframe
        df = pd.read_csv(dataframe_path)
        st.dataframe(df)

elif choice == "WINit":
    st.header("WINit")
    output_path = "output/FI_Dataframes/WINit.csv"
    tab1, tab2, tab3 = st.tabs(["Graph", "Plot", "Feature Importance"])
    with tab1:
        st.write("Graph")
        st.image("output/FI_Comparison_Plots/comparison_plot_WinIT.png")
        # Load your CSV (assuming it's local)
        df = pd.read_csv("output/FI_Comparison_Results/Comparison_Results_WinIT.csv")

        # Extract only the relevant columns
        simplified_df = df[["Technique", "Top Features"]]

        # Display the table
        st.title("Technique and Top Features")
        st.dataframe(simplified_df, hide_index=True, use_container_width=False, height=600, width=6000)
    with tab2:
        st.write("WINit Plot")
        image_folder = "output/FI_Plots/WinIT"
        # Get all .png files
        image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
        image_files.sort()

        # Function to format image names
        def format_image_name(filename):
            name_part, rest = filename.split("_WinIT_", 1)
            rest = rest.replace("WithCorr", "with correlation")
            rest = rest.replace("without_corr", "without correlation")
            
            
            rest = rest.replace("XGB_", "XGB ").replace("LSTM_", "LSTM ")
            
            return f"{name_part} - {rest.replace('.png', '')}"

        # Create a mapping from display name to file
        display_names = [format_image_name(f) for f in image_files]
        display_to_file = dict(zip(display_names, image_files))

        # Dropdown menu
        selected_display_name = st.selectbox("Choose a plot to display:", display_names)

        # Get the actual file path
        selected_file = display_to_file[selected_display_name]
        image_path = os.path.join(image_folder, selected_file)

        # Display image
        st.image(image_path, caption=selected_display_name, use_container_width=True)

    with tab3:
        st.write("Feature Importance DataFrame")
        dataframe_folder = "output/FI_Dataframes/WinIT"
        # Get all .csv files
        dataframe_files = [f for f in os.listdir(dataframe_folder) if f.endswith(".csv")]
        dataframe_files.sort()
        # Function to format image names
        def format_dataframe_name(filename):
            name_part, rest = filename.split("_WinIT_", 1)
            rest = rest.replace("WithCorr", "with correlation")
            rest = rest.replace("without_corr", "without correlation")
            
            
            rest = rest.replace("XGB_", "XGB ").replace("LSTM_", "LSTM ")
            
            return f"{name_part} - {rest.replace('.csv', '')}"

        # Create a mapping from display name to file
        display_names = [format_dataframe_name(f) for f in dataframe_files]
        display_to_file = dict(zip(display_names, dataframe_files))

        # Dropdown menu
        selected_display_name = st.selectbox("Choose a dataframe to display:", display_names)

        # Get the actual file path
        selected_file = display_to_file[selected_display_name]
        dataframe_path = os.path.join(dataframe_folder, selected_file)
        # Display dataframe
        df = pd.read_csv(dataframe_path)
        st.dataframe(df)


elif choice == "LIME":
    st.header("LIME")
    output_path = "output/FI_Dataframes/LIME.csv"

    tab1, tab2, tab3 = st.tabs(["Graph", "Plot", "Feature Importance"])
    with tab1:
        st.write("Graph")
        st.image("output/FI_Comparison_Plots/comparison_plot_LIME.png")
        # Load your CSV (assuming it's local)
        df = pd.read_csv("output/FI_Comparison_Results/Comparison_Results_LIME.csv")

        # Extract only the relevant columns
        simplified_df = df[["Technique", "Top Features"]]

        # Display the table
        st.title("Technique and Top Features")
        st.dataframe(simplified_df, hide_index=True, use_container_width=False, height=600, width=6000)
    with tab2:
        st.write("LIME Plot")
        image_folder = "output/FI_Plots/LIME"
        # Get all .png files
        image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
        image_files.sort() 

        # Function to format image names
        def format_image_name(filename):
            name_part, rest = filename.split("_lime_", 1)
            rest = rest.replace("withcorr", "with correlation")
            rest = rest.replace("without_corr", "without correlation")
            
            # Remove underscore after DT or RF
            rest = rest.replace("rf_", "RF ").replace("lstm_", "LSTM ").replace("xgb_", "XGB ")
            
            return f"{name_part} - {rest.replace('.png', '')}"

        # Create a mapping from display name to file
        display_names = [format_image_name(f) for f in image_files]
        display_to_file = dict(zip(display_names, image_files))

        # Dropdown menu
        selected_display_name = st.selectbox("Choose a plot to display:", display_names)

        # Get the actual file path
        selected_file = display_to_file[selected_display_name]
        image_path = os.path.join(image_folder, selected_file)

        # Display image
        st.image(image_path, caption=selected_display_name, use_container_width=True)

    with tab3:
        st.write("Feature Importance DataFrame")
        dataframe_folder = "output/FI_Dataframes/LIME"
        # Get all .csv files
        dataframe_files = [f for f in os.listdir(dataframe_folder) if f.endswith(".csv")]
        dataframe_files.sort()
        # Function to format image names
        def format_dataframe_name(filename):
            name_part, rest = filename.split("_lime_", 1)
            rest = rest.replace("withcorr", "with correlation")
            rest = rest.replace("without_corr", "without correlation")
            
            # Remove underscore after DT or RF
            rest = rest.replace("rf_", "RF ").replace("lstm_", "LSTM ").replace("xgb_", "XGB ")
            
            return f"{name_part} - {rest.replace('.csv', '')}"

        # Create a mapping from display name to file
        display_names = [format_dataframe_name(f) for f in dataframe_files]
        display_to_file = dict(zip(display_names, dataframe_files))

        # Dropdown menu
        selected_display_name = st.selectbox("Choose a dataframe to display:", display_names)

        # Get the actual file path
        selected_file = display_to_file[selected_display_name]
        dataframe_path = os.path.join(dataframe_folder, selected_file)
        # Display dataframe
        df = pd.read_csv(dataframe_path)
        st.dataframe(df)