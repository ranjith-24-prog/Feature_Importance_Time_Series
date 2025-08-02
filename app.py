import streamlit as st
import pandas as pd
import os


if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

if "selected_technique" not in st.session_state:
    st.session_state.selected_technique = None


technique_descriptions = {
    "Integrated Gradients": """
        Integrated Gradients is a gradient-based attribution method designed for neural networks. It
        quantifies feature importance by integrating the gradients of a model’s predictions with respect
        to input features, computed along a straight-line path from a baseline input to the actual
        input. This approach satisfies axioms like sensitivity and implementation invariance, making
        it theoretically grounded and suitable for complex models such as LSTMs and FNNs. IG is
        particularly useful in capturing subtle interactions in data, especially in cases involving temporal
        sequences
    """,
    "WINit": """
        WINIT is a feature removal-based explainability method specifically designed for time-series
        data. It computes the importance of each observation by analyzing the impact on predictions
        over a temporal window. By aggregating effects over multiple time steps, WINIT effectively
        captures delayed influences and long-range dependencies that are typical in industrial processes.
        Its design makes it particularly suitable for applications where temporal causality is critical,
        such as in CNC energy analysis
    """,
    "LIME": """
        LIME is a local surrogate model approach that approximates the behavior of complex models
        by training interpretable models (like linear regressors or decision trees) on perturbed samples
        around a prediction instance. This enables the estimation of local feature importances in a
        model-agnostic fashion. For tabular data, LIME generates explanations by drawing samples
        from replacement distributions, making it flexible for use with a variety of model architectures
        including LSTM, RF, and XGBoost 
    """,
    "Permutation Importance": """
        Permutation Importance is a model-agnostic technique that evaluates feature importance by
        measuring the change in model performance when the values of a specific feature are randomly
        shuffled. This process breaks the association between the feature and the target variable,
        allowing assessment of the drop in predictive accuracy due to the feature’s absence. It is
        particularly suitable for models like decision trees and ensemble methods, and offers a robust,
        interpretable approach, although at a higher computational cost 
    """
}


if not st.session_state.analysis_started:

    current_theme = st.get_option("theme.base")  # returns 'light' or 'dark'

    if current_theme == "dark":
        title_color = "#FFFFFF"      # white for dark mode
        subtitle_color = "#CCCCCC"   # lighter gray for dark mode subtitle
    else:
        title_color = "#000000"      # black for light mode
        subtitle_color = "#666666"   # original gray for light mode subtitle

    st.markdown(f"""
    <style>
        .title {{
            font-size: 35px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 20px;
        }}
        .subtitle {{
            font-size: 18px;
            color: {subtitle_color} !important;
            text-align: center;
            margin-bottom: 40px;
        }}
    </style>
""", unsafe_allow_html=True)

    st.markdown('<div class="title">Feature Importance in Time Series For Energy Consumption in CNC Machine</div>', unsafe_allow_html=True)
    st.image('CNC_machine.jpeg')
    st.markdown("#### 🔍 Select a Technique to Learn More")
    selected = st.radio("", list(technique_descriptions.keys()), index=0)
    st.session_state.selected_technique = selected

    st.markdown(f"""
    <div class="info-box">
        <strong>About {selected}</strong><br>
        {technique_descriptions[selected]}
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ Start Analysis", use_container_width=True):
            st.session_state.analysis_started = True
            st.rerun()


if st.session_state.analysis_started and st.session_state.selected_technique:

    selected = st.session_state.selected_technique

    st.sidebar.title("Feature Importance Techniques")
    st.sidebar.markdown(f"**Selected:** {selected}")

    with st.sidebar:
        if st.button('Back'):
            st.session_state.analysis_started = False
            st.rerun()

    def format_filename(filename, split_on, model_prefixes):
        name_part, rest = filename.split(split_on, 1)
        rest = rest.replace("withcorr", "with correlation")
        rest = rest.replace("WithCorr", "with correlation")
        rest = rest.replace("without_corr", "without correlation")
        for prefix in model_prefixes:
            rest = rest.replace(prefix + "_", prefix + " ")
        return f"{name_part} - {rest.replace('.csv', '').replace('.png', '')}"

    if selected == "Permutation Importance":
        st.header("Permutation Importance")
        tab1, tab2, tab3 = st.tabs(["Graph", "Plot", "Feature Importance"])
        with tab1:
            st.image("output/FI_Comparison_Plots/comparison_plot_PI.png")
            df = pd.read_csv("output/FI_Comparison_Results/Comparison_Results_PI.csv")
            st.header('Technique and Top Features')
            df["Dataset"] = df["Dataset"].str.extract(r"(DMC2_.*?).csv")
            st.dataframe(df[["Dataset","Technique", "Top Features"]], hide_index=True)

        with tab2:
            image_folder = "output/FI_Plots/Permutation"
            image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
            display_names = [format_filename(f, "_PI_", ["DT", "RF"]) for f in image_files]
            selected_file = dict(zip(display_names, image_files))[st.selectbox("Choose a plot to display:", display_names)]
            st.image(os.path.join(image_folder, selected_file), caption=selected_file, use_container_width=True)

        with tab3:
            dataframe_folder = "output/FI_Dataframes/Permutation"
            dataframe_files = sorted([f for f in os.listdir(dataframe_folder) if f.endswith(".csv")])
            display_names = [format_filename(f, "_PI_", ["DT", "RF"]) for f in dataframe_files]
            selected_file = dict(zip(display_names, dataframe_files))[st.selectbox("Choose a dataframe to display:", display_names)]
            df = pd.read_csv(os.path.join(dataframe_folder, selected_file))
            st.dataframe(df, hide_index=True)

    elif selected == "Integrated Gradients":
        st.header("Integrated Gradients")
        tab1, tab2, tab3 = st.tabs(["Graph", "Plot", "Feature Importance"])
        with tab1:
            st.image("output/FI_Comparison_Plots/comparison_plot_IG.png")
            df = pd.read_csv("output/FI_Comparison_Results/Comparison_Results_IG.csv")
            st.header('Technique and Top Features')
            df["Dataset"] = df["Dataset"].str.extract(r"(DMC2_.*?).csv")
            st.dataframe(df[["Dataset","Technique", "Top Features"]], hide_index=True)

        with tab2:
            image_folder = "output/FI_Plots/Integrated_Gradients"
            image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
            display_names = [format_filename(f, "_IG_", ["FNN", "lstm"]) for f in image_files]
            selected_file = dict(zip(display_names, image_files))[st.selectbox("Choose a plot to display:", display_names)]
            st.image(os.path.join(image_folder, selected_file), caption=selected_file, use_container_width=True)

        with tab3:
            dataframe_folder = "output/FI_Dataframes/Integrated_Gradients"
            dataframe_files = sorted([f for f in os.listdir(dataframe_folder) if f.endswith(".csv")])
            display_names = [format_filename(f, "_IG_", ["FNN", "lstm"]) for f in dataframe_files]
            selected_file = dict(zip(display_names, dataframe_files))[st.selectbox("Choose a dataframe to display:", display_names)]
            df = pd.read_csv(os.path.join(dataframe_folder, selected_file))
            st.dataframe(df, hide_index=True)

    elif selected == "WINit":
        st.header("WINit")
        tab1, tab2, tab3 = st.tabs(["Graph", "Plot", "Feature Importance"])
        with tab1:
            st.image("output/FI_Comparison_Plots/comparison_plot_WinIT.png")
            df = pd.read_csv("output/FI_Comparison_Results/Comparison_Results_WinIT.csv")
            st.header('Technique and Top Features')
            df["Dataset"] = df["Dataset"].str.extract(r"(DMC2_.*?).csv")
            st.dataframe(df[["Dataset","Technique", "Top Features"]], hide_index=True)

        with tab2:
            image_folder = "output/FI_Plots/WinIT"
            image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
            display_names = [format_filename(f, "_WinIT_", ["XGB", "LSTM"]) for f in image_files]
            selected_file = dict(zip(display_names, image_files))[st.selectbox("Choose a plot to display:", display_names)]
            st.image(os.path.join(image_folder, selected_file), caption=selected_file, use_container_width=True)

        with tab3:
            dataframe_folder = "output/FI_Dataframes/WinIT"
            dataframe_files = sorted([f for f in os.listdir(dataframe_folder) if f.endswith(".csv")])
            display_names = [format_filename(f, "_WinIT_", ["XGB", "LSTM"]) for f in dataframe_files]
            selected_file = dict(zip(display_names, dataframe_files))[st.selectbox("Choose a dataframe to display:", display_names)]
            df = pd.read_csv(os.path.join(dataframe_folder, selected_file))
            st.dataframe(df, hide_index=True)

    elif selected == "LIME":
        st.header("LIME")
        tab1, tab2, tab3 = st.tabs(["Graph", "Plot", "Feature Importance"])
        with tab1:
            st.image("output/FI_Comparison_Plots/comparison_plot_LIME.png")
            df = pd.read_csv("output/FI_Comparison_Results/Comparison_Results_LIME.csv")
            st.header('Technique and Top Features')
            df["Dataset"] = df["Dataset"].str.extract(r"(DMC2_.*?).csv")
            st.dataframe(df[["Dataset","Technique", "Top Features"]], hide_index=True)

        with tab2:
            image_folder = "output/FI_Plots/LIME"
            image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
            display_names = [format_filename(f, "_lime_", ["rf", "lstm", "xgb"]) for f in image_files]
            selected_file = dict(zip(display_names, image_files))[st.selectbox("Choose a plot to display:", display_names)]
            st.image(os.path.join(image_folder, selected_file), caption=selected_file, use_container_width=True)

        with tab3:
            dataframe_folder = "output/FI_Dataframes/LIME"
            dataframe_files = sorted([f for f in os.listdir(dataframe_folder) if f.endswith(".csv")])
            display_names = [format_filename(f, "_lime_", ["rf", "lstm", "xgb"]) for f in dataframe_files]
            selected_file = dict(zip(display_names, dataframe_files))[st.selectbox("Choose a dataframe to display:", display_names)]
            df = pd.read_csv(os.path.join(dataframe_folder, selected_file))
            st.dataframe(df, hide_index=True)
