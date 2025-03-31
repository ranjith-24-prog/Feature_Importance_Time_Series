import time
import pandas as pd
from prepare_data import prepare_data_with_correlation,prepare_data_with_correlation_PI, prepare_data_without_correlation_x_y_scaled,prepare_data_without_correlation
from Integrated_Gradient.IG_LSTM_with_correlation import process_dataset_with_integrated_gradients
from Integrated_Gradient.IG_FNN_with_correlation import process_dataset_with_IG_FNN_with_corr
from Integrated_Gradient.IG_FNN_without_correlation import process_dataset_with_IG_FNN_without_corr
from Integrated_Gradient.IG_LSTM_without_correlation import process_dataset_with_IG_lstm_without_corr
from Permutation_Importance.PI_DT_with_correlation import process_dataset_with_PI_DT
from Permutation_Importance.PI_RF_with_correlation import process_dataset_with_PI_RF
from Permutation_Importance.PI_DT_without_correlation import process_dataset_with_PI_DT_without_corr
from Permutation_Importance.PI_RF_without_correlation import process_dataset_with_PI_RF_without_corr
from LIME.LIME_LSTM_with_correlation import process_dataset_with_lime_agg_lstm
from LIME.LIME_LSTM_without_correlation import process_dataset_with_lime_agg_lstm_without_corr
from LIME.LIME_RF_with_correlation import process_dataset_with_lime_agg_rf
from LIME.LIME_RF_without_correlation import process_dataset_with_lime_agg_rf_without_corr
from LIME.LIME_XGB_with_correlation import process_dataset_with_lime_agg_xgb
from LIME.LIME_XGB_without_correlation import process_dataset_with_lime_agg_xgb_without_corr
from WINIT.WINIT_LSTM_with_correlation import process_dataset_with_winit_lstm_with_corr
from WINIT.WINIT_LSTM_without_correlation import process_dataset_with_winit_lstm_without_corr
from WINIT.WINIT_XGB_with_correlation import process_dataset_with_winit_xgb_with_corr
from WINIT.WINIT_XGB_without_correlation import process_dataset_with_winit_xgb_without_corr


dataset_paths = [
    "static/Dataset/DMC2_AL_CP1.csv",
    "static/Dataset/DMC2_AL_CP2.csv",
    "static/Dataset/DMC2_S_CP1.csv",
    "static/Dataset/DMC2_S_CP2.csv",
]


techniques = {
    "Integrated Gradients lstm with correlation": process_dataset_with_integrated_gradients,
    "Integrated Gradients fnn with correlation": process_dataset_with_IG_FNN_with_corr,
    "Integrated Gradients lstm without correlation": process_dataset_with_IG_lstm_without_corr,
    "Integrated Gradients fnn without correlation": process_dataset_with_IG_FNN_without_corr,
    "Permutation Importance DT with correlation": process_dataset_with_PI_DT,
    "Permutation Importance RF with correlation": process_dataset_with_PI_RF,
    "Permutation Importance DT without correlation": process_dataset_with_PI_DT_without_corr,
    "Permutation Importance RF without correlation": process_dataset_with_PI_RF_without_corr,
    "LIME LSTM with correlation": process_dataset_with_lime_agg_lstm,
    "LIME RF with correlation": process_dataset_with_lime_agg_rf,
    "LIME XGB with correlation": process_dataset_with_lime_agg_xgb,
    "LIME LSTM without correlation": process_dataset_with_lime_agg_lstm_without_corr,
    "LIME RF without correlation": process_dataset_with_lime_agg_rf_without_corr,
    "LIME XGB without correlation": process_dataset_with_lime_agg_xgb_without_corr,
    "WINIT LSTM with correlation": process_dataset_with_winit_lstm_with_corr,
    "WINIT XGB with correlation": process_dataset_with_winit_xgb_with_corr,
    "WINIT LSTM without correlation": process_dataset_with_winit_lstm_without_corr,
    "WINIT XGB without correlation": process_dataset_with_winit_xgb_without_corr
}

results = []

for dataset_path in dataset_paths:
    print(f"\n Processing dataset: {dataset_path}")

    for technique_name, technique_func in techniques.items():
        start_time = time.time()
        if 'without correlation' in technique_name:
            if 'LIME' in technique_name or 'WINIT' in technique_name or 'fnn' in technique_name or 'XGB' in technique_name:
                technique_results = technique_func(dataset_path, prepare_data_without_correlation_x_y_scaled)
            else:
                technique_results = technique_func(dataset_path, prepare_data_without_correlation)
            variation = "Without Correlation"
        else:
            if 'Permutation' in technique_name:
                technique_results = technique_func(dataset_path, prepare_data_with_correlation_PI)
            else:
                technique_results = technique_func(dataset_path, prepare_data_with_correlation)
            variation = "With Correlation"
        execution_time = time.time() - start_time

        results.append({
            "Dataset": dataset_path,
            "Technique": technique_name,
            "Variation": variation,
            "Test Loss": technique_results["test_loss"],
            "Top Features": ", ".join(technique_results["top_features"]),
            "Execution Time (Seconds)": execution_time
        })

        print(f" {technique_name} Done for {dataset_path} in {execution_time:.2f} sec")

results_df = pd.DataFrame(results)

output_path = f"output/FI_Comparison_Results/Comparison_Results_{technique_name}.csv"
results_df.to_csv(output_path, index=False)
print(f"\n Comparison results saved to {output_path}")