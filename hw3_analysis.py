import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
t1 = pd.read_csv("Data/t1_user_active_min.csv")
t2 = pd.read_csv("Data/t2_user_variant.csv")
t3 = pd.read_csv("Data/t3_user_active_min_pre.csv")
t4 = pd.read_csv("Data/t4_user_attributes.csv")


# Part 2: Organizing Data (Merging t1 and t2)
merged_data = t1.merge(t2[['uid', 'variant_number']], on='uid', how='left')
merged_data.to_csv("organized_user_activity.csv", index=False)

# Part 3: Statistical Analysis (Mean, Median, T-Test)
group_stats = merged_data.groupby("variant_number")["active_mins"].agg(["mean", "median"])
control_group = merged_data[merged_data["variant_number"] == 0]["active_mins"].dropna()
treatment_group = merged_data[merged_data["variant_number"] == 1]["active_mins"].dropna()
t_stat, p_value = stats.ttest_ind(control_group, treatment_group, equal_var=False)

# Part 4: Outlier Detection & Removal
Q1 = merged_data["active_mins"].quantile(0.25)
Q3 = merged_data["active_mins"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
cleaned_data = merged_data[(merged_data["active_mins"] >= lower_bound) & (merged_data["active_mins"] <= upper_bound)]
cleaned_data.to_csv("cleaned_user_activity.csv", index=False)

# Recalculate Statistics After Removing Outliers
cleaned_group_stats = cleaned_data.groupby("variant_number")["active_mins"].agg(["mean", "median"])
cleaned_control_group = cleaned_data[cleaned_data["variant_number"] == 0]["active_mins"].dropna()
cleaned_treatment_group = cleaned_data[cleaned_data["variant_number"] == 1]["active_mins"].dropna()
cleaned_t_stat, cleaned_p_value = stats.ttest_ind(cleaned_control_group, cleaned_treatment_group, equal_var=False)

# Part 5: Pre-Experiment Analysis (Merging t3 with t2)
pre_experiment_data = t3.merge(t2[['uid', 'variant_number']], on='uid', how='left')
pre_experiment_stats = pre_experiment_data.groupby("variant_number")["active_mins"].agg(["mean", "median"])
pre_control_group = pre_experiment_data[pre_experiment_data["variant_number"] == 0]["active_mins"].dropna()
pre_treatment_group = pre_experiment_data[pre_experiment_data["variant_number"] == 1]["active_mins"].dropna()
pre_t_stat, pre_p_value = stats.ttest_ind(pre_control_group, pre_treatment_group, equal_var=False)

# Part 6: User Attribute Analysis (Merging t4 with cleaned data)
merged_attributes_data = cleaned_data.merge(t4, on='uid', how='left')
attribute_group_stats = merged_attributes_data.groupby(["user_type", "gender"])["active_mins"].agg(["mean", "median"])

# Visualization: Box Plot
plt.figure(figsize=(8, 6))
sns.boxplot(x=merged_data["variant_number"], y=merged_data["active_mins"])
plt.xticks(ticks=[0, 1], labels=["Control", "Treatment"])
plt.xlabel("Experiment Group")
plt.ylabel("Active Minutes")
plt.title("Box Plot of Active Minutes by Group")
plt.savefig("box_plot.png")
plt.show()

# Save analysis results
with open("analysis_results.txt", "w") as f:
    f.write("Initial T-Test:\n")
    f.write(f"T-Statistic: {t_stat}, P-Value: {p_value}\n\n")
    f.write("Cleaned Data T-Test:\n")
    f.write(f"T-Statistic: {cleaned_t_stat}, P-Value: {cleaned_p_value}\n\n")
    f.write("Pre-Experiment T-Test:\n")
    f.write(f"T-Statistic: {pre_t_stat}, P-Value: {pre_p_value}\n")
