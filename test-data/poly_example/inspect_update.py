import pickle

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ert.storage import open_storage


def create_waterfall_chart(data_array, parameter_name, nresponses, title=""):
    # Extract data for the selected parameter and remove zeros
    parameter_data = data_array.sel(names=parameter_name).values
    column_labels = data_array.coords["response_index"].values
    non_zero_indices = np.nonzero(parameter_data)[0]
    non_zero_data = parameter_data[non_zero_indices]
    non_zero_labels = column_labels[non_zero_indices]

    # Calculate the base value as the sum of all values
    base_value = non_zero_data.sum()
    # Compute percentage changes
    percentage_changes = (non_zero_data / base_value) * 100

    # Sort by absolute value and limit to nresponses
    sorted_indices = np.abs(percentage_changes).argsort()[::-1]
    if len(percentage_changes) > nresponses:
        sorted_data = percentage_changes[sorted_indices][:nresponses]
        sorted_labels = non_zero_labels[sorted_indices][:nresponses]

        # Bundle remaining responses
        bundle_value = percentage_changes[sorted_indices][nresponses:].sum()
        sorted_data = np.append(sorted_data, bundle_value)
        sorted_labels = np.append(sorted_labels, "Bundle-Response")
    else:
        sorted_data = percentage_changes[sorted_indices]
        sorted_labels = non_zero_labels[sorted_indices]

    # Create the waterfall chart
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative"] * len(sorted_data),
            x=sorted_labels,
            textposition="outside",
            y=sorted_data,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Response Index"),
        yaxis=dict(title="Percentage Contribution", tickformat=".2f%"),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


# Set up the title of the web app
st.title("Parameter Update Decompoisition")
st.subheader("Using estimated Kalman-Gain")

# Open storage
storage = open_storage("storage")

# Fetch the list of ensemble names
ensemble_names = [x.name for x in storage.ensembles]

# Create a select box in the sidebar for "storage_prior_name"
storage_prior_name = st.sidebar.selectbox(
    "Select Storage Prior Name",
    ensemble_names,
    index=0,  # Default to the first ensemble name
)

# Create another select box in the sidebar for "storage_posterior_name"
storage_posterior_name = st.sidebar.selectbox(
    "Select Storage Posterior Name",
    ensemble_names,
    index=1,  # Default to the second ensemble name, change as needed
)

# Fetch the selected ensembles after selection
storage_prior = storage.get_ensemble_by_name(storage_prior_name)
storage_posterior = storage.get_ensemble_by_name(storage_posterior_name)

# Placeholder to confirm selections
# st.write(f"Selected Prior: {storage_prior_name}")
# st.write(f"Selected Posterior: {storage_posterior_name}")


# Load K_lasso_param_groups from a pickle file
try:
    with open("K_lasso_full.pkl", "rb") as file:
        K_lasso_param_groups = pickle.load(file)
    # st.write("K_lasso_param_groups loaded successfully.")
except Exception as e:
    st.error(f"Failed to load K_lasso_param_groups: {e}")

# Check if K_lasso_param_groups is loaded and not empty
if "K_lasso_param_groups" in locals() and K_lasso_param_groups:
    # Extract parameter-group names from the dict keys
    param_group_names = list(K_lasso_param_groups.keys())

    # Create a select box in the sidebar for selecting a parameter-group
    selected_param_group = st.sidebar.selectbox(
        "Select Parameter Group", param_group_names
    )

    # If a parameter group is selected, create selection for "parameter_name"
    if selected_param_group:
        # Extract parameter names for the selected parameter group
        parameter_names = (
            K_lasso_param_groups[selected_param_group].coords["names"].values.tolist()
        )

        # Sidebar for selecting a parameter name
        selected_parameter_name = st.sidebar.selectbox(
            "Select Parameter Name", parameter_names
        )

        # Display the selected parameter name
        # st.write(f"Selected Parameter Name: {selected_parameter_name}")


# Sidebar option to select the number of responses to display
nresponses = st.sidebar.number_input(
    "Number of Responses to Display", min_value=1, value=10
)  # You can adjust the default value and range as needed


# Attempt to load the average innovations from innovations.pkl
try:
    with open("innovations.pkl", "rb") as file:
        innovations_xr = pickle.load(file)
    # st.write("Average innovations loaded successfully.")
except Exception as e:
    st.error(f"Failed to load average innovations: {e}")


# Scale K_lasso_xr by the aligned average innovations
K_lasso_xr = K_lasso_param_groups[selected_param_group]
K_lasso_xr_scaled = K_lasso_xr * innovations_xr.values

# Use the existing create_waterfall_chart function with the scaled K_lasso_xr
if "K_lasso_xr_scaled" in locals() and selected_parameter_name:
    create_waterfall_chart(
        K_lasso_xr_scaled,
        selected_parameter_name,
        nresponses,
        title=f"Expected Update for {selected_param_group}:{selected_parameter_name}",
    )


# Create two columns for the plots
col1, col2 = st.columns(2)

with col1:
    # Use the function with selected parameters and nresponses
    if (
        "K_lasso_param_groups" in locals()
        and selected_param_group
        and selected_parameter_name
    ):
        create_waterfall_chart(
            K_lasso_xr,
            selected_parameter_name,
            nresponses,
            title=f"Kalman Gain for {selected_param_group}:{selected_parameter_name}",
        )

with col2:
    # Assuming innovations_xr is loaded and nresponses is defined
    if "innovations_xr" in locals():
        # Convert to a Pandas Series for easier manipulation
        innovations_series = innovations_xr.to_series()

        # Sort by absolute value in descending order
        sorted_innovations = innovations_series.abs().sort_values(ascending=False)

        # Limit to nresponses
        limited_innovations = sorted_innovations.head(nresponses)

        # Ensure the dataframe has the correct column names after resetting the index
        limited_innovations_df = limited_innovations.reset_index()
        limited_innovations_df.columns = ["Observation", "Average Innovation"]

        # Now, create the Plotly figure with the corrected column names
        fig = px.bar(
            limited_innovations_df,
            x="Observation",  # Make sure this matches the name of the column in your dataframe
            y="Average Innovation",  # This should also match exactly
            labels={
                "Observation": "Observation",
                "Average Innovation": "Average Innovation",
            },
            title="Average Innovations (Top Responses)",
        )

        # Display the figure in your Streamlit app
        st.plotly_chart(fig, use_container_width=True)
