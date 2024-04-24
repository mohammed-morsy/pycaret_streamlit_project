import streamlit as st
from dataloader import DataLoader
from datavisualizer import DataVisualizer
from preprocessing import clean_data, get_problem_type
from pycaret import regression, classification
        
def main():    
    st.title("Auto Machine Learning & Data Exploration App")

    # Initialize DataLoader
    data_loader = DataLoader()

    # Load data
    data = data_loader.load_data()

    if data is not None:
        # st.write("Loaded Data", data)
        
        # Display checkbox widget to select columns
        st.sidebar.subheader("## Select Columns")
        selected_columns = st.sidebar.multiselect("Columns", data.columns, default=data.columns.tolist())
        target_column_options = selected_columns
        target_column = st.selectbox("Select the target column:", 
                                     options=target_column_options, 
                                     index=len(target_column_options)-1)
        tmp_data = data[selected_columns]
        cleaned_data = clean_data(tmp_data, target_column)
        problem_type = get_problem_type(cleaned_data, target_column)

        st.session_state.processed_data = cleaned_data
        st.session_state.problem_type = get_problem_type(cleaned_data, target_column)
        st.write("Loaded Dataset:\n", data, "\nSelected Columns:\n", st.session_state.processed_data)
        st.write("Problem Type:", st.session_state.problem_type)

        actions = {"Exploratory Data Analysis": eda, "Supervised Learning": model_building, "Download": download_model}
        st.sidebar.subheader('## Options')
        choice = st.sidebar.radio("Options", ["Exploratory Data Analysis","Supervised Learning", "Download"])
        if choice == "Exploratory Data Analysis":
            eda(cleaned_data)
        elif choice == "Supervised Learning":
            model_building(cleaned_data, problem_type, target_column)
        else:
            download_model(download_model)
            



def model_building(data, problem_type, target_column):
    with st.spinner(text="Performing Model Building..."):
        # Setup PyCaret
        if problem_type == "Regression":
            s = regression.setup(data, target=target_column, remove_outliers=True)
            # Compare Models
            best_model = regression.compare_models()
            # Evaluate Model
            regression.evaluate_model(best_model)
        
            # Deploy Model
            model = regression.finalize_model(best_model)
            st.write(regression.pull())
        else:
            s = classification.setup(data, target=target_column, remove_outliers=True)
            # Compare Models
            best_model = classification.compare_models()
            # Evaluate Model
            classification.evaluate_model(best_model)
        
            # Deploy Model
            model = classification.finalize_model(best_model)
            st.write(classification.pull())
            
            
    st.write("Model Building Completed.")

def download_model(problem_type):
    st.write("Downloading Model...")
    if problem_type == "regression":
        best = regression.compare_models()
        st.markdown(regression.save_model(best, 'my_first_model'), unsafe_allow_html=True)
    else:
        best = classification.compare_models()
        st.markdown(classification.save_model(best, 'my_first_model'), unsafe_allow_html=True)
    st.write("Model Downloaded.")


# Define functions for different actions
def eda(data):
    st.write("Performing Exploratory Data Analysis...")
    st.subheader("Data Summary:")
    
    if data.select_dtypes(include=['number']).empty:
        st.warning("No numerical columns found.")
    else:
        describe = data.describe()
        st.write("Numerical Columns", describe)

    if data.select_dtypes(include=['object']).empty:
        st.warning("No categorical columns found.")
    else:
        describe_cat = data.describe(include="object")
        st.write("Categorical Columns", describe_cat)

            
    # Visualizations
    st.subheader("Customized Visualizations:")
    
    visualizer = DataVisualizer(data)
    plot_type = st.selectbox("Select plot type:", ["Scatter Plot", "Histogram", "Box Plot", "Bar Plot", "Heatmap", "Line Plot"])
    
    if plot_type == "Scatter Plot" or plot_type == "Box Plot" or plot_type == "Bar Plot":
        x_variable = st.selectbox("Select x variable:", data.columns)
        y_variable = st.selectbox("Select y variable:", data.columns)
        getattr(visualizer, plot_type.lower().replace(" ", "_"))(x_variable, y_variable)
    elif plot_type == "Line Plot":
        x_variable = st.selectbox("Select x variable for line plot:", data.columns)
        y_variable = st.selectbox("Select y variable for line plot:", data.columns)
        scale = st.selectbox("Select scale:", [None, "linear", "log", "symlog", "logit"])
        grid = st.checkbox("Show grid lines", value=False)
        visualizer.line_plot(x_variable, y_variable, scale=scale, grid=grid)
    elif plot_type == "Histogram":
        selected_variable = st.selectbox("Select variable for histogram:", data.columns)
        bins = st.slider("Number of bins:", min_value=1, max_value=100, value=10)
        visualizer.histogram(selected_variable, bins)
    elif plot_type == "Heatmap":
        visualizer.heatmap(data.columns)
    # Allow users to select a PyCaret dataset
    problem_types = ["Classification (Binary)", "Classification (Multiclass)", "Regression"]
    


if __name__ == "__main__":
    main()
