from joblib import dump, load
import time
import streamlit as st
from dataloader import DataLoader
from datavisualizer import DataVisualizer
from preprocessing import clean_data, get_problem_type
from pycaret import regression, classification
        
def main():    
    st.title("Auto Machine Learning & Data Exploration App")

    # Initialize DataLoader
    data_loader = DataLoader()
    st.session_state.models = {}

    # Load data
    data = data_loader.load_data()

    if data is not None:
        st.header("Loaded Dataset Sample:")
        st.write(data.sample(n=10))
        
        st.header("Action Menu")
        choice = st.radio("Pick an Action", ["Exploratory Data Analysis","Supervised Learning", "Download"], )
        
        # Display checkbox widget to select columns
        st.sidebar.subheader("Choose Columns")
        selected_columns = st.sidebar.multiselect("Available Columns", data.columns, default=data.columns.tolist())
        target_column_options = selected_columns
        target_column = st.sidebar.selectbox("Select the target column:", 
                                     options=target_column_options, 
                                     index=len(target_column_options)-1)
        tmp_data = data[selected_columns]
        cleaned_data = clean_data(tmp_data, target_column)
        problem_type = get_problem_type(cleaned_data, target_column)
        
        st.session_state.processed_data = cleaned_data
        st.session_state.problem_type = get_problem_type(cleaned_data, target_column)
        
        st.write("Problem Type:", st.session_state.problem_type)
        
        
        
        
        

        if choice == "Exploratory Data Analysis":
            eda()
        elif choice == "Supervised Learning":
            model_building(cleaned_data, problem_type, target_column)
        else:
            if "best_model" in st.session_state.keys():
                download_model()
            else: 
                st.error("There is no model to download")
            




def model_building(data, problem_type, target_column):
    st.header("Supervised Learning:")
    
    # Define available models based on problem type
    if problem_type == "Regression":
        available_models = ["lr", "lasso", "ridge", "rf", "knn", "dt", "svm", "lightgbm"]
    else:
        available_models = ["lr", "dt", "rf", "svm", "knn", "lightgbm"]

    # Allow user to select models
    selected_models = st.multiselect("Select models (2 to 3):", available_models, default=available_models[:2], key="selected_models")
    
    # Check if the number of selected models is within the allowed range
    if len(selected_models) < 2 or len(selected_models) > 3:
        st.warning("Please select between 2 and 3 models.")
        return

    # Show button to start model building
    if st.button("Start Model Building"):
        with st.spinner("Performing Model Building..."):
            # Setup PyCaret
            if problem_type == "Regression":
                regression.setup(data, target=target_column, remove_outliers=False, fold_strategy="stratifiedkfold")
                # Compare Models
                best_estimator = regression.compare_models(include=selected_models)
                # Evaluate Model
                regression.evaluate_model(best_estimator)
                # Deploy Model
                best_model = regression.finalize_model(best_estimator)
                estimators = regression.pull()
            else:
                classification.setup(data, target=target_column, remove_outliers=False, fold_strategy="stratifiedkfold")
                # Compare Models
                best_estimator = classification.compare_models(include=selected_models)
                # Evaluate Model
                classification.evaluate_model(best_estimator)
                # Deploy Model
                best_model = classification.finalize_model(best_estimator)
                estimators = classification.pull()

            # Display model information
            st.write(estimators)
            st.write(best_model)
            st.success("Model Building Completed.")
            st.session_state.best_model = best_model
            

import joblib

def download_model():
    st.header("Download: ")
    model_name = f"my_model_{int(time.time())}"
    with st.spinner(text="Downloading Model..."):        
        joblib.dump(st.session_state.best_model, model_name)
        with open(model_name, 'rb') as f:
            model_bytes = f.read()

        st.download_button(label="Download Model", data=model_bytes, file_name=model_name, mime="application/octet-stream")
        # Instructions to load the model
        st.markdown("""
        To load the downloaded model, please ensure that you have `joblib` version 1.3.2 installed. You can install it using pip:

        ```shell
        pip install joblib==1.3.2
        ```
        ```python
        import joblib
        ```
        Then, you can load the model in your Python script like this:
        ```python
        # Replace 'my_model.joblib' with the path to your downloaded model file
        model = joblib.load('my_model.joblib')
        ```
        """)




def eda():
    st.header("Exploratory Data Analysis:")
    
    st.write(st.session_state.processed_data)
    actions = ["Data Visualizations", "Data Summary"]
    eda_option = st.multiselect("Analysis Type:", actions, default=actions)
    
    if "Data Summary" in eda_option:
        st.subheader("\tData Summary:")
        
        if st.session_state.processed_data.select_dtypes(include=["number"]).empty:
            st.warning("No numerical columns found.")
        else:
            describe = st.session_state.processed_data.describe()
            st.write("Numerical Columns", describe)

        if st.session_state.processed_data.select_dtypes(include=["object"]).empty:
            st.warning("No categorical columns found.")
        else:
            describe_cat = st.session_state.processed_data.describe(include="object")
            st.write("Categorical Columns", describe_cat)

           
    # Visualizations
    if "Data Visualizations" in eda_option:
        st.subheader("\tCustomized Visualizations:")
        
        visualizer = DataVisualizer(st.session_state.processed_data)
        plot_type = st.selectbox("Select plot type:", ["Scatter Plot", "Histogram", "Box Plot", "Bar Plot", "Heatmap", "Line Plot"])
        
        if plot_type == "Scatter Plot" or plot_type == "Box Plot" or plot_type == "Bar Plot":
            x_variable = st.selectbox("Select x variable:", st.session_state.processed_data.columns)
            y_variable = st.selectbox("Select y variable:", st.session_state.processed_data.columns)
            grid = st.checkbox("Show grid lines", value=True)
            getattr(visualizer, plot_type.lower().replace(" ", "_"))(x_variable, y_variable, grid=bool(grid))
        elif plot_type == "Line Plot":
            x_variable = st.selectbox("Select x variable for line plot:", st.session_state.processed_data.columns)
            y_variable = st.selectbox("Select y variable for line plot:", st.session_state.processed_data.columns, )
            scale = st.selectbox("Select scale:", [None, "linear", "log", "symlog", "logit"])
            grid = st.checkbox("Show grid lines", value=True)
            visualizer.line_plot(x_variable, y_variable, scale=scale, grid=bool(grid))
        elif plot_type == "Histogram":
            selected_variable = st.selectbox("Select variable for histogram:", st.session_state.processed_data.columns)
            bins = st.slider("Number of bins:", min_value=1, max_value=100, value=10)
            grid = st.checkbox("Show grid lines", value=True)
            visualizer.histogram(selected_variable, bins, grid=bool(grid))
        elif plot_type == "Heatmap":
            visualizer.heatmap()    



if __name__ == "__main__":
    main()
