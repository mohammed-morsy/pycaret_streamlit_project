import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

class DataVisualizer:
    """
    A class for visualizing data using various plots.
    """

    def __init__(self, data):
        """
        Initializes the DataVisualizer class with the provided data.

        Parameters:
        - data (DataFrame): The dataset to visualize.
        """
        self.data = data

    def scatter_plot(self, x_variable, y_variable):
        """
        Create a scatter plot.

        Parameters:
        - x_variable (str): Name of the variable for the x-axis.
        - y_variable (str): Name of the variable for the y-axis.
        """
        fig, ax = plt.subplots()
        sns.scatterplot(x=x_variable, y=y_variable, data=self.data)
        st.pyplot(fig)

    def histogram(self, variable, bins):
        """
        Create a histogram.

        Parameters:
        - variable (str): Name of the variable for the histogram.
        """
        fig, ax = plt.subplots()
        sns.histplot(self.data[variable], bins = bins)
        st.pyplot(fig)

    def box_plot(self, x_variable, y_variable):
        """
        Create a box plot.

        Parameters:
        - x_variable (str): Name of the variable for the x-axis.
        - y_variable (str): Name of the variable for the y-axis.
        """
        fig, ax = plt.subplots()
        sns.boxplot(x=x_variable, y=y_variable, data=self.data)
        st.pyplot(fig)

    def bar_plot(self, x_variable, y_variable):
        """
        Create a bar plot.

        Parameters:
        - x_variable (str): Name of the variable for the x-axis.
        - y_variable (str): Name of the variable for the y-axis.
        """
        fig, ax = plt.subplots()
        sns.barplot(x=x_variable, y=y_variable, data=self.data)
        st.pyplot(fig)

    def heatmap(self, variables):
        """
        Create a heatmap.

        Parameters:
        - variables (list): List of variables for the heatmap.
        """
        st.write("Select variables for the heatmap:")
        heatmap_variables = st.multiselect("Select variables:", self.data.columns)
        if heatmap_variables:
            heatmap_data = self.data[heatmap_variables].corr()
            fig, ax = plt.subplots()
            sns.heatmap(heatmap_data, annot=True, cmap="coolwarm")
            st.pyplot(fig)
        else:
            st.warning("Please select at least one variable for the heatmap.")

    def line_plot(self, x_variable, y_variable, scale=None, grid=False):
        """
        Create a line plot.

        Parameters:
        - x_variable (str): Name of the variable for the x-axis.
        - y_variable (str): Name of the variable for the y-axis.
        - scale (str or None): Scaling of the plot (e.g., "linear", "log", "symlog", "logit").
        - grid (bool): Whether to display grid lines.
        """
        fig, ax = plt.subplots()
        if scale:
            ax.set_yscale(scale)
        if grid:
            ax.grid(True)
        sns.lineplot(x=x_variable, y=y_variable, data=self.data)
        st.pyplot(fig)
