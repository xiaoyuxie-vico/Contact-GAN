# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Dec, 2022
'''

import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd
import streamlit as st
import seaborn as sns
import mpld3
import streamlit.components.v1 as components

np.set_printoptions(precision=2)

matplotlib.use('agg')


def main():
    apptitle = 'Contact Optimization'
    st.set_page_config(
        page_title=apptitle,
        page_icon=':eyeglasses:',
        # layout='wide'
    )
    st.title('Welcome to Contact Optimization via Mechanistic Machine Learning')
    st.markdown('<p class="L2">- Developer: \nXiaoyu Xie, Northwestern University</p>', unsafe_allow_html=True)
    st.markdown('<p class="L2">- Date: December, 2022.</p>', unsafe_allow_html=True)
    st.image('images/schematic.png')

    # level 1 font
    st.markdown("""
        <style>
        .L1 {
            font-size:40px !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # level 2 font
    st.markdown("""
        <style>
        .L2 {
            font-size:20px !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    #########################Objectives#########################
    st.markdown('<p class="L1">Objectives</p>', unsafe_allow_html=True)
    str_1 = '''1. Background: The contact region plays an important role in the convergence and accuracy of the simulation. IT IS EXTREMELY HARD TO GET THE SOLUTION CONVERGE.'''
    st.markdown('<p class="L2">{}</p>'.format(str_1), unsafe_allow_html=True)

    str_1 = '''2. Goal: To avoid trial and error, we propose a mechanistic machine learning-based inverse approach to optimize parameters for surface-to-surface contact'''
    st.markdown('<p class="L2">{}</p>'.format(str_1), unsafe_allow_html=True)

    #########################Data generation#########################
    st.markdown('<p class="L1">Data Generation</p>', unsafe_allow_html=True)
    str_1 = '''To generate a database, we applied Finite Element Analysis software (Abaqus) to generate simulation data with different concate confirguations.'''
    st.markdown('<p class="L2">{}</p>'.format(str_1), unsafe_allow_html=True)
    st.image('images/data_generation.png')

    #########################Load dataset#########################
    st.markdown('<p class="L1">Load dataset</p>', unsafe_allow_html=True)
    st.markdown('<p class="L2">Upload simulation data</p>', unsafe_allow_html=True)

    data_str_1 = 'Default dataset (different angles)'
    data_str_2 = 'Default dataset (different loadings)'
    flag = ['New dataset', data_str_1, data_str_2]
    # use_new_data = st.selectbox('Chosse a new dataset or use default dataset', flag, 1)

    use_new_data = st.sidebar.selectbox('Chosse a new dataset or use default dataset', flag, 1)

    # load dataset
    if use_new_data == 'New dataset':
        uploaded_file = st.file_uploader('Choose a CSV file', accept_multiple_files=False)
    
    # load dataset
    if use_new_data == 'New dataset':
        data = io.BytesIO(uploaded_file.getbuffer())
        df = pd.read_csv(data)
    elif use_new_data == data_str_1:
        file_path = 'dataset/dataset_different_angle.csv'
        df = pd.read_csv(file_path)
    elif use_new_data == data_str_2:
        file_path = 'dataset/dataset_different_loading.csv'
        df = pd.read_csv(file_path)
    
    st.markdown('Please upload your own dataset in a cvs file.')

    #########################Data visualization#########################
    df_norm = (df - df.mean()) / df.std()

    st.markdown('<p class="L1">Data Visualization</p>', unsafe_allow_html=True)

    st.markdown('Visualize the dataset:')
    st.dataframe(df)
    
    st.markdown('Plot the dataset:')
    if use_new_data == data_str_1:
        fig = plt.figure()
        ax = sns.lineplot(data=df, x='Operation_angle', y='Maximum_displacement', hue='Contact_length', marker='o')
        plt.xlabel('Operation angle', fontsize=16)
        plt.ylabel('Maximum displacement', fontsize=16)
        plt.setp(ax.get_legend().get_texts(), fontsize='14')
        plt.setp(ax.get_legend().get_title(), fontsize='16')
        plt.tight_layout()
        # st.pyplot(fig, clear_figure=True)
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=600)
    elif use_new_data == data_str_2:
        fig = plt.figure()
        ax = sns.lineplot(data=df, x='Lifting_load', y='Maximum_displacement', hue='Contact_length', marker='o')
        plt.xlabel('Lifting load', fontsize=16)
        plt.ylabel('Maximum displacement', fontsize=16)
        plt.setp(ax.get_legend().get_texts(), fontsize='14')
        plt.setp(ax.get_legend().get_title(), fontsize='16')
        plt.tight_layout()
        # st.pyplot(fig, clear_figure=True)
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=600)

    #########################Model training#########################
    st.markdown('<p class="L1">Mechanistic Machine Learning Model Training</p>', unsafe_allow_html=True)
    st.image('images/model.png')

    st.markdown('**Hyperparameters:**')
    st.markdown('- The maximum decision tree depth is 3.')
    st.markdown('- The number of trees is 100.')
    st.markdown('- The loss function is mean squared error.')
    st.markdown('- The minimum number of samples required to split an internal node is 2.')

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    test_size = st.sidebar.slider("Model training: test ratio: ", 0.01, 1.0, 0.2)
    
    df_train, df_test = train_test_split(df_norm, test_size=test_size, random_state=5)
    if use_new_data == data_str_1:
        X_train, y_train = df_train[['Contact_length', 'Operation_angle']], df_train[['Maximum_displacement']]
        X_test, y_test = df_test[['Contact_length', 'Operation_angle']], df_test[['Maximum_displacement']]
    elif use_new_data == data_str_2:
        X_train, y_train = df_train[['Contact_length', 'Lifting_load']], df_train[['Maximum_Mises_stress', 'Maximum_displacement']]
        X_test, y_test = df_test[['Contact_length', 'Lifting_load']], df_test[['Maximum_Mises_stress', 'Maximum_displacement']]

    regr = RandomForestRegressor(max_depth=3).fit(X_train, y_train)

    pred_test = regr.predict(X_test)

    r2_train = regr.score(X_train, y_train)
    r2_test = regr.score(X_test, y_test)

    st.markdown('**Start training a mechanistic machine learning model:**')
    st.markdown('**Training results:**')
    st.markdown('- R2 score in the training set is **{:.4f}**.'.format(r2_train))
    st.markdown('- R2 score in the test set is **{:.4f}**.'.format(r2_test))

    #########################Design guidance#########################
    st.markdown('<p class="L1">Design guidance</p>', unsafe_allow_html=True)

    X_target_list = []

    test_range_1 = st.sidebar.slider("Select a lower bound for the contact length: ", 20, 50, 40)
    test_range_2 = st.sidebar.slider("Select a upper bound for the contact length: ", 60, 100, 70)

    for i in range(test_range_1, test_range_2):
        X_target_list.append([(i - df.mean()[0]) / df.std()[0], 0])

    X_target = np.array(X_target_list)
    pred_X_target = regr.predict(X_target)
    pred_y_target = pred_X_target * df.std()[2] + df.mean()[2]

    if use_new_data == data_str_1:
        st.markdown("**Choose ideal contact length based on users' preferences on the maximum displacement.**")
        fig = plt.figure()
        plt.plot(X_target[:,0] * df.std()[0] + df.mean()[0], pred_y_target)
        plt.scatter(X_target[:,0] * df.std()[0] + df.mean()[0], pred_y_target)
        plt.xlabel('Contact length', fontsize=16)
        plt.ylabel('Maximum_displacement', fontsize=16)
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=600)
    
    if use_new_data == data_str_2:
        st.markdown("**Choose ideal contact length based on users' preferences on the maximum Mises stress.**")
        fig = plt.figure()
        plt.plot(X_target[:,0] * df.std()[0] + df.mean()[0], pred_y_target[:,0])
        plt.scatter(X_target[:,0] * df.std()[0] + df.mean()[0], pred_y_target[:,0])
        plt.xlabel('Contact length', fontsize=16)
        plt.ylabel('Maximum_Mises_stress', fontsize=16)
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=600)
        
        st.markdown("**Choose ideal contact length based on users' preferences on the maximum displacement.**")
        fig = plt.figure()
        plt.plot(X_target[:,0] * df.std()[0] + df.mean()[0], pred_y_target[:,1])
        plt.scatter(X_target[:,0] * df.std()[0] + df.mean()[0], pred_y_target[:,1])
        plt.xlabel('Contact length', fontsize=16)
        plt.ylabel('Maximum_displacement', fontsize=16)
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=600)


if __name__ == '__main__':
    main()
