import streamlit as st
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from proccessing_bundle import processing_bundle
from processing_device import processing_device
from eval import predict_demo

FIRST_N_ROWS = 5
PATH_OUTPUT = "./"
NUM_SEGMENTS_IN_TASK = 5


def user_input_features(test_data):
    game_category = st.sidebar.text_input('gamecategory', test_data['gamecategory'][0])
    subgamecategory = st.sidebar.text_input('subgamecategory', test_data['subgamecategory'][0])
    bundle = st.sidebar.text_input('bundle', test_data['bundle'][0])
    created = st.sidebar.text_input('created', test_data['created'][0])
    shift = st.sidebar.text_input('shift', test_data['shift'][0])
    oblast = st.sidebar.text_input('oblast', test_data['oblast'][0])
    city = st.sidebar.text_input('city', test_data['city'][0])
    os = st.sidebar.text_input('os', test_data['os'][0])
    osv = st.sidebar.text_input('osv', test_data['osv'][0])
    data = {
        'game_category': game_category,
        'subgamecategory': subgamecategory,
        'bundle': bundle,
        'created': created,
        'shift': shift,
        'oblast': oblast,
        'os': os,
        'city': city,
        'osv': osv,
    }
    features = pd.DataFrame(data, index=[0])
    return features


def plot_left_slidebar(test_data):
    st.sidebar.header('User Input Parameters')
    df = user_input_features(test_data)


def load_csv_with_test_data():
    uploadedFile = st.file_uploader("Choose a file", type=['csv'], accept_multiple_files=False)  # , key="fileUploader"
    if uploadedFile:
        df_input = pd.read_csv(uploadedFile)
        st.subheader('Input users data')
        st.write(df_input.iloc[0:FIRST_N_ROWS, 1:])
        return uploadedFile, df_input
    else:
        return uploadedFile, None


def predict_segments(input_data):
    st.subheader('Predicted segment probability for user data')

    data = processing_bundle(input_data)
    data = processing_device(data)
    preds_proba = predict_demo(data)
    preds_proba_df = pd.DataFrame(data=preds_proba, index=np.arange(preds_proba.shape[0]))
    preds_proba_df = preds_proba_df.rename(columns={"0": "prob_seg1", "1": "prob_seg1",
                                                    "2": "prob_seg1", "3": "prob_seg1",
                                                    "4": "prob_seg1"})
    preds_proba_df.insert(0, "user_data_row", np.arange(0, preds_proba_df.shape[0], 1))
    st.write(preds_proba_df)
    fout_path = os.path.join(PATH_OUTPUT, "res_segment_probs.csv")
    preds_proba_df.to_csv(fout_path)
    st.write(f"Predicted segment probability for user data were save to file {fout_path}")
    return preds_proba_df


def process_user_coverage(is_uploaded_file, probs):
    push_button_segment_coverage = st.button("Segment coverage", key=None, help=None, on_click=None, args=None,
                                             kwargs=None)

    def calculate_percentage_coverage():
        percentage_arr = []
        probs_arr = probs.to_numpy()[:, 1:]
        print(probs.shape, probs_arr.shape)
        max_rows = np.argmax(probs_arr, axis=1)
        print(max_rows)
        for i in range(NUM_SEGMENTS_IN_TASK):
            percentage_arr.append(round(np.count_nonzero(max_rows == i) / probs_arr.shape[0], 3))
        return percentage_arr

    if push_button_segment_coverage and is_uploaded_file:
        coverage = calculate_percentage_coverage()
        df_result = pd.DataFrame({"№ Segment": np.arange(1, NUM_SEGMENTS_IN_TASK + 1, 1),
                                  "user coverage": coverage})
        df_result["% user coverage"] = [str(i * 100) + "%" for i in coverage]
        st.subheader('Segment coverage results')

        col1, col2 = st.columns(2)

        with col1:
            st.write(df_result[["№ Segment", "% user coverage"]])

        with col2:
            labels = df_result["№ Segment"].to_numpy()
            sizes = df_result["user coverage"].to_numpy() * 100
            pie_segment_plotly = go.Figure(data=[go.Pie(labels=labels, values=sizes, pull=[0, 0, 0.2, 0, 0])])
            colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
            pie_segment_plotly.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                                             marker=dict(colors=colors, line=dict(color='#000000', width=2)))
            st.plotly_chart(pie_segment_plotly, use_container_width=True)


def process_clusters(uploadedFile):
    loc = Nominatim(user_agent="GetLoc")

    push_button_cluster = st.button("Cluster", key=None, help=None, on_click=None, args=None, kwargs=None)
    if push_button_cluster and uploadedFile:

        def generate_cluster_info():
            info = []
            for i in df_result_cluster.index:
                info_string = ""
                for c in df_result_cluster.columns:
                    info_string += c + ":" + str(df_result_cluster[c][i]) + ", "
                info.append(info_string)
            return info

        def get_gps():
            lat = []
            lon = []
            for i in df_result_cluster.index:
                # entering the location name
                getLoc = loc.geocode(df_result_cluster["City"][i])
                lat.append(float(getLoc.latitude))
                lon.append(float(getLoc.longitude))
            return np.array(lat), np.array(lon)

        df_result_cluster = pd.DataFrame(
            {"Cluster": [1, 2, 3, 4, 5], "App": ["Games, Actions", "Games, Casino", "Games, Puzzle",
                                                 "Games, Actions", "Games, Casino", ],
             "OS": ["ios", "ios", "android", "ios", "android"], "Is weekend": ["yes", "no", "no", "no", "yes"],
             "Location": ["Свердловская область", "Санкт-Петербург", "Краснодарский край", "Москва", " Москва"],
             "City": ["Екатеринбург", "Санкт-Петербург", "Краснодар", "Москва", "Москва"]})

        x = df_result_cluster["Cluster"].to_numpy()
        fig = go.Figure(data=[go.Scatter(
            x=x,
            y=np.arange(10, 10 + x.shape[0], 1),
            text=generate_cluster_info(),
            mode='markers',
            marker=dict(
                color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)', 'rgb(44, 160, 101)', 'rgb(255, 65, 54)',
                       'rgb(255, 144, 210)'],
                size=[40, 60, 80, 100, 120],
            )
        )])  # TODO:размеры кластеров
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Clustering results')
        st.write(df_result_cluster)

        text = generate_cluster_info()
        lat, lon = get_gps()

        colors = ["royalblue", "crimson", "lightseagreen", "orange", "lightgrey"]

        fig2 = go.Figure(go.Scattergeo())

        print(lon)
        print(lat)
        print(text)
        fig2.add_scattergeo(
            lon=lon,
            lat=lat,
            text=text,
            marker=dict(
                size=10,
                color=colors,
                line_color='rgb(40,40,40)',
                line_width=0.5,
                sizemode='area'
            ))
        # name='{0} - {1}'.format(lim[0], lim[1])))

        fig2.update_layout(
            title_text='World map visualization',
            showlegend=True,
            geo=dict(
                scope='world',
                landcolor='rgb(217, 217, 217)',
            )
        )
        st.plotly_chart(fig2, use_container_width=True)
        # fig.show()

        # image_path_children = "kids-icon-png-11552334786q2mjiemnbp.png"
        # st.image(image_path_children, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


def run():
    st.write("""
    # User Segment Prediction App
    This app predicts the **user segment** type!
    """)

    # test_data = pd.read_csv("test_web.csv")  # TODO: Need to change on all data
    # plot_left_slidebar(test_data)

    # load file from user
    is_uploaded_file, input_data = load_csv_with_test_data()

    if is_uploaded_file and input_data is not None:
        probs_segment = predict_segments(input_data)

        process_user_coverage(is_uploaded_file, probs_segment)

        process_clusters(is_uploaded_file)


if __name__ == '__main__':
    run()
